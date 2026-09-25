//! Cross-individual likelihood refinement of mixed pseudobulk groups.
//!
//! [`RandProjOps::partition_columns_to_mixed_groups`] pools the cells of
//! many individuals by the sign code of a projection. A group can still
//! hold more than one cell state. This pass moves cells between groups
//! under the Poisson model
//!
//! ```text
//!   y_cg ~ Poisson(s_c mu_gp Lambda_gi)
//! ```
//!
//! for cell `c` of individual `i` in group `p`: `s_c` is the cell's depth,
//! `mu` the group's profile and `Lambda` the individual's gene offset. A
//! cell keeps its `Lambda_gi` in every group, so a move gains only from a
//! better match of `mu`, the cell state, and not from joining a group rich
//! in its own individual.
//!
//! Each sweep fits `mu` and `Lambda` for the current labels by alternating
//! closed-form updates (concave in their logs), then moves every cell to
//! its best candidate group with the parameters held fixed. Both steps
//! raise the same likelihood. Candidates are the groups of the codes one
//! bit away from the cell's own code. A move that would leave its group
//! with fewer than `min_batches` individuals is vetoed.

use crate::alg::batch_mixing::batch_indices;
use crate::alg::dc_poisson::compact_labels;
use crate::alg::hvg::select_hvg_streaming;
use crate::alg::random_projection::{mixed_group_codes, RandProjOps};
use crate::sparse_data_visitors::VisitColumnsOps;
use crate::sparse_io_vector::SparseIoVec;
use log::info;
use nalgebra_sparse::CscMatrix;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

#[derive(Clone, Debug)]
pub struct MixedRefineParams {
    /// most sweeps; stops early once no cell moves
    pub max_sweeps: usize,
    /// most alternating rounds of the `mu`/`Lambda` fit per sweep
    pub max_fit_rounds: usize,
    /// the fit stops once no `Lambda` changes by more than this ratio
    pub fit_tol: f64,
    /// highly variable genes scored
    pub num_genes: usize,
    /// prior weight: pseudo-depth for `mu`, pseudo-count for `Lambda`
    pub pseudocount: f64,
    /// least log-likelihood gain for a move
    pub min_gain: f64,
    pub block_size: Option<usize>,
}

impl Default for MixedRefineParams {
    fn default() -> Self {
        Self {
            max_sweeps: 5,
            max_fit_rounds: 50,
            fit_tol: 1e-4,
            num_genes: 2000,
            pseudocount: 1.0,
            min_gain: 1.0,
            block_size: None,
        }
    }
}

/////////////////////////
// Candidates by codes //
/////////////////////////

/// Candidate groups of each cell: the groups of its own code and of the
/// codes one bit away, as labelled by the initial partition.
pub struct CodeNeighbours {
    codes: Vec<usize>,
    label_of_code: HashMap<usize, usize>,
    bits: usize,
}

impl CodeNeighbours {
    /// `codes` - binary codes of `bits` bits; `labels` - group of each
    /// column (every column of one code shares its group)
    pub fn new(codes: &[usize], labels: &[usize], bits: usize) -> Self {
        let label_of_code = codes.iter().copied().zip(labels.iter().copied()).collect();
        Self {
            codes: codes.to_vec(),
            label_of_code,
            bits,
        }
    }

    /// Sorted candidate groups of `cell`, including `current`.
    pub fn candidates(&self, cell: usize, current: usize) -> Vec<usize> {
        let code = self.codes[cell];
        let mut out: Vec<usize> = std::iter::once(code)
            .chain((0..self.bits).map(|k| code ^ (1 << k)))
            .filter_map(|c| self.label_of_code.get(&c).copied())
            .chain(std::iter::once(current))
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    fn max_label(&self) -> Option<usize> {
        self.label_of_code.values().copied().max()
    }
}

//////////////////
// Count source //
//////////////////

/// Columns of counts visited in blocks.
pub trait CountBlocks: Sync {
    fn num_genes(&self) -> usize;
    fn num_cells(&self) -> usize;
    /// Call `visit(first_column, genes x cells)` on every block, possibly
    /// in parallel.
    fn visit_blocks<F>(&self, visit: &F) -> anyhow::Result<()>
    where
        F: Fn(usize, &CscMatrix<f32>) -> anyhow::Result<()> + Sync + Send;
}

impl CountBlocks for CscMatrix<f32> {
    fn num_genes(&self) -> usize {
        self.nrows()
    }
    fn num_cells(&self) -> usize {
        self.ncols()
    }
    fn visit_blocks<F>(&self, visit: &F) -> anyhow::Result<()>
    where
        F: Fn(usize, &CscMatrix<f32>) -> anyhow::Result<()> + Sync + Send,
    {
        visit(0, self)
    }
}

/// Keep the rows mapped to `Some(new_row)`; `row_to_sub` must be
/// increasing on the kept rows so row order is kept.
pub fn restrict_rows(
    csc: &CscMatrix<f32>,
    row_to_sub: &[Option<usize>],
    num_rows: usize,
) -> CscMatrix<f32> {
    let mut offsets = Vec::with_capacity(csc.ncols() + 1);
    let mut rows = Vec::new();
    let mut vals = Vec::new();
    offsets.push(0);
    for col in csc.col_iter() {
        for (&g, &v) in col.row_indices().iter().zip(col.values()) {
            if let Some(r) = row_to_sub[g] {
                rows.push(r);
                vals.push(v);
            }
        }
        offsets.push(rows.len());
    }
    CscMatrix::try_from_csc_data(num_rows, csc.ncols(), offsets, rows, vals)
        .expect("restricted rows keep a valid CSC layout")
}

/// A gene subset of a [`SparseIoVec`], read block by block.
struct GeneSubsetColumns<'a> {
    data: &'a SparseIoVec,
    row_to_sub: Vec<Option<usize>>,
    num_genes: usize,
    block_size: Option<usize>,
}

impl<'a> GeneSubsetColumns<'a> {
    fn new(data: &'a SparseIoVec, genes: &[usize], block_size: Option<usize>) -> Self {
        let mut row_to_sub = vec![None; data.num_rows()];
        let mut sorted = genes.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        for (r, &g) in sorted.iter().enumerate() {
            row_to_sub[g] = Some(r);
        }
        Self {
            data,
            row_to_sub,
            num_genes: sorted.len(),
            block_size,
        }
    }
}

impl CountBlocks for GeneSubsetColumns<'_> {
    fn num_genes(&self) -> usize {
        self.num_genes
    }
    fn num_cells(&self) -> usize {
        self.data.num_columns()
    }
    fn visit_blocks<F>(&self, visit: &F) -> anyhow::Result<()>
    where
        F: Fn(usize, &CscMatrix<f32>) -> anyhow::Result<()> + Sync + Send,
    {
        let visitor = |(lb, ub): (usize, usize),
                       data: &SparseIoVec,
                       _: &(),
                       _: std::sync::Arc<Mutex<&mut ()>>|
         -> anyhow::Result<()> {
            let csc = data.read_columns_csc(lb..ub)?;
            visit(lb, &restrict_rows(&csc, &self.row_to_sub, self.num_genes))
        };
        let mut unit = ();
        self.data
            .visit_columns_by_block(&visitor, &(), &mut unit, self.block_size)
    }
}

//////////////////
// Sufficient   //
// statistics   //
//////////////////

/// Counts summed by group and by individual, and cell depths.
struct Sums {
    /// K x G
    y_pg: Vec<f64>,
    /// I x G
    y_ig: Vec<f64>,
    depth: Vec<f64>,
}

fn accumulate<S: CountBlocks>(
    source: &S,
    batch: &[usize],
    labels: &[usize],
    k: usize,
    n_indv: usize,
) -> anyhow::Result<Sums> {
    let ng = source.num_genes();
    let sums = Mutex::new(Sums {
        y_pg: vec![0.0; k * ng],
        y_ig: vec![0.0; n_indv * ng],
        depth: vec![0.0; source.num_cells()],
    });
    source.visit_blocks(&|lb, csc| {
        let mut entries: Vec<(usize, usize, f64)> = Vec::with_capacity(csc.nnz());
        let mut depth = Vec::with_capacity(csc.ncols());
        for (j, col) in csc.col_iter().enumerate() {
            let c = lb + j;
            let mut s = 0.0;
            for (&g, &v) in col.row_indices().iter().zip(col.values()) {
                entries.push((labels[c] * ng + g, batch[c] * ng + g, f64::from(v)));
                s += f64::from(v);
            }
            depth.push(s);
        }
        let mut out = sums.lock().expect("sums lock");
        for (pg, ig, v) in entries {
            out.y_pg[pg] += v;
            out.y_ig[ig] += v;
        }
        out.depth[lb..lb + depth.len()].copy_from_slice(&depth);
        Ok(())
    })?;
    Ok(sums.into_inner().expect("sums lock"))
}

/////////
// Fit //
/////////

/// Fitted parameters for scoring: `ln mu` (K x G) and
/// `a[p * I + i] = sum_g mu_gp Lambda_gi`.
struct Scoring {
    ln_mu: Vec<f32>,
    a: Vec<f64>,
}

/// Fit `mu` and `Lambda` for the current labels by alternating their
/// closed-form updates. `lambda` (I x G) is updated in place so the next
/// sweep starts from it.
#[allow(clippy::too_many_arguments)]
fn fit(
    sums: &Sums,
    batch: &[usize],
    labels: &[usize],
    k: usize,
    n_indv: usize,
    ng: usize,
    lambda: &mut [f64],
    params: &MixedRefineParams,
) -> Scoring {
    let prior = params.pseudocount;
    let total_depth: f64 = sums.depth.iter().sum();
    let rate: Vec<f64> = (0..ng)
        .map(|g| (0..n_indv).map(|i| sums.y_ig[i * ng + g]).sum::<f64>() / total_depth.max(1.0))
        .collect();

    // depth of each occupied (group, individual) pair
    let mut n_pi = vec![0.0f64; k * n_indv];
    for (c, &s) in sums.depth.iter().enumerate() {
        n_pi[labels[c] * n_indv + batch[c]] += s;
    }
    let by_group: Vec<Vec<(usize, f64)>> = (0..k)
        .map(|p| {
            (0..n_indv)
                .filter_map(|i| Some((i, n_pi[p * n_indv + i])).filter(|&(_, n)| n > 0.0))
                .collect()
        })
        .collect();
    let by_indv: Vec<Vec<(usize, f64)>> = (0..n_indv)
        .map(|i| {
            (0..k)
                .filter_map(|p| Some((p, n_pi[p * n_indv + i])).filter(|&(_, n)| n > 0.0))
                .collect()
        })
        .collect();

    let mut mu = vec![0.0f64; k * ng];
    for _ in 0..params.max_fit_rounds.max(1) {
        mu.par_chunks_mut(ng).enumerate().for_each(|(p, mu_p)| {
            let mut den = vec![0.0f64; ng];
            for &(i, n) in &by_group[p] {
                for (d, l) in den.iter_mut().zip(&lambda[i * ng..(i + 1) * ng]) {
                    *d += n * l;
                }
            }
            for g in 0..ng {
                mu_p[g] = (sums.y_pg[p * ng + g] + prior * rate[g]) / (den[g] + prior);
            }
        });
        let change = lambda
            .par_chunks_mut(ng)
            .enumerate()
            .map(|(i, lambda_i)| {
                let mut den = vec![0.0f64; ng];
                for &(p, n) in &by_indv[i] {
                    for (d, m) in den.iter_mut().zip(&mu[p * ng..(p + 1) * ng]) {
                        *d += n * m;
                    }
                }
                let mut change = 0.0f64;
                for g in 0..ng {
                    let next = (sums.y_ig[i * ng + g] + prior) / (den[g] + prior);
                    change = change.max((next / lambda_i[g]).ln().abs());
                    lambda_i[g] = next;
                }
                change
            })
            .reduce(|| 0.0, f64::max);
        if change < params.fit_tol {
            break;
        }
    }

    let a: Vec<f64> = (0..k * n_indv)
        .into_par_iter()
        .map(|pi| {
            let (p, i) = (pi / n_indv, pi % n_indv);
            mu[p * ng..(p + 1) * ng]
                .iter()
                .zip(&lambda[i * ng..(i + 1) * ng])
                .map(|(m, l)| m * l)
                .sum()
        })
        .collect();
    let ln_mu = mu.iter().map(|&m| m.ln() as f32).collect();
    Scoring { ln_mu, a }
}

////////////
// Sweeps //
////////////

/// Move cells between groups in place; returns the number of moves.
///
/// * `source` - counts, genes x cells
/// * `batch` - individual of each cell, in `0..n_indv`
/// * `labels` - compact group of each cell, rewritten
/// * `neighbours` - candidate groups of each cell
/// * `min_batches` - individuals a group keeps
pub fn refine_mixed_labels<S: CountBlocks>(
    source: &S,
    batch: &[usize],
    labels: &mut [usize],
    neighbours: &CodeNeighbours,
    min_batches: usize,
    params: &MixedRefineParams,
) -> anyhow::Result<usize> {
    let n = source.num_cells();
    anyhow::ensure!(
        batch.len() == n && labels.len() == n,
        "one batch and label per cell"
    );
    if n == 0 || params.max_sweeps == 0 {
        return Ok(0);
    }
    let ng = source.num_genes();
    let n_indv = batch.iter().max().map_or(0, |&m| m + 1);
    let k = labels
        .iter()
        .copied()
        .chain(neighbours.max_label())
        .max()
        .map_or(0, |m| m + 1);

    let mut size = vec![0usize; k];
    let mut count = vec![0usize; k * n_indv];
    let mut distinct = vec![0usize; k];
    for (&p, &i) in labels.iter().zip(batch) {
        size[p] += 1;
        count[p * n_indv + i] += 1;
        if count[p * n_indv + i] == 1 {
            distinct[p] += 1;
        }
    }

    let mut lambda = vec![1.0f64; n_indv * ng];
    let mut total_moves = 0;
    for sweep in 0..params.max_sweeps {
        let sums = accumulate(source, batch, labels, k, n_indv)?;
        let scoring = fit(&sums, batch, labels, k, n_indv, ng, &mut lambda, params);

        // propose each cell's best candidate, parameters held fixed
        let proposals = Mutex::new(Vec::<(usize, usize)>::new());
        let current: &[usize] = labels;
        source.visit_blocks(&|lb, csc| {
            let block: Vec<(usize, usize)> = (0..csc.ncols())
                .into_par_iter()
                .filter_map(|j| {
                    let c = lb + j;
                    let (p, i) = (current[c], batch[c]);
                    let col = csc.col(j);
                    let score = |q: usize| -> f64 {
                        let ln_mu = &scoring.ln_mu[q * ng..(q + 1) * ng];
                        let fit: f64 = col
                            .row_indices()
                            .iter()
                            .zip(col.values())
                            .map(|(&g, &v)| f64::from(v) * f64::from(ln_mu[g]))
                            .sum();
                        fit - sums.depth[c] * scoring.a[q * n_indv + i]
                    };
                    let stay = score(p);
                    neighbours
                        .candidates(c, p)
                        .into_iter()
                        .filter(|&q| q != p && size[q] > 0)
                        .map(|q| (q, score(q) - stay))
                        .filter(|&(_, gain)| gain > params.min_gain)
                        .max_by(|x, y| x.1.total_cmp(&y.1))
                        .map(|(q, _)| (c, q))
                })
                .collect();
            proposals.lock().expect("proposals lock").extend(block);
            Ok(())
        })?;
        let mut proposals = proposals.into_inner().expect("proposals lock");
        proposals.sort_unstable();

        // apply in cell order, keeping every group's individuals
        let mut moves = 0;
        for (c, q) in proposals {
            let (p, i) = (labels[c], batch[c]);
            let last_of_indv = count[p * n_indv + i] == 1;
            if last_of_indv && distinct[p] <= min_batches {
                continue;
            }
            if last_of_indv {
                distinct[p] -= 1;
            }
            count[p * n_indv + i] -= 1;
            size[p] -= 1;
            if count[q * n_indv + i] == 0 {
                distinct[q] += 1;
            }
            count[q * n_indv + i] += 1;
            size[q] += 1;
            labels[c] = q;
            moves += 1;
        }
        info!(
            "mixed refine sweep {}: {} of {} cells moved",
            sweep + 1,
            moves,
            n
        );
        total_moves += moves;
        if moves == 0 {
            break;
        }
    }
    Ok(total_moves)
}

/////////////////
// Data entry  //
/////////////////

pub trait MixedRefineOps {
    /// Like [`RandProjOps::partition_columns_to_mixed_groups`], then move
    /// cells between the groups under the likelihood (see the module
    /// docs). Returns the number of nonempty groups.
    fn partition_columns_to_refined_mixed_groups<T>(
        &mut self,
        proj_kn: &nalgebra::DMatrix<f32>,
        num_features: Option<usize>,
        batch_membership: &[T],
        min_batches: usize,
        merge_levels: usize,
        params: &MixedRefineParams,
    ) -> anyhow::Result<usize>
    where
        T: std::hash::Hash + Eq + Clone;
}

impl MixedRefineOps for SparseIoVec {
    fn partition_columns_to_refined_mixed_groups<T>(
        &mut self,
        proj_kn: &nalgebra::DMatrix<f32>,
        num_features: Option<usize>,
        batch_membership: &[T],
        min_batches: usize,
        merge_levels: usize,
        params: &MixedRefineParams,
    ) -> anyhow::Result<usize>
    where
        T: std::hash::Hash + Eq + Clone,
    {
        let batch = batch_indices(batch_membership);
        let n_batches = batch.iter().max().map_or(0, |&m| m + 1);
        let min_batches = min_batches.min(n_batches);
        let (codes, labels, bits) =
            mixed_group_codes(proj_kn, num_features, &batch, min_batches, merge_levels)?;
        let (mut groups, _) = compact_labels(&labels);

        if params.max_sweeps > 0 && params.num_genes > 0 {
            let num_genes = params.num_genes.min(self.num_rows());
            let hvg = select_hvg_streaming(self, Some(num_genes), None, None, params.block_size)?;
            let source = GeneSubsetColumns::new(self, &hvg.selected_indices, params.block_size);
            let neighbours = CodeNeighbours::new(&codes, &groups, bits);
            let moves = refine_mixed_labels(
                &source,
                &batch,
                &mut groups,
                &neighbours,
                min_batches,
                params,
            )?;
            info!(
                "mixed refine: {} moves over {} genes, groups keep at least {} individuals",
                moves, source.num_genes, min_batches
            );
        }

        self.assign_group_labels(&groups);
        Ok(groups.iter().collect::<HashSet<_>>().len())
    }
}

#[cfg(test)]
#[path = "refine_mixed_tests.rs"]
mod tests;
