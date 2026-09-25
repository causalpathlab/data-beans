//! Cross-individual likelihood refinement of mixed pseudobulk groups.
//!
//! [`RandProjOps::partition_columns_to_mixed_groups`] pools the cells of
//! many individuals by the sign code of a projection. A group can still
//! hold more than one cell state. This pass moves cells between groups
//! under the Poisson model
//!
//! ```text
//!   y_cg ~ Poisson(s_c mu_gp omega_gi)
//! ```
//!
//! for cell `c` of individual `i` in group `p`: `s_c` is the cell's depth,
//! `mu` the group's profile and `omega` the individual's gene offset. A
//! cell keeps its `omega_gi` in every group, so a move gains only from a
//! better match of `mu`, the cell state, and not from joining a group rich
//! in its own individual.
//!
//! Each sweep fits `mu` and `omega` for the current labels by alternating
//! closed-form updates (concave in their logs), then moves every cell to
//! its best candidate group with the parameters held fixed. Both steps
//! raise the same likelihood. Candidates are the groups of the codes one
//! bit away from the cell's own code. A move that would leave its group
//! with fewer than `min_batches` individuals is vetoed.
//!
//! Unlike [`crate::alg::dc_poisson`], the score carries the individual
//! offset `omega`, and a cell's current group is scored with the cell
//! inside it (no leave-one-out), which only favours staying.

use crate::alg::dc_poisson::compact_labels;
use crate::alg::hvg::select_hvg_streaming;
use crate::alg::random_projection::{mixed_partition, RandProjOps};
use crate::sparse_data_visitors::VisitColumnsOps;
use crate::sparse_io_vector::SparseIoVec;
use log::info;
use nalgebra::DMatrix;
use nalgebra_sparse::CscMatrix;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Clone, Debug)]
pub struct MixedRefineParams {
    /// most sweeps; stops early once no cell moves
    pub max_sweeps: usize,
    /// most alternating rounds of the `mu`/`omega` fit per sweep
    pub max_fit_rounds: usize,
    /// the fit stops once no `omega` changes by more than this ratio
    pub fit_tol: f64,
    /// highly variable genes scored
    pub num_genes: usize,
    /// prior weight: pseudo-depth for `mu`, pseudo-count for `omega`
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
pub(crate) struct CodeNeighbours {
    codes: Vec<usize>,
    candidates_of_code: HashMap<usize, Vec<usize>>,
}

impl CodeNeighbours {
    /// `codes` - binary codes of `bits` bits; `labels` - group of each
    /// column (every column of one code shares its group)
    pub(crate) fn new(codes: Vec<usize>, labels: &[usize], bits: usize) -> Self {
        let label_of_code: HashMap<usize, usize> =
            codes.iter().copied().zip(labels.iter().copied()).collect();
        let candidates_of_code = label_of_code
            .keys()
            .map(|&code| {
                let mut groups: Vec<usize> = std::iter::once(code)
                    .chain((0..bits).map(|k| code ^ (1 << k)))
                    .filter_map(|c| label_of_code.get(&c).copied())
                    .collect();
                groups.sort_unstable();
                groups.dedup();
                (code, groups)
            })
            .collect();
        Self {
            codes,
            candidates_of_code,
        }
    }

    /// Sorted candidate groups of `cell`.
    pub(crate) fn candidates(&self, cell: usize) -> &[usize] {
        &self.candidates_of_code[&self.codes[cell]]
    }
}

//////////////////
// Count source //
//////////////////

/// Columns of counts visited in blocks.
pub(crate) trait CountBlocks: Sync {
    fn num_genes(&self) -> usize;
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
    fn visit_blocks<F>(&self, visit: &F) -> anyhow::Result<()>
    where
        F: Fn(usize, &CscMatrix<f32>) -> anyhow::Result<()> + Sync + Send,
    {
        visit(0, self)
    }
}

/// A data vector read block by block; the block size is the second field.
struct DataBlocks<'a>(&'a SparseIoVec, Option<usize>);

impl CountBlocks for DataBlocks<'_> {
    fn num_genes(&self) -> usize {
        self.0.num_rows()
    }
    fn visit_blocks<F>(&self, visit: &F) -> anyhow::Result<()>
    where
        F: Fn(usize, &CscMatrix<f32>) -> anyhow::Result<()> + Sync + Send,
    {
        let visitor = |(lb, ub): (usize, usize),
                       data: &SparseIoVec,
                       _: &(),
                       _: std::sync::Arc<Mutex<&mut ()>>|
         -> anyhow::Result<()> { visit(lb, &data.read_columns_csc(lb..ub)?) };
        self.0
            .visit_columns_by_block(&visitor, &(), &mut (), self.1)
    }
}

////////////////////////////
// Sufficient statistics //
////////////////////////////

/// Label-free totals: counts by individual, cell depths, gene rates.
struct Totals {
    /// I x G
    y_ig: DMatrix<f64>,
    depth: Vec<f64>,
    /// counts per unit depth, a prior centre for empty groups
    rate: Vec<f64>,
}

/// Counts summed by group (K x G) and, when `batch` is given, the
/// label-free totals in the same pass.
fn accumulate<S: CountBlocks>(
    source: &S,
    labels: &[usize],
    k: usize,
    batch: Option<(&[usize], usize)>,
) -> anyhow::Result<(DMatrix<f64>, Option<Totals>)> {
    let ng = source.num_genes();
    let n = labels.len();
    let out = Mutex::new((
        DMatrix::<f64>::zeros(k, ng),
        batch.map(|(_, n_indv)| (DMatrix::<f64>::zeros(n_indv, ng), vec![0.0; n])),
    ));
    source.visit_blocks(&|lb, csc| {
        let mut guard = out.lock().expect("sums lock");
        let (y_pg, totals) = &mut *guard;
        for (j, col) in csc.col_iter().enumerate() {
            let c = lb + j;
            for (&g, &v) in col.row_indices().iter().zip(col.values()) {
                y_pg[(labels[c], g)] += f64::from(v);
            }
            if let (Some((y_ig, depth)), Some((batch, _))) = (totals.as_mut(), batch) {
                for (&g, &v) in col.row_indices().iter().zip(col.values()) {
                    y_ig[(batch[c], g)] += f64::from(v);
                }
                depth[c] = col.values().iter().map(|&v| f64::from(v)).sum();
            }
        }
        Ok(())
    })?;
    let (y_pg, totals) = out.into_inner().expect("sums lock");
    let totals = totals.map(|(y_ig, depth)| {
        let total_depth: f64 = depth.iter().sum();
        let rate = y_ig
            .row_sum()
            .iter()
            .map(|&y| y / total_depth.max(1.0))
            .collect();
        Totals { y_ig, depth, rate }
    });
    Ok((y_pg, totals))
}

/////////
// Fit //
/////////

/// Fitted parameters for scoring: `ln mu` (K x G) and
/// `a[(p, i)] = sum_g mu_gp omega_gi` (K x I).
struct Scoring {
    ln_mu: DMatrix<f32>,
    a: DMatrix<f64>,
}

/// Fit `mu` and `omega` by alternating their closed-form updates, each a
/// matrix product with the group x individual depths `n_pi`. `omega`
/// (I x G) is updated in place so the next sweep starts from it.
fn fit(
    y_pg: &DMatrix<f64>,
    totals: &Totals,
    n_pi: &DMatrix<f64>,
    omega: &mut DMatrix<f64>,
    params: &MixedRefineParams,
) -> Scoring {
    let prior = params.pseudocount;
    let (k, ng) = y_pg.shape();
    let mut mu = DMatrix::<f64>::zeros(k, ng);
    for _ in 0..params.max_fit_rounds.max(1) {
        let den = n_pi * &*omega;
        mu = DMatrix::from_fn(k, ng, |p, g| {
            (y_pg[(p, g)] + prior * totals.rate[g]) / (den[(p, g)] + prior)
        });
        let den = n_pi.tr_mul(&mu);
        let next = DMatrix::from_fn(omega.nrows(), ng, |i, g| {
            (totals.y_ig[(i, g)] + prior) / (den[(i, g)] + prior)
        });
        let change = next
            .iter()
            .zip(omega.iter())
            .fold(0.0f64, |c, (a, b)| c.max((a / b).ln().abs()));
        *omega = next;
        if change < params.fit_tol {
            break;
        }
    }
    Scoring {
        a: &mu * omega.transpose(),
        ln_mu: mu.map(|m| m.ln() as f32),
    }
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
pub(crate) fn refine_mixed_labels<S: CountBlocks>(
    source: &S,
    batch: &[usize],
    labels: &mut [usize],
    neighbours: &CodeNeighbours,
    min_batches: usize,
    params: &MixedRefineParams,
) -> anyhow::Result<usize> {
    let n = batch.len();
    anyhow::ensure!(labels.len() == n, "one batch and label per cell");
    let n_indv = batch.iter().max().map_or(0, |&m| m + 1);
    let k = labels.iter().max().map_or(0, |&m| m + 1);

    // cells per (group, individual), and individuals per group
    let mut count = vec![0usize; k * n_indv];
    let mut distinct = vec![0usize; k];
    for (&p, &i) in labels.iter().zip(batch) {
        count[p * n_indv + i] += 1;
        if count[p * n_indv + i] == 1 {
            distinct[p] += 1;
        }
    }

    let mut totals: Option<Totals> = None;
    let mut omega = DMatrix::<f64>::from_element(n_indv, source.num_genes(), 1.0);
    let mut total_moves = 0;
    for sweep in 0..params.max_sweeps {
        let first = totals.is_none().then_some((batch, n_indv));
        let (y_pg, new_totals) = accumulate(source, labels, k, first)?;
        let totals = &*totals.get_or_insert_with(|| new_totals.expect("totals on first pass"));
        let mut n_pi = DMatrix::<f64>::zeros(k, n_indv);
        for (c, &s) in totals.depth.iter().enumerate() {
            n_pi[(labels[c], batch[c])] += s;
        }
        let scoring = fit(&y_pg, totals, &n_pi, &mut omega, params);

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
                        let fit: f64 = col
                            .row_indices()
                            .iter()
                            .zip(col.values())
                            .map(|(&g, &v)| f64::from(v) * f64::from(scoring.ln_mu[(q, g)]))
                            .sum();
                        fit - totals.depth[c] * scoring.a[(q, i)]
                    };
                    let stay = score(p);
                    neighbours
                        .candidates(c)
                        .iter()
                        .filter(|&&q| q != p && distinct[q] > 0)
                        .map(|&q| (q, score(q) - stay))
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
            if count[p * n_indv + i] == 1 {
                if distinct[p] <= min_batches {
                    continue;
                }
                distinct[p] -= 1;
            }
            count[p * n_indv + i] -= 1;
            if count[q * n_indv + i] == 0 {
                distinct[q] += 1;
            }
            count[q * n_indv + i] += 1;
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
    /// cells between the groups under the likelihood of this data's counts
    /// (see the module docs). Without sweeps the groups are exactly those
    /// of the unrefined partition. Returns the number of nonempty groups.
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
        let part = mixed_partition(
            proj_kn,
            num_features,
            batch_membership,
            min_batches,
            merge_levels,
        )?;
        let mut labels = part.labels;

        if params.max_sweeps > 0 && params.num_genes > 0 {
            let num_genes = params.num_genes.min(self.num_rows());
            let hvg = select_hvg_streaming(self, Some(num_genes), None, None, params.block_size)?;
            let mut keep = vec![false; self.num_rows()];
            for &g in &hvg.selected_indices {
                keep[g] = true;
            }
            let mut scored = self.clone_for_collapse();
            scored.mask_rows(&keep)?;

            let (mut groups, k) = compact_labels(&labels);
            let mut label_of_group = vec![0; k];
            for (&g, &l) in groups.iter().zip(&labels) {
                label_of_group[g] = l;
            }
            let neighbours = CodeNeighbours::new(part.codes, &groups, part.bits);
            let moves = refine_mixed_labels(
                &DataBlocks(&scored, params.block_size),
                &part.batch,
                &mut groups,
                &neighbours,
                part.min_batches,
                params,
            )?;
            labels = groups.iter().map(|&g| label_of_group[g]).collect();
            info!(
                "mixed refine: {} moves over {} genes, groups keep at least {} individuals",
                moves,
                scored.num_rows(),
                part.min_batches
            );
        }

        self.assign_group_labels(&labels);
        Ok(self.num_groups())
    }
}

#[cfg(test)]
#[path = "refine_mixed_tests.rs"]
mod tests;
