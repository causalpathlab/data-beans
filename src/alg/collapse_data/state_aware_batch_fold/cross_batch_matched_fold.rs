//! The fold itself, for either kind of unit: within each node, every
//! pseudobulk (a collapse group in one batch; pb samples for samples, pb pairs
//! for cell pairs) is matched to its nearest pseudobulks of every other batch
//! (in a batch-corrected projection), and the node's fold for a batch is its
//! base-adjusted sum over the matched counterfactual sum, centred over the
//! node's batches, on top of the node's base (δ, or its parent's fold).

use crate::sparse_io_vector::SparseIoVec;
use legume_numeric::matrix::knn_match::ColumnDict;
use legume_numeric::matrix::utils::generate_minibatch_intervals;
use log::warn;
use nalgebra::{DMatrix, SymmetricEigen};
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use std::ops::AddAssign;
use std::sync::Mutex;

use super::{StateAwareBatchFoldParams, COORD_DIM};

/// Standardize each projection coordinate, then keep the leading principal
/// components: `coords = basisᵀ ((proj − mean) / scale)`.
#[derive(Clone, Debug)]
pub struct MatchingCoordinates {
    pub mean: Vec<f32>,
    pub scale: Vec<f32>,
    /// dims × components
    pub basis: DMatrix<f32>,
}

impl MatchingCoordinates {
    /// Fit on `proj` (dims × columns).
    pub fn fit(proj: &DMatrix<f32>) -> Self {
        let (dim, n) = proj.shape();
        let nf = n.max(1) as f64;
        let mean: Vec<f32> = (0..dim)
            .map(|r| (proj.row(r).iter().map(|&v| f64::from(v)).sum::<f64>() / nf) as f32)
            .collect();
        let scale: Vec<f32> = (0..dim)
            .map(|r| {
                let m = f64::from(mean[r]);
                let var = proj
                    .row(r)
                    .iter()
                    .map(|&v| (f64::from(v) - m).powi(2))
                    .sum::<f64>()
                    / nf;
                (var.sqrt() + 1e-9) as f32
            })
            .collect();
        let z = Self::standardize(proj, &mean, &scale);
        let cov = (&z * z.transpose()) / (nf as f32);
        let basis = leading_eigenvectors(cov, COORD_DIM.min(dim));
        Self { mean, scale, basis }
    }

    fn standardize(proj: &DMatrix<f32>, mean: &[f32], scale: &[f32]) -> DMatrix<f32> {
        DMatrix::from_fn(proj.nrows(), proj.ncols(), |r, c| {
            (proj[(r, c)] - mean[r]) / scale[r]
        })
    }

    /// Coordinates (components × columns) of `proj` (dims × columns).
    pub fn apply(&self, proj: &DMatrix<f32>) -> DMatrix<f32> {
        self.basis.transpose() * Self::standardize(proj, &self.mean, &self.scale)
    }
}

/// The `k` leading eigenvectors (columns) of a symmetric matrix.
fn leading_eigenvectors(m: DMatrix<f32>, k: usize) -> DMatrix<f32> {
    let eig = SymmetricEigen::new(m);
    let mut order: Vec<usize> = (0..eig.eigenvalues.len()).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));
    let cols: Vec<_> = order
        .iter()
        .take(k)
        .map(|&i| eig.eigenvectors.column(i))
        .collect();
    DMatrix::from_columns(&cols)
}

/// A fold per (gene × node × batch) on top of δ.
#[derive(Clone, Debug)]
pub struct StateAwareBatchFold {
    pub n_batches: usize,
    pub n_nodes: usize,
    /// genes × (n_nodes · n_batches): log δ plus the node's own term; a
    /// (node, batch) the node never saw holds log δ.
    pub log_fold: DMatrix<f32>,
    /// genes × batches
    pub log_delta: DMatrix<f32>,
    /// Each unit's node; `None`: the unit keeps δ.
    pub node_of_unit: Vec<Option<usize>>,
}

impl StateAwareBatchFold {
    /// The log fold (genes) of a unit in `batch` at `node` (`None`: δ).
    pub fn log_fold_at(&self, node: Option<usize>, batch: usize) -> nalgebra::DVectorView<'_, f32> {
        match node {
            Some(k) => self.log_fold.column(k * self.n_batches + batch),
            None => self.log_delta.column(batch),
        }
    }

    /// The log fold (genes) of unit `u` in `batch`.
    pub fn unit_log_fold(&self, u: usize, batch: usize) -> nalgebra::DVectorView<'_, f32> {
        self.log_fold_at(self.node_of_unit[u], batch)
    }

    /// One level of a collapse: each pseudobulk's log fold, the mean over its
    /// units (`unit_to_pb[u]`, `n_pb` pseudobulks), genes × n_pb. A
    /// pseudobulk's units then come out unbiased on average, as δ does.
    pub fn per_level(&self, unit_to_pb: &[usize], batch: &[usize], n_pb: usize) -> DMatrix<f32> {
        let g = self.log_delta.nrows();
        let mut sum = DMatrix::<f32>::zeros(g, n_pb);
        let mut cnt = vec![0f32; n_pb];
        for (u, &pb) in unit_to_pb.iter().enumerate() {
            sum.column_mut(pb)
                .add_assign(self.unit_log_fold(u, batch[u]));
            cnt[pb] += 1.0;
        }
        for (pb, &c) in cnt.iter().enumerate() {
            if c > 0.0 {
                sum.column_mut(pb).scale_mut(1.0 / c);
            }
        }
        sum
    }
}

/// What one matched fold reads.
pub struct MatchedFoldInputs<'a> {
    /// Genes × units; only units with a node are read.
    pub data: &'a SparseIoVec,
    pub batch: &'a [usize],
    /// Components × units, batch-corrected.
    pub coords: &'a DMatrix<f32>,
    /// Each unit's node in `0..n_nodes`, `None` to keep δ.
    pub node: &'a [Option<usize>],
    pub n_nodes: usize,
    /// Each node's base fold per batch, log, genes × (nodes · batches): what
    /// its own fold is matched on top of (δ, or the parent node's fold).
    pub base_log_fold: &'a DMatrix<f32>,
    /// δ, log, genes × batches; its columns number the batches.
    pub log_delta: &'a DMatrix<f32>,
    /// Each unit's pseudobulk (a collapse group; crossed with its batch and
    /// node here): matching runs pseudobulk to pseudobulk, as the collapse
    /// matches. `None` matches unit to unit.
    pub pseudobulk_of_unit: Option<&'a [usize]>,
}

/// δ as every node's base: `log_delta` repeated for `n_nodes` nodes.
pub fn delta_as_base_of_every_node(log_delta: &DMatrix<f32>, n_nodes: usize) -> DMatrix<f32> {
    let nb = log_delta.ncols();
    let mut base = DMatrix::<f32>::zeros(log_delta.nrows(), n_nodes * nb);
    for k in 0..n_nodes {
        base.columns_mut(k * nb, nb).copy_from(log_delta);
    }
    base
}

/// The units a fold matches: pseudobulks (or single units) within a node and
/// batch, each with its summed base-adjusted depth and mean coordinates.
struct MatchingUnits {
    /// Each active unit's matching unit.
    of_unit: Vec<usize>,
    node: Vec<usize>,
    batch: Vec<usize>,
    depth: Vec<f32>,
    /// Components × matching units.
    coords: DMatrix<f32>,
}

/// Per active unit: its sum after dividing by its base fold.
fn base_adjusted_depths(
    inp: &MatchedFoldInputs<'_>,
    active: &[usize],
    inv: &DMatrix<f32>,
    block_size: Option<usize>,
) -> anyhow::Result<Vec<f32>> {
    let nb = inp.log_delta.ncols();
    let jobs = generate_minibatch_intervals(active.len(), inp.data.num_rows(), block_size);
    let parts: Vec<(usize, Vec<f32>)> = jobs
        .par_iter()
        .map(|&(lb, ub)| -> anyhow::Result<(usize, Vec<f32>)> {
            let yy = inp.data.read_columns_csc(active[lb..ub].iter().copied())?;
            let d = yy
                .col_iter()
                .zip(&active[lb..ub])
                .map(|(y, &u)| {
                    let col = inp.node[u].expect("active") * nb + inp.batch[u];
                    y.row_indices()
                        .iter()
                        .zip(y.values())
                        .map(|(&g, &x)| x * inv[(g, col)])
                        .sum()
                })
                .collect();
            Ok((lb, d))
        })
        .collect::<anyhow::Result<_>>()?;
    let mut out = vec![0f32; active.len()];
    for (lb, d) in parts {
        out[lb..lb + d.len()].copy_from_slice(&d);
    }
    Ok(out)
}

fn matching_units(inp: &MatchedFoldInputs<'_>, active: &[usize], depth: &[f32]) -> MatchingUnits {
    let mut id: HashMap<(usize, usize, usize), usize> = HashMap::default();
    let mut of_unit = Vec::with_capacity(active.len());
    let (mut node, mut batch, mut msum, mut csum) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::<Vec<f32>>::new());
    let dim = inp.coords.nrows();
    for (i, &u) in active.iter().enumerate() {
        let (k, b) = (inp.node[u].expect("active"), inp.batch[u]);
        let pb = inp.pseudobulk_of_unit.map_or(u, |p| p[u]);
        let m = *id.entry((pb, k, b)).or_insert_with(|| {
            node.push(k);
            batch.push(b);
            msum.push(0f32);
            csum.push(vec![0f32; dim]);
            node.len() - 1
        });
        msum[m] += depth[i];
        for (c, v) in csum[m].iter_mut().zip(inp.coords.column(u).iter()) {
            *c += v;
        }
        of_unit.push(m);
    }
    let mut count = vec![0f32; node.len()];
    for &m in &of_unit {
        count[m] += 1.0;
    }
    let coords = DMatrix::from_fn(dim, node.len(), |r, m| csum[m][r] / count[m].max(1.0));
    MatchingUnits {
        of_unit,
        node,
        batch,
        depth: msum,
        coords,
    }
}

/// Neighbour weights a query chunk may buffer before they are added up.
const QUERY_BUDGET: usize = 1 << 22;

/// One node's counterfactual weights over its matching units,
/// `[local matching unit · nb + b]`: how much each stands in for batch `b`'s
/// counterfactual (each matching unit of `b` draws its `knn` nearest from
/// every other batch, weighted by depth).
fn node_weights(
    mu: &MatchingUnits,
    mk: &[Vec<usize>],
    bs: &[usize],
    local_of: &[usize],
    n_local: usize,
    nb: usize,
    knn: usize,
) -> Vec<f32> {
    let m = (bs.len() - 1) as f32;
    let dicts: HashMap<usize, ColumnDict<usize>> = bs
        .iter()
        .map(|&b2| {
            let pts = DMatrix::from_columns(
                &mk[b2]
                    .iter()
                    .map(|&j| mu.coords.column(j))
                    .collect::<Vec<_>>(),
            );
            (b2, ColumnDict::from_dmatrix(pts, mk[b2].clone()))
        })
        .collect();
    let chunk_len = (QUERY_BUDGET / (knn * bs.len()).max(1)).max(64);
    let mut weight = vec![0f32; n_local * nb];
    for &b in bs {
        for chunk in mk[b].chunks(chunk_len) {
            let adds: Vec<(usize, f32)> = chunk
                .par_iter()
                .flat_map_iter(|&j| {
                    let q: Vec<f32> = mu.coords.column(j).iter().copied().collect();
                    let mut out = Vec::new();
                    for &b2 in bs.iter().filter(|&&b2| b2 != b) {
                        let kq = knn.min(mk[b2].len());
                        let Ok((nbrs, _)) = dicts[&b2].search_by_query_data(&q, kq) else {
                            continue;
                        };
                        // normalised by what came back (HNSW returns at most its beam)
                        let tot: f32 = nbrs.iter().map(|&v| mu.depth[v]).sum();
                        if tot > 0.0 {
                            let w = mu.depth[j] / (m * tot);
                            out.extend(nbrs.into_iter().map(|v| (local_of[v] * nb + b, w)));
                        }
                    }
                    out
                })
                .collect();
            for (slot, w) in adds {
                weight[slot] += w;
            }
        }
    }
    weight
}

/// The matched-counterfactual fold over nodes (see the module docs), on top
/// of each node's base fold: the residual of each node over its base, shrunk
/// toward it by the pseudo-count where the node has little to match.
pub fn estimate_fold_by_cross_batch_matching(
    inp: &MatchedFoldInputs<'_>,
    params: &StateAwareBatchFoldParams,
) -> anyhow::Result<StateAwareBatchFold> {
    let MatchedFoldInputs {
        data,
        batch,
        coords,
        node,
        n_nodes,
        base_log_fold,
        log_delta,
        pseudobulk_of_unit,
    } = *inp;
    let n_units = node.len();
    let nb = log_delta.ncols();
    let n_genes = log_delta.nrows();
    anyhow::ensure!(
        batch.len() == n_units && coords.ncols() == n_units,
        "one batch and coordinate per unit"
    );
    anyhow::ensure!(
        pseudobulk_of_unit.is_none_or(|p| p.len() == n_units),
        "one pseudobulk per unit"
    );
    anyhow::ensure!(
        base_log_fold.ncols() == n_nodes * nb && base_log_fold.nrows() == n_genes,
        "a base fold per (node, batch)"
    );
    anyhow::ensure!(params.knn >= 1, "batch fold: knn must be at least 1");

    let active: Vec<usize> = (0..n_units).filter(|&u| node[u].is_some()).collect();
    let inv = base_log_fold.map(|v| (-v).exp());
    let depth = base_adjusted_depths(inp, &active, &inv, params.block_size)?;
    let mu = matching_units(inp, &active, &depth);

    // members[k][b]: node k's matching units in batch b; units_of_node[k]: its units
    let mut members = vec![vec![Vec::<usize>::new(); nb]; n_nodes];
    for j in 0..mu.node.len() {
        members[mu.node[j]][mu.batch[j]].push(j);
    }
    let mut units_of_node = vec![Vec::<usize>::new(); n_nodes];
    for (i, &u) in active.iter().enumerate() {
        units_of_node[node[u].expect("active")].push(i);
    }

    let mut log_fold = base_log_fold.clone();
    let mut local_of = vec![usize::MAX; mu.node.len()];
    // a node at a time: memory is bounded by the largest node × batches
    for (k, mk) in members.iter().enumerate() {
        let bs: Vec<usize> = (0..nb).filter(|&b| !mk[b].is_empty()).collect();
        if bs.len() < 2 {
            continue;
        }
        let local: Vec<usize> = bs.iter().flat_map(|&b| mk[b].iter().copied()).collect();
        for (l, &j) in local.iter().enumerate() {
            local_of[j] = l;
        }
        let weight = node_weights(&mu, mk, &bs, &local_of, local.len(), nb, params.knn);

        // num: base-adjusted sums per batch; den: their matched counterfactual
        let positions = &units_of_node[k];
        let acc = Mutex::new((
            DMatrix::<f32>::zeros(n_genes, nb),
            DMatrix::<f32>::zeros(n_genes, nb),
        ));
        let jobs =
            generate_minibatch_intervals(positions.len(), data.num_rows(), params.block_size);
        jobs.par_iter()
            .try_for_each(|&(lb, ub)| -> anyhow::Result<()> {
                let yy = data.read_columns_csc(positions[lb..ub].iter().map(|&i| active[i]))?;
                let (mut num_t, mut den_t) = (Vec::new(), Vec::new());
                for (y, &i) in yy.col_iter().zip(&positions[lb..ub]) {
                    let b = batch[active[i]];
                    let l = local_of[mu.of_unit[i]];
                    let w = &weight[l * nb..(l + 1) * nb];
                    for (&g, &x) in y.row_indices().iter().zip(y.values()) {
                        let xt = x * inv[(g, k * nb + b)];
                        num_t.push((g, b, xt));
                        for (b2, &wb) in w.iter().enumerate() {
                            if wb > 0.0 {
                                den_t.push((g, b2, wb * xt));
                            }
                        }
                    }
                }
                let mut guard = acc.lock().expect("fold accumulator");
                for (g, b, v) in num_t {
                    guard.0[(g, b)] += v;
                }
                for (g, b, v) in den_t {
                    guard.1[(g, b)] += v;
                }
                Ok(())
            })?;
        let (num, den) = acc.into_inner().expect("fold accumulator");

        // a batch whose counterfactual drew no mass has nothing to be read against
        let matched: Vec<usize> = bs
            .iter()
            .copied()
            .filter(|&b| den.column(b).sum() > 0.0)
            .collect();
        if matched.len() < 2 {
            warn!("batch fold: node {k} has fewer than 2 matched batches; it keeps its base");
            continue;
        }
        let a = params.pseudo_count;
        for g in 0..n_genes {
            let r: Vec<f32> = matched
                .iter()
                .map(|&b| ((num[(g, b)] + a) / (den[(g, b)] + a)).ln())
                .collect();
            // r_b contrasts batch b with the mean of the others: on exact
            // data, centred, it is n/(n−1) · log f_b over a node's n batches.
            // Kept as is: the matching and the pseudo-count shrink it, and
            // the (n−1)/n rescale did worse on the sims (2026-11-21).
            let rm = r.iter().sum::<f32>() / r.len() as f32;
            for (&b, &rb) in matched.iter().zip(&r) {
                log_fold[(g, k * nb + b)] += rb - rm;
            }
        }
    }

    Ok(StateAwareBatchFold {
        n_batches: nb,
        n_nodes,
        log_fold,
        log_delta: log_delta.clone(),
        node_of_unit: node.to_vec(),
    })
}
