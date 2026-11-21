//! The fold itself, for either kind of unit: within each node, every unit is
//! matched to its nearest units of every other batch (in a batch-corrected
//! projection), and the node's fold for a batch is its δ-adjusted sum over the
//! matched counterfactual sum, centred over the node's batches, on top of δ.

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

/// Per column of `data`: its sum after dividing by its base fold, the column
/// `base_col[c]` of `inv` (1 / base, genes × (nodes · batches)).
fn adjusted_depths(
    data: &SparseIoVec,
    cols: &[usize],
    base_col: &[usize],
    inv: &DMatrix<f32>,
    block_size: Option<usize>,
) -> anyhow::Result<Vec<f32>> {
    let jobs = generate_minibatch_intervals(cols.len(), data.num_rows(), block_size);
    let parts: Vec<(usize, Vec<f32>)> = jobs
        .par_iter()
        .map(|&(lb, ub)| -> anyhow::Result<(usize, Vec<f32>)> {
            let yy = data.read_columns_csc(cols[lb..ub].iter().copied())?;
            let d = yy
                .col_iter()
                .zip(&cols[lb..ub])
                .map(|(y, &c)| {
                    let col = base_col[c];
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
    let mut out = vec![0f32; cols.len()];
    for (lb, d) in parts {
        out[lb..lb + d.len()].copy_from_slice(&d);
    }
    Ok(out)
}

/// Neighbour weights a query chunk may buffer before they are added up.
const QUERY_BUDGET: usize = 1 << 22;

/// What every node of one matched fold reads.
struct FoldCtx<'a> {
    data: &'a SparseIoVec,
    batch: &'a [usize],
    coords: &'a DMatrix<f32>,
    /// Units with a node, by position.
    active: &'a [usize],
    /// Base-adjusted depth of each active unit.
    depth: &'a [f32],
    /// 1 / base fold, genes × (nodes · batches).
    inv: &'a DMatrix<f32>,
    nb: usize,
    knn: usize,
    block_size: Option<usize>,
}

/// One node's counterfactual weights, `[local unit · nb + b]`: how much each
/// of its units stands in for batch `b`'s counterfactual (each unit of `b`
/// draws its `knn` nearest from every other batch, weighted by depth).
fn node_weights(
    ctx: &FoldCtx<'_>,
    mk: &[Vec<usize>],
    bs: &[usize],
    units_k: &[usize],
    local_of: &[usize],
) -> Vec<f32> {
    let FoldCtx {
        coords,
        active,
        depth,
        nb,
        knn,
        ..
    } = *ctx;
    let m = (bs.len() - 1) as f32;
    let dicts: HashMap<usize, ColumnDict<usize>> = bs
        .iter()
        .map(|&b2| {
            let pts = DMatrix::from_columns(
                &mk[b2]
                    .iter()
                    .map(|&i| coords.column(active[i]))
                    .collect::<Vec<_>>(),
            );
            (b2, ColumnDict::from_dmatrix(pts, mk[b2].clone()))
        })
        .collect();
    let chunk_len = (QUERY_BUDGET / (knn * bs.len()).max(1)).max(64);
    let mut weight = vec![0f32; units_k.len() * nb];
    for &b in bs {
        for chunk in mk[b].chunks(chunk_len) {
            let adds: Vec<(usize, f32)> = chunk
                .par_iter()
                .flat_map_iter(|&i| {
                    let q: Vec<f32> = coords.column(active[i]).iter().copied().collect();
                    let mut out = Vec::new();
                    for &b2 in bs.iter().filter(|&&b2| b2 != b) {
                        let kq = knn.min(mk[b2].len());
                        let Ok((nbrs, _)) = dicts[&b2].search_by_query_data(&q, kq) else {
                            continue;
                        };
                        // normalised by what came back (HNSW returns at most its beam)
                        let tot: f32 = nbrs.iter().map(|&v| depth[v]).sum();
                        if tot > 0.0 {
                            let w = depth[i] / (m * tot);
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

/// One node's δ-adjusted sums per batch (`num`, genes × nb) and their
/// matched counterfactual (`den`), reading only the node's units.
fn node_sums(
    ctx: &FoldCtx<'_>,
    k: usize,
    units_k: &[usize],
    weight: &[f32],
) -> anyhow::Result<(DMatrix<f32>, DMatrix<f32>)> {
    let FoldCtx {
        data,
        batch,
        active,
        inv,
        nb,
        block_size,
        ..
    } = *ctx;
    let n_genes = inv.nrows();
    let acc = Mutex::new((
        DMatrix::<f32>::zeros(n_genes, nb),
        DMatrix::<f32>::zeros(n_genes, nb),
    ));
    let jobs = generate_minibatch_intervals(units_k.len(), data.num_rows(), block_size);
    jobs.par_iter()
        .try_for_each(|&(lb, ub)| -> anyhow::Result<()> {
            let yy = data.read_columns_csc(units_k[lb..ub].iter().map(|&i| active[i]))?;
            let mut num_t = Vec::new();
            let mut den_t = Vec::new();
            for (y, l) in yy.col_iter().zip(lb..ub) {
                let b = batch[active[units_k[l]]];
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
    Ok(acc.into_inner().expect("fold accumulator"))
}

/// The matched-counterfactual fold over nodes (see the module docs), on top
/// of δ.
///
/// * `data`: genes × units; only units with a node are read.
/// * `coords`: components × units, batch-corrected.
/// * `node`: each unit's node in `0..n_nodes`, `None` to keep δ.
/// * `log_delta`: genes × batches; its columns number the batches.
pub fn estimate_fold_by_cross_batch_matching(
    data: &SparseIoVec,
    batch: &[usize],
    coords: &DMatrix<f32>,
    node: &[Option<usize>],
    n_nodes: usize,
    log_delta: &DMatrix<f32>,
    params: &StateAwareBatchFoldParams,
) -> anyhow::Result<StateAwareBatchFold> {
    let nb = log_delta.ncols();
    let mut base = DMatrix::<f32>::zeros(log_delta.nrows(), n_nodes * nb);
    for k in 0..n_nodes {
        base.columns_mut(k * nb, nb).copy_from(log_delta);
    }
    estimate_residual_fold_by_cross_batch_matching(
        data,
        batch,
        coords,
        node,
        n_nodes,
        &ResidualBase {
            base_log_fold: &base,
            log_delta,
        },
        params,
    )
}

/// What a residual matched fold reads besides the units: each node's base
/// fold per batch (log, genes × (nodes · batches)), what the node's own
/// residual is matched on top of, and δ (for the units without a node).
pub(crate) struct ResidualBase<'a> {
    pub base_log_fold: &'a DMatrix<f32>,
    pub log_delta: &'a DMatrix<f32>,
}

/// [`estimate_fold_by_cross_batch_matching`] on top of a base fold per
/// (node, batch) instead of δ: the residual of each node over its base, shrunk
/// toward it by the pseudo-count where the node has little to match.
pub(crate) fn estimate_residual_fold_by_cross_batch_matching(
    data: &SparseIoVec,
    batch: &[usize],
    coords: &DMatrix<f32>,
    node: &[Option<usize>],
    n_nodes: usize,
    base: &ResidualBase<'_>,
    params: &StateAwareBatchFoldParams,
) -> anyhow::Result<StateAwareBatchFold> {
    let ResidualBase {
        base_log_fold,
        log_delta,
    } = *base;
    let n_units = node.len();
    let nb = log_delta.ncols();
    anyhow::ensure!(
        batch.len() == n_units && coords.ncols() == n_units,
        "one batch and coordinate per unit"
    );
    anyhow::ensure!(
        base_log_fold.ncols() == n_nodes * nb && base_log_fold.nrows() == log_delta.nrows(),
        "a base fold per (node, batch)"
    );
    let n_genes = log_delta.nrows();
    anyhow::ensure!(params.knn >= 1, "group fold: knn must be at least 1");

    let active: Vec<usize> = (0..n_units).filter(|&u| node[u].is_some()).collect();
    let inv = base_log_fold.map(|v| (-v).exp());
    let mut base_col = vec![0usize; n_units];
    for &u in &active {
        base_col[u] = node[u].expect("active") * nb + batch[u];
    }
    let depth = adjusted_depths(data, &active, &base_col, &inv, params.block_size)?;

    // members[k][b]: positions (into `active`) of node k's units in batch b
    let mut members = vec![vec![Vec::<usize>::new(); nb]; n_nodes];
    for (i, &u) in active.iter().enumerate() {
        if let Some(k) = node[u] {
            members[k][batch[u]].push(i);
        }
    }

    let mut log_fold = base_log_fold.clone();
    let ctx = FoldCtx {
        data,
        batch,
        coords,
        active: &active,
        depth: &depth,
        inv: &inv,
        nb,
        knn: params.knn,
        block_size: params.block_size,
    };
    // a node at a time: memory is bounded by the largest node × batches
    let mut local_of = vec![usize::MAX; active.len()];
    for (k, mk) in members.iter().enumerate() {
        let bs: Vec<usize> = (0..nb).filter(|&b| !mk[b].is_empty()).collect();
        if bs.len() < 2 {
            continue;
        }
        let units_k: Vec<usize> = bs.iter().flat_map(|&b| mk[b].iter().copied()).collect();
        for (l, &i) in units_k.iter().enumerate() {
            local_of[i] = l;
        }
        let weight = node_weights(&ctx, mk, &bs, &units_k, &local_of);
        let (num, den) = node_sums(&ctx, k, &units_k, &weight)?;

        // a batch whose counterfactual drew no mass has nothing to be read against
        let matched: Vec<usize> = bs
            .iter()
            .copied()
            .filter(|&b| den.column(b).sum() > 0.0)
            .collect();
        if matched.len() < 2 {
            warn!("group fold: node {k} has fewer than 2 matched batches; it keeps δ");
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
        log::debug!(
            "group fold: node {k} matched over {} batches",
            matched.len()
        );
    }

    Ok(StateAwareBatchFold {
        n_batches: nb,
        n_nodes,
        log_fold,
        log_delta: log_delta.clone(),
        node_of_unit: node.to_vec(),
    })
}
