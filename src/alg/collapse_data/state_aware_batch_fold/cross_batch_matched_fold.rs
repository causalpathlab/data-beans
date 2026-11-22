//! The fold itself, for either kind of unit, on the pseudobulk sums: within
//! each node, every pseudobulk (pb sample, or pb pair) is matched to its
//! nearest pseudobulks of every other batch (in a batch-corrected
//! projection), and the node's fold for a batch is its base-adjusted sum over
//! the matched counterfactual sum, centred over the node's batches, on top of
//! the node's base (δ, or its parent's fold).

use super::pseudobulk_sums::PseudobulkSums;
use super::{StateAwareBatchFoldParams, COORD_DIM};
use legume_numeric::matrix::knn_match::ColumnDict;
use legume_numeric::matrix::traits::SymmetricTopEigen;
use log::warn;
use nalgebra::DMatrix;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use std::ops::AddAssign;

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
    pub fn fit(proj: &DMatrix<f32>) -> anyhow::Result<Self> {
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
        let basis = cov.top_eigen(COORD_DIM.min(dim))?.vectors;
        Ok(Self { mean, scale, basis })
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

    /// One level of a collapse: each pseudobulk sample's log fold, the mean
    /// over its units, genes × (n_groups · batches), column `group · nb + b`
    /// for the level's group `unit_to_group[u]` in batch `b` (a group is
    /// shared across batches; its sample in each batch has its own fold). A
    /// sample with no units holds log δ of its batch. Its units then come out
    /// unbiased on average, as δ does.
    pub fn per_level(
        &self,
        unit_to_group: &[usize],
        batch: &[usize],
        n_groups: usize,
    ) -> DMatrix<f32> {
        let (g, nb) = (self.log_delta.nrows(), self.n_batches);
        let mut sum = DMatrix::<f32>::zeros(g, n_groups * nb);
        let mut cnt = vec![0f32; n_groups * nb];
        for (u, &grp) in unit_to_group.iter().enumerate() {
            let col = grp * nb + batch[u];
            sum.column_mut(col)
                .add_assign(self.unit_log_fold(u, batch[u]));
            cnt[col] += 1.0;
        }
        for (col, &c) in cnt.iter().enumerate() {
            if c > 0.0 {
                sum.column_mut(col).scale_mut(1.0 / c);
            } else {
                sum.column_mut(col)
                    .copy_from(&self.log_delta.column(col % nb));
            }
        }
        sum
    }
}

/// δ as every node's base: `log_delta` repeated for `n_nodes` nodes.
pub(crate) fn delta_as_base_of_every_node(
    log_delta: &DMatrix<f32>,
    n_nodes: usize,
) -> DMatrix<f32> {
    let nb = log_delta.ncols();
    let mut base = DMatrix::<f32>::zeros(log_delta.nrows(), n_nodes * nb);
    for k in 0..n_nodes {
        base.columns_mut(k * nb, nb).copy_from(log_delta);
    }
    base
}

/// Each pseudobulk of one node, by batch, and the node's base in each batch.
struct NodeMembers<'a> {
    sums: &'a PseudobulkSums,
    by_batch: Vec<Vec<usize>>,
    /// 1 / base, genes × batches.
    inv: DMatrix<f32>,
}

impl NodeMembers<'_> {
    /// A pseudobulk's base-adjusted depth.
    fn depth(&self, pb: usize) -> f32 {
        let b = self.sums.batch[pb];
        self.sums.counts[pb]
            .iter()
            .map(|&(g, x)| x * self.inv[(g, b)])
            .sum()
    }

    /// Each pseudobulk's counterfactual weight per batch (`weight[pb][b]`):
    /// how much it stands in for batch `b`, each pseudobulk of `b` drawing its
    /// `knn` nearest of every other batch, weighted by depth and averaged over
    /// the batches that returned any.
    fn weights(&self, bs: &[usize], knn: usize) -> HashMap<usize, Vec<f32>> {
        let nb = self.by_batch.len();
        let depth: HashMap<usize, f32> = bs
            .iter()
            .flat_map(|&b| self.by_batch[b].iter().map(|&pb| (pb, self.depth(pb))))
            .collect();
        let dicts: HashMap<usize, ColumnDict<usize>> = bs
            .iter()
            .map(|&b| {
                let pts = DMatrix::from_columns(
                    &self.by_batch[b]
                        .iter()
                        .map(|&pb| self.sums.coords.column(pb))
                        .collect::<Vec<_>>(),
                );
                (b, ColumnDict::from_dmatrix(pts, self.by_batch[b].clone()))
            })
            .collect();
        let adds: Vec<(usize, usize, f32)> = bs
            .par_iter()
            .flat_map_iter(|&b| self.by_batch[b].iter().map(move |&pb| (b, pb)))
            .flat_map_iter(|(b, pb)| {
                let q = self.sums.coords.column(pb);
                let found: Vec<(Vec<usize>, f32)> = bs
                    .iter()
                    .filter(|&&b2| b2 != b)
                    .filter_map(|&b2| {
                        let kq = knn.min(self.by_batch[b2].len());
                        match dicts[&b2].search_by_query_data(q.as_slice(), kq) {
                            Ok((nbrs, _)) => {
                                let tot: f32 = nbrs.iter().map(|v| depth[v]).sum();
                                (tot > 0.0).then_some((nbrs, tot))
                            }
                            Err(e) => {
                                warn!("batch fold: a cross-batch search failed: {e}");
                                None
                            }
                        }
                    })
                    .collect();
                let m = found.len() as f32;
                let own = depth[&pb];
                found.into_iter().flat_map(move |(nbrs, tot)| {
                    nbrs.into_iter().map(move |v| (v, b, own / (m * tot)))
                })
            })
            .collect();
        let mut weight: HashMap<usize, Vec<f32>> = HashMap::default();
        for (v, b, w) in adds {
            weight.entry(v).or_insert_with(|| vec![0.0; nb])[b] += w;
        }
        weight
    }
}

/// The matched-counterfactual fold over nodes, on the pseudobulk sums:
/// `node_of_pb[pb]` its node (`None`: left out), `base_log_fold` each
/// node's base per batch (log, genes × (n_nodes · batches)). Returns each
/// node's fold, the base plus its residual, shrunk toward the base by the
/// pseudo-count where the node has little to match.
pub(crate) fn estimate_fold_on_pseudobulks(
    sums: &PseudobulkSums,
    node_of_pb: &[Option<usize>],
    n_nodes: usize,
    base_log_fold: &DMatrix<f32>,
    params: &StateAwareBatchFoldParams,
) -> anyhow::Result<DMatrix<f32>> {
    let n_genes = base_log_fold.nrows();
    anyhow::ensure!(
        n_nodes > 0 || node_of_pb.iter().all(Option::is_none),
        "no nodes"
    );
    let nb = base_log_fold.ncols() / n_nodes.max(1);
    anyhow::ensure!(params.knn >= 1, "batch fold: knn must be at least 1");
    anyhow::ensure!(node_of_pb.len() == sums.len(), "one node per pseudobulk");

    let mut members: Vec<Vec<Vec<usize>>> = vec![vec![Vec::new(); nb]; n_nodes];
    for (pb, k) in node_of_pb.iter().enumerate() {
        if let Some(k) = *k {
            members[k][sums.batch[pb]].push(pb);
        }
    }
    let residuals: Vec<Option<DMatrix<f32>>> = members
        .into_par_iter()
        .enumerate()
        .map(|(k, by_batch)| {
            let bs: Vec<usize> = (0..nb).filter(|&b| !by_batch[b].is_empty()).collect();
            if bs.len() < 2 {
                return None;
            }
            let node = NodeMembers {
                sums,
                inv: base_log_fold.columns(k * nb, nb).map(|v| (-v).exp()),
                by_batch,
            };
            let weight = node.weights(&bs, params.knn);
            // num: base-adjusted sums per batch; den: their matched counterfactual
            let mut num = DMatrix::<f32>::zeros(n_genes, nb);
            let mut den = DMatrix::<f32>::zeros(n_genes, nb);
            for &b in &bs {
                for &pb in &node.by_batch[b] {
                    let w = weight.get(&pb);
                    for &(g, x) in &sums.counts[pb] {
                        let xt = x * node.inv[(g, b)];
                        num[(g, b)] += xt;
                        if let Some(w) = w {
                            for (b2, &wb) in w.iter().enumerate() {
                                if wb > 0.0 {
                                    den[(g, b2)] += wb * xt;
                                }
                            }
                        }
                    }
                }
            }
            centred_log_ratio(&num, &den, &bs, params.pseudo_count, k)
        })
        .collect();

    let mut log_fold = base_log_fold.clone();
    for (k, r) in residuals.into_iter().enumerate() {
        if let Some(r) = r {
            log_fold.columns_mut(k * nb, nb).add_assign(&r);
        }
    }
    Ok(log_fold)
}

/// A node's residual log fold per batch (genes × batches): `ln((num + a) /
/// (den + a))` for each batch whose counterfactual drew mass, centred over
/// them. `None` when fewer than two batches drew any.
fn centred_log_ratio(
    num: &DMatrix<f32>,
    den: &DMatrix<f32>,
    bs: &[usize],
    a: f32,
    node: usize,
) -> Option<DMatrix<f32>> {
    let matched: Vec<usize> = bs
        .iter()
        .copied()
        .filter(|&b| den.column(b).sum() > 0.0)
        .collect();
    if matched.len() < 2 {
        warn!("batch fold: node {node} has fewer than 2 matched batches; it keeps its base");
        return None;
    }
    let mut r = DMatrix::<f32>::zeros(num.nrows(), num.ncols());
    for g in 0..num.nrows() {
        let lr: Vec<f32> = matched
            .iter()
            .map(|&b| ((num[(g, b)] + a) / (den[(g, b)] + a)).ln())
            .collect();
        // r_b contrasts batch b with the mean of the others: on exact data,
        // centred, it is n/(n−1) · log f_b over a node's n batches. Kept as
        // is: the matching and the pseudo-count shrink it, and the (n−1)/n
        // rescale did worse on the sims (2026-11-21).
        let mean = lr.iter().sum::<f32>() / lr.len() as f32;
        for (&b, &v) in matched.iter().zip(&lr) {
            r[(g, b)] = v - mean;
        }
    }
    Some(r)
}
