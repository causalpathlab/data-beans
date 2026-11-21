//! State-aware batch fold: one fold per (gene × batch × coarse node), beside δ.
//!
//! δ (gene × batch) cannot hold the part of a batch effect that depends on the
//! cell state (donor × state), and that part alone keeps batches apart in every
//! local neighbourhood. This module estimates it on coarse nodes that are
//! state-pure **and** span batches:
//!
//! 1. **Tree.** Ward linkage over the collapse's coarsest groups, on their
//!    δ-adjusted pooled profiles ([`CoarseTree::ward`]).
//! 2. **Cut.** Walk down from the root; split a node only while both children
//!    span enough batches (or donors) with enough units each
//!    ([`CoarseTree::cut`]). No user-facing count of nodes.
//! 3. **Matched fold.** Within a node, each unit is matched to its nearest
//!    units of every other batch in a batch-corrected projection; the node's
//!    fold for batch `b` is its δ-adjusted sum over the matched counterfactual
//!    sum, centred over the node's batches, on top of δ.
//!
//! Two kinds of unit go through the same estimator:
//!
//! * **samples** ([`estimate_sample_fold`]): columns as they are (cells);
//! * **graph** ([`estimate_pair_fold`]): pair units, each cell's reads split
//!   once over its pairs and one self-loop
//!   ([`crate::alg::cell_pairs::split_reads_over_pair_units`]). A pair's node
//!   is the unordered pair of its endpoints' cell nodes; a pair node that
//!   fails the spanning rule climbs both endpoints to their parents.
//!
//! If not even the root spans enough batches, every unit keeps δ, with a
//! warning; the estimate never fails for that reason.

use crate::sparse_io_vector::SparseIoVec;
use legume_numeric::matrix::knn_match::ColumnDict;
use legume_numeric::matrix::traits::RandomizedAlgs;
use legume_numeric::matrix::utils::generate_minibatch_intervals;
use log::{info, warn};
use nalgebra::{DMatrix, SymmetricEigen};
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use std::ops::AddAssign;
use std::sync::Mutex;

/// Nearest units drawn from each other batch for a unit's counterfactual.
pub const DEFAULT_FOLD_KNN: usize = 10;
/// Pseudo-count on both sides of a node's fold ratio (shrinks thin genes to 1).
pub const DEFAULT_FOLD_PSEUDO_COUNT: f32 = 10.0;
/// Units a batch (or donor) needs inside a node to count as spanned.
pub const DEFAULT_MIN_UNITS: usize = 20;
/// Principal components of the projection the matching runs in.
const COORD_DIM: usize = 30;
/// Principal components of the group profiles Ward links.
const PROFILE_DIM: usize = 20;
/// Queries matched per parallel chunk before their weights are accumulated.
const QUERY_CHUNK: usize = 4096;

#[derive(Clone, Debug)]
pub struct GroupFoldParams {
    pub knn: usize,
    pub pseudo_count: f32,
    pub min_units: usize,
    /// Label a node must span, per column of the samples (cells): a donor,
    /// say. `None` spans batches.
    pub span_labels: Option<Vec<usize>>,
    /// Distinct labels a node must span, each with `min_units` units. `None`:
    /// half the batches (rounded up), or 2 donors when `span_labels` is set.
    pub min_spans: Option<usize>,
    pub block_size: Option<usize>,
}

impl Default for GroupFoldParams {
    fn default() -> Self {
        Self {
            knn: DEFAULT_FOLD_KNN,
            pseudo_count: DEFAULT_FOLD_PSEUDO_COUNT,
            min_units: DEFAULT_MIN_UNITS,
            span_labels: None,
            min_spans: None,
            block_size: None,
        }
    }
}

/// Which label each unit spans with, how many there are, and how many a node needs.
struct SpanRule {
    label: Vec<usize>,
    n_labels: usize,
    min_spans: usize,
    min_units: usize,
}

impl SpanRule {
    fn new(params: &GroupFoldParams, batch: &[usize], n_batches: usize) -> anyhow::Result<Self> {
        let (label, n_labels, default_min) = match &params.span_labels {
            Some(l) => {
                anyhow::ensure!(l.len() == batch.len(), "span labels: one per column");
                let n = l.iter().max().map_or(0, |&m| m + 1);
                (l.clone(), n, 2)
            }
            None => (batch.to_vec(), n_batches, n_batches.div_ceil(2)),
        };
        Ok(Self {
            label,
            n_labels,
            min_spans: params.min_spans.unwrap_or(default_min).max(2),
            min_units: params.min_units,
        })
    }

    /// Does a node holding `count[label]` units span enough labels?
    fn spans(&self, count: &[usize]) -> bool {
        count.iter().filter(|&&c| c >= self.min_units).count() >= self.min_spans
    }
}

////////////////////////////////////////////////////////////////////
// Ward tree over the coarse groups
////////////////////////////////////////////////////////////////////

/// A binary tree over `n_groups` leaves (ids `0..n_groups`); internal nodes
/// take ids `n_groups..2·n_groups-1` in merge order; the root is the last.
#[derive(Clone, Debug)]
pub struct CoarseTree {
    pub n_groups: usize,
    /// `parent[node]`; the root is its own parent.
    pub parent: Vec<usize>,
    /// `children[node]` for an internal node.
    pub children: Vec<Option<(usize, usize)>>,
    pub root: usize,
}

impl CoarseTree {
    /// Ward linkage over the rows of `points` (groups × dims), by the
    /// nearest-neighbour chain with Lance–Williams updates on squared
    /// distances: the same tree as greedy Ward, in O(K²) time and memory for
    /// K groups.
    pub fn ward(points: &DMatrix<f32>) -> Self {
        let k = points.nrows();
        let total = (2 * k).saturating_sub(1).max(1);
        let mut parent: Vec<usize> = (0..total).collect();
        let mut children = vec![None; total];
        if k <= 1 {
            return Self {
                n_groups: k,
                parent,
                children,
                root: 0,
            };
        }
        let mut d = vec![0f64; k * k];
        d.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
            for (j, dij) in row.iter_mut().enumerate() {
                *dij = (0..points.ncols())
                    .map(|c| f64::from(points[(i, c)] - points[(j, c)]).powi(2))
                    .sum();
            }
        });
        let mut size = vec![1f64; k];
        let mut id: Vec<usize> = (0..k).collect();
        let mut active: Vec<bool> = vec![true; k];
        let mut chain: Vec<usize> = Vec::with_capacity(k);
        let mut next_id = k;
        let mut n_active = k;
        while n_active > 1 {
            if chain.is_empty() {
                chain.push(active.iter().position(|&a| a).expect("an active group"));
            }
            let a = *chain.last().expect("chain");
            let prev = chain.len().checked_sub(2).map(|i| chain[i]);
            // nearest active to `a`, preferring the previous link on ties so the chain ends
            let mut best = (usize::MAX, f64::INFINITY);
            for m in (0..k).filter(|&m| active[m] && m != a) {
                let v = d[a * k + m];
                if v < best.1 || (v == best.1 && Some(m) == prev) {
                    best = (m, v);
                }
            }
            let b = best.0;
            if Some(b) != prev {
                chain.push(b);
                continue;
            }
            chain.truncate(chain.len() - 2);
            // merge a and b into a's slot
            parent[id[a]] = next_id;
            parent[id[b]] = next_id;
            children[next_id] = Some((id[a], id[b]));
            let (na, nb, dab) = (size[a], size[b], d[a * k + b]);
            for m in (0..k).filter(|&m| active[m] && m != a && m != b) {
                let nm = size[m];
                let v = ((na + nm) * d[a * k + m] + (nb + nm) * d[b * k + m] - nm * dab)
                    / (na + nb + nm);
                d[a * k + m] = v;
                d[m * k + a] = v;
            }
            size[a] = na + nb;
            id[a] = next_id;
            active[b] = false;
            next_id += 1;
            n_active -= 1;
        }
        let root = total - 1;
        parent[root] = root;
        Self {
            n_groups: k,
            parent,
            children,
            root,
        }
    }

    /// Per tree node, the units of each label below it.
    fn subtree_counts(&self, group_count: &[Vec<usize>], n_labels: usize) -> Vec<Vec<usize>> {
        let mut sub = vec![vec![0usize; n_labels]; self.parent.len()];
        for (g, c) in group_count.iter().enumerate() {
            sub[g].clone_from(c);
        }
        // internal ids grow in merge order, so children come first
        for node in self.n_groups..self.parent.len() {
            if let Some((a, b)) = self.children[node] {
                for l in 0..n_labels {
                    sub[node][l] = sub[a][l] + sub[b][l];
                }
            }
        }
        sub
    }

    /// The tree node of every group after the batch-spanning cut: a node is
    /// split only if both its children span. `None` for every group when not
    /// even the root spans.
    fn cut(&self, group_count: &[Vec<usize>], rule: &SpanRule) -> Vec<Option<usize>> {
        let sub = self.subtree_counts(group_count, rule.n_labels);
        let mut out = vec![None; self.n_groups];
        if !rule.spans(&sub[self.root]) {
            return out;
        }
        let mut stack = vec![self.root];
        while let Some(nd) = stack.pop() {
            match self.children[nd] {
                Some((a, b)) if rule.spans(&sub[a]) && rule.spans(&sub[b]) => {
                    stack.push(a);
                    stack.push(b);
                }
                _ => {
                    for g in self.leaves(nd) {
                        out[g] = Some(nd);
                    }
                }
            }
        }
        out
    }

    fn leaves(&self, node: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut stack = vec![node];
        while let Some(nd) = stack.pop() {
            match self.children[nd] {
                Some((a, b)) => {
                    stack.push(a);
                    stack.push(b);
                }
                None => out.push(nd),
            }
        }
        out
    }
}

////////////////////////////////////////////////////////////////////
// coordinates the matching runs in
////////////////////////////////////////////////////////////////////

/// Standardize each projection coordinate, then keep the leading principal
/// components: `coords = basisᵀ ((proj − mean) / scale)`.
#[derive(Clone, Debug)]
pub struct CoordMap {
    pub mean: Vec<f32>,
    pub scale: Vec<f32>,
    /// dims × components
    pub basis: DMatrix<f32>,
}

impl CoordMap {
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

/// Principal-component scores (rows × k) of the rows of `x`, centred.
fn row_scores(x: &DMatrix<f32>, k: usize) -> DMatrix<f32> {
    let mut xc = x.clone();
    for c in 0..x.ncols() {
        let m = x.column(c).mean();
        xc.column_mut(c).add_scalar_mut(-m);
    }
    let k = k.min(x.nrows()).min(x.ncols());
    match xc.rsvd(k) {
        // randomized SVD: O(rows · cols · k), not cubic in the rows
        Ok((u, dv, _)) => DMatrix::from_fn(u.nrows(), u.ncols(), |r, c| u[(r, c)] * dv[c]),
        Err(_) => DMatrix::zeros(x.nrows(), k),
    }
}

////////////////////////////////////////////////////////////////////
// the fold
////////////////////////////////////////////////////////////////////

/// A fold per (gene × node × batch) on top of δ.
#[derive(Clone, Debug)]
pub struct GroupFold {
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

impl GroupFold {
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

/// The samples kind's result: the fold, plus what places an unseen unit.
#[derive(Clone, Debug)]
pub struct SampleFold {
    pub fold: GroupFold,
    pub tree: CoarseTree,
    /// Each unit's node in `tree` (before the dense node numbering), `None`
    /// when it keeps δ. The graph kind reads this for its endpoints.
    pub tree_node_of_unit: Vec<Option<usize>>,
    pub coord_map: CoordMap,
    /// Each unit's matching coordinates (components × units).
    pub coords: DMatrix<f32>,
    /// components × n_nodes: the mean coordinates of each node's units; an
    /// unseen unit takes its nearest node.
    pub node_centroids: DMatrix<f32>,
}

impl SampleFold {
    /// The node of an unseen unit with projection `proj` (dims): the nearest
    /// centroid.
    pub fn place(&self, proj: &DMatrix<f32>) -> Vec<Option<usize>> {
        let z = self.coord_map.apply(proj);
        (0..z.ncols())
            .map(|c| {
                (0..self.node_centroids.ncols())
                    .map(|k| {
                        (
                            k,
                            (z.column(c) - self.node_centroids.column(k)).norm_squared(),
                        )
                    })
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(k, _)| k)
            })
            .collect()
    }
}

/// Per column of `data`: its sum after dividing by δ of its batch.
fn adjusted_depths(
    data: &SparseIoVec,
    cols: &[usize],
    batch: &[usize],
    log_delta: &DMatrix<f32>,
    block_size: Option<usize>,
) -> anyhow::Result<Vec<f32>> {
    let inv = log_delta.map(|v| (-v).exp());
    let jobs = generate_minibatch_intervals(cols.len(), data.num_rows(), block_size);
    let parts: Vec<(usize, Vec<f32>)> = jobs
        .par_iter()
        .map(|&(lb, ub)| -> anyhow::Result<(usize, Vec<f32>)> {
            let yy = data.read_columns_csc(cols[lb..ub].iter().copied())?;
            let d = yy
                .col_iter()
                .zip(&cols[lb..ub])
                .map(|(y, &c)| {
                    let b = batch[c];
                    y.row_indices()
                        .iter()
                        .zip(y.values())
                        .map(|(&g, &x)| x * inv[(g, b)])
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

/// The matched-counterfactual fold over nodes (see the module docs).
///
/// * `data`: genes × units; only units with a node are read.
/// * `coords`: components × units, batch-corrected.
/// * `node`: each unit's node in `0..n_nodes`, `None` to keep δ.
/// * `log_delta`: genes × batches; its columns number the batches.
pub fn matched_fold(
    data: &SparseIoVec,
    batch: &[usize],
    coords: &DMatrix<f32>,
    node: &[Option<usize>],
    n_nodes: usize,
    log_delta: &DMatrix<f32>,
    params: &GroupFoldParams,
) -> anyhow::Result<GroupFold> {
    let n_units = node.len();
    let n_batches = log_delta.ncols();
    anyhow::ensure!(
        batch.len() == n_units && coords.ncols() == n_units,
        "one batch and coordinate per unit"
    );
    let n_genes = log_delta.nrows();
    let nb = n_batches;

    let active: Vec<usize> = (0..n_units).filter(|&u| node[u].is_some()).collect();
    let mut pos_of = vec![usize::MAX; n_units];
    for (i, &u) in active.iter().enumerate() {
        pos_of[u] = i;
    }
    let depth = adjusted_depths(data, &active, batch, log_delta, params.block_size)?;

    // members[k][b]: positions (into `active`) of node k's units in batch b
    let mut members = vec![vec![Vec::<usize>::new(); nb]; n_nodes];
    for (i, &u) in active.iter().enumerate() {
        if let Some(k) = node[u] {
            members[k][batch[u]].push(i);
        }
    }

    // weight[i·nb + b]: how much unit i stands in for batch b's counterfactual in its node
    let mut weight = vec![0f32; active.len() * nb];
    for (k, mk) in members.iter().enumerate() {
        let bs: Vec<usize> = (0..nb).filter(|&b| !mk[b].is_empty()).collect();
        if bs.len() < 2 {
            continue;
        }
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
        for &b in &bs {
            for chunk in mk[b].chunks(QUERY_CHUNK) {
                let adds: Vec<(usize, f32)> = chunk
                    .par_iter()
                    .flat_map_iter(|&i| {
                        let q: Vec<f32> = coords.column(active[i]).iter().copied().collect();
                        let mut out = Vec::new();
                        for &b2 in &bs {
                            if b2 == b {
                                continue;
                            }
                            let kq = params.knn.min(mk[b2].len());
                            let Ok((nbrs, _)) = dicts[&b2].search_by_query_data(&q, kq) else {
                                continue;
                            };
                            let tot: f32 = nbrs.iter().map(|&v| depth[v]).sum();
                            if tot > 0.0 {
                                let w = depth[i] / (m * tot);
                                out.extend(nbrs.into_iter().map(|v| (v * nb + b, w)));
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
        log::debug!("group fold: node {k} matched over {} batches", bs.len());
    }

    // num: δ-adjusted sums per (node, batch); den: their matched counterfactual
    let n_slots = n_nodes * nb;
    let inv = log_delta.map(|v| (-v).exp());
    let acc = Mutex::new((
        DMatrix::<f32>::zeros(n_genes, n_slots),
        DMatrix::<f32>::zeros(n_genes, n_slots),
    ));
    let jobs = generate_minibatch_intervals(active.len(), data.num_rows(), params.block_size);
    jobs.par_iter()
        .try_for_each(|&(lb, ub)| -> anyhow::Result<()> {
            let yy = data.read_columns_csc(active[lb..ub].iter().copied())?;
            let mut num_t = Vec::new();
            let mut den_t = Vec::new();
            for (y, i) in yy.col_iter().zip(lb..ub) {
                let u = active[i];
                let (k, b) = (node[u].expect("active unit"), batch[u]);
                let w = &weight[i * nb..(i + 1) * nb];
                for (&g, &x) in y.row_indices().iter().zip(y.values()) {
                    let xt = x * inv[(g, b)];
                    num_t.push((g, k * nb + b, xt));
                    for (b2, &wb) in w.iter().enumerate() {
                        if wb > 0.0 {
                            den_t.push((g, k * nb + b2, wb * xt));
                        }
                    }
                }
            }
            let mut guard = acc.lock().expect("fold accumulator");
            for (g, s, v) in num_t {
                guard.0[(g, s)] += v;
            }
            for (g, s, v) in den_t {
                guard.1[(g, s)] += v;
            }
            Ok(())
        })?;
    let (num, den) = acc.into_inner().expect("fold accumulator");

    let a = params.pseudo_count;
    let mut log_fold = DMatrix::<f32>::zeros(n_genes, n_slots);
    for k in 0..n_nodes {
        for b in 0..nb {
            log_fold
                .column_mut(k * nb + b)
                .copy_from(&log_delta.column(b));
        }
        let bs: Vec<usize> = (0..nb).filter(|&b| !members[k][b].is_empty()).collect();
        if bs.len() < 2 {
            continue;
        }
        for g in 0..n_genes {
            let r: Vec<f32> = bs
                .iter()
                .map(|&b| ((num[(g, k * nb + b)] + a) / (den[(g, k * nb + b)] + a)).ln())
                .collect();
            // r_b contrasts batch b with the mean of the others: on exact
            // data, centred, it is n/(n−1) · log f_b over a node's n batches.
            // Kept as is: the matching and the pseudo-count shrink it, and
            // the (n−1)/n rescale did worse on the sims (2026-11-21).
            let rm = r.iter().sum::<f32>() / r.len() as f32;
            for (&b, &rb) in bs.iter().zip(&r) {
                log_fold[(g, k * nb + b)] += rb - rm;
            }
        }
    }

    Ok(GroupFold {
        n_batches: nb,
        n_nodes,
        log_fold,
        log_delta: log_delta.clone(),
        node_of_unit: node.to_vec(),
    })
}

/// Dense numbering of the distinct `Some` values of `tree_node`.
fn renumber(tree_node: &[Option<usize>]) -> (Vec<Option<usize>>, usize) {
    let mut ids: HashMap<usize, usize> = HashMap::default();
    let out = tree_node
        .iter()
        .map(|t| {
            t.map(|t| {
                let n = ids.len();
                *ids.entry(t).or_insert(n)
            })
        })
        .collect();
    (out, ids.len())
}

/// Each coarse group's δ-adjusted pooled profile, log1p of counts per 10⁴.
fn group_profiles(
    data: &SparseIoVec,
    batch: &[usize],
    coarse_group: &[usize],
    n_groups: usize,
    log_delta: &DMatrix<f32>,
    block_size: Option<usize>,
) -> anyhow::Result<DMatrix<f32>> {
    let n_genes = log_delta.nrows();
    let inv = log_delta.map(|v| (-v).exp());
    let acc = Mutex::new(DMatrix::<f32>::zeros(n_groups, n_genes));
    let jobs = generate_minibatch_intervals(data.num_columns(), data.num_rows(), block_size);
    jobs.par_iter()
        .try_for_each(|&(lb, ub)| -> anyhow::Result<()> {
            let yy = data.read_columns_csc(lb..ub)?;
            let mut local: Vec<(usize, usize, f32)> = Vec::new();
            for (y, c) in yy.col_iter().zip(lb..ub) {
                let b = batch[c];
                for (&g, &x) in y.row_indices().iter().zip(y.values()) {
                    local.push((coarse_group[c], g, x * inv[(g, b)]));
                }
            }
            let mut p = acc.lock().expect("profiles");
            for (k, g, v) in local {
                p[(k, g)] += v;
            }
            Ok(())
        })?;
    let mut prof = acc.into_inner().expect("profiles");
    for mut row in prof.row_iter_mut() {
        let tot: f32 = row.sum();
        if tot > 0.0 {
            row.apply(|v| *v = (*v / tot * 1e4).ln_1p());
        }
    }
    Ok(prof)
}

/// The samples kind: a fold per (gene × node × batch) over columns as they are.
///
/// * `coarse_group`: each column's group at the collapse's coarsest level
///   (shared across batches), `0..K`.
/// * `log_delta`: the collapse's δ, log, genes × batches.
/// * `proj`: the batch-corrected projection the collapse ran on (dims × columns).
pub fn estimate_sample_fold(
    data: &SparseIoVec,
    batch: &[usize],
    coarse_group: &[usize],
    log_delta: &DMatrix<f32>,
    proj: &DMatrix<f32>,
    params: &GroupFoldParams,
) -> anyhow::Result<SampleFold> {
    let n = data.num_columns();
    let n_batches = log_delta.ncols();
    anyhow::ensure!(
        batch.len() == n && coarse_group.len() == n && proj.ncols() == n,
        "group fold: one batch, group and projection column per column"
    );
    let n_groups = coarse_group.iter().max().map_or(0, |&m| m + 1);
    let rule = SpanRule::new(params, batch, n_batches)?;

    let prof = group_profiles(
        data,
        batch,
        coarse_group,
        n_groups,
        log_delta,
        params.block_size,
    )?;
    let tree = CoarseTree::ward(&row_scores(
        &prof,
        PROFILE_DIM.min(n_groups.saturating_sub(1)).max(1),
    ));
    let mut group_count = vec![vec![0usize; rule.n_labels]; n_groups];
    for c in 0..n {
        group_count[coarse_group[c]][rule.label[c]] += 1;
    }
    let node_of_group = tree.cut(&group_count, &rule);
    let tree_node_of_unit: Vec<Option<usize>> =
        coarse_group.iter().map(|&k| node_of_group[k]).collect();
    let (node, n_nodes) = renumber(&tree_node_of_unit);
    if n_nodes == 0 {
        warn!(
            "group fold: no node spans {} of {} labels with {} units each; keeping δ",
            rule.min_spans, rule.n_labels, rule.min_units
        );
    } else {
        info!("group fold: {n_groups} coarse groups cut into {n_nodes} nodes");
    }

    let coord_map = CoordMap::fit(proj);
    let coords = coord_map.apply(proj);
    let fold = matched_fold(data, batch, &coords, &node, n_nodes, log_delta, params)?;

    let mut node_centroids = DMatrix::<f32>::zeros(coords.nrows(), n_nodes);
    let mut cnt = vec![0f32; n_nodes];
    for (c, k) in node.iter().enumerate() {
        if let Some(k) = *k {
            node_centroids.column_mut(k).add_assign(coords.column(c));
            cnt[k] += 1.0;
        }
    }
    for (k, &c) in cnt.iter().enumerate() {
        node_centroids.column_mut(k).scale_mut(1.0 / c.max(1.0));
    }

    Ok(SampleFold {
        fold,
        tree,
        tree_node_of_unit,
        coord_map,
        coords,
        node_centroids,
    })
}

/// Unordered (tree node, tree node) of a pair's endpoints → its pair node.
pub type NodeOfKey = HashMap<(usize, usize), usize>;

/// Each pair's node from its endpoints' tree nodes `at` (`None`: δ), climbing
/// both endpoints to their parents while the pair's key fails `rule`; past
/// the root a pair keeps δ. Returns the nodes, the settled keys and the rounds.
fn settle_pair_nodes(
    tree: &CoarseTree,
    mut at: Vec<Option<(usize, usize)>>,
    label_of_pair: &[usize],
    rule: &SpanRule,
) -> (Vec<Option<usize>>, NodeOfKey, usize) {
    let key = |(a, b): (usize, usize)| (a.min(b), a.max(b));
    let mut node = vec![None; at.len()];
    let mut node_of_key = NodeOfKey::default();
    let mut pending: Vec<usize> = (0..at.len()).filter(|&p| at[p].is_some()).collect();
    let mut rounds = 0;
    while !pending.is_empty() {
        let mut count: HashMap<(usize, usize), Vec<usize>> = HashMap::default();
        for &p in &pending {
            let k = key(at[p].expect("pending pair"));
            count.entry(k).or_insert_with(|| vec![0; rule.n_labels])[label_of_pair[p]] += 1;
        }
        let mut next = Vec::new();
        for &p in &pending {
            let (a, b) = at[p].expect("pending pair");
            let k = key((a, b));
            if rule.spans(&count[&k]) {
                let n = node_of_key.len();
                node[p] = Some(*node_of_key.entry(k).or_insert(n));
            } else if a == tree.root && b == tree.root {
                at[p] = None;
            } else {
                at[p] = Some((tree.parent[a], tree.parent[b]));
                next.push(p);
            }
        }
        pending = next;
        rounds += 1;
    }
    (node, node_of_key, rounds)
}

/// The graph kind's result: the fold over pair units, and how an unseen pair
/// finds its node.
#[derive(Clone, Debug)]
pub struct PairFold {
    pub fold: GroupFold,
    /// Unordered (tree node, tree node) of a pair's endpoints → its node, for
    /// every key a fitted pair settled at (after climbing).
    pub node_of_key: NodeOfKey,
}

impl PairFold {
    /// The node of a pair whose endpoints sit at tree nodes `a` and `b`:
    /// climb both until a settled key, `None` (δ) past the root.
    pub fn place(&self, tree: &CoarseTree, a: usize, b: usize) -> Option<usize> {
        let (mut a, mut b) = (a, b);
        loop {
            let key = (a.min(b), a.max(b));
            if let Some(&k) = self.node_of_key.get(&key) {
                return Some(k);
            }
            if a == tree.root && b == tree.root {
                return None;
            }
            (a, b) = (tree.parent[a], tree.parent[b]);
        }
    }
}

/// The graph kind: a fold per (gene × pair node × batch) over pair units.
///
/// * `units`: genes × pair units, the reads of every cell split once over its
///   pairs and its self-loop ([`crate::alg::cell_pairs::split_reads_over_pair_units`]):
///   columns `0..pairs.len()` are the pairs, the rest self-loops (estimate δ
///   only, never a fold).
/// * `cells`: the samples kind fitted on the cells (tree, cut, coordinates).
/// * `cell_span`: the span label of each cell (its batch, or its donor).
/// * `log_delta`: δ of the pair-unit collapse, log, genes × batches.
pub fn estimate_pair_fold(
    units: &SparseIoVec,
    pairs: &[(usize, usize)],
    cell_batch: &[usize],
    cells: &SampleFold,
    log_delta: &DMatrix<f32>,
    params: &GroupFoldParams,
) -> anyhow::Result<PairFold> {
    let n_units = units.num_columns();
    let n_batches = log_delta.ncols();
    let e = pairs.len();
    anyhow::ensure!(n_units >= e, "pair units: the pairs come first");
    let span_pair = SpanRule::new(params, cell_batch, n_batches)?;
    let label_of_pair: Vec<usize> = pairs.iter().map(|&(u, _)| span_pair.label[u]).collect();
    let tree = &cells.tree;

    let at: Vec<Option<(usize, usize)>> = pairs
        .iter()
        .map(|&(u, v)| Some((cells.tree_node_of_unit[u]?, cells.tree_node_of_unit[v]?)))
        .collect();
    let (pair_node, node_of_key, rounds) = settle_pair_nodes(tree, at, &label_of_pair, &span_pair);
    let mut node: Vec<Option<usize>> = vec![None; n_units];
    node[..e].copy_from_slice(&pair_node);
    let n_nodes = node_of_key.len();
    let kept = node.iter().take(e).filter(|n| n.is_some()).count();
    if n_nodes == 0 {
        warn!("pair group fold: no pair node spans enough labels; keeping δ");
    } else {
        info!(
            "pair group fold: {n_nodes} pair nodes after {rounds} rounds; {} of {e} pairs keep δ",
            e - kept
        );
    }

    let pair_batch: Vec<usize> = (0..n_units)
        .map(|k| if k < e { cell_batch[pairs[k].0] } else { 0 })
        .collect();
    let mut coords = DMatrix::<f32>::zeros(cells.coords.nrows(), n_units);
    for (k, &(u, v)) in pairs.iter().enumerate() {
        coords
            .column_mut(k)
            .copy_from(&((cells.coords.column(u) + cells.coords.column(v)) * 0.5));
    }
    let mut fold = matched_fold(
        units,
        &pair_batch,
        &coords,
        &node,
        n_nodes,
        log_delta,
        params,
    )?;
    fold.node_of_unit.truncate(e);
    Ok(PairFold { fold, node_of_key })
}

#[cfg(test)]
#[path = "group_fold_tests.rs"]
mod tests;
