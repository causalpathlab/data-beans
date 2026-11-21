//! The tree the state-aware batch fold cuts its nodes from: Ward linkage over
//! the sample collapse's coarsest groups (their δ-adjusted pooled profiles),
//! and the cut that splits a node only where both children span batches.
//!
//! Exact up to [`EXACT_WARD_MAX_GROUPS`] groups (nearest-neighbour chain,
//! O(K²)); beyond, Ward along legume-numeric's kNN graph (O(K·k·log K)).

use super::BatchSpanningRule;
use legume_numeric::matrix::knn_graph::{KnnGraph, KnnGraphArgs};
use legume_numeric::matrix::traits::RandomizedAlgs;
use nalgebra::{DMatrix, SymmetricEigen};
use rayon::prelude::*;

////////////////////////////////////////////////////////////////////
// Ward tree over the coarse groups
////////////////////////////////////////////////////////////////////

/// Groups up to this many get exact Ward ([`CoarseGroupWardTree::build_exact_by_nearest_neighbour_chain`]);
/// beyond, Ward along a kNN graph ([`CoarseGroupWardTree::build_along_knn_graph`]).
pub const EXACT_WARD_MAX_GROUPS: usize = 4096;
/// Neighbours per group in the graph Ward merges along.
const WARD_GRAPH_KNN: usize = 15;

/// The merges recorded so far: tree ids `0..k` are the leaves, the rest are
/// given in merge order.
struct WardBuild {
    parent: Vec<usize>,
    children: Vec<Option<(usize, usize)>>,
    next_id: usize,
}

impl WardBuild {
    fn merge(&mut self, a: usize, b: usize) {
        let n = self.next_id;
        self.parent[a] = n;
        self.parent[b] = n;
        self.children[n] = Some((a, b));
        self.next_id += 1;
    }
}

/// Ward's merge cost of two clusters: 2·nₐn_b/(nₐ+n_b)·‖c̄ₐ − c̄_b‖² (for
/// two single points, their squared distance).
fn ward_cost(ca: &[f32], na: f64, cb: &[f32], nb: f64) -> f64 {
    let d2: f64 = ca
        .iter()
        .zip(cb)
        .map(|(&x, &y)| f64::from(x - y).powi(2))
        .sum();
    2.0 * na * nb / (na + nb) * d2
}

/// A totally ordered merge cost for the heap.
#[derive(Clone, Copy, PartialEq)]
struct Cost(f64);
impl Eq for Cost {}
impl PartialOrd for Cost {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cost {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Exact Ward over clusters (`centroids`, `sizes`, tree `ids`) by the
/// nearest-neighbour chain with Lance–Williams updates on Ward costs,
/// recording every merge in `build`; O(n²) for n clusters.
fn ward_chain(
    centroids: &[Vec<f32>],
    mut size: Vec<f64>,
    mut id: Vec<usize>,
    build: &mut WardBuild,
) {
    let k = centroids.len();
    if k <= 1 {
        return;
    }
    let mut d = vec![0f64; k * k];
    d.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
        for (j, dij) in row.iter_mut().enumerate() {
            *dij = ward_cost(&centroids[i], size[i], &centroids[j], size[j]);
        }
    });
    let mut active: Vec<bool> = vec![true; k];
    let mut chain: Vec<usize> = Vec::with_capacity(k);
    let mut n_active = k;
    while n_active > 1 {
        if chain.is_empty() {
            chain.push(active.iter().position(|&a| a).expect("an active cluster"));
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
        build.merge(id[a], id[b]);
        let (na, nb, dab) = (size[a], size[b], d[a * k + b]);
        for m in (0..k).filter(|&m| active[m] && m != a && m != b) {
            let nm = size[m];
            let v =
                ((na + nm) * d[a * k + m] + (nb + nm) * d[b * k + m] - nm * dab) / (na + nb + nm);
            d[a * k + m] = v;
            d[m * k + a] = v;
        }
        size[a] = na + nb;
        id[a] = build.next_id - 1;
        active[b] = false;
        n_active -= 1;
    }
}

/// A binary tree over `n_groups` leaves (ids `0..n_groups`); internal nodes
/// take ids `n_groups..2·n_groups-1` in merge order; the root is the last.
#[derive(Clone, Debug)]
pub struct CoarseGroupWardTree {
    pub n_groups: usize,
    /// `parent[node]`; the root is its own parent.
    pub parent: Vec<usize>,
    /// `children[node]` for an internal node.
    pub children: Vec<Option<(usize, usize)>>,
    pub root: usize,
}

impl CoarseGroupWardTree {
    /// Ward linkage over the rows of `points` (groups × dims): exact up to
    /// [`EXACT_WARD_MAX_GROUPS`] groups ([`Self::build_exact_by_nearest_neighbour_chain`], O(K²)), along a
    /// kNN graph beyond ([`Self::build_along_knn_graph`], O(K·k·log K)).
    pub fn build(points: &DMatrix<f32>) -> Self {
        if points.nrows() <= EXACT_WARD_MAX_GROUPS {
            Self::build_exact_by_nearest_neighbour_chain(points)
        } else {
            Self::build_along_knn_graph(points, WARD_GRAPH_KNN)
        }
    }

    fn leaf_tree(k: usize) -> WardBuild {
        let total = (2 * k).saturating_sub(1).max(1);
        WardBuild {
            parent: (0..total).collect(),
            children: vec![None; total],
            next_id: k,
        }
    }

    fn finish(k: usize, build: WardBuild) -> Self {
        let mut parent = build.parent;
        let root = parent.len() - 1;
        parent[root] = root;
        Self {
            n_groups: k,
            parent,
            children: build.children,
            root,
        }
    }

    /// Exact Ward by the nearest-neighbour chain with Lance–Williams updates:
    /// the same tree as greedy Ward, in O(K²) time and memory.
    pub fn build_exact_by_nearest_neighbour_chain(points: &DMatrix<f32>) -> Self {
        let k = points.nrows();
        let mut build = Self::leaf_tree(k);
        let centroids: Vec<Vec<f32>> = points
            .row_iter()
            .map(|r| r.iter().copied().collect())
            .collect();
        ward_chain(&centroids, vec![1.0; k], (0..k).collect(), &mut build);
        Self::finish(k, build)
    }

    /// Ward along a kNN graph of the groups (`knn` neighbours, legume-numeric's
    /// [`KnnGraph`], as its layouts and clustering build it; an empty graph
    /// leaves every group to the exact join): clusters merge only along edges, cheapest Ward cost
    /// first, the cost exact from centroids and sizes; pieces the graph leaves
    /// apart are joined by [`Self::build_exact_by_nearest_neighbour_chain`]'s chain over their centroids.
    /// The same tree as exact Ward wherever each merge's partner is a graph
    /// neighbour.
    pub fn build_along_knn_graph(points: &DMatrix<f32>, knn: usize) -> Self {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        let k = points.nrows();
        let mut build = Self::leaf_tree(k);
        if k <= 1 {
            return Self::finish(k, build);
        }
        // the same kNN graph legume-numeric's layouts and clustering build (union edges)
        let graph = KnnGraph::from_rows(
            points,
            KnnGraphArgs {
                knn: knn.min(k - 1),
                block_size: 1000,
                reciprocal: false,
            },
        );
        let mut adj: Vec<rustc_hash::FxHashSet<usize>> = vec![Default::default(); k];
        for &(i, j) in graph
            .as_ref()
            .map(|g| g.edges.as_slice())
            .unwrap_or_default()
        {
            if i != j {
                adj[i].insert(j);
                adj[j].insert(i);
            }
        }
        let mut centroid: Vec<Vec<f32>> = points
            .row_iter()
            .map(|r| r.iter().copied().collect())
            .collect();
        let mut size = vec![1f64; k];
        let mut id: Vec<usize> = (0..k).collect();
        let mut alive = vec![true; k];
        let mut stamp = vec![0u32; k];
        let mut heap = BinaryHeap::new();
        for a in 0..k {
            for &b in adj[a].iter().filter(|&&b| b > a) {
                let c = ward_cost(&centroid[a], size[a], &centroid[b], size[b]);
                heap.push(Reverse((Cost(c), a, b, 0u32, 0u32)));
            }
        }
        while let Some(Reverse((_, a, b, sa, sb))) = heap.pop() {
            if !alive[a] || !alive[b] || stamp[a] != sa || stamp[b] != sb {
                continue;
            }
            build.merge(id[a], id[b]);
            let (na, nb) = (size[a], size[b]);
            let merged: Vec<f32> = centroid[a]
                .iter()
                .zip(&centroid[b])
                .map(|(&x, &y)| ((f64::from(x) * na + f64::from(y) * nb) / (na + nb)) as f32)
                .collect();
            centroid[a] = merged;
            size[a] = na + nb;
            id[a] = build.next_id - 1;
            alive[b] = false;
            stamp[a] += 1;
            let nb_of_b = std::mem::take(&mut adj[b]);
            for m in nb_of_b {
                adj[m].remove(&b);
                if m != a {
                    adj[m].insert(a);
                    adj[a].insert(m);
                }
            }
            adj[a].remove(&b);
            adj[a].remove(&a);
            for &m in &adj[a] {
                let c = ward_cost(&centroid[a], size[a], &centroid[m], size[m]);
                let (x, y) = if a < m { (a, m) } else { (m, a) };
                heap.push(Reverse((Cost(c), x, y, stamp[x], stamp[y])));
            }
        }
        // pieces the graph left apart: join them exactly
        let rest: Vec<usize> = (0..k).filter(|&i| alive[i]).collect();
        if rest.len() > 1 {
            ward_chain(
                &rest
                    .iter()
                    .map(|&i| centroid[i].clone())
                    .collect::<Vec<_>>(),
                rest.iter().map(|&i| size[i]).collect(),
                rest.iter().map(|&i| id[i]).collect(),
                &mut build,
            );
        }
        Self::finish(k, build)
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
    pub(crate) fn cut_where_both_children_span_batches(
        &self,
        group_count: &[Vec<usize>],
        rule: &BatchSpanningRule,
    ) -> Vec<Option<usize>> {
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

    pub(crate) fn leaves(&self, node: usize) -> Vec<usize> {
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

/// Groups up to this many get exact principal components (an eigen-
/// decomposition of their K × K Gram matrix); beyond, a randomized SVD.
const EXACT_SCORES_MAX_ROWS: usize = 1024;

/// Principal-component scores (rows × k) of the rows of `x`, centred: exact
/// up to [`EXACT_SCORES_MAX_ROWS`] rows, by randomized SVD beyond
/// (O(rows · cols · k), not cubic in the rows).
pub(super) fn row_scores(x: &DMatrix<f32>, k: usize) -> DMatrix<f32> {
    let n = x.nrows();
    let mut xc = x.clone();
    for c in 0..x.ncols() {
        let m = x.column(c).mean();
        xc.column_mut(c).add_scalar_mut(-m);
    }
    let k = k.min(n).min(x.ncols());
    if n <= EXACT_SCORES_MAX_ROWS {
        let eig = SymmetricEigen::new(&xc * xc.transpose());
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));
        return DMatrix::from_fn(n, k, |r, c| {
            let i = order[c];
            eig.eigenvectors[(r, i)] * eig.eigenvalues[i].max(0.0).sqrt()
        });
    }
    match xc.rsvd(k) {
        Ok((u, dv, _)) => DMatrix::from_fn(u.nrows(), u.ncols(), |r, c| u[(r, c)] * dv[c]),
        Err(_) => DMatrix::zeros(n, k),
    }
}
