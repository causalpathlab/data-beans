//! The fold estimated top down along the tree, from the root to each sample's
//! node: at every depth, each tree node's fold is a residual matched fold on
//! top of its parent's, shrunk toward the parent by the pseudo-count where
//! the node has little to match. A node the cut did not split is final; its
//! samples stop there.
//!
//! So a fine node never starts from δ alone: what its ancestors already saw
//! in common is carried down, and a node only adds what its own matching
//! shows beyond them.

use super::coarse_group_ward_tree::CoarseGroupWardTree;
use super::cross_batch_matched_fold::{
    estimate_fold_by_cross_batch_matching, MatchedFoldInputs, StateAwareBatchFold,
};
use super::StateAwareBatchFoldParams;
use crate::sparse_io_vector::SparseIoVec;
use log::info;
use nalgebra::DMatrix;
use rustc_hash::FxHashMap as HashMap;

/// Each tree node's depth below the root (the root at 0).
fn depth_of_tree_nodes(tree: &CoarseGroupWardTree) -> Vec<usize> {
    let n = tree.parent.len();
    let mut depth = vec![usize::MAX; n];
    depth[tree.root] = 0;
    // internal ids grow in merge order, so a parent comes after its children:
    // walk from the root down, largest id first
    for v in (0..n).rev() {
        if v == tree.root {
            continue;
        }
        let p = tree.parent[v];
        if depth[p] != usize::MAX {
            depth[v] = depth[p] + 1;
        }
    }
    depth
}

/// What the top-down fold reads.
pub(crate) struct TopDownInputs<'a> {
    pub data: &'a SparseIoVec,
    pub tree: &'a CoarseGroupWardTree,
    pub batch: &'a [usize],
    /// Matching coordinates, components × units.
    pub coords: &'a DMatrix<f32>,
    pub log_delta: &'a DMatrix<f32>,
    /// Each sample's pseudobulk (the collapse's finest group).
    pub pseudobulk_of_unit: &'a [usize],
    pub params: &'a StateAwareBatchFoldParams,
}

/// The fold of every sample, estimated top down along the tree:
/// `final_node[u]` is sample `u`'s tree node (where the cut and the swaps left
/// it; `None`: δ). Returns the fold over the distinct final nodes, numbered as
/// `dense_of_final` gives (tree node → dense node).
pub(crate) fn estimate_fold_top_down(
    inp: &TopDownInputs<'_>,
    final_node: &[Option<usize>],
    dense_of_final: &HashMap<usize, usize>,
) -> anyhow::Result<StateAwareBatchFold> {
    let tree = inp.tree;
    let depth = depth_of_tree_nodes(tree);
    let nb = inp.log_delta.ncols();
    let n_genes = inp.log_delta.nrows();
    let deepest = final_node
        .iter()
        .flatten()
        .map(|&t| depth[t])
        .max()
        .unwrap_or(0);
    // the ancestor of `t` at depth `d` (≤ its own)
    let ancestor_at = |mut t: usize, d: usize| {
        while depth[t] > d {
            t = tree.parent[t];
        }
        t
    };

    // each node's fold once estimated, genes × batches, log
    let mut fold_of: HashMap<usize, DMatrix<f32>> = HashMap::default();
    for d in 0..=deepest {
        // units still descending: their node at this depth
        let mut dense: HashMap<usize, usize> = HashMap::default();
        let mut node_of_dense: Vec<usize> = Vec::new();
        let node: Vec<Option<usize>> = final_node
            .iter()
            .map(|t| {
                let t = (*t).filter(|&t| depth[t] >= d)?;
                let a = ancestor_at(t, d);
                Some(*dense.entry(a).or_insert_with(|| {
                    node_of_dense.push(a);
                    node_of_dense.len() - 1
                }))
            })
            .collect();
        let n_nodes = node_of_dense.len();
        if n_nodes == 0 {
            break;
        }
        // each node's base: its parent's fold (δ at the root)
        let mut base = DMatrix::<f32>::zeros(n_genes, n_nodes * nb);
        for (k, &t) in node_of_dense.iter().enumerate() {
            let parent_fold = (d > 0)
                .then(|| fold_of.get(&tree.parent[t]))
                .flatten()
                .unwrap_or(inp.log_delta);
            base.columns_mut(k * nb, nb).copy_from(parent_fold);
        }
        let fold = estimate_fold_by_cross_batch_matching(
            &MatchedFoldInputs {
                data: inp.data,
                batch: inp.batch,
                coords: inp.coords,
                node: &node,
                n_nodes,
                base_log_fold: &base,
                log_delta: inp.log_delta,
                pseudobulk_of_unit: Some(inp.pseudobulk_of_unit),
            },
            inp.params,
        )?;
        for (k, &t) in node_of_dense.iter().enumerate() {
            fold_of.insert(t, fold.log_fold.columns(k * nb, nb).clone_owned());
        }
        info!("batch fold, top down: depth {d}: {n_nodes} nodes");
    }

    // the final nodes' folds, as one table
    let n_final = dense_of_final.len();
    let mut log_fold = DMatrix::<f32>::zeros(n_genes, n_final * nb);
    for (t, &k) in dense_of_final {
        let f = fold_of.get(t).unwrap_or(inp.log_delta);
        log_fold.columns_mut(k * nb, nb).copy_from(f);
    }
    Ok(StateAwareBatchFold {
        n_batches: nb,
        n_nodes: n_final,
        log_fold,
        log_delta: inp.log_delta.clone(),
        node_of_unit: final_node
            .iter()
            .map(|t| t.and_then(|t| dense_of_final.get(&t).copied()))
            .collect(),
    })
}
