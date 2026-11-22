//! Samples kind: the fold estimated top down along the tree, from the root to
//! each pseudobulk's node: at every depth, each tree node's fold is a residual
//! matched fold on top of its parent's, shrunk toward the parent by the
//! pseudo-count where the node has little to match. A node the cut (and the
//! swaps) left a pseudobulk in is final; the pseudobulk stops there.
//!
//! So a fine node never starts from δ alone: what its ancestors already saw
//! in common is carried down, and a node only adds what its own matching
//! shows beyond them. Every depth runs on the same pseudobulk sums.

use super::coarse_group_ward_tree::CoarseGroupWardTree;
use super::cross_batch_matched_fold::estimate_fold_on_pseudobulks;
use super::pseudobulk_sums::PseudobulkSums;
use super::{compact_present, StateAwareBatchFoldParams};
use log::info;
use nalgebra::DMatrix;
use rustc_hash::FxHashMap as HashMap;

/// Each final tree node's fold (log, genes × batches), estimated top down:
/// `final_node[pb]` is pseudobulk `pb`'s tree node (`None`: left out).
pub(crate) fn estimate_fold_top_down(
    sums: &PseudobulkSums,
    tree: &CoarseGroupWardTree,
    final_node: &[Option<usize>],
    log_delta: &DMatrix<f32>,
    params: &StateAwareBatchFoldParams,
) -> anyhow::Result<HashMap<usize, DMatrix<f32>>> {
    let nb = log_delta.ncols();
    let depth = &tree.depth;
    let deepest = final_node
        .iter()
        .flatten()
        .map(|&t| depth[t])
        .max()
        .unwrap_or(0);
    let ancestor_at = |mut t: usize, d: usize| {
        while depth[t] > d {
            t = tree.parent[t];
        }
        t
    };

    let mut fold_of: HashMap<usize, DMatrix<f32>> = HashMap::default();
    for d in 0..=deepest {
        // pseudobulks still descending: their node at this depth
        let at: Vec<Option<usize>> = final_node
            .iter()
            .map(|t| t.filter(|&t| depth[t] >= d).map(|t| ancestor_at(t, d)))
            .collect();
        let (node, node_of_dense) = compact_present(&at);
        let n_nodes = node_of_dense.len();
        if n_nodes == 0 {
            break;
        }
        // each node's base: its parent's fold (δ at the root)
        let mut base = DMatrix::<f32>::zeros(log_delta.nrows(), n_nodes * nb);
        for (k, &t) in node_of_dense.iter().enumerate() {
            let parent_fold = (d > 0)
                .then(|| fold_of.get(&tree.parent[t]))
                .flatten()
                .unwrap_or(log_delta);
            base.columns_mut(k * nb, nb).copy_from(parent_fold);
        }
        let log_fold = estimate_fold_on_pseudobulks(sums, &node, n_nodes, &base, params)?;
        for (k, &t) in node_of_dense.iter().enumerate() {
            fold_of.insert(t, log_fold.columns(k * nb, nb).into_owned());
        }
        info!("batch fold, top down: depth {d}: {n_nodes} nodes");
    }
    Ok(fold_of)
}
