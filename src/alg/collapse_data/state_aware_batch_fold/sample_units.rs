//! Samples kind: the state-aware batch fold over columns as they are (cells,
//! or carried pseudobulks), from the sample collapse's groups and δ. senna,
//! bge and pinto's baselines read this kind; the cell-pair kind
//! ([`super::cell_pair_units`]) builds its pair nodes on this kind's tree.
//!
//! 1. The data read once, into pb samples (finest group × batch) sums.
//! 2. Coarsen: Ward over the coarsest groups' δ-adjusted profiles, cut where
//!    both children span batches in coarsest pseudobulks.
//! 3. Uncoarsen: local swaps of pb samples between neighbouring nodes.
//! 4. The fold top down along the tree, pb sample to pb sample.

use super::coarse_group_ward_tree::{row_scores, CoarseGroupWardTree};
use super::cross_batch_matched_fold::{
    delta_as_base_of_every_node, estimate_fold_on_pseudobulks, MatchingCoordinates,
    StateAwareBatchFold,
};
use super::local_swap_refinement::refine_nodes_by_local_swaps;
use super::pseudobulk_sums::{sum_by_pseudobulk, PseudobulkKeys, PseudobulkSums};
use super::top_down_fold::estimate_fold_top_down;
use super::{
    compact_present, BatchSpanningRule, CollapsePseudobulks, StateAwareBatchFoldParams, PROFILE_DIM,
};
use crate::sparse_io_vector::SparseIoVec;
use log::{info, warn};
use nalgebra::DMatrix;
use std::ops::AddAssign;

/// The samples kind's result: the fold, plus what places an unseen unit.
#[derive(Clone, Debug)]
pub struct SampleBatchFold {
    pub fold: StateAwareBatchFold,
    pub tree: CoarseGroupWardTree,
    /// Each unit's node in `tree`, `None` when it keeps δ. The cell-pair kind
    /// reads this for its endpoints.
    pub tree_node_of_unit: Vec<Option<usize>>,
    pub coord_map: MatchingCoordinates,
    /// Each unit's matching coordinates (components × units).
    pub coords: DMatrix<f32>,
    /// components × n_nodes: the mean coordinates of each node's units; an
    /// unseen unit takes its nearest node.
    pub node_centroids: DMatrix<f32>,
    /// Each node's (dense id's) node in `tree`.
    pub tree_node_of_node: Vec<usize>,
}

/// Where an unseen unit lands: its node (the fold's column, with its batch)
/// and that node's place in the tree (what an unseen pair climbs from).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnseenPlacement {
    pub node: usize,
    pub tree_node: usize,
}

impl SampleBatchFold {
    /// Where unseen units land, from their projections `proj` (dims ×
    /// units): each takes the nearest node centroid.
    pub fn place_unseen_samples(&self, proj: &DMatrix<f32>) -> Vec<Option<UnseenPlacement>> {
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
                    .map(|(k, _)| UnseenPlacement {
                        node: k,
                        tree_node: self.tree_node_of_node[k],
                    })
            })
            .collect()
    }
}

/// Each coarse group's δ-adjusted pooled profile (groups × genes), log1p of
/// counts per 10⁴, from the pb sample sums.
fn delta_adjusted_group_profiles(
    sums: &PseudobulkSums,
    n_groups: usize,
    log_delta: &DMatrix<f32>,
) -> DMatrix<f32> {
    let mut prof = DMatrix::<f32>::zeros(n_groups, sums.n_genes);
    for pb in 0..sums.len() {
        let (k, b) = (sums.coarsest[pb], sums.batch[pb]);
        for &(g, x) in &sums.counts[pb] {
            prof[(k, g)] += x * (-log_delta[(g, b)]).exp();
        }
    }
    for mut row in prof.row_iter_mut() {
        let tot: f32 = row.sum();
        if tot > 0.0 {
            row.apply(|v| *v = (*v / tot * 1e4).ln_1p());
        }
    }
    prof
}

/// The samples kind: a fold per (gene × node × batch) over columns as they are.
///
/// * `pseudobulks`: each column's finest and coarsest collapse group
///   (shared across batches): its pb sample (finest × batch, what the fold
///   matches) and the tree's leaves (coarsest).
/// * `log_delta`: the collapse's δ, log, genes × batches.
/// * `proj`: the batch-corrected projection the collapse ran on (dims × columns).
pub fn estimate_state_aware_batch_fold_over_samples(
    data: &SparseIoVec,
    batch: &[usize],
    pseudobulks: &CollapsePseudobulks<'_>,
    log_delta: &DMatrix<f32>,
    proj: &DMatrix<f32>,
    params: &StateAwareBatchFoldParams,
) -> anyhow::Result<SampleBatchFold> {
    let n = data.num_columns();
    let nb = log_delta.ncols();
    anyhow::ensure!(
        batch.len() == n
            && pseudobulks.finest.len() == n
            && pseudobulks.coarsest.len() == n
            && proj.ncols() == n,
        "batch fold: one batch, group and projection column per column"
    );
    let (span_label, n_labels) = params.span_labels_of_samples(batch, nb)?;
    let rule = BatchSpanningRule::new(params, n_labels);

    let coord_map = MatchingCoordinates::fit(proj)?;
    let coords = coord_map.apply(proj);
    // the data read once: pb samples (finest group in one batch)
    let key: Vec<Option<(usize, usize, usize)>> = (0..n)
        .map(|u| Some((pseudobulks.finest[u], batch[u], 0)))
        .collect();
    let sums = sum_by_pseudobulk(
        data,
        &PseudobulkKeys {
            key: &key,
            batch,
            span_label: &span_label,
            coarsest: pseudobulks.coarsest,
            coords: &coords,
        },
    )?;

    // coarsen: Ward over the coarsest groups, cut where both children span
    let n_groups = pseudobulks.coarsest.iter().max().map_or(0, |&m| m + 1);
    let prof = delta_adjusted_group_profiles(&sums, n_groups, log_delta);
    let tree = CoarseGroupWardTree::build(&row_scores(
        &prof,
        PROFILE_DIM.min(n_groups.saturating_sub(1)).max(1),
    )?)?;
    let mut members = vec![Vec::new(); n_groups];
    for pb in 0..sums.len() {
        members[sums.coarsest[pb]].push((sums.coarsest[pb], sums.span_label[pb]));
    }
    let group_count: Vec<Vec<usize>> = members.into_iter().map(|m| rule.count(m)).collect();
    let node_of_group = tree.cut_where_both_children_span_batches(&group_count, &rule);
    let tree_node_of_pb: Vec<Option<usize>> =
        sums.coarsest.iter().map(|&g| node_of_group[g]).collect();
    let (node, tree_node_of_node) = compact_present(&tree_node_of_pb);
    let n_nodes = tree_node_of_node.len();
    if n_nodes == 0 {
        warn!(
            "batch fold: no node spans {} of {} labels with {} coarsest pseudobulks each; keeping δ",
            rule.min_spans, rule.n_labels, rule.min_pseudobulks
        );
    } else {
        info!("batch fold: {n_groups} coarse groups cut into {n_nodes} nodes");
    }

    // uncoarsen: a first fold, then local swaps of pb samples under it
    let first_fold = estimate_fold_on_pseudobulks(
        &sums,
        &node,
        n_nodes,
        &delta_as_base_of_every_node(log_delta, n_nodes),
        params,
    )?;
    let node = refine_nodes_by_local_swaps(&sums, &rule, node, n_nodes, &first_fold);

    // the fold top down along the tree, from the root to each refined node
    let final_tree_node: Vec<Option<usize>> = node
        .iter()
        .map(|k| k.map(|k| tree_node_of_node[k]))
        .collect();
    let fold_of = estimate_fold_top_down(&sums, &tree, &final_tree_node, log_delta, params)?;
    let mut log_fold = DMatrix::<f32>::zeros(log_delta.nrows(), n_nodes * nb);
    for (k, t) in tree_node_of_node.iter().enumerate() {
        log_fold
            .columns_mut(k * nb, nb)
            .copy_from(fold_of.get(t).unwrap_or(log_delta));
    }

    // each unit takes its pb sample's node
    let node_of_unit: Vec<Option<usize>> = sums
        .of_unit
        .iter()
        .map(|pb| pb.and_then(|pb| node[pb]))
        .collect();
    let tree_node_of_unit: Vec<Option<usize>> = node_of_unit
        .iter()
        .map(|k| k.map(|k| tree_node_of_node[k]))
        .collect();
    let mut node_centroids = DMatrix::<f32>::zeros(coords.nrows(), n_nodes);
    let mut cnt = vec![0f32; n_nodes];
    for (c, k) in node_of_unit.iter().enumerate() {
        if let Some(k) = *k {
            node_centroids.column_mut(k).add_assign(coords.column(c));
            cnt[k] += 1.0;
        }
    }
    for (k, &c) in cnt.iter().enumerate() {
        node_centroids.column_mut(k).scale_mut(1.0 / c.max(1.0));
    }

    Ok(SampleBatchFold {
        fold: StateAwareBatchFold {
            n_batches: nb,
            n_nodes,
            log_fold,
            log_delta: log_delta.clone(),
            node_of_unit,
        },
        tree,
        tree_node_of_unit,
        coord_map,
        coords,
        node_centroids,
        tree_node_of_node,
    })
}
