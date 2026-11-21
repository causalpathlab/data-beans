//! Samples kind: the state-aware batch fold over columns as they are (cells,
//! or carried pseudobulks), from the sample collapse's coarsest groups and δ.
//! senna, bge and pinto's baselines read this kind; the cell-pair kind
//! ([`super::cell_pair_units`]) builds its nodes on this kind's tree.

use super::coarse_group_ward_tree::{row_scores, CoarseGroupWardTree};
use super::cross_batch_matched_fold::{
    estimate_fold_by_cross_batch_matching, MatchingCoordinates, StateAwareBatchFold,
};
use super::local_swap_refinement::{refine_nodes_by_local_swaps, SwapInputs};
use super::top_down_fold::{estimate_fold_top_down, TopDownInputs};
use super::{BatchSpanningRule, StateAwareBatchFoldParams, PROFILE_DIM};
use crate::sparse_io_vector::SparseIoVec;
use legume_numeric::matrix::utils::generate_minibatch_intervals;
use log::{info, warn};
use nalgebra::DMatrix;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use std::ops::AddAssign;
use std::sync::Mutex;

/// The samples kind's result: the fold, plus what places an unseen unit.
#[derive(Clone, Debug)]
pub struct SampleBatchFold {
    pub fold: StateAwareBatchFold,
    pub tree: CoarseGroupWardTree,
    /// Each unit's node in `tree` (before the dense node numbering), `None`
    /// when it keeps δ. The graph kind reads this for its endpoints.
    pub tree_node_of_unit: Vec<Option<usize>>,
    pub coord_map: MatchingCoordinates,
    /// Each unit's matching coordinates (components × units).
    pub coords: DMatrix<f32>,
    /// components × n_nodes: the mean coordinates of each node's units; an
    /// unseen unit takes its nearest node.
    pub node_centroids: DMatrix<f32>,
}

impl SampleBatchFold {
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
fn delta_adjusted_group_profiles(
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
pub fn estimate_state_aware_batch_fold_over_samples(
    data: &SparseIoVec,
    batch: &[usize],
    coarse_group: &[usize],
    log_delta: &DMatrix<f32>,
    proj: &DMatrix<f32>,
    params: &StateAwareBatchFoldParams,
) -> anyhow::Result<SampleBatchFold> {
    let n = data.num_columns();
    let n_batches = log_delta.ncols();
    anyhow::ensure!(
        batch.len() == n && coarse_group.len() == n && proj.ncols() == n,
        "group fold: one batch, group and projection column per column"
    );
    let n_groups = coarse_group.iter().max().map_or(0, |&m| m + 1);
    let rule = BatchSpanningRule::new(params, batch, n_batches)?;

    let prof = delta_adjusted_group_profiles(
        data,
        batch,
        coarse_group,
        n_groups,
        log_delta,
        params.block_size,
    )?;
    let tree = CoarseGroupWardTree::build(&row_scores(
        &prof,
        PROFILE_DIM.min(n_groups.saturating_sub(1)).max(1),
    ));
    let mut group_count = vec![vec![0usize; rule.n_labels]; n_groups];
    for c in 0..n {
        group_count[coarse_group[c]][rule.label[c]] += 1;
    }
    let node_of_group = tree.cut_where_both_children_span_batches(&group_count, &rule);
    let tree_node_of_unit: Vec<Option<usize>> =
        coarse_group.iter().map(|&k| node_of_group[k]).collect();
    let (node, n_nodes) = renumber(&tree_node_of_unit);
    // each dense node's tree node, to keep the refined nodes on the tree
    let mut tree_node_of_node = vec![0usize; n_nodes];
    for (k, t) in node.iter().zip(&tree_node_of_unit) {
        if let (Some(k), Some(t)) = (k, t) {
            tree_node_of_node[*k] = *t;
        }
    }
    if n_nodes == 0 {
        warn!(
            "group fold: no node spans {} of {} labels with {} units each; keeping δ",
            rule.min_spans, rule.n_labels, rule.min_units
        );
    } else {
        info!("group fold: {n_groups} coarse groups cut into {n_nodes} nodes");
    }

    let coord_map = MatchingCoordinates::fit(proj);
    let coords = coord_map.apply(proj);
    // coarsened (Ward, bottom up, then the cut): the first fold
    let first_fold = estimate_fold_by_cross_batch_matching(
        data, batch, &coords, &node, n_nodes, log_delta, params,
    )?;
    // uncoarsened: local swaps between neighbouring nodes, scored by each
    // node's fold-corrected composition in the sample's own batch; then the
    // fold again, on the refined nodes
    let node = refine_nodes_by_local_swaps(
        &SwapInputs {
            data,
            batch,
            coords: &coords,
            rule: &rule,
            block_size: params.block_size,
        },
        node,
        n_nodes,
        &first_fold,
    )?;
    let tree_node_of_unit: Vec<Option<usize>> = node
        .iter()
        .map(|k| k.map(|k| tree_node_of_node[k]))
        .collect();
    // then the fold top down along the tree, from the root to each refined
    // node, every node a residual over its parent's (numbered as `node`)
    let dense_of_final: HashMap<usize, usize> = tree_node_of_node
        .iter()
        .enumerate()
        .map(|(k, &t)| (t, k))
        .collect();
    let fold = estimate_fold_top_down(
        &TopDownInputs {
            data,
            batch,
            coords: &coords,
            tree: &tree,
            log_delta,
            params,
        },
        &tree_node_of_unit,
        &dense_of_final,
    )?;

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

    Ok(SampleBatchFold {
        fold,
        tree,
        tree_node_of_unit,
        coord_map,
        coords,
        node_centroids,
    })
}
