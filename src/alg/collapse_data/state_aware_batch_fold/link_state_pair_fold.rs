//! Cell-pair kind, link states ([`super::cell_pair_units::PairStates::LinkStates`]):
//! a pair's state learned from the pair itself, not from its endpoints.
//!
//! A pair's batch effect is split in two:
//!
//! ```text
//! log fold(pair e, batch b) = cell-predicted(e, b) + leftover(link state of e, b)
//! cell-predicted(e, b)      = ½ (log cell fold(u) + log cell fold(v)),  e = (u, v)
//! ```
//!
//! 1. **Cell-predicted fold**: the mean log of the two endpoints' cell folds
//!    (the samples kind's result). Every pb pair is keyed by its endpoints'
//!    (unordered) cell nodes as well, so it has one cell-predicted fold.
//! 2. Each pb pair's sums divided by it: what is left is shared by the
//!    batches, up to what the contact itself adds.
//! 3. **Link states**: Ward over the coarsest pair groups on those divided
//!    sums, cut where both children span batches; local swaps of pb pairs.
//!    The cells' donor × state effect is divided out first, so the tree
//!    splits on what contacts share across donors, not on the donors.
//! 4. **Leftover fold**: matched pb pair to pb pair within each link state,
//!    on the divided sums, base 0.
//!
//! The fold's nodes are (link state, endpoints' cell nodes): the leftover is
//! the link state's, the cell-predicted part the endpoints'.

use super::cell_pair_units::{sum_cell_pairs_into_pseudobulks, CellPairUnits};
use super::cross_batch_matched_fold::{estimate_fold_on_pseudobulks, StateAwareBatchFold};
use super::local_swap_refinement::refine_nodes_by_local_swaps;
use super::node_compositions::{node_log_mean_compositions, place_by_composition};
use super::pseudobulk_sums::PseudobulkSums;
use super::sample_units::{
    cut_nodes_from_ward_tree_over_coarsest_groups, NodesCutFromWardTree, SampleBatchFold,
};
use super::{compact_present, BatchSpanningRule, CollapsePseudobulks, StateAwareBatchFoldParams};
use log::{info, warn};
use nalgebra::{DMatrix, DVector};
use std::ops::AddAssign;

/// The link states' part of a cell-pair fold, for predict and for reading
/// each pair's link state (stage 2 conditions on it).
#[derive(Clone, Debug)]
pub struct LinkStateDecomposition {
    /// Each given pair's link state (`None`: across batches, or in no link
    /// state that spans).
    pub link_state_of_pair: Vec<Option<usize>>,
    /// genes × (n_link_states · batches): each link state's leftover log
    /// fold over the cell-predicted fold, centred over its batches.
    pub leftover_log_fold: DMatrix<f32>,
    pub n_batches: usize,
    /// genes × n_link_states: each link state's composition, log, on counts
    /// divided by their cell-predicted fold and their leftover; an unseen
    /// pair takes the link state that explains its counts best.
    pub link_state_log_mean_composition: DMatrix<f32>,
}

impl LinkStateDecomposition {
    pub fn n_link_states(&self) -> usize {
        self.link_state_log_mean_composition.ncols()
    }

    /// Each unseen pair's link state, by composition
    /// ([`super::node_compositions`]): `counts[p]`, the pair's pooled counts
    /// split as the fit split them. `batch[p]`: `Some(b)` for a batch the fit
    /// saw, the counts divided by the pair's cell-predicted fold in `b`
    /// ([`cell_predicted_pair_log_fold`]); `None` for one it never saw, the
    /// counts divided by that batch's own δ alone (it has no cell-predicted
    /// fold). `None` for a pair with no read. Errors as
    /// [`place_by_composition`] does.
    pub fn place_unseen_pairs(
        &self,
        counts: &[Vec<(usize, f32)>],
        batch: &[Option<usize>],
    ) -> anyhow::Result<Vec<Option<usize>>> {
        place_by_composition(
            counts,
            batch,
            &self.link_state_log_mean_composition,
            &self.leftover_log_fold,
            self.n_batches,
        )
    }

    /// An unseen pair's log fold in a batch the fit saw: the cell-predicted
    /// fold of its endpoints' cell nodes `a`, `b` (the samples kind's nodes;
    /// `None`: δ), plus its link state's leftover (`None`: none). Errors on a
    /// batch, node or link state outside the fit.
    pub fn unseen_pair_log_fold(
        &self,
        cells: &StateAwareBatchFold,
        a: Option<usize>,
        b: Option<usize>,
        link_state: Option<usize>,
        batch: usize,
    ) -> anyhow::Result<DVector<f32>> {
        anyhow::ensure!(
            batch < self.n_batches && cells.n_batches == self.n_batches,
            "unseen pair: batch {batch}; the link states have {} batches, the cells {}",
            self.n_batches,
            cells.n_batches
        );
        if let Some(k) = [a, b].into_iter().flatten().find(|&k| k >= cells.n_nodes) {
            anyhow::bail!("unseen pair: cell node {k}, the fit has {}", cells.n_nodes);
        }
        let mut f = cell_predicted_pair_log_fold(cells, a, b, batch);
        if let Some(s) = link_state {
            anyhow::ensure!(
                s < self.n_link_states(),
                "unseen pair: link state {s}, the fit has {}",
                self.n_link_states()
            );
            f.add_assign(self.leftover_log_fold.column(s * self.n_batches + batch));
        }
        Ok(f)
    }
}

/// A pair's cell-predicted log fold in `batch`: the mean of its endpoints'
/// cell folds (cell nodes `a`, `b`; `None`: δ).
pub fn cell_predicted_pair_log_fold(
    cells: &StateAwareBatchFold,
    a: Option<usize>,
    b: Option<usize>,
    batch: usize,
) -> DVector<f32> {
    (cells.log_fold_at(a, batch) + cells.log_fold_at(b, batch)) * 0.5
}

/// Divide each pb pair's sums by its cell-predicted fold:
/// `cell_predicted_col_of_pb[pb]` is its column of `cell_predicted`
/// (log, genes × columns).
pub(crate) fn divide_pb_pairs_by_cell_predicted_fold(
    sums: &mut PseudobulkSums,
    cell_predicted: &DMatrix<f32>,
    cell_predicted_col_of_pb: &[usize],
) {
    use rayon::prelude::*;
    sums.counts
        .par_iter_mut()
        .zip(cell_predicted_col_of_pb)
        .for_each(|(counts, &col)| {
            for (g, x) in counts.iter_mut() {
                *x *= (-cell_predicted[(*g, col)]).exp();
            }
        });
}

/// Link states and their leftover folds on pb pairs already divided by the
/// cell-predicted fold: Ward over their coarsest groups (renumbered densely
/// here), cut where both children span `rule`, local swaps, then the fold
/// matched per link state with base 0. Returns each pb pair's link state
/// and the leftover (genes × (states · `n_batches`)).
pub(crate) fn link_states_and_leftover_fold(
    sums: &mut PseudobulkSums,
    n_batches: usize,
    rule: &BatchSpanningRule,
    params: &StateAwareBatchFoldParams,
) -> anyhow::Result<(Vec<Option<usize>>, usize, DMatrix<f32>)> {
    if sums.is_empty() {
        return Ok((Vec::new(), 0, DMatrix::zeros(sums.n_genes, 0)));
    }
    let (dense, n_groups) = crate::alg::dc_poisson::compact_labels(&sums.coarsest);
    sums.coarsest = dense;
    let no_delta = DMatrix::<f32>::zeros(sums.n_genes, n_batches);
    let NodesCutFromWardTree {
        node_of_pb,
        tree_node_of_node,
        ..
    } = cut_nodes_from_ward_tree_over_coarsest_groups(sums, n_groups, &no_delta, rule)?;
    let n_states = tree_node_of_node.len();
    if n_states == 0 {
        return Ok((node_of_pb, 0, DMatrix::zeros(sums.n_genes, 0)));
    }
    let zero = DMatrix::<f32>::zeros(sums.n_genes, n_states * n_batches);
    let first = estimate_fold_on_pseudobulks(sums, &node_of_pb, n_states, &zero, params)?;
    let node_of_pb = refine_nodes_by_local_swaps(sums, rule, node_of_pb, n_states, &first);
    let leftover = estimate_fold_on_pseudobulks(sums, &node_of_pb, n_states, &zero, params)?;
    info!(
        "cell-pair batch fold, link states: {} coarse pair groups cut into {n_states} link states",
        n_groups
    );
    Ok((node_of_pb, n_states, leftover))
}

/// The fold over pair units already built and collapsed, by link states:
/// see the module docs. Returns the fold (nodes = (link state, endpoints'
/// cell nodes), over the kept pairs) and the decomposition (over the kept
/// pairs; the caller re-indexes it to the given pairs). The fold's
/// `log_delta` is the pair units' δ, the base of pairs in no node (across
/// batches); a within-batch pair's base is its cell-predicted fold, where an
/// endpoint in no cell node carries the cells' δ.
pub(crate) fn estimate_fold_over_pair_units_by_link_states(
    pu: &CellPairUnits,
    groups: &CollapsePseudobulks<'_>,
    batch_of_cell: &[usize],
    cells: &SampleBatchFold,
    log_delta: &DMatrix<f32>,
    params: &StateAwareBatchFoldParams,
) -> anyhow::Result<(StateAwareBatchFold, LinkStateDecomposition)> {
    let nb = log_delta.ncols();
    let n_genes = log_delta.nrows();
    anyhow::ensure!(
        cells.fold.n_batches == nb,
        "link states: the cells have {} batches, the pair units {nb}",
        cells.fold.n_batches
    );
    let pairs = &pu.kept_pairs;
    let e = pairs.len();
    let (label_of_cell, n_labels) = params.span_labels_of_samples(batch_of_cell, nb)?;
    let rule = BatchSpanningRule::new(params, n_labels);
    let label_of_pair: Vec<usize> = pairs.iter().map(|&(u, _)| label_of_cell[u]).collect();

    // 1. endpoints' cell nodes (unordered) and their cell-predicted fold per batch
    let endpoints: Vec<Option<(Option<usize>, Option<usize>)>> = pairs
        .iter()
        .map(|&(u, v)| {
            let (a, b) = (cells.fold.node_of_unit[u], cells.fold.node_of_unit[v]);
            Some((a.min(b), a.max(b)))
        })
        .collect();
    let (endpoints_of_pair, distinct_endpoints) = compact_present(&endpoints);
    let mut cell_predicted = DMatrix::<f32>::zeros(n_genes, distinct_endpoints.len() * nb);
    for (k, &(a, c)) in distinct_endpoints.iter().enumerate() {
        for b in 0..nb {
            cell_predicted
                .column_mut(k * nb + b)
                .copy_from(&cell_predicted_pair_log_fold(&cells.fold, a, c, b));
        }
    }

    // 2. pb pairs keyed by their endpoints too, divided by their cell-predicted fold
    let mut sums = sum_cell_pairs_into_pseudobulks(
        pu,
        groups,
        batch_of_cell,
        &label_of_pair,
        &cells.coords,
        &endpoints_of_pair,
    )?;
    let mut endpoints_of_pb = vec![0usize; sums.len()];
    for (k, pb) in sums.of_unit.iter().take(e).enumerate() {
        if let (Some(pb), Some(ep)) = (*pb, endpoints_of_pair[k]) {
            endpoints_of_pb[pb] = ep;
        }
    }
    let col_of_pb: Vec<usize> = (0..sums.len())
        .map(|pb| endpoints_of_pb[pb] * nb + sums.batch[pb])
        .collect();
    divide_pb_pairs_by_cell_predicted_fold(&mut sums, &cell_predicted, &col_of_pb);

    // 3–4. link states and their leftover fold
    let (state_of_pb, n_states, leftover) =
        link_states_and_leftover_fold(&mut sums, nb, &rule, params)?;
    if n_states == 0 {
        warn!("cell-pair batch fold: no link state spans enough batches; pairs keep their cell-predicted fold");
    }
    let link_state_log_mean_composition =
        node_log_mean_compositions(&sums, &state_of_pb, n_states, &leftover);

    // the fold's nodes: (link state or none, endpoints' cell nodes)
    let node_key_of_pb: Vec<Option<(Option<usize>, usize)>> = (0..sums.len())
        .map(|pb| Some((state_of_pb[pb], endpoints_of_pb[pb])))
        .collect();
    let (node_of_pb, node_keys) = compact_present(&node_key_of_pb);
    let mut log_fold = DMatrix::<f32>::zeros(n_genes, node_keys.len() * nb);
    for (n, &(s, ep)) in node_keys.iter().enumerate() {
        for b in 0..nb {
            let mut col = cell_predicted.column(ep * nb + b).into_owned();
            if let Some(s) = s {
                col.add_assign(leftover.column(s * nb + b));
            }
            log_fold.column_mut(n * nb + b).copy_from(&col);
        }
    }
    let node_of_unit: Vec<Option<usize>> = (0..e)
        .map(|k| sums.of_unit[k].and_then(|pb| node_of_pb[pb]))
        .collect();
    let link_state_of_pair: Vec<Option<usize>> = (0..e)
        .map(|k| sums.of_unit[k].and_then(|pb| state_of_pb[pb]))
        .collect();
    info!(
        "cell-pair batch fold, link states: {} of {e} pairs in a link state; {} endpoint cell-node pairs",
        link_state_of_pair.iter().filter(|s| s.is_some()).count(),
        distinct_endpoints.len()
    );
    Ok((
        StateAwareBatchFold {
            n_batches: nb,
            n_nodes: node_keys.len(),
            log_fold,
            log_delta: log_delta.clone(),
            node_of_unit,
        },
        LinkStateDecomposition {
            link_state_of_pair,
            leftover_log_fold: leftover,
            n_batches: nb,
            link_state_log_mean_composition,
        },
    ))
}
