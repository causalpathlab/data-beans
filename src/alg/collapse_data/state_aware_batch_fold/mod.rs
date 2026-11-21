//! State-aware batch fold: one fold per (gene × node × batch), on top of δ.
//!
//! δ (gene × batch) cannot hold the part of a batch effect that depends on the
//! cell state (donor × state), and that part alone keeps batches apart in every
//! local neighbourhood. This module estimates it on coarse nodes that are
//! state-pure **and** span batches.
//!
//! Two kinds of unit, by meaning; keep them apart:
//!
//! * **samples** ([`sample_units`]): columns as they are, cells or carried
//!   pseudobulks; collapsed by the existing sample collapse
//!   (`collapse_columns_multilevel`). For senna, bge and pinto's baselines.
//!   Entry: [`estimate_state_aware_batch_fold_over_samples`].
//! * **cell pairs** ([`cell_pair_units`]): pair units, every cell's reads split
//!   once over its pairs and a self-loop, collapsed on their own for the pair
//!   δ; placed on the **samples kind's tree** by their endpoints. For
//!   link-based methods, where the pair is the unit. Entry:
//!   [`collapse_cell_pairs_and_estimate_state_aware_batch_fold`]; it needs
//!   the samples kind's result first.
//!
//! Shared by both: the tree over the coarsest sample groups and its
//! batch-spanning cut ([`coarse_group_ward_tree`], coarsening, bottom up), the
//! nodes refined by local swaps of samples ([`local_swap_refinement`],
//! uncoarsening, top down), and the fold matched across batches within each
//! node ([`cross_batch_matched_fold`]), estimated top down along the tree,
//! each node a residual over its parent's ([`top_down_fold`]).
//!
//! If not even the root spans enough batches, every unit keeps δ, with a
//! warning; the estimate never fails for that reason.

pub mod cell_pair_units;
pub mod coarse_group_ward_tree;
pub mod cross_batch_matched_fold;
pub(crate) mod local_swap_refinement;
pub mod sample_units;
pub(crate) mod top_down_fold;

pub use cell_pair_units::{
    collapse_cell_pairs_and_estimate_state_aware_batch_fold,
    split_cell_reads_over_pair_units_without_double_counting, CellPairBatchFold,
    PairNodeOfEndpointNodes,
};
pub use coarse_group_ward_tree::{CoarseGroupWardTree, EXACT_WARD_MAX_GROUPS};
pub use cross_batch_matched_fold::{
    estimate_fold_by_cross_batch_matching, MatchingCoordinates, StateAwareBatchFold,
};
pub use sample_units::{estimate_state_aware_batch_fold_over_samples, SampleBatchFold};

/// Nearest units drawn from each other batch for a unit's counterfactual.
pub const DEFAULT_FOLD_KNN: usize = 10;
/// Pseudo-count on both sides of a node's fold ratio (shrinks thin genes to 1).
pub const DEFAULT_FOLD_PSEUDO_COUNT: f32 = 10.0;
/// Units a batch (or donor) needs inside a node to count as spanned.
pub const DEFAULT_MIN_UNITS: usize = 20;
/// Principal components of the projection the matching runs in.
pub(crate) const COORD_DIM: usize = 30;
/// Principal components of the group profiles Ward links.
pub(crate) const PROFILE_DIM: usize = 20;

#[derive(Clone, Debug)]
pub struct StateAwareBatchFoldParams {
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

impl Default for StateAwareBatchFoldParams {
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
pub(crate) struct BatchSpanningRule {
    pub(crate) label: Vec<usize>,
    pub(crate) n_labels: usize,
    pub(crate) min_spans: usize,
    pub(crate) min_units: usize,
}

impl BatchSpanningRule {
    pub(crate) fn new(
        params: &StateAwareBatchFoldParams,
        batch: &[usize],
        n_batches: usize,
    ) -> anyhow::Result<Self> {
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
    pub(crate) fn spans(&self, count: &[usize]) -> bool {
        count.iter().filter(|&&c| c >= self.min_units).count() >= self.min_spans
    }
}

#[cfg(test)]
mod tests;
