//! State-aware batch fold: one fold per (gene × node × batch), on top of δ.
//!
//! δ (gene × batch) cannot hold the part of a batch effect that depends on the
//! cell state (donor × state), and that part alone keeps batches apart in every
//! local neighbourhood. This module estimates it on coarse nodes that are
//! state-pure **and** span batches, at pseudobulk sample sizes.
//!
//! Two kinds of unit, by meaning; keep them apart:
//!
//! * **samples** ([`sample_units`]): columns as they are, cells or carried
//!   pseudobulks, with the existing sample collapse's groups. For senna, bge
//!   and pinto's baselines. Entry: [`estimate_state_aware_batch_fold_over_samples`].
//! * **cell pairs** ([`cell_pair_units`]): pair units built once
//!   ([`build_cell_pair_units`]: every cell's reads split once over its pairs
//!   and a self-loop), collapsed on their own for the pair δ. A pair's state
//!   ([`PairStates`]) is its **link state** by default (learned from the
//!   pair after its cells' fold is divided out, [`link_state_pair_fold`]),
//!   or its **endpoint states** (placed on the samples kind's tree by its
//!   endpoints). For link-based methods, where the pair is the unit. Entry:
//!   [`collapse_cell_pairs_and_estimate_state_aware_batch_fold`]; it needs the
//!   samples kind's result first.
//!
//! Both read the data once, into per-pseudobulk sums
//! ([`pseudobulk_sums`]); everything after runs on those sums: the tree over
//! the coarsest sample groups and its batch-spanning cut
//! ([`coarse_group_ward_tree`], coarsening, bottom up), local swaps of whole
//! pseudobulks between nodes ([`local_swap_refinement`], uncoarsening, top
//! down), and the fold matched pseudobulk to pseudobulk across batches within
//! each node ([`cross_batch_matched_fold`]). The samples kind estimates the
//! fold top down along the tree, each node a residual over its parent's
//! ([`top_down_fold`]); the cell-pair kind matches once per pair node (top
//! down over pair keys did worse on the sims, 2026-11-21).
//!
//! If not even the root spans enough batches, every unit keeps δ, with a
//! warning; the estimate never fails for that reason.

pub mod cell_pair_units;
pub mod coarse_group_ward_tree;
pub mod cross_batch_matched_fold;
pub mod link_state_pair_fold;
pub(crate) mod local_swap_refinement;
pub mod node_compositions;
pub(crate) mod pseudobulk_sums;
pub mod sample_units;
pub(crate) mod top_down_fold;

pub use cell_pair_units::{
    build_cell_pair_units, collapse_cell_pairs_and_estimate_state_aware_batch_fold,
    split_cell_reads_over_pair_units_without_double_counting, CellPairBatchFold,
    CellPairFoldParams, CellPairUnits, PairNodeOfEndpointNodes, PairStateModel, PairStates,
};
pub use coarse_group_ward_tree::{CoarseGroupWardTree, EXACT_WARD_MAX_GROUPS};
pub use cross_batch_matched_fold::{MatchingCoordinates, StateAwareBatchFold};
pub use link_state_pair_fold::{cell_predicted_pair_log_fold, LinkStateDecomposition};
pub use node_compositions::{place_by_composition, CompositionPlacer};
pub use sample_units::{
    estimate_state_aware_batch_fold_over_samples, SampleBatchFold, UnseenPlacement,
};

/// Nearest pseudobulks drawn from each other batch for a counterfactual.
pub const DEFAULT_FOLD_KNN: usize = 10;
/// Pseudo-count on both sides of a node's fold ratio (shrinks thin genes to 1).
pub const DEFAULT_FOLD_PSEUDO_COUNT: f32 = 10.0;
/// Principal components of the projection the matching runs in.
pub(crate) const COORD_DIM: usize = 30;
/// Principal components of the group profiles Ward links.
pub(crate) const PROFILE_DIM: usize = 20;

#[derive(Clone, Debug)]
pub struct StateAwareBatchFoldParams {
    pub knn: usize,
    pub pseudo_count: f32,
    /// Coarsest pseudobulks (a collapse's coarsest group in one batch: pb
    /// samples for samples, pb pairs for pairs) a batch (or donor) needs
    /// inside a node to count as spanned: the matching's `knn` by default, so
    /// a node keeps pseudobulk sample sizes. With few coarsest groups (say
    /// 16) the cut makes few, large nodes; with fewer than this many, nothing
    /// spans and every unit keeps δ, with a warning.
    pub min_pseudobulks: usize,
    /// Label a node must span, per column of the samples (cells): a donor,
    /// say. `None` spans batches.
    pub span_labels: Option<Vec<usize>>,
    /// Distinct labels a node must span, each with `min_pseudobulks`
    /// pseudobulks. `None`: half the batches (rounded up), or 2 donors when
    /// `span_labels` is set.
    pub min_spans: Option<usize>,
    pub block_size: Option<usize>,
}

impl Default for StateAwareBatchFoldParams {
    fn default() -> Self {
        Self {
            knn: DEFAULT_FOLD_KNN,
            pseudo_count: DEFAULT_FOLD_PSEUDO_COUNT,
            min_pseudobulks: DEFAULT_FOLD_KNN,
            span_labels: None,
            min_spans: None,
            block_size: None,
        }
    }
}

impl StateAwareBatchFoldParams {
    /// Each sample's span label (its donor, or its batch) and how many there are.
    pub(crate) fn span_labels_of_samples(
        &self,
        batch: &[usize],
        n_batches: usize,
    ) -> anyhow::Result<(Vec<usize>, usize)> {
        match &self.span_labels {
            Some(l) => {
                anyhow::ensure!(l.len() == batch.len(), "span labels: one per column");
                Ok((l.clone(), l.iter().max().map_or(0, |&m| m + 1)))
            }
            None => Ok((batch.to_vec(), n_batches)),
        }
    }
}

/// A collapse's groups, per unit: the finest (each in one batch, the
/// pseudobulks the fold matches) and the coarsest (each in one batch, the
/// pseudobulks a node's size is counted in; the tree's leaves for samples).
#[derive(Clone, Copy, Debug)]
pub struct CollapsePseudobulks<'a> {
    pub finest: &'a [usize],
    pub coarsest: &'a [usize],
}

/// When a node spans: how many labels there are, how many it must span, and
/// how many coarsest pseudobulks a label needs in it.
pub(crate) struct BatchSpanningRule {
    pub(crate) n_labels: usize,
    pub(crate) min_spans: usize,
    pub(crate) min_pseudobulks: usize,
}

impl BatchSpanningRule {
    pub(crate) fn new(params: &StateAwareBatchFoldParams, n_labels: usize) -> Self {
        let default_min = if params.span_labels.is_some() {
            2
        } else {
            n_labels.div_ceil(2)
        };
        Self {
            n_labels,
            min_spans: params.min_spans.unwrap_or(default_min).max(2),
            min_pseudobulks: params.min_pseudobulks,
        }
    }

    /// Does a node holding `count[label]` coarsest pseudobulks span enough labels?
    pub(crate) fn spans(&self, count: &[usize]) -> bool {
        count.iter().filter(|&&c| c >= self.min_pseudobulks).count() >= self.min_spans
    }

    /// The distinct `(coarsest pseudobulk, label)` keys among `members`, per
    /// label: what [`Self::spans`] reads.
    pub(crate) fn count(&self, members: impl IntoIterator<Item = (usize, usize)>) -> Vec<usize> {
        let seen: rustc_hash::FxHashSet<(usize, usize)> = members.into_iter().collect();
        let mut count = vec![0usize; self.n_labels];
        for (_, l) in seen {
            count[l] += 1;
        }
        count
    }
}

/// Dense ids `0..` for the `Some` values of `labels`, in order of first
/// appearance (data-beans' `compact_labels`), and the distinct values in
/// that order.
pub(crate) fn compact_present<K: Copy + Eq + std::hash::Hash>(
    labels: &[Option<K>],
) -> (Vec<Option<usize>>, Vec<K>) {
    let present: Vec<K> = labels.iter().flatten().copied().collect();
    let (dense, n) = crate::alg::dc_poisson::compact_labels(&present);
    let mut distinct = Vec::with_capacity(n);
    let mut it = dense.into_iter();
    let out = labels
        .iter()
        .map(|l| {
            l.map(|k| {
                let d = it.next().expect("one dense id per present label");
                if d == distinct.len() {
                    distinct.push(k);
                }
                d
            })
        })
        .collect();
    (out, distinct)
}

#[cfg(test)]
mod tests;
