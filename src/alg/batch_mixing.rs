//! Spread batches across cell states.
//!
//! A batch (a donor, a sample, a run) shifts every one of its cells, and it
//! also has its own cell-state composition. Batch correction must remove the
//! first and keep the second. Subtracting each batch's plain mean removes
//! both, so a batch rich in one state has that state pulled to the origin,
//! and cells of one state stop lining up across batches. Here the
//! projection is fit as
//!
//! ```text
//!   proj_j = state(bin_j) + shift(batch_j)
//! ```
//!
//! by alternating the bins (binary codes of the projection) with the batch
//! shifts, so the shift is what is left after each cell's state is removed.
//!
//! The same codes form a binary tree over cell states (bit `k` is the sign
//! of the `k`-th component, so dropping the top bit gives the parent).
//! [`merge_poorly_mixed_bins`] cuts that tree so every bin holds enough
//! batches.

use crate::alg::random_projection::binary_sort_columns;
use nalgebra::DMatrix;
use std::collections::{HashMap, HashSet};

/// Rounds of the state/shift alternation; it stops early once no column
/// changes bin.
pub const CENTRING_ROUNDS: usize = 10;

/// Most state bits used while centring.
pub const MAX_STATE_BITS: usize = 10;

/// Levels a poorly mixed bin may merge up by default.
pub const DEFAULT_MERGE_LEVELS: usize = 8;

/// Batches a bin needs by default (capped by the number of batches).
pub const DEFAULT_MIN_BATCHES_PER_GROUP: usize = 3;

/// State bits for centring: about two columns per batch per bin, at least
/// one, at most the projection dimension and [`MAX_STATE_BITS`].
pub fn default_state_bits(n_cols: usize, n_batches: usize, dim: usize) -> usize {
    let per_bin = (2 * n_batches.max(1)) as f64;
    let bits = (n_cols as f64 / per_bin).log2().floor().max(1.0) as usize;
    bits.min(dim.max(1)).min(MAX_STATE_BITS)
}

/// Per-group column means of `x`; empty groups stay zero.
fn group_means(x: &DMatrix<f32>, group: &[usize], n_groups: usize) -> DMatrix<f32> {
    let mut sum = DMatrix::<f32>::zeros(x.nrows(), n_groups);
    let mut count = vec![0f32; n_groups];
    for (j, &g) in group.iter().enumerate() {
        let mut col = sum.column_mut(g);
        col += x.column(j);
        count[g] += 1.0;
    }
    for (g, &c) in count.iter().enumerate() {
        if c > 0.0 {
            sum.column_mut(g).unscale_mut(c);
        }
    }
    sum
}

/// `x_j - offsets[group_j]` for every column.
fn subtract(x: &DMatrix<f32>, offsets: &DMatrix<f32>, group: &[usize]) -> DMatrix<f32> {
    let mut out = x.clone();
    for (j, &g) in group.iter().enumerate() {
        let mut col = out.column_mut(j);
        col -= offsets.column(g);
    }
    out
}

/// Centre each batch within cell state (see the module docs). Starts from
/// the plain per-batch mean; the result has mean zero over all columns.
///
/// * `proj` - projection (feature x column)
/// * `batch` - column to batch index (`0..n_batches`)
/// * `bits` - state bits for the bins
pub fn centre_batches_within_state(
    proj: &DMatrix<f32>,
    batch: &[usize],
    bits: usize,
) -> anyhow::Result<DMatrix<f32>> {
    anyhow::ensure!(
        batch.len() == proj.ncols(),
        "batch membership size mismatch"
    );
    let n_batches = batch.iter().max().map_or(0, |&m| m + 1);
    let bits = bits.min(proj.nrows()).min(proj.ncols()).max(1);

    let mut shift = group_means(proj, batch, n_batches);
    let mut codes: Vec<usize> = Vec::new();
    for _ in 0..CENTRING_ROUNDS {
        let centred = subtract(proj, &shift, batch);
        let next = binary_sort_columns(&centred, bits)?;
        if next == codes {
            break;
        }
        codes = next;
        let n_bins = codes.iter().max().map_or(0, |&m| m + 1);
        let states = group_means(&centred, &codes, n_bins);
        shift = group_means(&subtract(proj, &states, &codes), batch, n_batches);
    }
    let mut centred = subtract(proj, &shift, batch);
    let mean = centred.column_mean();
    for mut col in centred.column_iter_mut() {
        col -= &mean;
    }
    Ok(centred)
}

/// Label of a node of the code tree: the code with a sentinel bit above its
/// `level` bits, so labels of different levels never collide.
fn node_label(level: usize, code: usize) -> usize {
    (1 << level) | code
}

/// Cut the binary code tree so that bins mix batches. Bottom-up, a node
/// holding fewer than `min_batches` batches collapses its whole parent
/// subtree into the parent, for at most `levels` levels or up to the root.
/// A bin still short after that is left as is. Returns one label per
/// column.
///
/// * `codes` - binary codes of `bits` bits, bit `k` from component `k`
/// * `batch` - column to batch index
pub fn merge_poorly_mixed_bins(
    codes: &[usize],
    batch: &[usize],
    bits: usize,
    levels: usize,
    min_batches: usize,
) -> Vec<usize> {
    let mut node: Vec<(usize, usize)> = codes.iter().map(|&c| (bits, c)).collect();
    let lowest = bits.saturating_sub(levels);
    for b in (lowest + 1..=bits).rev() {
        let mut members: HashMap<usize, HashSet<usize>> = HashMap::default();
        for (&(level, code), &k) in node.iter().zip(batch) {
            if level == b {
                members.entry(code).or_default().insert(k);
            }
        }
        let parent_mask = (1 << (b - 1)) - 1;
        let poor_parents: HashSet<usize> = members
            .iter()
            .filter(|(_, ks)| ks.len() < min_batches)
            .map(|(&code, _)| code & parent_mask)
            .collect();
        if poor_parents.is_empty() {
            continue;
        }
        for (level, code) in node.iter_mut() {
            let parent = *code & parent_mask;
            if *level >= b && poor_parents.contains(&parent) {
                *level = b - 1;
                *code = parent;
            }
        }
    }
    node.into_iter().map(|(l, c)| node_label(l, c)).collect()
}

/// Map arbitrary batch keys to `0..n_batches` in order of first appearance.
pub fn batch_indices<T>(batch_membership: &[T]) -> Vec<usize>
where
    T: std::hash::Hash + Eq + Clone,
{
    let mut index: HashMap<T, usize> = HashMap::default();
    batch_membership
        .iter()
        .map(|t| {
            let next = index.len();
            *index.entry(t.clone()).or_insert(next)
        })
        .collect()
}
