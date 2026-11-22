//! Node compositions, and scoring counts against them: what the local swaps
//! move pseudobulks by, and what places a unit the fit never saw (predict,
//! impute) on a node or a link state.
//!
//! ```text
//! ln p_{k,b} = ln normalize(μ_k ⊙ f_{k,b}),   score(x → k) = Σ_g x_g · ln p_{k,b,g}
//! ```
//!
//! with `μ_k` the node's fold-corrected composition and `f_{k,b}` its fold
//! in batch `b` (carrying δ). A batch the fit never saw has no fold: its
//! units are scored against `μ_k` alone, on counts the caller has divided by
//! that batch's own δ.

use super::pseudobulk_sums::PseudobulkSums;
use nalgebra::DMatrix;
use rayon::prelude::*;

/// Each node's fold-corrected composition, log, genes × nodes:
/// `ln normalize(μ_k + floor)`, `μ_k` the node's sum with each pseudobulk's
/// fold (`log_fold`, genes × (nodes · batches)) divided out. The floor makes
/// a gene the node never showed unlikely, not impossible.
pub(crate) fn node_log_mean_compositions(
    sums: &PseudobulkSums,
    node: &[Option<usize>],
    n_nodes: usize,
    log_fold: &DMatrix<f32>,
) -> DMatrix<f32> {
    let n_genes = sums.n_genes;
    let nb = log_fold.ncols() / n_nodes.max(1);
    let mut mu = DMatrix::<f32>::zeros(n_genes, n_nodes);
    for (pb, k) in node.iter().enumerate() {
        if let Some(k) = *k {
            let col = k * nb + sums.batch[pb];
            for &(g, x) in &sums.counts[pb] {
                mu[(g, k)] += x * (-log_fold[(g, col)]).exp();
            }
        }
    }
    for mut c in mu.column_iter_mut() {
        let floor = (c.sum() / n_genes.max(1) as f32 * 1e-3).max(1e-12);
        c.apply(|v| *v += floor);
        let tot: f32 = c.sum();
        c.apply(|v| *v = (*v / tot).ln());
    }
    mu
}

/// A node's composition in one batch, log: `ln normalize(exp(log_mean) ⊙
/// exp(log_fold))`, from its fold-corrected composition `log_mean` and its
/// fold in that batch.
pub(crate) fn log_composition_in_batch(
    log_mean: nalgebra::DVectorView<'_, f32>,
    log_fold: nalgebra::DVectorView<'_, f32>,
) -> Vec<f32> {
    let s: Vec<f32> = log_mean
        .iter()
        .zip(log_fold.iter())
        .map(|(m, f)| m + f)
        .collect();
    let top = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = top + s.iter().map(|v| (v - top).exp()).sum::<f32>().ln();
    s.into_iter().map(|v| v - lse).collect()
}

/// Each node's composition in each batch, log, genes × (nodes · batches):
/// `ln normalize(μ_k ⊙ fold_{k,b})`.
pub(crate) fn node_log_compositions(
    sums: &PseudobulkSums,
    node: &[Option<usize>],
    n_nodes: usize,
    log_fold: &DMatrix<f32>,
) -> DMatrix<f32> {
    let n_genes = sums.n_genes;
    let nb = log_fold.ncols() / n_nodes;
    let log_mean = node_log_mean_compositions(sums, node, n_nodes, log_fold);
    let cols: Vec<Vec<f32>> = (0..n_nodes * nb)
        .into_par_iter()
        .map(|col| log_composition_in_batch(log_mean.column(col / nb), log_fold.column(col)))
        .collect();
    DMatrix::from_fn(n_genes, n_nodes * nb, |g, col| cols[col][g])
}

/// The log likelihood of `counts` (gene, count) under the composition in
/// column `col` of `log_p` (genes × columns, each a log composition), up
/// to the multinomial constant: `Σ_g x_g · log_p[g, col]`. Swaps score a
/// pseudobulk by it; placement scores a unit the fit never saw by it.
pub(crate) fn composition_score(
    counts: impl Iterator<Item = (usize, f32)>,
    log_p: &DMatrix<f32>,
    col: usize,
) -> f32 {
    counts.map(|(g, x)| x * log_p[(g, col)]).sum()
}

/// The column of `log_p` that explains `counts` best ([`composition_score`]);
/// `None` when `counts` hold no read.
pub(crate) fn best_column_by_composition(
    counts: &[(usize, f32)],
    log_p: &DMatrix<f32>,
) -> Option<usize> {
    if counts.iter().all(|&(_, x)| x <= 0.0) {
        return None;
    }
    (0..log_p.ncols())
        .map(|k| (k, composition_score(counts.iter().copied(), log_p, k)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(k, _)| k)
}

/// Each node's composition in batch `b`, log, genes × nodes, from their
/// fold-corrected compositions `log_mean` (genes × nodes) and their folds
/// `log_fold` (genes × (nodes · `n_batches`)).
fn node_log_compositions_in_batch(
    log_mean: &DMatrix<f32>,
    log_fold: &DMatrix<f32>,
    n_batches: usize,
    b: usize,
) -> DMatrix<f32> {
    let cols: Vec<Vec<f32>> = (0..log_mean.ncols())
        .into_par_iter()
        .map(|k| log_composition_in_batch(log_mean.column(k), log_fold.column(k * n_batches + b)))
        .collect();
    DMatrix::from_fn(log_mean.nrows(), log_mean.ncols(), |g, k| cols[k][g])
}

/// Where units the fit never saw land: each count column (`(gene, count)`
/// pairs) on the node whose composition explains it best
/// ([`composition_score`]). `batch[c]` is `Some(b)` for a batch the fit saw
/// (scored against each node's composition in `b`, the counts as read) or
/// `None` for one it never saw (scored against `log_mean` alone, the counts
/// divided by that batch's own δ). `None` for a column with no read.
pub fn place_by_composition(
    counts: &[Vec<(usize, f32)>],
    batch: &[Option<usize>],
    log_mean: &DMatrix<f32>,
    log_fold: &DMatrix<f32>,
    n_batches: usize,
) -> anyhow::Result<Vec<Option<usize>>> {
    let n_genes = log_mean.nrows();
    anyhow::ensure!(
        counts.len() == batch.len(),
        "placement: {} count columns, {} batch entries",
        counts.len(),
        batch.len()
    );
    anyhow::ensure!(
        log_fold.nrows() == n_genes && log_fold.ncols() == log_mean.ncols() * n_batches,
        "placement: fold {}×{}, compositions {n_genes}×{} over {n_batches} batches",
        log_fold.nrows(),
        log_fold.ncols(),
        log_mean.ncols()
    );
    if let Some(b) = batch.iter().flatten().find(|&&b| b >= n_batches) {
        anyhow::bail!("placement: batch {b}, the fit saw {n_batches}");
    }
    if let Some(g) = counts
        .iter()
        .flatten()
        .map(|&(g, _)| g)
        .find(|&g| g >= n_genes)
    {
        anyhow::bail!("placement: gene {g}, the fit has {n_genes}");
    }
    let mut log_p_of: rustc_hash::FxHashMap<Option<usize>, DMatrix<f32>> = Default::default();
    for &b in batch {
        log_p_of.entry(b).or_insert_with(|| match b {
            Some(b) => node_log_compositions_in_batch(log_mean, log_fold, n_batches, b),
            None => log_mean.clone(),
        });
    }
    Ok(counts
        .par_iter()
        .zip(batch)
        .map(|(x, b)| best_column_by_composition(x, &log_p_of[b]))
        .collect())
}
