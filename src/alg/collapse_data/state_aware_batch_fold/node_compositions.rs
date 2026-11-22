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
use legume_numeric::matrix::traits::MatOps;
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
        c.add_scalar_mut(floor);
    }
    mu.sum_to_one_columns_inplace();
    mu.apply(|v| *v = v.ln());
    mu
}

/// Each node's composition in each batch, log, genes × (nodes · batches):
/// `ln normalize(μ_k ⊙ fold_{k,b})`.
pub(crate) fn node_log_compositions(
    sums: &PseudobulkSums,
    node: &[Option<usize>],
    n_nodes: usize,
    log_fold: &DMatrix<f32>,
) -> DMatrix<f32> {
    let nb = log_fold.ncols() / n_nodes;
    let log_mean = node_log_mean_compositions(sums, node, n_nodes, log_fold);
    log_compositions_with_fold(&log_mean, log_fold, (0..n_nodes * nb).map(|col| col / nb))
}

/// `ln normalize(exp(log_mean[:, k]) ⊙ exp(log_fold[:, col]))` for each
/// column `col` of `log_fold` taken, `k` its node (`node_of_col`, in the
/// order taken: column `j` of the result is the `j`-th taken).
fn log_compositions_with_fold(
    log_mean: &DMatrix<f32>,
    log_fold: &DMatrix<f32>,
    node_of_col: impl Iterator<Item = usize>,
) -> DMatrix<f32> {
    let cols: Vec<nalgebra::DVector<f32>> = node_of_col
        .enumerate()
        .map(|(col, k)| log_mean.column(k) + log_fold.column(col))
        .collect();
    let mut p = DMatrix::from_columns(&cols);
    p.log_softmax_columns_inplace();
    p
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

/// Places units the fit never saw, by composition: each node's composition
/// in each batch, built once, then any number of count columns scored
/// against them ([`Self::place`]).
pub struct CompositionPlacer {
    /// Per batch the fit saw: each node's composition in it, log, genes × nodes.
    log_p_of_batch: Vec<DMatrix<f32>>,
    /// Each node's fold-corrected composition, log, genes × nodes: what a
    /// batch the fit never saw is scored against.
    log_mean: DMatrix<f32>,
}

impl CompositionPlacer {
    /// From the nodes' fold-corrected compositions `log_mean` (genes ×
    /// nodes) and their folds `log_fold` (genes × (nodes · `n_batches`)).
    pub fn new(
        log_mean: &DMatrix<f32>,
        log_fold: &DMatrix<f32>,
        n_batches: usize,
    ) -> anyhow::Result<Self> {
        let (n_genes, n_nodes) = log_mean.shape();
        anyhow::ensure!(
            log_fold.nrows() == n_genes && log_fold.ncols() == n_nodes * n_batches,
            "placement: fold {}×{}, compositions {n_genes}×{n_nodes} over {n_batches} batches",
            log_fold.nrows(),
            log_fold.ncols(),
        );
        let log_p_of_batch = (0..n_batches)
            .into_par_iter()
            .map(|b| {
                let cols: Vec<usize> = (0..n_nodes).map(|k| k * n_batches + b).collect();
                let fold = log_fold.select_columns(&cols);
                log_compositions_with_fold(log_mean, &fold, 0..n_nodes)
            })
            .collect();
        Ok(Self {
            log_p_of_batch,
            log_mean: log_mean.clone(),
        })
    }

    /// Each count column (`(gene, count)` pairs) on the node whose
    /// composition explains it best ([`composition_score`]). `batch[c]` is
    /// `Some(b)` for a batch the fit saw (scored against each node's
    /// composition in `b`, the counts as read) or `None` for one it never
    /// saw (scored against the fold-corrected compositions alone, the counts
    /// divided by that batch's own δ). `None` for a column with no read.
    /// Errors on a length mismatch, a batch or gene outside the fit.
    pub fn place(
        &self,
        counts: &[Vec<(usize, f32)>],
        batch: &[Option<usize>],
    ) -> anyhow::Result<Vec<Option<usize>>> {
        let (n_genes, nb) = (self.log_mean.nrows(), self.log_p_of_batch.len());
        anyhow::ensure!(
            counts.len() == batch.len(),
            "placement: {} count columns, {} batch entries",
            counts.len(),
            batch.len()
        );
        if let Some(b) = batch.iter().flatten().find(|&&b| b >= nb) {
            anyhow::bail!("placement: batch {b}, the fit saw {nb}");
        }
        if let Some(g) = counts
            .iter()
            .flatten()
            .map(|&(g, _)| g)
            .find(|&g| g >= n_genes)
        {
            anyhow::bail!("placement: gene {g}, the fit has {n_genes}");
        }
        Ok(counts
            .par_iter()
            .zip(batch)
            .map(|(x, b)| {
                let log_p = b.map_or(&self.log_mean, |b| &self.log_p_of_batch[b]);
                best_column_by_composition(x, log_p)
            })
            .collect())
    }
}

/// One placement by [`CompositionPlacer`]: built, then used once.
pub fn place_by_composition(
    counts: &[Vec<(usize, f32)>],
    batch: &[Option<usize>],
    log_mean: &DMatrix<f32>,
    log_fold: &DMatrix<f32>,
    n_batches: usize,
) -> anyhow::Result<Vec<Option<usize>>> {
    CompositionPlacer::new(log_mean, log_fold, n_batches)?.place(counts, batch)
}
