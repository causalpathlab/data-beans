//! Uncoarsening: the nodes the cut picked are refined by local swaps of whole
//! pseudobulks between neighbouring nodes, after which the fold is
//! re-estimated. Moving pseudobulks, not single units, keeps every node at
//! pseudobulk sample sizes.
//!
//! Coarsening (Ward, bottom up) groups whole coarse groups, so a pseudobulk
//! placed in the wrong node stays there. Here each pseudobulk may move to a
//! node held by one of its nearest pseudobulks (in the matching coordinates)
//! when that node explains it better. The score is the node's composition
//! **in the pseudobulk's own batch, with that node's fold applied**:
//!
//! ```text
//! score(s → k) = Σ_g x_sg · ln p_{k,b(s),g},   p_{k,b} ∝ μ_k ⊙ f_{k,b}
//! ```
//!
//! with `μ_k` the node's fold-corrected sum and `f` carrying δ. A donor's
//! shifted pseudobulk stays with its state, since the node's fold for that
//! donor already carries the shift. A move that leaves a node no longer
//! spanning batches is undone.

use super::pseudobulk_sums::PseudobulkSums;
use super::BatchSpanningRule;
use legume_numeric::matrix::knn::knn_rows;
use log::info;
use nalgebra::DMatrix;
use rayon::prelude::*;

/// Sweeps of swaps (each: refit the node compositions, then move).
const SWAP_SWEEPS: usize = 2;
/// Nearest pseudobulks whose nodes are a pseudobulk's move candidates.
const SWAP_CANDIDATE_KNN: usize = 15;
/// Rounds of undoing moves that broke a node's batch spanning.
const MAX_REPAIR_ROUNDS: usize = 10;

/// Each node's composition in each batch, log, genes × (nodes · batches):
/// `ln normalize(μ_k ⊙ fold_{k,b})`, `μ_k` the node's fold-corrected sum.
fn node_log_compositions(
    sums: &PseudobulkSums,
    node: &[Option<usize>],
    n_nodes: usize,
    log_fold: &DMatrix<f32>,
) -> DMatrix<f32> {
    let n_genes = sums.n_genes;
    let nb = log_fold.ncols() / n_nodes;
    let mut mu = DMatrix::<f32>::zeros(n_genes, n_nodes);
    for (pb, k) in node.iter().enumerate() {
        if let Some(k) = *k {
            let col = k * nb + sums.batch[pb];
            for &(g, x) in &sums.counts[pb] {
                mu[(g, k)] += x * (-log_fold[(g, col)]).exp();
            }
        }
    }
    let cols: Vec<Vec<f32>> = (0..n_nodes * nb)
        .into_par_iter()
        .map(|col| {
            let k = col / nb;
            // a small floor so a gene the node never showed is unlikely, not impossible
            let floor = (mu.column(k).sum() / n_genes.max(1) as f32 * 1e-3).max(1e-12);
            let p: Vec<f32> = (0..n_genes)
                .map(|g| (mu[(g, k)] + floor) * log_fold[(g, col)].exp())
                .collect();
            let tot: f32 = p.iter().sum();
            p.into_iter().map(|v| (v / tot).ln()).collect()
        })
        .collect();
    DMatrix::from_fn(n_genes, n_nodes * nb, |g, col| cols[col][g])
}

/// Each pseudobulk's best node among its candidates, when it beats its own.
fn propose_moves(
    sums: &PseudobulkSums,
    node: &[Option<usize>],
    candidates: &[Vec<usize>],
    log_p: &DMatrix<f32>,
    nb: usize,
) -> Vec<(usize, usize)> {
    (0..node.len())
        .into_par_iter()
        .filter_map(|pb| {
            let own = node[pb]?;
            let b = sums.batch[pb];
            let score = |k: usize| -> f32 {
                let col = k * nb + b;
                sums.counts[pb]
                    .iter()
                    .map(|&(g, x)| x * log_p[(g, col)])
                    .sum()
            };
            let own_score = score(own);
            let (k, s) = candidates[pb]
                .iter()
                .filter(|&&k| k != own)
                .map(|&k| (k, score(k)))
                .max_by(|a, b| a.1.total_cmp(&b.1))?;
            (s > own_score + 1e-4).then_some((pb, k))
        })
        .collect()
}

/// Undo moves out of any node they would leave not spanning; returns how
/// many were undone.
fn undo_moves_breaking_span(
    sums: &PseudobulkSums,
    rule: &BatchSpanningRule,
    node: &mut [Option<usize>],
    before: &[Option<usize>],
    moves: &[(usize, usize)],
    n_nodes: usize,
) -> usize {
    let mut undone = 0usize;
    for _ in 0..MAX_REPAIR_ROUNDS {
        let mut members = vec![Vec::new(); n_nodes];
        for (pb, k) in node.iter().enumerate() {
            if let Some(k) = *k {
                members[k].push((sums.coarsest[pb], sums.span_label[pb]));
            }
        }
        let failing: Vec<bool> = members
            .into_iter()
            .map(|m| !rule.spans(&rule.count(m)))
            .collect();
        let mut changed = false;
        for &(pb, _) in moves {
            let from = before[pb].expect("moved from a node");
            if failing[from] && node[pb] != before[pb] {
                node[pb] = before[pb];
                undone += 1;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    undone
}

/// The nodes after local swaps of pseudobulks: `node` refined over
/// [`SWAP_SWEEPS`] sweeps, each refitting the node compositions under
/// `log_fold` (genes × (nodes · batches)), then moving every pseudobulk a
/// neighbouring node explains better.
pub(crate) fn refine_nodes_by_local_swaps(
    sums: &PseudobulkSums,
    rule: &BatchSpanningRule,
    mut node: Vec<Option<usize>>,
    n_nodes: usize,
    log_fold: &DMatrix<f32>,
) -> Vec<Option<usize>> {
    if n_nodes < 2 {
        return node;
    }
    let nb = log_fold.ncols() / n_nodes;
    let (nbrs, _) = knn_rows(&sums.coords.transpose(), SWAP_CANDIDATE_KNN);
    for sweep in 0..SWAP_SWEEPS {
        let log_p = node_log_compositions(sums, &node, n_nodes, log_fold);
        let candidates: Vec<Vec<usize>> = (0..node.len())
            .map(|pb| {
                let Some(own) = node[pb] else {
                    return Vec::new();
                };
                let mut c: Vec<usize> = std::iter::once(own)
                    .chain(nbrs[pb].iter().filter_map(|&v| node[v]))
                    .collect();
                c.sort_unstable();
                c.dedup();
                c
            })
            .collect();
        let moves = propose_moves(sums, &node, &candidates, &log_p, nb);
        let before = node.clone();
        for &(pb, k) in &moves {
            node[pb] = Some(k);
        }
        let undone = undo_moves_breaking_span(sums, rule, &mut node, &before, &moves, n_nodes);
        info!(
            "batch fold, local swaps: sweep {}/{SWAP_SWEEPS}: {} of {} pseudobulks moved ({} undone to keep nodes spanning)",
            sweep + 1,
            moves.len() - undone,
            node.iter().filter(|n| n.is_some()).count(),
            undone
        );
        if moves.len() == undone {
            break;
        }
    }
    node
}
