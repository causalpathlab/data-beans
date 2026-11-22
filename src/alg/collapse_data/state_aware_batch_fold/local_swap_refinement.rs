//! Uncoarsening: the nodes the cut picked are refined by local swaps of
//! samples between neighbouring nodes, after which the fold is re-estimated.
//!
//! Coarsening (Ward, bottom up) groups whole coarse groups, so a group placed
//! in the wrong node stays there. Here each sample may move to a node held by
//! one of its nearest neighbours (in the matching coordinates) when that node
//! explains it better. The score is the node's composition **in the sample's
//! own batch, with that node's fold applied**:
//!
//! ```text
//! score(u → k) = Σ_g x_ug · ln p_{k,b(u),g},   p_{k,b} ∝ μ_k ⊙ δ_b ⊙ f_{k,b}
//! ```
//!
//! so a sample is compared with the node as its own batch would see it: a
//! donor's shifted cell stays with its state, since the node's fold for that
//! donor already carries the shift. (A δ-adjusted profile alone would still
//! hold the donor × state residual and pull such cells into nodes of their
//! own.) A move that leaves a node no longer spanning batches is undone.

use super::cross_batch_matched_fold::StateAwareBatchFold;
use super::{pseudobulks_per_label, BatchSpanningRule};
use crate::sparse_io_vector::SparseIoVec;
use legume_numeric::matrix::knn::knn_rows;
use legume_numeric::matrix::utils::generate_minibatch_intervals;
use log::info;
use nalgebra::DMatrix;
use rayon::prelude::*;
use std::sync::Mutex;

/// Sweeps of swaps (each: refit the node compositions, then move).
pub(crate) const SWAP_SWEEPS: usize = 2;
/// Nearest samples whose nodes are a sample's move candidates.
pub(crate) const SWAP_CANDIDATE_KNN: usize = 15;
/// Rounds of undoing moves that broke a node's batch spanning.
const MAX_REPAIR_ROUNDS: usize = 10;

/// What the swaps read.
pub(crate) struct SwapInputs<'a> {
    pub data: &'a SparseIoVec,
    pub batch: &'a [usize],
    /// Matching coordinates, components × samples.
    pub coords: &'a DMatrix<f32>,
    pub rule: &'a BatchSpanningRule,
    /// Each sample's pseudobulk: nodes must keep spanning in pseudobulks.
    pub pseudobulk_of_unit: &'a [usize],
    pub block_size: Option<usize>,
}

/// Each node's composition in each batch, log, genes × (nodes · batches):
/// `ln p_{k,b} = ln normalize(μ_k ⊙ fold_{k,b})`, with `μ_k` the node's
/// fold-corrected sum and `fold` carrying δ.
fn node_log_compositions(
    inp: &SwapInputs<'_>,
    node: &[Option<usize>],
    n_nodes: usize,
    fold: &StateAwareBatchFold,
) -> anyhow::Result<DMatrix<f32>> {
    let n_genes = fold.log_delta.nrows();
    let nb = fold.n_batches;
    let inv_fold = fold.log_fold.map(|v| (-v).exp());
    let active: Vec<usize> = (0..node.len()).filter(|&u| node[u].is_some()).collect();
    let acc = Mutex::new(DMatrix::<f32>::zeros(n_genes, n_nodes));
    let jobs = generate_minibatch_intervals(active.len(), inp.data.num_rows(), inp.block_size);
    jobs.par_iter()
        .try_for_each(|&(lb, ub)| -> anyhow::Result<()> {
            let yy = inp.data.read_columns_csc(active[lb..ub].iter().copied())?;
            let mut local = Vec::new();
            for (y, &u) in yy.col_iter().zip(&active[lb..ub]) {
                let k = node[u].expect("active");
                let col = k * nb + inp.batch[u];
                for (&g, &x) in y.row_indices().iter().zip(y.values()) {
                    local.push((g, k, x * inv_fold[(g, col)]));
                }
            }
            let mut m = acc.lock().expect("node sums");
            for (g, k, v) in local {
                m[(g, k)] += v;
            }
            Ok(())
        })?;
    let mu = acc.into_inner().expect("node sums");
    let mut out = DMatrix::<f32>::zeros(n_genes, n_nodes * nb);
    for k in 0..n_nodes {
        // a small floor so a gene the node never showed is unlikely, not impossible
        let floor = (mu.column(k).sum() / n_genes.max(1) as f32 * 1e-3).max(1e-12);
        for b in 0..nb {
            let col = k * nb + b;
            let mut p: Vec<f32> = (0..n_genes)
                .map(|g| (mu[(g, k)] + floor) * fold.log_fold[(g, col)].exp())
                .collect();
            let tot: f32 = p.iter().sum();
            p.iter_mut().for_each(|v| *v = (*v / tot).ln());
            out.column_mut(col).copy_from_slice(&p);
        }
    }
    Ok(out)
}

/// Each sample's best node among its candidates, when it beats its own.
fn propose_moves(
    inp: &SwapInputs<'_>,
    node: &[Option<usize>],
    candidates: &[Vec<usize>],
    log_p: &DMatrix<f32>,
    nb: usize,
) -> anyhow::Result<Vec<(usize, usize)>> {
    let movable: Vec<usize> = (0..node.len())
        .filter(|&u| node[u].is_some() && candidates[u].len() > 1)
        .collect();
    let jobs = generate_minibatch_intervals(movable.len(), inp.data.num_rows(), inp.block_size);
    let parts: Vec<Vec<(usize, usize)>> = jobs
        .par_iter()
        .map(|&(lb, ub)| -> anyhow::Result<Vec<(usize, usize)>> {
            let yy = inp.data.read_columns_csc(movable[lb..ub].iter().copied())?;
            let mut out = Vec::new();
            for (y, &u) in yy.col_iter().zip(&movable[lb..ub]) {
                let b = inp.batch[u];
                let score = |k: usize| -> f32 {
                    let col = k * nb + b;
                    y.row_indices()
                        .iter()
                        .zip(y.values())
                        .map(|(&g, &x)| x * log_p[(g, col)])
                        .sum()
                };
                let own = node[u].expect("movable");
                let own_score = score(own);
                let best = candidates[u]
                    .iter()
                    .filter(|&&k| k != own)
                    .map(|&k| (k, score(k)))
                    .max_by(|a, b| a.1.total_cmp(&b.1));
                if let Some((k, s)) = best {
                    if s > own_score + 1e-4 {
                        out.push((u, k));
                    }
                }
            }
            Ok(out)
        })
        .collect::<anyhow::Result<_>>()?;
    Ok(parts.into_iter().flatten().collect())
}

/// The nodes after local swaps: `node` refined over [`SWAP_SWEEPS`] sweeps,
/// each refitting the node compositions under `fold`, then moving every
/// sample that a neighbouring node explains better, undoing the moves out of
/// any node they would leave no longer spanning batches.
pub(crate) fn refine_nodes_by_local_swaps(
    inp: &SwapInputs<'_>,
    mut node: Vec<Option<usize>>,
    n_nodes: usize,
    fold: &StateAwareBatchFold,
) -> anyhow::Result<Vec<Option<usize>>> {
    if n_nodes < 2 {
        return Ok(node);
    }
    let nb = fold.n_batches;
    let (nbrs, _) = knn_rows(&inp.coords.transpose(), SWAP_CANDIDATE_KNN);
    for sweep in 0..SWAP_SWEEPS {
        let log_p = node_log_compositions(inp, &node, n_nodes, fold)?;
        let candidates: Vec<Vec<usize>> = (0..node.len())
            .map(|u| {
                let Some(own) = node[u] else {
                    return Vec::new();
                };
                let mut c: Vec<usize> = std::iter::once(own)
                    .chain(nbrs[u].iter().filter_map(|&v| node[v]))
                    .collect();
                c.sort_unstable();
                c.dedup();
                c
            })
            .collect();
        let moves = propose_moves(inp, &node, &candidates, &log_p, nb)?;
        let before = node.clone();
        for &(u, k) in &moves {
            node[u] = Some(k);
        }
        // undo the moves out of every node they would leave not spanning
        let mut undone = 0usize;
        for _ in 0..MAX_REPAIR_ROUNDS {
            let mut units_of_node = vec![Vec::new(); n_nodes];
            for (u, k) in node.iter().enumerate() {
                if let Some(k) = *k {
                    units_of_node[k].push(u);
                }
            }
            let count: Vec<Vec<usize>> = units_of_node
                .into_iter()
                .map(|units| {
                    pseudobulks_per_label(
                        units,
                        inp.pseudobulk_of_unit,
                        &inp.rule.label,
                        inp.rule.n_labels,
                    )
                })
                .collect();
            let failing: Vec<bool> = count.iter().map(|c| !inp.rule.spans(c)).collect();
            let mut changed = false;
            for &(u, _) in &moves {
                let from = before[u].expect("moved from a node");
                if failing[from] && node[u] != before[u] {
                    node[u] = before[u];
                    undone += 1;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        info!(
            "batch fold, local swaps: sweep {}/{SWAP_SWEEPS}: {} of {} samples moved ({} undone to keep nodes spanning)",
            sweep + 1,
            moves.len() - undone,
            node.iter().filter(|n| n.is_some()).count(),
            undone
        );
        if moves.len() == undone {
            break;
        }
    }
    Ok(node)
}
