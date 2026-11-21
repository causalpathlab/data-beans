//! Cell-pair kind: the state-aware batch fold over cell pairs, for the
//! link-based methods (pinto's lc, dsvd, cage), where the pair is the unit.
//!
//! 1. Every cell's reads are split once over its pairs and one self-loop
//!    ([`split_cell_reads_over_pair_units_without_double_counting`]), so no
//!    read counts twice.
//! 2. Those pair units are collapsed on their own (the self-loops a stratum
//!    of their own): the pair δ and the pair-unit levels.
//! 3. Each pair is placed on the **sample** tree ([`super::sample_units`]) by
//!    its endpoints' nodes, climbing while a pair node does not span batches.
//! 4. The fold is matched pair to pair within each pair node.

use super::coarse_group_ward_tree::CoarseGroupWardTree;
use super::cross_batch_matched_fold::{estimate_fold_by_cross_batch_matching, StateAwareBatchFold};
use super::sample_units::SampleBatchFold;
use super::{BatchSpanningRule, StateAwareBatchFoldParams};
use crate::sparse_io_vector::SparseIoVec;
use legume_numeric::matrix::utils::generate_minibatch_intervals;
use log::{info, warn};
use nalgebra::DMatrix;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;

/// Split every read of every cell once over its **pair units**: its pairs and
/// one self-loop.
///
/// A cell in `deg(u)` pairs has `deg(u) + 1` units; each of its reads goes to
/// one of them uniformly (multinomial, by sequential binomials), so the units
/// together hold every read exactly once — the batch estimate on pairs counts
/// nothing twice. Returns `(gene, unit, count)` triplets: units
/// `0..pairs.len()` are the pairs (each summing its two endpoints' shares),
/// `pairs.len() + c` the self-loop of cell `c`. Counts are rounded to whole
/// reads. Seeds are drawn fresh (not reproducible by design).
pub fn split_cell_reads_over_pair_units_without_double_counting(
    cells: &SparseIoVec,
    pairs: &[(usize, usize)],
    block_size: Option<usize>,
) -> anyhow::Result<Vec<(u64, u64, f32)>> {
    use rand::SeedableRng;
    use rand_distr::{Binomial, Distribution};

    let n = cells.num_columns();
    let e = pairs.len();
    let mut incident: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (k, &(u, v)) in pairs.iter().enumerate() {
        anyhow::ensure!(u < n && v < n, "pair {k} ({u}, {v}) outside {n} cells");
        incident[u].push(k);
        incident[v].push(k);
    }
    for (c, inc) in incident.iter_mut().enumerate() {
        inc.push(e + c);
    }

    let jobs = generate_minibatch_intervals(n, cells.num_rows(), block_size);
    let parts: Vec<Vec<(u64, u64, f32)>> = jobs
        .par_iter()
        .map(|&(lb, ub)| -> anyhow::Result<Vec<(u64, u64, f32)>> {
            let mut rng = rand::rngs::SmallRng::seed_from_u64(rand::random());
            let yy = cells.read_columns_csc(lb..ub)?;
            let mut out = Vec::new();
            for (y, c) in yy.col_iter().zip(lb..ub) {
                let units = &incident[c];
                for (&g, &x) in y.row_indices().iter().zip(y.values()) {
                    let mut left = x.round().max(0.0) as u64;
                    for (j, &unit) in units.iter().enumerate() {
                        if left == 0 {
                            break;
                        }
                        let rest = (units.len() - j) as f64;
                        let k = if rest <= 1.0 {
                            left
                        } else {
                            Binomial::new(left, 1.0 / rest)?.sample(&mut rng)
                        };
                        if k > 0 {
                            out.push((g as u64, unit as u64, k as f32));
                            left -= k;
                        }
                    }
                }
            }
            Ok(out)
        })
        .collect::<anyhow::Result<_>>()?;

    // a pair unit takes shares from both endpoints: merge (gene, unit) repeats
    let mut trip: Vec<(u64, u64, f32)> = parts.into_iter().flatten().collect();
    trip.par_sort_unstable_by_key(|&(g, u, _)| (u, g));
    trip.dedup_by(|next, kept| {
        if next.0 == kept.0 && next.1 == kept.1 {
            kept.2 += next.2;
            true
        } else {
            false
        }
    });
    Ok(trip)
}

/// Unordered (tree node, tree node) of a pair's endpoints → its pair node.
pub type PairNodeOfEndpointNodes = HashMap<(usize, usize), usize>;

/// Each pair's node from its endpoints' tree nodes `at` (`None`: δ), climbing
/// both endpoints to their parents while the pair's key fails `rule`; past
/// the root a pair keeps δ. Returns the nodes, the settled keys and the rounds.
pub(crate) fn place_cell_pairs_on_sample_tree(
    tree: &CoarseGroupWardTree,
    mut at: Vec<Option<(usize, usize)>>,
    label_of_pair: &[usize],
    rule: &BatchSpanningRule,
) -> (Vec<Option<usize>>, PairNodeOfEndpointNodes, usize) {
    let key = |(a, b): (usize, usize)| (a.min(b), a.max(b));
    let mut node = vec![None; at.len()];
    let mut pair_node_of_endpoint_nodes = PairNodeOfEndpointNodes::default();
    let mut pending: Vec<usize> = (0..at.len()).filter(|&p| at[p].is_some()).collect();
    let mut rounds = 0;
    // pairs that reached the root in any round, pooled and tested once at the end
    let mut at_root: Vec<usize> = Vec::new();
    while !pending.is_empty() {
        let mut count: HashMap<(usize, usize), Vec<usize>> = HashMap::default();
        for &p in &pending {
            let k = key(at[p].expect("pending pair"));
            count.entry(k).or_insert_with(|| vec![0; rule.n_labels])[label_of_pair[p]] += 1;
        }
        let mut next = Vec::new();
        for &p in &pending {
            let (a, b) = at[p].expect("pending pair");
            let k = key((a, b));
            if let Some(&n) = pair_node_of_endpoint_nodes.get(&k) {
                // a key settled earlier already spans: a later arrival joins it
                node[p] = Some(n);
            } else if rule.spans(&count[&k]) {
                let n = pair_node_of_endpoint_nodes.len();
                node[p] = Some(*pair_node_of_endpoint_nodes.entry(k).or_insert(n));
            } else if a == tree.root && b == tree.root {
                at_root.push(p);
            } else {
                at[p] = Some((tree.parent[a], tree.parent[b]));
                next.push(p);
            }
        }
        pending = next;
        rounds += 1;
    }
    let mut root_count = vec![0; rule.n_labels];
    for &p in &at_root {
        root_count[label_of_pair[p]] += 1;
    }
    if !at_root.is_empty() && rule.spans(&root_count) {
        let n = pair_node_of_endpoint_nodes.len();
        let n = *pair_node_of_endpoint_nodes
            .entry((tree.root, tree.root))
            .or_insert(n);
        for &p in &at_root {
            node[p] = Some(n);
        }
    }
    (node, pair_node_of_endpoint_nodes, rounds)
}

/// The cell-pair kind's result.
pub struct CellPairBatchFold {
    /// The fold of each pair (unit `e` is pair `e`): its node, and the table.
    pub fold: StateAwareBatchFold,
    /// Unordered (sample-tree node, sample-tree node) of a pair's endpoints →
    /// its pair node, for every key a fitted pair settled at (after climbing).
    pub pair_node_of_endpoint_nodes: PairNodeOfEndpointNodes,
    /// The pair-unit collapse's own δ, log, genes × batches (the fold's base).
    pub pair_unit_log_delta: DMatrix<f32>,
    /// Each pair's group at every level of the pair-unit collapse, finest
    /// first (`group_of_pair_per_level[level][pair]`), for per-level folds.
    pub group_of_pair_per_level: Vec<Vec<usize>>,
}

impl CellPairBatchFold {
    /// The node of an unseen pair whose endpoints sit at sample-tree nodes
    /// `a` and `b`: climb both until a settled key, `None` (δ) past the root.
    pub fn place_unseen_pair(
        &self,
        tree: &CoarseGroupWardTree,
        a: usize,
        b: usize,
    ) -> Option<usize> {
        let (mut a, mut b) = (a, b);
        loop {
            let key = (a.min(b), a.max(b));
            if let Some(&k) = self.pair_node_of_endpoint_nodes.get(&key) {
                return Some(k);
            }
            if a == tree.root && b == tree.root {
                return None;
            }
            (a, b) = (tree.parent[a], tree.parent[b]);
        }
    }

    /// The fold of each group at `level` of the pair-unit collapse (genes ×
    /// groups): the mean log fold of its pairs.
    pub fn fold_per_pair_collapse_level(
        &self,
        level: usize,
        batch_of_pair: &[usize],
    ) -> DMatrix<f32> {
        let groups = &self.group_of_pair_per_level[level];
        let n_groups = groups.iter().max().map_or(0, |&m| m + 1);
        self.fold.per_level(groups, batch_of_pair, n_groups)
    }
}

/// The cell-pair kind, end to end: split every cell's reads once over its
/// pairs and a self-loop, collapse those pair units on their own (self-loops
/// a stratum of their own) for the pair δ and the pair levels, place each pair
/// on the samples kind's tree (`sample_fold`, which must come from the same
/// cells), and match the fold pair to pair within each pair node.
///
/// * `cells`: genes × cells, the columns `sample_fold` was estimated on.
/// * `cell_pairs`: pairs of cell indices; a pair's batch is its first cell's.
/// * `batch_label_of_cell`: each cell's batch label; the pair δ's columns are
///   the sorted distinct labels, as the sample collapse's are.
/// * `pair_unit_collapse`: the collapse settings for the pair units (strata
///   are set here); `proj_dim` its batch-corrected projection's width.
///
/// The pair units live in a temporary file for the estimate only.
pub fn collapse_cell_pairs_and_estimate_state_aware_batch_fold(
    cells: &SparseIoVec,
    cell_pairs: &[(usize, usize)],
    batch_label_of_cell: &[Box<str>],
    sample_fold: &SampleBatchFold,
    pair_unit_collapse: &crate::alg::collapse_data::MultilevelParams,
    proj_dim: usize,
    params: &StateAwareBatchFoldParams,
) -> anyhow::Result<CellPairBatchFold> {
    use crate::alg::collapse_data::collapse_columns_multilevel_with_hierarchy;
    use crate::alg::random_projection::RandProjOps;
    use crate::sparse_io::{create_sparse_from_triplets_owned, SparseIoBackend};
    use legume_numeric::param::traits::Inference;

    let (n_rows, n_cells, n_pairs) = (cells.num_rows(), cells.num_columns(), cell_pairs.len());
    anyhow::ensure!(
        batch_label_of_cell.len() == n_cells,
        "one batch label per cell"
    );
    info!("cell-pair batch fold: splitting reads over {n_pairs} pairs and {n_cells} self-loops");
    let triplets = split_cell_reads_over_pair_units_without_double_counting(
        cells,
        cell_pairs,
        params.block_size,
    )?;
    let n_units = n_pairs + n_cells;
    let nnz = triplets.len();
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("pair_units.zarr.zip");
    let mut backend = create_sparse_from_triplets_owned(
        triplets,
        (n_rows, n_units, nnz),
        Some(
            path.to_str()
                .ok_or_else(|| anyhow::anyhow!("non-utf-8 temporary path"))?,
        ),
        Some(&SparseIoBackend::Zarr),
    )?;
    backend.register_row_names_vec(&cells.row_names()?);
    backend.register_column_names_vec(
        &(0..n_units)
            .map(|u| format!("pair_unit_{u}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    let mut units = SparseIoVec::new();
    units.push(std::sync::Arc::from(backend), None)?;

    // pairs first, then each cell's self-loop; self-loops are their own stratum
    let cell_of_unit = |u: usize| {
        if u < n_pairs {
            cell_pairs[u].0
        } else {
            u - n_pairs
        }
    };
    let unit_label: Vec<Box<str>> = (0..n_units)
        .map(|u| batch_label_of_cell[cell_of_unit(u)].clone())
        .collect();
    let strata: Vec<usize> = (0..n_units).map(|u| usize::from(u >= n_pairs)).collect();
    let proj = units.project_columns_with_batch_correction(proj_dim, None, Some(&unit_label))?;
    let collapsed = collapse_columns_multilevel_with_hierarchy(
        &mut units,
        &proj.proj,
        &unit_label,
        &crate::alg::collapse_data::MultilevelParams {
            strata: Some(strata),
            ..pair_unit_collapse.clone()
        },
    )?;
    let pair_unit_log_delta = collapsed
        .levels
        .first()
        .and_then(|l| l.delta.as_ref())
        .ok_or_else(|| anyhow::anyhow!("the pair-unit collapse gave no batch effect"))?
        .posterior_mean()
        .map(|v| v.max(1e-8).ln());

    let mut labels: Vec<&str> = batch_label_of_cell.iter().map(AsRef::as_ref).collect();
    labels.sort_unstable();
    labels.dedup();
    anyhow::ensure!(
        pair_unit_log_delta.ncols() == labels.len(),
        "pair units span {} of {} batches",
        pair_unit_log_delta.ncols(),
        labels.len()
    );
    let batch_of_cell: Vec<usize> = batch_label_of_cell
        .iter()
        .map(|b| {
            labels
                .binary_search(&b.as_ref())
                .expect("a cell's label is listed")
        })
        .collect();

    let (fold, pair_node_of_endpoint_nodes) = estimate_fold_over_split_pair_units(
        &units,
        cell_pairs,
        &batch_of_cell,
        sample_fold,
        &pair_unit_log_delta,
        params,
    )?;
    let group_of_pair_per_level = collapsed
        .cell_to_pb_per_level
        .into_iter()
        .map(|mut groups| {
            groups.truncate(n_pairs);
            groups
        })
        .collect();
    Ok(CellPairBatchFold {
        fold,
        pair_node_of_endpoint_nodes,
        pair_unit_log_delta,
        group_of_pair_per_level,
    })
}

/// The fold over pair units already split and collapsed: each pair placed on
/// the samples kind's tree by its endpoints, then matched pair to pair.
///
/// * `units`: genes × pair units, columns `0..pairs.len()` the pairs, the
///   rest self-loops (in the collapse's δ only, never a fold).
/// * `log_delta`: δ of the pair-unit collapse, log, genes × batches.
pub(crate) fn estimate_fold_over_split_pair_units(
    units: &SparseIoVec,
    pairs: &[(usize, usize)],
    cell_batch: &[usize],
    cells: &SampleBatchFold,
    log_delta: &DMatrix<f32>,
    params: &StateAwareBatchFoldParams,
) -> anyhow::Result<(StateAwareBatchFold, PairNodeOfEndpointNodes)> {
    let n_units = units.num_columns();
    let e = pairs.len();
    anyhow::ensure!(n_units >= e, "pair units: the pairs come first");
    let n_batches = log_delta.ncols();
    let span_pair = BatchSpanningRule::new(params, cell_batch, n_batches)?;
    let label_of_pair: Vec<usize> = pairs.iter().map(|&(u, _)| span_pair.label[u]).collect();
    let tree = &cells.tree;

    let at: Vec<Option<(usize, usize)>> = pairs
        .iter()
        .map(|&(u, v)| Some((cells.tree_node_of_unit[u]?, cells.tree_node_of_unit[v]?)))
        .collect();
    let (pair_node, pair_node_of_endpoint_nodes, rounds) =
        place_cell_pairs_on_sample_tree(tree, at, &label_of_pair, &span_pair);
    let mut node: Vec<Option<usize>> = vec![None; n_units];
    node[..e].copy_from_slice(&pair_node);
    let n_nodes = pair_node_of_endpoint_nodes.len();
    let kept = pair_node.iter().filter(|n| n.is_some()).count();
    if n_nodes == 0 {
        warn!("cell-pair batch fold: no pair node spans enough batches; every pair keeps δ");
    } else {
        info!(
            "cell-pair batch fold: {n_nodes} pair nodes after {rounds} rounds; {} of {e} pairs keep δ",
            e - kept
        );
    }

    let pair_batch: Vec<usize> = (0..n_units)
        .map(|k| if k < e { cell_batch[pairs[k].0] } else { 0 })
        .collect();
    let mut coords = DMatrix::<f32>::zeros(cells.coords.nrows(), n_units);
    for (k, &(u, v)) in pairs.iter().enumerate() {
        coords
            .column_mut(k)
            .copy_from(&((cells.coords.column(u) + cells.coords.column(v)) * 0.5));
    }
    let mut fold = estimate_fold_by_cross_batch_matching(
        units,
        &pair_batch,
        &coords,
        &node,
        n_nodes,
        log_delta,
        params,
    )?;
    fold.node_of_unit.truncate(e);
    Ok((fold, pair_node_of_endpoint_nodes))
}
