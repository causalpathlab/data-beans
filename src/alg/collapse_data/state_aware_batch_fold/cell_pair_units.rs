//! Cell-pair kind: the state-aware batch fold over cell pairs, for the
//! link-based methods (pinto's lc, dsvd, cage), where the pair is the unit.
//!
//! 1. Pair units, built once ([`build_cell_pair_units`]): every cell's reads
//!    split once over its pairs and one self-loop, so no read counts twice.
//!    A pair across two batches has no one batch to be read in: it is left
//!    out here, its reads going to its cells' other units, and keeps δ.
//! 2. Those units collapsed on their own (self-loops a stratum of their
//!    own): the pair δ, and the pb pairs (finest and coarsest groups).
//! 3. Each pair's state, by [`PairStates`]:
//!    * [`PairStates::LinkStates`] (default): learned from the pair itself,
//!      after its cells' fold is divided out
//!      ([`super::link_state_pair_fold`]);
//!    * [`PairStates::EndpointStates`]: the pair of its endpoints' nodes on
//!      the **sample** tree ([`super::sample_units`]); a key that does not
//!      span batches moves up.
//! 4. The fold matched pb pair to pb pair within each pair state, once.

use super::coarse_group_ward_tree::CoarseGroupWardTree;
use super::cross_batch_matched_fold::{
    delta_as_base_of_every_node, estimate_fold_on_pseudobulks, StateAwareBatchFold,
};
use super::link_state_pair_fold::{
    estimate_fold_over_pair_units_by_link_states, LinkStateDecomposition,
};
use super::pseudobulk_sums::{sum_by_pseudobulk, PseudobulkKeys, PseudobulkSums};
use super::sample_units::SampleBatchFold;
use super::{BatchSpanningRule, CollapsePseudobulks, StateAwareBatchFoldParams};
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
                    anyhow::ensure!(
                        x >= 0.0 && x.fract() == 0.0,
                        "cell {c}: count {x} is not a whole number of reads; the pair split needs raw counts"
                    );
                    let mut left = x as u64;
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

/// Unordered (sample-tree node, sample-tree node) of a pair's endpoints → its
/// pair node.
pub type PairNodeOfEndpointNodes = HashMap<(usize, usize), usize>;

/// A pair key's depth: its deeper endpoint's.
fn key_depth(tree: &CoarseGroupWardTree, (a, b): (usize, usize)) -> usize {
    tree.depth[a].max(tree.depth[b])
}

/// A pair key one step up: each endpoint at the key's depth moves to its
/// parent (a shallower endpoint stays), so the depth drops by one; (root,
/// root) is its own parent.
fn parent_key(tree: &CoarseGroupWardTree, (a, b): (usize, usize)) -> (usize, usize) {
    let d = key_depth(tree, (a, b));
    let up = |x: usize| {
        if tree.depth[x] == d {
            tree.parent[x]
        } else {
            x
        }
    };
    let (a, b) = (up(a), up(b));
    (a.min(b), a.max(b))
}

/// Each pair's node from its endpoints' tree nodes `at` (`None`: δ). Keys are
/// tested deepest first, each with every pair that reached it: a key that
/// spans `rule` (counted in each pair's `(coarsest pb pair, label)`) becomes
/// a pair node; the pairs of one that does not move to its parent key; past
/// (root, root) a pair keeps δ.
pub(crate) fn place_cell_pairs_on_sample_tree(
    tree: &CoarseGroupWardTree,
    at: &[Option<(usize, usize)>],
    count_key_of_pair: &[(usize, usize)],
    rule: &BatchSpanningRule,
) -> (Vec<Option<usize>>, PairNodeOfEndpointNodes) {
    let mut node = vec![None; at.len()];
    let mut pair_node_of_endpoint_nodes = PairNodeOfEndpointNodes::default();
    // pairs waiting at each key, by key depth
    let mut waiting: std::collections::BTreeMap<usize, HashMap<(usize, usize), Vec<usize>>> =
        Default::default();
    for (p, k) in at.iter().enumerate() {
        if let Some((a, b)) = *k {
            let key = (a.min(b), a.max(b));
            waiting
                .entry(key_depth(tree, key))
                .or_default()
                .entry(key)
                .or_default()
                .push(p);
        }
    }
    while let Some((d, keys)) = waiting.pop_last() {
        for (key, pairs) in keys {
            if rule.spans(&rule.count(pairs.iter().map(|&p| count_key_of_pair[p]))) {
                let n = pair_node_of_endpoint_nodes.len();
                let n = *pair_node_of_endpoint_nodes.entry(key).or_insert(n);
                for p in pairs {
                    node[p] = Some(n);
                }
            } else if d > 0 {
                waiting
                    .entry(d - 1)
                    .or_default()
                    .entry(parent_key(tree, key))
                    .or_default()
                    .extend(pairs);
            }
        }
    }
    (node, pair_node_of_endpoint_nodes)
}

/// Pair units: the within-batch pairs, then one self-loop per cell, in a
/// temporary file that lives as long as this.
pub struct CellPairUnits {
    /// genes × units: `kept_pairs.len()` pairs, then a self-loop per cell.
    pub units: SparseIoVec,
    /// The pairs given that are within one batch, in their order.
    pub kept_pairs: Vec<(usize, usize)>,
    /// Each given pair's unit (`None`: across batches, left out).
    pub unit_of_pair: Vec<Option<usize>>,
    /// Each unit's batch label.
    pub batch_label: Vec<Box<str>>,
    /// Each unit's stratum: 0 a pair, 1 a self-loop.
    pub stratum: Vec<usize>,
    _dir: tempfile::TempDir,
}

/// Build the pair units of `cell_pairs` over `cells`: every cell's reads split
/// once over its within-batch pairs and its self-loop
/// ([`split_cell_reads_over_pair_units_without_double_counting`]).
pub fn build_cell_pair_units(
    cells: &SparseIoVec,
    cell_pairs: &[(usize, usize)],
    batch_label_of_cell: &[Box<str>],
    block_size: Option<usize>,
) -> anyhow::Result<CellPairUnits> {
    use crate::sparse_io::{create_sparse_from_triplets_owned, SparseIoBackend};
    let (n_rows, n_cells) = (cells.num_rows(), cells.num_columns());
    anyhow::ensure!(
        batch_label_of_cell.len() == n_cells,
        "one batch label per cell"
    );
    let mut kept_pairs = Vec::new();
    let unit_of_pair: Vec<Option<usize>> = cell_pairs
        .iter()
        .map(|&(u, v)| {
            (batch_label_of_cell[u] == batch_label_of_cell[v]).then(|| {
                kept_pairs.push((u, v));
                kept_pairs.len() - 1
            })
        })
        .collect();
    let n_pairs = kept_pairs.len();
    if n_pairs < cell_pairs.len() {
        info!(
            "cell-pair batch fold: {} of {} pairs cross batches; they keep δ",
            cell_pairs.len() - n_pairs,
            cell_pairs.len()
        );
    }
    let triplets =
        split_cell_reads_over_pair_units_without_double_counting(cells, &kept_pairs, block_size)?;
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
    let cell_of_unit = |u: usize| {
        if u < n_pairs {
            kept_pairs[u].0
        } else {
            u - n_pairs
        }
    };
    Ok(CellPairUnits {
        units,
        batch_label: (0..n_units)
            .map(|u| batch_label_of_cell[cell_of_unit(u)].clone())
            .collect(),
        stratum: (0..n_units).map(|u| usize::from(u >= n_pairs)).collect(),
        kept_pairs,
        unit_of_pair,
        _dir: dir,
    })
}

/// Where a cell pair's state (its node, the fold's column) comes from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PairStates {
    /// Learned from the pair itself: each pair divided by the fold its two
    /// cells predict (the mean log of their cell folds), then Ward over the
    /// coarsest pair groups on what is left, cut where both children span
    /// batches. A pair's fold = its cell-predicted fold + its link state's
    /// leftover ([`super::link_state_pair_fold`]).
    #[default]
    LinkStates,
    /// The pair of its endpoints' nodes on the samples kind's tree, climbing
    /// while that key does not span batches. A pair's fold = the pair δ + its
    /// endpoint key's matched fold.
    EndpointStates,
}

/// The cell-pair kind's result.
pub struct CellPairBatchFold {
    /// The fold of each given pair (`node_of_unit[e]` for pair `e`; a pair
    /// across batches keeps δ).
    pub fold: StateAwareBatchFold,
    /// Which states the fold was matched within.
    pub pair_states: PairStates,
    /// [`PairStates::EndpointStates`]: unordered (sample-tree node,
    /// sample-tree node) of a pair's endpoints → its pair node, for every key
    /// a fitted pair settled at. Empty for link states.
    pub pair_node_of_endpoint_nodes: PairNodeOfEndpointNodes,
    /// [`PairStates::LinkStates`]: the fold's decomposition, cell-predicted +
    /// leftover, with each given pair's link state. `None` for endpoint
    /// states.
    pub link_states: Option<LinkStateDecomposition>,
    /// The pair-unit collapse's own δ, log, genes × batches (the endpoint
    /// states' base; a pair across batches keeps it).
    pub pair_unit_log_delta: DMatrix<f32>,
}

impl CellPairBatchFold {
    /// [`PairStates::EndpointStates`]: the node of an unseen pair whose
    /// endpoints sit at sample-tree nodes `a` and `b`
    /// ([`SampleBatchFold::place_unseen_samples`]'s `tree_node`): up its keys
    /// as placement walked them, to the first settled one; `None` (δ) past
    /// the root. Link states place an unseen pair through
    /// [`LinkStateDecomposition::unseen_pair_log_fold`].
    pub fn place_unseen_pair(
        &self,
        tree: &CoarseGroupWardTree,
        a: usize,
        b: usize,
    ) -> Option<usize> {
        let mut key = (a.min(b), a.max(b));
        loop {
            if let Some(&k) = self.pair_node_of_endpoint_nodes.get(&key) {
                return Some(k);
            }
            if key_depth(tree, key) == 0 {
                return None;
            }
            key = parent_key(tree, key);
        }
    }
}

/// The cell-pair kind's settings: which states pairs are matched within, and
/// the fold's settings (shared with the samples kind).
#[derive(Clone, Debug, Default)]
pub struct CellPairFoldParams {
    pub pair_states: PairStates,
    pub fold: StateAwareBatchFoldParams,
}

/// The cell-pair kind, end to end: build the pair units, collapse them
/// (self-loops their own stratum) for the pair δ and the pb pairs, find each
/// pair's state (`pair_params.pair_states`; both build on `sample_fold`, from
/// the same cells), and match the fold pb pair to pb pair within each state.
///
/// * `cells`: genes × cells, the columns `sample_fold` was estimated on.
/// * `cell_pairs`: pairs of cell indices.
/// * `batch_label_of_cell`: each cell's batch label.
/// * `pair_unit_collapse`: the collapse settings for the pair units;
///   `proj_dim` its batch-corrected projection's width.
pub fn collapse_cell_pairs_and_estimate_state_aware_batch_fold(
    cells: &SparseIoVec,
    cell_pairs: &[(usize, usize)],
    batch_label_of_cell: &[Box<str>],
    sample_fold: &SampleBatchFold,
    pair_unit_collapse: &crate::alg::collapse_data::MultilevelParams,
    proj_dim: usize,
    pair_params: &CellPairFoldParams,
) -> anyhow::Result<CellPairBatchFold> {
    let (pair_states, params) = (pair_params.pair_states, &pair_params.fold);
    use crate::alg::collapse_data::collapse_columns_multilevel_with_strata;
    use crate::alg::random_projection::RandProjOps;
    use legume_numeric::param::traits::Inference;

    let mut pu = build_cell_pair_units(cells, cell_pairs, batch_label_of_cell, params.block_size)?;
    info!(
        "cell-pair batch fold: {} pairs and {} self-loops",
        pu.kept_pairs.len(),
        cells.num_columns()
    );
    let proj =
        pu.units
            .project_columns_with_batch_correction(proj_dim, None, Some(&pu.batch_label))?;
    let collapsed = collapse_columns_multilevel_with_strata(
        &mut pu.units,
        &proj.proj,
        &pu.batch_label,
        pair_unit_collapse,
        &pu.stratum,
    )?;
    let pair_unit_log_delta = collapsed
        .levels
        .first()
        .and_then(|l| l.delta.as_ref())
        .ok_or_else(|| anyhow::anyhow!("the pair-unit collapse gave no batch effect"))?
        .posterior_mean()
        .map(|v| v.max(1e-8).ln());
    // δ's columns are the collapse's batches, as the units registered them
    let names = pu
        .units
        .batch_names()
        .ok_or_else(|| anyhow::anyhow!("the pair units carry no batch names"))?;
    anyhow::ensure!(
        pair_unit_log_delta.ncols() == names.len(),
        "pair δ has {} batches for {} names",
        pair_unit_log_delta.ncols(),
        names.len()
    );
    let index: HashMap<&str, usize> = names
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_ref(), i))
        .collect();
    let batch_of_cell: Vec<usize> = batch_label_of_cell
        .iter()
        .map(|b| {
            index
                .get(b.as_ref())
                .copied()
                .ok_or_else(|| anyhow::anyhow!("cell batch {b} not among the pair units' batches"))
        })
        .collect::<anyhow::Result<_>>()?;
    let groups = CollapsePseudobulks {
        finest: collapsed
            .cell_to_pb_per_level
            .first()
            .ok_or_else(|| anyhow::anyhow!("the pair-unit collapse gave no level"))?,
        coarsest: collapsed
            .cell_to_pb_per_level
            .last()
            .ok_or_else(|| anyhow::anyhow!("the pair-unit collapse gave no level"))?,
    };
    // estimated over the kept (within-batch) pairs; re-indexed to the given pairs
    let given = |kept: &[Option<usize>]| -> Vec<Option<usize>> {
        pu.unit_of_pair
            .iter()
            .map(|e| e.and_then(|e| kept[e]))
            .collect()
    };
    let (kept_fold, pair_node_of_endpoint_nodes, link_states) = match pair_states {
        PairStates::EndpointStates => {
            let (fold, keys) = estimate_fold_over_pair_units(
                &pu,
                &groups,
                &batch_of_cell,
                sample_fold,
                &pair_unit_log_delta,
                params,
            )?;
            (fold, keys, None)
        }
        PairStates::LinkStates => {
            let (fold, decomposition) = estimate_fold_over_pair_units_by_link_states(
                &pu,
                &groups,
                &batch_of_cell,
                sample_fold,
                &pair_unit_log_delta,
                params,
            )?;
            let link_states = LinkStateDecomposition {
                link_state_of_pair: given(&decomposition.link_state_of_pair),
                ..decomposition
            };
            (fold, PairNodeOfEndpointNodes::default(), Some(link_states))
        }
    };
    Ok(CellPairBatchFold {
        fold: StateAwareBatchFold {
            node_of_unit: given(&kept_fold.node_of_unit),
            ..kept_fold
        },
        pair_states,
        pair_node_of_endpoint_nodes,
        link_states,
        pair_unit_log_delta,
    })
}

/// The kept pairs' units read once into pb pairs, keyed (finest pair group,
/// batch, `slot_of_pair[k]`): a pair whose slot is `None` is left out, and so
/// is every self-loop. Each pb pair carries its pairs' span label
/// (`label_of_pair`), coarsest pair group and mean matching coordinates (a
/// pair's: the mean of its endpoints' `cell_coords`, components × cells).
pub(crate) fn sum_cell_pairs_into_pseudobulks(
    pu: &CellPairUnits,
    groups: &CollapsePseudobulks<'_>,
    batch_of_cell: &[usize],
    label_of_pair: &[usize],
    cell_coords: &DMatrix<f32>,
    slot_of_pair: &[Option<usize>],
) -> anyhow::Result<PseudobulkSums> {
    let pairs = &pu.kept_pairs;
    let e = pairs.len();
    let n_units = pu.units.num_columns();
    anyhow::ensure!(
        label_of_pair.len() == e && slot_of_pair.len() == e,
        "pb pairs: one label and slot per kept pair"
    );
    let mut coords = DMatrix::<f32>::zeros(cell_coords.nrows(), n_units);
    for (k, &(u, v)) in pairs.iter().enumerate() {
        coords
            .column_mut(k)
            .copy_from(&((cell_coords.column(u) + cell_coords.column(v)) * 0.5));
    }
    let batch: Vec<usize> = (0..n_units)
        .map(|k| if k < e { batch_of_cell[pairs[k].0] } else { 0 })
        .collect();
    let span: Vec<usize> = (0..n_units)
        .map(|k| if k < e { label_of_pair[k] } else { 0 })
        .collect();
    let key: Vec<Option<(usize, usize, usize)>> = (0..n_units)
        .map(|k| {
            let slot = *slot_of_pair.get(k)?;
            slot.map(|s| (groups.finest[k], batch[k], s))
        })
        .collect();
    sum_by_pseudobulk(
        &pu.units,
        &PseudobulkKeys {
            key: &key,
            batch: &batch,
            span_label: &span,
            coarsest: groups.coarsest,
            coords: &coords,
        },
    )
}

/// The fold over pair units already built and collapsed: each kept pair
/// placed on the samples kind's tree by its endpoints, its pb pairs summed
/// once, and the fold matched pb pair to pb pair per pair node.
pub(crate) fn estimate_fold_over_pair_units(
    pu: &CellPairUnits,
    groups: &CollapsePseudobulks<'_>,
    batch_of_cell: &[usize],
    cells: &SampleBatchFold,
    log_delta: &DMatrix<f32>,
    params: &StateAwareBatchFoldParams,
) -> anyhow::Result<(StateAwareBatchFold, PairNodeOfEndpointNodes)> {
    let nb = log_delta.ncols();
    let pairs = &pu.kept_pairs;
    let e = pairs.len();
    let (label_of_cell, n_labels) = params.span_labels_of_samples(batch_of_cell, nb)?;
    let rule = BatchSpanningRule::new(params, n_labels);
    let label_of_pair: Vec<usize> = pairs.iter().map(|&(u, _)| label_of_cell[u]).collect();
    let count_key: Vec<(usize, usize)> = (0..e)
        .map(|p| (groups.coarsest[p], label_of_pair[p]))
        .collect();
    let at: Vec<Option<(usize, usize)>> = pairs
        .iter()
        .map(|&(u, v)| Some((cells.tree_node_of_unit[u]?, cells.tree_node_of_unit[v]?)))
        .collect();
    let (pair_node, pair_node_of_endpoint_nodes) =
        place_cell_pairs_on_sample_tree(&cells.tree, &at, &count_key, &rule);
    let n_nodes = pair_node_of_endpoint_nodes.len();
    let kept = pair_node.iter().filter(|n| n.is_some()).count();
    if n_nodes == 0 {
        warn!("cell-pair batch fold: no pair node spans enough batches; every pair keeps δ");
    } else {
        info!(
            "cell-pair batch fold: {n_nodes} pair nodes; {} of {e} pairs keep δ",
            e - kept
        );
    }

    // the pair units read once: pb pairs (finest pair group in one batch and pair node)
    let sums = sum_cell_pairs_into_pseudobulks(
        pu,
        groups,
        batch_of_cell,
        &label_of_pair,
        &cells.coords,
        &pair_node,
    )?;
    let mut node_of_pb = vec![None; sums.len()];
    for (k, pb) in sums.of_unit.iter().enumerate() {
        if let Some(pb) = *pb {
            node_of_pb[pb] = pair_node[k];
        }
    }
    let log_fold = estimate_fold_on_pseudobulks(
        &sums,
        &node_of_pb,
        n_nodes,
        &delta_as_base_of_every_node(log_delta, n_nodes),
        params,
    )?;
    Ok((
        StateAwareBatchFold {
            n_batches: nb,
            n_nodes,
            log_fold,
            log_delta: log_delta.clone(),
            node_of_unit: pair_node,
        },
        pair_node_of_endpoint_nodes,
    ))
}
