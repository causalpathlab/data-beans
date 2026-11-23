//! Everything upstream of a graph embedding's training: the batch-corrected
//! projection, the multilevel collapse it feeds, and the per-level
//! pseudobulk rates the embedding's phase 1 is built from.
//!
//! One module because it is one dependency chain — the projection exists only
//! to hash cells into the collapse, and the collapse exists only to produce
//! these levels. None of it touches a model.

use super::UnifiedData;
use crate::alg::collapse_data::{
    collapse_columns_multilevel_with_hierarchy, collapse_columns_multilevel_with_partition,
    CollapsedOut, MultilevelCollapseOut, MultilevelParams,
};
use crate::alg::random_projection::{RandColProjOut, RandProjOps};
use crate::alg::refine_multilevel::RefineParams;
use crate::sparse_io_vector::SparseIoVec;
use legume_numeric::param::traits::Inference;
use log::info;
use nalgebra::DMatrix;
use std::borrow::Cow;

/// How the pseudobulks are built: the projection, then the collapse.
#[derive(Clone, Debug)]
pub struct EmbeddingPseudobulkArgs {
    /// Target rank of the random-projection sketch that seeds batch
    /// correction and the multilevel collapse.
    pub proj_dim: usize,
    /// Columns read per block by the projection (`None`: the backend's own).
    pub block_size: Option<usize>,
    /// Per-feature projection weights on the compact feature axis (HVG
    /// weighting); `None`: every feature weighs 1.
    pub hvg_weights: Option<Vec<f32>>,
    /// The features the sketch reads, on the compact feature axis: on a
    /// multi-track axis the base track's rows, since the other tracks are
    /// offsets from the base model, not independent measurements. `None`:
    /// every feature. The collapse itself always runs on every row.
    pub projection_rows: Option<Vec<bool>>,
    pub seed: u64,
    /// Multilevel-collapse levels (coarse → fine).
    pub num_levels: usize,
    /// Binary-tree partition depth at the finest level: at most
    /// `2^sort_dim + 1` pseudobulk leaves.
    pub sort_dim: usize,
    /// In-batch k-NN used when merging cells into pseudobulk samples.
    pub knn_pb_samples: usize,
    /// Coordinate-descent iterations of the per-batch δ inside the collapse.
    pub num_opt_iter: usize,
    pub refine: RefineParams,
    /// See [`MultilevelParams::anchor_batches`].
    pub anchor_batches: Option<Vec<Box<str>>>,
    /// See [`MultilevelParams::bulk_batches`].
    pub bulk_batches: Option<Vec<Box<str>>>,
    /// Per-cell CNV stratum (`0` = mixable); see [`MultilevelParams::strata`].
    pub strata: Option<Vec<usize>>,
    /// A given partition, finest first, one cell → pseudobulk map per level,
    /// each indexing the backend's columns: collapsed on as is, refinement
    /// skipped. `None`: the hash partition, refined.
    pub partition: Option<Vec<Vec<usize>>>,
}

/// The collapse, ordered **coarsest → finest**, with each cell's pseudobulk
/// per level. Read on the unified (compact) feature axis through its methods.
pub struct EmbeddingPseudobulks {
    pub collapsed_levels: Vec<CollapsedOut>,
    /// `cell_to_pb_per_level[l][c]` is cell `c`'s pseudobulk at level `l`.
    pub cell_to_pb_per_level: Vec<Vec<usize>>,
    /// The collapse's batch names, in the column order of its `δ`.
    pub collapse_batch_names: Vec<Box<str>>,
    /// The features the sketch read ([`EmbeddingPseudobulkArgs::projection_rows`]).
    pub sketched_rows: Option<Vec<bool>>,
    n_features: usize,
    feature_to_backend: Vec<usize>,
}

impl EmbeddingPseudobulks {
    /// Level `level`'s pseudobulk rates `[n_features × n_pb]` (batch-adjusted
    /// when the collapse fit a δ, observed otherwise), on the compact feature
    /// axis: borrowed when it is the backend's own axis, gathered otherwise.
    #[must_use]
    pub fn rates(&self, level: usize) -> Cow<'_, DMatrix<f32>> {
        let collapsed = &self.collapsed_levels[level];
        let full = match &collapsed.mu_adjusted {
            Some(adj) => adj.posterior_mean(),
            None => collapsed.mu_observed.posterior_mean(),
        };
        self.on_compact_axis(full)
    }

    /// The finest level's observed counts `[n_features × n_pb]` on the compact
    /// feature axis, with each pseudobulk's size.
    pub fn finest_counts(&self) -> anyhow::Result<(DMatrix<f32>, Vec<f32>)> {
        let (finest, cell_to_pb) = self.finest_collapse();
        let (counts, sizes) = finest.observed_counts(cell_to_pb)?;
        Ok((self.on_compact_axis(&counts).into_owned(), sizes))
    }

    /// The finest collapse's δ `[n_features × collapse batches]` on the compact
    /// feature axis, columns in [`Self::collapse_batch_names`] order; `None`
    /// when the collapse fit none.
    #[must_use]
    pub fn finest_delta(&self) -> Option<Cow<'_, DMatrix<f32>>> {
        let (finest, _) = self.finest_collapse();
        finest
            .delta
            .as_ref()
            .map(|d| self.on_compact_axis(d.posterior_mean()))
    }

    /// The finest collapse and each cell's pseudobulk in it.
    #[must_use]
    pub fn finest_collapse(&self) -> (&CollapsedOut, &[usize]) {
        (
            self.collapsed_levels.last().expect("at least one level"),
            self.cell_to_pb_per_level
                .last()
                .expect("membership per level"),
        )
    }

    fn on_compact_axis<'a>(&self, backend: &'a DMatrix<f32>) -> Cow<'a, DMatrix<f32>> {
        if backend.nrows() == self.n_features {
            Cow::Borrowed(backend)
        } else {
            Cow::Owned(gather_to_unified_axis(
                backend,
                self.n_features,
                &self.feature_to_backend,
            ))
        }
    }
}

/// Project, collapse, and keep the levels coarsest → finest.
pub fn build_embedding_pseudobulks(
    unified: &mut UnifiedData,
    args: &EmbeddingPseudobulkArgs,
) -> anyhow::Result<EmbeddingPseudobulks> {
    let batch_labels: Vec<Box<str>> = unified.batch_labels();
    let proj_out = project(unified, args, &batch_labels)?;

    let params = MultilevelParams {
        knn_pb_samples: args.knn_pb_samples,
        // A given partition carries its own level count.
        num_levels: args
            .partition
            .as_ref()
            .map_or(args.num_levels, Vec::len)
            .max(1),
        sort_dim: args.sort_dim,
        num_opt_iter: args.num_opt_iter,
        refine: args.refine.clone(),
        // Only `posterior_mean()` is ever read off this, so skip the sd / log_mean /
        // log_sd planes — that is the bulk of the coarsen-stage memory at high
        // pb-sample counts.
        output_calibration: legume_numeric::param::traits::CalibrateTarget::MeanOnly,
        anchor_batches: args.anchor_batches.clone(),
        bulk_batches: args.bulk_batches.clone(),
        observe_panels: true,
        // The feature partition reads the finest level's counts
        // (`CollapsedOut::observed_counts`), and the pseudobulk
        // reference serializes them; both need the sufficient statistics.
        keep_finest_stats: true,
        pb_tree: None,
        strata: args.strata.clone(),
    };
    let collapse_out = collapse(
        unified.count_backend_mut(),
        &proj_out.proj,
        &batch_labels,
        &params,
        args.partition.as_deref(),
    )?;
    let mut collapsed_levels = collapse_out.levels;
    let mut cell_to_pb_per_level = collapse_out.cell_to_pb_per_level;
    // The collapse emits finest-first. Reverse both so levels run coarsest..finest —
    // `senna topic` uses the same order, so its curriculum trains coarse first.
    collapsed_levels.reverse();
    cell_to_pb_per_level.reverse();
    let collapse_batch_names = match unified.count_backend().batch_names() {
        Some(names) => names,
        None => {
            anyhow::ensure!(
                collapsed_levels.iter().all(|c| c.delta.is_none()),
                "the collapse fit a δ but the backend has no batch names"
            );
            Vec::new()
        }
    };
    Ok(EmbeddingPseudobulks {
        collapsed_levels,
        cell_to_pb_per_level,
        collapse_batch_names,
        sketched_rows: args.projection_rows.clone(),
        n_features: unified.n_features(),
        feature_to_backend: unified.feature_to_backend_row.clone(),
    })
}

/// The multilevel collapse of `backend`'s columns: built from the hash partition and
/// refined, or — given `partition` (finest first, one cell → pseudobulk map per level,
/// each indexing `backend`'s columns) — collapsed on that partition as is, the
/// refinement skipped. `proj` still seeds the cross-batch matching either way.
fn collapse(
    backend: &mut SparseIoVec,
    proj: &DMatrix<f32>,
    batch_labels: &[Box<str>],
    params: &MultilevelParams,
    partition: Option<&[Vec<usize>]>,
) -> anyhow::Result<MultilevelCollapseOut> {
    let Some(partition) = partition else {
        info!(
            "Multilevel collapse (sort_dim={}, {} levels requested)...",
            params.sort_dim, params.num_levels
        );
        return collapse_columns_multilevel_with_hierarchy(backend, proj, batch_labels, params);
    };
    check_partition(partition, backend.num_columns())?;
    info!(
        "Multilevel collapse on a given partition ({} levels, {} cells; refinement skipped)...",
        partition.len(),
        backend.num_columns()
    );
    collapse_columns_multilevel_with_partition(backend, proj, batch_labels, params, partition)
        .map_err(|e| e.context("collapsing on the given partition"))
}

/// A given partition must hold at least the finest level, and every level must map each
/// of the backend's `n_cells` columns.
fn check_partition(partition: &[Vec<usize>], n_cells: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        !partition.is_empty(),
        "the given partition has no levels (it needs at least the finest)"
    );
    for (level, cell_to_pb) in partition.iter().enumerate() {
        anyhow::ensure!(
            cell_to_pb.len() == n_cells,
            "the given partition's level {level} (finest first) maps {} cells, but the count \
             backend has {n_cells} columns",
            cell_to_pb.len()
        );
    }
    Ok(())
}

/// The batch-corrected random projection the collapse hashes on, HVG-weighted when the
/// caller supplied weights, on `args.projection_rows` alone when given. The collapse
/// itself still runs on the FULL backend with this sketch, so refinement, `mu_adjusted`
/// and `δ` cover every row.
fn project(
    unified: &UnifiedData,
    args: &EmbeddingPseudobulkArgs,
    batch_labels: &[Box<str>],
) -> anyhow::Result<RandColProjOut> {
    info!(
        "Batch-corrected projection (proj_dim={}, {} batches)...",
        args.proj_dim,
        unified.n_batches()
    );
    let batch_arg = (unified.n_batches() > 1).then_some(batch_labels);
    let backend = unified.count_backend();
    // The projection runs on the full backend row axis, which may be wider than the
    // compact feature axis when a prior pass dropped features. Scatter the compact
    // weights to backend rows; rows not in the current feature axis get 0 so they sit
    // out the projection basis.
    let backend_w: Option<Vec<f32>> = match args.hvg_weights.as_deref() {
        None => None,
        Some(w) => {
            anyhow::ensure!(
                w.len() == unified.n_features(),
                "hvg_weights length {} != n_features {} (the HVG mask must be aligned to the \
                 unified feature axis BEFORE any subset/coarsening — pass full-axis weights from \
                 the wrapper)",
                w.len(),
                unified.n_features()
            );
            info!(
                "HVG-weighted projection: {} weighted features (>= 1.0)",
                w.iter().filter(|&&x| x > 0.0).count()
            );
            let mut backend_w = vec![0.0f32; backend.num_rows()];
            for (compact_i, &brow) in unified.feature_to_backend_row.iter().enumerate() {
                backend_w[brow] = w[compact_i];
            }
            Some(backend_w)
        }
    };

    let keep = match args.projection_rows.as_deref() {
        None => None,
        Some(rows) => Some(projection_row_mask(
            rows,
            &unified.feature_to_backend_row,
            backend.num_rows(),
        )?),
    };
    let Some(keep) = keep else {
        return project_backend(
            backend,
            args.proj_dim,
            args.block_size,
            batch_arg,
            backend_w.as_deref(),
            args.seed,
        );
    };
    info!(
        "Sketching on {} of {} backend rows",
        keep.iter().filter(|&&k| k).count(),
        keep.len()
    );
    let mut view = backend.clone_for_collapse();
    view.mask_rows(&keep)?;
    // `mask_rows` RENUMBERS the kept rows compactly, so the weight vector has to be
    // subset the same way — a full-axis vector would misalign every row.
    let view_w = backend_w.map(|w| subset_kept(&w, &keep));
    project_backend(
        &view,
        args.proj_dim,
        args.block_size,
        batch_arg,
        view_w.as_deref(),
        args.seed,
    )
}

/// The backend rows the sketch reads: those of the compact features `rows` marks.
fn projection_row_mask(
    rows: &[bool],
    feature_to_backend: &[usize],
    backend_rows: usize,
) -> anyhow::Result<Vec<bool>> {
    anyhow::ensure!(
        rows.len() == feature_to_backend.len(),
        "projection_rows has {} entries for {} features",
        rows.len(),
        feature_to_backend.len()
    );
    let mut keep = vec![false; backend_rows];
    for (&on, &brow) in rows.iter().zip(feature_to_backend) {
        if on {
            keep[brow] = true;
        }
    }
    Ok(keep)
}

/// `values` restricted to the `keep`ed positions, in the same order — the order
/// [`SparseIoVec::mask_rows`] renumbers them into.
fn subset_kept(values: &[f32], keep: &[bool]) -> Vec<f32> {
    values
        .iter()
        .zip(keep)
        .filter(|&(_, &k)| k)
        .map(|(&v, _)| v)
        .collect()
}

/// One random projection over `backend`, weighted when `row_weights` is given (length =
/// `backend.num_rows()`).
fn project_backend<T>(
    backend: &SparseIoVec,
    proj_dim: usize,
    block_size: Option<usize>,
    batch_arg: Option<&[T]>,
    row_weights: Option<&[f32]>,
    seed: u64,
) -> anyhow::Result<RandColProjOut>
where
    T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
{
    match row_weights {
        None => backend
            .project_columns_with_batch_correction_seeded(proj_dim, block_size, batch_arg, seed),
        Some(w) => {
            backend.project_columns_weighted_seeded(proj_dim, block_size, batch_arg, w, seed)
        }
    }
}

/// Gather a backend-row matrix onto the compact unified feature axis.
fn gather_to_unified_axis(
    backend: &DMatrix<f32>,
    n_features: usize,
    feature_to_backend: &[usize],
) -> DMatrix<f32> {
    let cols = backend.ncols();
    let mut out = DMatrix::<f32>::zeros(n_features, cols);
    for (new_i, &old_i) in feature_to_backend.iter().enumerate() {
        for s in 0..cols {
            out[(new_i, s)] = backend[(old_i, s)];
        }
    }
    out
}

#[cfg(test)]
#[path = "pseudobulks_tests.rs"]
mod tests;
