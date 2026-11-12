//! Retrieval-based imputation: pool a sparse reference's full-feature
//! counts onto query cells matched in a shared latent space.
//!
//! The caller owns the latent semantics — both matrices must already be
//! mapped onto whatever space makes their rows comparable under L2
//! (softmaxed topic proportions, L2-normalized embeddings, …). This
//! module only does the model-agnostic part:
//!
//! 1. Build an approximate kNN index over the reference rows.
//! 2. For each query row, find the K nearest reference rows and convert
//!    their L2 distances to softmax weights with temperature τ.
//! 3. Stream the reference backend's columns chunk by chunk and
//!    accumulate each query row's weighted average of its neighbours'
//!    counts into a dense `[N_query, G_ref]` matrix.
//!
//! An all-zero row means "no evidence" on BOTH sides: a zero query row
//! is skipped (its imputed row stays zero), and a zero reference row is
//! excluded from the index — left in, it would sit at one fixed
//! coordinate capturing every otherwise-distant query.
//!
//! The information-theoretic point of retrieval over a decoder readout:
//! the imputed expression inherits the reference's full-rank gene–gene
//! covariance through the retrieved cells, rather than being confined to
//! the rank of the model's dictionary.

use crate::sparse_io_vector::SparseIoVec;
use legume_numeric::matrix::knn_match::ColumnDict;
use log::{info, warn};
use rayon::prelude::*;

type Mat = nalgebra::DMatrix<f32>;

pub struct RetrievalImputeConfig {
    /// Reference nearest neighbours pooled per query row.
    pub knn: usize,
    /// Softmax temperature on kNN distances (lower = sharper weights).
    pub temperature: f32,
    /// Reference columns per streamed read.
    pub chunk: usize,
}

/// A searched query row's matched reference rows and their pooling weights.
struct QueryNeighbours {
    ids: Vec<u32>,
    weights: Vec<f32>,
}

/// How many query rows were not searched for having a zero latent (`None`),
/// and how many searched ones got fewer than `knn` neighbours, none at all
/// included.
fn count_zero_latent_and_short(
    neighbours: &[Option<QueryNeighbours>],
    knn: usize,
) -> (usize, usize) {
    let n_zero_latent = neighbours.iter().filter(|n| n.is_none()).count();
    let n_short = neighbours
        .iter()
        .flatten()
        .filter(|n| n.ids.len() < knn)
        .count();
    (n_zero_latent, n_short)
}

/// Weighted-kNN imputation of `ref_data`'s features onto the query rows.
///
/// `query_latent` is `[N_query, K]` and `ref_latent` is `[N_ref, K]`, in
/// the same (caller-established) matching space; `ref_data` holds the
/// reference counts with one column per `ref_latent` row. Returns the
/// dense imputed `[N_query, G_ref]` matrix.
pub fn retrieval_impute(
    query_latent: &Mat,
    ref_latent: &Mat,
    ref_data: &SparseIoVec,
    cfg: &RetrievalImputeConfig,
) -> anyhow::Result<Mat> {
    let (n_query, k_query) = (query_latent.nrows(), query_latent.ncols());
    let (n_ref, k_ref) = (ref_latent.nrows(), ref_latent.ncols());
    anyhow::ensure!(
        k_query == k_ref,
        "latent dimension mismatch: query K={k_query} vs reference K={k_ref}"
    );
    anyhow::ensure!(cfg.knn >= 1, "knn must be at least 1");
    anyhow::ensure!(n_ref > 0, "reference latent has no rows");
    anyhow::ensure!(
        ref_data.num_columns() == n_ref,
        "reference data has {} cells but reference latent has {n_ref}; \
         the data files don't match the latent",
        ref_data.num_columns(),
    );
    let g_ref = ref_data.num_rows();
    anyhow::ensure!(g_ref > 0, "reference data has no features");

    // Index only the reference rows that carry evidence.
    let live: Vec<usize> = (0..n_ref)
        .filter(|&i| ref_latent.row(i).iter().any(|&x| x != 0.0))
        .collect();
    anyhow::ensure!(!live.is_empty(), "every reference latent row is zero");
    if live.len() < n_ref {
        warn!(
            "{} of {n_ref} reference rows have a zero latent and are excluded \
             from the index",
            n_ref - live.len()
        );
    }
    info!(
        "Building kNN index over {} reference rows (k={}, τ={})",
        live.len(),
        cfg.knn,
        cfg.temperature
    );
    // One allocation for the [K, n_live] point set the dict copies from;
    // `transpose().select_columns(&live)` would build the same thing through
    // an intermediate full-matrix copy.
    let ref_dict = ColumnDict::<u32>::from_dmatrix(
        Mat::from_fn(k_ref, live.len(), |r, c| ref_latent[(live[c], r)]),
        live.iter().map(|&i| i as u32).collect(),
    );

    let knn = cfg.knn.min(live.len());
    // `None` marks a zero-latent query row, which is not searched.
    let neighbours: Vec<Option<QueryNeighbours>> = (0..n_query)
        .into_par_iter()
        // One search scratch per rayon worker: re-growing the backend's
        // visited set is O(n_ref) per query otherwise.
        .map_init(
            legume_numeric::matrix::knn_match::SearchScratch::default,
            |scratch, i| {
                let query: Vec<f32> = query_latent.row(i).iter().copied().collect();
                if query.iter().all(|&x| x == 0.0) {
                    return Ok(None);
                }
                let (ids, distances) = ref_dict
                    .search_by_query_data_reuse(&query, knn, scratch)
                    .map_err(|e| anyhow::anyhow!("kNN search for query row {i}: {e}"))?;
                let weights = dist_to_softmax_weights(&distances, cfg.temperature);
                Ok(Some(QueryNeighbours { ids, weights }))
            },
        )
        .collect::<anyhow::Result<_>>()?;
    let (n_zero_latent, n_short) = count_zero_latent_and_short(&neighbours, knn);
    if n_zero_latent > 0 {
        warn!(
            "{n_zero_latent} of {n_query} query rows have a zero latent and were skipped; \
             their imputed rows stay zero"
        );
    }
    // The index may return fewer neighbours than asked for (an approximate
    // search has a fixed candidate budget); the weights are then spread over
    // the ones it did return, and a row it returned none for stays zero.
    if n_short > 0 {
        warn!(
            "{n_short} of {n_query} query rows got fewer than the {knn} neighbours \
             asked for; they pool the neighbours the index returned, and stay zero \
             where it returned none"
        );
    }

    // Heads-up before the big allocation: dense [N_query, G_ref] f32 grows
    // fast. Streaming-sparse output is a future optimization; for now we
    // materialize and let the caller write whole.
    let bytes_est = (n_query * g_ref).saturating_mul(4);
    if bytes_est > (1 << 30) {
        warn!(
            "imputed dense matrix will allocate ~{} MB ({n_query} × {g_ref} f32). \
             Consider reducing the query size or the reference feature set if \
             memory is tight.",
            bytes_est >> 20,
        );
    }

    // Invert the (query row → neighbours) map so column-streaming the
    // reference can do `consumers[ref_id]` lookups in O(1) and skip whole
    // chunks where no consumer touches any of the chunk's cells.
    let mut cell_to_consumers: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_ref];
    for (query_id, nbr) in neighbours.iter().enumerate() {
        let Some(nbr) = nbr else { continue };
        for (&c, &w) in nbr.ids.iter().zip(nbr.weights.iter()) {
            cell_to_consumers[c as usize].push((query_id as u32, w));
        }
    }

    // Accumulate TRANSPOSED, [G_ref, N_query]: each query row is then one
    // contiguous column, so a reference column's ascending row indices write
    // ascending contiguous addresses, and disjoint query rows let the
    // per-chunk accumulation run in parallel with no locking. One transpose
    // at the end pays a single O(N×G) pass for a cache-friendly hot loop.
    let mut imputed_t = Mat::zeros(g_ref, n_query);
    // Per-chunk (reference-local column, weight) work lists, indexed by
    // query; allocated once, cleared inside the parallel pass below.
    let mut groups: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_query];
    let chunk_size = cfg.chunk.max(64);
    let mut col_lb = 0;
    while col_lb < n_ref {
        let col_ub = (col_lb + chunk_size).min(n_ref);
        // Skip the read entirely if no consumer touches this chunk —
        // common when N_query × knn ≪ N_ref.
        if cell_to_consumers[col_lb..col_ub]
            .iter()
            .all(std::vec::Vec::is_empty)
        {
            col_lb = col_ub;
            continue;
        }
        let csc = ref_data.read_columns_csc(col_lb..col_ub)?;

        for c_local in 0..(col_ub - col_lb) {
            for &(query_id, w) in &cell_to_consumers[col_lb + c_local] {
                groups[query_id as usize].push((c_local as u32, w));
            }
        }

        imputed_t
            .as_mut_slice()
            .par_chunks_mut(g_ref)
            .zip(groups.par_iter_mut())
            .for_each(|(query_col, grp)| {
                for &(c_local, w) in grp.iter() {
                    let col = csc.col(c_local as usize);
                    for (&row_id, &v) in col.row_indices().iter().zip(col.values().iter()) {
                        query_col[row_id] += w * v;
                    }
                }
                // Cleared here (in parallel, capacity kept) so the next
                // chunk's regroup starts from empty lists.
                grp.clear();
            });

        col_lb = col_ub;
    }

    Ok(imputed_t.transpose())
}

/// Convert kNN L2 distances to a positive-weight simplex via
/// `w_k ∝ exp(-d_k² / (2 τ²))`, normalised to sum to 1.
fn dist_to_softmax_weights(distances: &[f32], temperature: f32) -> Vec<f32> {
    if distances.is_empty() {
        return Vec::new();
    }
    let tau = temperature.max(1e-6);
    let scale = 1.0 / (2.0 * tau * tau);
    let mut max = f32::NEG_INFINITY;
    let mut out: Vec<f32> = distances
        .iter()
        .map(|d| {
            let v = -d * d * scale;
            if v > max {
                max = v;
            }
            v
        })
        .collect();
    let mut sum = 0.0f32;
    for x in &mut out {
        *x = (*x - max).exp();
        sum += *x;
    }
    let inv = 1.0 / sum.max(1e-12);
    for x in &mut out {
        *x *= inv;
    }
    out
}

#[cfg(test)]
#[path = "retrieval_impute_tests.rs"]
mod tests;
