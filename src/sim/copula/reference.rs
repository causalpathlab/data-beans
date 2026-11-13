//! Reference loading, per-gene stats, HVG selection, and the library-size
//! coordinate for the global copula fit used by `run_simulate --reference`.

use crate::sparse_io::*;
use legume_numeric::matrix::common_io::file_ext;
use legume_numeric::matrix::sparse_stat::SparseRunningStatistics;
use nalgebra_sparse::CscMatrix;

/// Boxed dyn handle to whatever sparse backend the reference lives in.
pub type SparseRef = Box<dyn SparseIo<IndexIter = Vec<usize>>>;

/// Open a `.h5`, `.zarr`, or `.zarr.zip` reference. The zip case is handled
/// natively by `crate::zarr_io` — no separate unzip step.
pub fn open_reference(path: &str) -> anyhow::Result<SparseRef> {
    let ext = file_ext(path)?.to_string();
    let backend = match ext.as_str() {
        "h5" => SparseIoBackend::HDF5,
        // `.zarr` directories report ext == "zarr"; `.zarr.zip` archives
        // report ext == "zip" — both go through the Zarr backend, which
        // detects and handles zip stores in `crate::zarr_io`.
        "zarr" | "zip" => SparseIoBackend::Zarr,
        other => anyhow::bail!(
            "unsupported reference extension '{}': expected h5 / zarr / zarr.zip",
            other
        ),
    };
    open_sparse_matrix(path, &backend)
}

/// Per-gene moment summary over a cell subset.
#[derive(Debug, Clone, Copy)]
pub struct GeneStats {
    pub mu: f64,
    pub var: f64,
    pub nnz: usize,
}

/// Per-gene mean / variance / nnz over a cell subset via
/// `legume_numeric::matrix::sparse_stat::SparseRunningStatistics` (one CSC
/// pass), with each cell's library size. The CSC itself is returned too:
/// the latent fit reads the same counts again.
///
/// The reference is read as counts: every value is rounded to the nearest
/// whole number (negatives to 0) first, so the statistics, the histograms
/// and the marginals all see one matrix.
pub fn per_gene_stats(
    sc: &SparseRef,
    cells: &[usize],
    n_genes: usize,
) -> anyhow::Result<(Vec<GeneStats>, Vec<f64>, CscMatrix<f32>)> {
    let mut mat = sc.read_columns_csc(cells.to_vec())?;
    for v in mat.values_mut() {
        *v = v.round().max(0.0);
    }
    let mut acc = SparseRunningStatistics::<f64>::new(n_genes);
    let mut libs = Vec::with_capacity(cells.len());
    for col in mat.col_iter() {
        let vals_f64: Vec<f64> = col.values().iter().map(|&v| v as f64).collect();
        libs.push(vals_f64.iter().sum());
        acc.add_sparse_column(col.row_indices(), &vals_f64);
    }
    // Pad implicit-zero columns so the mean/variance denominators reflect
    // every requested cell, not just CSC columns that contain at least one nz.
    for _ in mat.ncols()..cells.len() {
        acc.add_sparse_column(&[], &[]);
    }
    libs.resize(cells.len(), 0.0);
    let (npos, _sum, mean, std) = acc.to_vecs();
    let stats = (0..n_genes)
        .map(|g| GeneStats {
            mu: mean[g],
            var: std[g] * std[g],
            nnz: npos[g] as usize,
        })
        .collect();
    Ok((stats, libs, mat))
}

/// Library-size coordinate of each cell: `Φ⁻¹` of its mid-rank, ties
/// averaged, so it is standard normal across cells by construction.
pub fn library_coordinate(libs: &[f64]) -> Vec<f64> {
    let n = libs.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| libs[i].total_cmp(&libs[j]));
    let mut z = vec![0.0_f64; n];
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && libs[order[end]] == libs[order[start]] {
            end += 1;
        }
        // Mid-rank of the tie block, on (0, 1).
        let u = (start + end) as f64 / (2.0 * n as f64);
        let zu = crate::sim::copula::marginals::inv_phi(u);
        for &c in &order[start..end] {
            z[c] = zu;
        }
        start = end;
    }
    z
}

/// Select up to `n_hvg` highly variable genes via the workspace's NB
/// dispersion-trend HVG ranker (`crate::alg::hvg`). It fits
/// `σ²(μ) = μ + φ(μ)·μ²` and ranks each gene by its excess dispersion above
/// the trend — the same `φ(μ)` fit that the rest of the workspace uses for
/// Fisher-info gene weighting and DC-Poisson clustering, so the copula sim
/// is consistent with how HVGs are picked in cocoa / pinto / senna.
///
/// Only genes detected (non-zero) in at least `min_detection` of the
/// `n_cells` reference cells are eligible: the excess-dispersion score
/// otherwise favours genes seen in a handful of cells, whose correlations
/// the reference cannot estimate.
pub fn select_hvg(
    stats: &[GeneStats],
    n_cells: usize,
    n_hvg: usize,
    min_detection: f32,
) -> Vec<usize> {
    let floor = ((min_detection.max(0.0) as f64 * n_cells as f64).ceil() as usize)
        .max(HVG_MIN_CELLS.min(n_cells));
    let means: Vec<f32> = stats.iter().map(|s| s.mu as f32).collect();
    let vars: Vec<f32> = stats.iter().map(|s| s.var as f32).collect();
    let trend = crate::alg::nb_dispersion::DispersionTrend::fit(&means, &vars);
    let mut ranked: Vec<(usize, f32)> = (0..stats.len())
        .filter(|&g| stats[g].nnz >= floor.max(1))
        .map(|g| (g, log_excess_dispersion(&trend, means[g], vars[g])))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut hvg: Vec<usize> = ranked.into_iter().take(n_hvg).map(|(g, _)| g).collect();
    hvg.sort_unstable();
    hvg
}

/// Fewest reference cells an HVG may be detected in, whatever the share.
pub const HVG_MIN_CELLS: usize = 50;

/// `ln(φ̂ / φ(μ))`: a gene's dispersion relative to the trend at its mean.
/// The difference `φ̂ − φ(μ)` would rank genes seen in a handful of cells
/// first, whose `φ̂` is huge only because their mean is tiny.
fn log_excess_dispersion(
    trend: &crate::alg::nb_dispersion::DispersionTrend,
    mu: f32,
    var: f32,
) -> f32 {
    if mu.is_nan() || mu <= 0.0 || !var.is_finite() {
        return f32::NEG_INFINITY;
    }
    let phi_hat = (var - mu) / (mu * mu);
    let phi_trend = trend.phi_at(mu);
    if phi_hat <= 0.0 || phi_trend.is_nan() || phi_trend <= 0.0 {
        return f32::NEG_INFINITY;
    }
    (phi_hat / phi_trend).ln()
}
