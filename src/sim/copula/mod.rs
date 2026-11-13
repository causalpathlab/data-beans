//! Reference-conditioned copula structure used by `run_simulate` when a
//! `--reference` is supplied. The simulator's GLM (`β·θ·δ` + housekeeping)
//! is unchanged — the reference contributes:
//!
//! 1. each gene's count distribution over all reference cells, a Poisson
//!    mixture fitted by nonparametric maximum likelihood (see [`mixture`]),
//!    so every gene's mean, variance and zero fraction are the reference's;
//! 2. `ρ_gs`, each active gene's latent correlation with the cell's
//!    library-size coordinate `z_s` (the normal score of its library-size
//!    rank), so cells differ in depth the way the reference's do, gene by
//!    gene;
//! 3. `Σ̂`, the HVGs' latent correlation beyond depth, as a low-rank factor
//!    `F` plus a per-gene residual (see [`gaussian`]). Latent correlations
//!    come from [`latent`], which corrects for the discreteness of sparse
//!    counts.
//!
//! Sampling, per synthetic cell `j` with GLM mean `λ_{g,j}`:
//!
//! ```text
//! z_s ~ N(0, 1);  e ~ F·η + diag(ridge)·ε on the HVGs, iid N(0, 1) elsewhere
//! z*_g = ρ_gs · z_s + √(1 − ρ_gs²) · e_g
//! y_{g,j} = F⁻¹_g(Φ(z*_g);  rates × λ_{g,j} / μ̂_g)
//! ```
//!
//! with `F_g` the gene's mixture, tempered to leave room for the topic and
//! noise terms (see [`sampler`]).
//!
//! References:
//! - Li & Li 2019. *A statistical simulator scDesign for rational scRNA-seq
//!   experimental design.* Bioinformatics 35(14):i41–i50.
//! - Sun, Song, Li & Li 2021. *scDesign2.* Genome Biology 22:163.
//! - Song et al. 2024. *scDesign3.* Nature Biotechnology 42(2):247–252.

pub mod fidelity;
pub mod gaussian;
pub mod latent;
pub mod marginals;
pub mod mixture;
pub mod reference;
pub mod sampler;

use gaussian::CopulaCovariance;
use latent::ReferenceLatent;
use log::info;
use mixture::PoissonMixture;
use rayon::prelude::*;
use reference::SparseRef;

/// Genes with `μ̂_g` below this threshold are excluded from the per-cell
/// sampling loop entirely (they can't produce a nonzero count even with the
/// maximum log-mean perturbation `≈ exp(3.5)` from topic+batch+noise stages).
/// Saves the cost of CDF construction + inverse-CDF lookup for genes that
/// would always yield 0.
const MU_HAT_ACTIVE_THRESHOLD: f32 = 1e-5;

/// Output of [`fit_global_copula`]. Owns the fitted reference dispersions,
/// the per-gene reference mean `μ̂_g`, and the low-rank Gaussian copula
/// factor — everything needed to drive the two-stage GLM + NB+copula sampler.
pub struct GlobalCopulaFit {
    pub gene_names: Vec<Box<str>>,
    pub n_genes: usize,
    /// Indices into `gene_names` chosen as HVGs for the copula structure.
    pub hvg_indices: Vec<usize>,
    /// Per-gene reference mean `μ̂_g`, length `n_genes`. Used as the
    /// stage-1 baseline `log μ̂_g` in the two-stage simulator.
    pub mu_hat: Vec<f32>,
    /// Per-gene count distribution in the reference, a Poisson mixture
    /// (see [`mixture`]), length `n_genes`; all mass at 0 for inactive genes.
    pub marginals: Vec<PoissonMixture>,
    /// Per-gene latent correlation `ρ_gs` with the library-size coordinate,
    /// length `n_genes`; 0 for inactive genes.
    pub depth_rho: Vec<f32>,
    /// Subset of gene indices `g` with `μ̂_g >= MU_HAT_ACTIVE_THRESHOLD`.
    /// The per-cell sampling loop iterates only these — undetectable genes
    /// (always-zero) skip CDF construction.
    pub active_genes: Vec<usize>,
    /// `hvg_pos[g] = Some(h)` iff gene `g` is the `h`-th HVG; `None` for
    /// non-HVG genes. Length = `n_genes`.
    pub hvg_pos: Vec<Option<u32>>,
    pub copula: CopulaCovariance,
    /// The reference's side of the fidelity report, over `active_genes`.
    pub profile: fidelity::Profile,
    /// Indices into `active_genes` of the genes whose correlations the
    /// fidelity report compares.
    pub correlation_genes: Vec<usize>,
}

impl GlobalCopulaFit {
    /// Each gene's NB size with its marginal's extra-Poisson variance,
    /// `1 / cv²` of the mixture's rates (`∞` without any): what the
    /// `.r.parquet` output reports.
    pub fn effective_nb_size(&self) -> Vec<f32> {
        self.marginals
            .iter()
            .map(|m| {
                let cv2 = m.rate_cv2();
                if cv2 > 0.0 {
                    (1.0 / cv2) as f32
                } else {
                    f32::INFINITY
                }
            })
            .collect()
    }

    /// Write the fidelity report of `triplets`, a simulation of `n_cells`
    /// cells from this fit, to [`fidelity::report_path`] of `prefix`; that
    /// path.
    pub fn write_fidelity_report(
        &self,
        triplets: &[(u64, u64, f32)],
        n_cells: usize,
        reference: &str,
        prefix: &str,
    ) -> anyhow::Result<String> {
        let simulated = fidelity::profile(
            triplets,
            self.n_genes,
            n_cells,
            &self.active_genes,
            &self.correlation_genes,
        );
        let report = fidelity::FidelityReport {
            reference: reference.to_string(),
            output: prefix.to_string(),
            gene_names: self
                .active_genes
                .iter()
                .map(|&g| self.gene_names[g].to_string())
                .collect(),
            correlation_genes: self.correlation_genes.clone(),
            profiles: fidelity::Paired {
                reference: self.profile.clone(),
                simulated,
            },
        };
        let path = fidelity::report_path(prefix);
        report.write(&path)?;
        Ok(path)
    }
}

pub struct GlobalCopulaArgs<'a> {
    pub sc: &'a SparseRef,
    /// HVGs included in the copula dependence structure. Genes outside the
    /// HVG set are sampled independently from their NB fits at run time.
    pub n_hvg: usize,
    /// Maximum rank of the low-rank Σ̂ factor. Effective rank is at most
    /// the number of HVGs.
    pub copula_rank: usize,
    /// Smallest share of each HVG's latent variance kept as its own,
    /// outside the shared factors.
    pub regularization: f32,
    /// Smallest share of reference cells in which an HVG is detected.
    pub hvg_min_detection: f32,
}

/// Fit a single global copula on the entire reference: Poisson-mixture marginals,
/// the genes' coupling to depth, and a low-rank factor of the HVGs' latent
/// correlation beyond depth.
pub fn fit_global_copula(args: &GlobalCopulaArgs) -> anyhow::Result<GlobalCopulaFit> {
    let n_genes = args
        .sc
        .num_rows()
        .ok_or_else(|| anyhow::anyhow!("reference has no num_rows"))?;
    let n_cells = args
        .sc
        .num_columns()
        .ok_or_else(|| anyhow::anyhow!("reference has no num_columns"))?;
    if n_cells < 2 {
        anyhow::bail!(
            "reference has only {} cells; need ≥2 to fit a copula",
            n_cells
        );
    }
    let gene_names = args.sc.row_names()?;
    info!("reference: {} genes × {} cells", n_genes, n_cells);

    let cells: Vec<usize> = (0..n_cells).collect();
    let (stats, libs, mat) = reference::per_gene_stats(args.sc, &cells, n_genes)?;
    let mu_hat: Vec<f32> = stats.iter().map(|s| s.mu as f32).collect();
    let active_genes: Vec<usize> = mu_hat
        .iter()
        .enumerate()
        .filter_map(|(g, &m)| (m >= MU_HAT_ACTIVE_THRESHOLD).then_some(g))
        .collect();
    // Copula genes are sampled, so they are active: a gene detected in many
    // cells can still have a tiny mean in fractional (normalised) data.
    let mut hvg_indices =
        reference::select_hvg(&stats, n_cells, args.n_hvg, args.hvg_min_detection);
    hvg_indices.retain(|&g| mu_hat[g] >= MU_HAT_ACTIVE_THRESHOLD);
    anyhow::ensure!(
        !hvg_indices.is_empty(),
        "no gene is detected in {:.1}% (and 50) of the {} reference cells; lower --hvg-min-detection",
        100.0 * args.hvg_min_detection,
        n_cells
    );
    info!(
        "active genes (μ̂ ≥ {:.0e}): {}/{} ({:.1}% — undetectable genes skipped from sampling)",
        MU_HAT_ACTIVE_THRESHOLD,
        active_genes.len(),
        n_genes,
        100.0 * active_genes.len() as f32 / n_genes as f32
    );

    let hist = latent::gene_histograms(&mat, n_cells, &active_genes);
    let marginals: Vec<PoissonMixture> = hist
        .par_iter()
        .map(|h| {
            h.as_deref()
                .map_or_else(PoissonMixture::zero, PoissonMixture::fit)
        })
        .collect();
    let latent = ReferenceLatent::fit(&mat, &libs, &hist, &hvg_indices);
    drop(hist);
    let zero: Vec<f32> = active_genes
        .iter()
        .map(|&g| 1.0 - stats[g].nnz as f32 / n_cells as f32)
        .collect();
    let correlation_genes = fidelity::correlation_genes(&active_genes, &hvg_indices, &zero);
    let profile = fidelity::profile(&mat, n_genes, n_cells, &active_genes, &correlation_genes);
    drop(mat);
    let copula = CopulaCovariance::from_correlation(
        &latent.hvg_residual,
        args.copula_rank,
        args.regularization,
    )?;
    let mean_abs_depth = active_genes
        .iter()
        .map(|&g| latent.depth_rho[g].abs())
        .sum::<f32>()
        / active_genes.len().max(1) as f32;
    info!(
        "fit global copula: {} HVGs (detected in ≥ {:.1}% of cells), rank {} (cap {}), \
         mean residual sd {:.3}; mean |ρ| with depth {:.3}",
        hvg_indices.len(),
        100.0 * args.hvg_min_detection,
        copula.rank(),
        args.copula_rank,
        copula.ridge_sd.mean(),
        mean_abs_depth
    );
    let mut hvg_pos = vec![None; n_genes];
    for (h, &g) in hvg_indices.iter().enumerate() {
        hvg_pos[g] = Some(h as u32);
    }
    Ok(GlobalCopulaFit {
        gene_names,
        n_genes,
        hvg_indices,
        mu_hat,
        marginals,
        depth_rho: latent.depth_rho,
        active_genes,
        hvg_pos,
        copula,
        profile,
        correlation_genes,
    })
}
