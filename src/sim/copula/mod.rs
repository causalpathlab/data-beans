//! Reference-conditioned copula structure used by `run_simulate` when a
//! `--reference` is supplied. The simulator's GLM (`β·θ·δ` + housekeeping)
//! is unchanged — the reference contributes:
//!
//! 0. cell groups ([`groups`]): the reference is fitted group by group
//!    (k-means on its HVGs' principal components, or given labels), and
//!    each simulated cell is drawn from one group in the reference's
//!    proportions, as in scDesign2, so a reference of cell types of very
//!    different size keeps its spread of library sizes and detected genes;
//! 1. each gene's count distribution within each group, a Poisson
//!    mixture fitted by nonparametric maximum likelihood (see [`mixture`]),
//!    so every gene's mean, variance and zero fraction are the reference's;
//! 2. `ρ_gs`, each active gene's latent correlation with the cell's
//!    library-size coordinate `z_s` (the normal score of its library-size
//!    rank within the group), so cells differ in depth the way the
//!    reference's do, gene by gene;
//! 3. `Σ̂`, the HVGs' latent correlation beyond depth, pooled within the
//!    groups, as a low-rank factor `F` plus a per-gene residual (see
//!    [`gaussian`]). Latent correlations come from [`latent`], which
//!    corrects for the discreteness of sparse counts.
//!
//! Sampling, per synthetic cell `j` of group `k` with GLM mean `λ_{g,j}`:
//!
//! ```text
//! z_s ~ N(0, 1);  e ~ F·η + diag(ridge)·ε on the HVGs, iid N(0, 1) elsewhere
//! z*_g = ρ_{gs,k} · z_s + √(1 − ρ_{gs,k}²) · e_g
//! y_{g,j} = F⁻¹_{g,k}(Φ(z*_g);  rates × λ_{g,j} / μ̂_g)
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
pub mod groups;
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

/// Output of [`fit_global_copula`]: the reference's Poisson-mixture
/// marginals, each gene's coupling to depth, and the low-rank latent copula
/// of the HVGs — everything the reference sampler ([`sampler`]) needs.
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
    /// The reference's cell groups, each with its own count distributions
    /// and depth coupling ([`groups`]); one group of all cells unless the
    /// fit is asked to split them.
    pub groups: Vec<GroupFit>,
    /// Genes detected in at least one reference cell. The per-cell sampling
    /// loop iterates only these; a gene never seen is never drawn.
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

/// One cell group of a reference fit.
#[derive(Debug, Clone)]
pub struct GroupFit {
    pub name: String,
    /// Share of the reference's cells in this group.
    pub share: f64,
    /// Per-gene count distribution within the group, length `n_genes`.
    pub marginals: Vec<PoissonMixture>,
    /// Per-gene latent correlation `ρ_gs` with the group's library-size
    /// coordinate, length `n_genes`; 0 for genes the group never shows.
    pub depth_rho: Vec<f32>,
}

impl GlobalCopulaFit {
    /// Each of `n` simulated cells' group, drawn in the reference's
    /// proportions with the seed `seed`.
    pub fn draw_groups(&self, n: usize, seed: u64) -> Vec<usize> {
        use rand::distr::Distribution;
        use rand::SeedableRng;
        if self.groups.len() <= 1 {
            return vec![0; n];
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let pick = rand::distr::weighted::WeightedIndex::new(self.groups.iter().map(|g| g.share))
            .expect("groups with positive shares");
        (0..n).map(|_| pick.sample(&mut rng)).collect()
    }

    /// Write each gene's NB size with its marginal's extra-Poisson
    /// variance, `1 / cv²` of the mixture's rates, as the one-column
    /// `r_hat` parquet `path`, rows named `names`. A gene with none
    /// (`r = ∞`) is written as 1e9, which parquet readers take.
    pub fn write_effective_nb_size(&self, path: &str, names: &[Box<str>]) -> anyhow::Result<()> {
        use legume_numeric::matrix::traits::IoOps;
        let r = nalgebra::DMatrix::<f32>::from_iterator(
            self.n_genes,
            1,
            self.marginals.iter().map(|m| {
                let cv2 = m.rate_cv2();
                if cv2 > 0.0 {
                    (1.0 / cv2) as f32
                } else {
                    1e9
                }
            }),
        );
        let label = ["r_hat".to_string().into_boxed_str()];
        r.to_parquet_with_names(path, (Some(names), Some("feature")), Some(&label))
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

/// The reference-fit settings `topic --reference` and `multiome
/// --reference-*` share, on the command line.
#[derive(clap::Args, Debug, Clone)]
pub struct ReferenceFitArgs {
    #[arg(
        long,
        default_value_t = 2000,
        help = "Features (HVGs) in the latent copula",
        long_help = "Features (HVGs) in the latent copula, per reference.\n\
                     Features outside it are drawn independently given depth.\n\
                     This is used only with a reference."
    )]
    pub n_hvg: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Maximum rank of the copula's latent correlation factor",
        long_help = "Maximum rank of the factor fitted to the HVGs' latent correlation.\n\
                     Only factors with positive eigenvalues are kept, so the effective rank\n\
                     is at most `n_hvg`. This is used only with a reference."
    )]
    pub copula_rank: usize,

    #[arg(
        long,
        default_value_t = 1e-3,
        help = "Smallest share of each HVG's latent variance kept outside the shared factors",
        long_help = "Smallest share of each HVG's latent variance kept as its own,\n\
                     outside the shared factors, so no feature is a copy of another.\n\
                     This is used only with a reference."
    )]
    pub regularization: f32,

    #[arg(
        long,
        default_value_t = 0.005,
        help = "Smallest share of reference cells an HVG must be detected in",
        long_help = "Smallest share of reference cells an HVG must be detected in,\n\
                     and never fewer than 50 cells. Features seen in fewer cells carry no\n\
                     correlation the reference can estimate; they are drawn independently\n\
                     given depth. This is used only with a reference."
    )]
    pub hvg_min_detection: f32,

    #[arg(
        long,
        default_value_t = false,
        help = "Do not couple features to a per-cell depth (reference mode)",
        long_help = "Do not couple features to a per-cell depth.\n\
                     By default each cell draws a depth coordinate (one per modality), and\n\
                     every feature follows it as strongly as it follows library size in the\n\
                     reference, so library sizes and detected features vary as the\n\
                     reference's do. With this flag every cell has the same expected depth.\n\
                     Each feature keeps the reference's distribution either way.\n\
                     This is used only with a reference."
    )]
    pub fixed_depth: bool,

    #[arg(
        long,
        hide = true,
        help = "No longer used: reference marginals are Poisson mixtures, with no NB size to bound"
    )]
    pub r_floor: Option<f32>,

    #[arg(
        long,
        help = "Cell groups the reference is fitted in (1: one population) [topic: 8; trajectory, multiome: 1]",
        long_help = "Cell groups the reference is fitted in, found by k-means on the leading\n\
                     principal components of its HVGs. Each group gets its own count\n\
                     distributions and depth coupling, and simulated cells are drawn group by\n\
                     group in the reference's proportions, so a reference of cell types of\n\
                     very different size keeps its library-size and detected-gene spread.\n\
                     Each cell's group is written to `{output}.ref_group.gz`.\n\
                     Groups of fewer than 100 cells join their nearest. 1 fits one\n\
                     population. The default is 8 for `topic`, and 1 for `trajectory`,\n\
                     whose tree is meant to be the cells' structure (more groups cross the\n\
                     reference's cell types with it), and for `multiome`, whose modalities\n\
                     would each draw a cell's group on their own.\n\
                     This is used only with a reference."
    )]
    pub reference_groups: Option<usize>,

    #[arg(
        long,
        help = "Cell labels to fit the reference by, instead of --reference-groups",
        long_help = "Cell labels to fit the reference by, instead of --reference-groups: one\n\
                     label per line in the reference's column order, or `cell<TAB>label`\n\
                     lines matched by cell name. This is used only with a reference."
    )]
    pub reference_labels: Option<Box<str>>,
}

/// Cell groups a reference is fitted in when the command adds its own
/// structure on top of the reference's (`topic`).
pub const DEFAULT_GROUPS: usize = 8;

/// How a reference's cells are grouped for the fit.
#[derive(Debug, Clone)]
pub enum GroupSpec {
    /// `k` groups by k-means; 1 is one population.
    Cluster(usize),
    /// Labels from a file ([`groups::CellGroups::from_labels`]).
    Labels(Box<str>),
}

impl ReferenceFitArgs {
    /// The fit's settings for the reference `sc`, in `default_groups` cell
    /// groups unless asked otherwise; warns about a retired flag.
    pub fn global_args<'a>(
        &self,
        sc: &'a SparseRef,
        default_groups: usize,
    ) -> GlobalCopulaArgs<'a> {
        if self.r_floor.is_some() {
            log::warn!("--r-floor is no longer used: reference marginals are Poisson mixtures");
        }
        GlobalCopulaArgs {
            sc,
            n_hvg: self.n_hvg,
            copula_rank: self.copula_rank,
            regularization: self.regularization,
            hvg_min_detection: self.hvg_min_detection,
            groups: match &self.reference_labels {
                Some(path) => GroupSpec::Labels(path.clone()),
                None => GroupSpec::Cluster(self.reference_groups.unwrap_or(default_groups)),
            },
        }
    }
}

pub struct GlobalCopulaArgs<'a> {
    pub sc: &'a SparseRef,
    /// HVGs included in the copula dependence structure. Genes outside the
    /// HVG set are drawn independently given depth.
    pub n_hvg: usize,
    /// Maximum rank of the low-rank Σ̂ factor. Effective rank is at most
    /// the number of HVGs.
    pub copula_rank: usize,
    /// Smallest share of each HVG's latent variance kept as its own,
    /// outside the shared factors.
    pub regularization: f32,
    /// Smallest share of reference cells in which an HVG is detected.
    pub hvg_min_detection: f32,
    pub groups: GroupSpec,
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
    let active_genes: Vec<usize> = (0..n_genes).filter(|&g| stats[g].nnz > 0).collect();
    // HVGs are detected in at least 50 cells, so they are active.
    let hvg_indices = reference::select_hvg(&stats, n_cells, args.n_hvg, args.hvg_min_detection);
    anyhow::ensure!(
        !hvg_indices.is_empty(),
        "no gene is detected in {:.1}% (and 50) of the {} reference cells; lower --hvg-min-detection",
        100.0 * args.hvg_min_detection,
        n_cells
    );
    info!(
        "active genes (detected in the reference): {}/{} ({:.1}%)",
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
    let cell_groups = match &args.groups {
        GroupSpec::Cluster(k) => groups::CellGroups::cluster(&mat, &libs, &hvg_indices, *k),
        GroupSpec::Labels(path) => groups::CellGroups::from_labels(path, args.sc)?,
    };
    let (group_fits, hvg_residual) = fit_groups(
        &mat,
        &libs,
        &cell_groups,
        &active_genes,
        &hvg_indices,
        &marginals,
        &hist,
    );
    drop(hist);
    let zero: Vec<f32> = active_genes
        .iter()
        .map(|&g| 1.0 - stats[g].nnz as f32 / n_cells as f32)
        .collect();
    let correlation_genes = fidelity::correlation_genes(&active_genes, &hvg_indices, &zero);
    let profile = fidelity::profile(&mat, n_genes, n_cells, &active_genes, &correlation_genes);
    drop(mat);
    let copula =
        CopulaCovariance::from_correlation(&hvg_residual, args.copula_rank, args.regularization)?;
    let mean_abs_depth = group_fits
        .iter()
        .map(|f| {
            f.share as f32
                * active_genes
                    .iter()
                    .map(|&g| f.depth_rho[g].abs())
                    .sum::<f32>()
                / active_genes.len().max(1) as f32
        })
        .sum::<f32>();
    info!(
        "fit global copula: {} HVGs (detected in ≥ {:.1}% of cells), rank {} (cap {}), \
         mean residual sd {:.3}; mean |ρ| with depth {:.3}; {} cell group(s)",
        hvg_indices.len(),
        100.0 * args.hvg_min_detection,
        copula.rank(),
        args.copula_rank,
        copula.ridge_sd.mean(),
        mean_abs_depth,
        group_fits.len()
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
        groups: group_fits,
        active_genes,
        hvg_pos,
        copula,
        profile,
        correlation_genes,
    })
}

/// Each group's count distributions and depth coupling (`whole` and
/// `whole_hist`, the whole reference's, for a single group), and the HVGs'
/// latent correlation beyond depth pooled within the groups: each pair's
/// group estimates averaged with weights `n_k · c_{1,i,k} · c_{1,h,k}`, the
/// information one group's cells carry on it ([`latent::LatentScores`]), so
/// a group in which a gene barely varies says little about its pairs.
fn fit_groups(
    mat: &nalgebra_sparse::CscMatrix<f32>,
    libs: &[f64],
    cell_groups: &groups::CellGroups,
    active: &[usize],
    hvg: &[usize],
    whole: &[PoissonMixture],
    whole_hist: &[Option<Vec<f64>>],
) -> (Vec<GroupFit>, nalgebra::DMatrix<f64>) {
    if cell_groups.n_groups() == 1 {
        // One group is the whole reference, already fitted.
        let lat = ReferenceLatent::fit(mat, libs, whole_hist, hvg);
        let fit = GroupFit {
            name: cell_groups.names[0].clone(),
            share: 1.0,
            marginals: whole.to_vec(),
            depth_rho: lat.depth_rho,
        };
        return (vec![fit], lat.hvg_residual);
    }
    let n_total = libs.len().max(1) as f64;
    let h = hvg.len();
    let mut num = nalgebra::DMatrix::<f64>::zeros(h, h);
    let mut den = nalgebra::DMatrix::<f64>::zeros(h, h);
    let mut fits = Vec::with_capacity(cell_groups.n_groups());
    for (k, cells) in cell_groups.cells().into_iter().enumerate() {
        let sub = groups::select_columns(mat, &cells);
        let sub_libs: Vec<f64> = cells.iter().map(|&c| libs[c]).collect();
        let n = cells.len();
        let hist = latent::gene_histograms(&sub, n, active);
        let marginals: Vec<PoissonMixture> = hist
            .par_iter()
            .map(|h| {
                h.as_deref()
                    .map_or_else(PoissonMixture::zero, PoissonMixture::fit)
            })
            .collect();
        let lat = ReferenceLatent::fit(&sub, &sub_libs, &hist, hvg);
        let c1 = &lat.hvg_information;
        for i in 0..h {
            for j in 0..h {
                if i != j {
                    let w = n as f64 * c1[i] * c1[j];
                    num[(i, j)] += w * lat.hvg_residual[(i, j)];
                    den[(i, j)] += w;
                }
            }
        }
        fits.push(GroupFit {
            name: cell_groups.names[k].clone(),
            share: n as f64 / n_total,
            marginals,
            depth_rho: lat.depth_rho,
        });
    }
    let residual = nalgebra::DMatrix::from_fn(h, h, |i, j| {
        if i == j {
            1.0
        } else if den[(i, j)] > 0.0 {
            num[(i, j)] / den[(i, j)]
        } else {
            0.0
        }
    });
    (fits, residual)
}
