//! `data-beans-sim trajectory`: arguments, the synthetic and reference
//! samplers, and the ground-truth files.

use super::{CellPlacement, GeneCurves, LineageTree};
use crate::hdf5_io::*;
use crate::sim::copula::reference::open_reference;
use crate::sim::copula::sampler::{
    batch_centre, depth_log_offset, sample_reference_counts, CellPrograms, CellTerm, TermScaling,
    Triplets,
};
use crate::sim::copula::{fit_global_copula, GlobalCopulaFit, ReferenceFitArgs};
use crate::sim::core::{library_scale, sample_lognormal_dictionary};
use crate::sim::handlers::{
    batch_covariance, build_and_write_backend, write_batch_truth, write_reference_groups,
    BatchProgram,
};
use crate::sparse_io::*;
use crate::zarr_io::apply_zip_flag;

use clap::Args;
use indicatif::ParallelProgressIterator;
use legume_numeric::matrix::common_io::*;
use legume_numeric::matrix::traits::*;
use log::info;
use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Poisson, Uniform};
use rayon::prelude::*;

/// Expected library size in synthetic mode when `--depth` is not given.
const DEFAULT_DEPTH: usize = 1000;

/// Pseudotimes at which `{out}.curves.parquet` tabulates every curve.
const CURVE_GRID: usize = 11;

#[derive(Args, Debug)]
pub struct TrajectoryArgs {
    #[arg(
        short,
        long,
        help = "Number of genes (synthetic mode; the reference's with --reference)"
    )]
    pub rows: Option<usize>,

    #[arg(short, long, help = "Number of cells")]
    pub cols: usize,

    #[arg(
        long,
        default_value_t = 3,
        help = "Lineages: a root branch that splits into this many (1: one linear branch)"
    )]
    pub lineages: usize,

    #[arg(
        long,
        value_delimiter = ',',
        conflicts_with = "lineages",
        help = "Tree as the parent of each branch after the root, e.g. `0,0,1,1`",
        long_help = "Tree topology as the parent of each branch after the root.\n\
                     Branch 0 is the root; `0,0,1,1` makes branches 1 and 2 start at\n\
                     the end of the root, and 3 and 4 at the end of branch 1.\n\
                     Each parent must be an earlier branch. Every leaf ends a lineage.\n\
                     Overrides `--lineages`."
    )]
    pub tree: Option<Vec<usize>>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Relative share of cells per branch, one per branch [equal]"
    )]
    pub branch_weights: Option<Vec<f64>>,

    #[arg(
        long,
        value_delimiter = ',',
        default_value = "1,1",
        help = "Beta(a,b) shape of the pseudotime density within a branch [uniform]",
        long_help = "Beta(a,b) shape of the pseudotime density within each branch, as `a,b`.\n\
                     `1,1` is uniform; `2,1` puts more cells late in a branch,\n\
                     `1,2` more early."
    )]
    pub time_shape: Vec<f64>,

    #[arg(
        long,
        default_value_t = 0.2,
        help = "Share of genes that change along the trajectory",
        long_help = "Share of genes that change along the trajectory, in [0, 1].\n\
                     The others keep one mean over all cells. In reference mode they\n\
                     are drawn among the genes with extra-Poisson variation in the\n\
                     reference: a gene without any has no room to change."
    )]
    pub dynamic_genes: f32,

    #[arg(
        long,
        default_value_t = 3,
        help = "Basis functions per branch: 1 gives monotone curves, more wigglier ones"
    )]
    pub basis: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Trajectory-PVE π_traj: log-variance of a dynamic gene's curve",
        long_help = "Trajectory-PVE π_traj: the log-variance the trajectory adds to a\n\
                     dynamic gene. Each curve is standardised over the simulated cells,\n\
                     then scaled by √π_traj, and centred so the gene's mean stays.\n\
                     \n\
                     In reference mode it is capped, like `topic --pve-topic`, so it never\n\
                     takes more log-variance than the gene's extra-Poisson log-variance\n\
                     in the reference; the gene's count distribution keeps the rest."
    )]
    pub pve_trajectory: f32,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "Noise-PVE π_noise: log-variance of an iid per-cell log-mean term",
        long_help = "Noise-PVE π_noise: adds `√π_noise · ε_{g,j}`, ε ~ N(0, 1) iid per gene\n\
                     and cell, centred so the mean stays.\n\
                     In reference mode it is capped by what the trajectory left of the\n\
                     gene's extra-Poisson log-variance."
    )]
    pub pve_noise: f32,

    #[arg(short, long, default_value_t = 1, help = "Number of batches B")]
    pub batches: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Batch-PVE π_batch: log-variance of a gene's batch shift",
        long_help = "Batch-PVE π_batch: each gene's log mean shifts by `√π_batch · z(g,b)`\n\
                     in batch b, z ~ N(0, 1) from a low-rank gene program (`--batch-rank`),\n\
                     centred across batches so the gene's mean over all cells stays.\n\
                     With one batch there is no shift. It is on top of the reference's\n\
                     dispersion, not capped by it."
    )]
    pub pve_batch: f32,

    #[arg(
        long,
        default_value_t = 2,
        help = "Rank of the batch-program subspace (0: iid shifts per gene)"
    )]
    pub batch_rank: usize,

    #[arg(
        long,
        value_enum,
        default_value_t = BatchProgram::Random,
        help = "Where the batch-program subspace comes from: `random`, or `empirical` (the reference copula's top factors)"
    )]
    pub batch_program: BatchProgram,

    #[arg(
        long,
        help = "Expected library size per cell [synthetic: 1000; reference: the reference's]"
    )]
    pub depth: Option<usize>,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Log-sd σ_β of the genes' baseline means (synthetic mode)"
    )]
    pub beta_scale: f32,

    #[arg(
        long,
        help = "Real single-cell reference (.h5, .zarr, .zarr.zip)",
        long_help = "Real single-cell reference: `.h5`, `.zarr` or `.zarr.zip`. Each gene\n\
                     keeps its reference count distribution (a Poisson mixture), cells\n\
                     vary in depth as the reference's do, and the copula genes keep their\n\
                     latent correlation; the trajectory moves genes within their\n\
                     reference dispersion, as `topic --reference` does with topics."
    )]
    pub reference: Option<Box<str>>,

    #[command(flatten)]
    pub fit: ReferenceFitArgs,

    #[arg(short, long, help = "Output file header")]
    pub output: Box<str>,

    #[arg(long, default_value_t = 42, help = "Random seed")]
    pub rseed: u64,

    #[arg(
        long,
        default_value_t = false,
        help = "Open the fidelity view when done (reference mode)"
    )]
    pub view: bool,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Backend format for output"
    )]
    pub backend: SparseIoBackend,

    #[arg(
        long = "no-zip",
        default_value_t = true,
        action = clap::ArgAction::SetFalse,
        help = "Write a plain `.zarr` directory instead of the default `.zarr.zip` archive"
    )]
    pub zip: bool,
}

impl TrajectoryArgs {
    fn tree(&self) -> anyhow::Result<LineageTree> {
        match &self.tree {
            Some(parents) => LineageTree::from_parents(parents),
            None => LineageTree::star(self.lineages),
        }
    }

    fn time_shape(&self) -> anyhow::Result<(f64, f64)> {
        match self.time_shape.as_slice() {
            &[a, b] if a > 0.0 && b > 0.0 => Ok((a, b)),
            s => anyhow::bail!("--time-shape takes two positive numbers `a,b`, not {s:?}"),
        }
    }

    fn output(&self) -> (Box<str>, Box<str>) {
        let effective = apply_zip_flag(&self.output, self.zip, &self.backend);
        let prefix = strip_backend_suffix(&effective).into();
        (effective, prefix)
    }
}

/// Where `trajectory --reference` writes its fidelity report; `None`
/// without a reference.
pub fn fidelity_report_path(args: &TrajectoryArgs) -> Option<String> {
    args.reference.as_ref()?;
    Some(crate::sim::copula::fidelity::report_path(&args.output().1))
}

/// The trajectory's structure, shared by both modes.
struct Structure {
    tree: LineageTree,
    cells: CellPlacement,
    curves: GeneCurves,
    batch_membership: Vec<usize>,
    /// Per batch, every gene's log shift `√π_batch · z_{·,b}`.
    batch_delta: Vec<DVector<f32>>,
}

impl Structure {
    /// The tree, cells, curves and batches for `n_genes` genes; with a
    /// reference `fit`, dynamic genes are drawn among [`dynamic_candidates`].
    fn sample(
        args: &TrajectoryArgs,
        n_genes: usize,
        fit: Option<&GlobalCopulaFit>,
        rng: &mut rand::rngs::StdRng,
    ) -> anyhow::Result<Self> {
        let tree = args.tree()?;
        let nn = args.cols;
        let cells = CellPlacement::sample(
            &tree,
            nn,
            args.branch_weights.as_deref().unwrap_or(&[]),
            args.time_shape()?,
            rng,
        )?;
        let candidates = fit.map(dynamic_candidates);
        let mut curves = GeneCurves::sample(
            &tree,
            n_genes,
            candidates.as_deref(),
            args.dynamic_genes,
            args.basis,
            rng,
        )?;
        curves.standardize(&cells);
        let bb = args.batches.max(1);
        let runif = Uniform::new(0, bb)?;
        let batch_membership: Vec<usize> = (0..nn).map(|_| runif.sample(rng)).collect();
        let pve_batch = args.pve_batch.clamp(0.0, 1.0);
        let batch_delta = batch_covariance(n_genes, args.batch_rank, args.batch_program, fit, rng)?
            .sample_batch_shifts(bb, pve_batch.sqrt(), rng);
        info!(
            "trajectory: {} branches, {} lineages, {} cells, {} of {} genes dynamic, {} batches",
            tree.n_branches(),
            tree.n_lineages(),
            nn,
            curves.dynamic.len(),
            n_genes,
            bb
        );
        Ok(Self {
            tree,
            cells,
            curves,
            batch_membership,
            batch_delta,
        })
    }

    /// Cell `j`'s standardised curve values `f̃(b_j, t_j)`.
    fn term(&self, j: usize) -> DVector<f32> {
        self.curves.at(self.cells.branch[j], self.cells.time[j])
    }

    fn cell_names(&self) -> Vec<Box<str>> {
        (0..self.cells.n_cells())
            .map(|j| {
                format!(
                    "synthetic_{}_{}@{}",
                    j, self.cells.branch[j], self.batch_membership[j]
                )
                .into_boxed_str()
            })
            .collect()
    }
}

/// Simulate counts along a lineage tree; see [`crate::sim::trajectory`].
pub fn run_trajectory(args: &TrajectoryArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        !(args.reference.is_some() && args.rows.is_some()),
        "`--rows` and `--reference` are mutually exclusive: the reference sets the genes"
    );
    if args.reference.is_some() {
        if args.beta_scale != 1.0 {
            log::warn!("--beta-scale is ignored with --reference: the reference sets the means");
        }
    } else if args.fit.fixed_depth || args.view {
        log::warn!("--fixed-depth and --view apply to reference mode only");
    }
    let (effective, prefix) = args.output();
    dirname(&prefix).as_deref().map(mkdir).transpose()?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.rseed);
    let pve_trajectory = args.pve_trajectory.clamp(0.0, 1.0);
    let pve_noise = args.pve_noise.clamp(0.0, 1.0);

    let (structure, triplets, gene_names, scaling) = match args.reference.as_deref() {
        None => {
            let dd = args
                .rows
                .ok_or_else(|| anyhow::anyhow!("--rows is required without --reference"))?;
            let structure = Structure::sample(args, dd, None, &mut rng)?;
            let amp = pve_trajectory.sqrt();
            let mut scaling = TermScaling {
                amp: vec![0.0; dd],
                centre: structure.curves.log_mean_exp(&structure.cells, amp),
            };
            for &g in &structure.curves.dynamic {
                scaling.amp[g] = amp;
            }
            let depth = args.depth.unwrap_or(DEFAULT_DEPTH);
            let baseline = sample_lognormal_dictionary(dd, 1, 0.0, args.beta_scale, &mut rng)
                * library_scale(depth, dd, 1, false);
            let triplets = sample_poisson(
                &structure,
                baseline.as_slice(),
                &scaling,
                pve_noise,
                args.rseed,
            )?;
            let names: Vec<Box<str>> = (0..dd).map(|g| g.to_string().into_boxed_str()).collect();
            (structure, triplets, names, scaling)
        }
        Some(reference) => {
            info!("opening reference: {reference}");
            let sc = open_reference(reference)?;
            // The tree is the cells' structure: one population unless asked.
            let fit = fit_global_copula(&args.fit.global_args(&sc, 1))?;
            let structure = Structure::sample(args, fit.n_genes, Some(&fit), &mut rng)?;
            let term = |j: usize| structure.term(j);
            let programs = CellPrograms {
                term: CellTerm::PerCell(&term),
                pve_topic: pve_trajectory,
                cap_spread: true,
                pve_noise,
                batch_delta: &structure.batch_delta,
                batch_membership: &structure.batch_membership,
            };
            let sample = sample_reference_counts(
                &fit,
                depth_log_offset(&fit, args.depth),
                &programs,
                !args.fit.fixed_depth,
                args.rseed,
                0x7a7a_7a7a,
            )?;
            let report =
                fit.write_fidelity_report(&sample.triplets, args.cols, reference, &prefix)?;
            info!("fidelity report: {report}");
            write_reference_groups(&fit, &sample.groups, &format!("{prefix}.ref_group.gz"))?;
            // Each group scales the curves within its own dispersion; the
            // ground truth reports their cell-weighted mean.
            let scaling = TermScaling::pooled(&sample.scaling, &sample.groups);
            (structure, sample.triplets, fit.gene_names.clone(), scaling)
        }
    };

    let cell_names = structure.cell_names();
    write_ground_truth(&structure, &prefix, &gene_names, &cell_names, &scaling)?;
    build_and_write_backend(
        &triplets,
        &gene_names,
        &cell_names,
        &args.backend,
        &effective,
    )
}

/// Active genes with extra-Poisson variation in the reference: a gene
/// without any has no room to move along the trajectory.
pub(crate) fn dynamic_candidates(fit: &GlobalCopulaFit) -> Vec<usize> {
    fit.active_genes
        .iter()
        .copied()
        .filter(|&g| fit.marginals[g].rate_cv2() > 0.0)
        .collect()
}

/// Synthetic counts, `y_{g,j} ~ Poisson(λ_{g,j})`; cell `j` uses the seed
/// `rseed + j`.
fn sample_poisson(
    s: &Structure,
    baseline: &[f32],
    scaling: &TermScaling,
    pve_noise: f32,
    rseed: u64,
) -> anyhow::Result<Triplets> {
    let dd = baseline.len();
    let nn = s.cells.n_cells();
    let b_centre = batch_centre(&s.batch_delta, &s.batch_membership, dd);
    let noise = pve_noise.sqrt();
    let normal = Normal::new(0.0_f32, 1.0)?;
    Ok((0..nn)
        .into_par_iter()
        .progress_count(nn as u64)
        .map(|j| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(rseed.wrapping_add(j as u64));
            let f = s.term(j);
            let shift = &s.batch_delta[s.batch_membership[j]];
            let mut out = Triplets::new();
            for g in 0..dd {
                let mut ln_m = scaling.amp[g] * f[g] - scaling.centre[g] + shift[g] - b_centre[g];
                if noise > 0.0 {
                    ln_m += noise * normal.sample(&mut rng) - 0.5 * pve_noise;
                }
                let lambda = baseline[g] * ln_m.exp();
                if !(lambda.is_finite() && lambda > 0.0) {
                    continue;
                }
                let y: f32 = Poisson::new(lambda)
                    .map(|p| p.sample(&mut rng))
                    .unwrap_or(0.0);
                if y > 0.5 {
                    out.push((g as u64, j as u64, y));
                }
            }
            out
        })
        .flatten()
        .collect())
}

fn names(v: impl IntoIterator<Item = String>) -> Vec<Box<str>> {
    v.into_iter().map(String::into_boxed_str).collect()
}

/// The ground truth next to the counts:
///
/// - `{out}.cells.parquet`: per cell its branch, pseudotime `t`, tree time
///   (branch depth + `t`), batch, and `lineage_{l}` (1 when the cell's
///   branch is on lineage `l`'s path);
/// - `{out}.tree.tsv`: per branch its parent, depth and lineages;
/// - `{out}.genes.parquet`: per gene whether it was drawn dynamic, the
///   amplitude and centre of its trajectory term (`amplitude · f̃ −
///   centre`, after the reference caps: a dynamic gene with little room in
///   the reference has a small amplitude), and its standardised curve's coefficients `b{b}_k{k}`
///   and branch starts `b{b}_start`;
/// - `{out}.curves.parquet`: that log-mean term on a pseudotime grid per
///   branch, `b{b}_t{t}`;
/// - `{out}.batch.gz` and `{out}.ln_batch.parquet`, as `topic` writes.
fn write_ground_truth(
    s: &Structure,
    prefix: &str,
    gene_names: &[Box<str>],
    cell_names: &[Box<str>],
    scaling: &TermScaling,
) -> anyhow::Result<()> {
    let (amp, centre) = (&scaling.amp, &scaling.centre);
    let tree = &s.tree;
    let (nb, nl) = (tree.n_branches(), tree.n_lineages());
    let nn = s.cells.n_cells();
    let dd = gene_names.len();

    let mut cells = DMatrix::<f32>::zeros(nn, 4 + nl);
    for j in 0..nn {
        let b = s.cells.branch[j];
        cells[(j, 0)] = b as f32;
        cells[(j, 1)] = s.cells.time[j];
        cells[(j, 2)] = s.cells.tree_time(tree, j);
        cells[(j, 3)] = s.batch_membership[j] as f32;
        for l in 0..nl {
            cells[(j, 4 + l)] = f32::from(u8::from(tree.on_lineage(b, l)));
        }
    }
    let cell_cols = names(
        ["branch", "pseudotime", "tree_time", "batch"]
            .map(String::from)
            .into_iter()
            .chain((0..nl).map(|l| format!("lineage_{l}"))),
    );
    let cells_file = format!("{prefix}.cells.parquet");
    cells.to_parquet_with_names(
        &cells_file,
        (Some(cell_names), Some("cell")),
        Some(&cell_cols),
    )?;

    let mut tree_lines: Vec<Box<str>> = vec!["branch\tparent\tdepth\tlineages".into()];
    for b in 0..nb {
        let lineages: Vec<String> = (0..nl)
            .filter(|&l| tree.on_lineage(b, l))
            .map(|l| l.to_string())
            .collect();
        tree_lines.push(
            format!(
                "{b}\t{}\t{}\t{}",
                tree.parent[b].map_or("-".to_string(), |p| p.to_string()),
                tree.depth[b],
                lineages.join(",")
            )
            .into(),
        );
    }
    write_lines(&tree_lines, &format!("{prefix}.tree.tsv"))?;

    let c = &s.curves;
    let kk = c.n_basis;
    let mut genes = DMatrix::<f32>::zeros(dd, 3 + nb * (kk + 1));
    for &g in &c.dynamic {
        genes[(g, 0)] = 1.0;
    }
    for g in 0..dd {
        genes[(g, 1)] = amp[g];
        genes[(g, 2)] = centre[g];
        for b in 0..nb {
            genes[(g, 3 + b * (kk + 1))] = c.start[(g, b)];
            for k in 0..kk {
                genes[(g, 3 + b * (kk + 1) + 1 + k)] = c.coef[(g, b * kk + k)];
            }
        }
    }
    let gene_cols = names(
        ["dynamic", "amplitude", "centre"]
            .map(String::from)
            .into_iter()
            .chain((0..nb).flat_map(|b| {
                std::iter::once(format!("b{b}_start"))
                    .chain((0..kk).map(move |k| format!("b{b}_k{}", k + 1)))
            })),
    );
    let genes_file = format!("{prefix}.genes.parquet");
    genes.to_parquet_with_names(
        &genes_file,
        (Some(gene_names), Some("feature")),
        Some(&gene_cols),
    )?;

    let grid: Vec<f32> = (0..CURVE_GRID)
        .map(|i| i as f32 / (CURVE_GRID - 1) as f32)
        .collect();
    let mut curves = DMatrix::<f32>::zeros(dd, nb * CURVE_GRID);
    for &g in &c.dynamic {
        for b in 0..nb {
            for (i, &t) in grid.iter().enumerate() {
                curves[(g, b * CURVE_GRID + i)] = amp[g] * c.value(g, b, t) - centre[g];
            }
        }
    }
    let curve_cols = names((0..nb).flat_map(|b| grid.iter().map(move |t| format!("b{b}_t{t:.1}"))));
    let curves_file = format!("{prefix}.curves.parquet");
    curves.to_parquet_with_names(
        &curves_file,
        (Some(gene_names), Some("feature")),
        Some(&curve_cols),
    )?;

    write_batch_truth(
        &s.batch_membership,
        &s.batch_delta,
        gene_names,
        &format!("{prefix}.batch.gz"),
        &format!("{prefix}.ln_batch.parquet"),
    )?;
    info!(
        "ground truth: {cells_file}, {prefix}.tree.tsv, {genes_file}, {curves_file}, {prefix}.batch.gz"
    );
    Ok(())
}
