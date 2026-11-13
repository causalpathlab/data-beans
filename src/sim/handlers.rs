use crate::hdf5_io::*;
use crate::sim::copula::reference::open_reference;
use crate::sim::copula::sampler::{
    depth_log_offset, sample_reference_counts, CellPrograms, CellTerm,
};
use crate::sim::copula::{fit_global_copula, ReferenceFitArgs};
use crate::sim::core as simulate;
use crate::sim::multimodal as simulate_multimodal;
use crate::sparse_io::*;
use crate::zarr_io::{apply_zip_flag, finalize_zarr_output};

use clap::{Args, ValueEnum};
use legume_numeric::matrix::common_io::*;
use legume_numeric::matrix::mtx_io;
use legume_numeric::matrix::traits::*;
use log::info;
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_distr::Distribution;
use rayon::prelude::*;

use crate::sim::core::{inject_housekeeping, sample_lognormal_dictionary};

use crate::sim::copula::gaussian::CopulaCovariance;

/// Expected library size in synthetic mode when `--depth` is not given.
const DEFAULT_DEPTH: usize = 1000;

/// How the batch-program covariance `F_batch` is constructed.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum BatchProgram {
    /// `F_batch` is a fresh `(D, batch_rank)` random factor independent of
    /// the reference's gene-gene structure. Batch shifts ride an arbitrary
    /// low-dim subspace — easier to disentangle from real co-expression,
    /// useful as a sanity baseline.
    Random,
    /// `F_batch` reuses the top `batch_rank` columns of the reference's
    /// fitted gene-gene copula factor `Σ̂_gene` — i.e. the leading PCs of
    /// the *empirical* gene-gene correlation in the reference. Batch shifts
    /// align with the same axes as real co-expression, the worst-case
    /// stress test for batch-correction methods.
    Empirical,
}

#[derive(Args, Debug)]
pub struct RunSimulateArgs {
    #[arg(
        short,
        long,
        help = "Number of rows, genes, or features (ignored when --reference is set)"
    )]
    pub rows: Option<usize>,

    #[arg(short, long, help = "Number of columns or cells")]
    pub cols: usize,

    #[arg(
        long,
        help = "Real single-cell reference (.h5, .zarr, .zarr.zip)",
        long_help = "Real single-cell reference: `.h5`, `.zarr` or `.zarr.zip`. When set,\n\
                     the GLM pipeline is unchanged.\n\
                     Only the final count generation step differs.\n\
                     It swaps `Poisson(λ)` for a copula-coupled draw from each gene's\n\
                     reference count distribution (a Poisson mixture, its rates scaled to λ),\n\
                     coupled to a per-cell depth and by a global Σ̂ fitted from this reference."
    )]
    pub reference: Option<Box<str>>,

    #[command(flatten)]
    pub fit: ReferenceFitArgs,

    #[arg(
        long,
        help = "Expected library size E[Σ_g Y(g,j)] per cell [synthetic default: 1000]",
        long_help = "Expected library size E[Σ_g Y(g,j)] per cell. It is emergent;\n\
                     nothing is rescaled per cell.\n\
                     In synthetic mode it enters as λ_scale = depth/G, or depth·K with\n\
                     `--hierarchical-depth`, whose dictionary sums to 1 over all genes and topics.\n\
                     It defaults to 1000 there.\n\
                     In reference mode it scales every per-gene mean μ̂_g so the mean library\n\
                     size is `depth`; unset keeps the reference's."
    )]
    pub depth: Option<usize>,

    #[arg(
        short,
        long,
        default_value_t = 1,
        help = "Number of factors K (cell types / topics / states); width of β and θ"
    )]
    pub factors: usize,

    #[arg(short, long, default_value_t = 1, help = "Number of batches B")]
    pub batches: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Topic-PVE π_topic ∈ [0, 1]",
        long_help = "Topic-PVE π_topic ∈ [0, 1].\n\
                     It splits log β in two.\n\
                     One part is topic-specific, the other topic-invariant:\n\
                     \x20 log β(g,k) = σ_β·[√π_topic·u(g,k) + √(1−π_topic)·v(g)] − σ_β²/2\n\
                     \n\
                     It also softens θ from one-hot toward uniform:\n\
                     \x20 θ(k*,j) = π_topic + (1−π_topic)/K\n\
                     \n\
                     π_topic = 0 gives no topic structure.\n\
                     π_topic = 1 gives pure topic structure.\n\
                     It is independent of pve_batch; both can be 1.\n\
                     \n\
                     In reference mode √π_topic scales the z-scored log topic loading of each\n\
                     cell, centred per gene, and capped so it never takes more log-variance\n\
                     than the gene's extra-Poisson log-variance in the reference."
    )]
    pub pve_topic: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Batch-PVE π_batch ∈ [0, 1]",
        long_help = "Batch-PVE π_batch ∈ [0, 1].\n\
                     It splits log δ in two.\n\
                     One part is batch-specific, the other batch-invariant:\n\
                     \x20 log δ(g,b) = √π_batch·z(g,b) + √(1−π_batch)·w(g)\n\
                     \n\
                     π_batch = 0 makes batches share one per-gene shift exp(w).\n\
                     π_batch = 1 is fully batch-specific.\n\
                     It is independent of pve_topic.\n\
                     \n\
                     In reference mode the reference's per-gene mean is the baseline:\n\
                     \x20 log δ(g,b) = √π_batch·z(g,b)\n\
                     with two or more batches, z centred across batches per gene,\n\
                     and no shift with one."
    )]
    pub pve_batch: f32,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "PVE-style magnitude for the per-cell residual log-mean noise term",
        long_help = "PVE-style magnitude for the per-cell residual noise term.\n\
                     It applies to the log-mean, in reference mode.\n\
                     \n\
                     Above 0 it adds `√pve_noise · ε_{g,j}`, centred so the mean stays.\n\
                     `ε ~ N(0,1)` is iid per gene per cell.\n\
                     The term sits on top of the topic perturbation, and applies before batch.\n\
                     Together they never take more log-variance than the gene's reference\n\
                     extra-Poisson log-variance; the gene's count distribution keeps the rest.\n\
                     \n\
                     The default of 0 leaves topics and the reference alone."
    )]
    pub pve_noise: f32,

    #[arg(
        long,
        default_value_t = 2,
        help = "Rank of the batch-program subspace (reference mode)",
        long_help = "Rank of the batch-program subspace, in reference mode.\n\
                     `0` makes each gene's batch shift iid log-normal,\n\
                     which is the Splatter style.\n\
                     `2-3` makes genes co-shift along a low-dim subspace,\n\
                     a \"batch program\".\n\
                     Higher ranks make batch effects look more structured."
    )]
    pub batch_rank: usize,

    #[arg(
        long,
        value_enum,
        default_value_t = BatchProgram::Random,
        help = "Where the batch-program subspace comes from when --batch-rank > 0",
        long_help = "Where the batch-program subspace comes from.\n\
                     This applies when `--batch-rank > 0`.\n\
                     \n\
                     `random` draws a fresh low-dim random factor. It is the default,\n\
                     and an arbitrary subspace.\n\
                     \n\
                     `empirical` takes the top columns of the reference's fitted gene-gene copula factor.\n\
                     That is the worst case:\n\
                     batch shifts then ride the reference's real co-expression PCs."
    )]
    pub batch_program: BatchProgram,

    #[arg(short, long, help = "Output file header")]
    pub output: Box<str>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Route cells whose dominant topic is in this list to a second `<out>.holdout` backend",
        long_help = "Comma-separated 0-indexed topic ids.\n\
                     A cell whose dominant topic (argmax θ) is in this set goes to `<out>.holdout.<backend>`.\n\
                     It is kept out of the primary `<out>.<backend>`.\n\
                     So a model trained on the primary file provably never sees these topics.\n\
                     \n\
                     The β and θ ground-truth parquets stay full.\n\
                     Column names in both backends are the original cell indices.\n\
                     Cross-reference them against `<out>.prop.parquet`.\n\
                     \n\
                     Unset by default, giving a single output file. Synthetic mode only."
    )]
    pub holdout_topics: Option<Vec<usize>>,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Log-normal scale σ_β",
        long_help = "Log-normal scale σ_β. Total log-variance per gene-topic entry is σ_β².\n\
                     That is independent of pve_topic.\n\
                     Centering gives E[β(g,k)] = 1. Higher values vary expression more across genes and topics."
    )]
    pub beta_scale: f32,

    #[arg(long, default_value_t = 42, help = "Random seed")]
    pub rseed: u64,

    #[arg(long, help = "Hierarchical tree depth for binary tree dictionary")]
    pub hierarchical_depth: Option<usize>,

    #[arg(long, default_value_t = 0, help = "Number of housekeeping genes")]
    pub n_housekeeping: usize,

    #[arg(long, default_value_t = 10.0, help = "Housekeeping fold change")]
    pub housekeeping_fold: f32,

    #[arg(long, default_value_t = false, help = "Save output in MTX format")]
    pub save_mtx: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Open the fidelity view when done (reference mode)",
        long_help = "Open the fidelity view when done, in reference mode.\n\
                     It draws the simulation against its reference from `{output}.fidelity.json`:\n\
                     library size, detected genes, gene moments and gene-gene correlation.\n\
                     `data-beans-sim view {output}.fidelity.json` opens it again later."
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

#[derive(Args, Debug)]
pub struct RunSimulateMultimodalArgs {
    #[arg(short, long, help = "Number of features (shared across modalities)")]
    pub rows: usize,

    #[arg(short, long, help = "Number of cells")]
    pub cols: usize,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Expected library size per modality, comma-separated",
        long_help = "Expected library size per modality, comma-separated.\n\
                     An example is 1000,500. There is one entry per modality,\n\
                     and the length defines M. Each modality's β columns are softmax-normalized over genes.\n\
                     So depth_m directly sets E[lib(j)|m]."
    )]
    pub depth: Vec<usize>,

    #[arg(
        short,
        long,
        default_value_t = 5,
        help = "Number of topics K (shared across modalities)"
    )]
    pub factors: usize,

    #[arg(short, long, default_value_t = 1, help = "Number of batches B")]
    pub batches: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Scale σ_base of base-dictionary logits",
        long_help = "Scale σ_base of the base-dictionary logits.\n\
                     They are drawn W_base ~ N(0, σ_base²).\n\
                     This applies when hierarchical mode is off."
    )]
    pub base_scale: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Scale σ_Δ of non-zero spike entries Δ_m[k,g] ~ N(0, σ_Δ²)"
    )]
    pub delta_scale: f32,

    #[arg(
        long,
        default_value_t = 5,
        help = "Spike-and-slab support: number of non-zero genes per topic in each Δ_m"
    )]
    pub n_delta_features: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Topic-PVE π_topic ∈ [0, 1]",
        long_help = "Topic-PVE π_topic ∈ [0, 1].\n\
                     It softens θ from one-hot toward uniform:\n\
                     \x20 θ(k*,j) = π_topic + (1−π_topic)/K\n\
                     It is independent of pve_batch."
    )]
    pub pve_topic: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Batch-PVE π_batch ∈ [0, 1]",
        long_help = "Batch-PVE π_batch ∈ [0, 1].\n\
                     It sets the variance share in log δ:\n\
                     \x20 log δ_m(g,b) = √π_batch·z(g,b) + √(1−π_batch)·w(g)\n\
                     It is independent of pve_topic."
    )]
    pub pve_batch: f32,

    #[arg(long, default_value_t = 42, help = "Random seed")]
    pub rseed: u64,

    #[arg(
        long,
        default_value_t = false,
        help = "Share batch effects across modalities"
    )]
    pub shared_batch_effects: bool,

    #[arg(long, help = "Hierarchical tree depth for base dictionary")]
    pub hierarchical_depth: Option<usize>,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Log-normal scale σ_β for the hierarchical-base dictionary"
    )]
    pub beta_scale: f32,

    #[arg(long, default_value_t = 0, help = "Number of housekeeping genes")]
    pub n_housekeeping: usize,

    #[arg(long, default_value_t = 10.0, help = "Housekeeping fold change")]
    pub housekeeping_fold: f32,

    #[arg(short, long, help = "Output file header")]
    pub output: Box<str>,

    #[arg(long, default_value_t = false, help = "Save output in MTX format")]
    pub save_mtx: bool,

    #[arg(long, value_enum, default_value = "zarr", help = "Backend format")]
    pub backend: SparseIoBackend,

    #[arg(
        long = "no-zip",
        default_value_t = true,
        action = clap::ArgAction::SetFalse,
        help = "Write a plain `.zarr` directory instead of the default `.zarr.zip` archive"
    )]
    pub zip: bool,
}

/// The batch-program covariance of the per-batch log shifts: rank 0 gives
/// iid shifts per gene (Splatter-style); a positive rank a low-rank factor
/// plus an iid residual, random or the reference copula's top factors.
pub(crate) fn batch_covariance(
    dd: usize,
    rank: usize,
    program: BatchProgram,
    fit: Option<&crate::sim::copula::GlobalCopulaFit>,
    rng: &mut impl rand::Rng,
) -> anyhow::Result<CopulaCovariance> {
    Ok(match (rank, program, fit) {
        (0, ..) | (_, BatchProgram::Random, _) => CopulaCovariance::random_low_rank(dd, rank, rng),
        (_, BatchProgram::Empirical, Some(fit)) => {
            fit.copula.truncate_rank(rank).embed(&fit.hvg_indices, dd)?
        }
        (_, BatchProgram::Empirical, None) => {
            anyhow::bail!("--batch-program empirical needs a --reference")
        }
    })
}

/// Each cell's batch, one per line, and the per-batch log shifts already
/// scaled by `√π_batch`, gene × batch, as `topic` writes them.
pub(crate) fn write_batch_truth(
    membership: &[usize],
    delta: &[nalgebra::DVector<f32>],
    row_names: &[Box<str>],
    batch_file: &str,
    ln_batch_file: &str,
) -> anyhow::Result<()> {
    let lines: Vec<Box<str>> = membership
        .iter()
        .map(|b| b.to_string().into_boxed_str())
        .collect();
    write_lines(&lines, batch_file)?;
    let mut ln_delta_db = DMatrix::<f32>::zeros(row_names.len(), delta.len());
    for (b, col) in delta.iter().enumerate() {
        ln_delta_db.set_column(b, col);
    }
    ln_delta_db.to_parquet_with_names(ln_batch_file, (Some(row_names), Some("feature")), None)?;
    info!("batch membership: {batch_file}; log shifts: {ln_batch_file}");
    Ok(())
}

/// Build a sparse backend from column-remapped triplets and register
/// row/column names. Shared by the single-file and holdout-split write
/// paths of [`run_simulate`].
pub(crate) fn build_and_write_backend(
    triplets: &[(u64, u64, f32)],
    row_names: &[Box<str>],
    col_names: &[Box<str>],
    backend: &SparseIoBackend,
    effective_output: &str,
) -> anyhow::Result<()> {
    let (_, backend_file) = resolve_backend_file(effective_output, Some(backend.clone()))?;
    remove_all_files(&vec![backend_file.clone()]).ok();

    let shape = (row_names.len(), col_names.len(), triplets.len());
    let mut data =
        create_sparse_from_triplets(triplets, shape, Some(&backend_file), Some(backend))?;
    data.register_row_names_vec(row_names);
    data.register_column_names_vec(col_names);

    finalize_zarr_output(&backend_file, effective_output)?;
    info!(
        "wrote backend: {} ({} cells)",
        backend_file,
        col_names.len()
    );
    Ok(())
}

/// Simulate log-normal factored count data.
///
/// Without `--reference`: pure log-normal · Poisson model
/// `y(g,j) ~ Poisson( λ_scale · δ(g,B(j)) · Σ_k β(g,k) θ(k,j) )`, λ_scale
/// from [`crate::sim::core::library_scale`], with
/// log β decomposed into topic-specific + topic-invariant components by
/// `pve_topic`, and log δ decomposed into batch-specific + batch-invariant
/// by `pve_batch`. Library size is emergent.
///
/// With `--reference`: same β / θ / δ structure, but counts are drawn from
/// each gene's reference count distribution, coupled to depth and across
/// HVGs by the reference-fitted copula. Per-gene baseline becomes log μ̂_g
/// (replacing log(depth/G)).
pub fn run_simulate(cmd_args: &RunSimulateArgs) -> anyhow::Result<()> {
    if cmd_args.reference.is_some() && cmd_args.rows.is_some() {
        anyhow::bail!(
            "`--rows` and `--reference` are mutually exclusive. \
             Under `--reference` the gene count is taken from the reference."
        );
    }
    if cmd_args.reference.is_some() {
        return run_simulate_with_reference(cmd_args);
    }

    let rows = cmd_args
        .rows
        .ok_or_else(|| anyhow::anyhow!("--rows is required when --reference is not set"))?;

    let effective_output = apply_zip_flag(&cmd_args.output, cmd_args.zip, &cmd_args.backend);
    let output: Box<str> = strip_backend_suffix(&effective_output).into();

    dirname(&output).as_deref().map(mkdir).transpose()?;

    let backend = cmd_args.backend.clone();
    let (_, backend_file) = resolve_backend_file(&effective_output, Some(backend.clone()))?;

    let mtx_file = output.to_string() + ".mtx.gz";
    let row_file = output.to_string() + ".rows.gz";
    let col_file = output.to_string() + ".cols.gz";

    let dict_file = mtx_file.replace(".mtx.gz", ".dict.parquet");
    let prop_file = mtx_file.replace(".mtx.gz", ".prop.parquet");
    let batch_memb_file = mtx_file.replace(".mtx.gz", ".batch.gz");
    let ln_batch_file = mtx_file.replace(".mtx.gz", ".ln_batch.parquet");

    remove_all_files(&vec![
        backend_file.clone(),
        mtx_file.clone().into_boxed_str(),
        dict_file.clone().into_boxed_str(),
        prop_file.clone().into_boxed_str(),
        batch_memb_file.clone().into_boxed_str(),
        ln_batch_file.clone().into_boxed_str(),
    ])
    .expect("failed to clean up existing output files");

    let sim_args = simulate::SimArgs {
        rows,
        cols: cmd_args.cols,
        depth: cmd_args.depth.unwrap_or(DEFAULT_DEPTH),
        factors: cmd_args.factors,
        batches: cmd_args.batches,
        beta_scale: cmd_args.beta_scale,
        pve_topic: cmd_args.pve_topic,
        pve_batch: cmd_args.pve_batch,
        rseed: cmd_args.rseed,
        hierarchical_depth: cmd_args.hierarchical_depth,
        n_housekeeping: cmd_args.n_housekeeping,
        housekeeping_fold: cmd_args.housekeeping_fold,
    };

    let sim = simulate::generate_factored_poisson_gamma_data(&sim_args)?;
    info!("successfully generated log-normal factored Poisson data");

    let batch_out: Vec<Box<str>> = sim
        .batch_membership
        .iter()
        .map(|&x| Box::from(x.to_string()))
        .collect();

    write_lines(&batch_out, &batch_memb_file)?;
    info!("batch membership: {:?}", batch_memb_file);

    let rows: Vec<Box<str>> = (0..sim_args.rows)
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    let cols: Vec<Box<str>> = (0..cmd_args.cols)
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    sim.ln_delta_db
        .to_parquet_with_names(&ln_batch_file, (Some(&rows), Some("feature")), None)?;
    sim.theta_kn.transpose().to_parquet_with_names(
        &prop_file,
        (Some(&cols), Some("cell")),
        None,
    )?;
    sim.beta_dk
        .to_parquet_with_names(&dict_file, (Some(&rows), Some("feature")), None)?;

    if let Some(ref node_probs) = sim.hierarchy_node_probs {
        let hierarchy_file = mtx_file.replace(".mtx.gz", ".hierarchy.parquet");
        node_probs.to_parquet_with_names(&hierarchy_file, (Some(&rows), Some("feature")), None)?;
        info!("wrote hierarchy node probabilities: {:?}", hierarchy_file);
    }

    info!(
        "wrote parameter files:\n{:?},\n{:?},\n{:?}",
        ln_batch_file, dict_file, prop_file
    );

    if cmd_args.save_mtx {
        let mut triplets = sim.triplets.clone();
        triplets.par_sort_by_key(|&(row, _, _)| row);
        triplets.par_sort_by_key(|&(_, col, _)| col);

        mtx_io::write_mtx_triplets(&triplets, sim_args.rows, sim_args.cols, &mtx_file)?;
        write_lines(&rows, &row_file)?;
        write_lines(&cols, &col_file)?;

        info!(
            "save mtx, row, and column files:\n{}\n{}\n{}",
            mtx_file, row_file, col_file
        );
    }

    info!("registering triplets ...");

    let holdout_set: Option<std::collections::HashSet<usize>> = cmd_args
        .holdout_topics
        .as_ref()
        .map(|v| v.iter().copied().collect());

    match holdout_set {
        None => {
            build_and_write_backend(&sim.triplets, &rows, &cols, &backend, &effective_output)?;
        }
        Some(holdout) => {
            // Route each cell by its dominant (argmax θ) topic.
            let dom: Vec<usize> = (0..sim_args.cols)
                .map(|j| sim.theta_kn.column(j).imax())
                .collect();

            let mut prim_map = vec![usize::MAX; sim_args.cols];
            let mut hold_map = vec![usize::MAX; sim_args.cols];
            let mut prim_cols: Vec<Box<str>> = Vec::new();
            let mut hold_cols: Vec<Box<str>> = Vec::new();
            for j in 0..sim_args.cols {
                if holdout.contains(&dom[j]) {
                    hold_map[j] = hold_cols.len();
                    hold_cols.push(cols[j].clone());
                } else {
                    prim_map[j] = prim_cols.len();
                    prim_cols.push(cols[j].clone());
                }
            }

            let mut prim_tr: Vec<(u64, u64, f32)> = Vec::with_capacity(sim.triplets.len());
            let mut hold_tr: Vec<(u64, u64, f32)> = Vec::new();
            for &(g, j, v) in &sim.triplets {
                let jj = j as usize;
                if holdout.contains(&dom[jj]) {
                    hold_tr.push((g, hold_map[jj] as u64, v));
                } else {
                    prim_tr.push((g, prim_map[jj] as u64, v));
                }
            }

            build_and_write_backend(&prim_tr, &rows, &prim_cols, &backend, &effective_output)?;

            let holdout_effective = apply_zip_flag(
                &format!("{}.holdout", cmd_args.output),
                cmd_args.zip,
                &backend,
            );
            build_and_write_backend(&hold_tr, &rows, &hold_cols, &backend, &holdout_effective)?;

            info!(
                "holdout split by dominant topic {:?}: {} primary + {} holdout cells",
                cmd_args.holdout_topics.as_ref().unwrap(),
                prim_cols.len(),
                hold_cols.len()
            );
        }
    }

    info!("done");
    Ok(())
}

/// Where `topic --reference` writes its fidelity report,
/// `{output}.fidelity.json`; `None` without a reference.
pub fn fidelity_report_path(cmd_args: &RunSimulateArgs) -> Option<String> {
    cmd_args.reference.as_ref()?;
    let effective_output = apply_zip_flag(&cmd_args.output, cmd_args.zip, &cmd_args.backend);
    Some(crate::sim::copula::fidelity::report_path(
        strip_backend_suffix(&effective_output),
    ))
}

/// Reference-conditioned variant of [`run_simulate`]. Each gene keeps the
/// reference count distribution, a Poisson mixture; on top of it:
///
/// - **topics and noise** move a gene between cells within its reference
///   dispersion: `√pve_topic` scales the z-scored topic loading and
///   `√pve_noise` an iid term, both mean-centred and capped so their
///   log-variance never exceeds the gene's extra-Poisson log-variance; the
///   mixture keeps the rest;
/// - **batches** shift the log mean by `√pve_batch · δ_{g, b(j)}`, where
///   `δ_{:,b} ~ N(0, F_b · F_bᵀ + diag(1 − ‖F_b‖²))`, centred so a gene's mean
///   over all cells stays the reference's. `F_b`'s rank is `--batch-rank`;
///   `--batch-program` picks `random` (a fresh low-rank factor) or
///   `empirical` (the top factors of the reference's HVG copula);
/// - **depth and co-expression** come from the latent: each cell draws a
///   depth coordinate that every gene follows as it follows library size in
///   the reference, and the HVGs share the reference's latent correlation
///   beyond depth.
///
/// Counts are `y = F⁻¹_g(Φ(z*))` with the mixture's rates scaled to `λ`; see
/// [`crate::sim::copula::sampler`].
fn run_simulate_with_reference(cmd_args: &RunSimulateArgs) -> anyhow::Result<()> {
    if cmd_args.holdout_topics.is_some() {
        log::warn!("--holdout-topics is ignored in reference mode (synthetic mode only)");
    }
    let reference_path = cmd_args
        .reference
        .as_ref()
        .expect("run_simulate_with_reference called without --reference");

    let effective_output = apply_zip_flag(&cmd_args.output, cmd_args.zip, &cmd_args.backend);
    let output: Box<str> = strip_backend_suffix(&effective_output).into();
    dirname(&output).as_deref().map(mkdir).transpose()?;

    let backend = cmd_args.backend.clone();
    let (_, backend_file) = resolve_backend_file(&effective_output, Some(backend.clone()))?;

    info!("opening reference: {}", reference_path);
    let sc = open_reference(reference_path)?;

    let fit = fit_global_copula(&cmd_args.fit.global_args(&sc))?;

    let dd = fit.n_genes;
    let nn = cmd_args.cols;
    let bb = cmd_args.batches.max(1);
    let kk = if let Some(depth) = cmd_args.hierarchical_depth {
        1usize << (depth - 1)
    } else {
        cmd_args.factors.max(1)
    };
    let pve_topic = cmd_args.pve_topic.clamp(0.0, 1.0);
    let pve_batch = cmd_args.pve_batch.clamp(0.0, 1.0);
    let pve_noise = cmd_args.pve_noise.clamp(0.0, 1.0);
    let alpha_batch = pve_batch.sqrt();

    let mut rng = rand::rngs::StdRng::seed_from_u64(cmd_args.rseed);

    let runif = rand_distr::Uniform::new(0, bb).expect("unif [0 .. bb)");
    let batch_membership: Vec<usize> = (0..nn).map(|_| runif.sample(&mut rng)).collect();

    // Every gene's mean shifted so the mean library size is `--depth`.
    let depth_offset = depth_log_offset(&fit, cmd_args.depth);
    info!("gene means scaled by {:.4} for --depth", depth_offset.exp());

    // Stage-2 batch covariance. rank=0 → all-isotropic (Splatter-style);
    // rank>0 → low-rank factor + iid residual.
    let batch_cov = batch_covariance(
        dd,
        cmd_args.batch_rank,
        cmd_args.batch_program,
        Some(&fit),
        &mut rng,
    )?;
    info!(
        "stage-2 batch program: rank={} ({:?}), {} batches",
        batch_cov.rank(),
        cmd_args.batch_program,
        bb
    );
    // Pre-sample the per-batch log shift `log δ_{g,b} = √π_batch · z_{g,b}`,
    // z_{g,b} from the (gene-gene-correlated) batch copula. The reference's
    // `μ̂_g` is the gene baseline, so nothing batch-invariant is added.
    let batch_delta = batch_cov.sample_batch_shifts(bb, alpha_batch, &mut rng);

    // Topic dictionary β: log-normal with explicit topic-PVE decomposition,
    // matching synthetic mode. Hierarchical mode uses the stick-breaking
    // tree (which encodes topic structure by construction; no extra blend).
    let (mut beta_dk, hierarchy_node_probs) = if let Some(tree_depth) = cmd_args.hierarchical_depth
    {
        let (beta, node_probs) = crate::sim::core::generate_hierarchical_dictionary(
            dd,
            tree_depth,
            cmd_args.beta_scale,
            &mut rng,
        );
        info!(
            "generated hierarchical dictionary: depth={}, K={} leaves",
            tree_depth,
            beta.ncols()
        );
        (beta, Some(node_probs))
    } else {
        (
            sample_lognormal_dictionary(dd, kk, pve_topic, cmd_args.beta_scale, &mut rng),
            None,
        )
    };

    inject_housekeeping(
        &mut beta_dk,
        cmd_args.n_housekeeping,
        cmd_args.housekeeping_fold,
        &mut rng,
    );

    let theta_kn = crate::sim::core::sample_theta_kn(kk, nn, pve_topic, &mut rng)?;

    let programs = CellPrograms {
        term: CellTerm::Topics {
            dict_dk: &beta_dk,
            theta_kn: &theta_kn,
        },
        pve_topic,
        cap_spread: false,
        pve_noise,
        batch_delta: &batch_delta,
        batch_membership: &batch_membership,
    };
    let (triplets, _) = sample_reference_counts(
        &fit,
        depth_offset,
        &programs,
        !cmd_args.fit.fixed_depth,
        cmd_args.rseed,
        0x5a5a_5a5a,
    )?;

    info!(
        "sampled {} cells producing {} nonzero triplets from the reference",
        nn,
        triplets.len()
    );

    // Output: same companion files as before. `.ln_batch.parquet` now stores
    // `α_batch · δ_{g, b}` — the actual log-shift applied per gene per batch.
    let mtx_file = output.to_string() + ".mtx.gz";
    let row_file = output.to_string() + ".rows.gz";
    let col_file = output.to_string() + ".cols.gz";
    let dict_file = format!("{}.dict.parquet", output);
    let prop_file = format!("{}.prop.parquet", output);
    let batch_memb_file = format!("{}.batch.gz", output);
    let ln_batch_file = format!("{}.ln_batch.parquet", output);
    let r_file = format!("{}.r.parquet", output);
    let hvg_file = format!("{}.hvg.gz", output);

    remove_all_files(&vec![
        backend_file.clone(),
        mtx_file.clone().into_boxed_str(),
        dict_file.clone().into_boxed_str(),
        prop_file.clone().into_boxed_str(),
        batch_memb_file.clone().into_boxed_str(),
        ln_batch_file.clone().into_boxed_str(),
        r_file.clone().into_boxed_str(),
        hvg_file.clone().into_boxed_str(),
    ])
    .expect("failed to clean up existing output files");

    let row_names: Vec<Box<str>> = fit.gene_names.clone();
    let col_names: Vec<Box<str>> = (0..nn)
        .map(|j| {
            let argmax_k = theta_kn.column(j).imax();
            format!("synthetic_{}_{}@{}", j, argmax_k, batch_membership[j]).into_boxed_str()
        })
        .collect();

    write_batch_truth(
        &batch_membership,
        &batch_delta,
        &row_names,
        &batch_memb_file,
        &ln_batch_file,
    )?;
    theta_kn.transpose().to_parquet_with_names(
        &prop_file,
        (Some(&col_names), Some("cell")),
        None,
    )?;
    beta_dk.to_parquet_with_names(&dict_file, (Some(&row_names), Some("feature")), None)?;

    if let Some(node_probs) = hierarchy_node_probs.as_ref() {
        let hierarchy_file = format!("{}.hierarchy.parquet", output);
        node_probs.to_parquet_with_names(
            &hierarchy_file,
            (Some(&row_names), Some("feature")),
            None,
        )?;
        info!("hierarchy node probabilities: {}", hierarchy_file);
    }

    fit.write_effective_nb_size(&r_file, &row_names)?;
    info!("per-gene NB size of the same variance: {}", r_file);

    let hvg_lines: Vec<Box<str>> = fit
        .hvg_indices
        .iter()
        .map(|&g| row_names[g].clone())
        .collect();
    write_lines(&hvg_lines, &hvg_file)?;
    info!("HVGs used by copula: {}", hvg_file);

    let fidelity_file = fit.write_fidelity_report(&triplets, nn, reference_path, &output)?;
    info!("fidelity report: {}", fidelity_file);

    let mut triplets = triplets;
    if cmd_args.save_mtx {
        triplets.par_sort_by_key(|&(row, _, _)| row);
        triplets.par_sort_by_key(|&(_, col, _)| col);
        mtx_io::write_mtx_triplets(&triplets, dd, nn, &mtx_file)?;
        write_lines(&row_names, &row_file)?;
        write_lines(&col_names, &col_file)?;
        info!(
            "save mtx, row, and column files:\n{}\n{}\n{}",
            mtx_file, row_file, col_file
        );
    }

    let mtx_shape = (dd, nn, triplets.len());
    let mut data =
        create_sparse_from_triplets(&triplets, mtx_shape, Some(&backend_file), Some(&backend))?;
    data.register_row_names_vec(&row_names);
    data.register_column_names_vec(&col_names);
    finalize_zarr_output(&backend_file, &effective_output)?;
    info!("wrote sparse backend: {}", backend_file);

    info!("done");
    Ok(())
}

/// Run multimodal simulation with shared base + delta dictionaries.
pub fn run_simulate_multimodal(cmd_args: &RunSimulateMultimodalArgs) -> anyhow::Result<()> {
    let output = cmd_args.output.clone();
    dirname(&output).as_deref().map(mkdir).transpose()?;

    let sim_args = simulate_multimodal::MultimodalSimArgs {
        rows: cmd_args.rows,
        cols: cmd_args.cols,
        depth_per_modality: cmd_args.depth.clone(),
        factors: cmd_args.factors,
        batches: cmd_args.batches,
        base_scale: cmd_args.base_scale,
        delta_scale: cmd_args.delta_scale,
        n_delta_features: cmd_args.n_delta_features,
        pve_topic: cmd_args.pve_topic,
        pve_batch: cmd_args.pve_batch,
        rseed: cmd_args.rseed,
        shared_batch_effects: cmd_args.shared_batch_effects,
        hierarchical_depth: cmd_args.hierarchical_depth,
        beta_scale: cmd_args.beta_scale,
        n_housekeeping: cmd_args.n_housekeeping,
        housekeeping_fold: cmd_args.housekeeping_fold,
    };

    let mm = sim_args.depth_per_modality.len();
    let sim = simulate_multimodal::generate_multimodal_data(&sim_args)?;
    info!("generated multimodal data: {} modalities", mm);

    let rows: Vec<Box<str>> = (0..cmd_args.rows)
        .map(|i| i.to_string().into_boxed_str())
        .collect();
    let cols: Vec<Box<str>> = (0..cmd_args.cols)
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    // Shared outputs
    let prop_file = format!("{}.prop.parquet", output);
    sim.theta_kn.transpose().to_parquet_with_names(
        &prop_file,
        (Some(&cols), Some("cell")),
        None,
    )?;

    let batch_file = format!("{}.batch.gz", output);
    let batch_out: Vec<Box<str>> = sim
        .batch_membership
        .iter()
        .map(|&x| Box::from(x.to_string()))
        .collect();
    write_lines(&batch_out, &batch_file)?;

    // W_base
    let base_file = format!("{}.w_base.parquet", output);
    sim.w_base_kd
        .to_parquet_with_names(&base_file, (None, None), None)?;

    // Per-modality outputs
    let backend = cmd_args.backend.clone();
    for m in 0..mm {
        let suffix = format!(".m{}", m);
        let modality_output =
            apply_zip_flag(&format!("{}{}", output, suffix), cmd_args.zip, &backend);
        let (_, backend_file) = resolve_backend_file(&modality_output, Some(backend.clone()))?;

        let mtx_shape = (cmd_args.rows, cmd_args.cols, sim.triplets[m].len());

        // Dictionary
        let dict_file = format!("{}{}.dict.parquet", output, suffix);
        sim.beta_dk[m].to_parquet_with_names(&dict_file, (Some(&rows), Some("feature")), None)?;

        // Batch effects
        let ln_batch_file = format!("{}{}.ln_batch.parquet", output, suffix);
        sim.ln_delta_db[m].to_parquet_with_names(
            &ln_batch_file,
            (Some(&rows), Some("feature")),
            None,
        )?;

        // Delta (non-reference only)
        if m > 0 {
            let delta_file = format!("{}.w_delta{}.parquet", output, suffix);
            sim.w_delta_kd[m - 1].to_parquet_with_names(&delta_file, (None, None), None)?;

            let mask_file = format!("{}.spike_mask{}.parquet", output, suffix);
            sim.spike_mask_kd[m - 1].to_parquet_with_names(&mask_file, (None, None), None)?;
        }

        // MTX
        if cmd_args.save_mtx {
            let mtx_file = format!("{}{}.mtx.gz", output, suffix);
            let mut triplets = sim.triplets[m].clone();
            triplets.par_sort_by_key(|&(row, col, _)| (col, row));
            mtx_io::write_mtx_triplets(&triplets, cmd_args.rows, cmd_args.cols, &mtx_file)?;
        }

        // Sparse backend
        let mut data = create_sparse_from_triplets(
            &sim.triplets[m],
            mtx_shape,
            Some(&backend_file),
            Some(&backend),
        )?;

        data.register_row_names_vec(&rows);
        data.register_column_names_vec(&cols);

        finalize_zarr_output(&backend_file, &modality_output)?;
        info!("modality {}: {}", m, backend_file);
    }

    info!("done");
    Ok(())
}
