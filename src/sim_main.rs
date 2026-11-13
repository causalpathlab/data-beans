use clap::{Parser, Subcommand};

use data_beans::sim::deconv::{generate_convoluted_data, SimConvArgs};
use data_beans::sim::faba::{run_faba, FabaArgs};
use data_beans::sim::handlers::{
    fidelity_report_path, run_simulate, run_simulate_multimodal, RunSimulateArgs,
    RunSimulateMultimodalArgs,
};
use data_beans::sim::multiome::{fidelity_report_paths, run_multiome, MultiomeArgs};
use data_beans::sim::trajectory::run::{
    fidelity_report_path as trajectory_report_path, run_trajectory, TrajectoryArgs,
};

#[cfg(feature = "mcp")]
mod sim_mcp;

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if cli.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    if let Some(n) = cli.n_threads {
        if n == 0 {
            anyhow::bail!("--n-threads must be >= 1");
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok();
    }

    // A reference-mode run's fidelity reports, and whether `--view` asked
    // to open them.
    let (reports, view) = match &cli.commands {
        Commands::Topic(args) => {
            run_simulate(args)?;
            (fidelity_report_path(args).into_iter().collect(), args.view)
        }
        Commands::Trajectory(args) => {
            run_trajectory(args)?;
            (
                trajectory_report_path(args).into_iter().collect(),
                args.view,
            )
        }
        Commands::Bulk(args) => {
            generate_convoluted_data(args)?;
            (Vec::new(), false)
        }
        Commands::Multimodal(args) => {
            run_simulate_multimodal(args)?;
            (Vec::new(), false)
        }
        Commands::Multiome(args) => {
            run_multiome(args)?;
            (fidelity_report_paths(args), args.view)
        }
        Commands::Faba(args) => {
            run_faba(args)?;
            (Vec::new(), false)
        }
        Commands::View(args) => return view_reports(std::slice::from_ref(&args.report)),
        #[cfg(feature = "mcp")]
        Commands::Mcp => return sim_mcp::run_mcp(),
    };
    for path in &reports {
        eprintln!("fidelity report: {path} (open with `data-beans-sim view {path}`)");
    }
    if view {
        // The simulation's outputs are written: a view that cannot open is
        // a note, not a failure.
        if reports.is_empty() {
            eprintln!("no fidelity report to view: --view draws reference-mode runs only");
        } else if let Err(e) = view_reports(&reports) {
            eprintln!("{e}");
        }
    }
    Ok(())
}

/// Open the fidelity view of each report in turn (`q` moves on); an error
/// when there is no terminal to draw it on.
#[cfg(feature = "tui")]
fn view_reports(reports: &[String]) -> anyhow::Result<()> {
    anyhow::ensure!(
        data_beans::interactive::tui_available(),
        "the fidelity view needs a terminal; the reports are written: {reports:?}"
    );
    for path in reports {
        data_beans::interactive::sim_tui::view_fidelity(path)?;
    }
    Ok(())
}

#[cfg(not(feature = "tui"))]
fn view_reports(reports: &[String]) -> anyhow::Result<()> {
    anyhow::bail!(
        "data-beans-sim was built without the `tui` feature; the reports are written: {reports:?}"
    )
}

#[derive(clap::Args, Debug)]
struct ViewArgs {
    #[arg(help = "Fidelity report of a reference-mode run, `{output}.fidelity.json`")]
    report: String,
}

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Synthetic single-cell-like data generators for the data-beans ecosystem.",
    long_about = "data-beans-sim simulates sparse count matrices. Available structure:\n\
                  factor-model, lineage trajectory, CNV, multimodal, and topic-conditioned\n\
                  bulk-deconvolution.\n\
                  \n\
                  Outputs use the same zarr/h5 backend formats `data-beans` reads.\n\
                  Simulated and real datasets therefore share one toolchain."
)]
struct Cli {
    #[arg(short = 'v', long, global = true)]
    verbose: bool,

    #[arg(
        long = "n-threads",
        visible_aliases = ["threads", "num-threads"],
        global = true,
        value_name = "N",
        help = "Limit the number of CPU threads"
    )]
    n_threads: Option<usize>,

    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(
        about = "Log-normal topic factor model (with optional reference-conditioned sampling)",
        long_about = "Synthetic mode (no `--reference`):\n\
                      `Y(g,j) ~ Poisson( (depth/G) · δ(g,B(j)) · Σ_k β(g,k) θ(k,j) )`,\n\
            with an explicit log-space variance decomposition for both β and δ:\n\
                      log β(g,k) = σ_β · [√pve_topic · u_{g,k} + √(1−pve_topic) · v_g] − σ_β²/2\n\
                      log δ(g,b) =        √pve_batch · z_{g,b} + √(1−pve_batch) · w_g\n\
                      with u, v, z, w ~ N(0, 1) iid. `--beta-scale` controls σ_β.\n\
                      `pve_topic` and `pve_batch` are independent variance shares;\n\
                      both can be 1. `depth` is the **expected** library size. It is emergent,\n\
                      with no per-cell rescaling.\n\
                      \n\
                      Reference mode (`--reference <h5/zarr>`): Poisson-mixture marginals + latent Gaussian copula.\n\
                      Each gene keeps its reference count distribution F_g, a Poisson mixture:\n\
                      \x20 log λ = log μ̂_g + a_g·t − ln M1_g + b_g·ε − b_g²/2 + δ_{g,b} − c_g\n\
                      \x20 z*_g  = ρ_gs·z_s + √(1−ρ_gs²)·e_g,   y = F⁻¹_g(Φ(z*_g); rates × λ/μ̂_g)\n\
                      t is z-scored log(β·θ) per cell, ε iid N(0, 1); a_g = √pve_topic and\n\
                      b_g = √pve_noise, both capped so their log-variance stays within the gene's\n\
                      extra-Poisson log-variance; F_g is tempered to leave them room, and M1_g\n\
                      and c_g centre the means.\n\
                      z_s is one depth coordinate per cell, and ρ_gs each gene's latent coupling\n\
                      to library size in the reference. e is the HVGs' latent correlation beyond\n\
                      depth (iid off the HVGs), estimated on the reference's empirical counts.\n\
                      δ_{g,b} comes from the gene-gene copula factor at rank `--batch-rank`,\n\
                      on axes chosen by `--batch-program`, centred across batches per gene.\n\
                      With one batch there is no shift: the means are the reference's.\n\
                      \n\
                      `--depth` rescales every μ̂_g so the mean library size is `depth`; unset,\n\
                      it is the reference's. This follows the scDesign, scDesign2 and scDesign3\n\
                      lineage. The run writes `{output}.fidelity.json`; `--view` or\n\
                      `data-beans-sim view` draws it against the reference.\n\
                      \n\
                      See data-beans-sim/docs/topic.md for the full derivation."
    )]
    Topic(RunSimulateArgs),

    #[command(
        about = "Lineage tree / pseudotime trajectory (with optional reference-conditioned sampling)",
        long_about = "Cells sit on the branches of a lineage tree (`--lineages k`: a root that\n\
                      splits into k; or `--tree` parents), each at a pseudotime t ∈ [0, 1]\n\
                      in its branch. A share of genes (`--dynamic-genes`) follows a smooth\n\
                      random curve f_g(branch, t), continuous at branch points:\n\
                      \x20 f_g(b, t) = f_g(parent(b), 1) + Σ_k c_{g,b,k}·sin((k−½)πt)/((k−½)π)\n\
                      a smoothed Brownian motion down the tree (`--basis` terms per branch),\n\
                      standardised over the cells to f̃_g. Per cell j and gene g:\n\
                      \x20 log λ = log μ_g + √pve_traj·f̃_g(b_j,t_j) − c_g + √pve_noise·ε − pve_noise/2\n\
                      \x20         + √pve_batch·z_{g,B(j)} − c_{g,batch}\n\
                      with every c centring its term so the gene's mean stays μ_g.\n\
                      \n\
                      Synthetic mode: μ_g = (depth/G)·β_g, log-normal β, y ~ Poisson(λ).\n\
                      Reference mode (`--reference`): the sampler of `topic --reference`, with\n\
                      the trajectory in place of topics. Each gene keeps its reference Poisson\n\
                      mixture, depth coupling and latent copula; trajectory and noise are capped\n\
                      by its extra-Poisson log-variance. It writes `{output}.fidelity.json`.\n\
                      \n\
                      Ground truth: `{output}.cells.parquet` (branch, pseudotime, tree time,\n\
                      batch, lineage membership), `{output}.tree.tsv`, `{output}.genes.parquet`\n\
                      (dynamic, amplitude, centre, curve coefficients) and\n\
                      `{output}.curves.parquet` (each gene's log-mean term on a pseudotime grid).\n\
                      This follows scDesign3's pseudotime mode, Splatter's paths and PROSSTT."
    )]
    Trajectory(TrajectoryArgs),

    #[command(
        about = "Bulk (convoluted) data matrix from real SC reference (experimental)",
        long_about = "Synthesise bulk pseudo-samples from real single-cell counts.\n\
                      Cells are Dirichlet-mixed under a supplied per-cell topic membership.\n\
                      \n\
                      Each sample is an exact weighted sum of cells with known topic memberships.\n\
                      There is no extra noise model. So ground-truth fractions are recovered,\n\
                      up to whatever the cell-pool sampling implies.\n\
                      \n\
                      See data-beans-sim/docs/bulk.md for the full derivation."
    )]
    Bulk(SimConvArgs),

    #[command(
        about = "Multimodal count data with shared base + delta dictionaries",
        long_about = "Generate M count matrices from shared latent topics θ with modality-specific dictionaries:\n\
                      β_0(:,k) = softmax_g( W_base[k,:]            )    (reference modality)\n\
                      β_m(:,k) = softmax_g( W_base[k,:] + Δ_m[k,:] )    (m = 1..M-1)\n\
                      where W_base ~ N(0, base_scale²).\n\
                      In hierarchical mode the logits are stick-breaking-derived.\n\
                      Δ_m is sparse spike-and-slab.\n\
                      `n_delta_features` genes per topic carry an iid N(0, delta_scale²) perturbation;\n\
                      the rest are zero.\n\
                      \n\
                      Per-modality counts (batch effects independent per modality unless `--shared-batch-effects`):\n\
                      log δ_m(g,b) = √pve_batch · z_{g,b} + √(1−pve_batch) · w_g\n\
                      Y_m(g,j)    ~ Poisson( depth_m · δ_m(g,B(j)) · Σ_k β_m(g,k) θ(k,j) )\n\
                      depth_m is the **expected** library size for modality m. It is emergent,\n\
                      with no per-cell rescaling. Each β_m(:,k) sums to 1 over genes.\n\
                      So the cell-level total of (β·θ) sums to 1 deterministically,\n\
                      and `depth_m` directly sets E[lib(j) | m].\n\
                      \n\
                      See data-beans-sim/docs/multimodal.md for the full derivation."
    )]
    Multimodal(RunSimulateMultimodalArgs),

    #[command(
        about = "Paired ATAC + RNA simulator: cell types switch peaks on/off,\n\
                 genes inherit their enhancers (two-step), with peak-gene ground truth",
        long_about = "Synthetic mode, used when no `--reference-*` is given.\n\
                      It is a two-step generative model. Cis links are cell-type-INVARIANT.\n\
                      Cell-type-specific expression arises because upstream peaks switch on and off per cell type.\n\
                      \n\
                      Step 1 — ATAC from topics. Per cell j, topic mix θ_j (concentration\n\
                      --topic-concentration). Each peak p:\n\
                      \u{20} A_pj = base_p + σ·( √π_topic·T_p + √π_priv·P_p + √π_noise·N_p [+ batch] )\n\
                      \u{20} T = std(log(β_p·θ))  cell-type on/off;  P = peak-PRIVATE fluctuation\n\
                      \u{20} peak budget {topic, private, noise, batch} normalized to 1\n\
                      --invariant-causal-fraction sets a share of causal peaks\n\
                      to be topic-invariant, so purely private.\n\
                      Their links are cleanly recoverable.\n\
                      \n\
                      Step 2 — RNA conditional on enhancers.\n\
                      A linked gene inherits its causal peaks' signal,\n\
                      sig = √π_topic·T + √π_priv·P, via an invariant cis link:\n\
                      \u{20} E_gj = σ·( √pve_cis·std(Σ_{p∈M_g} sig_p) + √(1−pve_cis)·N_g [+ batch] ) The gene has no topic path of its own.\n\
                      Cell-type specificity propagates through its peaks.\n\
                      Unlinked genes are noise. Counts are Poisson(depth·softmax).\n\
                      \n\
                      Identifiability. Only an enhancer's PRIVATE part reaches its gene.\n\
                      Co-active bystanders share only T. So --pve-private is the recoverability dial.\n\
                      --pve-cis sets the gene's cis-dependence strength.\n\
                      Ground truth is M[G,P].\n\
                      \n\
                      Reference mode uses `--reference-rna` or `--reference-atac`.\n\
                      Each modality with a reference draws from its features' reference count\n\
                      distributions (Poisson mixtures), coupled by depth and a latent copula.\n\
                      A modality without one keeps its synthetic step.\n\
                      A normalized {topic, noise, batch} budget weights the log-rate;\n\
                      there is no cis term.\n\
                      Reference row counts override --n-genes and --n-peaks.\n\
                      \n\
                      See data-beans-sim/docs/multiome.md for the full derivation."
    )]
    Multiome(MultiomeArgs),

    #[command(
        about = "RNA modification + processing simulator (counts + m6A + A-to-I + APA)",
        long_about = "Generate sparse per-track count matrices.\n\
                      They are shaped like a `faba all` run.\n\
                      There is one .zarr.zip per RNA track: expression counts, m6A methylation,\n\
                      A-to-I editing, and alternative polyadenylation.\n\
                      Rows are named '{gene}/{track}/{detail}'.\n\
                      A full set of ground-truth parquets ships alongside.\n\
                      \n\
                      Substrate-level coupling is encoded.\n\
                      m6A and pA share a long-3'UTR substrate axis. A-to-I rides on Alu/dsRNA.\n\
                      Writer and editor programs are shared,\n\
                      so one cell-state topic can drive multiple tracks.\n\
                      \n\
                      Generative model summary:\n\
                      \u{20} cell state    θ_{k,j} ~ Dirichlet (shared with writer/editor activity)\n\
                      \u{20} substrate     s_g ~ N(0, I_S);  φ_{g,m} = Bernoulli(σ(s_g·w_m + b_m))\n\
                      \u{20} programs      A_{m,k} ~ N(0,σ_A²)·Bern(π_A);  z_{g,k} ~ N(0,σ_z²)·Bern(π_z)\n\
                      \u{20} mRNA pool     log μ_{g,j} = β_g + log((β_topic·θ)_{g,j}) + δ_{g,B(j)}\n\
                      \u{20} mod. rate     log r_{g,m,j} = base_{g,m} + φ_{g,m}·Σ_k z·A·θ + δ_m\n\
                      \u{20} mixture       α_{g,m} ~ Dir(α_mix·1_{C_m})\n\
                      \u{20} counts:       λ ∝ α · μ                      → Poisson, depth_count\n\
                      \u{20} modifiers:    λ ∝ α · μ · r  (only if φ=1)    → Poisson, depth_modifier\n\
                      \n\
                      See data-beans-sim/docs/faba.md for the full derivation."
    )]
    Faba(FabaArgs),

    #[command(
        about = "Draw a reference-mode simulation against its reference (terminal view)",
        long_about = "Open the fidelity view of a reference-mode simulation.\n\
                      `topic --reference`, `trajectory --reference` and `multiome --reference-*`\n\
                      write a report,\n\
                      `{output}.fidelity.json` (`{out}.{rna,atac}.fidelity.json` for multiome).\n\
                      The view shows a summary table, library-size and detected-gene histograms,\n\
                      per-gene mean, variance and zero fraction against the reference,\n\
                      and the gene-gene correlation of the densest copula genes.\n\
                      ←/→ or Tab switch panels, 1-7 jump, y changes the histogram scale, q quits."
    )]
    View(ViewArgs),

    #[cfg(feature = "mcp")]
    #[command(
        about = "Serve the simulators and their fidelity diagnosis as MCP tools over stdio",
        long_about = "Serve the simulators as Model Context Protocol tools over stdio, for an agent\n\
                      (`claude mcp add data-beans-sim -- data-beans-sim mcp`).\n\
                      simulate_from_reference fits a reference and simulates from it, returning\n\
                      the fidelity summary; fidelity reads a run's report, with the genes and\n\
                      pairs furthest from the reference; simulate runs any simulator with its\n\
                      command-line options; command_help lists them."
    )]
    Mcp,
}
