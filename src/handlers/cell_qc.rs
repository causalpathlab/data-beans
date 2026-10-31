//! `data-beans qc`: senna's and pinto's cell QC (`data_beans::qc_lib`, the
//! same flags) run on one backend, or on several pooled into one, its
//! outcome summarized on stdout, its figure written with --figure, and the
//! cells (and features) it keeps written to a new backend when asked. The
//! nnz cutoffs that `squeeze` took are its floors: --qc-min-cell-nnz on
//! cells, --qc-feature-min-cells on genes. It never rewrites its input.
//!
//! On a terminal it opens its view first (`data_beans::interactive::qc_tui`):
//! the rules move on their histograms, Enter writes the kept cells (asking
//! where, without -o) and q cancels. `--batch-process`, or no terminal, runs straight through with
//! the flags given.
//!
//! It works in the library's types, since `qc_lib` does: the binary's own
//! copies of the modules are other types.

use crate::handlers::explore::{off_screen, open_session};
use crate::handlers::merging::{batch_file, run_merge_backend, same_file, MergeBackendArgs};
use clap::Args;
use data_beans::figure::qc::{figure, shown};
use data_beans::figure::save;
use data_beans::hdf5_io::resolve_backend_file;
use data_beans::interactive::qc_tui::{choose_rules, shell_word, stem, Decision};
use data_beans::interactive::ui::Tui;
use data_beans::qc_lib::{
    compute_qc_metrics, qc_report, write_qc_report, QcArgs, QcConfig, QcMetrics,
};
use data_beans::qc_rules::{CellRules, Rule};
use data_beans::sparse_io::{open_sparse_matrix, SparseIo, SparseIoBackend};
use data_beans::sparse_io_vector::SparseIoVec;
use data_beans::zarr_io::{finalize_output, output_target, prepare_output};
use legume_numeric::matrix::common_io::{basename, read_lines, write_lines};
use log::info;
use std::sync::Arc;

#[derive(Args, Debug)]
pub struct CellQcArgs {
    /// data files -- `.zarr`, `.zarr.zip`, or `.h5`; several are pooled
    /// (merged, as `merge-backend` does) and QC'd as one
    #[arg(required = true, value_delimiter = ',')]
    pub data_files: Vec<Box<str>>,

    /// with several data files, keep only the rows (genes) every one has;
    /// otherwise the pool has them all
    #[arg(long, default_value_t = false)]
    pub common_rows: bool,

    #[command(flatten)]
    pub qc: QcArgs,

    /// output: the cells (and features) QC keeps, as a new backend, never
    /// the input; without it, the view asks where on Enter, and
    /// --batch-process only prints the summary
    #[arg(short, long)]
    pub output: Option<Box<str>>,

    /// backend for the output file
    #[arg(long, value_enum, default_value = "zarr")]
    pub backend: SparseIoBackend,

    /// keep a `.zarr` directory instead of producing a `.zarr.zip` archive
    #[arg(long = "no-zip", default_value_t = true, action = clap::ArgAction::SetFalse)]
    pub zip: bool,

    /// cells per rayon job (omit for auto-scaling by feature count)
    #[arg(long)]
    pub block_size: Option<usize>,

    /// write the QC figure under this name: `NAME.pdf` and `NAME.png`, or
    /// the one file NAME names when it ends in .pdf, .png or .svg
    #[arg(long)]
    pub figure: Option<Box<str>>,

    /// run straight through with the flags given: no view, even on a
    /// terminal
    #[arg(long, default_value_t = false)]
    pub batch_process: bool,

    #[command(flatten)]
    pub session: legume_tui::cli::SessionArgs,

    /// Whether --qc-feature-min-cells was given on the command line, so
    /// --qc-auto-cutoff leaves the gene rule as given (0 included).
    #[arg(skip)]
    pub gene_rule_given: bool,
}

/// The mito pattern `qc` starts from, where senna's and pinto's QC has
/// none; an empty `--qc-mito-pattern=` turns it off.
pub const MITO: &str = "(?i)^MT-";

/// The QC config the flags ask for: `--no-qc` leaves nothing to do, and an
/// empty `--qc-mito-pattern=` turns the mito metric off.
pub fn qc_config(qc: &QcArgs) -> anyhow::Result<QcConfig> {
    let mut qc = qc.clone();
    if qc.qc_mito_pattern.as_deref() == Some("") {
        qc.qc_mito_pattern = None;
    }
    qc.to_config()
        .ok_or_else(|| anyhow::anyhow!("--no-qc leaves nothing to do"))
}

/// Whether to open the view: on a terminal, unless `--batch-process`.
fn opens_view(args: &CellQcArgs) -> bool {
    data_beans::interactive::opens_view("qc", args.batch_process)
}

/// Several data files merged into one backend, in a folder of its own
/// that goes with it.
struct Pool {
    _dir: tempfile::TempDir,
    file: String,
}

/// The data files pooled, when there are several: merged as `merge-backend`
/// merges them (all rows, or `--common-rows`), into a hidden folder here,
/// which a big pool needs room for.
fn pool(args: &CellQcArgs) -> anyhow::Result<Option<Pool>> {
    if args.data_files.len() < 2 {
        return Ok(None);
    }
    let dir = tempfile::Builder::new()
        .prefix(".data-beans-qc-")
        .tempdir_in(".")?;
    let file = dir.path().join("pool.zarr").to_string_lossy().into_owned();
    info!("pooling {} files ...", args.data_files.len());
    run_merge_backend(&MergeBackendArgs {
        data_files: args.data_files.clone(),
        backend: crate::sparse_io::SparseIoBackend::Zarr,
        output: file.clone().into(),
        zip: false,
        common_rows: args.common_rows,
        do_squeeze: false,
        row_nnz_cutoff: 0,
        column_nnz_cutoff: 0,
        block_size: args.block_size,
        view: crate::handlers::explore::PlanArgs::batch(),
    })?;
    Ok(Some(Pool { _dir: dir, file }))
}

pub fn run_cell_qc(args: &CellQcArgs) -> anyhow::Result<()> {
    let view = opens_view(args);
    let session = view
        .then(|| open_session("qc", &args.session))
        .transpose()?;
    let mut tui = session.as_ref().map(Tui::new);

    // Whatever was asked, the input is never written over.
    if let Some(out) = &args.output {
        refuse_an_input(args, out)?;
    }

    let mut cfg = qc_config(&args.qc)?;
    let pool = off_screen(tui.as_mut(), || pool(args))?;
    let input = pool
        .as_ref()
        .map_or(&*args.data_files[0], |p| p.file.as_str());
    let title = args
        .data_files
        .iter()
        .map(|f| basename(f).unwrap_or_else(|_| f.clone()).to_string())
        .collect::<Vec<_>>()
        .join(" + ");
    let (backend, file) = resolve_backend_file(input, None)?;
    let matrix: Arc<dyn SparseIo<IndexIter = Vec<usize>>> =
        Arc::from(open_sparse_matrix(&file, &backend)?);
    let mut data = SparseIoVec::new();
    data.push(matrix.clone(), None)?;
    // Cells per gene: for the gene rule, the view's and the figure's gene
    // panel, the gene trough and its histogram.
    let gene_cells = view
        || args.figure.is_some()
        || cfg.feature_min_cells > 0
        || cfg.auto_cell_cutoff
        || cfg.qc_histogram;
    let metrics = off_screen(tui.as_mut(), || {
        compute_qc_metrics(&data, &cfg, args.block_size, gene_cells)
    })?;
    // --qc-auto-cutoff puts the gene rule at the trough of cells per gene,
    // as it puts the cell floor at the trough of genes per cell; a gene
    // rule given stands, 0 (off) included.
    if cfg.auto_cell_cutoff && !args.gene_rule_given {
        let cut = metrics
            .feature_n_cells
            .as_deref()
            .and_then(crate::qc::suggest_nnz_cutoff);
        if let Some(cut) = cut {
            info!("gene rule at the trough of cells per gene: {cut}");
            cfg.feature_min_cells = cut;
        }
    }

    // The command that decides the same cells without the view, said once
    // the screen is back.
    let mut same = None;
    let mut output = args.output.as_ref().map(|o| o.to_string());
    let cfg = match tui.as_mut() {
        Some(tui) => {
            let suggest = match &pool {
                Some(_) => "pooled.qc".to_string(),
                None => format!("{}.qc", stem(&args.data_files[0])),
            };
            let histogram = cfg.qc_histogram;
            let rules = CellRules::new(metrics.clone(), cfg);
            let (rules, decision, chosen) = choose_rules(tui, rules, &title, output, &suggest)?;
            if decision == Decision::Cancel {
                info!("qc cancelled: nothing written");
                return Ok(());
            }
            output = chosen;
            let mut words = vec!["data-beans".to_string(), "qc".into()];
            words.push(shell_word(&args.data_files.join(",")));
            if pool.is_some() && args.common_rows {
                words.push("--common-rows".into());
            }
            words.extend(rules.flags().iter().map(|f| shell_word(f)));
            if let Some(out) = &output {
                words.extend(["-o".into(), shell_word(out)]);
                if args.backend != SparseIoBackend::Zarr {
                    words.extend(["--backend".into(), "hdf5".into()]);
                }
                if !args.zip {
                    words.push("--no-zip".into());
                }
            }
            if histogram {
                words.push("--qc-histogram".into());
            }
            words.push("--batch-process".into());
            same = Some(words.join(" "));
            // The rules hold no printing; the histogram asked for stands.
            let mut cfg = rules.config().clone();
            cfg.qc_histogram = histogram;
            cfg
        }
        None => cfg,
    };
    let output = output.as_deref();
    off_screen(tui.as_mut(), || {
        finish(
            args,
            &title,
            matrix.as_ref(),
            metrics,
            cfg,
            output,
            pool.as_ref(),
        )
    })?;
    drop(tui);
    if let Some(same) = same {
        eprintln!("the same without the view:\n  {same}");
    }
    Ok(())
}

/// After the view (or without one): the figure, the per-cell report, the
/// summary on stdout, and the kept cells written to `-o`.
fn finish(
    args: &CellQcArgs,
    title: &str,
    matrix: &dyn SparseIo<IndexIter = Vec<usize>>,
    metrics: QcMetrics,
    cfg: QcConfig,
    output: Option<&str>,
    pool: Option<&Pool>,
) -> anyhow::Result<()> {
    // The rules the figure is drawn from and the summary says.
    let rules = CellRules::new(metrics.clone(), cfg.clone());
    if let Some(name) = &args.figure {
        let paths = save(&figure(&rules, title), name)?;
        info!("wrote the QC figure: {}", paths.join(", "));
    }
    // After the cells' histogram, which qc_lib prints with the report,
    // the genes'.
    let gene_cells = cfg
        .qc_histogram
        .then(|| metrics.feature_n_cells.clone())
        .flatten();
    let report = qc_report(metrics, &cfg);
    if let Some(cells) = gene_cells {
        let suggested = crate::qc::suggest_nnz_cutoff(&cells);
        let cut = cfg.feature_min_cells;
        crate::qc::eprint_nnz_summary("Gene", "cells", &cells, cut, suggested);
    }
    let report = &report;
    // The cell names, read only when something is written with them.
    let cells = match args.qc.qc_report.is_some() || output.is_some() {
        true => matrix.column_names()?,
        false => vec![],
    };
    if let Some(path) = &args.qc.qc_report {
        write_qc_report(path, &cells, report)?;
        info!("wrote the per-cell report: {path}");
    }

    let keep_cells = report.emit_idx_unmasked();
    let keep_rows: Vec<usize> = (0..report.feature_keep.len())
        .filter(|&r| report.feature_keep[r])
        .collect();
    let summary = summary(&rules, keep_cells.len(), keep_rows.len());
    write_lines(&summary, "stdout")?;

    let Some(output) = output else {
        return Ok(());
    };
    anyhow::ensure!(!keep_cells.is_empty(), "QC keeps no cells; nothing written");
    // The output is cleared before the input is read: it must not be it
    // (the view may have asked for this one).
    refuse_an_input(args, output)?;
    let rows = matrix.row_names()?;
    let out_rows: Vec<Box<str>> = keep_rows.iter().map(|&r| rows[r].clone()).collect();
    let out_cells: Vec<Box<str>> = keep_cells.iter().map(|&c| cells[c].clone()).collect();
    let (effective, backend_out, file_out) =
        prepare_output(output, args.backend.clone(), args.zip)?;
    let row_filter = (keep_rows.len() < rows.len()).then_some(keep_rows.as_slice());
    let (_, _, nnz) = data_beans::column_subset::stream_column_selection(
        matrix,
        &keep_cells,
        row_filter,
        &out_rows,
        &out_cells,
        &file_out,
        &backend_out,
    )?;
    let written = finalize_output(&file_out, &effective)?;
    // A pool's batch labels, for the cells kept: one a column, as `merge`
    // writes them.
    if let Some(pool) = pool {
        let labels = read_lines(&batch_file(&pool.file))?;
        let kept: Vec<Box<str>> = keep_cells.iter().map(|&c| labels[c].clone()).collect();
        let path = batch_file(&effective);
        write_lines(&kept, &path)?;
        info!("wrote the batch labels: {path}");
    }
    info!(
        "done: {written} ({} features x {} cells, {nnz} non-zeros)",
        out_rows.len(),
        out_cells.len()
    );
    Ok(())
}

/// Refuse an output that is, or writes over, one of the inputs.
fn refuse_an_input(args: &CellQcArgs, output: &str) -> anyhow::Result<()> {
    let target = output_target(output, args.backend.clone(), args.zip)?;
    for p in [&target.effective, &target.working, &target.found] {
        if let Some(input) = args.data_files.iter().find(|i| same_file(i, p)) {
            anyhow::bail!("the output {p} is the input {input}: give -o another name");
        }
    }
    Ok(())
}

/// The outcome, as `key:\tvalue` lines: the counts kept and dropped, and for
/// each metric its median, the edge its MAD band held it to and how many
/// cells fell past it (a cell can fall past several), and its cap. The
/// bands are the rules' own, as the view and the figure show them.
fn summary(rules: &CellRules, kept_cells: usize, kept_features: usize) -> Vec<Box<str>> {
    let v = rules.verdicts();
    let cfg = rules.config();
    let mut lines: Vec<Box<str>> = vec![];
    let mut say = |key: &str, value: String| lines.push(format!("{key}:\t{value}").into());
    let near_empty = v.near_empty.iter().filter(|&&e| e).count();
    say("cells", v.train_keep.len().to_string());
    say("cells_kept", kept_cells.to_string());
    say("near_empty", near_empty.to_string());
    // Dropped as outliers, apart from the near-empty: what the figure shows.
    let outliers = (v.near_empty.iter().zip(&v.train_keep))
        .filter(|&(&e, &k)| !e && !k)
        .count();
    say("outliers", outliers.to_string());
    say("features", v.feature_keep.len().to_string());
    say("features_kept", kept_features.to_string());

    for &rule in rules.rules().iter().filter(|r| r.on_cells()) {
        let (name, banded, cap) = match rule {
            Rule::Genes => ("n_genes", cfg.mad_on_n_genes, None),
            Rule::Counts => ("total_counts", cfg.mad_on_counts, None),
            Rule::Mito => ("mito_frac", cfg.mad_on_mito, cfg.mito_max_frac),
            Rule::Ribo => ("ribo_frac", false, cfg.ribo_max_frac),
            Rule::GeneCells => continue,
        };
        let values = rules.values(rule);
        // Cells above the near-empty floor past `edge`.
        let past = |edge: f32| {
            let within = |x: f32| {
                if rule.keeps_above() {
                    x >= edge
                } else {
                    x <= edge
                }
            };
            values
                .iter()
                .zip(&v.near_empty)
                .filter(|&(&x, &e)| !e && !within(x))
                .count()
        };
        let band = rules.band(rule);
        if let Some(b) = band {
            say(&format!("{name}.median"), shown(rule, b.median));
        }
        if let Some(b) = band.filter(|_| banded) {
            let (bound, side, edge) = match rule.keeps_above() {
                true => ("min", "below", b.lo),
                false => ("max", "above", b.hi),
            };
            say(&format!("{name}.{bound}"), shown(rule, edge));
            say(&format!("{name}.{side}"), past(edge).to_string());
        }
        if let Some(cap) = cap {
            say(&format!("{name}.cap"), cap.to_string());
            say(&format!("{name}.above_cap"), past(cap).to_string());
        }
    }
    lines
}
