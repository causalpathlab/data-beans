//! `data-beans qc`: senna's and pinto's cell QC (`data_beans::qc_lib`, the
//! same flags) run on one backend, its outcome summarized on stdout, and the
//! cells (and features) it keeps written to a new backend when asked.
//!
//! It works in the library's types, since `qc_lib` does: the binary's own
//! copies of the modules are other types.

use clap::Args;
use data_beans::hdf5_io::resolve_backend_file;
use data_beans::qc_lib::{
    compute_qc, robust_band, write_qc_report, QcArgs, QcConfig, QcReport, RobustBand,
};
use data_beans::sparse_io::{open_sparse_matrix, SparseIo, SparseIoBackend};
use data_beans::sparse_io_vector::SparseIoVec;
use data_beans::zarr_io::{finalize_output, prepare_output};
use legume_numeric::matrix::common_io::write_lines;
use log::info;
use std::sync::Arc;

#[derive(Args, Debug)]
pub struct CellQcArgs {
    /// data file -- `.zarr`, `.zarr.zip`, or `.h5`
    pub data_file: Box<str>,

    #[command(flatten)]
    pub qc: QcArgs,

    /// output: the cells (and features) QC keeps, as a new backend; without
    /// it, only the summary is printed
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
}

/// The mito pattern `qc` and `plot qc` start from, where senna's and
/// pinto's QC has none; an empty `--qc-mito-pattern=` turns it off.
pub const MITO: &str = "(?i)^MT-";

/// A backend's QC: what [`compute_qc`] made of it, under which config.
pub struct CellQc {
    pub matrix: Arc<dyn SparseIo<IndexIter = Vec<usize>>>,
    pub cfg: QcConfig,
    pub report: QcReport,
}

impl CellQc {
    pub fn compute(
        data_file: &str,
        qc: &QcArgs,
        block_size: Option<usize>,
    ) -> anyhow::Result<Self> {
        let mut qc = qc.clone();
        if qc.qc_mito_pattern.as_deref() == Some("") {
            qc.qc_mito_pattern = None;
        }
        let cfg = qc
            .to_config()
            .ok_or_else(|| anyhow::anyhow!("--no-qc leaves nothing to do"))?;
        let (backend, file) = resolve_backend_file(data_file, None)?;
        let matrix: Arc<dyn SparseIo<IndexIter = Vec<usize>>> =
            Arc::from(open_sparse_matrix(&file, &backend)?);
        let mut data = SparseIoVec::new();
        data.push(matrix.clone(), None)?;
        let report = compute_qc(&data, &cfg, block_size)?;
        Ok(Self {
            matrix,
            cfg,
            report,
        })
    }

    /// The per-cell metrics QC measured, in a fixed order.
    pub fn metrics(&self) -> Vec<Metric<'_>> {
        let (r, cfg) = (&self.report, &self.cfg);
        let mut v = vec![
            Metric::counts("n_genes", "detected genes", &r.n_genes, cfg.mad_on_n_genes),
            Metric::counts(
                "total_counts",
                "total counts",
                &r.total_counts,
                cfg.mad_on_counts,
            ),
        ];
        if let Some(f) = &r.mito_frac {
            v.push(Metric::fraction(
                ("mito_frac", "mitochondrial fraction"),
                f,
                cfg.mad_on_mito,
                cfg.mito_max_frac,
            ));
        }
        if let Some(f) = &r.ribo_frac {
            v.push(Metric::fraction(
                ("ribo_frac", "ribosomal fraction"),
                f,
                false,
                cfg.ribo_max_frac,
            ));
        }
        v
    }

    /// The robust band of `m`, on its own scale, as qc_lib fits it: over
    /// the cells above the near-empty floor.
    pub fn band(&self, m: &Metric) -> Option<RobustBand> {
        let consider: Vec<bool> = self.report.near_empty.iter().map(|&e| !e).collect();
        robust_band(m.values, self.cfg.n_mads, m.count, Some(&consider)).map(|b| b.raw(m.count))
    }

    /// The edge of `m`'s band QC drops cells past, when it does: its floor
    /// for a count, its ceiling for a fraction.
    pub fn edge(&self, m: &Metric) -> Option<f32> {
        let band = self.band(m).filter(|_| m.banded)?;
        Some(if m.count { band.lo } else { band.hi })
    }
}

/// A per-cell metric: a count, banded on the log1p scale and held from
/// below, or a fraction, banded as is and held from above (and capped).
pub struct Metric<'a> {
    pub name: &'static str,
    /// What a figure calls it.
    pub label: &'static str,
    pub values: &'a [f32],
    pub count: bool,
    /// Whether QC drops cells outside its band.
    pub banded: bool,
    pub cap: Option<f32>,
}

impl<'a> Metric<'a> {
    fn counts(name: &'static str, label: &'static str, values: &'a [f32], banded: bool) -> Self {
        Self {
            name,
            label,
            values,
            count: true,
            banded,
            cap: None,
        }
    }

    fn fraction(
        (name, label): (&'static str, &'static str),
        values: &'a [f32],
        banded: bool,
        cap: Option<f32>,
    ) -> Self {
        Self {
            name,
            label,
            values,
            count: false,
            banded,
            cap,
        }
    }

    /// What its edge is, and the side of it a dropped cell is on.
    pub fn bound(&self) -> (&'static str, &'static str) {
        if self.count {
            ("min", "below")
        } else {
            ("max", "above")
        }
    }

    /// Whether `v` is on the kept side of `edge`.
    pub fn within(&self, v: f32, edge: f32) -> bool {
        if self.count {
            v >= edge
        } else {
            v <= edge
        }
    }

    /// `v` shown past the f32 noise of the log1p round trip: counts to a
    /// tenth, fractions to four places.
    fn shown(&self, v: f32) -> String {
        if self.count {
            format!("{v:.1}")
        } else {
            format!("{v:.4}")
        }
    }
}

pub fn run_cell_qc(args: &CellQcArgs) -> anyhow::Result<()> {
    let qc = CellQc::compute(&args.data_file, &args.qc, args.block_size)?;
    let (matrix, report) = (&qc.matrix, &qc.report);
    // The cell names, read only when something is written with them.
    let cells = match args.qc.qc_report.is_some() || args.output.is_some() {
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
    let summary = summary(&qc, keep_cells.len(), keep_rows.len());
    write_lines(&summary, "stdout")?;

    let Some(output) = &args.output else {
        return Ok(());
    };
    let rows = matrix.row_names()?;
    let out_rows: Vec<Box<str>> = keep_rows.iter().map(|&r| rows[r].clone()).collect();
    let out_cells: Vec<Box<str>> = keep_cells.iter().map(|&c| cells[c].clone()).collect();
    let (effective, backend_out, file_out) =
        prepare_output(output, args.backend.clone(), args.zip)?;
    let row_filter = (keep_rows.len() < rows.len()).then_some(keep_rows.as_slice());
    let (_, _, nnz) = data_beans::column_subset::stream_column_selection(
        &**matrix,
        &keep_cells,
        row_filter,
        &out_rows,
        &out_cells,
        &file_out,
        &backend_out,
    )?;
    let written = finalize_output(&file_out, &effective)?;
    info!(
        "done: {written} ({} features x {} cells, {nnz} non-zeros)",
        out_rows.len(),
        out_cells.len()
    );
    Ok(())
}

/// The outcome, as `key:\tvalue` lines: the counts kept and dropped, and for
/// each metric its median, the edge QC held it to, and how many cells fell
/// past it (a cell can fall past several).
fn summary(qc: &CellQc, kept_cells: usize, kept_features: usize) -> Vec<Box<str>> {
    let report = &qc.report;
    let mut lines: Vec<Box<str>> = vec![];
    let mut say = |key: &str, value: String| lines.push(format!("{key}:\t{value}").into());
    let near_empty = report.near_empty.iter().filter(|&&e| e).count();
    say("cells", report.train_keep.len().to_string());
    say("cells_kept", kept_cells.to_string());
    say("near_empty", near_empty.to_string());
    say("outliers", report.n_cells_dropped.to_string());
    say("features", report.feature_keep.len().to_string());
    say("features_kept", kept_features.to_string());

    // Cells above the near-empty floor past `edge`.
    let past = |m: &Metric, edge: f32| {
        m.values
            .iter()
            .zip(&report.near_empty)
            .filter(|&(&v, &e)| !e && !m.within(v, edge))
            .count()
    };
    for m in qc.metrics() {
        let name = m.name;
        if let Some(band) = qc.band(&m) {
            say(&format!("{name}.median"), m.shown(band.median));
        }
        if let Some(edge) = qc.edge(&m) {
            let (bound, side) = m.bound();
            say(&format!("{name}.{bound}"), m.shown(edge));
            say(&format!("{name}.{side}"), past(&m, edge).to_string());
        }
        if let Some(cap) = m.cap {
            say(&format!("{name}.cap"), cap.to_string());
            say(&format!("{name}.above_cap"), past(&m, cap).to_string());
        }
    }
    lines
}
