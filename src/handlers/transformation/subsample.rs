use crate::hdf5_io::*;
use crate::sparse_io::*;
use crate::zarr_io::{finalize_output, prepare_output};

use clap::Args;
use log::info;
use rand::rngs::SmallRng;
use rand::seq::index::sample;
use rand::SeedableRng;

use crate::handlers::explore::{off_screen, PlanArgs};
use crate::interactive::plan_tui::{plan_in, Control, Plan, PlanView, Planner, Setting};
use crate::interactive::ui::Tui;
use crate::zarr_io::output_path;

#[derive(Args, Debug)]
pub struct SubsampleArgs {
    /// data file -- `.zarr`, `.zarr.zip`, or `.h5`
    pub data_file: Box<str>,

    /// number of cells (columns) to keep at random
    #[arg(long)]
    pub cells: Option<usize>,

    /// fraction of cells to keep at random, in (0, 1]; ignored when --cells is set
    #[arg(long)]
    pub cell_frac: Option<f64>,

    /// number of genes/features (rows) to keep at random
    #[arg(long)]
    pub genes: Option<usize>,

    /// fraction of genes to keep at random, in (0, 1]; ignored when --genes is set
    #[arg(long)]
    pub gene_frac: Option<f64>,

    /// RNG seed for reproducible sampling
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// backend for the output file
    #[arg(long, value_enum, default_value = "zarr")]
    pub backend: SparseIoBackend,

    /// output header: {output}.zarr.zip by default; pass --no-zip to keep a {output}.zarr directory
    #[arg(short, long)]
    pub output: Box<str>,

    /// keep a `.zarr` directory instead of producing a `.zarr.zip` archive
    #[arg(long = "no-zip", default_value_t = true, action = clap::ArgAction::SetFalse)]
    pub zip: bool,

    #[command(flatten)]
    pub view: PlanArgs,
}

/// Randomly subsample cells and/or genes into a new, smaller backend.
///
/// Only the selected columns are read, so the cost scales with the *output*
/// size rather than the input — handy for carving quick test/demo datasets
/// out of a large backend. An unset dimension keeps all of its entries.
pub fn run_subsample(args: &SubsampleArgs) -> anyhow::Result<()> {
    let session = args.view.session("subsample")?;
    let (backend_in, file_in) = resolve_backend_file(&args.data_file, None)?;
    let data = open_sparse_matrix(&file_in, &backend_in)?;

    let nrow = data
        .num_rows()
        .ok_or_else(|| anyhow::anyhow!("backend has no `nrow`"))?;
    let ncol = data
        .num_columns()
        .ok_or_else(|| anyhow::anyhow!("backend has no `ncol`"))?;

    let mut n_cells = resolve_target(args.cells, args.cell_frac, ncol, "cells")?;
    let mut n_genes = resolve_target(args.genes, args.gene_frac, nrow, "genes")?;
    let mut seed = args.seed;
    let mut tui = session.as_ref().map(Tui::new);
    if let Some(tui) = tui.as_mut() {
        let size = Size {
            nrow,
            ncol,
            nnz: data.num_non_zeros(),
        };
        match choose(tui, args, size, (n_cells, n_genes, seed))? {
            Some(c) => (n_cells, n_genes, seed) = c,
            None => {
                info!("subsample: cancelled; nothing written");
                return Ok(());
            }
        }
    }
    if n_cells.is_none() && n_genes.is_none() {
        anyhow::bail!("specify at least one of --cells / --cell-frac / --genes / --gene-frac");
    }
    off_screen(tui.as_mut(), || {
        write_subsample(args, &*data, (nrow, ncol), (n_cells, n_genes), seed)
    })
}

/// The input's size, for the plan.
#[derive(Debug, Clone, Copy)]
struct Size {
    nrow: usize,
    ncol: usize,
    nnz: Option<usize>,
}

/// Cells and genes to keep (`None`: all) and the seed.
type Draw = (Option<usize>, Option<usize>, u64);

/// A count from a setting: whole, at least 1, at most `total`.
fn count(s: &Setting, what: &str, total: usize) -> anyhow::Result<Option<usize>> {
    match s.number() {
        None => Ok(None),
        Some(v) if v.fract() == 0.0 && v >= 1.0 => Ok(Some((v as usize).min(total))),
        Some(v) => anyhow::bail!("{what} is a whole number, at least 1; not {v}"),
    }
}

fn draw_of(s: &[Setting], size: Size) -> anyhow::Result<Draw> {
    let cells = count(&s[0], "cells (n)", size.ncol)?;
    let genes = count(&s[1], "genes (g)", size.nrow)?;
    let seed = match s[2].number() {
        Some(v) if v.fract() == 0.0 && v >= 0.0 => v as u64,
        Some(v) => anyhow::bail!("the seed (s) is a whole number; not {v}"),
        None => 42,
    };
    Ok((cells, genes, seed))
}

/// What `draw` keeps of an input of `size`.
fn plan_of(output: &str, size: Size, (cells, genes, seed): Draw) -> Plan {
    let (rows, cols) = (genes.unwrap_or(size.nrow), cells.unwrap_or(size.ncol));
    let share = rows as f64 / size.nrow.max(1) as f64 * cols as f64 / size.ncol.max(1) as f64;
    let entries =
        |n: Option<usize>, f: f64| n.map_or("?".into(), |n| format!("{:.0}", n as f64 * f));
    Plan {
        columns: ["", "rows", "columns", "entries"]
            .map(String::from)
            .to_vec(),
        rows: vec![
            vec![
                "input".into(),
                size.nrow.to_string(),
                size.ncol.to_string(),
                entries(size.nnz, 1.0),
            ],
            vec![
                output.into(),
                rows.to_string(),
                cols.to_string(),
                format!("≈ {}", entries(size.nnz, share)),
            ],
        ],
        notes: vec![format!(
            "{rows} of {} rows and {cols} of {} columns at random, seed {seed}; \
             the entries scale with both",
            size.nrow, size.ncol
        )],
        blocked: (cells.is_none() && genes.is_none())
            .then(|| "set the cells (n) or the genes (g) to keep".into()),
    }
}

/// Show the plan of `draw` on `tui`; the draw it ended on, or `None` when
/// cancelled.
fn choose(
    tui: &mut Tui<'_>,
    args: &SubsampleArgs,
    size: Size,
    (cells, genes, seed): Draw,
) -> anyhow::Result<Option<Draw>> {
    let output = output_path(&args.output, args.zip, &args.backend);
    let controls = vec![
        Control::number('n', "cells", cells.map(|n| n as f64)),
        Control::number('g', "genes", genes.map(|n| n as f64)),
        Control::number('s', "seed", Some(seed as f64)),
    ];
    let planner: Planner =
        Box::new(move |s: &[Setting]| Ok(plan_of(&output, size, draw_of(s, size)?)));
    let view = PlanView::new("subsample", &args.data_file, "write", controls, planner);
    match plan_in(tui, view)? {
        Some(s) => draw_of(&s, size).map(Some),
        None => Ok(None),
    }
}

/// Write the subsample `draw` picks.
fn write_subsample(
    args: &SubsampleArgs,
    data: &dyn SparseIo<IndexIter = Vec<usize>>,
    (nrow, ncol): (usize, usize),
    (n_cells, n_genes): (Option<usize>, Option<usize>),
    seed: u64,
) -> anyhow::Result<()> {
    let mut rng = SmallRng::seed_from_u64(seed);

    // Sample ascending index sets so the output preserves the original order.
    let cell_idx = sample_sorted(&mut rng, ncol, n_cells);
    let gene_idx = sample_sorted(&mut rng, nrow, n_genes);

    info!(
        "subsampling to {} genes x {} cells (seed {})",
        gene_idx.len(),
        cell_idx.len(),
        seed
    );

    let out_nrow = gene_idx.len();
    let out_ncol = cell_idx.len();

    let row_names_all = data.row_names()?;
    let col_names_all = data.column_names()?;
    let out_row_names: Vec<Box<str>> = gene_idx.iter().map(|&i| row_names_all[i].clone()).collect();
    let out_col_names: Vec<Box<str>> = cell_idx.iter().map(|&i| col_names_all[i].clone()).collect();

    let (effective_output, backend_out, file_out) =
        prepare_output(&args.output, args.backend.clone(), args.zip)?;

    // Streamed: the survivors never sit in memory at once. `sample_sorted`
    // returns ascending indices, which is what lets the row renumbering stay
    // monotone and the columns keep their order without a per-column sort.
    let row_filter = (gene_idx.len() < nrow).then_some(gene_idx.as_slice());
    let (_, _, nnz) = crate::column_subset::stream_column_selection(
        data,
        &cell_idx,
        row_filter,
        &out_row_names,
        &out_col_names,
        file_out.as_ref(),
        &backend_out,
    )?;

    let final_path = finalize_output(&file_out, &effective_output)?;
    info!(
        "done: {} ({} genes x {} cells, {} non-zeros)",
        final_path, out_nrow, out_ncol, nnz
    );
    Ok(())
}

/// Resolve a target keep-count from an explicit count or a fraction. `None`
/// means "keep all". A count is capped at `total`; a fraction rounds to at
/// least one.
fn resolve_target(
    count: Option<usize>,
    frac: Option<f64>,
    total: usize,
    label: &str,
) -> anyhow::Result<Option<usize>> {
    if let Some(k) = count {
        if k == 0 {
            anyhow::bail!("--{} must be >= 1", label);
        }
        Ok(Some(k.min(total)))
    } else if let Some(f) = frac {
        if !(f > 0.0 && f <= 1.0) {
            anyhow::bail!("--{}-frac must be in (0, 1]", label);
        }
        Ok(Some(((f * total as f64).round() as usize).clamp(1, total)))
    } else {
        Ok(None)
    }
}

/// Draw `target` distinct indices from `0..total` (or all of them when
/// `target` is `None` or `>= total`), returned in ascending order.
fn sample_sorted(rng: &mut SmallRng, total: usize, target: Option<usize>) -> Vec<usize> {
    match target {
        Some(k) if k < total => {
            let mut v = sample(rng, total, k).into_vec();
            v.sort_unstable();
            v
        }
        _ => (0..total).collect(),
    }
}

#[cfg(all(test, feature = "ndarray"))]
mod tests;
