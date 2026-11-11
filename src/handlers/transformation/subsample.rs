use crate::hdf5_io::*;
use crate::sparse_io::*;
use crate::zarr_io::{finalize_output, prepare_output};

use clap::Args;
use legume_numeric::matrix::common_io::write_lines;
use log::{info, warn};
use rand::rngs::SmallRng;
use rand::seq::index::sample;
use rand::SeedableRng;

use crate::handlers::explore::{off_screen, PlanArgs};
use crate::handlers::merging::batch_file;
use crate::interactive::plan_tui::{plan_in, Control, Plan, PlanView, Planner, Setting};
use crate::interactive::ui::Tui;
use crate::zarr_io::output_path;

mod sketch;
use sketch::Sketch;

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

    /// draw the cells within each group of this table, with --per-group or
    /// --per-group-frac
    #[arg(
        long,
        value_name = "TABLE",
        conflicts_with_all = ["cells", "cell_frac"],
        long_help = "Draw the cells within each group of TABLE, with --per-group or\n\
                     --per-group-frac: a quota per donor or sample, not one over the pool.\n\
                     TABLE is what `split --groups` reads: two columns (cell name, group\n\
                     label), .parquet or delimited text, optionally .gz, matched exactly;\n\
                     or one label per line in column order, as `qc` and `merge-backend`\n\
                     write in {out}.batch.gz. Every cell needs a label. The kept cells'\n\
                     labels are written to {output}.batch.gz, for senna's -b.\n\
                     A quota is per group, so --cells and --cell-frac do not combine with it."
    )]
    pub groups: Option<Box<str>>,

    /// cells to keep in each group; a group with fewer keeps all of them
    #[arg(long, requires = "groups", conflicts_with = "per_group_frac")]
    pub per_group: Option<usize>,

    /// fraction of each group's cells to keep, in (0, 1] (at least one each)
    #[arg(long, requires = "groups")]
    pub per_group_frac: Option<f64>,

    /// draw evenly across sketch bins, to keep rare states (never the default)
    #[arg(
        long,
        default_value_t = false,
        long_help = "Draw evenly across sketch bins instead of uniformly, to keep rare states.\n\
                     Cells are projected (log1p of the counts, each cell scaled to unit\n\
                     length, times a Gaussian matrix of --sketch-dim dimensions, as senna's\n\
                     --proj-dim sketch) and binned by the signs of the top --sketch-bits\n\
                     components (as senna's --sort-dim). The quota is shared evenly over\n\
                     the bins; a bin with fewer cells than its share keeps all of them.\n\
                     With --groups the bins are the same for all cells and each group's\n\
                     quota is shared over the bins it has.\n\
                     \n\
                     CAVEAT: this changes the cell-type composition on purpose (rare states\n\
                     up-weighted, common ones down). Abundances, and integration and mixing\n\
                     scores computed on the result, describe a population no donor has."
    )]
    pub sketch: bool,

    /// dimensions of the sketch's random projection
    #[arg(long, default_value_t = 50,
          value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub sketch_dim: usize,

    /// top components whose signs bin the sketch (at most 2^bits bins)
    #[arg(long, default_value_t = 10,
          value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..=63))]
    pub sketch_bits: usize,

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
/// `--groups` draws a quota within each group; `--sketch` draws evenly over
/// sketch bins, which reads every cell once to bin it.
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
    // The names, read now only for a table to match them against.
    let column_names = args
        .groups
        .is_some()
        .then(|| data.column_names())
        .transpose()?;
    let pool = Pool::of(args, ncol, column_names.as_deref())?;
    if pool.grouped() {
        // Never over a file: an earlier output's labels, or the table itself.
        let labels = batch_file(&output_path(&args.output, args.zip, &args.backend));
        anyhow::ensure!(
            !std::path::Path::new(&labels).exists(),
            "{labels} exists; the kept cells' labels would replace it: choose another output"
        );
    }
    let mut draw = Draw {
        cells: match pool.grouped() {
            true => resolve_target(args.per_group, None, ncol, "per-group")?,
            false => resolve_target(args.cells, args.cell_frac, ncol, "cells")?,
        },
        genes: resolve_target(args.genes, args.gene_frac, nrow, "genes")?,
        seed: args.seed,
        sketch: args.sketch,
    };
    let sketch = Sketch {
        dim: args.sketch_dim,
        bits: args.sketch_bits,
    };
    let mut tui = session.as_ref().map(Tui::new);
    if let Some(tui) = tui.as_mut() {
        let size = Size {
            nrow,
            ncol,
            nnz: data.num_non_zeros(),
        };
        match choose(tui, args, size, &pool, sketch, draw)? {
            Some(c) => draw = c,
            None => {
                info!("subsample: cancelled; nothing written");
                return Ok(());
            }
        }
    }
    if let Some(why) = pool.blocked(draw) {
        anyhow::bail!("{why}");
    }
    off_screen(tui.as_mut(), || {
        write_subsample(args, &*data, column_names, &pool, sketch, draw)
    })
}

/// The input's size, for the plan.
#[derive(Debug, Clone, Copy)]
struct Size {
    nrow: usize,
    ncol: usize,
    nnz: Option<usize>,
}

/// What to draw: cells (per group, under `--groups`) and genes (`None`:
/// all, or `--per-group-frac` of each group), the seed, and whether over
/// sketch bins.
#[derive(Debug, Clone, Copy)]
struct Draw {
    cells: Option<usize>,
    genes: Option<usize>,
    seed: u64,
    sketch: bool,
}

/// The groups cells are drawn within: those of a `--groups` table, or one
/// group of every cell.
struct Pool {
    ncol: usize,
    table: Option<Groups>,
    /// `--per-group-frac`.
    frac: Option<f64>,
}

/// The groups of a `--groups` table.
struct Groups {
    /// Each cell's group.
    of_column: Vec<usize>,
    /// Each group's cells, ascending.
    columns: Vec<Vec<usize>>,
    /// Each group's label.
    names: Vec<Box<str>>,
}

impl Pool {
    /// `column_names` is read for a table only.
    fn of(
        args: &SubsampleArgs,
        ncol: usize,
        column_names: Option<&[Box<str>]>,
    ) -> anyhow::Result<Self> {
        let (Some(path), Some(column_names)) = (args.groups.as_deref(), column_names) else {
            return Ok(Self {
                ncol,
                table: None,
                frac: None,
            });
        };
        if let Some(f) = args.per_group_frac {
            check_frac(f, "per-group")?;
        }
        let labels = super::group_table::column_labels(path, column_names)?;
        let (of_column, columns, names) = super::group_table::groups_of(&labels);
        info!("{ncol} cells in {} groups from {path}", columns.len());
        Ok(Self {
            ncol,
            table: Some(Groups {
                of_column,
                columns,
                names,
            }),
            frac: args.per_group_frac,
        })
    }

    fn grouped(&self) -> bool {
        self.table.is_some()
    }

    /// How many groups.
    fn len(&self) -> usize {
        self.table.as_ref().map_or(1, |g| g.columns.len())
    }

    /// How many of a group of `size` cells `draw` keeps.
    fn quota(&self, size: usize, draw: Draw) -> usize {
        match (draw.cells, self.frac) {
            (Some(n), _) => n.min(size),
            (None, Some(f)) => frac_count(f, size),
            (None, None) => size,
        }
    }

    /// How many cells `draw` keeps.
    fn kept(&self, draw: Draw) -> usize {
        match &self.table {
            Some(g) => g.columns.iter().map(|c| self.quota(c.len(), draw)).sum(),
            None => self.quota(self.ncol, draw),
        }
    }

    /// The table's groups smaller than the `--per-group` count, kept whole.
    fn short(&self, draw: Draw) -> Vec<(&str, usize)> {
        match (&self.table, draw.cells) {
            (Some(g), Some(n)) => (g.names.iter().zip(&g.columns))
                .filter(|(_, c)| c.len() < n)
                .map(|(name, c)| (name.as_ref(), c.len()))
                .collect(),
            _ => vec![],
        }
    }

    /// Why `draw` cannot be written, if it cannot.
    fn blocked(&self, draw: Draw) -> Option<String> {
        let counted = draw.cells.is_some() || self.frac.is_some();
        match self.grouped() {
            true if !counted => {
                Some("set the cells per group (n): --per-group or --per-group-frac".into())
            }
            false if draw.sketch && !counted => Some(
                "the sketch draws a number of cells: set the cells (n), --cells or --cell-frac"
                    .into(),
            ),
            false if !counted && draw.genes.is_none() => {
                Some("specify at least one of --cells / --cell-frac / --genes / --gene-frac".into())
            }
            _ => None,
        }
    }

    /// What each group keeps, for the plan.
    fn quota_text(&self, draw: Draw) -> String {
        match (draw.cells, self.frac) {
            (Some(n), _) => format!("{n} cells"),
            (None, Some(f)) => format!("{f} of the cells"),
            (None, None) => "? cells".into(),
        }
    }
}

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
    Ok(Draw {
        cells,
        genes,
        seed,
        sketch: s[3].flag(),
    })
}

/// What `draw` keeps of an input of `size`.
fn plan_of(output: &str, size: Size, pool: &Pool, sketch: Sketch, draw: Draw) -> Plan {
    let (rows, cols) = (draw.genes.unwrap_or(size.nrow), pool.kept(draw));
    let share = rows as f64 / size.nrow.max(1) as f64 * cols as f64 / size.ncol.max(1) as f64;
    let entries =
        |n: Option<usize>, f: f64| n.map_or("?".into(), |n| format!("{:.0}", n as f64 * f));
    let how = if draw.sketch {
        "evenly over sketch bins"
    } else {
        "at random"
    };
    let mut notes = vec![format!(
        "{rows} of {} rows and {cols} of {} columns {how}, seed {}; \
         the entries scale with both",
        size.nrow, size.ncol, draw.seed
    )];
    if pool.grouped() {
        notes.push(format!(
            "{} from each of {} groups; their labels go in the .batch.gz beside it",
            pool.quota_text(draw),
            pool.len()
        ));
        let short = pool.short(draw);
        if !short.is_empty() {
            notes.push(format!(
                "{} groups have fewer and are kept whole: {}",
                short.len(),
                names_shown(&short)
            ));
        }
    }
    if draw.sketch {
        notes.push(format!(
            "sketch: {} dimensions, bins by the signs of the top {} components; \
             rare states are up-weighted, so the composition is not the data's",
            sketch.dim, sketch.bits
        ));
    }
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
        notes,
        blocked: pool.blocked(draw),
    }
}

/// `name (size)` for the first few, and how many more.
fn names_shown(groups: &[(&str, usize)]) -> String {
    const SHOWN: usize = 8;
    let mut s = groups
        .iter()
        .take(SHOWN)
        .map(|(name, n)| format!("{name} ({n})"))
        .collect::<Vec<_>>()
        .join(", ");
    if groups.len() > SHOWN {
        s.push_str(&format!(" and {} more", groups.len() - SHOWN));
    }
    s
}

/// Show the plan of `draw` on `tui`; the draw it ended on, or `None` when
/// cancelled.
fn choose(
    tui: &mut Tui<'_>,
    args: &SubsampleArgs,
    size: Size,
    pool: &Pool,
    sketch: Sketch,
    draw: Draw,
) -> anyhow::Result<Option<Draw>> {
    let output = output_path(&args.output, args.zip, &args.backend);
    let cells = match pool.grouped() {
        true => "cells per group",
        false => "cells",
    };
    let controls = vec![
        Control::number('n', cells, draw.cells.map(|n| n as f64)),
        Control::number('g', "genes", draw.genes.map(|n| n as f64)),
        Control::number('s', "seed", Some(draw.seed as f64)),
        Control::flag('k', "sketch", draw.sketch),
    ];
    let planner: Planner =
        Box::new(move |s: &[Setting]| Ok(plan_of(&output, size, pool, sketch, draw_of(s, size)?)));
    let view = PlanView::new("subsample", &args.data_file, "write", controls, planner);
    match plan_in(tui, view)? {
        Some(s) => draw_of(&s, size).map(Some),
        None => Ok(None),
    }
}

/// The cells `draw` keeps, ascending.
fn pick_cells(
    rng: &mut SmallRng,
    data_file: &str,
    pool: &Pool,
    sketch: Sketch,
    draw: Draw,
) -> anyhow::Result<Vec<usize>> {
    let ncol = pool.ncol;
    // Nothing to choose between when every group is kept whole.
    let bins = if draw.sketch && pool.kept(draw) < ncol {
        info!(
            "sketch: projecting {ncol} cells to {} dimensions, {} bits ...",
            sketch.dim, sketch.bits
        );
        Some(sketch::bins(data_file, sketch, draw.seed)?)
    } else {
        None
    };
    let short = pool.short(draw);
    if !short.is_empty() {
        warn!(
            "{} of {} groups have fewer than {} cells and are kept whole: {}",
            short.len(),
            pool.len(),
            draw.cells.unwrap_or_default(),
            names_shown(&short)
        );
    }
    let draw_from = |rng: &mut SmallRng, cells: &[usize], k: usize| -> Vec<usize> {
        match &bins {
            _ if k >= cells.len() => cells.to_vec(),
            Some(b) => sketch::draw_across_bins(rng, cells, b, k),
            None => sample(rng, cells.len(), k)
                .into_iter()
                .map(|i| cells[i])
                .collect(),
        }
    };
    let mut kept: Vec<usize> = match &pool.table {
        Some(g) => g
            .columns
            .iter()
            .flat_map(|cells| draw_from(rng, cells, pool.quota(cells.len(), draw)))
            .collect(),
        // The whole backend, drawn by index: no list of every cell.
        None if bins.is_none() => {
            return Ok(sample_sorted(rng, ncol, Some(pool.quota(ncol, draw))));
        }
        None => draw_from(rng, &(0..ncol).collect::<Vec<_>>(), pool.quota(ncol, draw)),
    };
    kept.sort_unstable();
    Ok(kept)
}

/// Write the subsample `draw` picks.
fn write_subsample(
    args: &SubsampleArgs,
    data: &dyn SparseIo<IndexIter = Vec<usize>>,
    column_names: Option<Vec<Box<str>>>,
    pool: &Pool,
    sketch: Sketch,
    draw: Draw,
) -> anyhow::Result<()> {
    let mut rng = SmallRng::seed_from_u64(draw.seed);
    let nrow = data.num_rows().unwrap_or_default();

    // Ascending index sets so the output preserves the original order.
    let cell_idx = pick_cells(&mut rng, &args.data_file, pool, sketch, draw)?;
    let gene_idx = sample_sorted(&mut rng, nrow, draw.genes);

    info!(
        "subsampling to {} genes x {} cells (seed {})",
        gene_idx.len(),
        cell_idx.len(),
        draw.seed
    );

    let out_nrow = gene_idx.len();
    let out_ncol = cell_idx.len();

    let row_names_all = data.row_names()?;
    let col_names_all = match column_names {
        Some(names) => names,
        None => data.column_names()?,
    };
    let out_row_names: Vec<Box<str>> = gene_idx.iter().map(|&i| row_names_all[i].clone()).collect();
    let out_col_names: Vec<Box<str>> = cell_idx.iter().map(|&i| col_names_all[i].clone()).collect();

    let (effective_output, backend_out, file_out) =
        prepare_output(&args.output, args.backend.clone(), args.zip)?;

    // Streamed: the survivors never sit in memory at once. Both index sets
    // are ascending, which is what lets the row renumbering stay monotone
    // and the columns keep their order without a per-column sort.
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

    // The labels first: should they fail, the backend is not yet final.
    if let Some(g) = &pool.table {
        let labels: Vec<Box<str>> = cell_idx
            .iter()
            .map(|&c| g.names[g.of_column[c]].clone())
            .collect();
        let path = batch_file(&effective_output);
        write_lines(&labels, &path)?;
        info!("group labels of the kept cells: {path}");
    }
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
        check_frac(f, label)?;
        Ok(Some(frac_count(f, total)))
    } else {
        Ok(None)
    }
}

fn check_frac(f: f64, label: &str) -> anyhow::Result<()> {
    anyhow::ensure!(f > 0.0 && f <= 1.0, "--{label}-frac must be in (0, 1]");
    Ok(())
}

/// `f` of `total`, rounded, at least one.
fn frac_count(f: f64, total: usize) -> usize {
    ((f * total as f64).round() as usize).clamp(1, total.max(1))
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
