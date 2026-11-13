//! Train/test and K-fold splitting of a backend, by CELL.
//!
//! # Why this lives in data-beans
//!
//! Splitting is a data operation, not a model one: the folds have to be
//! identical no matter which engine consumes them, or two methods are compared
//! on different data and the comparison means nothing. Putting it beside
//! `subsample` and `subset-columns` gives every downstream crate — `senna`,
//! `pinto`, anything later — one splitter and one seed convention.
//!
//! # The grouped split is the point on spatial data
//!
//! A random cell split leaves nearly every test cell ringed by training cells.
//! On spatial data adjacent cells are near-duplicates, so the model has
//! effectively seen each test cell already and the score comes out optimistic —
//! it measures interpolation between neighbours, not generalization.
//!
//! So the split is by *region*, not by cell. Give it coordinates and a grid
//! resolution — `--coord positions.csv --grid 8` tiles the bounding box 8×8 and
//! assigns whole tiles — and a test region gets a training boundary rather than
//! a training interior. The gap between a random and a tiled split is worth
//! reporting rather than choosing between.
//!
//! `--groups` is the escape hatch for a grouping that is not geometric: donor,
//! sample, slide, or any precomputed labelling.
//!
//! # Ablating features
//!
//! Deliberately not a flag here. Dropping genes from the training half composes
//! out of what already exists:
//!
//! ```text
//! data-beans split data.zarr -o cv --test-frac 0.2
//! data-beans subsample cv.train.zarr.zip --gene-frac 0.8 -o cv.train.ablated
//! ```
//!
//! Keeping the two operations separate means the split stays reproducible from
//! its own seed regardless of what the ablation does.

use crate::hdf5_io::*;
use crate::sparse_io::*;
use crate::zarr_io::finalize_output;

use clap::Args;
use log::info;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rustc_hash::FxHashMap as HashMap;

use crate::handlers::explore::{off_screen, PlanArgs};
use crate::interactive::ui::Tui;

#[path = "split_plan.rs"]
mod plan;

#[derive(Args, Debug)]
pub struct SplitArgs {
    /// data file -- `.zarr`, `.zarr.zip`, or `.h5`
    pub data_file: Box<str>,

    /// fraction of cells (columns) held out as the test half, in (0, 1)
    #[arg(long, conflicts_with = "folds")]
    pub test_frac: Option<f64>,

    /// number of cross-validation folds; writes {out}.fold{k}.{train,test}
    #[arg(long, conflicts_with = "test_frac")]
    pub folds: Option<usize>,

    /// cell coordinates -- split by REGION instead of by cell (CSV/TSV/parquet)
    ///
    /// First column is the cell name; the coordinate columns are chosen with
    /// --coord-columns. Pair with --grid.
    #[arg(long, value_name = "FILE", conflicts_with = "groups")]
    pub coord: Option<Box<str>>,

    /// 0-based coordinate column indices, comma separated [default: 1,2]
    #[arg(long, value_name = "I,J", value_delimiter = ',', requires = "coord")]
    pub coord_columns: Option<Vec<usize>>,

    /// tiles per axis over the coordinate bounding box; 8 gives an 8x8 grid
    #[arg(long, default_value_t = 8, requires = "coord")]
    pub grid: usize,

    /// two-column table (cell name, group label), or one label per line in
    /// column order (qc's {out}.batch.gz) -- keep cells sharing a label on
    /// the same side: donor, sample, slide. Use --coord for geometry instead.
    #[arg(long, value_name = "TABLE")]
    pub groups: Option<Box<str>>,

    /// RNG seed for a reproducible split
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// backend for the output files
    #[arg(long, value_enum, default_value = "zarr")]
    pub backend: SparseIoBackend,

    /// output prefix: writes {output}.train.* and {output}.test.*
    #[arg(short, long)]
    pub output: Box<str>,

    /// keep `.zarr` directories instead of producing `.zarr.zip` archives
    #[arg(long = "no-zip", default_value_t = true, action = clap::ArgAction::SetFalse)]
    pub zip: bool,

    #[command(flatten)]
    pub view: PlanArgs,
}

/// How the cells are split: what `--test-frac`, `--folds`, `--grid` and
/// `--seed` say, or what the plan ended on.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Choice {
    test_frac: Option<f64>,
    folds: Option<usize>,
    grid: usize,
    seed: u64,
}

/// Split a backend into train and test halves by cell, or into K folds.
///
/// Every cell lands in exactly one side (or exactly one fold's test half), so
/// the halves are disjoint and together cover the input — checked, not assumed.
pub fn run_split(args: &SplitArgs) -> anyhow::Result<()> {
    let session = args.view.session("split")?;
    let (backend_in, file_in) = resolve_backend_file(&args.data_file, None)?;
    let data = open_sparse_matrix(&file_in, &backend_in)?;
    let n_columns = data
        .num_columns()
        .ok_or_else(|| anyhow::anyhow!("backend has no `ncol`"))?;
    let column_names = data.column_names()?;
    let mut choice = Choice {
        test_frac: args.test_frac,
        folds: args.folds,
        grid: args.grid,
        seed: args.seed,
    };
    let grouping = Grouping::read(args, &column_names)?;
    let mut tui = session.as_ref().map(Tui::new);
    if let Some(tui) = tui.as_mut() {
        match plan::choose(tui, args, &grouping, column_names.len(), choice)? {
            Some(c) => choice = c,
            None => {
                info!("split: cancelled; nothing written");
                return Ok(());
            }
        }
    }
    let (_, splits) = split_columns(&grouping, n_columns, choice)?;
    off_screen(tui.as_mut(), || {
        write_splits(args, &*data, &column_names, &splits, choice)
    })
}

/// The cells of each train and test half: their columns, by fold.
type Splits = Vec<(Vec<usize>, Vec<usize>)>;

/// What the cells are grouped by, read once: the units shuffled and assigned
/// whole. One column each by default, one grid tile under `--coord`, one
/// label under `--groups`. Everything after is the same in all three cases,
/// which is why grouped splitting costs nothing extra to support.
enum Grouping {
    Cells,
    /// `--coord`: the file, and each column's position in it.
    Coords {
        path: Box<str>,
        of_column: Vec<(f64, f64)>,
    },
    /// `--groups`: the columns of each label.
    Table(Vec<Vec<usize>>),
}

impl Grouping {
    /// The grouping `args` asks for over the cells `column_names`.
    fn read(args: &SplitArgs, column_names: &[Box<str>]) -> anyhow::Result<Self> {
        Ok(match (args.coord.as_deref(), args.groups.as_deref()) {
            (Some(path), _) => {
                let coord_columns = args.coord_columns.clone().unwrap_or_else(|| vec![1, 2]);
                Grouping::Coords {
                    path: path.into(),
                    of_column: coords_of_columns(path, &coord_columns, column_names)?,
                }
            }
            (_, Some(path)) => Grouping::Table(column_groups_from_table(path, column_names)?),
            _ => Grouping::Cells,
        })
    }

    /// The columns of each group, of `n_columns`; `grid` tiles an axis under
    /// `--coord`.
    fn column_groups(&self, n_columns: usize, grid: usize) -> anyhow::Result<Vec<Vec<usize>>> {
        Ok(match self {
            Grouping::Coords { path, of_column } => column_groups_from_grid(path, grid, of_column)?,
            Grouping::Table(groups) => groups.clone(),
            Grouping::Cells => (0..n_columns).map(|i| vec![i]).collect(),
        })
    }
}

/// The halves `choice` makes of the `n_columns` columns `grouping` groups,
/// and how many groups they were dealt from.
fn split_columns(
    grouping: &Grouping,
    n_columns: usize,
    choice: Choice,
) -> anyhow::Result<(usize, Splits)> {
    let column_groups = grouping.column_groups(n_columns, choice.grid)?;
    // Two, not one: a single group cannot be divided, and `--test-frac` would
    // reach `clamp(1, 0)` and panic. One group is a real outcome of a coarse
    // `--grid` or a `--groups` table with a single label, so it gets a message
    // that names the cause rather than a backtrace.
    anyhow::ensure!(
        column_groups.len() >= 2,
        "only {} group(s) to split: every cell falls in the same tile or group, \
         so there is no way to hold any of them out. Raise --grid, or check that \
         --groups / --coord really vary across cells.",
        column_groups.len()
    );

    // Group ids first, then the columns they cover: the partition never sees a
    // column, so a grouped split cannot leak one across the boundary by
    // construction rather than by a later check.
    let splits = partition_groups(
        column_groups.len(),
        choice.test_frac,
        choice.folds,
        choice.seed,
    )?
    .into_iter()
    .map(|(train_groups, test_groups)| {
        (
            columns_in_groups(&column_groups, &train_groups),
            columns_in_groups(&column_groups, &test_groups),
        )
    })
    .collect();
    Ok((column_groups.len(), splits))
}

/// `{output}.train` and `{output}.test`, or `{output}.fold{k}.*` for folds.
fn half_name(args: &SplitArgs, choice: Choice, fold: usize, half: &str) -> String {
    match choice.folds {
        Some(_) => format!("{}.fold{fold}.{half}", args.output),
        None => format!("{}.{half}", args.output),
    }
}

fn write_splits(
    args: &SplitArgs,
    data: &dyn SparseIo<IndexIter = Vec<usize>>,
    column_names: &[Box<str>],
    splits: &Splits,
    choice: Choice,
) -> anyhow::Result<()> {
    let n_columns = column_names.len();
    let row_names = data.row_names()?;
    let names = (row_names.as_slice(), column_names);
    // Every half is checked before the first is written: an input named
    // like a later half would otherwise be refused with the earlier halves
    // already on disk.
    for fold in 0..splits.len() {
        for half in ["train", "test"] {
            super::ensure_output_is_no_input(
                &[&args.data_file],
                &half_name(args, choice, fold, half),
                args.backend.clone(),
                args.zip,
            )?;
        }
    }
    for (fold, (train_columns, test_columns)) in splits.iter().enumerate() {
        anyhow::ensure!(
            train_columns.len() + test_columns.len() == n_columns,
            "split lost cells: {} train + {} test != {n_columns}",
            train_columns.len(),
            test_columns.len()
        );
        let tag = |half: &str| half_name(args, choice, fold, half);
        write_half(data, names, train_columns, &tag("train"), args)?;
        write_half(data, names, test_columns, &tag("test"), args)?;
        info!(
            "fold {fold}: {} train cells, {} test cells",
            train_columns.len(),
            test_columns.len()
        );
    }
    Ok(())
}

/// Flatten group ids to the column indices they cover, sorted so the backend
/// read stays sequential.
fn columns_in_groups(column_groups: &[Vec<usize>], picked: &[usize]) -> Vec<usize> {
    let mut columns: Vec<usize> = picked
        .iter()
        .flat_map(|&g| column_groups[g].iter().copied())
        .collect();
    columns.sort_unstable();
    columns
}

/// Each column's position, from the coordinate file `path`; every cell
/// needs one, and a finite one.
fn coords_of_columns(
    path: &str,
    coord_columns: &[usize],
    column_names: &[Box<str>],
) -> anyhow::Result<Vec<(f64, f64)>> {
    anyhow::ensure!(
        coord_columns.len() == 2,
        "--coord-columns needs exactly two indices; got {}",
        coord_columns.len()
    );
    let coords = read_coords(path, coord_columns)?;

    let of_column: Vec<(f64, f64)> = column_names
        .iter()
        .filter_map(|name| coords.get(name.as_ref()).copied())
        .collect();
    let n_missing = column_names.len() - of_column.len();
    anyhow::ensure!(
        n_missing == 0,
        "{path}: {n_missing} of {} cells have no coordinate",
        column_names.len()
    );
    Ok(of_column)
}

/// Cells grouped into a regular grid over their coordinates (each column's,
/// read from `path`).
///
/// A grid rather than a clustering on purpose: the tiles are then a property of
/// the tissue's geometry alone, reproducible from the coordinate file without a
/// model run, and a reader can say exactly how large a held-out region was.
/// Empty tiles simply do not appear.
fn column_groups_from_grid(
    path: &str,
    tiles_per_axis: usize,
    coord_of_column: &[(f64, f64)],
) -> anyhow::Result<Vec<Vec<usize>>> {
    anyhow::ensure!(tiles_per_axis >= 2, "--grid must be at least 2");

    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for &(x, y) in coord_of_column {
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    // A degenerate span would put every cell in tile 0; widen it so the grid is
    // still well defined rather than silently collapsing to one group.
    let span = |lo: f64, hi: f64| if hi > lo { hi - lo } else { 1.0 };
    let (width, height) = (span(x0, x1), span(y0, y1));

    let mut group_of_tile: HashMap<usize, usize> = HashMap::default();
    let mut column_groups: Vec<Vec<usize>> = Vec::new();
    for (column, &(x, y)) in coord_of_column.iter().enumerate() {
        let tx = (((x - x0) / width) * tiles_per_axis as f64)
            .floor()
            .clamp(0.0, (tiles_per_axis - 1) as f64) as usize;
        let ty = (((y - y0) / height) * tiles_per_axis as f64)
            .floor()
            .clamp(0.0, (tiles_per_axis - 1) as f64) as usize;
        let tile = ty * tiles_per_axis + tx;
        let next = column_groups.len();
        let group = *group_of_tile.entry(tile).or_insert(next);
        if group == next {
            column_groups.push(Vec::new());
        }
        column_groups[group].push(column);
    }
    info!(
        "spatial split: {} cells in {} non-empty tiles of a \
         {tiles_per_axis}x{tiles_per_axis} grid ({path})",
        coord_of_column.len(),
        column_groups.len()
    );
    Ok(column_groups)
}

/// `cell -> (x, y)` from a CSV/TSV/parquet whose first column is the cell
/// name. A cell whose position is not finite (`NaN`, `inf`) has none.
fn read_coords(path: &str, cols: &[usize]) -> anyhow::Result<HashMap<Box<str>, (f64, f64)>> {
    let (i, j) = (cols[0], cols[1]);
    let mut out: HashMap<Box<str>, (f64, f64)> = HashMap::default();
    if path.ends_with(".parquet") {
        let names = legume_numeric::matrix::parquet::read_parquet_string_column(path, 0)?;
        let mat =
            <nalgebra::DMatrix<f32> as legume_numeric::matrix::traits::IoOps>::from_parquet(path)?;
        // The matrix holds the columns after the names': file column 1 is
        // its column 0.
        let n_file_columns = mat.mat.ncols() + 1;
        anyhow::ensure!(
            (1..n_file_columns).contains(&i) && (1..n_file_columns).contains(&j),
            "{path}: --coord-columns {i},{j} are not among its value columns 1 to {} \
             (column 0 is the cell name)",
            n_file_columns - 1
        );
        for (r, name) in names.iter().enumerate() {
            let (x, y) = (
                f64::from(mat.mat[(r, i - 1)]),
                f64::from(mat.mat[(r, j - 1)]),
            );
            if x.is_finite() && y.is_finite() {
                out.insert(name.clone(), (x, y));
            }
        }
        anyhow::ensure!(!out.is_empty(), "{path}: no coordinate rows parsed");
        return Ok(out);
    }
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("reading coordinates {path}: {e}"))?;
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = if line.contains('\t') {
            line.split('\t').collect()
        } else if line.contains(',') {
            line.split(',').collect()
        } else {
            line.split_whitespace().collect()
        };
        let max = i.max(j);
        anyhow::ensure!(
            f.len() > max,
            "{path}:{}: {} columns, need index {max}",
            lineno + 1,
            f.len()
        );
        // A header line has non-numeric coordinates; skip it rather than failing.
        let (Ok(x), Ok(y)) = (f[i].trim().parse::<f64>(), f[j].trim().parse::<f64>()) else {
            continue;
        };
        if x.is_finite() && y.is_finite() {
            out.insert(f[0].trim().into(), (x, y));
        }
    }
    anyhow::ensure!(!out.is_empty(), "{path}: no coordinate rows parsed");
    Ok(out)
}

/// Cells grouped by their group label, from a group table (see
/// [`super::group_table`]).
fn column_groups_from_table(
    path: &str,
    column_names: &[Box<str>],
) -> anyhow::Result<Vec<Vec<usize>>> {
    let labels = super::group_table::column_labels(path, column_names)?;
    let (_, column_groups, _) = super::group_table::groups_of(&labels);
    info!(
        "grouped split: {} cells in {} groups from {path}",
        column_names.len(),
        column_groups.len()
    );
    Ok(column_groups)
}

/// Write one half as a new backend, streamed. The caller has checked that
/// `output` is not the input.
///
/// The selected columns' exact nnz comes off the resident indptr and the
/// survivors flow through the validated streaming writer in byte-budgeted
/// slabs, so peak residency is bounded by the slab budget rather than scaling
/// with the half — the wall the old triplet-materialising version hit at
/// imaging scale is gone, and so is the doc comment that described it.
fn write_half(
    data: &dyn SparseIo<IndexIter = Vec<usize>>,
    (row_names, all_column_names): (&[Box<str>], &[Box<str>]),
    selected_columns: &[usize],
    output: &str,
    args: &SplitArgs,
) -> anyhow::Result<()> {
    let kept_column_names: Vec<Box<str>> = selected_columns
        .iter()
        .map(|&c| all_column_names[c].clone())
        .collect();

    let (effective_output, backend_out, file_out) =
        super::prepare_output(output, args.backend.clone(), args.zip)?;
    info!("Streaming {} columns to {output}", selected_columns.len());
    let (n_rows, n_cols, nnz) = crate::column_subset::stream_column_selection(
        data,
        selected_columns,
        None,
        row_names,
        &kept_column_names,
        file_out.as_ref(),
        &backend_out,
    )?;

    let final_path = finalize_output(&file_out, &effective_output)?;
    info!("wrote {final_path} ({n_rows} genes x {n_cols} cells, {nnz} non-zeros)");
    Ok(())
}

/// Partition `n_groups` shuffled groups into `(train, test)` group-index pairs.
///
/// Separated from the column bookkeeping so the part that decides who is held
/// out can be tested without a backend on disk — this is the piece where an
/// off-by-one silently produces an overlapping or incomplete fold, which no
/// downstream number would reveal.
fn partition_groups(
    n_groups: usize,
    test_frac: Option<f64>,
    folds: Option<usize>,
    seed: u64,
) -> anyhow::Result<Vec<(Vec<usize>, Vec<usize>)>> {
    let mut shuffled_groups: Vec<usize> = (0..n_groups).collect();
    let mut rng = SmallRng::seed_from_u64(seed);
    shuffled_groups.shuffle(&mut rng);

    match (test_frac, folds) {
        (Some(f), _) => {
            anyhow::ensure!(
                f > 0.0 && f < 1.0,
                "--test-frac must be in (0, 1); got {f}. A percentage goes in as e.g. 0.2"
            );
            // Clamped so neither side is ever empty: a fraction that rounds to
            // zero, or to everything, is a silently useless split rather than an
            // error the user would see.
            let n_test = ((f * n_groups as f64).round() as usize).clamp(1, n_groups - 1);
            let (test_groups, train_groups) = shuffled_groups.split_at(n_test);
            Ok(vec![(train_groups.to_vec(), test_groups.to_vec())])
        }
        (_, Some(k)) => {
            anyhow::ensure!(k >= 2, "--folds must be at least 2");
            anyhow::ensure!(
                k <= n_groups,
                "--folds {k} exceeds the {n_groups} groups available to split"
            );
            Ok((0..k)
                .map(|f| {
                    let take = |in_test: bool| -> Vec<usize> {
                        shuffled_groups
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| (i % k == f) == in_test)
                            .map(|(_, &g)| g)
                            .collect()
                    };
                    (take(false), take(true))
                })
                .collect())
        }
        (None, None) => anyhow::bail!("specify one of --test-frac or --folds"),
    }
}

#[cfg(all(test, feature = "ndarray"))]
mod tests;
