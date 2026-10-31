//! Opening data for the statistics explorer, and picking rows or columns
//! with it for the commands that take a selection.

use crate::hdf5_io::resolve_backend_file;
use crate::interactive::stat_tui::{
    explore_in, Dataset, Picked, Purpose, Side, StatExplorer, Values, ValuesReader,
};
use crate::interactive::ui::Tui;
use crate::interactive::{tui_available, TOOL};
use crate::qc::*;
use crate::sparse_io::*;
use crate::sparse_io_vector::*;

use legume_numeric::matrix::common_io::basename;
use legume_numeric::matrix::traits::RunningStatOps;
use log::info;
use std::sync::Arc;

/// The session of interactive command `screen` (`data-beans-<screen>-N`),
/// opened before any of its work. Hosted, this process goes no further:
/// it relays the terminal and exits with the command, which runs again in
/// the session's host. Always on something named, so a new one starts.
pub fn open_session(
    screen: &str,
    args: &legume_tui::cli::SessionArgs,
) -> anyhow::Result<legume_tui::Session> {
    Ok(TOOL.session(screen, &args.options(true))?)
}

/// Whether a command that writes shows its plan first, and how that view
/// runs as a session.
#[derive(clap::Args, Debug, Clone, Default)]
pub struct PlanArgs {
    #[arg(
        long,
        default_value_t = false,
        help = "Write straight away: no plan first, even on a terminal",
        long_help = "On a terminal, a plan of what will be written opens first: the inputs\n\
                     and what they share, the output and its size, and the settings on keys,\n\
                     each change redrawing the plan. Enter writes with the settings shown; q\n\
                     writes nothing. --batch-process (or no terminal) writes straight away."
    )]
    pub batch_process: bool,

    #[command(flatten)]
    pub session: legume_tui::cli::SessionArgs,
}

impl PlanArgs {
    /// Straight through, for a command run by another.
    pub fn batch() -> Self {
        Self {
            batch_process: true,
            ..Self::default()
        }
    }

    /// The session `command`'s plan opens in, opened before any work; none
    /// when it writes straight away.
    pub fn session(&self, command: &str) -> anyhow::Result<Option<legume_tui::Session>> {
        if !crate::interactive::opens_view(command, self.batch_process) {
            return Ok(None);
        }
        open_session(command, &self.session).map(Some)
    }
}

/// Run `f`, a command's work after its views, off the screen of `tui` when
/// there is one: a `kill` meanwhile lets it finish.
pub fn off_screen<T>(
    tui: Option<&mut Tui<'_>>,
    f: impl FnOnce() -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    match tui {
        Some(tui) => tui.work(f)?,
        None => f(),
    }
}

/// Open `files` as one matrix. Several files keep their column names
/// distinct by appending each file's name.
pub fn open_data(files: &[Box<str>], preload: bool) -> anyhow::Result<SparseIoVec> {
    let attach_data_name = files.len() > 1;
    let mut data = SparseIoVec::new();
    for file in files {
        let (backend, path) = resolve_backend_file(file, None)?;
        let mut this = open_sparse_matrix(&path, &backend)?;
        if preload {
            info!("Preloading data from {} ...", path);
            this.preload_columns()?;
        }
        let data_name = attach_data_name.then(|| basename(&path)).transpose()?;
        data.push(Arc::from(this), data_name)?;
    }
    Ok(data)
}

fn dataset<S: RunningStatOps<f32, Output = Vec<f32>>>(names: Vec<Box<str>>, stat: &S) -> Dataset {
    Dataset {
        names,
        values: [stat.count_positives(), stat.sum(), stat.mean(), stat.std()],
    }
}

/// Collect one side's nnz/sum/mean/sd, also saving them to `save_to` when
/// given. Column statistics count only `select_rows`, if given.
pub fn side_stats(
    data: &SparseIoVec,
    side: Side,
    select_rows: Option<&[usize]>,
    block_size: Option<usize>,
    save_to: Option<&str>,
) -> anyhow::Result<Dataset> {
    Ok(match side {
        Side::Rows => {
            let stat = collect_row_stat_across_vec(data, block_size)?;
            let names = data.row_names()?;
            if let Some(out) = save_to {
                stat.save(out, &names, "\t")?;
            }
            dataset(names, &stat)
        }
        Side::Columns => {
            let stat = collect_column_stat_across_vec(data, select_rows, block_size)?;
            let names = data.column_names()?;
            if let Some(out) = save_to {
                stat.save(out, &names, "\t")?;
            }
            dataset(names, &stat)
        }
    })
}

/// Reads the dense values of chosen rows (or columns) against every column
/// (or row) of `data`, for the explorer's values view.
pub fn values_reader(data: &SparseIoVec) -> ValuesReader<'_> {
    Box::new(move |side, entries: &[usize]| {
        Ok(match side {
            Side::Rows => {
                // entries x columns
                let m = data.read_rows_dmatrix(entries.iter().copied())?;
                Values {
                    names: data.column_names()?,
                    columns: m.row_iter().map(|r| r.iter().copied().collect()).collect(),
                }
            }
            Side::Columns => {
                // rows x entries
                let m = data.read_columns_dmatrix(entries.iter().copied())?;
                Values {
                    names: data.row_names()?,
                    columns: m
                        .column_iter()
                        .map(|c| c.iter().copied().collect())
                        .collect(),
                }
            }
        })
    })
}

/// Let the user mark rows or columns of `data_file` in the explorer, for a
/// command labelled `verb`. Returns the marked indices (ascending), or
/// `None` when the user quits without finishing.
pub fn pick_entries(
    tui: &mut Tui<'_>,
    data_file: &str,
    side: Side,
    verb: &'static str,
) -> anyhow::Result<Option<Vec<usize>>> {
    if !tui_available() {
        anyhow::bail!("picking needs a terminal");
    }
    let data = open_data(&[data_file.into()], false)?;
    let first = side_stats(&data, side, None, None, None)?;
    let explorer = StatExplorer::new(
        data_file,
        side,
        first,
        None,
        Some(values_reader(&data)),
        Purpose::Pick { verb },
    );
    let picked = explore_in(tui, explorer)?.map(|p| p.side(side));
    match &picked {
        Some(entries) => info!("picked {} entries", entries.len()),
        None => info!("nothing picked; cancelled"),
    }
    Ok(picked)
}

/// Let the user mark rows and columns of `data_file` in the explorer (Tab
/// switches sides; the rows are computed on first use), for a command
/// labelled `verb`. Returns the marks of both sides, or `None` when the user
/// quits without finishing.
pub fn pick_rows_and_columns(
    tui: &mut Tui<'_>,
    data_file: &str,
    verb: &'static str,
) -> anyhow::Result<Option<Picked>> {
    if !tui_available() {
        anyhow::bail!("this command needs a terminal");
    }
    let data = open_data(&[data_file.into()], false)?;
    let first = side_stats(&data, Side::Columns, None, None, None)?;
    let data = &data;
    let loader = Box::new(move |side| side_stats(data, side, None, None, None));
    let picked = explore_in(
        tui,
        StatExplorer::new(
            data_file,
            Side::Columns,
            first,
            Some(loader),
            Some(values_reader(data)),
            Purpose::PickBoth { verb },
        ),
    )?;
    match &picked {
        Some(p) => info!(
            "picked {} rows and {} columns",
            p.rows.len(),
            p.columns.len()
        ),
        None => info!("nothing picked; cancelled"),
    }
    Ok(picked)
}
