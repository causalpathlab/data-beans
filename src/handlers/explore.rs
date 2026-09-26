//! Opening data for the statistics explorer, and picking rows or columns
//! with it for the commands that take a selection.

use crate::hdf5_io::resolve_backend_file;
use crate::interactive::stat_tui::{
    explore, Dataset, Purpose, Side, StatExplorer, Values, ValuesReader,
};
use crate::interactive::tui_available;
use crate::qc::*;
use crate::sparse_io::*;
use crate::sparse_io_vector::*;

use legume_numeric::matrix::common_io::basename;
use legume_numeric::matrix::traits::RunningStatOps;
use log::info;
use std::sync::Arc;

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
                let m = data.read_rows_ndarray(entries.iter().copied())?;
                Values {
                    names: data.column_names()?,
                    columns: m.rows().into_iter().map(|r| r.to_vec()).collect(),
                }
            }
            Side::Columns => {
                // rows x entries
                let m = data.read_columns_ndarray(entries.iter().copied())?;
                Values {
                    names: data.row_names()?,
                    columns: m.columns().into_iter().map(|c| c.to_vec()).collect(),
                }
            }
        })
    })
}

/// Let the user mark rows or columns of `data_file` in the explorer, for a
/// command labelled `verb`. Returns the marked indices (ascending), or
/// `None` when the user quits without finishing.
pub fn pick_entries(
    data_file: &str,
    side: Side,
    verb: &'static str,
) -> anyhow::Result<Option<Vec<usize>>> {
    if !tui_available() {
        anyhow::bail!("--interactive needs a terminal");
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
    let picked = explore(explorer)?;
    match &picked {
        Some(entries) => info!("picked {} entries", entries.len()),
        None => info!("nothing picked; cancelled"),
    }
    Ok(picked)
}
