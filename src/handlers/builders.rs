use log::info;

use crate::handlers::explore::off_screen;
use crate::interactive::ui::Tui;
use legume_numeric::matrix::traits::RunningStatOps;

#[path = "import_preview.rs"]
mod import;

#[cfg(feature = "hdf5")]
mod from_10x_matrix;
#[cfg(feature = "hdf5")]
mod from_10x_molecule;
mod from_fragments;
#[cfg(feature = "hdf5")]
mod from_h5ad;
mod from_mtx;
mod from_zarr;

#[cfg(feature = "hdf5")]
pub use from_10x_matrix::*;
#[cfg(feature = "hdf5")]
pub use from_10x_molecule::*;
pub use from_fragments::*;
#[cfg(feature = "hdf5")]
pub use from_h5ad::*;
pub use from_mtx::*;
pub use from_zarr::*;

////////////////////////////////////////////////////////////////////
// Shared helpers used across all build-from-* subcommands.       //
// Each subcommand lives in its own submodule (`from_*`); helpers //
// that are reused by more than one of them stay here.            //
////////////////////////////////////////////////////////////////////

pub(super) fn log_feature_type_histogram(label: &str, row_types: &[Box<str>]) {
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for t in row_types {
        *counts.entry(t.as_ref()).or_insert(0) += 1;
    }
    info!("Feature types in {}: {:?}", label, counts);
}

/// What `--do-squeeze` asks of a build.
#[derive(Debug, Clone, Copy)]
pub(super) struct Squeeze {
    pub(super) on: bool,
    pub(super) rows: usize,
    pub(super) columns: usize,
    pub(super) block_size: Option<usize>,
}

impl Squeeze {
    fn apply(self, backend_file: &str) -> anyhow::Result<()> {
        if self.on {
            crate::handlers::transformation::squeeze_in_place(
                backend_file,
                self.rows,
                self.columns,
                self.block_size,
            )?;
        }
        Ok(())
    }
}

/// The end of a build, before the output is finalized: squeezed as asked,
/// or, with a `session` (on a terminal), the import preview first: the
/// rows' and columns' non-zeros as histograms against the squeeze cutoffs,
/// each a setting. Enter keeps the build (squeezed when on); `q` deletes
/// it. Whether to go on and finalize.
pub(super) fn keep_import(
    command: &str,
    session: Option<&legume_tui::Session>,
    squeeze: Squeeze,
    backend_file: &str,
) -> anyhow::Result<bool> {
    let Some(session) = session else {
        squeeze.apply(backend_file)?;
        return Ok(true);
    };
    let mut tui = Tui::new(session);
    let (rows, columns) = off_screen(Some(&mut tui), || nnz_by_axis(backend_file))?;
    let Some(s) = import::preview(&mut tui, command, backend_file, &rows, &columns, squeeze)?
    else {
        info!("{command}: discarded; deleting {backend_file}");
        open_backend(backend_file)?.remove_backend_file()?;
        return Ok(false);
    };
    off_screen(Some(&mut tui), || s.apply(backend_file))?;
    Ok(true)
}

type Backend = Box<dyn crate::sparse_io::SparseIo<IndexIter = Vec<usize>>>;

fn open_backend(file: &str) -> anyhow::Result<Backend> {
    let (backend, file) = crate::hdf5_io::resolve_backend_file(file, None)?;
    crate::sparse_io::open_sparse_matrix(&file, &backend)
}

/// Non-zeros per row and per column of the backend at `file`: off its
/// indexes, else counted in a pass.
fn nnz_by_axis(file: &str) -> anyhow::Result<(Vec<usize>, Vec<usize>)> {
    let data = open_backend(file)?;
    let counts = |p: &[u64], n: usize| -> Option<Vec<usize>> {
        (p.len() == n + 1).then(|| p.windows(2).map(|w| (w[1] - w[0]) as usize).collect())
    };
    let as_counts = |v: Vec<f32>| -> Vec<usize> { v.into_iter().map(|n| n as usize).collect() };
    let (nrow, ncol) = (
        data.num_rows().unwrap_or(0),
        data.num_columns().unwrap_or(0),
    );
    let rows = match counts(data.row_indptr(), nrow) {
        Some(c) => c,
        None => as_counts(crate::qc::collect_row_stat(data.as_ref(), None)?.count_positives()),
    };
    let columns = match counts(data.column_indptr(), ncol) {
        Some(c) => c,
        None => as_counts(crate::qc::collect_column_stat(data.as_ref(), None)?.count_positives()),
    };
    Ok((rows, columns))
}
