use log::info;

use crate::handlers::explore::off_screen;
use crate::interactive::ui::Tui;
use legume_numeric::matrix::traits::RunningStatOps;

#[path = "import_preview.rs"]
mod import;

// The `from-*` commands `data_beans::import` builds live in `imports`.
mod from_fragments;
mod from_zarr;
mod imports;

pub use from_fragments::*;
pub use from_zarr::*;
pub use imports::*;

////////////////////////////////////////////////////////////////////
// Shared helpers used across all build-from-* subcommands.       //
// Each subcommand lives in its own submodule (`from_*`); helpers //
// that are reused by more than one of them stay here.            //
////////////////////////////////////////////////////////////////////

/// What `--do-squeeze` asks of a build.
#[derive(Debug, Clone, Copy)]
pub(super) struct Squeeze {
    pub(super) on: bool,
    /// The cutoffs, kept while `on` is off so the preview can turn it on.
    pub(super) cut: crate::import::Squeeze,
}

impl Squeeze {
    /// `--do-squeeze`, `--row-nnz-cutoff`, `--column-nnz-cutoff`,
    /// `--block-size`.
    pub(super) fn new(on: bool, rows: usize, columns: usize, block_size: Option<usize>) -> Self {
        let cut = crate::import::Squeeze {
            rows,
            columns,
            block_size,
        };
        Self { on, cut }
    }

    /// The library's squeeze, when on.
    fn cut(self) -> Option<crate::import::Squeeze> {
        self.on.then_some(self.cut)
    }

    fn apply(self, backend_file: &str) -> anyhow::Result<()> {
        match self.cut() {
            Some(cut) => cut.apply(backend_file),
            None => Ok(()),
        }
    }
}

/// Build with `whole`, the library's import squeezed as asked and put in
/// place, or, on a terminal, `stage` it and keep it after the import
/// preview ([`keep_import`]).
pub(super) fn run_import<O>(
    command: &str,
    session: Option<&legume_tui::Session>,
    opts: &O,
    out: crate::import::Output,
    squeeze: Squeeze,
    whole: fn(&O, &crate::import::Output) -> anyhow::Result<Box<str>>,
    stage: fn(&O, &crate::import::Output) -> anyhow::Result<crate::import::Staged>,
) -> anyhow::Result<()> {
    if session.is_none() {
        let squeeze = squeeze.cut();
        whole(opts, &crate::import::Output { squeeze, ..out })?;
    } else {
        let staged = stage(opts, &out)?;
        if keep_import(command, session, squeeze, staged.working())? {
            staged.finish()?;
        } else {
            staged.discard()?;
        }
    }
    Ok(())
}

/// The end of a build, before the output is finalized: squeezed as asked,
/// or, with a `session` (on a terminal), the import preview first: the
/// rows' and columns' non-zeros as histograms against the squeeze cutoffs,
/// each a setting. Enter keeps the build (squeezed when on); `q`
/// discards it. Whether to keep it: false for the caller to delete it
/// ([`crate::import::remove_build`]).
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
        info!("{command}: discarded");
        return Ok(false);
    };
    off_screen(Some(&mut tui), || s.apply(backend_file))?;
    Ok(true)
}

/// Non-zeros per row and per column of the backend at `file`: off its
/// indexes, else counted in a pass.
fn nnz_by_axis(file: &str) -> anyhow::Result<(Vec<usize>, Vec<usize>)> {
    let data = crate::sparse_io::open_sparse_matrix_by_path(file)?;
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
