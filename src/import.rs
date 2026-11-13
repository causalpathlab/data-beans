//! Build a backend from the formats single-cell tools write.
//!
//! * [`from_mtx`]: a MatrixMarket file with its feature and barcode TSVs,
//!   such as a 10x Cell Ranger MEX triplet ([`MtxImport`]).
//! * [`from_h5ad`] (feature `hdf5`): an AnnData `.h5ad` file
//!   ([`H5adImport`]), its `obs` written beside the backend as
//!   `{stem}.cell_metadata.tsv.gz`.
//! * [`from_10x_matrix`] (feature `hdf5`): a Cell Ranger feature-barcode
//!   matrix `.h5` ([`TenxMatrixImport`]).
//! * [`from_10x_molecule`] (feature `hdf5`): a Cell Ranger
//!   `molecule_info.h5`, counted as Cell Ranger counts it
//!   ([`TenxMoleculeImport`]).
//!
//! Each writes the backend [`Output`] names, squeezed when it asks, and
//! returns the path of the file written: `{output}.zarr.zip` by default. The
//! `stage_*` functions stop short of the last step and return a [`Staged`]
//! build, for a caller that looks at the backend before keeping it (as the
//! CLI's import preview does). These are what `data-beans from-mtx`,
//! `from-h5ad`, `from-10x-matrix` and `from-10x-molecule` run.
//!
//! ```no_run
//! use data_beans::import::{self, MtxImport, Output};
//!
//! let mtx = MtxImport {
//!     row: Some("filtered/features.tsv.gz".into()),
//!     col: Some("filtered/barcodes.tsv.gz".into()),
//!     ..MtxImport::new("filtered/matrix.mtx.gz")
//! };
//! let backend = import::from_mtx(&mtx, &Output::new("pbmc"))?; // "pbmc.zarr.zip"
//!
//! #[cfg(feature = "hdf5")]
//! let backend = import::from_h5ad(&import::H5adImport::new("pbmc.h5ad"), &Output::new("pbmc"))?;
//! # anyhow::Ok(())
//! ```

#[cfg(feature = "hdf5")]
mod h5ad;
mod mtx;
#[cfg(feature = "hdf5")]
mod tenx_matrix;
#[cfg(feature = "hdf5")]
mod tenx_molecule;

#[cfg(feature = "hdf5")]
pub use h5ad::{from_h5ad, stage_h5ad, H5adImport};
pub use mtx::{from_mtx, stage_mtx, MtxImport};
#[cfg(feature = "hdf5")]
pub use tenx_matrix::{from_10x_matrix, stage_10x_matrix, TenxMatrixImport};
#[cfg(feature = "hdf5")]
pub use tenx_molecule::{from_10x_molecule, stage_10x_molecule, TenxMoleculeImport};

use crate::sparse_io::{open_sparse_matrix_by_path, SparseIoBackend};
use log::info;

/// Where an import writes its backend.
#[derive(Clone, Debug)]
pub struct Output {
    /// The output's header or name: `{output}.zarr.zip`, `{output}.zarr` or
    /// `{output}.h5`, a backend suffix it already has kept. Files written
    /// beside the backend (metadata tables) are named by its stem.
    pub output: Box<str>,
    pub backend: SparseIoBackend,
    /// Zip a zarr backend into `.zarr.zip`; else keep the `.zarr` directory.
    pub zip: bool,
    /// Drop rows and columns with too few non-zeros once it is written.
    pub squeeze: Option<Squeeze>,
}

impl Output {
    /// `{output}.zarr.zip`, not squeezed: the CLI's defaults.
    pub fn new(output: &str) -> Self {
        Self {
            output: output.into(),
            backend: SparseIoBackend::Zarr,
            zip: true,
            squeeze: None,
        }
    }

    /// Clear what a previous run left at the working file, and stage the
    /// build there.
    fn stage(&self) -> anyhow::Result<(Staged, SparseIoBackend)> {
        let (effective, backend, working) =
            crate::zarr_io::prepare_output(&self.output, self.backend.clone(), self.zip)?;
        let squeeze = self.squeeze;
        Ok((
            Staged {
                effective,
                working,
                squeeze,
            },
            backend,
        ))
    }
}

/// A squeeze: the rows and columns with fewer non-zeros than these go.
#[derive(Clone, Copy, Debug)]
pub struct Squeeze {
    pub rows: usize,
    pub columns: usize,
    /// Cells per job; `None` scales it by the number of features.
    pub block_size: Option<usize>,
}

impl Default for Squeeze {
    /// Rows and columns without a single non-zero go.
    fn default() -> Self {
        Self {
            rows: 1,
            columns: 1,
            block_size: None,
        }
    }
}

impl Squeeze {
    /// Squeeze the backend `backend_file` in place.
    pub fn apply(&self, backend_file: &str) -> anyhow::Result<()> {
        crate::qc::squeeze_in_place(backend_file, self.rows, self.columns, self.block_size)
    }
}

/// A backend written but not yet in place: for a `.zarr.zip` output, the
/// `.zarr` directory it is zipped from. [`finish`](Self::finish) puts it in
/// place; [`discard`](Self::discard) deletes it.
#[derive(Debug)]
#[must_use = "a staged backend is only in place once finished"]
pub struct Staged {
    /// The output as named, `.zarr.zip` added under zip.
    effective: Box<str>,
    /// The file the backend was written to.
    working: Box<str>,
    /// The squeeze [`Output`] asked for, applied when it is finished.
    squeeze: Option<Squeeze>,
}

impl Staged {
    /// The backend as written, to open and read before it is finished.
    pub fn working(&self) -> &str {
        &self.working
    }

    /// Squeeze the backend as its [`Output`] asked, and put it in place;
    /// the path of the file to open it by.
    pub fn finish(self) -> anyhow::Result<Box<str>> {
        if let Some(squeeze) = &self.squeeze {
            squeeze.apply(&self.working)?;
        }
        let found = crate::zarr_io::finalize_output(&self.working, &self.effective)?;
        info!("done");
        Ok(found.into())
    }

    /// Delete the backend, kept nowhere.
    pub fn discard(self) -> anyhow::Result<()> {
        remove_build(&self.working)
    }
}

/// Delete the backend written at `working`, and the stage it was built in
/// once nothing else is built there.
pub(crate) fn remove_build(working: &str) -> anyhow::Result<()> {
    info!("deleting {working}");
    open_sparse_matrix_by_path(working)?.remove_backend_file()?;
    crate::zarr_io::leave_stage(working);
    Ok(())
}

/// Log how many rows of each feature type `label` lists.
pub(crate) fn log_feature_type_histogram(label: &str, row_types: &[Box<str>]) {
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for t in row_types {
        *counts.entry(t.as_ref()).or_insert(0) += 1;
    }
    info!("Feature types in {}: {:?}", label, counts);
}

#[cfg(test)]
mod tests;
