//! Write a backend out to the interchange formats scanpy and Seurat read.
//!
//! * [`to_mtx`]: a 10x Genomics Cell Ranger MEX triplet directory
//!   (`matrix.mtx`, `features.tsv`, `barcodes.tsv`, gzipped by default),
//!   loaded by `scanpy.read_10x_mtx(dir)` and Seurat `Read10X(dir)`.
//! * [`to_h5ad`] (feature `hdf5`): an AnnData `.h5ad` file, loaded by
//!   `anndata.read_h5ad`, with optional `obs` / `var` metadata tables.
//!
//! Both take a backend already open, so a caller holding one does not open
//! it again; [`open_sparse_matrix_by_path`](crate::sparse_io::open_sparse_matrix_by_path)
//! opens one from its file. These are what `data-beans to-mtx` and
//! `data-beans to-h5ad` run, and write the same bytes.
//!
//! ```no_run
//! use data_beans::export::{self, MtxExport};
//! use data_beans::sparse_io::open_sparse_matrix_by_path;
//!
//! let data = open_sparse_matrix_by_path("sim.zarr.zip")?;
//! // sim_mtx/{matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz}
//! export::to_mtx(data.as_ref(), "sim_mtx", &MtxExport::default())?;
//!
//! #[cfg(feature = "hdf5")]
//! {
//!     let mut data = data;
//!     let out = export::h5ad_path("sim.zarr.zip", None); // "sim.h5ad"
//!     export::to_h5ad(data.as_mut(), &out, &export::H5adExport::default())?;
//! }
//! # anyhow::Ok(())
//! ```

#[cfg(feature = "hdf5")]
pub(crate) mod h5ad;
mod mtx;

#[cfg(feature = "hdf5")]
pub use h5ad::{h5ad_path, to_h5ad, H5adExport};
pub use mtx::{to_mtx, MtxExport};
