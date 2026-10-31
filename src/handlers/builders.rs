use log::info;

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

pub(super) fn run_squeeze_if_needed(
    do_squeeze: bool,
    row_nnz_cutoff: usize,
    column_nnz_cutoff: usize,
    block_size: Option<usize>,
    backend_file: &str,
) -> anyhow::Result<()> {
    if do_squeeze {
        crate::handlers::transformation::squeeze_in_place(
            backend_file,
            row_nnz_cutoff,
            column_nnz_cutoff,
            block_size,
        )?;
    }
    Ok(())
}
