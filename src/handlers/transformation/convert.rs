use super::prepare_output_beside;
use crate::hdf5_io::*;
use crate::sparse_io::*;
use crate::zarr_io::{finalize_output, materialize_writable_backend};

use clap::Args;
use log::info;

#[derive(Args, Debug)]
pub struct ConvertArgs {
    /// input data file -- `.zarr`, `.zarr.zip`, or `.h5`
    pub data_file: Box<str>,

    /// backend for the output file
    #[arg(long, value_enum, default_value = "zarr")]
    pub backend: SparseIoBackend,

    /// output header: {output}.zarr.zip by default; pass --no-zip to keep a {output}.zarr directory
    #[arg(short, long)]
    pub output: Box<str>,

    /// keep a `.zarr` directory instead of producing a `.zarr.zip` archive
    #[arg(long = "no-zip", default_value_t = true, action = clap::ArgAction::SetFalse)]
    pub zip: bool,
}

/// Convert a backend to a different on-disk format (`zarr` <-> `h5`,
/// `.zarr` <-> `.zarr.zip`), preserving the matrix and row/column names.
///
/// When the source and target share the same backend the data is copied
/// (or unzipped / re-zipped) directly; a cross-backend conversion streams the
/// columns into the other format, a slab at a time.
pub fn run_convert(args: &ConvertArgs) -> anyhow::Result<()> {
    let (backend_in, file_in) = resolve_backend_file(&args.data_file, None)?;
    let (effective_output, backend_out, file_out) = prepare_output_beside(
        &args.data_file,
        &args.output,
        args.backend.clone(),
        args.zip,
    )?;

    if backend_in == backend_out {
        // Same on-disk format: copy the file / extract the archive directly.
        // `finalize_zarr_output` re-zips afterwards when the target is `.zarr.zip`.
        info!(
            "same backend ({:?}); staging {} -> {}",
            backend_in, file_in, file_out
        );
        materialize_writable_backend(&file_in, &file_out)?;
    } else {
        info!(
            "re-encoding {} ({:?}) -> {} ({:?})",
            file_in, backend_in, file_out, backend_out
        );
        let data = open_sparse_matrix(&file_in, &backend_in)?;
        let ncol = data
            .num_columns()
            .ok_or_else(|| anyhow::anyhow!("backend has no `ncol`"))?;
        let every_column: Vec<usize> = (0..ncol).collect();
        let (nrow, ncol, nnz) = crate::column_subset::stream_column_selection(
            data.as_ref(),
            &every_column,
            None,
            &data.row_names()?,
            &data.column_names()?,
            file_out.as_ref(),
            &backend_out,
        )?;
        info!("re-encoded {} non-zeros in {} x {}", nnz, nrow, ncol);
    }

    let final_path = finalize_output(&file_out, &effective_output)?;
    info!("done: {}", final_path);
    Ok(())
}

#[cfg(test)]
#[path = "convert/tests.rs"]
mod tests;
