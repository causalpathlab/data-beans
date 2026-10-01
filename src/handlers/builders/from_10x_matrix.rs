use super::run_squeeze_if_needed;
use crate::convert::{build_from_h5_matrix, MatrixLayout};
use crate::sparse_io::*;
use crate::sparse_util::*;
use crate::zarr_io::*;

use clap::Args;
use log::info;

#[derive(Args, Debug)]
pub struct From10xMatrixArgs {
    #[arg(
        help = "Input HDF5 file containing sparse matrix triplets",
        long_help = "Specify the HDF5 file where triplets of sparse matrix data are stored.\n\
                     Supports 10X Genomics and H5AD formats."
    )]
    pub h5_file: Box<str>,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Backend format for output",
        long_help = "Choose the backend format for the output file.\n\
                     Supported formats include 'zarr' and 'h5'"
    )]
    pub backend: SparseIoBackend,

    #[arg(
        short,
        long,
        help = "Output file header or name",
        long_help = "Specify the output file header.\n\
                     The zarr backend produces {output}.zarr.zip by default;\n\
                     pass --no-zip to keep a {output}.zarr directory instead.\n\
                     Redundant {backend} names will be ignored."
    )]
    pub output: Box<str>,

    /// keep a `.zarr` directory instead of producing a `.zarr.zip` archive
    #[arg(long = "no-zip", default_value_t = true, action = clap::ArgAction::SetFalse)]
    pub zip: bool,

    #[arg(
        short = 'x',
        long,
        default_value = MatrixLayout::H5_ROOT,
        help = "Root group name for sparse data triplets",
        long_help = "Set the root group name under which sparse data triplets are stored in the HDF5 file.\n\
                     Use the 'list-h5' command to inspect available groups."
    )]
    pub root_group_name: Box<str>,

    #[arg(
        short = 'd',
        long,
        default_value = MatrixLayout::H5_DATA,
        help = "Data field name",
        long_help = "Name of the dataset containing triplet values X(i,j) under the root group."
    )]
    pub data_field: Box<str>,

    #[arg(
        short = 'i',
        long,
        default_value = MatrixLayout::H5_INDICES,
        help = "Indices field name",
        long_help = "Name of the dataset containing indices. Row indices for CSC,\n\
                     column indices for CSR, under the root group."
    )]
    pub indices_field: Box<str>,

    #[arg(
        short = 'p',
        long,
        default_value = MatrixLayout::H5_INDPTR,
        help = "Indptr field name",
        long_help = "Name of the dataset containing indptr. Column pointers for CSC,\n\
                     row pointers for CSR, under the root group."
    )]
    pub indptr_field: Box<str>,

    #[arg(
        short = 't',
        long,
        value_enum,
        default_value = "column",
        help = "Pointer type (row or column)",
        long_help = "Specify whether the pointers are for row (gene) or column (cell) indices."
    )]
    pub pointer_type: IndexPointerType,

    #[arg(
        short = 'r',
        long,
        default_value = MatrixLayout::H5_ROW_IDS,
        help = "Row ID field name",
        long_help = "Group or dataset name for row, gene, or feature IDs under the root group."
    )]
    pub row_id_field: Box<str>,

    #[arg(
        short = 'n',
        long,
        default_value = MatrixLayout::H5_ROW_NAMES,
        help = "Row name field name",
        long_help = "Group or dataset name for row, gene,\n\
                     or feature names under the root group."
    )]
    pub row_name_field: Box<str>,

    #[arg(
        short = 'f',
        long,
        default_value = MatrixLayout::H5_ROW_TYPES,
        help = "Row type field name",
        long_help = "Group or dataset name for row, gene,\n\
                     or feature types under the root group."
    )]
    pub row_type_field: Box<str>,

    #[arg(
        long,
        default_value = MatrixLayout::SELECT_ROW_TYPES,
        help = "Select row type (comma-separated patterns; ANY match keeps the row)",
        long_help = "Select which row types to include. Patterns are comma-separated,\n\
                     case-insensitive substrings.\n\
                     A row is kept if its type contains any pattern.\n\
                     The default 'gene,peak' keeps Gene Expression and ATAC Peaks."
    )]
    pub select_row_type: Box<str>,

    #[arg(
        long,
        default_value = MatrixLayout::REMOVE_ROW_TYPES,
        help = "Remove row type (comma-separated patterns; ANY match drops the row)",
        long_help = "Remove rows if their type contains any of these comma-separated patterns."
    )]
    pub remove_row_type: Box<str>,

    #[arg(
        short = 'c',
        long,
        default_value = MatrixLayout::H5_COLUMN_NAMES,
        help = "Column name field",
        long_help = "Group or dataset name for columns or cells under the root group."
    )]
    pub column_name_field: Box<str>,

    #[arg(
        long,
        default_value_t = false,
        help = "Squeeze sparse rows or columns",
        long_help = "Enable squeezing to remove rows and columns with too few non-zeros.\n\
                     This can help reduce file size and improve performance."
    )]
    pub do_squeeze: bool,

    #[arg(
        long,
        default_value_t = 1,
        help = "Row non-zero cutoff",
        long_help = "Minimum number of non-zero elements required for rows.\n\
                     Rows with fewer non-zeros will be removed if squeezing is enabled."
    )]
    pub row_nnz_cutoff: usize,

    #[arg(
        long,
        default_value_t = 1,
        help = "Column non-zero cutoff",
        long_help = "Minimum number of non-zero elements required for columns.\n\
                     Columns with fewer non-zeros will be removed if squeezing is enabled."
    )]
    pub column_nnz_cutoff: usize,

    #[arg(
        long,
        help = "Cells per rayon job for the post-build squeeze pass",
        long_help = "Cells per rayon job for the post-build squeeze pass.\n\
                     Omit it for auto-scaling by feature count."
    )]
    pub block_size: Option<usize>,
}
pub fn run_build_from_10x_matrix(args: &From10xMatrixArgs) -> anyhow::Result<()> {
    let (effective_output, backend, backend_file) =
        prepare_output(&args.output, args.backend.clone(), args.zip)?;

    let layout = MatrixLayout {
        data_field: args.data_field.clone(),
        indices_field: args.indices_field.clone(),
        indptr_field: args.indptr_field.clone(),
        pointer_type: args.pointer_type,
        row_id_field: args.row_id_field.clone(),
        row_name_field: args.row_name_field.clone(),
        row_type_field: args.row_type_field.clone(),
        select_row_type: args.select_row_type.clone(),
        remove_row_type: args.remove_row_type.clone(),
        column_name_field: args.column_name_field.clone(),
    };
    build_from_h5_matrix(
        &args.h5_file,
        &args.root_group_name,
        &layout,
        &backend_file,
        &backend,
    )?;

    run_squeeze_if_needed(
        args.do_squeeze,
        args.row_nnz_cutoff,
        args.column_nnz_cutoff,
        args.block_size,
        &backend_file,
    )?;
    finalize_output(&backend_file, &effective_output)?;
    info!("done");
    Ok(())
}
