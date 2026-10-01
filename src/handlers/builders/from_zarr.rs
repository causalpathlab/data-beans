use super::run_squeeze_if_needed;
use crate::convert::{build_from_zarr_matrix, MatrixLayout};
use crate::sparse_io::*;
use crate::sparse_util::*;
use crate::zarr_io::*;

use log::info;

#[derive(clap::Args, Debug)]
pub struct FromZarrArgs {
    #[arg(
        help = "Input Zarr file containing sparse matrix triplets",
        long_help = "Specify the Zarr file where triplets of sparse matrix data are stored.\n\
                     For example, 10X Genomics Xenium's 'cell_feature_matrix.zarr.zip'."
    )]
    pub zarr_file: Box<str>,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Backend format for output",
        long_help = "Choose the backend format for the output file.\n\
                     Supported formats include 'zarr' and 'h5'."
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
        short = 'd',
        long,
        default_value = MatrixLayout::XENIUM_DATA,
        help = "Data field path",
        long_help = "Path to the dataset containing triplet values.\n\
                     Use the 'list-zarr' subcommand to inspect available fields."
    )]
    pub data_field: Box<str>,

    #[arg(
        short = 'i',
        long,
        default_value = MatrixLayout::XENIUM_INDICES,
        help = "Indices field path",
        long_help = "Path to the dataset containing indices. Row indices for CSC,\n\
                     column indices for CSR."
    )]
    pub indices_field: Box<str>,

    #[arg(
        short = 'p',
        long,
        default_value = MatrixLayout::XENIUM_INDPTR,
        help = "Indptr field path",
        long_help = "Path to the dataset containing indptr. Column pointers for CSC,\n\
                     row pointers for CSR."
    )]
    pub indptr_field: Box<str>,

    #[arg(
        short = 't',
        long,
        value_enum,
        default_value = "row",
        help = "Pointer type (row or column)",
        long_help = "Specify whether the pointers keep track of row (gene) or column (cell) indices."
    )]
    pub pointer_type: IndexPointerType,

    #[arg(
        short = 'r',
        long,
        default_value = MatrixLayout::XENIUM_ROW_IDS,
        help = "Row ID field path",
        long_help = "Path to the group or dataset for row, gene, or feature IDs."
    )]
    pub row_id_field: Box<str>,

    #[arg(
        short = 'n',
        long,
        default_value = MatrixLayout::XENIUM_ROW_NAMES,
        help = "Row name field path",
        long_help = "Path to the group or dataset for row, gene, or feature names."
    )]
    pub row_name_field: Box<str>,

    #[arg(
        short = 'f',
        long,
        default_value = MatrixLayout::XENIUM_ROW_TYPES,
        help = "Row type field path",
        long_help = "Path to the group or dataset for row, gene, or feature types."
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
        default_value = MatrixLayout::XENIUM_COLUMN_NAMES,
        help = "Column name field path",
        long_help = "Path to the group or dataset for columns or cells.\n\
                     Will first attempt Xenium's Cell ID format mapping."
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
pub fn run_build_from_zarr_triplets(args: &FromZarrArgs) -> anyhow::Result<()> {
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
    build_from_zarr_matrix(&args.zarr_file, &layout, &backend_file, &backend)?;

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
