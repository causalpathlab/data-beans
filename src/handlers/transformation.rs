use crate::hdf5_io::resolve_backend_file;
use crate::qc::{squeeze_by_nnz, SqueezeCutoffs};
use crate::sparse_io::open_sparse_matrix;
use clap::Args;
use log::info;

mod convert;
mod split;
mod subsample;
mod subset;

pub use convert::*;
pub use split::*;
pub use subsample::*;
pub use subset::*;

#[derive(Args, Debug)]
pub struct ReorderRowsArgs {
    /// Data file -- either `.zarr` or `.h5`
    pub data_file: Box<str>,

    /// Row/feature name file (name per each line; `.tsv.gz` or `.tsv`)
    #[arg(short, long, required = true)]
    pub row_file: Box<str>,

    /// output header
    #[arg(short, long, required = true)]
    pub output: Box<str>,

    /// keep a `.zarr` directory instead of producing a `.zarr.zip` archive
    #[arg(long = "no-zip", default_value_t = true, action = clap::ArgAction::SetFalse)]
    pub zip: bool,
}

#[derive(Args, Debug)]
pub struct SubsetArgs {
    /// data file -- `.zarr`, `.zarr.zip`, or `.h5`
    pub data_file: Box<str>,

    /// output file, in the input's format unless the name gives `.zarr` or
    /// `.zarr.zip`
    #[arg(short, long, required = true)]
    pub output: Box<str>,

    /// never open the explorer; subset only picks there, so it stops,
    /// naming subset-rows and subset-columns, which take a selection
    #[arg(long, default_value_t = false)]
    pub batch_process: bool,

    #[command(flatten)]
    pub session: legume_tui::cli::SessionArgs,
}

#[derive(Args, Debug)]
pub struct SubsetColumnsArgs {
    /// data file -- either `.zarr` or `.h5`
    pub data_file: Box<str>,

    /// column indices to take: singles and inclusive ranges, e.g. `0,1,2,3`,
    /// `1-10`, or `1-20,50-55` (past the last column, cut)
    #[arg(short = 'i', long, conflicts_with_all = ["name_file", "pattern"])]
    pub column_indices: Option<Box<str>>,

    /// column name file where each line is a column name
    #[arg(short = 'f', long, conflicts_with = "pattern")]
    pub name_file: Option<Box<str>>,

    /// the columns whose names this regex finds (case-insensitive), e.g.
    /// `/count/spliced$` or `^MT-`
    #[arg(long)]
    pub pattern: Option<Box<str>>,

    /// never open the explorer: what to keep is given on the command line
    #[arg(long, default_value_t = false)]
    pub batch_process: bool,

    #[command(flatten)]
    pub session: legume_tui::cli::SessionArgs,

    /// delimiter for base-key extraction (e.g., '@' to match "ACGT-1@batch" with "ACGT-1")
    #[arg(short = 'd', long, default_value = "@")]
    pub delimiter: char,

    /// enable prefix matching (stored name is prefix of query or vice
    /// versa); `--allow-prefix=false` turns it off, as `--exact-names` does
    #[arg(
        long,
        action = clap::ArgAction::Set,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "true",
        default_value_t = true
    )]
    pub allow_prefix: bool,

    /// Match names exactly (plus the delimiter base key): no prefix matching.
    /// `--allow-prefix` defaults to on and, being a bare flag, cannot be turned
    /// off from the command line; this is the off switch. Numeric or short gene
    /// names are prefixes of many others, so a name list can otherwise keep far
    /// more rows than it names.
    #[arg(long, default_value_t = false)]
    pub exact_names: bool,

    /// squeeze
    #[arg(long, default_value_t = false)]
    pub do_squeeze: bool,

    /// minimum number of non-zero cutoff for rows
    #[arg(long, default_value_t = 1)]
    pub row_nnz_cutoff: usize,

    /// minimum number of non-zero cutoff for columns
    #[arg(long, default_value_t = 1)]
    pub column_nnz_cutoff: usize,

    /// output file
    #[arg(short, long, required = true)]
    pub output: Box<str>,

    /// keep a `.zarr` directory instead of producing a `.zarr.zip` archive
    #[arg(long = "no-zip", default_value_t = true, action = clap::ArgAction::SetFalse)]
    pub zip: bool,
}

#[derive(Args, Debug)]
pub struct SubsetRowsArgs {
    /// data file -- either `.zarr` or `.h5`
    pub data_file: Box<str>,

    /// row indices to take: singles and inclusive ranges, e.g. `0,1,2,3`,
    /// `1-10`, or `1-20,50-55` (past the last row, cut)
    #[arg(short = 'i', long, conflicts_with_all = ["name_file", "pattern"])]
    pub row_indices: Option<Box<str>>,

    /// row name file where each line is a row name
    #[arg(short = 'f', long, conflicts_with = "pattern")]
    pub name_file: Option<Box<str>>,

    /// the rows whose names this regex finds (case-insensitive), e.g.
    /// `/count/spliced$` or `^MT-`
    #[arg(long)]
    pub pattern: Option<Box<str>>,

    /// never open the explorer: what to keep is given on the command line
    #[arg(long, default_value_t = false)]
    pub batch_process: bool,

    #[command(flatten)]
    pub session: legume_tui::cli::SessionArgs,

    /// delimiter for base-key extraction (e.g., '@' to match "gene@batch" with "gene")
    #[arg(short = 'd', long, default_value = "@")]
    pub delimiter: char,

    /// enable prefix matching (stored name is prefix of query or vice
    /// versa); `--allow-prefix=false` turns it off, as `--exact-names` does
    #[arg(
        long,
        action = clap::ArgAction::Set,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "true",
        default_value_t = true
    )]
    pub allow_prefix: bool,

    /// Match names exactly (plus the delimiter base key): no prefix matching.
    /// `--allow-prefix` defaults to on and, being a bare flag, cannot be turned
    /// off from the command line; this is the off switch. Numeric or short gene
    /// names are prefixes of many others, so a name list can otherwise keep far
    /// more rows than it names.
    #[arg(long, default_value_t = false)]
    pub exact_names: bool,

    /// squeeze
    #[arg(long, default_value_t = false)]
    pub do_squeeze: bool,

    /// minimum number of non-zero cutoff for rows
    #[arg(long, default_value_t = 1)]
    pub row_nnz_cutoff: usize,

    /// minimum number of non-zero cutoff for columns
    #[arg(long, default_value_t = 1)]
    pub column_nnz_cutoff: usize,

    /// output file
    #[arg(short, long, required = true)]
    pub output: Box<str>,

    /// keep a `.zarr` directory instead of producing a `.zarr.zip` archive
    #[arg(long = "no-zip", default_value_t = true, action = clap::ArgAction::SetFalse)]
    pub zip: bool,
}

/// Cut, in place, the rows and columns of the backend at `backend_file`
/// with fewer than `row` and `column` non-zeros: the `--squeeze` of the
/// commands that build or derive a backend. The path is one the caller
/// resolved and finalizes itself, so nothing is zipped here.
pub fn squeeze_in_place(
    backend_file: &str,
    row: usize,
    column: usize,
    block_size: Option<usize>,
) -> anyhow::Result<()> {
    info!("Squeeze the backend data {backend_file}");
    let (backend, file) = resolve_backend_file(backend_file, None)?;
    let data = open_sparse_matrix(&file, &backend)?;
    squeeze_by_nnz(
        data.as_ref(),
        SqueezeCutoffs { row, column },
        block_size,
        true,
    )?;
    let data = open_sparse_matrix(&file, &backend)?;
    info!(
        "after squeeze -- data: {} rows x {} columns",
        data.num_rows().unwrap_or_default(),
        data.num_columns().unwrap_or_default()
    );
    Ok(())
}
