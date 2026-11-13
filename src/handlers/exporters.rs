// `to-mtx` and `to-h5ad`: the command-line face of `data_beans::export` (`crate::export`).
// The h5ad exporter needs libhdf5, so it compiles out without the feature.

use crate::export::{to_mtx, MtxExport};
use crate::sparse_io::open_sparse_matrix_by_path;

use clap::Args;
use log::info;

#[derive(Args, Debug)]
pub struct ToMtxArgs {
    /// backend data file to export (`.zarr`, `.zarr.zip`, or `.h5`)
    pub data_file: Box<str>,

    #[arg(
        short,
        long,
        help = "Output directory for the 10x-style triplet",
        long_help = "Output directory. The 10x Cell Ranger MEX triplet is written inside it:\n\
                     matrix.mtx.gz, features.tsv.gz and barcodes.tsv.gz.\n\
                     `scanpy.read_10x_mtx(dir)` and Seurat `Read10X(dir)` load it.\n\
                     The directory is created if missing."
    )]
    pub output: Box<str>,

    /// keep the triplet uncompressed (`matrix.mtx` / `features.tsv` /
    /// `barcodes.tsv`) instead of the default gzipped Cell Ranger v3 layout
    #[arg(long = "no-gzip", default_value_t = true, action = clap::ArgAction::SetFalse)]
    pub gzip: bool,

    #[arg(
        long,
        default_value = "Gene Expression",
        help = "feature_type written to column 3 of features.tsv",
        long_help = "Value written to the third column of features.tsv.\n\
                     That column is feature_type, and every feature gets this value.\n\
                     10x uses 'Gene Expression', 'Antibody Capture' or 'Peaks'.\n\
                     `from-mtx` reads it back for --select-row-type filtering."
    )]
    pub feature_type: Box<str>,
}

/// `to-mtx`: [`to_mtx`] of the backend `data_file`.
pub fn run_export_to_mtx(args: &ToMtxArgs) -> anyhow::Result<()> {
    info!("Exporting {}", args.data_file);
    let data = open_sparse_matrix_by_path(&args.data_file)?;
    let opts = MtxExport {
        gzip: args.gzip,
        feature_type: args.feature_type.clone(),
    };
    to_mtx(data.as_ref(), &args.output, &opts)?;
    Ok(())
}

#[cfg(feature = "hdf5")]
pub use h5ad::*;

#[cfg(feature = "hdf5")]
mod h5ad {
    use crate::export::{h5ad_path, to_h5ad, H5adExport};
    use crate::sparse_io::open_sparse_matrix_by_path;

    use clap::Args;
    use log::info;

    #[derive(Args, Debug)]
    pub struct ToH5adArgs {
        /// backend data file to export (`.zarr`, `.zarr.zip`, or `.h5`)
        pub data_file: Box<str>,

        #[arg(
            short,
            long,
            help = "Output `.h5ad` file (default: <input stem>.h5ad)",
            long_help = "Output AnnData h5ad path.\n\
                         `.h5ad` is appended when the name lacks that suffix.\n\
                         It defaults to the input file's stem, plus `.h5ad`."
        )]
        pub output: Option<Box<str>>,

        #[arg(
            long,
            help = "Cell-metadata TSV to attach as `obs` columns",
            long_help = "Optional TSV whose first column is the cell key; gzip is fine.\n\
                         Keys match against the backend column names.\n\
                         A cell named `barcode@donor` also matches the row keyed `barcode`\n\
                         whose `donor_id` column is `donor`, or else the row keyed `barcode`.\n\
                         \n\
                         Remaining columns become `obs` columns. Unmatched cells get `NA`,\n\
                         as do cells whose key is listed on rows that differ.\n\
                         \n\
                         Pair this with the `*.cell_metadata.tsv.gz` that `from-h5ad` wrote,\n\
                         to round-trip cell annotations."
        )]
        pub obs: Option<Box<str>>,

        #[arg(
            long,
            help = "Feature-metadata TSV to attach as `var` columns",
            long_help = "Optional TSV whose first column is the feature key; gzip is fine.\n\
                         Keys match against the backend row names.\n\
                         Remaining columns become `var` columns. Unmatched features get `NA`,\n\
                         as do features whose key is listed on rows that differ."
        )]
        pub var: Option<Box<str>>,
    }

    /// `to-h5ad`: [`to_h5ad`] of the backend `data_file`, to `--output` or
    /// beside it ([`h5ad_path`]).
    pub fn run_export_to_h5ad(args: &ToH5adArgs) -> anyhow::Result<()> {
        let output = h5ad_path(&args.data_file, args.output.as_deref());
        info!("Exporting {}", args.data_file);
        let mut data = open_sparse_matrix_by_path(&args.data_file)?;
        let opts = H5adExport {
            obs: args.obs.clone(),
            var: args.var.clone(),
        };
        to_h5ad(data.as_mut(), &output, &opts)
    }
}
