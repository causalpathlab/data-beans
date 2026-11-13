//! `from-mtx`, `from-h5ad`, `from-10x-matrix` and `from-10x-molecule`: the
//! command-line face of `data_beans::import`. Each turns its flags into the
//! library's options, stages the build there, and keeps it as
//! [`run_import`] does: after the import preview on a terminal.

use super::{run_import, Squeeze};
use crate::import::{self, MtxImport, Output};
use crate::sparse_io::SparseIoBackend;

use clap::Args;

/// The library's [`Output`] for `-o`, `--backend` and `--no-zip`;
/// [`run_import`] adds the squeeze.
fn output(output: &str, backend: &SparseIoBackend, zip: bool) -> Output {
    Output {
        backend: backend.clone(),
        zip,
        ..Output::new(output)
    }
}

#[derive(Args, Debug)]
pub struct FromMtxArgs {
    /// matrix market-formatted data file (`.mtx.gz` or `.mtx`)
    pub mtx: Box<str>,

    /// row/feature name file (name per each line; `.tsv.gz` or `.tsv`).
    /// For 10x `features.tsv[.gz]` the columns are `id`, `name`, `feature_type`;
    /// the third column, when present, is used for `--select-row-type` /
    /// `--remove-row-type` / `--hto-row-type` filtering.
    #[arg(short, long)]
    pub row: Option<Box<str>>,

    /// column/cell/barcode file (name per each line; `.tsv.gz` or `.tsv`)
    #[arg(short, long)]
    pub col: Option<Box<str>>,

    /// backend for the output file
    #[arg(long, value_enum, default_value = "zarr")]
    pub backend: SparseIoBackend,

    /// output file header: {output}.zarr.zip by default; pass --no-zip to keep a {output}.zarr directory
    #[arg(short, long)]
    pub output: Box<str>,

    /// keep a `.zarr` directory instead of producing a `.zarr.zip` archive
    #[arg(long = "no-zip", default_value_t = true, action = clap::ArgAction::SetFalse)]
    pub zip: bool,

    /// maximum number of columns to read from the row/feature name file
    /// (columns are joined with '_'); e.g. 2 reads "ENSG…<tab>SYMBOL"
    #[arg(long, default_value_t = 2)]
    pub row_name_columns: usize,

    /// Comma-separated patterns; row is kept if its feature_type (column 3
    /// of `--row`) contains ANY of them (case-insensitive). Default
    /// `Gene Expression,Peaks` covers both 10x scRNA and 10x ATAC.
    /// Set to empty to keep everything.
    #[arg(long, default_value = "Gene Expression,Peaks")]
    pub select_row_type: Box<str>,

    /// Drop rows whose feature_type contains this string. Applied after
    /// `--select-row-type`.
    #[arg(long, default_value = "")]
    pub remove_row_type: Box<str>,

    /// Feature type used as HTO/multiplexing tags (e.g. `Antibody Capture`).
    /// When set, each cell is assigned a batch label `barcode@{id}_{name}`
    /// based on the HTO row with the highest count. HTO rows are then
    /// removed from the final backend and cells with zero HTO signal are
    /// dropped.
    #[arg(long, default_value = "")]
    pub hto_row_type: Box<str>,

    /// squeeze
    #[arg(long, default_value_t = false)]
    pub do_squeeze: bool,

    /// minimum number of non-zero cutoff for rows
    #[arg(long, default_value_t = 1)]
    pub row_nnz_cutoff: usize,

    /// minimum number of non-zero cutoff for columns
    #[arg(long, default_value_t = 1)]
    pub column_nnz_cutoff: usize,

    /// Cells per rayon job for the post-build squeeze pass.
    /// Omit for auto-scaling by feature count.
    #[arg(long)]
    pub block_size: Option<usize>,

    #[command(flatten)]
    pub view: crate::handlers::explore::PlanArgs,
}

/// `from-mtx`: [`import::from_mtx`].
pub fn run_build_from_mtx(args: &FromMtxArgs) -> anyhow::Result<()> {
    // The session first: hosted, the build runs there.
    let session = args.view.session("from-mtx")?;
    let mtx = MtxImport {
        row: args.row.clone(),
        col: args.col.clone(),
        row_name_columns: args.row_name_columns,
        select_row_type: args.select_row_type.clone(),
        remove_row_type: args.remove_row_type.clone(),
        hto_row_type: args.hto_row_type.clone(),
        ..MtxImport::new(&args.mtx)
    };
    let out = output(&args.output, &args.backend, args.zip);
    let squeeze = Squeeze::new(
        args.do_squeeze,
        args.row_nnz_cutoff,
        args.column_nnz_cutoff,
        args.block_size,
    );
    run_import(
        "from-mtx",
        session.as_ref(),
        &mtx,
        out,
        squeeze,
        import::from_mtx,
        import::stage_mtx,
    )
}

#[cfg(feature = "hdf5")]
pub use hdf5_formats::*;

#[cfg(feature = "hdf5")]
mod hdf5_formats {
    use super::{output, run_import, Squeeze};
    use crate::convert::MatrixLayout;
    use crate::import::{self, H5adImport, TenxMatrixImport, TenxMoleculeImport};
    use crate::sparse_io::SparseIoBackend;
    use crate::sparse_util::IndexPointerType;

    use clap::Args;

    #[derive(Args, Debug)]
    pub struct FromH5adArgs {
        #[arg(
            help = "Input AnnData h5ad file",
            long_help = "Specify the AnnData h5ad file (CELLxGENE schema v7). Expected layout:\n\
                         X/{data,indices,indptr}, obs/, var/. Prefers raw/X over X when available."
        )]
        pub h5ad_file: Box<str>,

        #[arg(
            long,
            value_enum,
            default_value = "zarr",
            help = "Backend format for output",
            long_help = "Choose the backend format for the output file."
        )]
        pub backend: SparseIoBackend,

        #[arg(
            short,
            long,
            help = "Output file header or name",
            long_help = "Specify the output file header.\n\
                         The zarr backend produces {output}.zarr.zip by default;\n\
                         pass --no-zip to keep a {output}.zarr directory instead.\n\
                         Metadata files will be named {output}.cell_metadata.tsv.gz, etc."
        )]
        pub output: Box<str>,

        /// keep a `.zarr` directory instead of producing a `.zarr.zip` archive
        #[arg(long = "no-zip", default_value_t = true, action = clap::ArgAction::SetFalse)]
        pub zip: bool,

        #[arg(
            long,
            value_delimiter = ',',
            default_values_t = [Box::<str>::from("_index"), Box::from("gene_id")],
            help = "var/ field(s) for feature ID (fallback list)",
            long_help = "Comma-separated list of var/ dataset names to try for the feature ID (e.g. Ensembl ID).\n\
                         The first one found is used. `_index` is the index of var/,\n\
                         under whatever name the file stores it. Default: '_index,gene_id'."
        )]
        pub row_id_field: Vec<Box<str>>,

        #[arg(
            long,
            value_delimiter = ',',
            default_values_t = [Box::<str>::from("feature_name"), Box::from("gene_name")],
            help = "var/ field(s) for gene symbol (fallback list)",
            long_help = "Comma-separated list of var/ column names to try for the gene symbol.\n\
                         The first column found is used.\n\
                         Joined with the ID to form 'ID_SYMBOL' row names. Default:\n\
                         'feature_name,gene_name'."
        )]
        pub row_name_field: Vec<Box<str>>,

        #[arg(
            long,
            value_delimiter = ',',
            default_values_t = [Box::<str>::from("_index"), Box::from("cell")],
            help = "obs/ field(s) for cell barcode (fallback list)",
            long_help = "Comma-separated list of obs/ dataset names to try for the cell barcode.\n\
                         The first one found is used. `_index` is the index of obs/,\n\
                         under whatever name the file stores it. Default: '_index,cell'."
        )]
        pub col_name_field: Vec<Box<str>>,

        #[arg(
            long,
            default_value = "",
            help = "Select row type (biotype) for filtering",
            long_help = "Filter features by biotype (e.g., 'protein_coding', 'lncRNA').\n\
                         Leave empty to keep all features (default).\n\
                         CELLxGENE uses biotype annotations rather than 'Gene Expression'."
        )]
        pub select_row_type: Box<str>,

        #[arg(
            long,
            default_value = "",
            help = "Remove row type",
            long_help = "Remove rows if their biotype contains this value."
        )]
        pub remove_row_type: Box<str>,

        #[arg(
            long,
            default_value_t = false,
            help = "Squeeze sparse rows or columns",
            long_help = "Enable squeezing to remove rows and columns with too few non-zeros."
        )]
        pub do_squeeze: bool,

        #[arg(
            long,
            default_value_t = 1,
            help = "Row non-zero cutoff",
            long_help = "Minimum number of non-zero elements required for rows."
        )]
        pub row_nnz_cutoff: usize,

        #[arg(
            long,
            default_value_t = 1,
            help = "Column non-zero cutoff",
            long_help = "Minimum number of non-zero elements required for columns."
        )]
        pub column_nnz_cutoff: usize,

        #[arg(
            long,
            help = "Cells per rayon job for the post-build squeeze pass",
            long_help = "Cells per rayon job for the post-build squeeze pass.\n\
                         Omit it for auto-scaling by feature count."
        )]
        pub block_size: Option<usize>,

        #[command(flatten)]
        pub view: crate::handlers::explore::PlanArgs,
    }

    /// `from-h5ad`: [`import::from_h5ad`].
    pub fn run_build_from_h5ad(args: &FromH5adArgs) -> anyhow::Result<()> {
        let session = args.view.session("from-h5ad")?;
        let h5ad = H5adImport {
            row_id_field: args.row_id_field.clone(),
            row_name_field: args.row_name_field.clone(),
            col_name_field: args.col_name_field.clone(),
            select_row_type: args.select_row_type.clone(),
            remove_row_type: args.remove_row_type.clone(),
            ..H5adImport::new(&args.h5ad_file)
        };
        let out = output(&args.output, &args.backend, args.zip);
        let squeeze = Squeeze::new(
            args.do_squeeze,
            args.row_nnz_cutoff,
            args.column_nnz_cutoff,
            args.block_size,
        );
        run_import(
            "from-h5ad",
            session.as_ref(),
            &h5ad,
            out,
            squeeze,
            import::from_h5ad,
            import::stage_h5ad,
        )
    }

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

        #[command(flatten)]
        pub view: crate::handlers::explore::PlanArgs,
    }

    /// `from-10x-matrix`: [`import::from_10x_matrix`].
    pub fn run_build_from_10x_matrix(args: &From10xMatrixArgs) -> anyhow::Result<()> {
        let session = args.view.session("from10x-matrix")?;
        let h5 = TenxMatrixImport {
            root_group: args.root_group_name.clone(),
            layout: MatrixLayout {
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
            },
            ..TenxMatrixImport::new(&args.h5_file)
        };
        let out = output(&args.output, &args.backend, args.zip);
        let squeeze = Squeeze::new(
            args.do_squeeze,
            args.row_nnz_cutoff,
            args.column_nnz_cutoff,
            args.block_size,
        );
        run_import(
            "from10x-matrix",
            session.as_ref(),
            &h5,
            out,
            squeeze,
            import::from_10x_matrix,
            import::stage_10x_matrix,
        )
    }

    #[derive(Args, Debug)]
    pub struct From10xMoleculeArgs {
        #[arg(
            help = "Input 10X molecule_info.h5 file",
            long_help = "Specify the molecule_info.h5 file from Cell Ranger count/multi.\n\
                         Contains per-molecule data: barcode_idx, feature_idx, count, gem_group,\n\
                         etc."
        )]
        pub h5_file: Box<str>,

        #[arg(
            long,
            value_enum,
            default_value = "zarr",
            help = "Backend format for output",
            long_help = "Choose the backend format for the output file."
        )]
        pub backend: SparseIoBackend,

        #[arg(
            short,
            long,
            help = "Output file header or name",
            long_help = "Specify the output file header.\n\
                         The zarr backend produces {output}.zarr.zip by default;\n\
                         pass --no-zip to keep a {output}.zarr directory instead."
        )]
        pub output: Box<str>,

        /// keep a `.zarr` directory instead of producing a `.zarr.zip` archive
        #[arg(long = "no-zip", default_value_t = true, action = clap::ArgAction::SetFalse)]
        pub zip: bool,

        #[arg(
            long,
            default_value = "Gene Expression",
            help = "Library type to include",
            long_help = "Filter molecules to only those from libraries of this type. Common types:\n\
                         'Gene Expression', 'Antibody Capture', 'CRISPR Guide Capture'.\n\
                         Reads library_info JSON to determine which library indices match."
        )]
        pub library_type: Box<str>,

        #[arg(
            long,
            default_value = "",
            help = "Select row type (feature_type)",
            long_help = "Filter features by type.\n\
                         Rows are included if their type contains this value.\n\
                         Empty (default) keeps all features. 10X uses 'Gene Expression',\n\
                         'Antibody Capture', etc."
        )]
        pub select_row_type: Box<str>,

        #[arg(
            long,
            default_value = "",
            help = "Remove row type",
            long_help = "Remove rows if their type contains this value.\n\
                         Empty (default) removes nothing."
        )]
        pub remove_row_type: Box<str>,

        #[arg(
            long,
            default_value_t = false,
            help = "Skip pass_filter and include all barcodes",
            long_help = "By default,\n\
                         only barcodes that passed Cell Ranger cell calling are included.\n\
                         Set this flag to include ALL barcodes with at least one molecule."
        )]
        pub no_pass_filter: bool,

        #[arg(
            long,
            default_value_t = false,
            help = "Sum read counts instead of counting molecules (UMIs)",
            long_help = "By default each entry is the number of molecules (UMIs) of a\n\
                         feature in a barcode, as in Cell Ranger's feature-barcode\n\
                         matrices. With this flag it is the sum of their read counts."
        )]
        pub sum_reads: bool,

        #[arg(
            long,
            default_value_t = false,
            help = "Squeeze sparse rows or columns",
            long_help = "Enable squeezing to remove rows and columns with too few non-zeros."
        )]
        pub do_squeeze: bool,

        #[arg(
            long,
            default_value_t = 1,
            help = "Row non-zero cutoff",
            long_help = "Minimum number of non-zero elements required for rows."
        )]
        pub row_nnz_cutoff: usize,

        #[arg(
            long,
            default_value_t = 1,
            help = "Column non-zero cutoff",
            long_help = "Minimum number of non-zero elements required for columns."
        )]
        pub column_nnz_cutoff: usize,

        #[arg(
            long,
            help = "Cells per rayon job for the post-build squeeze pass",
            long_help = "Cells per rayon job for the post-build squeeze pass.\n\
                         Omit it for auto-scaling by feature count."
        )]
        pub block_size: Option<usize>,

        #[command(flatten)]
        pub view: crate::handlers::explore::PlanArgs,
    }

    /// `from-10x-molecule`: [`import::from_10x_molecule`].
    pub fn run_build_from_10x_molecule(args: &From10xMoleculeArgs) -> anyhow::Result<()> {
        let session = args.view.session("from10x-molecule")?;
        let h5 = TenxMoleculeImport {
            library_type: args.library_type.clone(),
            select_row_type: args.select_row_type.clone(),
            remove_row_type: args.remove_row_type.clone(),
            pass_filter: !args.no_pass_filter,
            sum_reads: args.sum_reads,
            ..TenxMoleculeImport::new(&args.h5_file)
        };
        let out = output(&args.output, &args.backend, args.zip);
        let squeeze = Squeeze::new(
            args.do_squeeze,
            args.row_nnz_cutoff,
            args.column_nnz_cutoff,
            args.block_size,
        );
        run_import(
            "from10x-molecule",
            session.as_ref(),
            &h5,
            out,
            squeeze,
            import::from_10x_molecule,
            import::stage_10x_molecule,
        )
    }

    #[cfg(test)]
    mod h5ad_tests;
    #[cfg(test)]
    mod molecule_tests;
}
