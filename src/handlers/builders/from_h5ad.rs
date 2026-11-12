use super::{keep_import, log_feature_type_histogram, Squeeze};
use crate::hdf5_io::*;
use crate::sparse_io::*;
use crate::sparse_util::*;
use crate::utilities::name_matching::{
    colon_peak_names, compose_id_name, filter_row_indices_by_type, make_names_unique,
};
use crate::zarr_io::*;

use clap::Args;
use legume_numeric::matrix::common_io::*;
use log::{info, warn};

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
pub fn run_build_from_h5ad(args: &FromH5adArgs) -> anyhow::Result<()> {
    // The session first: hosted, the build runs there.
    let session = args.view.session("from-h5ad")?;
    let file = hdf5::File::open(args.h5ad_file.to_string())?;
    info!("Opened AnnData h5ad file: {}", args.h5ad_file);

    let effective_output = apply_zip_flag(&args.output, args.zip, &args.backend);
    let (backend, backend_file) =
        resolve_backend_file(&effective_output, Some(args.backend.clone()))?;

    if std::path::Path::new(backend_file.as_ref()).exists() {
        info!("Removing existing backend file: {}", backend_file);
        remove_file(&backend_file)?;
    }

    // Prefer raw/X (raw counts) over X (processed/normalized)
    let (x_path, var_path) = if file.group("raw/X").is_ok() {
        info!("Using raw/X (raw counts)");
        ("raw/X", "raw/var")
    } else {
        info!("Using X");
        ("X", "var")
    };

    let x_group = file.group(x_path)?;
    let var_group = file.group(var_path)?;
    let obs_group = file.group("obs")?;

    // AnnData stores (obs x var) = (cells x features), but data-beans
    // convention is (features x cells). We swap the pointer type to
    // transpose: CSR -> Column pointers, CSC -> Row pointers.
    let pointer_type = sparse_pointer_type(&x_group, x_path)?;
    let declared = declared_shape(&x_group);

    // Read triplets
    let CooTripletsShape { triplets, shape } = {
        let values = x_group.dataset("data")?.read_1d::<f32>()?;
        let indices = x_group.dataset("indices")?.read_1d::<u64>()?;
        let indptr = x_group.dataset("indptr")?.read_1d::<u64>()?;
        ValuesIndicesPointers {
            values: values.as_slice().expect("values not contiguous"),
            indices: indices.as_slice().expect("indices not contiguous"),
            indptr: indptr.as_slice().expect("indptr not contiguous"),
        }
        .to_coo(pointer_type)?
    };
    let nnz = shape.nnz;
    info!(
        "Read {} non-zero elements reaching {} x {}",
        nnz, shape.nrows, shape.ncols
    );

    // Feature IDs, names and types from var/, barcodes from obs/ (each
    // list of candidates tried in order). Types are var/feature_type or
    // var/feature_types (scanpy, muon), often categorical.
    let row_ids = resolve_h5ad_field(
        &var_group,
        &index_candidates(&var_group, &args.row_id_field),
        "feature IDs",
    );
    let row_names = resolve_h5ad_field(&var_group, &args.row_name_field, "gene symbols");
    let typed = read_h5ad_column(&var_group, "feature_type")
        .or_else(|_| read_h5ad_column(&var_group, "feature_types"))
        .ok();
    let column_names = resolve_h5ad_field(
        &obs_group,
        &index_candidates(&obs_group, &args.col_name_field),
        "barcodes",
    );

    // The matrix is as large as the file says, so features and cells with
    // no count past the last one that has any are kept.
    let nrows = axis_len(
        declared.map(|(_, n_var)| n_var),
        row_ids.as_ref().map(Vec::len),
        shape.nrows,
        pointer_type == IndexPointerType::Row,
        "features",
    )?;
    let ncols = axis_len(
        declared.map(|(n_obs, _)| n_obs),
        column_names.as_ref().map(Vec::len),
        shape.ncols,
        pointer_type == IndexPointerType::Column,
        "cells",
    )?;
    info!("Matrix: {} x {}", nrows, ncols);
    let listed = |what: &str, names: &[Box<str>], n: usize| {
        anyhow::ensure!(
            names.len() == n,
            "{} lists {} {what} but {x_path} has {n}",
            args.h5ad_file,
            names.len()
        );
        Ok(())
    };
    let index =
        |n: usize| -> Vec<Box<str>> { (0..n).map(|x| x.to_string().into_boxed_str()).collect() };

    let mut row_ids = row_ids.unwrap_or_else(|| {
        warn!("No feature ID column found in {var_path}/; using numeric IDs");
        index(nrows)
    });
    listed("feature IDs", &row_ids, nrows)?;
    let mut row_names = row_names.unwrap_or_else(|| {
        info!("No gene symbol column found in {var_path}/; using index only");
        vec![Box::from(""); nrows]
    });
    listed("gene symbols", &row_names, nrows)?;
    let has_types = typed.is_some();
    let row_types: Vec<Box<str>> = typed.unwrap_or_else(|| vec![Box::from(""); nrows]);
    listed("feature types", &row_types, nrows)?;

    // Peak rows in chr:start-end form, then composite row names: id_name
    let n_peaks = colon_peak_names(
        &mut row_ids,
        &mut row_names,
        has_types.then_some(&row_types[..]),
    );
    if n_peaks > 0 {
        info!("{n_peaks} peak names rewritten in chr:start-end form");
    }
    let mut row_ids = compose_id_name(row_ids, row_names);
    make_names_unique(&mut row_ids);

    let column_names = column_names.unwrap_or_else(|| {
        warn!("No barcode column found in obs/; using numeric IDs");
        index(ncols)
    });
    listed("barcodes", &column_names, ncols)?;
    info!("Read {} barcodes", column_names.len());

    // Create sparse backend
    let mut out = create_sparse_from_triplets_owned(
        triplets,
        (nrows, ncols, nnz),
        Some(&backend_file),
        Some(&backend),
    )?;
    info!("created sparse matrix: {}", backend_file);
    out.register_row_names_vec(&row_ids);

    log_feature_type_histogram(args.h5ad_file.as_ref(), &row_types);

    let select_rows =
        filter_row_indices_by_type(&row_types, &args.select_row_type, &args.remove_row_type);

    // Read all obs metadata columns
    let obs_df = read_h5ad_dataframe(&obs_group)?;
    info!(
        "Read {} obs columns: {:?}",
        obs_df.col_names.len(),
        obs_df.col_names
    );

    // Output stem for metadata files (strip .zarr/.h5)
    let output_stem = crate::hdf5_io::strip_backend_suffix(&effective_output);

    // Handle donor_id: barcode@donor_id column names + donor mapping files
    let donor_idx = obs_df
        .col_names
        .iter()
        .position(|c| c.as_ref() == "donor_id");

    if let Some(di) = donor_idx {
        let donors = &obs_df.col_data[di];
        let new_col_names: Vec<Box<str>> = column_names
            .iter()
            .zip(donors.iter())
            .map(|(b, d)| format!("{}@{}", b, d).into_boxed_str())
            .collect();
        out.register_column_names_vec(&new_col_names);
        info!("Column names set to barcode@donor_id");

        // barcode_to_donor.tsv.gz
        let donor_file = format!("{}.barcode_to_donor.tsv.gz", output_stem);
        let mut lines: Vec<Box<str>> = Vec::with_capacity(column_names.len() + 1);
        lines.push("barcode\tdonor_id".into());
        for (b, d) in column_names.iter().zip(donors.iter()) {
            lines.push(format!("{}\t{}", b, d).into());
        }
        write_lines(&lines, &donor_file)?;
        info!("Wrote {}", donor_file);

        // sample_metadata.tsv.gz (one row per unique donor)
        let sample_file = format!("{}.sample_metadata.tsv.gz", output_stem);
        write_sample_metadata(&sample_file, &obs_df.col_names, &obs_df.col_data, di)?;
        info!("Wrote {}", sample_file);
    } else {
        out.register_column_names_vec(&column_names);
    }

    if select_rows.len() < nrows {
        info!(
            "filtering features: {} -> {} of `{}` type",
            nrows,
            select_rows.len(),
            args.select_row_type
        );
        out.subset_columns_rows(None, Some(&select_rows))?;
    }

    // cell_metadata.tsv.gz — all obs columns indexed by barcode
    let meta_file = format!("{}.cell_metadata.tsv.gz", output_stem);
    let mut lines: Vec<Box<str>> = Vec::with_capacity(column_names.len() + 1);
    // header
    let header = std::iter::once("barcode")
        .chain(obs_df.col_names.iter().map(|s| s.as_ref()))
        .collect::<Vec<&str>>()
        .join("\t");
    lines.push(header.into());
    // rows
    for i in 0..column_names.len() {
        let row = std::iter::once(column_names[i].as_ref())
            .chain(obs_df.col_data.iter().map(|col| col[i].as_ref()))
            .collect::<Vec<&str>>()
            .join("\t");
        lines.push(row.into());
    }
    write_lines(&lines, &meta_file)?;
    info!("Wrote {}", meta_file);

    let squeeze = Squeeze {
        on: args.do_squeeze,
        rows: args.row_nnz_cutoff,
        columns: args.column_nnz_cutoff,
        block_size: args.block_size,
    };
    if !keep_import("from-h5ad", session.as_ref(), squeeze, &backend_file)? {
        return Ok(());
    }
    finalize_zarr_output(&backend_file, &effective_output)?;
    info!("done");
    Ok(())
}

/// A string attribute of an h5ad group, however the writer encoded it.
fn str_attr(loc: &hdf5::Location, name: &str) -> Option<String> {
    use hdf5::types::{FixedAscii, FixedUnicode, VarLenAscii, VarLenUnicode};
    let attr = loc.attr(name).ok()?;
    attr.read_scalar::<VarLenUnicode>()
        .map(|s| s.to_string())
        .or_else(|_| attr.read_scalar::<VarLenAscii>().map(|s| s.to_string()))
        .or_else(|_| {
            attr.read_scalar::<FixedUnicode<64>>()
                .map(|s| s.to_string())
        })
        .or_else(|_| attr.read_scalar::<FixedAscii<64>>().map(|s| s.to_string()))
        .ok()
}

/// Which axis the sparse group `x` (at `x_path`) is compressed along, as
/// data-beans sees it: its `encoding-type` (`csr_matrix` / `csc_matrix`),
/// or the `h5sparse_format` older anndata wrote. AnnData's CSR, pointers
/// over cells, has pointers over the columns here; CSC over the rows. An
/// error when the group names neither format.
fn sparse_pointer_type(x: &hdf5::Group, x_path: &str) -> anyhow::Result<IndexPointerType> {
    let format = str_attr(x, "encoding-type")
        .or_else(|| str_attr(x, "h5sparse_format"))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{x_path} has no `encoding-type` or `h5sparse_format` attribute \
                 to tell CSR from CSC"
            )
        })?
        .to_ascii_lowercase();
    if format.contains("csr") {
        info!("Sparse format: CSR -> transposing to (features x cells)");
        Ok(IndexPointerType::Column)
    } else if format.contains("csc") {
        info!("Sparse format: CSC -> transposing to (features x cells)");
        Ok(IndexPointerType::Row)
    } else {
        anyhow::bail!("{x_path} is stored as `{format}`, neither CSR nor CSC")
    }
}

/// `(n_obs, n_var)` as the sparse group declares it: its `shape`
/// attribute, or the `h5sparse_shape` older anndata wrote.
fn declared_shape(x: &hdf5::Group) -> Option<(usize, usize)> {
    ["shape", "h5sparse_shape"].iter().find_map(|name| {
        let shape: Vec<u64> = x.attr(name).ok()?.read_raw().ok()?;
        (shape.len() == 2).then(|| (shape[0] as usize, shape[1] as usize))
    })
}

/// The length of one axis of the matrix: what the sparse group declares,
/// else the number of names obs/ or var/ lists for it, else as far as the
/// data reaches. An error when the data reaches past that length, or, on
/// the axis the matrix is `compressed` along, holds another number of
/// vectors.
fn axis_len(
    declared: Option<usize>,
    named: Option<usize>,
    reach: usize,
    compressed: bool,
    what: &str,
) -> anyhow::Result<usize> {
    let n = declared.or(named).unwrap_or(reach);
    anyhow::ensure!(
        if compressed { reach == n } else { reach <= n },
        "the matrix should have {n} {what} but its data {} {reach}",
        if compressed { "holds" } else { "reaches" }
    );
    Ok(n)
}

/// `fields` with `_index` read as the index of the dataframe `group`: the
/// dataset its `_index` attribute names (anndata stores a named index under
/// that name), ahead of one literally called `_index`.
fn index_candidates(group: &hdf5::Group, fields: &[Box<str>]) -> Vec<Box<str>> {
    let named = str_attr(group, "_index").filter(|name| name != "_index");
    fields
        .iter()
        .flat_map(|field| {
            let index = named
                .as_deref()
                .filter(|_| field.trim() == "_index")
                .map(Box::from);
            index.into_iter().chain([field.clone()])
        })
        .collect()
}

/// Write sample-level metadata: one row per unique donor_id.
fn write_sample_metadata(
    path: &str,
    col_names: &[Box<str>],
    col_data: &[Vec<Box<str>>],
    donor_col_idx: usize,
) -> anyhow::Result<()> {
    use std::collections::BTreeMap;

    let donors = &col_data[donor_col_idx];
    let mut donor_first_row: BTreeMap<&str, usize> = BTreeMap::new();
    for (i, d) in donors.iter().enumerate() {
        donor_first_row.entry(d.as_ref()).or_insert(i);
    }

    let mut lines: Vec<Box<str>> = Vec::with_capacity(donor_first_row.len() + 1);

    // header: donor_id + all other columns
    let header = std::iter::once("donor_id")
        .chain(
            col_names
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != donor_col_idx)
                .map(|(_, s)| s.as_ref()),
        )
        .collect::<Vec<&str>>()
        .join("\t");
    lines.push(header.into());

    // one row per unique donor
    for (donor_id, &row_i) in &donor_first_row {
        let row = std::iter::once(*donor_id)
            .chain(
                col_data
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != donor_col_idx)
                    .map(|(_, col)| col[row_i].as_ref()),
            )
            .collect::<Vec<&str>>()
            .join("\t");
        lines.push(row.into());
    }

    write_lines(&lines, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hdf5::types::VarLenUnicode;
    use std::path::Path;

    fn vlu(s: &str) -> VarLenUnicode {
        s.parse().unwrap()
    }

    fn write_attr(loc: &hdf5::Location, name: &str, value: &str) {
        loc.new_attr::<VarLenUnicode>()
            .create(name)
            .unwrap()
            .write_scalar(&vlu(value))
            .unwrap();
    }

    fn write_strings(group: &hdf5::Group, name: &str, values: &[&str]) {
        let values: Vec<VarLenUnicode> = values.iter().map(|v| vlu(v)).collect();
        group
            .new_dataset::<VarLenUnicode>()
            .shape(values.len())
            .create(name)
            .unwrap()
            .write_raw(&values)
            .unwrap();
    }

    /// An AnnData dataframe group: its index stored under `index.0`, as the
    /// `_index` attribute says, and string columns.
    fn write_dataframe(
        file: &hdf5::File,
        name: &str,
        index: (&str, &[&str]),
        columns: &[(&str, &[&str])],
    ) {
        let group = file.create_group(name).unwrap();
        write_attr(&group, "_index", index.0);
        write_strings(&group, index.0, index.1);
        let order: Vec<VarLenUnicode> = columns.iter().map(|(c, _)| vlu(c)).collect();
        group
            .new_attr::<VarLenUnicode>()
            .shape([order.len()])
            .create("column-order")
            .unwrap()
            .write_raw(&order)
            .unwrap();
        for (column, values) in columns {
            write_strings(&group, column, values);
        }
    }

    /// The sparse group `X`, its format and shape under the attributes
    /// named.
    fn write_x(
        file: &hdf5::File,
        format: Option<(&str, &str)>,
        shape: Option<(&str, [i64; 2])>,
        data: &[f32],
        indices: &[i32],
        indptr: &[i32],
    ) {
        let x = file.create_group("X").unwrap();
        if let Some((name, value)) = format {
            write_attr(&x, name, value);
        }
        if let Some((name, value)) = shape {
            x.new_attr::<i64>()
                .shape([2])
                .create(name)
                .unwrap()
                .write_raw(&value)
                .unwrap();
        }
        let ds = |name: &str, n: usize| x.new_dataset::<i32>().shape(n).create(name).unwrap();
        x.new_dataset::<f32>()
            .shape(data.len())
            .create("data")
            .unwrap()
            .write_raw(data)
            .unwrap();
        ds("indices", indices.len()).write_raw(indices).unwrap();
        ds("indptr", indptr.len()).write_raw(indptr).unwrap();
    }

    // 3 cells x 4 genes, CSR over cells; the last gene has no count:
    //   BC1 [1 0 2 0], BC2 [0 3 0 0], BC3 [4 0 0 0]
    const DATA: [f32; 4] = [1., 2., 3., 4.];
    const INDICES: [i32; 4] = [0, 2, 1, 0];
    const INDPTR: [i32; 4] = [0, 2, 3, 4];
    const CELLS: [&str; 3] = ["BC1", "BC2", "BC3"];
    const GENES: [&str; 4] = ["G1", "G2", "G3", "G4"];

    fn args(h5ad: &Path, output: &Path) -> FromH5adArgs {
        let fields = |xs: [&str; 2]| xs.iter().map(|&x| Box::from(x)).collect();
        FromH5adArgs {
            h5ad_file: h5ad.to_str().unwrap().into(),
            backend: SparseIoBackend::Zarr,
            output: output.to_str().unwrap().into(),
            zip: false,
            row_id_field: fields(["_index", "gene_id"]),
            row_name_field: fields(["feature_name", "gene_name"]),
            col_name_field: fields(["_index", "cell"]),
            select_row_type: "".into(),
            remove_row_type: "".into(),
            do_squeeze: false,
            row_nnz_cutoff: 1,
            column_nnz_cutoff: 1,
            block_size: None,
            view: crate::handlers::explore::PlanArgs::batch(),
        }
    }

    type Built = (Vec<Box<str>>, Vec<Box<str>>, Vec<Vec<f32>>);

    /// Build `h5ad` and read the backend back: its row names, column names
    /// and the dense (features x cells) matrix, row by row.
    fn build(h5ad: &Path) -> anyhow::Result<Built> {
        let output = h5ad.with_extension("");
        run_build_from_h5ad(&args(h5ad, &output))?;
        let data = open_sparse_matrix(
            &format!("{}.zarr", output.to_str().unwrap()),
            &SparseIoBackend::Zarr,
        )?;
        let cols = data.column_names()?;
        let m = data.read_columns_dmatrix((0..cols.len()).collect())?;
        let rows = m.row_iter().map(|r| r.iter().copied().collect()).collect();
        Ok((data.row_names()?, cols, rows))
    }

    fn genes_by_cells() -> Vec<Vec<f32>> {
        vec![
            vec![1., 0., 4.],
            vec![0., 3., 0.],
            vec![2., 0., 0.],
            vec![0., 0., 0.],
        ]
    }

    #[test]
    fn a_named_index_names_the_cells_and_features() {
        let dir = tempfile::tempdir().unwrap();
        let h5ad = dir.path().join("named.h5ad");
        let file = hdf5::File::create(&h5ad).unwrap();
        let csr = Some(("encoding-type", "csr_matrix"));
        write_x(
            &file,
            csr,
            Some(("shape", [3, 4])),
            &DATA,
            &INDICES,
            &INDPTR,
        );
        write_dataframe(&file, "obs", ("barcode", &CELLS), &[]);
        write_dataframe(&file, "var", ("gene_ids", &GENES), &[]);
        drop(file);

        let (rows, cols, m) = build(&h5ad).unwrap();
        assert_eq!(cols, CELLS.map(Box::from));
        assert_eq!(rows, GENES.map(Box::from));
        assert_eq!(m, genes_by_cells());
    }

    #[test]
    fn features_and_cells_past_the_last_count_are_kept() {
        let dir = tempfile::tempdir().unwrap();
        // No shape on X: var/ and obs/ say how large the matrix is.
        let h5ad = dir.path().join("csr.h5ad");
        let file = hdf5::File::create(&h5ad).unwrap();
        let csr = Some(("encoding-type", "csr_matrix"));
        write_x(&file, csr, None, &DATA, &INDICES, &INDPTR);
        write_dataframe(&file, "obs", ("_index", &CELLS), &[]);
        write_dataframe(&file, "var", ("_index", &GENES), &[]);
        drop(file);
        let (rows, _, m) = build(&h5ad).unwrap();
        assert_eq!(rows, GENES.map(Box::from));
        assert_eq!(m, genes_by_cells());

        // The same arrays as CSC over genes: 4 cells x 3 genes, the last
        // cell empty.
        let h5ad = dir.path().join("csc.h5ad");
        let file = hdf5::File::create(&h5ad).unwrap();
        let csc = Some(("encoding-type", "csc_matrix"));
        write_x(
            &file,
            csc,
            Some(("shape", [4, 3])),
            &DATA,
            &INDICES,
            &INDPTR,
        );
        write_dataframe(&file, "obs", ("_index", &GENES), &[]);
        write_dataframe(&file, "var", ("_index", &CELLS), &[]);
        drop(file);
        let (rows, cols, m) = build(&h5ad).unwrap();
        assert_eq!(rows, CELLS.map(Box::from));
        assert_eq!(cols, GENES.map(Box::from));
        assert_eq!(
            m,
            [[1., 0., 2., 0.], [0., 3., 0., 0.], [4., 0., 0., 0.]].map(Vec::from)
        );
    }

    #[test]
    fn data_reaching_past_the_names_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let h5ad = dir.path().join("short.h5ad");
        let file = hdf5::File::create(&h5ad).unwrap();
        let csr = Some(("encoding-type", "csr_matrix"));
        write_x(&file, csr, None, &DATA, &INDICES, &INDPTR);
        write_dataframe(&file, "obs", ("_index", &CELLS), &[]);
        write_dataframe(&file, "var", ("_index", &GENES[..2]), &[]);
        drop(file);
        let e = build(&h5ad).unwrap_err().to_string();
        assert!(e.contains("2 features"), "{e}");
    }

    #[test]
    fn the_sparse_format_is_read_not_assumed() {
        let dir = tempfile::tempdir().unwrap();
        let frames = |file: &hdf5::File| {
            write_dataframe(file, "obs", ("_index", &CELLS), &[]);
            write_dataframe(file, "var", ("_index", &GENES), &[]);
        };
        // As anndata before 0.7 wrote a CSR matrix.
        let h5ad = dir.path().join("legacy.h5ad");
        let file = hdf5::File::create(&h5ad).unwrap();
        let legacy = Some(("h5sparse_format", "csr"));
        let shape = Some(("h5sparse_shape", [3, 4]));
        write_x(&file, legacy, shape, &DATA, &INDICES, &INDPTR);
        frames(&file);
        drop(file);
        let (rows, cols, m) = build(&h5ad).unwrap();
        assert_eq!(rows, GENES.map(Box::from));
        assert_eq!(cols, CELLS.map(Box::from));
        assert_eq!(m, genes_by_cells());

        for (name, format) in [("none.h5ad", None), ("coo.h5ad", Some("coo_matrix"))] {
            let h5ad = dir.path().join(name);
            let file = hdf5::File::create(&h5ad).unwrap();
            let format = format.map(|f| ("encoding-type", f));
            write_x(&file, format, None, &DATA, &INDICES, &INDPTR);
            frames(&file);
            drop(file);
            let e = build(&h5ad).unwrap_err().to_string();
            assert!(e.contains("CSR"), "{e}");
        }
    }

    /// Two donors share a barcode. The cell metadata `from-h5ad` writes,
    /// given back to `to-h5ad`, annotates each cell with its own donor's
    /// row.
    #[test]
    fn cell_metadata_round_trips_when_donors_share_a_barcode() {
        use crate::handlers::exporters::{run_export_to_h5ad, ToH5adArgs};
        let dir = tempfile::tempdir().unwrap();
        let h5ad = dir.path().join("donors.h5ad");
        let file = hdf5::File::create(&h5ad).unwrap();
        let csr = Some(("encoding-type", "csr_matrix"));
        write_x(
            &file,
            csr,
            Some(("shape", [3, 4])),
            &DATA,
            &INDICES,
            &INDPTR,
        );
        let obs: [(&str, &[&str]); 2] = [
            ("donor_id", &["donorA", "donorB", "donorA"]),
            ("cell_type", &["T", "B", "NK"]),
        ];
        let barcodes = ["AAAC-1", "AAAC-1", "GGGT-1"];
        write_dataframe(&file, "obs", ("_index", &barcodes), &obs);
        write_dataframe(&file, "var", ("_index", &GENES), &[]);
        drop(file);

        let (_, cols, _) = build(&h5ad).unwrap();
        let cells = ["AAAC-1@donorA", "AAAC-1@donorB", "GGGT-1@donorA"].map(Box::from);
        assert_eq!(cols, cells);

        let stem = dir.path().join("donors");
        let stem = stem.to_str().unwrap();
        let back = dir.path().join("back.h5ad");
        run_export_to_h5ad(&ToH5adArgs {
            data_file: format!("{stem}.zarr").into(),
            output: Some(back.to_str().unwrap().into()),
            obs: Some(format!("{stem}.cell_metadata.tsv.gz").into()),
            var: None,
        })
        .unwrap();
        let obs = hdf5::File::open(&back).unwrap().group("obs").unwrap();
        assert_eq!(read_h5ad_column(&obs, "_index").unwrap(), cells);
        assert_eq!(
            read_h5ad_column(&obs, "cell_type").unwrap(),
            ["T", "B", "NK"].map(Box::from)
        );
    }
}
