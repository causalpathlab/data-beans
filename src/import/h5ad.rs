use super::{log_feature_type_histogram, Output, Staged};
use crate::hdf5_io::*;
use crate::sparse_io::*;
use crate::sparse_util::*;
use crate::utilities::name_matching::{
    colon_peak_names, compose_id_name, filter_row_indices_by_type, make_names_unique,
};

use legume_numeric::matrix::common_io::*;
use log::{info, warn};

/// An AnnData file, and where in it [`from_h5ad`] finds the names.
#[derive(Clone, Debug)]
pub struct H5adImport {
    /// The `.h5ad` file (CELLxGENE schema v7): `X/{data,indices,indptr}`,
    /// `obs/`, `var/`; `raw/X` is read over `X` when there is one.
    pub h5ad_file: Box<str>,
    /// `var/` fields to try, in order, for the feature id; `_index` is the
    /// index of `var/`, under whatever name the file stores it.
    pub row_id_field: Vec<Box<str>>,
    /// `var/` columns to try, in order, for the gene symbol, joined with the
    /// id as `id_name`.
    pub row_name_field: Vec<Box<str>>,
    /// `obs/` fields to try, in order, for the cell barcode.
    pub col_name_field: Vec<Box<str>>,
    /// Keep features whose biotype contains this; empty keeps all.
    pub select_row_type: Box<str>,
    /// Drop features whose biotype contains this; empty drops none.
    pub remove_row_type: Box<str>,
}

impl H5adImport {
    /// `h5ad_file` with the CLI's defaults: ids from `_index,gene_id`,
    /// symbols from `feature_name,gene_name`, barcodes from `_index,cell`,
    /// every feature kept.
    pub fn new(h5ad_file: &str) -> Self {
        let fields = |xs: &[&str]| xs.iter().map(|&x| Box::from(x)).collect();
        Self {
            h5ad_file: h5ad_file.into(),
            row_id_field: fields(&["_index", "gene_id"]),
            row_name_field: fields(&["feature_name", "gene_name"]),
            col_name_field: fields(&["_index", "cell"]),
            select_row_type: "".into(),
            remove_row_type: "".into(),
        }
    }
}

/// Build the backend `out` names from the AnnData file `h5ad`; the path
/// written. Beside it go `{stem}.cell_metadata.tsv.gz` (every `obs`
/// column, by barcode) and, when `obs` has a `donor_id`, cells named
/// `barcode@donor`, `{stem}.barcode_to_donor.tsv.gz` and
/// `{stem}.sample_metadata.tsv.gz`.
pub fn from_h5ad(h5ad: &H5adImport, out: &Output) -> anyhow::Result<Box<str>> {
    stage_h5ad(h5ad, out)?.finish()
}

/// [`from_h5ad`] short of squeezing and putting the backend in place.
pub fn stage_h5ad(args: &H5adImport, out: &Output) -> anyhow::Result<Staged> {
    let file = hdf5::File::open(args.h5ad_file.to_string())?;
    info!("Opened AnnData h5ad file: {}", args.h5ad_file);

    let (staged, backend) = out.stage()?;
    let backend_file = staged.working();
    let effective_output = &staged.effective;

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

    // AnnData stores (obs x var) = (cells x features), but data-beans
    // convention is (features x cells). We swap the pointer type to
    // transpose: CSR -> Column pointers, CSC -> Row pointers.
    let (n_obs, n_var) = declared_shape(&x_group).unzip();
    let pointer_type = sparse_pointer_type(
        &x_group,
        x_path,
        n_obs.or(column_names.as_ref().map(Vec::len)),
        n_var.or(row_ids.as_ref().map(Vec::len)),
    )?;

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

    // The matrix is as large as the file says, so features and cells with
    // no count past the last one that has any are kept.
    let nrows = crate::convert::axis_len(
        n_var,
        row_ids.as_ref().map(Vec::len),
        shape.nrows,
        pointer_type == IndexPointerType::Row,
        "features",
    )?;
    let ncols = crate::convert::axis_len(
        n_obs,
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

    // The features to keep, settled before anything is written: a backend
    // without a row is of no use.
    log_feature_type_histogram(args.h5ad_file.as_ref(), &row_types);
    let select_rows =
        filter_row_indices_by_type(&row_types, &args.select_row_type, &args.remove_row_type);
    crate::convert::ensure_rows_kept(
        select_rows.len(),
        nrows,
        has_types.then_some(&row_types[..]),
        (&args.select_row_type, &args.remove_row_type),
        None,
    )?;

    // Create sparse backend
    let mut data = create_sparse_from_triplets_owned(
        triplets,
        (nrows, ncols, nnz),
        Some(backend_file),
        Some(&backend),
    )?;
    info!("created sparse matrix: {}", backend_file);
    data.register_row_names_vec(&row_ids);

    // Read all obs metadata columns
    let obs_df = read_h5ad_dataframe(&obs_group)?;
    info!(
        "Read {} obs columns: {:?}",
        obs_df.col_names.len(),
        obs_df.col_names
    );

    // Output stem for metadata files (strip .zarr/.h5)
    let output_stem = crate::hdf5_io::strip_backend_suffix(effective_output);

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
        data.register_column_names_vec(&new_col_names);
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
        data.register_column_names_vec(&column_names);
    }

    if select_rows.len() < nrows {
        info!(
            "filtering features: {} -> {} of `{}` type",
            nrows,
            select_rows.len(),
            args.select_row_type
        );
        data.subset_columns_rows(None, Some(&select_rows))?;
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

    drop(data);
    Ok(staged)
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
/// over cells, has pointers over the columns here; CSC over the rows.
///
/// A group that names no format is read by its number of compressed
/// vectors, one less than the length of `indptr`: CSR when that is `n_obs`
/// (the cells) and not `n_var` (the features), CSC when it is `n_var` and
/// not `n_obs`. An error when the count matches both or neither, or the
/// group names a format that is neither.
fn sparse_pointer_type(
    x: &hdf5::Group,
    x_path: &str,
    n_obs: Option<usize>,
    n_var: Option<usize>,
) -> anyhow::Result<IndexPointerType> {
    let named = str_attr(x, "encoding-type").or_else(|| str_attr(x, "h5sparse_format"));
    let format = match named {
        Some(format) => format.to_ascii_lowercase(),
        None => {
            let vectors = x.dataset("indptr")?.size().checked_sub(1);
            let over = |n: Option<usize>| vectors.is_some() && vectors == n;
            match (over(n_obs), over(n_var)) {
                (true, false) => "csr".into(),
                (false, true) => "csc".into(),
                _ => anyhow::bail!(
                    "{x_path} has no `encoding-type` or `h5sparse_format` attribute to tell \
                     CSR from CSC, and its {} compressed vectors do not tell either for \
                     {} cells and {} features",
                    vectors.map_or("no".into(), |n| n.to_string()),
                    n_obs.map_or("unknown".into(), |n| n.to_string()),
                    n_var.map_or("unknown".into(), |n| n.to_string()),
                ),
            }
        }
    };
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
