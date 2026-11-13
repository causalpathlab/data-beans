use crate::hdf5_io::*;
use crate::sparse_io::*;
use crate::sparse_util::*;
use crate::utilities::name_matching::{
    colon_peak_names, compose_id_name, filter_row_indices_by_type, make_names_unique,
};
use crate::zarr_io::*;

use legume_numeric::matrix::common_io::*;
use log::info;

/// The sparse matrix of a 10x file as read, before rows are named and kept:
/// its triplets, how far they reach, and whatever names the file lists.
pub struct TenxMatrix {
    pub triplets: Vec<(u64, u64, f32)>,
    /// Rows and columns the triplets span: every compressed vector, and up
    /// to the largest index on the other axis.
    pub reach: (usize, usize),
    pub row_ids: Option<Vec<Box<str>>>,
    pub row_names: Option<Vec<Box<str>>>,
    pub row_types: Option<Vec<Box<str>>>,
    pub column_names: Option<Vec<Box<str>>>,
    /// Rows to keep whatever their type, as a probe set's targets; `None`
    /// for all.
    pub keep_rows: Option<Vec<bool>>,
}

/// The length of one axis: the length the file declares for it, else the
/// number of names it lists (one per row or column, so trailing all-zero
/// rows and columns, which leave no trace in the triplets, still count),
/// else as far as the triplets reach. An error when the triplets reach past
/// that length, or, with `compressed` (the axis the file's pointers run
/// over, one vector per row or column), hold another number of vectors.
pub(crate) fn axis_len(
    declared: Option<usize>,
    named: Option<usize>,
    reach: usize,
    compressed: bool,
    what: &str,
) -> anyhow::Result<usize> {
    let (n, source) = match (declared, named) {
        (Some(n), _) => (n, "declares"),
        (None, Some(n)) => (n, "names"),
        (None, None) => return Ok(reach),
    };
    anyhow::ensure!(
        if compressed { reach == n } else { reach <= n },
        "the file {source} {n} {what} but its data {} {reach}",
        if compressed { "holds" } else { "reaches" }
    );
    Ok(n)
}

/// An error when a matrix of `nrows` rows keeps none of them: `n_kept` is
/// zero. It names the type `patterns` (select, remove) and the file's
/// types; `row_set` is `(rows of a kept type, rows in the set)` when the
/// rows are also limited to a set of their own ([`TenxMatrix::keep_rows`]),
/// which may be what leaves none.
pub(crate) fn ensure_rows_kept(
    n_kept: usize,
    nrows: usize,
    row_types: Option<&[Box<str>]>,
    patterns: (&str, &str),
    row_set: Option<(usize, usize)>,
) -> anyhow::Result<()> {
    if n_kept > 0 || nrows == 0 {
        return Ok(());
    }
    let types: std::collections::BTreeSet<&str> = row_types
        .iter()
        .flat_map(|t| t.iter())
        .map(|t| t.as_ref())
        .collect();
    let patterns = format!("select `{}`, remove `{}`", patterns.0, patterns.1);
    match row_set {
        Some((n_typed, n_set)) => anyhow::bail!(
            "none of the {nrows} rows is kept: {n_typed} are of a kept type ({patterns}) and \
             {n_set} are in `keep_rows`, the rows the file itself selects (a probe set's \
             targets), but none is both; the file's feature types are {types:?}. Name the \
             types to keep with --select-row-type"
        ),
        None => anyhow::bail!(
            "none of the {nrows} rows is kept ({patterns}); the file's feature types are \
             {types:?}. Name the types to keep with --select-row-type"
        ),
    }
}

/// Write `m` as a new backend at `backend_file`: rows named `{id}_{name}`
/// (just `id` when the name is empty or repeats it; made unique) and kept by
/// type (comma-separated, case-insensitive patterns; a file without types
/// keeps all), columns named as the file names them. Shared by the 10x
/// readers, so every format and entry point names and keeps rows alike.
pub fn write_10x_matrix(
    m: TenxMatrix,
    select_row_type: &str,
    remove_row_type: &str,
    backend_file: &str,
    backend: &SparseIoBackend,
) -> anyhow::Result<()> {
    let TenxMatrix {
        mut triplets,
        reach,
        row_ids,
        row_names,
        row_types,
        column_names,
        keep_rows,
    } = m;
    // A 10x file declares no shape apart from its names, and a list of
    // names longer than the pointers is taken as trailing empty vectors.
    let nrows = axis_len(None, row_ids.as_ref().map(Vec::len), reach.0, false, "rows")?;
    let named_columns = column_names.as_ref().map(Vec::len);
    let ncols = axis_len(None, named_columns, reach.1, false, "columns")?;
    let index = |n: usize| {
        (0..n)
            .map(|i| i.to_string().into_boxed_str())
            .collect::<Vec<_>>()
    };
    let mut row_ids = row_ids.unwrap_or_else(|| index(nrows));
    let mut row_names = row_names.unwrap_or_else(|| vec![Box::from(""); nrows]);
    anyhow::ensure!(
        row_names.len() == nrows
            && row_types.as_ref().is_none_or(|t| t.len() == nrows)
            && keep_rows.as_ref().is_none_or(|k| k.len() == nrows),
        "the file lists {nrows} row ids but {} row names and {} row types",
        row_names.len(),
        row_types.as_ref().map_or(0, Vec::len)
    );
    let column_names = column_names.unwrap_or_else(|| index(ncols));

    let n_peaks = colon_peak_names(&mut row_ids, &mut row_names, row_types.as_deref());
    if n_peaks > 0 {
        info!("{n_peaks} peak names rewritten in chr:start-end form");
    }
    let mut row_ids = compose_id_name(row_ids, row_names);
    make_names_unique(&mut row_ids);
    let mut keep = match &row_types {
        Some(types) => filter_row_indices_by_type(types, select_row_type, remove_row_type),
        None => (0..nrows).collect(),
    };
    let row_set = keep_rows.as_ref().map(|keep_rows| {
        let n_typed = keep.len();
        keep.retain(|&i| keep_rows[i]);
        (n_typed, keep_rows.iter().filter(|&&k| k).count())
    });
    ensure_rows_kept(
        keep.len(),
        nrows,
        row_types.as_deref(),
        (select_row_type, remove_row_type),
        row_set,
    )?;
    // The other rows leave the triplets before anything is written.
    if keep.len() < nrows {
        info!("Keeping {} of {nrows} rows", keep.len());
        let mut new_row = vec![None; nrows];
        for (new, &old) in keep.iter().enumerate() {
            new_row[old] = Some(new as u64);
        }
        triplets.retain_mut(|(i, _, _)| match new_row[*i as usize] {
            Some(new) => {
                *i = new;
                true
            }
            None => false,
        });
        row_ids = keep
            .iter()
            .map(|&i| std::mem::take(&mut row_ids[i]))
            .collect();
    }

    let nnz = triplets.len();
    info!("Matrix: {} x {ncols}, {nnz} non-zeros", row_ids.len());
    let mut out = create_sparse_from_triplets_owned(
        triplets,
        (row_ids.len(), ncols, nnz),
        Some(backend_file),
        Some(backend),
    )?;
    info!("Created sparse matrix: {}", backend_file);
    out.register_row_names_vec(&row_ids);
    out.register_column_names_vec(&column_names);
    Ok(())
}

/// Where a 10x file keeps its sparse matrix and the names of its rows and
/// columns, and which rows to keep: [`MatrixLayout::cell_ranger_h5`] and
/// [`MatrixLayout::xenium_zarr`].
#[derive(Clone, Debug)]
pub struct MatrixLayout {
    pub data_field: Box<str>,
    pub indices_field: Box<str>,
    pub indptr_field: Box<str>,
    pub pointer_type: IndexPointerType,
    pub row_id_field: Box<str>,
    pub row_name_field: Box<str>,
    pub row_type_field: Box<str>,
    /// Comma-separated, case-insensitive; a row whose type contains any is kept.
    pub select_row_type: Box<str>,
    /// Comma-separated, case-insensitive; a row whose type contains any is dropped.
    pub remove_row_type: Box<str>,
    pub column_name_field: Box<str>,
}

impl MatrixLayout {
    /// The group a Cell Ranger HDF5 file keeps its matrix under; the
    /// `H5_*` fields are relative to it.
    pub const H5_ROOT: &'static str = "matrix";
    pub const H5_DATA: &'static str = "data";
    pub const H5_INDICES: &'static str = "indices";
    pub const H5_INDPTR: &'static str = "indptr";
    pub const H5_ROW_IDS: &'static str = "features/id";
    pub const H5_ROW_NAMES: &'static str = "features/name";
    pub const H5_ROW_TYPES: &'static str = "features/feature_type";
    pub const H5_COLUMN_NAMES: &'static str = "barcodes";
    pub const XENIUM_DATA: &'static str = "/cell_features/data";
    pub const XENIUM_INDICES: &'static str = "/cell_features/indices";
    pub const XENIUM_INDPTR: &'static str = "/cell_features/indptr";
    pub const XENIUM_ROW_IDS: &'static str = "/cell_features/feature_ids";
    pub const XENIUM_ROW_NAMES: &'static str = "/cell_features/feature_keys";
    pub const XENIUM_ROW_TYPES: &'static str = "/cell_features/feature_types";
    pub const XENIUM_COLUMN_NAMES: &'static str = "/cell_features/cell_id";
    /// Gene Expression (Xenium `gene`) and ATAC Peaks.
    pub const SELECT_ROW_TYPES: &'static str = "gene,peak";
    /// Xenium's per-gene aggregates (`aggregate_gene`).
    pub const REMOVE_ROW_TYPES: &'static str = "aggregate";

    /// Cell Ranger / Space Ranger / Xenium `*feature_bc_matrix.h5`:
    /// features × barcodes under [`Self::H5_ROOT`], pointers over barcodes.
    pub fn cell_ranger_h5() -> Self {
        Self {
            data_field: Self::H5_DATA.into(),
            indices_field: Self::H5_INDICES.into(),
            indptr_field: Self::H5_INDPTR.into(),
            pointer_type: IndexPointerType::Column,
            row_id_field: Self::H5_ROW_IDS.into(),
            row_name_field: Self::H5_ROW_NAMES.into(),
            row_type_field: Self::H5_ROW_TYPES.into(),
            select_row_type: Self::SELECT_ROW_TYPES.into(),
            remove_row_type: Self::REMOVE_ROW_TYPES.into(),
            column_name_field: Self::H5_COLUMN_NAMES.into(),
        }
    }

    /// Xenium's `cell_feature_matrix.zarr`: features × cells, pointers over
    /// features, feature names as attributes of `/cell_features`.
    pub fn xenium_zarr() -> Self {
        Self {
            data_field: Self::XENIUM_DATA.into(),
            indices_field: Self::XENIUM_INDICES.into(),
            indptr_field: Self::XENIUM_INDPTR.into(),
            pointer_type: IndexPointerType::Row,
            row_id_field: Self::XENIUM_ROW_IDS.into(),
            row_name_field: Self::XENIUM_ROW_NAMES.into(),
            row_type_field: Self::XENIUM_ROW_TYPES.into(),
            select_row_type: Self::SELECT_ROW_TYPES.into(),
            remove_row_type: Self::REMOVE_ROW_TYPES.into(),
            column_name_field: Self::XENIUM_COLUMN_NAMES.into(),
        }
    }
}

/// Read the sparse matrix of a 10x HDF5 file, its fields under
/// `root_group`, into a new backend at `backend_file`
/// ([`write_10x_matrix`]). Shared by `data-beans from-10x-matrix` and
/// [`try_open_or_convert`]. The caller prepares and finalizes the output
/// ([`prepare_output`], [`finalize_output`]).
#[cfg(feature = "hdf5")]
pub fn build_from_h5_matrix(
    h5_file: &str,
    root_group: &str,
    layout: &MatrixLayout,
    backend_file: &str,
    backend: &SparseIoBackend,
) -> anyhow::Result<()> {
    let file = hdf5::File::open(h5_file)?;
    info!("Opened 10x HDF5 file: {}", h5_file);
    let root = file
        .group(root_group)
        .map_err(|_| anyhow::anyhow!("no group `{root_group}` in {h5_file}"))?;
    let dataset = |field: &str| {
        root.dataset(field)
            .map_err(|_| anyhow::anyhow!("no dataset `{root_group}/{field}` in {h5_file}"))
    };
    // The arrays go once their triplets are made.
    let CooTripletsShape { triplets, shape } = {
        let values: Vec<f32> = dataset(&layout.data_field)?.read_raw()?;
        let indices: Vec<u64> = dataset(&layout.indices_field)?.read_raw()?;
        let indptr: Vec<u64> = dataset(&layout.indptr_field)?.read_raw()?;
        ValuesIndicesPointers {
            values: &values,
            indices: &indices,
            indptr: &indptr,
        }
        .to_coo(layout.pointer_type)?
    };
    info!(
        "Read {} non-zero elements reaching {} x {}",
        shape.nnz, shape.nrows, shape.ncols
    );
    let names = |field: &str| root.dataset(field).ok().map(read_hdf5_strings).transpose();
    write_10x_matrix(
        TenxMatrix {
            triplets,
            reach: (shape.nrows, shape.ncols),
            row_ids: names(&layout.row_id_field)?,
            row_names: names(&layout.row_name_field)?,
            row_types: names(&layout.row_type_field)?,
            column_names: names(&layout.column_name_field)?,
            keep_rows: None,
        },
        &layout.select_row_type,
        &layout.remove_row_type,
        backend_file,
        backend,
    )
}

/// Read the sparse matrix of a 10x-style Zarr store into a new backend at
/// `backend_file` ([`write_10x_matrix`]), columns named by Xenium cell id
/// when they are encoded so. Shared by `data-beans from-zarr` and
/// [`try_open_or_convert`], so both read a store alike. The caller prepares
/// and finalizes the output ([`prepare_output`], [`finalize_output`]).
pub fn build_from_zarr_matrix(
    zarr_file: &str,
    layout: &MatrixLayout,
    backend_file: &str,
    backend: &SparseIoBackend,
) -> anyhow::Result<()> {
    if !std::path::Path::new(zarr_file).exists() {
        let zip_variant = format!("{}.zip", zarr_file);
        let hint: Box<str> = if std::path::Path::new(&zip_variant).exists() {
            format!(" (did you mean {}?)", zip_variant).into()
        } else {
            Box::from("")
        };
        anyhow::bail!("Zarr file not found: {}{}", zarr_file, hint);
    }
    let store = open_zarr_store(zarr_file)?;
    info!("Opened zarr store: {}", zarr_file);

    // The arrays go once their triplets are made.
    let CooTripletsShape { triplets, shape } = {
        let indices: Vec<u64> = read_zarr_numerics(store.clone(), &layout.indices_field)?;
        let indptr: Vec<u64> = read_zarr_numerics(store.clone(), &layout.indptr_field)?;
        let values: Vec<f32> = read_zarr_numerics(store.clone(), &layout.data_field)?;
        ValuesIndicesPointers {
            values: &values,
            indices: &indices,
            indptr: &indptr,
        }
        .to_coo(layout.pointer_type)?
    };
    let TripletsShape { nrows, ncols, nnz } = shape;
    info!("Read {nnz} non-zero elements reaching {nrows} x {ncols}");

    // Names stored as a group attribute (Xenium) or as a string array.
    let names = |field: &str| {
        read_zarr_group_attr::<Vec<Box<str>>>(store.clone(), field)
            .or_else(|_| read_zarr_strings(store.clone(), field))
            .ok()
    };
    let column_names = read_zarr_flat_u32(store.clone(), &layout.column_name_field)
        .and_then(|(ids, shape)| {
            anyhow::ensure!(shape.len() == 2 && shape[1] == 2, "cell_id must be [N, 2]");
            parse_10x_cell_id_flat(&ids, shape[0] as usize)
        })
        .ok()
        .or_else(|| names(&layout.column_name_field));
    write_10x_matrix(
        TenxMatrix {
            triplets,
            reach: (nrows, ncols),
            row_ids: names(&layout.row_id_field),
            row_names: names(&layout.row_name_field),
            row_types: names(&layout.row_type_field),
            column_names,
            keep_rows: None,
        },
        &layout.select_row_type,
        &layout.remove_row_type,
        backend_file,
        backend,
    )
}

/// Convert to a Zarr backend at `output` with `build`, which writes it.
fn convert_with(
    output: &str,
    build: impl FnOnce(&str, &SparseIoBackend) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let (output, backend, backend_file) = prepare_output(output, SparseIoBackend::Zarr, false)?;
    build(&backend_file, &backend)?;
    finalize_output(&backend_file, &output)?;
    info!("Conversion done: {}", output);
    Ok(())
}

/// Convert a 10x HDF5 file (Cell Ranger / Space Ranger / Xenium
/// `*feature_bc_matrix.h5`) to a data-beans backend, read as `data-beans
/// from-10x-matrix` reads it with its defaults
/// ([`MatrixLayout::cell_ranger_h5`]).
#[cfg(feature = "hdf5")]
pub fn convert_h5_to_backend(h5_file: &str, output: &str) -> anyhow::Result<()> {
    convert_with(output, |backend_file, backend| {
        let layout = MatrixLayout::cell_ranger_h5();
        build_from_h5_matrix(
            h5_file,
            MatrixLayout::H5_ROOT,
            &layout,
            backend_file,
            backend,
        )
    })
}

/// Convert a 10x-style Zarr file (Xenium `cell_feature_matrix.zarr.zip` or
/// directory) to a data-beans backend, read as `data-beans from-zarr` reads
/// it with its defaults ([`MatrixLayout::xenium_zarr`]).
pub fn convert_zarr_to_backend(zarr_file: &str, output: &str) -> anyhow::Result<()> {
    convert_with(output, |backend_file, backend| {
        build_from_zarr_matrix(
            zarr_file,
            &MatrixLayout::xenium_zarr(),
            backend_file,
            backend,
        )
    })
}

/// The backend cached at `converted`, when it is there and whole: it opens
/// and knows its shape. `None` for a missing or unusable one.
fn cached_conversion(converted: &str) -> Option<Box<dyn SparseIo<IndexIter = Vec<usize>>>> {
    if !std::path::Path::new(converted).exists() {
        return None;
    }
    let data = open_sparse_matrix(converted, &SparseIoBackend::Zarr).ok()?;
    let whole =
        data.num_rows().is_some() && data.num_columns().is_some() && data.num_non_zeros().is_some();
    whole.then_some(data)
}

/// Where a conversion cached as `{base}.db.zarr` is built: a sibling no
/// other conversion writes to, in this process or another, so the cache
/// path only ever holds a finished backend.
fn staged_conversion(base: &str) -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NTH: AtomicUsize = AtomicUsize::new(0);
    format!(
        "{base}.db.partial-{}-{}.zarr",
        std::process::id(),
        NTH.fetch_add(1, Ordering::Relaxed)
    )
}

/// Move the backend built at `staged` to the cache path `converted`, in one
/// rename. The rename fails when a backend is already there, which is looked
/// at only then: a usable one is another conversion of the same input that
/// finished first, and it stays while `staged` is dropped. An unusable one
/// is moved aside, in one rename too, before `staged` takes its place, and
/// only what was moved aside is deleted; a backend another conversion
/// installs in between is therefore never deleted, it makes the second
/// rename fail and is then the cache.
fn install_conversion(staged: &str, converted: &str) -> anyhow::Result<()> {
    let rename = || Ok(std::fs::rename(staged, converted)?);
    let occupied = || std::path::Path::new(converted).exists();
    let usable = || cached_conversion(converted).is_some();
    let mut renamed: anyhow::Result<()> = rename();
    if renamed.is_err() && occupied() && !usable() {
        let stale = format!("{staged}.stale");
        if std::fs::rename(converted, &stale).is_ok() {
            renamed = rename();
            remove_file(&stale)?;
        }
    }
    let Err(e) = renamed else {
        return Ok(());
    };
    remove_file(staged)?;
    anyhow::ensure!(
        usable(),
        "cannot install the conversion at {converted}: {e}"
    );
    Ok(())
}

/// Try to open a data file directly; if that fails, attempt automatic
/// conversion from raw 10x formats (Cell Ranger h5, Xenium zarr/zarr.zip).
/// An AnnData `.h5ad` is not converted here: `data-beans from-h5ad` reads
/// it, with its own layout and metadata files.
///
/// Converted backends are cached as `{data_file}.db.zarr` next to
/// the original file so subsequent calls skip conversion. The conversion is
/// built beside the cache path and renamed onto it when done, so an
/// interrupted or concurrent conversion never leaves a partial cache; a
/// cache that does not open is converted again.
pub fn try_open_or_convert(
    data_file: &str,
) -> anyhow::Result<Box<dyn SparseIo<IndexIter = Vec<usize>>>> {
    let ext = file_ext(data_file)?;
    let backend = match ext.as_ref() {
        "h5" | "h5ad" => SparseIoBackend::HDF5,
        _ => SparseIoBackend::Zarr,
    };

    let original_err = match open_sparse_matrix(data_file, &backend) {
        Ok(data) => return Ok(data),
        Err(e) => e,
    };
    let base = strip_backend_suffix(data_file);
    let converted = format!("{}.db.zarr", base);

    if let Some(data) = cached_conversion(&converted) {
        info!("Using cached conversion: {}", converted);
        return Ok(data);
    }
    if std::path::Path::new(&converted).exists() {
        log::warn!(
            "the cached conversion {} is incomplete; converting {} again",
            converted,
            data_file
        );
    }

    // The conversion, and the command that does the same with options.
    type Convert = fn(&str, &str) -> anyhow::Result<()>;
    let (convert, command): (Convert, &str) = match ext.as_ref() {
        #[cfg(feature = "hdf5")]
        "h5" => (convert_h5_to_backend, "from-10x-matrix"),
        #[cfg(feature = "hdf5")]
        "h5ad" => anyhow::bail!(
            "{data_file} does not open as a data-beans backend ({original_err}), and an \
             AnnData .h5ad is not converted automatically; run `data-beans from-h5ad \
             {data_file} -o <output>` and use its output"
        ),
        #[cfg(not(feature = "hdf5"))]
        "h5" | "h5ad" => anyhow::bail!(
            "{} is an HDF5 file but data-beans was built without the `hdf5` \
             feature. Reinstall with `--features hdf5` (and a working libhdf5) \
             to read .h5/.h5ad inputs.",
            data_file
        ),
        "zarr" | "zip" => (convert_zarr_to_backend, "from-zarr"),
        _ => return Err(original_err),
    };

    let staged = staged_conversion(base);
    info!("Converting {} to backend: {}", data_file, converted);
    if let Err(e) = convert(data_file, &staged) {
        let _ = remove_file(&staged);
        // Converted here, the file is read with the defaults of `command`,
        // which the caller has no options to change.
        return Err(e.context(format!(
            "{data_file} is not converted automatically with the default options; run \
             `data-beans {command} {data_file} -o <output>` with its options, such as \
             --select-row-type for the feature types to keep, and use its output"
        )));
    }
    install_conversion(&staged, &converted)?;
    open_sparse_matrix(&converted, &SparseIoBackend::Zarr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// An uncompressed Zarr v2 array of little-endian u32 under `dir/name`.
    fn write_u32_array(dir: &Path, name: &str, shape: &[usize], values: &[u32]) {
        let d = dir.join(name);
        std::fs::create_dir_all(&d).unwrap();
        let meta = serde_json::json!({
            "zarr_format": 2, "shape": shape, "chunks": shape, "dtype": "<u4",
            "compressor": null, "fill_value": 0, "order": "C", "filters": null
        });
        std::fs::write(d.join(".zarray"), meta.to_string()).unwrap();
        let chunk = vec!["0"; shape.len()].join(".");
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(d.join(chunk), bytes).unwrap();
    }

    /// A store laid out as Xenium writes `cell_feature_matrix.zarr`: 3
    /// features × 4 cells, pointers over features, names as attributes of
    /// `/cell_features`, cell ids as a `[N, 2]` u32 array.
    fn xenium_like_store(dir: &Path) -> Vec<u32> {
        let cf = dir.join("cell_features");
        std::fs::create_dir_all(&cf).unwrap();
        std::fs::write(dir.join(".zgroup"), r#"{"zarr_format": 2}"#).unwrap();
        std::fs::write(cf.join(".zgroup"), r#"{"zarr_format": 2}"#).unwrap();
        let attrs = serde_json::json!({
            "feature_ids": ["FID1", "FID2", "FID3"],
            "feature_keys": ["GENE1", "GENE2", "NEG1"],
            "feature_types": ["gene", "gene", "negative_control_probe"],
            "number_cells": 4, "number_features": 3
        });
        std::fs::write(cf.join(".zattrs"), attrs.to_string()).unwrap();
        // Rows (features): GENE1 [1 0 2 0], GENE2 [0 3 0 4], NEG1 [5 0 0 0].
        write_u32_array(&cf, "indptr", &[4], &[0, 2, 4, 5]);
        write_u32_array(&cf, "indices", &[5], &[0, 2, 1, 3, 0]);
        write_u32_array(&cf, "data", &[5], &[1, 2, 3, 4, 5]);
        let ids = vec![16844, 1, 22527, 1, 16845, 1, 22528, 1];
        write_u32_array(&cf, "cell_id", &[4, 2], &ids);
        ids
    }

    #[test]
    fn a_xenium_store_converts_to_genes_by_cells_with_their_names() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("matrix.zarr");
        let ids = xenium_like_store(&store);
        let out = dir.path().join("out.db.zarr");
        let out = out.to_str().unwrap();
        convert_zarr_to_backend(store.to_str().unwrap(), out).unwrap();

        let data = open_sparse_matrix(out, &SparseIoBackend::Zarr).unwrap();
        // The control feature is dropped; genes are rows, cells columns.
        assert_eq!(
            data.row_names().unwrap(),
            vec![Box::from("FID1_GENE1"), Box::from("FID2_GENE2")]
        );
        assert_eq!(
            data.column_names().unwrap(),
            parse_10x_cell_id_flat(&ids, 4).unwrap()
        );
        let m = data.read_columns_dmatrix((0..4).collect()).unwrap();
        assert_eq!(m.shape(), (2, 4));
        assert_eq!(
            m.row(0).iter().copied().collect::<Vec<f32>>(),
            [1., 0., 2., 0.]
        );
        assert_eq!(
            m.row(1).iter().copied().collect::<Vec<f32>>(),
            [0., 3., 0., 4.]
        );
    }

    fn file_names(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn a_partial_cached_conversion_is_converted_again() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("matrix.zarr");
        xenium_like_store(&store);
        // What an interrupted conversion left at the cache path.
        let cache = dir.path().join("matrix.db.zarr");
        std::fs::create_dir_all(cache.join("by_column")).unwrap();
        std::fs::write(cache.join(".zgroup"), r#"{"zarr_format": 2}"#).unwrap();

        let data = try_open_or_convert(store.to_str().unwrap()).unwrap();
        assert_eq!(
            data.row_names().unwrap(),
            ["FID1_GENE1", "FID2_GENE2"].map(Box::from)
        );
        assert_eq!(data.num_columns(), Some(4));
        drop(data);
        // Nothing is left of the build but the cache, which the next call
        // opens as it is.
        assert_eq!(file_names(dir.path()), ["matrix.db.zarr", "matrix.zarr"]);
        let stamp = |p: &Path| std::fs::metadata(p).unwrap().modified().unwrap();
        let built = stamp(&cache);
        let again = try_open_or_convert(store.to_str().unwrap()).unwrap();
        assert_eq!(again.num_non_zeros(), Some(4));
        assert_eq!(stamp(&cache), built);
    }

    #[test]
    fn a_failed_conversion_leaves_no_cache() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("matrix.zarr");
        xenium_like_store(&store);
        // No feature is of a kept type, so the conversion fails.
        let attrs = serde_json::json!({
            "feature_ids": ["FID1", "FID2", "FID3"],
            "feature_keys": ["A", "B", "C"],
            "feature_types": vec!["negative_control_probe"; 3],
        });
        std::fs::write(store.join("cell_features/.zattrs"), attrs.to_string()).unwrap();

        let e = try_open_or_convert(store.to_str().unwrap())
            .err()
            .expect("no row is kept");
        // The error names the command whose options keep other types, and
        // still says what went wrong.
        let e = format!("{e:#}");
        assert!(e.contains("data-beans from-zarr"), "{e}");
        assert!(e.contains("--select-row-type"), "{e}");
        assert!(e.contains("negative_control_probe"), "{e}");
        assert_eq!(file_names(dir.path()), ["matrix.zarr"]);
    }

    #[test]
    fn a_cache_installed_meanwhile_stays_unless_it_is_unusable() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("matrix.zarr");
        xenium_like_store(&store);
        let path = |name: &str| dir.path().join(name).to_str().unwrap().to_string();
        let (cache, staged) = (path("matrix.db.zarr"), path("staged.zarr"));
        let convert = |to: &str| convert_zarr_to_backend(store.to_str().unwrap(), to).unwrap();

        // Another conversion finished first: its backend is the one kept.
        convert(&cache);
        let marker = Path::new(&cache).join("installed-first");
        std::fs::write(&marker, "").unwrap();
        convert(&staged);
        install_conversion(&staged, &cache).unwrap();
        assert!(marker.exists());
        assert!(!Path::new(&staged).exists());

        // What is there does not open: the staged backend takes its place.
        remove_file(&cache).unwrap();
        std::fs::create_dir_all(Path::new(&cache).join("by_column")).unwrap();
        convert(&staged);
        install_conversion(&staged, &cache).unwrap();
        assert!(cached_conversion(&cache).is_some());
        assert!(!Path::new(&staged).exists());
    }

    #[cfg(feature = "hdf5")]
    #[test]
    fn an_h5ad_file_is_sent_to_from_h5ad() {
        let dir = tempfile::tempdir().unwrap();
        let h5ad = dir.path().join("cells.h5ad");
        let file = hdf5::File::create(&h5ad).unwrap();
        file.create_group("X").unwrap();
        file.create_group("obs").unwrap();
        file.create_group("var").unwrap();
        drop(file);

        let e = try_open_or_convert(h5ad.to_str().unwrap())
            .err()
            .expect("an AnnData file is not a backend")
            .to_string();
        assert!(e.contains("from-h5ad"), "{e}");
        assert_eq!(file_names(dir.path()), ["cells.h5ad"]);
    }

    fn names(xs: &[&str]) -> Option<Vec<Box<str>>> {
        Some(xs.iter().map(|&x| Box::from(x)).collect())
    }

    #[test]
    fn trailing_empty_rows_and_columns_are_kept() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.zarr");
        let out = out.to_str().unwrap();
        // The data reaches 2 x 2, but the file names 3 rows and 3 columns;
        // the last row and column have nothing in them. Rows whose name is
        // empty or repeats the id are named by id alone.
        let m = TenxMatrix {
            triplets: vec![(0, 0, 1.0), (1, 1, 2.0)],
            reach: (2, 2),
            row_ids: names(&["FID1", "FID2", "FID3"]),
            row_names: names(&["GENE1", "FID2", ""]),
            row_types: names(&["Gene Expression"; 3]),
            column_names: names(&["BC1", "BC2", "BC3"]),
            keep_rows: None,
        };
        write_10x_matrix(m, "gene", "", out, &SparseIoBackend::Zarr).unwrap();

        let data = open_sparse_matrix(out, &SparseIoBackend::Zarr).unwrap();
        assert_eq!(
            data.row_names().unwrap(),
            ["FID1_GENE1", "FID2", "FID3"].map(Box::from)
        );
        assert_eq!(
            data.column_names().unwrap(),
            ["BC1", "BC2", "BC3"].map(Box::from)
        );
        let m = data.read_columns_dmatrix((0..3).collect()).unwrap();
        assert_eq!(m.shape(), (3, 3));
        assert_eq!(
            m.row(2).iter().sum::<f32>() + m.column(2).iter().sum::<f32>(),
            0.
        );
    }

    #[test]
    fn data_reaching_past_the_names_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.zarr");
        let m = TenxMatrix {
            triplets: vec![(2, 0, 1.0)],
            reach: (3, 1),
            row_ids: names(&["FID1", "FID2"]),
            row_names: None,
            row_types: None,
            column_names: None,
            keep_rows: None,
        };
        let e = write_10x_matrix(m, "gene", "", out.to_str().unwrap(), &SparseIoBackend::Zarr);
        assert!(e.unwrap_err().to_string().contains("names 2 rows"));
    }

    #[test]
    fn keeping_no_row_is_an_error_naming_the_types() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.zarr");
        let m = TenxMatrix {
            triplets: vec![(0, 0, 1.0), (1, 1, 2.0)],
            reach: (2, 2),
            row_ids: names(&["HTO1", "CD3"]),
            row_names: None,
            row_types: names(&["Multiplexing Capture", "Antibody Capture"]),
            column_names: None,
            keep_rows: None,
        };
        let e = write_10x_matrix(
            m,
            MatrixLayout::SELECT_ROW_TYPES,
            MatrixLayout::REMOVE_ROW_TYPES,
            out.to_str().unwrap(),
            &SparseIoBackend::Zarr,
        )
        .unwrap_err()
        .to_string();
        assert!(
            e.contains("Antibody Capture") && e.contains("gene,peak"),
            "{e}"
        );
        assert!(!out.exists());
    }

    #[test]
    fn keeping_no_row_names_the_row_set_that_leaves_none() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.zarr");
        // Two rows are genes, and the only row the file selects is of
        // another type.
        let m = TenxMatrix {
            triplets: vec![(0, 0, 1.0), (1, 1, 2.0), (2, 0, 3.0)],
            reach: (3, 2),
            row_ids: names(&["G1", "G2", "CD3"]),
            row_names: None,
            row_types: names(&["Gene Expression", "Gene Expression", "Antibody Capture"]),
            column_names: None,
            keep_rows: Some(vec![false, false, true]),
        };
        let e = write_10x_matrix(m, "gene", "", out.to_str().unwrap(), &SparseIoBackend::Zarr)
            .unwrap_err()
            .to_string();
        assert!(e.contains("2 are of a kept type"), "{e}");
        assert!(e.contains("1 are in `keep_rows`"), "{e}");
        assert!(!out.exists());
    }
}
