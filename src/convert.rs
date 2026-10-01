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

/// The length of one axis: the number of names the file lists for it (one
/// per row or column, so trailing all-zero rows and columns, which leave no
/// trace in the triplets, still count), else as far as the triplets reach.
/// An error when the triplets reach past the names.
fn axis_len(named: Option<usize>, reach: usize, what: &str) -> anyhow::Result<usize> {
    match named {
        Some(n) if n >= reach => Ok(n),
        Some(n) => anyhow::bail!("the file names {n} {what} but its data reaches {reach}"),
        None => Ok(reach),
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
    let nrows = axis_len(row_ids.as_ref().map(Vec::len), reach.0, "rows")?;
    let ncols = axis_len(column_names.as_ref().map(Vec::len), reach.1, "columns")?;
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
    if let Some(keep_rows) = &keep_rows {
        keep.retain(|&i| keep_rows[i]);
    }
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

/// Try to open a data file directly; if that fails, attempt automatic
/// conversion from raw 10x formats (h5/h5ad, zarr/zarr.zip).
///
/// Converted backends are cached as `{data_file}.db.zarr` next to
/// the original file so subsequent calls skip conversion.
pub fn try_open_or_convert(
    data_file: &str,
) -> anyhow::Result<Box<dyn SparseIo<IndexIter = Vec<usize>>>> {
    let ext = file_ext(data_file)?;
    let backend = match ext.as_ref() {
        "h5" | "h5ad" => SparseIoBackend::HDF5,
        _ => SparseIoBackend::Zarr,
    };

    match open_sparse_matrix(data_file, &backend) {
        Ok(data) => Ok(data),
        Err(original_err) => {
            let base = strip_backend_suffix(data_file);
            let converted = format!("{}.db.zarr", base);

            if std::path::Path::new(&converted).exists() {
                info!("Using cached conversion: {}", converted);
                return open_sparse_matrix(&converted, &SparseIoBackend::Zarr);
            }

            match ext.as_ref() {
                "h5" | "h5ad" => {
                    #[cfg(feature = "hdf5")]
                    {
                        info!(
                            "Converting h5/h5ad to backend: {} -> {}",
                            data_file, converted
                        );
                        convert_h5_to_backend(data_file, &converted)?;
                    }
                    #[cfg(not(feature = "hdf5"))]
                    {
                        anyhow::bail!(
                            "{} is an HDF5 file but data-beans was built without the `hdf5` \
                             feature. Reinstall with `--features hdf5` (and a working libhdf5) \
                             to read .h5/.h5ad inputs.",
                            data_file
                        );
                    }
                }
                "zarr" | "zip" => {
                    info!("Converting zarr to backend: {} -> {}", data_file, converted);
                    convert_zarr_to_backend(data_file, &converted)?;
                }
                _ => return Err(original_err),
            }

            open_sparse_matrix(&converted, &SparseIoBackend::Zarr)
        }
    }
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
}
