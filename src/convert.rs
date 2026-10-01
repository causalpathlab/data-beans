use crate::hdf5_io::*;
use crate::sparse_io::*;
use crate::sparse_util::*;
use crate::utilities::name_matching::{
    compose_id_name, filter_row_indices_by_type, make_names_unique,
};
use crate::zarr_io::*;

use legume_numeric::matrix::common_io::*;
use log::info;

/// The sparse matrix of a 10x file as read, before rows are named and kept:
/// its triplets, how far they reach, and whatever names the file lists.
pub struct TenxMatrix {
    pub triplets: Vec<(u64, u64, f32)>,
    /// Rows and columns the triplets reach (`max index + 1`).
    pub reach: (usize, usize),
    pub row_ids: Option<Vec<Box<str>>>,
    pub row_names: Option<Vec<Box<str>>>,
    pub row_types: Option<Vec<Box<str>>>,
    pub column_names: Option<Vec<Box<str>>>,
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
/// type (comma-separated, case-insensitive patterns), columns named as the
/// file names them. Shared by the 10x readers, so every format and entry
/// point names and keeps rows alike.
pub fn write_10x_matrix(
    m: TenxMatrix,
    select_row_type: &str,
    remove_row_type: &str,
    backend_file: &str,
    backend: &SparseIoBackend,
) -> anyhow::Result<()> {
    let nrows = axis_len(m.row_ids.as_ref().map(Vec::len), m.reach.0, "rows")?;
    let ncols = axis_len(m.column_names.as_ref().map(Vec::len), m.reach.1, "columns")?;
    let nnz = m.triplets.len();
    info!("Matrix: {nrows} x {ncols}");
    let index = |n: usize| {
        (0..n)
            .map(|i| i.to_string().into_boxed_str())
            .collect::<Vec<_>>()
    };
    let row_ids = m.row_ids.unwrap_or_else(|| index(nrows));
    let row_names = m.row_names.unwrap_or_else(|| vec![Box::from(""); nrows]);
    let row_types = m
        .row_types
        .unwrap_or_else(|| vec![Box::from(select_row_type); nrows]);
    anyhow::ensure!(
        row_names.len() == nrows && row_types.len() == nrows,
        "the file lists {nrows} row ids but {} row names and {} row types",
        row_names.len(),
        row_types.len()
    );
    let column_names = m.column_names.unwrap_or_else(|| index(ncols));

    let mut row_ids = compose_id_name(row_ids, row_names);
    make_names_unique(&mut row_ids);
    let select_rows = filter_row_indices_by_type(&row_types, select_row_type, remove_row_type);

    let mut out = create_sparse_from_triplets_owned(
        m.triplets,
        (nrows, ncols, nnz),
        Some(backend_file),
        Some(backend),
    )?;
    info!("Created sparse matrix: {}", backend_file);
    out.register_row_names_vec(&row_ids);
    out.register_column_names_vec(&column_names);
    if select_rows.len() < nrows {
        info!(
            "Keeping {} of {nrows} rows of type `{select_row_type}`",
            select_rows.len()
        );
        out.subset_columns_rows(None, Some(&select_rows))?;
    }
    Ok(())
}

/// Where a 10x HDF5 file keeps its sparse matrix and the names of its rows
/// and columns, and which rows to keep. [`Default`] is Cell Ranger / Space
/// Ranger / Xenium `*feature_bc_matrix.h5`: features × barcodes under
/// `matrix`, pointers over barcodes.
#[derive(Clone, Debug)]
pub struct H5MatrixLayout {
    pub root_group: Box<str>,
    pub data_field: Box<str>,
    pub indices_field: Box<str>,
    pub indptr_field: Box<str>,
    pub pointer_type: IndexPointerType,
    pub row_id_field: Box<str>,
    pub row_name_field: Box<str>,
    pub row_type_field: Box<str>,
    pub select_row_type: Box<str>,
    pub remove_row_type: Box<str>,
    pub column_name_field: Box<str>,
}

impl H5MatrixLayout {
    pub const ROOT: &'static str = "matrix";
    pub const DATA: &'static str = "data";
    pub const INDICES: &'static str = "indices";
    pub const INDPTR: &'static str = "indptr";
    pub const ROW_IDS: &'static str = "features/id";
    pub const ROW_NAMES: &'static str = "features/name";
    pub const ROW_TYPES: &'static str = "features/feature_type";
    pub const COLUMN_NAMES: &'static str = "barcodes";
}

impl Default for H5MatrixLayout {
    fn default() -> Self {
        Self {
            root_group: Self::ROOT.into(),
            data_field: Self::DATA.into(),
            indices_field: Self::INDICES.into(),
            indptr_field: Self::INDPTR.into(),
            pointer_type: IndexPointerType::Column,
            row_id_field: Self::ROW_IDS.into(),
            row_name_field: Self::ROW_NAMES.into(),
            row_type_field: Self::ROW_TYPES.into(),
            select_row_type: ZarrMatrixLayout::SELECT_ROW_TYPES.into(),
            remove_row_type: ZarrMatrixLayout::REMOVE_ROW_TYPES.into(),
            column_name_field: Self::COLUMN_NAMES.into(),
        }
    }
}

/// Read the sparse matrix of a 10x HDF5 file into a new backend at
/// `backend_file` ([`write_10x_matrix`]). Shared by `data-beans
/// from-10x-matrix` and [`try_open_or_convert`]. The caller clears
/// `backend_file` beforehand and finalizes after.
#[cfg(feature = "hdf5")]
pub fn build_from_h5_matrix(
    h5_file: &str,
    layout: &H5MatrixLayout,
    backend_file: &str,
    backend: &SparseIoBackend,
) -> anyhow::Result<()> {
    let file = hdf5::File::open(h5_file)?;
    info!("Opened 10x HDF5 file: {}", h5_file);
    let root = file
        .group(&layout.root_group)
        .map_err(|_| anyhow::anyhow!("no group `{}` in {}", layout.root_group, h5_file))?;
    let read = |field: &str| -> anyhow::Result<Vec<u64>> {
        Ok(root
            .dataset(field)
            .map_err(|_| {
                anyhow::anyhow!("no dataset `{}/{field}` in {h5_file}", layout.root_group)
            })?
            .read_1d::<u64>()?
            .to_vec())
    };
    let values: Vec<f32> = root
        .dataset(&layout.data_field)
        .map_err(|_| {
            anyhow::anyhow!(
                "no dataset `{}/{}` in {h5_file}",
                layout.root_group,
                layout.data_field
            )
        })?
        .read_1d::<f32>()?
        .to_vec();
    let (indices, indptr) = (read(&layout.indices_field)?, read(&layout.indptr_field)?);
    let CooTripletsShape { triplets, shape } = ValuesIndicesPointers {
        values: &values,
        indices: &indices,
        indptr: &indptr,
    }
    .to_coo(layout.pointer_type)?;
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
        },
        &layout.select_row_type,
        &layout.remove_row_type,
        backend_file,
        backend,
    )
}

/// Convert a 10x HDF5 file (Cell Ranger / Space Ranger / Xenium
/// `*feature_bc_matrix.h5`) to a data-beans backend, read as `data-beans
/// from-10x-matrix` reads it with its defaults ([`H5MatrixLayout::default`]).
#[cfg(feature = "hdf5")]
pub fn convert_h5_to_backend(h5_file: &str, output: &str) -> anyhow::Result<()> {
    let (backend, backend_file) =
        resolve_backend_file(&Box::from(output), Some(SparseIoBackend::Zarr))?;
    if std::path::Path::new(backend_file.as_ref()).exists() {
        info!("Removing existing backend file: {}", &backend_file);
        remove_file(&backend_file)?;
    }
    build_from_h5_matrix(h5_file, &H5MatrixLayout::default(), &backend_file, &backend)?;
    finalize_zarr_output(&backend_file, output)?;
    info!("Conversion done: {}", output);
    Ok(())
}

/// Where a 10x-style Zarr store keeps its sparse matrix and the names of
/// its rows and columns, and which rows to keep. [`Default`] is Xenium's
/// `cell_feature_matrix.zarr`: features × cells, pointers over features,
/// feature names as attributes of `/cell_features`.
#[derive(Clone, Debug)]
pub struct ZarrMatrixLayout {
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

impl ZarrMatrixLayout {
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
}

impl Default for ZarrMatrixLayout {
    fn default() -> Self {
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

/// Read the sparse matrix of a 10x-style Zarr store into a new backend at
/// `backend_file` ([`write_10x_matrix`]), columns named by Xenium cell id
/// when they are encoded so. Shared by
/// `data-beans from-zarr` and [`try_open_or_convert`], so both read a store
/// alike. The caller clears `backend_file` beforehand and finalizes after.
pub fn build_from_zarr_matrix(
    zarr_file: &str,
    layout: &ZarrMatrixLayout,
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

    let indices: Vec<u64> = read_zarr_numerics(store.clone(), &layout.indices_field)?;
    let indptr: Vec<u64> = read_zarr_numerics(store.clone(), &layout.indptr_field)?;
    let values: Vec<f32> = read_zarr_numerics(store.clone(), &layout.data_field)?;

    let CooTripletsShape { triplets, shape } = ValuesIndicesPointers {
        values: &values,
        indices: &indices,
        indptr: &indptr,
    }
    .to_coo(layout.pointer_type)?;
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
        },
        &layout.select_row_type,
        &layout.remove_row_type,
        backend_file,
        backend,
    )
}

/// Convert a 10x-style Zarr file (Xenium `cell_feature_matrix.zarr.zip` or
/// directory) to a data-beans backend, read as `data-beans from-zarr` reads
/// it with its defaults ([`ZarrMatrixLayout::default`]).
pub fn convert_zarr_to_backend(zarr_file: &str, output: &str) -> anyhow::Result<()> {
    let (backend, backend_file) =
        resolve_backend_file(&Box::from(output), Some(SparseIoBackend::Zarr))?;
    if std::path::Path::new(backend_file.as_ref()).exists() {
        info!("Removing existing backend file: {}", &backend_file);
        remove_file(&backend_file)?;
    }
    build_from_zarr_matrix(
        zarr_file,
        &ZarrMatrixLayout::default(),
        &backend_file,
        &backend,
    )?;
    finalize_zarr_output(&backend_file, output)?;
    info!("Conversion done: {}", output);
    Ok(())
}

/// What conversion writes, as stamped on the caches it leaves: bumped
/// whenever that changes, so a cache an older conversion wrote is converted
/// again rather than read.
const CONVERSION: u64 = 2;
const CONVERSION_ATTR: &str = "data_beans_conversion";

/// Whether the cache at `path` was written by this conversion or a later one.
fn conversion_is_current(path: &str) -> bool {
    open_zarr_store(path)
        .and_then(|store| read_zarr_group_attr::<u64>(store, CONVERSION_ATTR))
        .is_ok_and(|v| v >= CONVERSION)
}

fn stamp_conversion(path: &str) -> anyhow::Result<()> {
    let store = std::sync::Arc::new(zarrs::filesystem::FilesystemStore::new(path)?);
    let mut group = zarrs::group::Group::open(store, "/")?;
    group
        .attributes_mut()
        .insert(CONVERSION_ATTR.into(), CONVERSION.into());
    group.store_metadata()?;
    Ok(())
}

/// Try to open a data file directly; if that fails, attempt automatic
/// conversion from raw 10x formats (h5/h5ad, zarr/zarr.zip).
///
/// Converted backends are cached as `{data_file}.db.zarr` next to
/// the original file so subsequent calls skip conversion; a cache an older
/// conversion wrote is converted again.
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
                if conversion_is_current(&converted) {
                    info!("Using cached conversion: {}", converted);
                    return open_sparse_matrix(&converted, &SparseIoBackend::Zarr);
                }
                info!("Converting again: {converted} was written by an older conversion");
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

            stamp_conversion(&converted)?;
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
        };
        write_10x_matrix(m, "gene", "", out, &SparseIoBackend::Zarr).unwrap();

        let data = open_sparse_matrix(out, &SparseIoBackend::Zarr).unwrap();
        assert_eq!(
            data.row_names().unwrap(),
            names(&["FID1_GENE1", "FID2", "FID3"]).unwrap()
        );
        assert_eq!(
            data.column_names().unwrap(),
            names(&["BC1", "BC2", "BC3"]).unwrap()
        );
        let m = data.read_columns_dmatrix((0..3).collect()).unwrap();
        assert_eq!(m.shape(), (3, 3));
        assert_eq!(
            m.row(2).iter().sum::<f32>() + m.column(2).iter().sum::<f32>(),
            0.
        );
    }

    #[test]
    fn a_cache_from_an_older_conversion_is_converted_again() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("matrix.zarr");
        xenium_like_store(&store);
        let store = store.to_str().unwrap();
        // As an older conversion left it: unstamped, its row named by index.
        let cache = dir.path().join("matrix.db.zarr");
        let stale = TenxMatrix {
            triplets: vec![(0, 0, 1.0)],
            reach: (1, 1),
            row_ids: None,
            row_names: None,
            row_types: None,
            column_names: None,
        };
        write_10x_matrix(
            stale,
            "gene",
            "",
            cache.to_str().unwrap(),
            &SparseIoBackend::Zarr,
        )
        .unwrap();

        let genes = names(&["FID1_GENE1", "FID2_GENE2"]).unwrap();
        let rows = |store| try_open_or_convert(store).unwrap().row_names().unwrap();
        assert_eq!(rows(store), genes);
        // Current now, the cache is read even with the store gone.
        std::fs::remove_dir_all(store).unwrap();
        assert_eq!(rows(store), genes);
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
        };
        let e = write_10x_matrix(m, "gene", "", out.to_str().unwrap(), &SparseIoBackend::Zarr);
        assert!(e.unwrap_err().to_string().contains("names 2 rows"));
    }
}
