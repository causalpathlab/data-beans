// #![allow(dead_code)]

use crate::sparse_io::*;
use hdf5::filters::blosc_set_nthreads;
use legume_numeric::matrix::common_io::*;
use log::info;
use std::ops::Range;
use std::sync::Arc;

use anyhow::anyhow;

use crate::sparse_backend::shared;
use crate::utilities::io_helpers::{chunk_elems, parse_name_file};

const COMPRESSION_LEVEL: u8 = 5;

/// The rayon pool size as a Blosc thread count: Blosc takes a `u8`, so a
/// larger pool is capped at 255 threads instead of wrapping around.
fn blosc_threads(pool_threads: usize) -> u8 {
    u8::try_from(pool_threads.max(1)).unwrap_or(u8::MAX)
}

/// Match Blosc compression threads to the active rayon pool so
/// `--n-threads` (and `RAYON_NUM_THREADS`) cap total CPU usage.
fn match_blosc_threads_to_pool() {
    blosc_set_nthreads(blosc_threads(rayon::current_num_threads()));
}

/// 10x-like cell-feature matrix with hdf5 (feature x cell)
///
/// ```text
/// (root)
///     ├── nrow
///     ├── ncell
///     ├── by_column
///     │   ├── data
///     │   ├── indices (row indices)
///     │   └── indptr (column pointers)
///     └── by_row
///         ├── data
///         ├── indices (column indices)
///         └── indptr (row pointers)
/// ```
///
#[derive(Debug, Clone)]
pub struct SparseMtxData {
    backend: Arc<hdf5::File>,
    file_name: String,
    max_row_name_idx: usize,
    max_column_name_idx: usize,
    by_column_indptr: Vec<u64>,
    /// Streaming-write cursor: entries appended so far (see `note_streamed_nnz`).
    streamed_nnz: u64,
    by_row_indptr: Vec<u64>,
    by_column_indices: Option<Vec<u64>>,
    by_column_data: Option<Vec<f32>>,
    by_row_indices: Option<Vec<u64>>,
    by_row_data: Option<Vec<f32>>,
    /// Budget share of the preloaded column arrays, given back with them.
    column_preload: Option<crate::sparse_io::PreloadReservation>,
    /// Budget share of the preloaded row arrays, given back with them.
    row_preload: Option<crate::sparse_io::PreloadReservation>,
}

impl SparseMtxData {
    /// Create an empty new `SparseMtxData` with HDF5 file. If no
    /// `backend_file` is provided, a temporary file will be created.
    ///
    /// * `backend_file`: HDF5 file to be associated with
    pub fn new(backend_file: Option<&str>) -> anyhow::Result<Self> {
        let ret = match backend_file {
            Some(backend_file) => Self::register_backend_file(backend_file)?,
            None => {
                let backend_file = create_temp_dir_file(".h5")?;
                let backend_file = backend_file.to_str().expect("to_str failed");
                Self::register_backend_file(backend_file)?
            }
        };
        Ok(ret)
    }

    /// Open existing `SparseMtxData` from a backend HDF5 file
    pub fn open(backend_file: &str) -> anyhow::Result<Self> {
        let hdf5_backend = hdf5::File::open(backend_file)?;

        if let (Some(nrow), Some(ncol), Some(nnz)) = (
            Self::_num_rows(&hdf5_backend),
            Self::_num_columns(&hdf5_backend),
            Self::_num_nnz(&hdf5_backend),
        ) {
            info!("#rows: {}, #columns: {}, #non-zeros: {}", nrow, ncol, nnz);
        } else {
            anyhow::bail!("Couldn't figure out the size of this sparse matrix data");
        }

        let mut ret = Self {
            backend: hdf5_backend.into(),
            file_name: backend_file.to_string(),
            max_row_name_idx: MAX_ROW_NAME_IDX,
            max_column_name_idx: MAX_COLUMN_NAME_IDX,
            by_column_indptr: vec![],
            streamed_nnz: 0,
            by_row_indptr: vec![],
            by_column_indices: None,
            by_column_data: None,
            by_row_indices: None,
            by_row_data: None,
            column_preload: None,
            row_preload: None,
        };

        ret.read_column_indptr()?;
        ret.read_row_indptr()?;

        Ok(ret)
    }

    /// Create `SparseMtxData` from mtx file with `backend_file` as
    /// the backend file.  If no `backend_file` is provided, it will
    /// be the same as `mtx_file` with `.h5` extension.
    /// * `mtx_file`: mtx file to be read into HDF5 backend
    /// * `backend_file`: HDF5 file to be associated with
    /// * `index_by_row`: if true, the matrix will be indexed by row
    pub fn from_mtx_file(
        mtx_file: &str,
        backend_file: Option<&str>,
        index_by_row: Option<bool>,
    ) -> anyhow::Result<Self> {
        // create an object
        let mut ret = match backend_file {
            Some(backend_file) => {
                info!("backend file : {}", backend_file);
                Self::register_backend_file(backend_file)?
            }
            None => {
                let backend_file = mtx_file.to_string() + ".h5";
                info!("backend file : {}", backend_file);
                Self::register_backend_file(backend_file.as_ref())?
            }
        };

        // populate data from mtx file
        ret.import_mtx_file(mtx_file, index_by_row == Some(true))?;

        info!("created sparse backend from {}", mtx_file);
        Ok(ret)
    }

    /// Create a new `SparseMtxData` from ndarray with its backend
    /// HDF5 file. If no `backend_file` is provided, a temporary file
    /// will be created.
    /// * `array`: 2D array to be written into HDF5 backend
    /// * `backend_file`: HDF5 file to be associated with
    /// * `index_by_row`: if true, the matrix will be indexed by row
    ///
    pub fn from_ndarray(
        array: &Array2<f32>,
        backend_file: Option<&str>,
        index_by_row: Option<bool>,
    ) -> anyhow::Result<Self> {
        let mut ret = match backend_file {
            Some(backend_file) => Self::register_backend_file(backend_file)?,
            None => {
                let backend_file = create_temp_dir_file(".h5")?;
                let backend_file = backend_file.to_str().expect("to_str failed");
                Self::register_backend_file(backend_file)?
            }
        };

        ret.import_ndarray_by_col(array)?; // populate data from mtx file
        ret.read_column_indptr()?; // reload column indptr

        if Some(true) == index_by_row {
            ret.import_ndarray_by_row(array)?; //
            ret.read_row_indptr()?; //
        }

        Ok(ret)
    }

    /// Create a new `SparseMtxData` from DMatrix with its backend
    /// HDF5 file. If no `backend_file` is provided, a temporary file
    /// will be created.
    /// * `array`: 2D array to be written into HDF5 backend
    /// * `backend_file`: HDF5 file to be associated with
    /// * `index_by_row`: if true, the matrix will be indexed by row
    ///
    pub fn from_dmatrix(
        matrix: &DMatrix<f32>,
        backend_file: Option<&str>,
        index_by_row: Option<bool>,
    ) -> anyhow::Result<Self> {
        let mut ret = match backend_file {
            Some(backend_file) => Self::register_backend_file(backend_file)?,
            None => {
                let backend_file = create_temp_dir_file(".h5")?;
                let backend_file = backend_file.to_str().expect("to_str failed");
                Self::register_backend_file(backend_file)?
            }
        };

        ret.import_dmatrix_by_col(matrix)?; // populate data from mtx file
        ret.read_column_indptr()?; // reload column indptr

        if Some(true) == index_by_row {
            ret.import_dmatrix_by_row(matrix)?; //
            ret.read_row_indptr()?; //
        }

        Ok(ret)
    }

    /////////////////////////////
    // purely helper functions //
    /////////////////////////////

    fn _num_rows(file: &hdf5::File) -> Option<usize> {
        file.attr("nrow").ok()?.read_scalar().ok()
    }

    fn _num_columns(file: &hdf5::File) -> Option<usize> {
        file.attr("ncol").ok()?.read_scalar().ok()
    }

    fn _num_nnz(file: &hdf5::File) -> Option<usize> {
        file.attr("nnz").ok()?.read_scalar().ok()
    }

    fn set_attrs(&mut self, attr_name: &str, value: usize) -> anyhow::Result<()> {
        if self.backend.attr(attr_name).is_err() {
            self.backend
                .new_attr::<usize>()
                .create(attr_name)?
                .write_scalar(&value)?;
        } else if self.backend.attr(attr_name)?.read_scalar::<usize>()? != value {
            return Err(anyhow!(format!("{} mismatch", attr_name)));
        }
        Ok(())
    }

    /// Read a whole 1-D dataset into a `Vec`. The array the read fills owns
    /// one contiguous buffer, which is handed over as it is rather than
    /// copied, so the dataset is held in memory once.
    fn read_vec<T: hdf5::H5Type>(dataset: &hdf5::Dataset) -> anyhow::Result<Vec<T>> {
        let (vec, offset) = dataset.read_1d::<T>()?.into_raw_vec_and_offset();
        debug_assert!(matches!(offset, None | Some(0)));
        Ok(vec)
    }

    /// Store `names` as the string dataset `key` under the root, replacing
    /// one already there: names are registered again when they change after
    /// the first registration (cell names gaining a hashtag suffix, say).
    fn write_names(
        &mut self,
        key: &str,
        names: &[hdf5::types::VarLenUnicode],
    ) -> anyhow::Result<()> {
        use hdf5::types::VarLenUnicode;

        let root = self.backend.group("/")?;
        if root.link_exists(key) {
            root.unlink(key)?;
        }
        // A chunk dimension must be positive, also for an empty list.
        root.new_dataset::<VarLenUnicode>()
            .shape(names.len())
            .chunk([names.len().max(1)])
            .create(key)?
            .write(names)?;
        Ok(())
    }

    /////////////////////
    // backend related //
    /////////////////////

    /// Associate sparse matrix data with a HDF5 file
    /// * `hdf5_file`: HDF5 file to be associated with
    fn register_backend_file(hdf5_file: &str) -> anyhow::Result<Self> {
        let hdf5_backend = hdf5::File::create(hdf5_file)?;

        Ok(Self {
            backend: hdf5_backend.into(),
            file_name: hdf5_file.to_string(),
            max_row_name_idx: MAX_ROW_NAME_IDX,
            max_column_name_idx: MAX_COLUMN_NAME_IDX,
            by_column_indptr: vec![],
            streamed_nnz: 0,
            by_row_indptr: vec![],
            by_column_indices: None,
            by_column_data: None,
            by_row_indices: None,
            by_row_data: None,
            column_preload: None,
            row_preload: None,
        })
    }

    /// Forget everything held in memory about the file's contents: the
    /// preloaded columns and rows and the streaming cursor. For when the
    /// file behind this handle has been replaced; arrays preloaded from the
    /// old file would otherwise be served against the new index pointers.
    fn forget_file_contents(&mut self) {
        self.clean_preloaded_columns();
        self.clean_preloaded_rows();
        self.streamed_nnz = 0;
    }
}

impl SparseIo for SparseMtxData {
    type IndexIter = Vec<usize>;

    /// Helper function to create a new backend file
    fn initialize_backend(&mut self) -> anyhow::Result<()> {
        self.remove_backend_file()?;
        self.backend = hdf5::File::create(&self.file_name)?.into();
        self.max_column_name_idx = MAX_COLUMN_NAME_IDX;
        self.max_row_name_idx = MAX_ROW_NAME_IDX;
        self.by_column_indptr = vec![];
        self.by_row_indptr = vec![];
        // Nothing read from the removed file describes the new one.
        self.forget_file_contents();

        Ok(())
    }

    /// Record the shape of the mtx file into the HDF5 backend
    fn record_mtx_shape(&mut self, mtx_shape: Option<(usize, usize, usize)>) -> anyhow::Result<()> {
        if let Some((nrow, ncol, nnz)) = mtx_shape {
            self.set_attrs("nrow", nrow)?;
            self.set_attrs("ncol", ncol)?;
            self.set_attrs("nnz", nnz)?;
            self.backend.flush()?;
        }

        Ok(())
    }

    /// Read column index pointers
    fn column_indptr(&self) -> &[u64] {
        &self.by_column_indptr
    }

    fn row_indptr(&self) -> &[u64] {
        &self.by_row_indptr
    }

    fn metadata(&self) -> Metadata {
        use hdf5::types::VarLenUnicode;
        self.backend
            .attr(meta::ATTR)
            .and_then(|a| a.read_scalar::<VarLenUnicode>())
            .ok()
            .and_then(|json| serde_json::from_str(json.as_str()).ok())
            .unwrap_or_default()
    }

    fn set_metadata(&mut self, values: &Metadata) -> anyhow::Result<()> {
        use hdf5::types::VarLenUnicode;
        anyhow::ensure!(
            !self.backend.is_read_only(),
            "{} is open for reading only: set its metadata while creating it",
            self.file_name
        );
        if self.backend.attr(meta::ATTR).is_ok() {
            self.backend.delete_attr(meta::ATTR)?;
        }
        if !values.is_empty() {
            let json: VarLenUnicode = serde_json::to_string(values)?
                .parse()
                .map_err(|e| anyhow!("metadata: {e}"))?;
            self.backend
                .new_attr::<VarLenUnicode>()
                .create(meta::ATTR)?
                .write_scalar(&json)?;
        }
        self.backend.flush()?;
        Ok(())
    }

    fn reopen_backend(&mut self) -> anyhow::Result<()> {
        // The open handle points at the deleted inode after the swap; reopen on
        // the path and refresh the resident indptrs.
        self.backend = hdf5::File::open_rw(&self.file_name)?.into();
        self.forget_file_contents();
        self.read_column_indptr()?;
        self.read_row_indptr()?;
        Ok(())
    }

    fn note_streamed_nnz(&mut self, n: u64) {
        self.streamed_nnz += n;
    }

    fn streamed_nnz(&self) -> u64 {
        self.streamed_nnz
    }

    fn reset_streamed_nnz(&mut self) {
        self.streamed_nnz = 0;
    }

    fn read_column_indptr(&mut self) -> anyhow::Result<()> {
        if let Ok(by_column) = self.backend.group("/by_column") {
            let indptr = by_column.dataset("indptr")?.read_1d::<u64>()?;
            self.by_column_indptr.clear();
            self.by_column_indptr.extend(indptr);
        }
        Ok(())
    }

    /// Read row index pointers
    fn read_row_indptr(&mut self) -> anyhow::Result<()> {
        if let Ok(by_row) = self.backend.group("/by_row") {
            let indptr = by_row.dataset("indptr")?.read_1d::<u64>()?;
            self.by_row_indptr.clear();
            self.by_row_indptr.extend(indptr);
        }
        Ok(())
    }

    fn preload_columns(&mut self) -> anyhow::Result<()> {
        if self.by_column_data.is_some() && self.by_column_indices.is_some() {
            return Ok(());
        }
        let reservation = match self.num_non_zeros() {
            Some(nnz) => match crate::sparse_io::reserve_preload(nnz, "column") {
                Some(r) => Some(r),
                None => return Ok(()),
            },
            None => None,
        };
        let by_column = self.backend.group("/by_column")?;
        let data = Self::read_vec::<f32>(&by_column.dataset("data")?)?;
        let indices = Self::read_vec::<u64>(&by_column.dataset("indices")?)?;

        index_audit::check_preload(
            index_audit::Major::Column,
            self.num_rows(),
            self.num_columns(),
            &self.by_column_indptr,
            &indices,
            data.len(),
        )?;

        self.by_column_data = Some(data);
        self.column_preload = reservation;
        self.by_column_indices = Some(indices);
        Ok(())
    }

    fn clean_preloaded_columns(&mut self) {
        self.by_column_data = None;
        self.by_column_indices = None;
        self.column_preload = None;
    }

    fn preload_rows(&mut self) -> anyhow::Result<()> {
        if self.by_row_data.is_some() && self.by_row_indices.is_some() {
            return Ok(());
        }
        let reservation = match self.num_non_zeros() {
            Some(nnz) => match crate::sparse_io::reserve_preload(nnz, "row") {
                Some(r) => Some(r),
                None => return Ok(()),
            },
            None => None,
        };
        let by_row = self.backend.group("/by_row")?;
        let data = Self::read_vec::<f32>(&by_row.dataset("data")?)?;
        let indices = Self::read_vec::<u64>(&by_row.dataset("indices")?)?;

        index_audit::check_preload(
            index_audit::Major::Row,
            self.num_rows(),
            self.num_columns(),
            &self.by_row_indptr,
            &indices,
            data.len(),
        )?;

        self.by_row_data = Some(data);
        self.row_preload = reservation;
        self.by_row_indices = Some(indices);
        Ok(())
    }

    fn clean_preloaded_rows(&mut self) {
        self.by_row_data = None;
        self.by_row_indices = None;
        self.row_preload = None;
    }

    /// Remove backend file to free up disk space
    fn remove_backend_file(&self) -> anyhow::Result<()> {
        let backend = std::path::Path::new(&self.file_name);
        if backend.exists() {
            std::fs::remove_file(backend)?;
        }
        Ok(())
    }

    /// Access file name of the hdf5 backend file
    fn get_backend_file_name(&self) -> &str {
        &self.file_name
    }

    fn backend_type(&self) -> SparseIoBackend {
        SparseIoBackend::HDF5
    }

    /// Export the data to a mtx file. This will take time.
    /// * `mtx_file`: mtx file to be written
    fn to_mtx_file(&self, mtx_file: &str) -> anyhow::Result<()> {
        let by_column = self.backend.group("/by_column")?;

        let indptr = by_column.dataset("indptr")?.read_1d::<u64>()?;
        let data = by_column.dataset("data")?;
        let indices = by_column.dataset("indices")?;

        if let (Some(ncol), Some(nrow), Some(nnz)) =
            (self.num_columns(), self.num_rows(), self.num_non_zeros())
        {
            let indptr = indptr
                .as_slice()
                .ok_or_else(|| anyhow!("column pointers are not contiguous in memory"))?;

            shared::write_mtx_file(mtx_file, |buf| {
                shared::write_mtx(
                    buf,
                    (nrow, ncol, nnz),
                    indptr,
                    shared::MTX_STREAM_BLOCK,
                    |s, e| {
                        let range = (s as usize)..(e as usize);
                        Ok((
                            data.read_slice_1d::<f32, _>(range.clone())?,
                            indices.read_slice_1d::<u64, _>(range)?,
                        ))
                    },
                )
            })?;
            info!(
                "{}: {} rows, {} columns, {} non-zeros",
                mtx_file, nrow, ncol, nnz
            );

            Ok(())
        } else {
            Err(anyhow!("Unable to figure out the size of the backend data"))
        }
    }

    /// Set row names for the matrix
    /// * `row_name_file`: a file each line contains row name words
    fn register_row_names_file(&mut self, row_name_file: &str) {
        self.register_names_file(
            "/row_names",
            row_name_file,
            0..self.max_row_name_idx,
            ROW_SEP,
        )
        .expect("failed to add row names");
    }

    /// Set row names for the matrix
    /// * `rows`: a vector of row names
    fn register_row_names_vec(&mut self, rows: &[Box<str>]) {
        self.register_names_vec("/row_names", rows)
            .expect("failed to add row names");
    }

    /// Set column names for the matrix
    /// * `column_name_file`: a file each line contains column name words
    fn register_column_names_file(&mut self, column_name_file: &str) {
        self.register_names_file(
            "/column_names",
            column_name_file,
            0..self.max_column_name_idx,
            COLUMN_SEP,
        )
        .expect("failed to add column names");
    }

    /// Set column names for the matrix
    /// * `columns`: a vector of column names
    fn register_column_names_vec(&mut self, columns: &[Box<str>]) {
        self.register_names_vec("/column_names", columns)
            .expect("failed to add column names");
    }

    /// Number of rows in the underlying data matrix
    fn num_rows(&self) -> Option<usize> {
        Self::_num_rows(&self.backend)
    }

    /// Number of columns in the underlying data matrix
    fn num_columns(&self) -> Option<usize> {
        Self::_num_columns(&self.backend)
    }

    /// Number of non-zero elements
    fn num_non_zeros(&self) -> Option<usize> {
        Self::_num_nnz(&self.backend)
    }

    /// Add arbitrary names (a vector of strings)
    /// * `group_name`: group name
    /// * `name_file`: a file each line contains name words
    /// * `name_columns`: range of columns to be used for name
    /// * `name_sep`: separator for name columns
    fn register_names_file(
        &mut self,
        key: &str,
        name_file: &str,
        name_columns: Range<usize>,
        name_sep: &str,
    ) -> anyhow::Result<()> {
        use hdf5::types::VarLenUnicode;

        let _names: Vec<VarLenUnicode> = parse_name_file(name_file, name_columns, name_sep)?
            .iter()
            .map(|x| x.parse().expect("invalid name"))
            .collect();

        self.write_names(key, &_names)
    }

    /// Add arbitrary names (a vector of strings)
    /// * `group_name`: group name
    /// * `names`: a file each line contains name words
    fn register_names_vec(&mut self, key: &str, names: &[Box<str>]) -> anyhow::Result<()> {
        use hdf5::types::VarLenUnicode;

        let _names: Vec<VarLenUnicode> = names
            .iter()
            .map(|x| x.to_string().parse().expect("invalid name"))
            .collect::<Vec<_>>();

        self.write_names(key, &_names)
    }

    fn row_names(&self) -> anyhow::Result<Vec<Box<str>>> {
        self.retrieve_registered_names("/row_names")
    }

    fn column_names(&self) -> anyhow::Result<Vec<Box<str>>> {
        self.retrieve_registered_names("/column_names")
    }

    /// Get back the registered names
    /// * `key`: key for the registered names
    fn retrieve_registered_names(&self, key: &str) -> anyhow::Result<Vec<Box<str>>> {
        use hdf5::types::VarLenUnicode;

        let root = self.backend.group("/")?;

        let ret = root.dataset(key)?.read_1d::<VarLenUnicode>()?;

        Ok(ret.iter().map(|x| x.to_string().into_boxed_str()).collect())
    }

    /// Read columns within the range and return a vector of triplets (row, col, value)
    /// * `col` : usize
    ///
    fn read_triplets_by_single_column(
        &self,
        j_data: usize,
    ) -> anyhow::Result<(usize, usize, Vec<(u64, u64, f32)>)> {
        let by_column = self.backend.group("/by_column")?;
        let indptr = &self.by_column_indptr;

        let nrow = self
            .num_rows()
            .ok_or(anyhow!("can't figure out the number of rows"))?;
        index_audit::check_requested_slots("column", [j_data], indptr.len())?;

        if let (Some(data), Some(indices)) = (&self.by_column_data, &self.by_column_indices) {
            let ncol_out = 1;
            let jj = 0;

            let mut ret = Vec::new();
            index_audit::emit_slot(
                "preloaded column",
                j_data,
                indptr,
                indices,
                data,
                nrow,
                &mut ret,
                |ii, x| (ii, jj, x),
            )?;
            Ok((nrow, ncol_out, ret))
        } else {
            let data = by_column.dataset("data")?;
            let indices = by_column.dataset("indices")?;

            let mut ret = Vec::new();
            let ncol_out = 1;
            let jj = 0;

            // [start, end)
            anyhow::ensure!(
                indptr[j_data] <= indptr[j_data + 1],
                "column read: slot {j_data} points at [{}, {})",
                indptr[j_data],
                indptr[j_data + 1]
            );
            let start = indptr[j_data] as usize;
            let end = indptr[j_data + 1] as usize;

            if start < end {
                let data_slice = data.read_slice_1d::<f32, _>(start..end)?;
                let indices_slice = indices.read_slice_1d::<u64, _>(start..end)?;

                for k in 0..(end - start) {
                    let x_ij = data_slice[k];
                    let ii = indices_slice[k];
                    if ii >= nrow as u64 {
                        return Err(index_audit::inner_out_of_range(
                            "column read",
                            start + k,
                            ii,
                            nrow,
                        ));
                    }
                    ret.push((ii, jj, x_ij));
                }
            }

            Ok((nrow, ncol_out, ret))
        }
    }

    /// Read columns within the range and return a vector of triplets (row, col, value)
    /// * `columns` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_triplets_by_columns(
        &self,
        columns: Self::IndexIter,
    ) -> anyhow::Result<(usize, usize, Vec<(u64, u64, f32)>)> {
        // need to open backend again?
        // let backend = hdf5::File::open(&self.file_name)?;
        let by_column = self.backend.group("/by_column")?;

        let indptr = &self.by_column_indptr;

        let columns_vec = columns.into_iter().collect::<Vec<usize>>();

        let nrow = self
            .num_rows()
            .ok_or(anyhow!("can't figure out the number of rows"))?;

        let ncol = self
            .num_columns()
            .ok_or(anyhow!("can't figure out the number of columns"))?;
        if let Some(&bad) = columns_vec.iter().find(|&&j| j >= ncol) {
            anyhow::bail!("read column {bad}: outside {ncol} columns");
        }
        index_audit::check_requested_slots("column", columns_vec.iter().copied(), indptr.len())?;

        let min_start = columns_vec
            .iter()
            .map(|&j_data| indptr[j_data])
            .min()
            .unwrap_or(0);

        let max_end = columns_vec
            .iter()
            .map(|&j_data| indptr[j_data + 1])
            .max()
            .unwrap_or(0);

        if let (Some(data), Some(indices)) = (&self.by_column_data, &self.by_column_indices) {
            let ncol_out = columns_vec.len();

            let mut ret: Vec<(u64, u64, f32)> =
                Vec::with_capacity(max_end.saturating_sub(min_start) as usize);

            for (jj, &j_data) in columns_vec.iter().enumerate() {
                let jj = jj as u64;
                index_audit::emit_slot(
                    "preloaded column",
                    j_data,
                    indptr,
                    indices,
                    data,
                    nrow,
                    &mut ret,
                    |ii, x| (ii, jj, x),
                )?;
            }

            Ok((nrow, ncol_out, ret))
        } else {
            // Cold path: coalesce abutting/overlapping indptr ranges into a
            // single `read_slice_1d` call. For a contiguous block of N cells
            // this collapses N HDF5 retrievals into 1 — same idea zarr's
            // chunk-store path uses (shared `coalesce_and_emit`).
            let data = by_column.dataset("data")?;
            let indices = by_column.dataset("indices")?;

            let ncol_out = columns_vec.len();

            let mut tagged: Vec<(u64, u64, u64)> = columns_vec
                .iter()
                .enumerate()
                .filter_map(|(jj, &j_data)| {
                    if j_data >= ncol {
                        return None;
                    }
                    let start = indptr[j_data];
                    let end = indptr[j_data + 1];
                    (start < end).then_some((jj as u64, start, end))
                })
                .collect();
            tagged.sort_by_key(|&(_, start, _)| start);

            let ret = shared::coalesce_and_emit(
                &tagged,
                nrow,
                |jj, ii, val| (ii, jj, val),
                |s, e| {
                    let data_buf = data.read_slice_1d::<f32, _>((s as usize)..(e as usize))?;
                    let indices_buf =
                        indices.read_slice_1d::<u64, _>((s as usize)..(e as usize))?;
                    Ok((data_buf, indices_buf))
                },
            )?;
            Ok((nrow, ncol_out, ret))
        }
    }

    fn csc_column_arrays(&self) -> Option<(&[u64], &[u64], &[f32])> {
        match (
            self.by_column_data.as_ref(),
            self.by_column_indices.as_ref(),
        ) {
            (Some(data), Some(indices)) if !self.by_column_indptr.is_empty() => Some((
                self.by_column_indptr.as_slice(),
                indices.as_slice(),
                data.as_slice(),
            )),
            _ => None,
        }
    }

    /// Read rows within the range and return a vector of triplets (row, column, value)
    /// * `rows` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_triplets_by_rows(
        &self,
        rows: Self::IndexIter,
    ) -> anyhow::Result<(usize, usize, Vec<(u64, u64, f32)>)> {
        let indptr = &self.by_row_indptr;

        let rows_vec = rows.into_iter().collect::<Vec<usize>>();

        let (ncol, nrow) = match (self.num_columns(), self.num_rows()) {
            (Some(ncol), Some(nrow)) => (ncol, nrow),
            _ => return Err(anyhow!("Unable to figure out the size of the backend data")),
        };
        let nrow_out = rows_vec.len();
        index_audit::check_requested_slots(
            "row",
            rows_vec.iter().copied().filter(|&i| i < nrow),
            indptr.len(),
        )?;

        if let (Some(data), Some(indices)) = (&self.by_row_data, &self.by_row_indices) {
            let mut nnz_total: usize = 0;
            let valid: Vec<(u64, usize)> = rows_vec
                .iter()
                .enumerate()
                .filter_map(|(ii, &i_data)| {
                    if i_data >= nrow {
                        return None;
                    }
                    nnz_total += indptr[i_data + 1].saturating_sub(indptr[i_data]) as usize;
                    Some((ii as u64, i_data))
                })
                .collect();

            let mut ret: Vec<(u64, u64, f32)> = Vec::with_capacity(nnz_total);
            for (ii, i_data) in valid {
                index_audit::emit_slot(
                    "preloaded row",
                    i_data,
                    indptr,
                    indices,
                    data,
                    ncol,
                    &mut ret,
                    |jj, x| (ii, jj, x),
                )?;
            }
            return Ok((nrow_out, ncol, ret));
        }

        // Cold path: coalesce abutting/overlapping indptr ranges into single
        // read_slice_1d calls, matching read_triplets_by_columns and zarr's
        // by-row path (shared `coalesce_and_emit`). CSR: tag = output row,
        // inner = column.
        let by_row = self.backend.group("/by_row")?;
        let data = by_row.dataset("data")?;
        let indices = by_row.dataset("indices")?;

        let mut tagged: Vec<(u64, u64, u64)> = rows_vec
            .iter()
            .enumerate()
            .filter_map(|(ii, &i_data)| {
                if i_data >= nrow {
                    return None;
                }
                let start = indptr[i_data];
                let end = indptr[i_data + 1];
                (start < end).then_some((ii as u64, start, end))
            })
            .collect();
        tagged.sort_by_key(|&(_, start, _)| start);

        let ret = shared::coalesce_and_emit(
            &tagged,
            ncol,
            |ii, jj, val| (ii, jj, val),
            |s, e| {
                let data_buf = data.read_slice_1d::<f32, _>((s as usize)..(e as usize))?;
                let indices_buf = indices.read_slice_1d::<u64, _>((s as usize)..(e as usize))?;
                Ok((data_buf, indices_buf))
            },
        )?;
        Ok((nrow_out, ncol, ret))
    }

    /// Helper function to add CSR dataset to HDF5 backend
    ///
    /// ```text
    ///     └── by_row
    ///         ├── data
    ///         ├── indices (column indices)
    ///         └── isndptr (row pointers)
    /// ```
    fn record_csr_dataset_backend(
        &mut self,
        csr_cols: &[u64],
        csr_vals: &[f32],
        csr_rowptr: &[u64],
    ) -> anyhow::Result<()> {
        // Populate them into HDF5
        if self.backend.group("/by_row").is_err() {
            let _root = self.backend.create_group("/by_row")?;
            // info!("Group: {:?} created", root);
        }

        match_blosc_threads_to_pool();

        let csr = self.backend.group("/by_row")?;

        csr.new_dataset::<f32>()
            .shape(csr_vals.len())
            .chunk([chunk_elems(csr_vals.len(), std::mem::size_of::<f32>())])
            .blosc_blosclz(COMPRESSION_LEVEL, true)
            .create("data")?
            .write(&csr_vals)?;

        csr.new_dataset::<u64>()
            .shape(csr_rowptr.len())
            .chunk([chunk_elems(csr_rowptr.len(), std::mem::size_of::<u64>())])
            .blosc_blosclz(COMPRESSION_LEVEL, true)
            .create("indptr")?
            .write(&csr_rowptr)?;

        csr.new_dataset::<u64>()
            .shape(csr_cols.len())
            .chunk([chunk_elems(csr_cols.len(), std::mem::size_of::<u64>())])
            .blosc_blosclz(COMPRESSION_LEVEL, true)
            .create("indices")?
            .write(&csr_cols)?;

        self.backend.flush()?;

        Ok(())
    }

    fn cs_create(&mut self, key: CsKey, len: usize) -> anyhow::Result<()> {
        let (group_path, ds_name) = match key {
            CsKey::CscData => ("/by_column", "data"),
            CsKey::CscIndices => ("/by_column", "indices"),
            CsKey::CscIndptr => ("/by_column", "indptr"),
            CsKey::CsrData => ("/by_row", "data"),
            CsKey::CsrIndices => ("/by_row", "indices"),
            CsKey::CsrIndptr => ("/by_row", "indptr"),
        };

        if self.backend.group(group_path).is_err() {
            self.backend.create_group(group_path)?;
        }

        match_blosc_threads_to_pool();

        // Exactly `len` elements, none for an empty array, so what a reader
        // finds matches the index pointers.
        let group = self.backend.group(group_path)?;
        match key {
            CsKey::CscData | CsKey::CsrData => {
                group
                    .new_dataset::<f32>()
                    .shape(len)
                    .chunk([chunk_elems(len, std::mem::size_of::<f32>())])
                    .blosc_blosclz(COMPRESSION_LEVEL, true)
                    .create(ds_name)?;
            }
            _ => {
                group
                    .new_dataset::<u64>()
                    .shape(len)
                    .chunk([chunk_elems(len, std::mem::size_of::<u64>())])
                    .blosc_blosclz(COMPRESSION_LEVEL, true)
                    .create(ds_name)?;
            }
        }
        self.backend.flush()?;
        Ok(())
    }

    fn cs_write_u64(&mut self, key: CsKey, offset: u64, data: &[u64]) -> anyhow::Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let (group_path, ds_name) = match key {
            CsKey::CscIndices => ("/by_column", "indices"),
            CsKey::CscIndptr => ("/by_column", "indptr"),
            CsKey::CsrIndices => ("/by_row", "indices"),
            CsKey::CsrIndptr => ("/by_row", "indptr"),
            CsKey::CscData | CsKey::CsrData => {
                return Err(anyhow!("cs_write_u64 called on f32 slot {:?}", key));
            }
        };
        let group = self.backend.group(group_path)?;
        let ds = group.dataset(ds_name)?;
        let lo = offset as usize;
        let hi = lo + data.len();
        ds.write_slice(data, ndarray::s![lo..hi])?;
        Ok(())
    }

    fn cs_write_f32(&mut self, key: CsKey, offset: u64, data: &[f32]) -> anyhow::Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let (group_path, ds_name) = match key {
            CsKey::CscData => ("/by_column", "data"),
            CsKey::CsrData => ("/by_row", "data"),
            _ => {
                return Err(anyhow!("cs_write_f32 called on u64 slot {:?}", key));
            }
        };
        let group = self.backend.group(group_path)?;
        let ds = group.dataset(ds_name)?;
        let lo = offset as usize;
        let hi = lo + data.len();
        ds.write_slice(data, ndarray::s![lo..hi])?;
        Ok(())
    }

    /// Helper function to add CSC dataset to HDF5 backend
    ///
    /// ```text
    /// Helper function to record the CSC dataset
    ///     ├── by_column
    ///     │   ├── data
    ///     │   ├── indices (row indices)
    ///     │   └── indptr (column pointers)
    /// ```
    fn record_csc_dataset_backend(
        &mut self,
        csc_rows: &[u64],
        csc_vals: &[f32],
        csc_colptr: &[u64],
    ) -> anyhow::Result<()> {
        // Populate them into HDF5
        if self.backend.group("/by_column").is_err() {
            let _root = self.backend.create_group("/by_column")?;
            // info!("Group: {:?} created", root);
        }

        match_blosc_threads_to_pool();

        let csc = self.backend.group("/by_column")?;

        csc.new_dataset::<f32>()
            .shape(csc_vals.len())
            .chunk([chunk_elems(csc_vals.len(), std::mem::size_of::<f32>())])
            .blosc_blosclz(COMPRESSION_LEVEL, true)
            .create("data")?
            .write(&csc_vals)?;

        csc.new_dataset::<u64>()
            .shape(csc_colptr.len())
            .chunk([chunk_elems(csc_colptr.len(), std::mem::size_of::<u64>())])
            .blosc_blosclz(COMPRESSION_LEVEL, true)
            .create("indptr")?
            .write(&csc_colptr)?;

        csc.new_dataset::<u64>()
            .shape(csc_rows.len())
            .chunk([chunk_elems(csc_rows.len(), std::mem::size_of::<u64>())])
            .blosc_blosclz(COMPRESSION_LEVEL, true)
            .create("indices")?
            .write(&csc_rows)?;

        self.backend.flush()?;

        Ok(())
    }
}

#[cfg(test)]
#[path = "hdf5_tests.rs"]
mod tests;
