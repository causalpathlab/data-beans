#![cfg(feature = "ndarray")]

//! Behaviour every sparse backend must share, run against zarr and (when
//! built in) HDF5: what a handle reads after its file was rebuilt in place,
//! an all-zero matrix written by the streaming path, single-column reads,
//! name registration, the Matrix Market export, and a handle whose backend is
//! initialised again.

use data_beans::sparse_io::*;
use legume_numeric::matrix::mtx_io::read_mtx_triplets;
use ndarray::Array2;

type Backend = Box<dyn SparseIo<IndexIter = Vec<usize>>>;

fn backends() -> Vec<(SparseIoBackend, &'static str)> {
    vec![
        (SparseIoBackend::Zarr, "zarr"),
        #[cfg(feature = "hdf5")]
        (SparseIoBackend::HDF5, "h5"),
    ]
}

fn names(prefix: &str, n: usize) -> Vec<Box<str>> {
    (0..n).map(|i| format!("{prefix}{i}").into()).collect()
}

/// A small matrix with an empty row (3), an empty column (2) and columns of
/// several entries.
fn reference() -> Array2<f32> {
    let mut raw = Array2::<f32>::zeros((5, 6));
    raw[(0, 0)] = 1.0;
    raw[(4, 0)] = 2.0;
    raw[(1, 1)] = 3.0;
    raw[(2, 1)] = 4.0;
    raw[(4, 1)] = 5.0;
    raw[(0, 3)] = 6.0;
    raw[(1, 4)] = 7.0;
    raw[(2, 4)] = 8.0;
    raw[(2, 5)] = 9.0;
    raw
}

/// `raw` in a named backend file under `dir`, with rows `r0..` and columns
/// `c0..`; nothing preloaded.
fn build(
    raw: &Array2<f32>,
    dir: &tempfile::TempDir,
    backend: &SparseIoBackend,
    ext: &str,
) -> anyhow::Result<(Backend, String)> {
    let path = dir.path().join(format!("x.{ext}"));
    let path = path.to_str().expect("utf8").to_string();
    let mut data = create_sparse_from_ndarray(raw, Some(&path), Some(backend))?;
    data.register_row_names_vec(&names("r", raw.nrows()));
    data.register_column_names_vec(&names("c", raw.ncols()));
    Ok((data, path))
}

/// `(row, column, value)` of the nonzeros of `raw`, column-major.
fn dense_triplets(raw: &Array2<f32>) -> Vec<(u64, u64, f32)> {
    sorted(ndarray_to_triplets(raw))
}

/// Stream `raw` into `data` as one column band, then build its CSR half.
fn stream(
    data: &mut dyn SparseIo<IndexIter = Vec<usize>>,
    raw: &Array2<f32>,
) -> anyhow::Result<()> {
    let triplets = dense_triplets(raw);
    let mut colptr = vec![0u64; raw.ncols()];
    for &(_, j, _) in &triplets {
        for start in &mut colptr[j as usize + 1..] {
            *start += 1;
        }
    }
    let rows: Vec<u64> = triplets.iter().map(|t| t.0).collect();
    let values: Vec<f32> = triplets.iter().map(|t| t.2).collect();
    data.begin_streaming_csc((raw.nrows(), raw.ncols(), triplets.len()))?;
    data.append_csc_slab(0, 0, &colptr, &rows, &values)?;
    data.finalize_streaming_csc()?;
    data.build_csr_from_csc_streaming()
}

fn sorted(mut t: Vec<(u64, u64, f32)>) -> Vec<(u64, u64, f32)> {
    t.sort_by_key(|a| (a.1, a.0));
    t
}

/// Every way of reading `data` gives `raw`.
fn assert_reads(data: &dyn SparseIo<IndexIter = Vec<usize>>, raw: &Array2<f32>, what: &str) {
    let (nrow, ncol) = raw.dim();
    let expected = dense_triplets(raw);
    assert_eq!(data.num_rows(), Some(nrow), "{what}: rows");
    assert_eq!(data.num_columns(), Some(ncol), "{what}: columns");
    assert_eq!(data.num_non_zeros(), Some(expected.len()), "{what}: nnz");

    let (nr, nc, by_col) = data.read_triplets_by_columns((0..ncol).collect()).unwrap();
    assert_eq!((nr, nc), (nrow, ncol), "{what}: column read shape");
    assert_eq!(sorted(by_col), expected, "{what}: by columns");

    let (nr, nc, by_row) = data.read_triplets_by_rows((0..nrow).collect()).unwrap();
    assert_eq!((nr, nc), (nrow, ncol), "{what}: row read shape");
    assert_eq!(sorted(by_row), expected, "{what}: by rows");

    for j in 0..ncol {
        let (nr, nc, column) = data.read_triplets_by_single_column(j).unwrap();
        assert_eq!((nr, nc), (nrow, 1), "{what}: column {j} shape");
        // rows ascending, as stored, and the one output column is 0
        let expected_j: Vec<(u64, u64, f32)> = expected
            .iter()
            .filter(|t| t.1 == j as u64)
            .map(|&(i, _, x)| (i, 0, x))
            .collect();
        assert_eq!(column, expected_j, "{what}: column {j}");
    }
}

#[test]
fn every_read_path_gives_the_matrix_before_and_after_a_preload() -> anyhow::Result<()> {
    for (backend, ext) in backends() {
        let dir = tempfile::tempdir()?;
        let raw = reference();
        let (mut data, _) = build(&raw, &dir, &backend, ext)?;
        assert_reads(&*data, &raw, &format!("{ext}, from the file"));
        // a second pass is served by whatever the first left in memory
        assert_reads(&*data, &raw, &format!("{ext}, again"));

        data.preload_columns()?;
        data.preload_rows()?;
        let (indptr, indices, values) = data.csc_column_arrays().expect("preloaded");
        assert_eq!(indptr, [0, 2, 5, 5, 6, 8, 9], "{ext}");
        assert_eq!(indices, [0, 4, 1, 2, 4, 0, 1, 2, 2], "{ext}");
        assert_eq!(values, [1., 2., 3., 4., 5., 6., 7., 8., 9.], "{ext}");
        assert_reads(&*data, &raw, &format!("{ext}, preloaded"));
    }
    Ok(())
}

/// `reorder_rows` rebuilds the backend file under the open handle. Reads made
/// before it must not colour reads made after: the handle has to serve the
/// rebuilt matrix, here with rows dropped and reversed.
#[test]
fn reads_after_reorder_rows_see_the_rebuilt_matrix() -> anyhow::Result<()> {
    for (backend, ext) in backends() {
        let dir = tempfile::tempdir()?;
        let raw = reference();
        let (mut data, path) = build(&raw, &dir, &backend, ext)?;
        assert_reads(&*data, &raw, &format!("{ext}, before"));

        let kept = [4usize, 2, 0];
        let order: Vec<Box<str>> = kept.iter().map(|i| format!("r{i}").into()).collect();
        data.reorder_rows(&order)?;

        let mut expected = Array2::<f32>::zeros((kept.len(), raw.ncols()));
        for (new, &old) in kept.iter().enumerate() {
            expected.row_mut(new).assign(&raw.row(old));
        }
        assert_reads(&*data, &expected, &format!("{ext}, same handle"));
        assert_eq!(data.row_names()?, order, "{ext}");
        assert_eq!(data.column_names()?, names("c", raw.ncols()), "{ext}");
        drop(data);

        let reopened = open_sparse_matrix(&path, &backend)?;
        assert_reads(&*reopened, &expected, &format!("{ext}, reopened"));
    }
    Ok(())
}

/// A subset that keeps only an empty column goes through the streaming
/// writer with no entries. The stored arrays must then be empty too, so the
/// preload audit accepts the backend.
#[test]
fn an_all_zero_streamed_matrix_reads_and_preloads() -> anyhow::Result<()> {
    for (backend, ext) in backends() {
        let dir = tempfile::tempdir()?;
        let raw = reference();
        let (mut data, path) = build(&raw, &dir, &backend, ext)?;
        data.subset_columns_rows(Some(&vec![2]), None)?;

        let expected = Array2::<f32>::zeros((raw.nrows(), 1));
        assert_reads(&*data, &expected, &format!("{ext}, same handle"));
        drop(data);

        let mut reopened = open_sparse_matrix(&path, &backend)?;
        assert_reads(&*reopened, &expected, &format!("{ext}, reopened"));
        reopened.preload_columns()?;
        reopened.preload_rows()?;
        let (indptr, indices, values) = reopened.csc_column_arrays().expect("preloaded");
        assert_eq!(indptr, [0, 0], "{ext}");
        assert!(indices.is_empty() && values.is_empty(), "{ext}");
        assert_reads(&*reopened, &expected, &format!("{ext}, preloaded"));
    }
    Ok(())
}

/// Names are registered a second time when they change after the first
/// registration; the later list replaces the earlier one.
#[test]
fn registering_names_again_replaces_them() -> anyhow::Result<()> {
    for (backend, ext) in backends() {
        let dir = tempfile::tempdir()?;
        let raw = reference();
        let (mut data, path) = build(&raw, &dir, &backend, ext)?;

        // longer names, and a list of a different length
        let rows = names("gene_with_a_longer_name_", raw.nrows());
        let columns = names("barcode@tag", raw.ncols() + 2);
        data.register_row_names_vec(&rows);
        data.register_column_names_vec(&columns);
        assert_eq!(data.row_names()?, rows, "{ext}");
        assert_eq!(data.column_names()?, columns, "{ext}");
        drop(data);

        let reopened = open_sparse_matrix(&path, &backend)?;
        assert_eq!(reopened.row_names()?, rows, "{ext}");
        assert_eq!(reopened.column_names()?, columns, "{ext}");
    }
    Ok(())
}

#[test]
fn an_empty_name_list_is_stored() -> anyhow::Result<()> {
    for (backend, ext) in backends() {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join(format!("empty.{ext}"));
        let mut data = create_sparse_streaming_empty(path.to_str(), Some(&backend))?;
        data.register_names_vec("/column_names", &[])?;
        data.register_row_names_vec(&[]);
        assert!(data.column_names()?.is_empty(), "{ext}");
        assert!(data.row_names()?.is_empty(), "{ext}");
    }
    Ok(())
}

/// The export writes the header's dimensions and every stored entry, with
/// empty columns skipped, on both backends.
#[test]
fn to_mtx_file_round_trips() -> anyhow::Result<()> {
    for (backend, ext) in backends() {
        let dir = tempfile::tempdir()?;
        let raw = reference();
        let (data, _) = build(&raw, &dir, &backend, ext)?;
        let mtx = dir.path().join("out.mtx");
        let mtx = mtx.to_str().expect("utf8");
        data.to_mtx_file(mtx)?;

        let expected = dense_triplets(&raw);
        let (triplets, shape) = read_mtx_triplets(mtx)?;
        assert_eq!(shape, (raw.nrows(), raw.ncols(), expected.len()), "{ext}");
        assert_eq!(sorted(triplets), expected, "{ext}");
    }
    Ok(())
}

/// An export that fails part-way, here on a stored row index outside the
/// matrix, leaves no file behind.
#[test]
fn a_failed_to_mtx_file_leaves_no_file() -> anyhow::Result<()> {
    for (backend, ext) in backends() {
        let dir = tempfile::tempdir()?;
        let raw = reference();
        let (mut data, _) = build(&raw, &dir, &backend, ext)?;
        data.cs_write_u64(CsKey::CscIndices, 0, &[raw.nrows() as u64])?;

        let mtx = dir.path().join("out.mtx");
        let err = data
            .to_mtx_file(mtx.to_str().expect("utf8"))
            .expect_err("a row index past the last row");
        assert!(format!("{err:#}").contains("to_mtx_file"), "{ext}: {err:#}");
        assert!(!mtx.exists(), "{ext}: the incomplete file is removed");
    }
    Ok(())
}

/// `initialize_backend` starts the file over under the open handle. Nothing
/// the handle held of the old matrix, preloaded arrays and streaming cursor
/// included, carries into the matrix written next.
#[test]
fn an_initialized_backend_keeps_nothing_of_the_old_matrix() -> anyhow::Result<()> {
    for (backend, ext) in backends() {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join(format!("again.{ext}"));
        let mut data = create_sparse_streaming_empty(path.to_str(), Some(&backend))?;
        let raw = reference();
        stream(&mut *data, &raw)?;
        data.preload_columns()?;
        data.preload_rows()?;
        assert!(data.csc_column_arrays().is_some(), "{ext}: preloaded");
        assert_eq!(data.streamed_nnz(), 9, "{ext}");

        data.initialize_backend()?;
        assert!(data.csc_column_arrays().is_none(), "{ext}: preload dropped");
        assert_eq!(data.streamed_nnz(), 0, "{ext}");

        let mut next = Array2::<f32>::zeros((3, 2));
        next[(2, 0)] = -1.0;
        next[(0, 1)] = -2.0;
        next[(1, 1)] = -3.0;
        stream(&mut *data, &next)?;
        assert_reads(&*data, &next, &format!("{ext}, same handle"));
    }
    Ok(())
}

/// `reopen_backend` opens the file at the handle's path again. Columns and
/// rows preloaded from the file that was there before describe the old
/// matrix; the handle serves the new one.
#[test]
fn a_reopened_backend_serves_the_new_store_not_the_old_preload() -> anyhow::Result<()> {
    for (backend, ext) in backends() {
        let dir = tempfile::tempdir()?;
        let raw = reference();
        let (mut data, path) = build(&raw, &dir, &backend, ext)?;
        data.preload_columns()?;
        data.preload_rows()?;
        assert!(data.csc_column_arrays().is_some(), "{ext}: preloaded");
        assert_reads(&*data, &raw, &format!("{ext}, before"));

        let mut next = Array2::<f32>::zeros((3, 2));
        next[(2, 0)] = -1.0;
        next[(0, 1)] = -2.0;
        next[(1, 1)] = -3.0;
        let other = dir.path().join(format!("y.{ext}"));
        drop(create_sparse_from_ndarray(
            &next,
            other.to_str(),
            Some(&backend),
        )?);
        data.remove_backend_file()?;
        std::fs::rename(&other, &path)?;

        data.reopen_backend()?;
        assert!(data.csc_column_arrays().is_none(), "{ext}: preload dropped");
        assert_reads(&*data, &next, &format!("{ext}, reopened"));
    }
    Ok(())
}
