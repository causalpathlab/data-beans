#![cfg(feature = "ndarray")]

//! `column_subset`'s streamed selections against the dense matrix they are
//! taken from: every way the output's size is found (off the by-column
//! indptr, off the by-row indptr, by a counting pass) and a row map that
//! reorders, drops and leaves rows empty.

use data_beans::column_subset::{stream_column_selection, stream_mapped_selection, RowMap};
use data_beans::sparse_io::*;

type Data = Box<dyn SparseIo<IndexIter = Vec<usize>>>;

/// 5 rows x 6 columns, about half the cells filled, all values distinct.
fn fixture() -> (ndarray::Array2<f32>, Data) {
    let mut arr = ndarray::Array2::<f32>::zeros((5, 6));
    for r in 0..5 {
        for c in 0..6 {
            if (r * 3 + c * 5) % 7 < 4 {
                arr[(r, c)] = (r * 10 + c + 1) as f32;
            }
        }
    }
    let mut data = create_sparse_from_ndarray(&arr, None, None).expect("fixture");
    let rows: Vec<Box<str>> = (0..5).map(|r| format!("r{r}").into()).collect();
    let cols: Vec<Box<str>> = (0..6).map(|c| format!("c{c}").into()).collect();
    data.register_row_names_vec(&rows);
    data.register_column_names_vec(&cols);
    (arr, data)
}

fn names(prefix: &str, idx: &[usize]) -> Vec<Box<str>> {
    idx.iter().map(|i| format!("{prefix}{i}").into()).collect()
}

/// The backend at `path` equals `want`, in its by-column and by-row forms.
fn assert_holds(path: &str, want: &ndarray::Array2<f32>) -> anyhow::Result<()> {
    let out = open_sparse_matrix(path, &SparseIoBackend::Zarr)?;
    assert_eq!(out.num_rows(), Some(want.nrows()));
    assert_eq!(out.num_columns(), Some(want.ncols()));
    let nnz = want.iter().filter(|&&x| x != 0.0).count();
    assert_eq!(out.num_non_zeros(), Some(nnz));
    let by_column = out.read_columns_dmatrix((0..want.ncols()).collect())?;
    let by_row = out.read_rows_dmatrix((0..want.nrows()).collect())?;
    for r in 0..want.nrows() {
        for c in 0..want.ncols() {
            assert_eq!(by_column[(r, c)], want[(r, c)], "by column ({r}, {c})");
            assert_eq!(by_row[(r, c)], want[(r, c)], "by row ({r}, {c})");
        }
    }
    Ok(())
}

#[test]
fn a_row_filter_over_every_column_is_sized_without_counting() -> anyhow::Result<()> {
    let (arr, data) = fixture();
    let dir = tempfile::tempdir()?;
    let out = dir.path().join("out.zarr");
    let out = out.to_str().expect("utf8");
    let (rows, cols) = (vec![0usize, 2, 3], (0..6).collect::<Vec<_>>());
    let (nrow, ncol, nnz) = stream_column_selection(
        data.as_ref(),
        &cols,
        Some(&rows),
        &names("r", &rows),
        &names("c", &cols),
        out,
        &SparseIoBackend::Zarr,
    )?;
    let want = arr.select(ndarray::Axis(0), &rows);
    assert_eq!((nrow, ncol), (3, 6));
    assert_eq!(nnz, want.iter().filter(|&&x| x != 0.0).count());
    assert_holds(out, &want)?;
    let written = open_sparse_matrix(out, &SparseIoBackend::Zarr)?;
    assert_eq!(written.row_names()?, names("r", &rows));
    assert_eq!(written.column_names()?, names("c", &cols));
    Ok(())
}

#[test]
fn a_row_filter_over_some_columns_is_counted() -> anyhow::Result<()> {
    let (arr, data) = fixture();
    let dir = tempfile::tempdir()?;
    let out = dir.path().join("out.zarr");
    let out = out.to_str().expect("utf8");
    // Columns out of order, too: the selection's order is the output's.
    let (rows, cols) = (vec![1usize, 4], vec![5usize, 0, 3]);
    stream_column_selection(
        data.as_ref(),
        &cols,
        Some(&rows),
        &names("r", &rows),
        &names("c", &cols),
        out,
        &SparseIoBackend::Zarr,
    )?;
    let want = arr
        .select(ndarray::Axis(0), &rows)
        .select(ndarray::Axis(1), &cols);
    assert_holds(out, &want)
}

#[test]
fn rows_are_reordered_dropped_and_left_empty_by_a_row_map() -> anyhow::Result<()> {
    let (arr, data) = fixture();
    let dir = tempfile::tempdir()?;
    // Input rows 0..5 go to output rows 3, -, 0, 2, -; output row 1 is fed
    // by none. Over every column, and over a few.
    let to = vec![Some(3), None, Some(0), Some(2), None];
    for cols in [(0..6).collect::<Vec<_>>(), vec![4usize, 1]] {
        let out = dir.path().join(format!("out{}.zarr", cols.len()));
        let out = out.to_str().expect("utf8");
        let rows = RowMap {
            to: to.clone(),
            nrow: 4,
        };
        stream_mapped_selection(
            data.as_ref(),
            &cols,
            Some(&rows),
            &names("new", &[0, 1, 2, 3]),
            &names("c", &cols),
            out,
            &SparseIoBackend::Zarr,
        )?;
        let mut want = ndarray::Array2::<f32>::zeros((4, cols.len()));
        for (old, new) in to.iter().enumerate() {
            if let Some(new) = new {
                for (k, &c) in cols.iter().enumerate() {
                    want[(*new as usize, k)] = arr[(old, c)];
                }
            }
        }
        assert_holds(out, &want)?;
    }
    Ok(())
}

#[test]
fn a_row_map_feeding_one_row_twice_is_refused() -> anyhow::Result<()> {
    let (_, data) = fixture();
    let dir = tempfile::tempdir()?;
    let out = dir.path().join("out.zarr");
    let rows = RowMap {
        to: vec![Some(0), Some(0), None, None, None],
        nrow: 1,
    };
    let err = stream_mapped_selection(
        data.as_ref(),
        &[0, 1],
        Some(&rows),
        &names("new", &[0]),
        &names("c", &[0, 1]),
        out.to_str().expect("utf8"),
        &SparseIoBackend::Zarr,
    )
    .unwrap_err();
    assert!(
        format!("{err:#}").contains("land on output row 0"),
        "{err:#}"
    );
    assert!(RowMap::keeping(&[1, 1], 5).is_err());
    assert!(RowMap::keeping(&[5], 5).is_err());
    Ok(())
}
