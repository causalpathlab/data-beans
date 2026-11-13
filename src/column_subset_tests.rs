//! The pieces of a streamed selection on their own: how the output's size
//! is found, how slabs are cut, and how one slab is laid out.

use super::*;
use crate::sparse_io::*;

type Data = Box<dyn SparseIo<IndexIter = Vec<usize>>>;

/// `(row, column, value)` of a 4 x 5 matrix; column 2 is empty.
const ENTRIES: [(u64, u64, f32); 7] = [
    (0, 0, 10.0),
    (2, 0, 20.0),
    (1, 1, 30.0),
    (3, 1, 40.0),
    (0, 3, 50.0),
    (2, 3, 60.0),
    (3, 4, 70.0),
];

/// [`ENTRIES`] as a backend at `fixture.zarr` of `dir`.
fn fixture(dir: &std::path::Path) -> Data {
    let path = dir.join("fixture.zarr");
    create_sparse_from_triplets(
        &ENTRIES,
        (4, 5, ENTRIES.len()),
        path.to_str(),
        Some(&SparseIoBackend::Zarr),
    )
    .expect("fixture")
}

/// The entries of the backend at `path`, sorted by (column, row).
fn entries_at(path: &str) -> Vec<(u64, u64, f32)> {
    let data = open_sparse_matrix(path, &SparseIoBackend::Zarr).expect("open");
    let ncol = data.num_columns().expect("ncol");
    let (_, _, mut triplets) = data
        .read_triplets_by_columns((0..ncol).collect())
        .expect("read");
    triplets.sort_by_key(|&(r, c, _)| (c, r));
    triplets
}

#[test]
fn the_row_index_counts_the_entries_of_the_rows_kept() {
    let dir = tempfile::tempdir().unwrap();
    let data = fixture(dir.path());
    assert_eq!(data.row_indptr().len(), 5, "the fixture has a by-row index");

    // Rows 0 and 3 hold 10, 50 and 40, 70.
    let rows = RowMap::keeping(&[3, 0], 4).unwrap();
    assert_eq!(nnz_of_kept_rows(data.row_indptr(), &rows.to), Some(4));
    let every_row = RowMap::keeping(&[0, 1, 2, 3], 4).unwrap();
    assert_eq!(nnz_of_kept_rows(data.row_indptr(), &every_row.to), Some(7));

    // An index that does not describe the map's rows says nothing.
    let other_rows = RowMap::keeping(&[0, 1], 3).unwrap();
    assert_eq!(nnz_of_kept_rows(data.row_indptr(), &other_rows.to), None);
}

#[test]
fn a_selection_is_counted_and_written_in_blocks_cut_by_the_budget() {
    let dir = tempfile::tempdir().unwrap();
    let data = fixture(dir.path());
    // Some columns and some rows: sized by the counting pass. A budget of
    // one triplet makes every column a counting block and a slab of its own.
    let cols = [3usize, 1, 0, 4];
    let rows = RowMap::keeping(&[3, 0], 4).unwrap();
    let plan = plan_within(data.as_ref(), &cols, Some(&rows), TRIPLET_BYTES).unwrap();
    assert_eq!(plan.slabs, [(0, 1), (1, 2), (2, 3), (3, 4)]);
    assert_eq!((plan.nrow, plan.nnz), (2, 4));

    let out = dir.path().join("out.zarr");
    let out = out.to_str().unwrap();
    let mut written = create_sparse_streaming_empty(Some(out), Some(&SparseIoBackend::Zarr))
        .expect("empty output");
    let shape =
        write_csc_selection(data.as_ref(), &cols, Some(&rows), &plan, &mut *written).unwrap();
    drop(written);
    assert_eq!(shape, (2, 4, 4));
    // Old row 3 is new row 0, old row 0 new row 1.
    assert_eq!(
        entries_at(out),
        [(1, 0, 50.0), (0, 1, 40.0), (1, 2, 10.0), (0, 3, 70.0)]
    );
}

#[test]
fn a_column_past_the_data_is_refused_before_the_output_exists() {
    let dir = tempfile::tempdir().unwrap();
    let data = fixture(dir.path());
    let out = dir.path().join("out.zarr");
    let names =
        |p: &str, n: usize| -> Vec<Box<str>> { (0..n).map(|i| format!("{p}{i}").into()).collect() };
    let err = stream_mapped_selection(
        data.as_ref(),
        &[1, 5],
        None,
        &names("r", 4),
        &names("c", 2),
        out.to_str().unwrap(),
        &SparseIoBackend::Zarr,
    )
    .unwrap_err();
    assert!(
        format!("{err:#}").contains("column 5 is past the 5 columns"),
        "{err:#}"
    );
    assert!(!out.exists(), "{} was created", out.display());

    // A row map that is not one is refused as early.
    let rows = RowMap {
        to: vec![Some(0), Some(0), None, None],
        nrow: 1,
    };
    let err = stream_mapped_selection(
        data.as_ref(),
        &[1],
        Some(&rows),
        &names("r", 1),
        &names("c", 1),
        out.to_str().unwrap(),
        &SparseIoBackend::Zarr,
    )
    .unwrap_err();
    assert!(
        format!("{err:#}").contains("land on output row 0"),
        "{err:#}"
    );
    assert!(!out.exists(), "{} was created", out.display());
}

#[test]
fn a_selection_names_each_index_once_and_within_range() {
    assert!(distinct_within(&[2, 0, 1], 3, "row").is_ok());
    assert!(distinct_within(&[], 3, "row").is_ok());
    let err = distinct_within(&[0, 3], 3, "column").unwrap_err();
    assert_eq!(err.to_string(), "column 3 is past the 3 columns");
    let err = distinct_within(&[1, 0, 1], 3, "row").unwrap_err();
    assert_eq!(err.to_string(), "the row selection repeats row 1");
    // The row map of a selection refuses the same way.
    let err = RowMap::keeping(&[1, 1], 3).err().expect("row 1 twice");
    assert_eq!(err.to_string(), "the row selection repeats row 1");
}

#[test]
fn rows_are_placed_by_name_and_ambiguous_names_refused() {
    let names = |xs: &[&str]| -> Vec<Box<str>> { xs.iter().map(|&x| x.into()).collect() };
    let old = names(&["a", "b", "c", "b"]);

    let map = RowMap::by_names(&old, &names(&["c", "x", "a"]), 4).unwrap();
    assert_eq!(map.to, [Some(2), None, Some(0), None]);
    assert_eq!(map.nrow, 3);

    // A row past the names is dropped; a name past the rows names no row,
    // so the second `b` is no clash.
    let map = RowMap::by_names(&old, &names(&["c", "a"]), 6).unwrap();
    assert_eq!(map.to, [Some(1), None, Some(0), None, None, None]);
    let map = RowMap::by_names(&old, &names(&["b", "c"]), 2).unwrap();
    assert_eq!(map.to, [None, Some(0)]);
    assert_eq!(map.nrow, 2);

    let err = RowMap::by_names(&old, &names(&["a", "c", "a"]), 4)
        .err()
        .expect("a listed twice");
    assert!(err.to_string().contains("lists `a` twice"), "{err}");
    let err = RowMap::by_names(&old, &names(&["b", "a"]), 4)
        .err()
        .expect("two rows named b");
    assert!(
        err.to_string().contains("rows 1 and 3 are both named `b`"),
        "{err}"
    );
}

#[test]
fn a_slab_is_laid_out_by_column_with_rows_ascending() {
    // Columns out of order, and column 0's rows descending.
    let triplets = vec![(2, 1, 5.0), (3, 0, 1.0), (0, 0, 2.0)];
    let (colptr, rows, values) = csc_slab(triplets, 3, None);
    assert_eq!(colptr, [0, 2, 3]);
    assert_eq!(rows, [0, 3, 2]);
    assert_eq!(values, [2.0, 1.0, 5.0]);

    // Out of order by column and by row, one cell twice, one column empty:
    // the slab is the triplets in a stable sort by (column, row).
    let triplets: Vec<(u64, u64, f32)> = vec![
        (2, 3, 1.0),
        (0, 0, 2.0),
        (4, 3, 3.0),
        (1, 3, 4.0),
        (3, 0, 5.0),
        (1, 3, 6.0),
        (0, 1, 7.0),
    ];
    let mut sorted = triplets.clone();
    sorted.sort_by_key(|&(row, col, _)| (col, row));
    let (colptr, rows, values) = csc_slab(triplets, 4, None);
    assert_eq!(colptr, [0, 2, 3, 3]);
    assert_eq!(rows, sorted.iter().map(|t| t.0).collect::<Vec<_>>());
    assert_eq!(values, sorted.iter().map(|t| t.2).collect::<Vec<_>>());

    // A row map drops, renumbers and so reorders; entries that land on one
    // row keep the order they came in.
    let to = [Some(1), None, Some(0), Some(0)];
    let triplets = vec![(0, 0, 1.0), (1, 0, 2.0), (2, 0, 3.0), (3, 0, 4.0)];
    let (colptr, rows, values) = csc_slab(triplets, 1, Some(&to));
    assert_eq!(colptr, [0]);
    assert_eq!(rows, [0, 0, 1]);
    assert_eq!(values, [3.0, 4.0, 1.0]);
}
