#![cfg(feature = "ndarray")]

//! `subset_columns_rows` round-trips, written against the CURRENT
//! implementation so they pin behaviour before the streaming rewrite.
//!
//! This trait default method is the highest-leverage write path in the crate —
//! ten call sites across the builders, subset, convert, merging and qc — and it
//! materialises every surviving triplet in RAM before deleting the original
//! backend and rewriting it. The tests fix what the rewrite must preserve:
//! shapes, names, values, and the row/column renumbering.

use data_beans::sparse_io::*;

fn named_fixture() -> Box<dyn SparseIo<IndexIter = Vec<usize>>> {
    // 4 rows x 5 columns, one empty column, distinct values everywhere so a
    // renumbering mistake cannot alias.
    let mut arr = ndarray::Array2::<f32>::zeros((4, 5));
    arr[(0, 0)] = 10.0;
    arr[(2, 0)] = 20.0;
    arr[(1, 1)] = 30.0;
    arr[(3, 1)] = 40.0;
    // column 2 empty
    arr[(0, 3)] = 50.0;
    arr[(2, 3)] = 60.0;
    arr[(3, 4)] = 70.0;
    let mut data = create_sparse_from_ndarray(&arr, None, None).expect("fixture");
    let rows: Vec<Box<str>> = ["r0", "r1", "r2", "r3"].map(Box::from).to_vec();
    let cols: Vec<Box<str>> = ["c0", "c1", "c2", "c3", "c4"].map(Box::from).to_vec();
    data.register_row_names_vec(&rows);
    data.register_column_names_vec(&cols);
    data
}

fn dense(data: &dyn SparseIo<IndexIter = Vec<usize>>) -> nalgebra::DMatrix<f32> {
    let ncol = data.num_columns().expect("ncol");
    data.read_columns_dmatrix((0..ncol).collect())
        .expect("read")
}

#[test]
fn a_column_subset_keeps_values_names_and_order() -> anyhow::Result<()> {
    let mut data = named_fixture();
    data.subset_columns_rows(Some(&vec![0usize, 3, 4]), None)?;

    assert_eq!(data.num_rows(), Some(4));
    assert_eq!(data.num_columns(), Some(3));
    assert_eq!(data.num_non_zeros(), Some(5));
    let col_names = data.column_names()?;
    assert_eq!(
        col_names.iter().map(AsRef::as_ref).collect::<Vec<_>>(),
        vec!["c0", "c3", "c4"]
    );
    let m = dense(&*data);
    assert_eq!(m[(0, 0)], 10.0);
    assert_eq!(m[(2, 0)], 20.0);
    assert_eq!(m[(0, 1)], 50.0);
    assert_eq!(m[(2, 1)], 60.0);
    assert_eq!(m[(3, 2)], 70.0);
    assert_eq!(m.sum(), 10.0 + 20.0 + 50.0 + 60.0 + 70.0, "nothing extra");
    Ok(())
}

#[test]
fn a_row_subset_renumbers_without_aliasing() -> anyhow::Result<()> {
    let mut data = named_fixture();
    data.subset_columns_rows(None, Some(&vec![2usize, 3]))?;

    assert_eq!(data.num_rows(), Some(2));
    assert_eq!(data.num_columns(), Some(5));
    assert_eq!(data.num_non_zeros(), Some(4), "20, 40, 60, 70 survive");
    let row_names = data.row_names()?;
    assert_eq!(
        row_names.iter().map(AsRef::as_ref).collect::<Vec<_>>(),
        vec!["r2", "r3"]
    );
    let m = dense(&*data);
    assert_eq!(m[(0, 0)], 20.0, "old r2 is new row 0");
    assert_eq!(m[(1, 1)], 40.0, "old r3 is new row 1");
    assert_eq!(m[(0, 3)], 60.0);
    assert_eq!(m[(1, 4)], 70.0);
    Ok(())
}

#[test]
fn rows_and_columns_together_compose() -> anyhow::Result<()> {
    let mut data = named_fixture();
    data.subset_columns_rows(Some(&vec![1usize, 3]), Some(&vec![0usize, 3]))?;

    assert_eq!(data.num_rows(), Some(2));
    assert_eq!(data.num_columns(), Some(2));
    // survivors: (r3,c1)=40, (r0,c3)=50
    assert_eq!(data.num_non_zeros(), Some(2));
    let m = dense(&*data);
    assert_eq!(m[(1, 0)], 40.0);
    assert_eq!(m[(0, 1)], 50.0);
    Ok(())
}

#[test]
fn the_csr_side_survives_the_rewrite() -> anyhow::Result<()> {
    // The subset rewrites both orientations; a stale CSR would serve the OLD
    // matrix to every row reader while the CSC serves the new one.
    let mut data = named_fixture();
    data.subset_columns_rows(Some(&vec![0usize, 3]), None)?;
    let (_, _, by_rows) = data.read_triplets_by_rows((0..4).collect())?;
    let mut got: Vec<_> = by_rows.iter().map(|&(i, j, x)| (i, j, x)).collect();
    got.sort_by(|a, b| (a.0, a.1).partial_cmp(&(b.0, b.1)).expect("ord"));
    assert_eq!(
        got,
        vec![(0, 0, 10.0), (0, 1, 50.0), (2, 0, 20.0), (2, 1, 60.0)]
    );
    Ok(())
}

#[test]
fn an_empty_selection_is_refused_not_written() -> anyhow::Result<()> {
    // Deleting the original and writing a 0-column husk would destroy data on
    // what is almost certainly a caller mistake.
    let mut data = named_fixture();
    assert!(data.subset_columns_rows(Some(&vec![]), None).is_err());
    Ok(())
}

#[test]
fn a_permuted_column_selection_defines_the_output_order() -> anyhow::Result<()> {
    // Selection order IS the new order — and a swapped old/new map direction
    // would pass every in-order test while scrambling exactly this one.
    let mut data = named_fixture();
    data.subset_columns_rows(Some(&vec![3usize, 0]), None)?;
    let col_names = data.column_names()?;
    assert_eq!(
        col_names.iter().map(AsRef::as_ref).collect::<Vec<_>>(),
        vec!["c3", "c0"]
    );
    let m = dense(&*data);
    assert_eq!(m[(0, 0)], 50.0, "new column 0 is old c3");
    assert_eq!(m[(0, 1)], 10.0, "new column 1 is old c0");
    Ok(())
}

#[test]
fn a_reordering_row_selection_keeps_columns_sorted() -> anyhow::Result<()> {
    // Old r3 becomes new row 0 and old r0 becomes new row 1: the renumbering
    // inverts within-column order, so the writer's ascending-rows invariant
    // only holds if the subset re-sorts each column. Before the streaming
    // rewrite this case was covered by the global triplet sort; now it is an
    // explicit path, and this is its test.
    let mut data = named_fixture();
    data.subset_columns_rows(None, Some(&vec![3usize, 0]))?;
    let row_names = data.row_names()?;
    assert_eq!(
        row_names.iter().map(AsRef::as_ref).collect::<Vec<_>>(),
        vec!["r3", "r0"]
    );
    let m = dense(&*data);
    assert_eq!(m[(1, 0)], 10.0, "old r0 is new row 1");
    assert_eq!(m[(0, 1)], 40.0, "old r3 is new row 0");
    assert_eq!(m[(1, 3)], 50.0);
    assert_eq!(m[(0, 4)], 70.0);
    assert_eq!(data.num_non_zeros(), Some(4));
    Ok(())
}

/// The regression the block-read rewrite exposed: reads before the swap warm
/// the zarr decoded-chunk caches, and a reopen that kept them served pre-swap
/// chunk contents against post-swap indptrs — row indices past the matrix's
/// own nrow, from a file that was byte-for-byte correct on disk. The exact
/// sequence: subset with a row filter (warms the cache on the source), then a
/// cold read of the swapped-in result.
#[test]
fn a_cold_read_after_the_swap_serves_the_new_contents() -> anyhow::Result<()> {
    let mut data = named_fixture();
    data.subset_columns_rows(None, Some(&vec![2usize, 3]))?;
    let (_, _, triplets) = data.read_triplets_by_columns((0..5).collect())?;
    let nrow = data.num_rows().expect("nrow") as u64;
    assert!(
        triplets.iter().all(|&(r, _, _)| r < nrow),
        "a stale chunk cache leaks pre-swap rows: {triplets:?}"
    );
    assert_eq!(triplets.len(), 4);
    Ok(())
}

/// A duplicated index collapses in the old-to-new map while still counting
/// toward the new shape, so slabs land at wrong offsets and the build fails
/// with an nnz-tiling message that names nothing the caller did. Refuse it up
/// front with the actual cause — this method destroys the original.
#[test]
fn a_duplicated_selection_is_refused_by_name() -> anyhow::Result<()> {
    let mut data = named_fixture();
    let err = data
        .subset_columns_rows(Some(&vec![0usize, 3, 0]), None)
        .expect_err("duplicates must be refused");
    assert!(
        err.to_string().contains("repeats an index"),
        "the error must name the cause, got: {err}"
    );
    let mut data = named_fixture();
    assert!(data
        .subset_columns_rows(None, Some(&vec![1usize, 1]))
        .is_err());
    Ok(())
}

/// An index past the matrix was dropped in silence, shifting every later
/// column of the result against the caller's selection.
#[test]
fn an_out_of_range_selection_is_refused_and_the_data_kept() -> anyhow::Result<()> {
    let mut data = named_fixture();
    let before = dense(&*data);

    let err = data
        .subset_columns_rows(Some(&vec![1usize, 2, 5]), None)
        .expect_err("column 5 of 5 must be refused");
    assert!(err.to_string().contains("column index 5"), "got: {err}");
    let err = data
        .subset_columns_rows(None, Some(&vec![0usize, 4]))
        .expect_err("row 4 of 4 must be refused");
    assert!(err.to_string().contains("row index 4"), "got: {err}");

    assert_eq!(data.num_columns(), Some(5));
    assert_eq!(data.num_rows(), Some(4));
    assert_eq!(dense(&*data), before);
    Ok(())
}

fn names(xs: &[&str]) -> Vec<Box<str>> {
    xs.iter().map(|&x| Box::from(x)).collect()
}

#[test]
fn reordering_rows_moves_values_names_and_both_indexes() -> anyhow::Result<()> {
    let mut data = named_fixture();
    data.set_meta("sample", "s1")?;
    // r1 is dropped, `new` is a row the data never had.
    data.reorder_rows(&names(&["r3", "new", "r0", "r2"]))?;

    assert_eq!(data.num_rows(), Some(4));
    assert_eq!(data.num_columns(), Some(5));
    assert_eq!(data.num_non_zeros(), Some(6), "30 on r1 is gone");
    assert_eq!(data.row_names()?, names(&["r3", "new", "r0", "r2"]));
    assert_eq!(data.column_names()?, names(&["c0", "c1", "c2", "c3", "c4"]));
    assert_eq!(data.meta("sample").as_deref(), Some("s1"));

    let mut expected = nalgebra::DMatrix::<f32>::zeros(4, 5);
    expected[(2, 0)] = 10.0;
    expected[(3, 0)] = 20.0;
    expected[(0, 1)] = 40.0;
    expected[(2, 3)] = 50.0;
    expected[(3, 3)] = 60.0;
    expected[(0, 4)] = 70.0;
    assert_eq!(dense(&*data), expected);
    assert_eq!(data.read_rows_dmatrix((0..4).collect())?, expected);

    // Rows ascend within every column, as readers assume.
    let (_, _, triplets) = data.read_triplets_by_columns((0..5).collect())?;
    assert!(triplets
        .windows(2)
        .all(|w| w[0].1 != w[1].1 || w[0].0 < w[1].0));
    Ok(())
}

#[test]
fn reordering_refuses_a_name_listed_twice() -> anyhow::Result<()> {
    let mut data = named_fixture();
    let before = dense(&*data);
    let err = data
        .reorder_rows(&names(&["r0", "r1", "r0"]))
        .expect_err("r0 twice");
    assert!(
        err.to_string().contains("`r0` is listed twice"),
        "got: {err}"
    );
    assert_eq!(data.row_names()?, names(&["r0", "r1", "r2", "r3"]));
    assert_eq!(dense(&*data), before);
    Ok(())
}

#[test]
fn reordering_refuses_two_rows_of_one_name_and_keeps_the_data() -> anyhow::Result<()> {
    let mut data = named_fixture();
    data.register_row_names_vec(&names(&["r0", "TBCE", "r2", "TBCE"]));
    let before = dense(&*data);

    let err = data
        .reorder_rows(&names(&["TBCE", "r0"]))
        .expect_err("two rows named TBCE");
    assert!(err.to_string().contains("both named `TBCE`"), "got: {err}");
    assert_eq!(data.num_rows(), Some(4));
    assert_eq!(data.num_non_zeros(), Some(7));
    assert_eq!(dense(&*data), before);

    // The repeated name is harmless when the new order leaves it out.
    data.reorder_rows(&names(&["r2", "r0"]))?;
    let m = dense(&*data);
    assert_eq!(m[(0, 0)], 20.0);
    assert_eq!(m[(1, 0)], 10.0);
    assert_eq!(m.sum(), 20.0 + 10.0 + 60.0 + 50.0);
    Ok(())
}

#[test]
fn reordering_onto_names_the_data_lacks_leaves_an_empty_matrix() -> anyhow::Result<()> {
    let mut data = named_fixture();
    data.reorder_rows(&names(&["x", "y"]))?;
    assert_eq!(data.num_rows(), Some(2));
    assert_eq!(data.num_columns(), Some(5));
    assert_eq!(data.num_non_zeros(), Some(0));
    assert_eq!(dense(&*data).sum(), 0.0);
    Ok(())
}

/// The column-selection writer, with and without a row filter, against the
/// dense matrix it selects from.
#[test]
fn a_streamed_selection_equals_the_dense_selection() -> anyhow::Result<()> {
    use data_beans::column_subset::stream_column_selection;
    let data = named_fixture();
    let full = dense(&*data);
    let dir = tempfile::tempdir()?;

    let cases: [(&[usize], Option<&[usize]>); 4] = [
        (&[4, 0, 3], None),
        // Every column kept: the total comes from the row index.
        (&[0, 1, 2, 3, 4], Some(&[0, 2, 3])),
        (&[2, 4, 0, 1, 3], Some(&[1, 3])),
        // Both filtered: the total comes from a counting pass.
        (&[3, 1], Some(&[0, 3])),
    ];
    for (k, (cols, rows)) in cases.into_iter().enumerate() {
        let kept_rows: Vec<usize> = rows.map_or((0..4).collect(), <[usize]>::to_vec);
        let row_names: Vec<Box<str>> = kept_rows.iter().map(|r| format!("r{r}").into()).collect();
        let col_names: Vec<Box<str>> = cols.iter().map(|c| format!("c{c}").into()).collect();
        let path = dir.path().join(format!("sel{k}.zarr"));
        let path = path.to_str().expect("utf8");
        let (nrow, ncol, nnz) = stream_column_selection(
            &*data,
            cols,
            rows,
            &row_names,
            &col_names,
            path,
            &SparseIoBackend::Zarr,
        )?;

        let expected = nalgebra::DMatrix::<f32>::from_fn(kept_rows.len(), cols.len(), |i, j| {
            full[(kept_rows[i], cols[j])]
        });
        assert_eq!((nrow, ncol), expected.shape());
        assert_eq!(nnz, expected.iter().filter(|&&x| x != 0.0).count());

        let out = open_sparse_matrix(path, &SparseIoBackend::Zarr)?;
        assert_eq!(out.num_non_zeros(), Some(nnz));
        assert_eq!(dense(&*out), expected, "case {k}, by columns");
        assert_eq!(
            out.read_rows_dmatrix((0..nrow).collect())?,
            expected,
            "case {k}, by rows"
        );
        assert_eq!(out.row_names()?, row_names);
        assert_eq!(out.column_names()?, col_names);
    }
    Ok(())
}
