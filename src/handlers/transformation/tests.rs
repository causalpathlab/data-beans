//! `squeeze_in_place` drops what falls below its cutoffs and nothing else:
//! a backend with nothing to drop is left as it is, not written again.

use super::squeeze_in_place;
use crate::sparse_io::*;

/// A 4 x 4 backend at `squeeze.zarr` of `dir`, rows g0..g3 and columns
/// c0..c3: row g3 and column c3 are empty, g1 and c1 hold one entry each.
fn fixture(dir: &std::path::Path) -> String {
    let path = dir.join("squeeze.zarr").to_str().unwrap().to_string();
    let triplets = [
        (0, 0, 1.0),
        (2, 0, 2.0),
        (1, 1, 3.0),
        (0, 2, 4.0),
        (2, 2, 5.0),
    ];
    let mut data = create_sparse_from_triplets(
        &triplets,
        (4, 4, triplets.len()),
        Some(&path),
        Some(&SparseIoBackend::Zarr),
    )
    .unwrap();
    let names = |p: &str| -> Vec<Box<str>> { (0..4).map(|i| format!("{p}{i}").into()).collect() };
    data.register_row_names_vec(&names("g"));
    data.register_column_names_vec(&names("c"));
    path
}

/// The backend at `path`: row names, column names and sorted triplets.
type Content = (Vec<Box<str>>, Vec<Box<str>>, Vec<(u64, u64, f32)>);

fn content(path: &str) -> Content {
    let data = open_sparse_matrix(path, &SparseIoBackend::Zarr).unwrap();
    let ncol = data.num_columns().unwrap();
    let (_, _, mut triplets) = data.read_triplets_by_columns((0..ncol).collect()).unwrap();
    triplets.sort_by_key(|&(r, c, _)| (c, r));
    (
        data.row_names().unwrap(),
        data.column_names().unwrap(),
        triplets,
    )
}

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|&s| s.into()).collect()
}

#[test]
fn a_squeeze_that_drops_nothing_leaves_the_backend_as_it_is() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(dir.path());
    let before = content(&path);
    // Something of the store's that a rewrite would not carry over.
    let marker = std::path::Path::new(&path).join("marker");
    std::fs::write(&marker, "here").unwrap();
    squeeze_in_place(&path, 0, 0, None).unwrap();
    assert_eq!(content(&path), before);
    assert!(marker.exists(), "the backend was written again");
}

#[test]
fn a_squeeze_drops_the_rows_and_columns_below_their_cutoffs() {
    // Empty rows and columns only.
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(dir.path());
    squeeze_in_place(&path, 1, 1, None).unwrap();
    assert_eq!(
        content(&path),
        (
            names(&["g0", "g1", "g2"]),
            names(&["c0", "c1", "c2"]),
            vec![
                (0, 0, 1.0),
                (2, 0, 2.0),
                (1, 1, 3.0),
                (0, 2, 4.0),
                (2, 2, 5.0)
            ]
        )
    );

    // One side only: every column stays.
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(dir.path());
    squeeze_in_place(&path, 2, 0, None).unwrap();
    assert_eq!(
        content(&path),
        (
            names(&["g0", "g2"]),
            names(&["c0", "c1", "c2", "c3"]),
            vec![(0, 0, 1.0), (1, 0, 2.0), (0, 2, 4.0), (1, 2, 5.0)]
        )
    );

    // Both sides, by the counts of the backend as it was.
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(dir.path());
    squeeze_in_place(&path, 2, 2, None).unwrap();
    assert_eq!(
        content(&path),
        (
            names(&["g0", "g2"]),
            names(&["c0", "c2"]),
            vec![(0, 0, 1.0), (1, 0, 2.0), (0, 1, 4.0), (1, 1, 5.0)]
        )
    );
}
