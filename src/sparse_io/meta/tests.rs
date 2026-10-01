use super::*;
use crate::sparse_io::{SparseIo, SparseIoBackend};

fn meta(pairs: &[(&str, &str)]) -> Metadata {
    pairs
        .iter()
        .map(|&(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[test]
fn a_merge_keeps_only_what_every_part_agrees_on() {
    let a = meta(&[(SAMPLE, "s1"), (PRODUCER, "p1"), (CONTENT, GENE_COUNT)]);
    let b = meta(&[(SAMPLE, "s2"), (PRODUCER, "p1"), (CONTENT, GENE_COUNT)]);
    let c = meta(&[(PRODUCER, "p1"), (CONTENT, GENE_COUNT)]);
    assert_eq!(
        common_metadata(&[a.clone(), b]),
        meta(&[(PRODUCER, "p1"), (CONTENT, GENE_COUNT)])
    );
    assert_eq!(common_metadata(&[a.clone(), a.clone()]), a);
    assert_eq!(
        common_metadata(&[a, c]),
        meta(&[(PRODUCER, "p1"), (CONTENT, GENE_COUNT)])
    );
    assert!(common_metadata(&[]).is_empty());
}

/// A 3 x 4 backend at `path`, its rows and columns named.
fn backend(
    path: &std::path::Path,
    kind: &SparseIoBackend,
) -> Box<dyn SparseIo<IndexIter = Vec<usize>>> {
    use crate::sparse_io::create_sparse_from_triplets_owned;
    let triplets = vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0), (0, 3, 4.0)];
    let mut data = create_sparse_from_triplets_owned(
        triplets,
        (3, 4, 4),
        Some(path.to_str().unwrap()),
        Some(kind),
    )
    .unwrap();
    data.register_row_names_vec(&["GENE1", "GENE2", "GENE3"].map(Box::from));
    data.register_column_names_vec(&["c1", "c2", "c3", "c4"].map(Box::from));
    data
}

/// Set on the backend as it is created, read back once it is reopened.
fn stored_and_read_back(path: &std::path::Path, kind: SparseIoBackend) {
    use crate::sparse_io::open_sparse_matrix;
    let mut data = backend(path, &kind);
    assert!(data.metadata().is_empty());
    data.set_meta(SAMPLE, "s1").unwrap();
    data.set_meta(CONTENT, GENE_COUNT).unwrap();
    data.set_meta(SAMPLE, "s2").unwrap();
    data.set_meta(PRODUCER, "p1").unwrap();
    data.set_metadata(&meta(&[(CONTENT, GENE_COUNT), (SAMPLE, "s2")]))
        .unwrap();
    drop(data);

    let data = open_sparse_matrix(path.to_str().unwrap(), &kind).unwrap();
    assert_eq!(
        data.metadata(),
        meta(&[(CONTENT, GENE_COUNT), (SAMPLE, "s2")])
    );
    assert_eq!(data.meta(SAMPLE).as_deref(), Some("s2"));
    assert_eq!(data.meta(PRODUCER), None);
    assert_eq!(data.num_rows(), Some(3));
}

#[test]
fn a_zarr_backend_stores_its_metadata() {
    let dir = tempfile::tempdir().unwrap();
    stored_and_read_back(&dir.path().join("x.zarr"), SparseIoBackend::Zarr);
}

#[cfg(feature = "hdf5")]
#[test]
fn an_hdf5_backend_stores_its_metadata() {
    let dir = tempfile::tempdir().unwrap();
    stored_and_read_back(&dir.path().join("x.h5"), SparseIoBackend::HDF5);
}

#[test]
fn metadata_survives_rewrites_copies_and_zipping() {
    use crate::sparse_io::open_sparse_matrix;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.zarr");
    let mut data = backend(&path, &SparseIoBackend::Zarr);
    let expected = meta(&[(SAMPLE, "s1"), (PRODUCER, "p1")]);
    data.set_metadata(&expected).unwrap();

    // Rewritten in place.
    data.subset_columns_rows(Some(&vec![0, 2, 3]), None)
        .unwrap();
    assert_eq!(data.metadata(), expected);
    data.reorder_rows(&["GENE3", "GENE1", "GENE2"].map(Box::from))
        .unwrap();
    assert_eq!(data.metadata(), expected);

    // Copied into a new backend.
    let copy = dir.path().join("y.zarr");
    crate::column_subset::stream_column_selection(
        &*data,
        &[0, 1],
        None,
        &data.row_names().unwrap(),
        &["c1", "c3"].map(Box::from),
        copy.to_str().unwrap(),
        &SparseIoBackend::Zarr,
    )
    .unwrap();
    let copied = open_sparse_matrix(copy.to_str().unwrap(), &SparseIoBackend::Zarr).unwrap();
    assert_eq!(copied.metadata(), expected);

    // Cleared.
    data.set_metadata(&Metadata::new()).unwrap();
    assert!(data.metadata().is_empty());
    data.set_metadata(&expected).unwrap();

    // Zipped, and read from the archive.
    drop(data);
    let zipped = dir.path().join("x.zarr.zip");
    crate::zarr_io::finalize_zarr_output(path.to_str().unwrap(), zipped.to_str().unwrap()).unwrap();
    let zipped = open_sparse_matrix(zipped.to_str().unwrap(), &SparseIoBackend::Zarr).unwrap();
    assert_eq!(zipped.metadata(), expected);
}
