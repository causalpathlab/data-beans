//! `convert` writes the same matrix in another container or format, and
//! never over the file it reads.

use super::*;

/// A 3 x 4 backend at `in.zarr` of `dir`, rows g0..g2 and columns c0..c3.
fn fixture(dir: &std::path::Path) -> String {
    let path = dir.join("in.zarr").to_str().unwrap().to_string();
    let triplets = [(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0), (0, 3, 4.0)];
    let mut data = create_sparse_from_triplets(
        &triplets,
        (3, 4, triplets.len()),
        Some(&path),
        Some(&SparseIoBackend::Zarr),
    )
    .unwrap();
    data.register_row_names_vec(&["g0".into(), "g1".into(), "g2".into()]);
    data.register_column_names_vec(&["c0".into(), "c1".into(), "c2".into(), "c3".into()]);
    data.set_meta(meta::PRODUCER, "p1").unwrap();
    path
}

#[test]
fn an_output_that_is_the_input_is_refused_and_the_input_kept() {
    let dir = tempfile::tempdir().unwrap();
    let input = fixture(dir.path());
    // `-o in --no-zip` is in.zarr itself, cleared before it is read.
    let err = run_convert(&ConvertArgs {
        data_file: input.as_str().into(),
        backend: SparseIoBackend::Zarr,
        output: input.strip_suffix(".zarr").unwrap().into(),
        zip: false,
    })
    .unwrap_err();
    assert!(format!("{err:#}").contains("is the input"), "{err:#}");
    let kept = open_sparse_matrix(&input, &SparseIoBackend::Zarr).unwrap();
    assert_eq!(kept.num_non_zeros(), Some(4));
}

/// Into the other format and back: the matrix, its names and its metadata.
#[cfg(feature = "hdf5")]
#[test]
fn the_other_format_holds_the_same_matrix() {
    let dir = tempfile::tempdir().unwrap();
    let input = fixture(dir.path());
    let h5 = dir.path().join("out.h5");
    let h5 = h5.to_str().unwrap();
    let back = dir.path().join("back.zarr");
    let back = back.to_str().unwrap();
    for (from, to, backend) in [
        (input.as_str(), h5, SparseIoBackend::HDF5),
        (h5, back, SparseIoBackend::Zarr),
    ] {
        run_convert(&ConvertArgs {
            data_file: from.into(),
            backend,
            output: to.into(),
            zip: false,
        })
        .unwrap();
    }
    let read = |path: &str| {
        let data = open_sparse_matrix_by_path(path).unwrap();
        let ncol = data.num_columns().unwrap();
        (
            data.read_triplets_by_columns((0..ncol).collect()).unwrap(),
            data.read_rows_dmatrix((0..data.num_rows().unwrap()).collect())
                .unwrap(),
            data.row_names().unwrap(),
            data.column_names().unwrap(),
            data.meta(meta::PRODUCER),
        )
    };
    let want = read(&input);
    assert_eq!(read(h5), want);
    assert_eq!(read(back), want);
}
