use super::{build_csr_from_csc_banded, slab_end};
use crate::sparse_io::{create_sparse_streaming_empty, SparseIoBackend};

fn by_col(entries: &[(u64, u64)]) -> Vec<(u64, u64, f32)> {
    let mut t: Vec<(u64, u64, f32)> = entries.iter().map(|&(r, c)| (r, c, 1.0)).collect();
    t.sort_unstable_by_key(|&(r, c, _)| (c, r));
    t
}

/// A slab boundary never falls inside a column, however small the slab.
#[test]
fn a_slab_never_splits_a_column() {
    let t = by_col(&[(0, 0), (1, 0), (2, 0), (0, 1), (5, 1), (3, 2)]);
    assert_eq!(
        slab_end(&t, 0, 1, 3, |t| t.1),
        (3, 1),
        "column 0 has three entries"
    );
    assert_eq!(slab_end(&t, 3, 1, 3, |t| t.1), (5, 2));
}

/// The final slab runs to the axis length, so trailing empties are tiled too.
#[test]
fn the_last_slab_extends_to_the_axis_length() {
    let t = by_col(&[(0, 0), (1, 2)]);
    assert_eq!(
        slab_end(&t, 0, 1, 7, |t| t.1),
        (1, 2),
        "column 1 is empty: next band starts there"
    );
    assert_eq!(
        slab_end(&t, 1, 1, 7, |t| t.1),
        (2, 7),
        "columns 3..7 are empty and belong to the last band"
    );
}

#[test]
fn a_slab_larger_than_the_matrix_is_one_band() {
    let t = by_col(&[(0, 0), (1, 1), (2, 2)]);
    assert_eq!(slab_end(&t, 0, 1 << 20, 3, |t| t.1), (3, 3));
}

/// The same walker serves CSR once the vector is sorted row-major.
#[test]
fn the_walker_follows_the_row_axis_for_csr() {
    let mut t = by_col(&[(2, 0), (2, 1), (0, 3), (4, 2)]);
    t.sort_unstable_by_key(|&(r, c, _)| (r, c));
    assert_eq!(slab_end(&t, 0, 1, 6, |t| t.0), (1, 2), "row 1 is empty");
    assert_eq!(
        slab_end(&t, 1, 1, 6, |t| t.0),
        (3, 4),
        "row 2 has two entries, row 3 is empty"
    );
    assert_eq!(
        slab_end(&t, 3, 1, 6, |t| t.0),
        (4, 6),
        "row 5 is empty and belongs to the last band"
    );
}

/// The transpose in row bands of every size, down to one entry per band,
/// against the single-band result.
#[test]
fn the_transpose_does_not_depend_on_the_band_size() -> anyhow::Result<()> {
    // 37 rows x 23 columns with empty rows, empty columns and one dense row,
    // column-major.
    let (nrow, ncol) = (37u64, 23u64);
    let mut triplets = Vec::new();
    let mut colptr = Vec::new();
    for j in 0..ncol {
        colptr.push(triplets.len() as u64);
        for i in 0..nrow {
            let dense_row = i == 5 && j % 2 == 0;
            let scattered = (i * 7 + j * 13) % 11 == 0 && i % 9 != 4 && j != 17;
            if dense_row || scattered {
                triplets.push((i, j, (i * 100 + j + 1) as f32));
            }
        }
    }
    let rows: Vec<u64> = triplets.iter().map(|t| t.0).collect();
    let vals: Vec<f32> = triplets.iter().map(|t| t.2).collect();
    let shape = (nrow as usize, ncol as usize, triplets.len());
    let mut by_rows = triplets.clone();
    by_rows.sort_by_key(|t| (t.0, t.1));

    let dir = tempfile::tempdir()?;
    // 12 bytes per entry: one entry, one small row, several rows, everything.
    for (k, band_bytes) in [1usize, 12, 60, 300, usize::MAX].into_iter().enumerate() {
        let path = dir.path().join(format!("band{k}.zarr"));
        let mut out = create_sparse_streaming_empty(
            Some(path.to_str().expect("utf8")),
            Some(&SparseIoBackend::Zarr),
        )?;
        out.begin_streaming_csc(shape)?;
        out.append_csc_slab(0, 0, &colptr, &rows, &vals)?;
        out.finalize_streaming_csc()?;
        build_csr_from_csc_banded(&mut *out, band_bytes)?;

        // As stored: each row's columns ascending, rows in request order.
        let (_, _, got) = out.read_triplets_by_rows((0..nrow as usize).collect())?;
        assert_eq!(got, by_rows, "band of {band_bytes} bytes");
        let (_, _, one) = out.read_triplets_by_rows(vec![5])?;
        let dense_row = by_rows.iter().filter(|t| t.0 == 5).count();
        assert!(dense_row >= 12);
        assert_eq!(
            one.len(),
            dense_row,
            "the dense row, band of {band_bytes} bytes"
        );
    }
    Ok(())
}
