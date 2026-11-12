//! Read-time column scales and the depth factors they come from.

use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use data_beans::sparse_io_vector::scales::RowFilter;
use data_beans::sparse_io_vector::{ColumnAlignment, SparseIoVec};
use std::sync::Arc;

type Triplets = Vec<(u64, u64, f32)>;

/// A backend `{tag}.zarr` in `dir` with rows `rows`, columns `{tag}_0..`,
/// holding `triplets`; `preload` puts its column arrays in memory, so
/// reads take the fast path rather than the cold one.
fn backend(
    dir: &tempfile::TempDir,
    tag: &str,
    rows: &[&str],
    n_cols: usize,
    triplets: &[(u64, u64, f32)],
    preload: bool,
) -> Arc<dyn data_beans::sparse_io::SparseIo<IndexIter = Vec<usize>>> {
    let path = dir.path().join(format!("{tag}.zarr"));
    let mut b = create_sparse_from_triplets(
        triplets,
        (rows.len(), n_cols, triplets.len()),
        Some(path.to_str().unwrap()),
        Some(&SparseIoBackend::Zarr),
    )
    .unwrap();
    let rows: Vec<Box<str>> = rows.iter().map(|&s| s.into()).collect();
    let cols: Vec<Box<str>> = (0..n_cols).map(|j| format!("{tag}_{j}").into()).collect();
    b.register_row_names_vec(&rows);
    b.register_column_names_vec(&cols);
    if preload {
        b.preload_columns().unwrap();
    }
    Arc::from(b)
}

/// Dense `n_rows × n_cols` from triplets, column-major.
fn dense(n_rows: usize, n_cols: usize, triplets: &[(u64, u64, f32)]) -> Vec<Vec<f32>> {
    let mut x = vec![vec![0.0f32; n_rows]; n_cols];
    for &(r, c, v) in triplets {
        x[c as usize][r as usize] += v;
    }
    x
}

fn csc_dense(m: &nalgebra_sparse::CscMatrix<f32>) -> Vec<Vec<f32>> {
    let mut x = vec![vec![0.0f32; m.nrows()]; m.ncols()];
    for (r, c, &v) in m.triplet_iter() {
        x[c][r] += v;
    }
    x
}

fn dmatrix_dense(m: &nalgebra::DMatrix<f32>) -> Vec<Vec<f32>> {
    (0..m.ncols())
        .map(|c| (0..m.nrows()).map(|r| m[(r, c)]).collect())
        .collect()
}

/// Two backends of 4 rows: 3 cells in `a`, 4 in `b`, every count distinct.
fn two_backends(dir: &tempfile::TempDir, preload: bool) -> (SparseIoVec, Vec<Vec<f32>>) {
    let rows = ["G0", "G1", "G2", "G3"];
    let mut ta: Triplets = Vec::new();
    let mut tb: Triplets = Vec::new();
    for c in 0..3u64 {
        for r in 0..4u64 {
            if (r + c) % 3 != 2 {
                ta.push((r, c, (1 + r + 4 * c) as f32));
            }
        }
    }
    for c in 0..4u64 {
        for r in 0..4u64 {
            if (r + 2 * c) % 3 != 1 {
                tb.push((r, c, (50 + r + 4 * c) as f32));
            }
        }
    }
    let mut data = SparseIoVec::new();
    data.push(backend(dir, "a", &rows, 3, &ta, preload), None)
        .unwrap();
    data.push(backend(dir, "b", &rows, 4, &tb, preload), None)
        .unwrap();
    let mut x = dense(4, 3, &ta);
    x.extend(dense(4, 4, &tb));
    (data, x)
}

const FACTOR: [f32; 7] = [0.5, 2.0, 4.0, 1.0, 0.25, 8.0, 3.0];

/// `raw` columns `cells`, each divided by `FACTOR` of its global column.
fn expected(raw: &[Vec<f32>], cells: &[usize]) -> Vec<Vec<f32>> {
    cells
        .iter()
        .map(|&g| raw[g].iter().map(|v| v / FACTOR[g]).collect())
        .collect()
}

// A subset out of order and across both backends, so a read that takes a
// column's factor from its output position rather than its global index
// gets it wrong.
const CELLS: [usize; 4] = [5, 0, 3, 1];

#[test]
fn no_scales_reads_the_counts_as_stored() {
    let dir = tempfile::tempdir().unwrap();
    let (data, raw) = two_backends(&dir, true);
    assert!(data.column_scales().is_none());
    let x = csc_dense(&data.read_columns_csc(0..7).unwrap());
    assert_eq!(x, raw);
}

#[test]
fn clearing_the_scales_restores_the_stored_counts_bit_for_bit() {
    let dir = tempfile::tempdir().unwrap();
    let (mut data, _) = two_backends(&dir, true);
    let before = data.read_columns_csc(0..7).unwrap();
    data.set_column_scales(Some(FACTOR.to_vec())).unwrap();
    data.set_column_scales(None).unwrap();
    let after = data.read_columns_csc(0..7).unwrap();
    assert_eq!(before.values(), after.values());
    assert_eq!(before.row_indices(), after.row_indices());
}

#[test]
fn every_column_read_divides_each_column_by_its_factor() {
    for preload in [true, false] {
        let dir = tempfile::tempdir().unwrap();
        let (mut data, raw) = two_backends(&dir, preload);
        data.set_column_scales(Some(FACTOR.to_vec())).unwrap();
        let want = expected(&raw, &CELLS);
        let csc = csc_dense(&data.read_columns_csc(CELLS.into_iter()).unwrap());
        assert_eq!(csc, want, "csc, preload {preload}");
        let dm = dmatrix_dense(&data.read_columns_dmatrix(CELLS.into_iter()).unwrap());
        assert_eq!(dm, want, "dmatrix, preload {preload}");
        let mut streamed = vec![vec![0.0f32; 4]; CELLS.len()];
        data.for_each_triplet(CELLS.into_iter(), 2, |r, c, v| {
            streamed[c as usize][r as usize] += v
        })
        .unwrap();
        assert_eq!(streamed, want, "for_each_triplet, preload {preload}");
    }
}

#[test]
fn row_reads_divide_each_column_by_its_factor() {
    let dir = tempfile::tempdir().unwrap();
    let (mut data, raw) = two_backends(&dir, true);
    data.set_column_scales(Some(FACTOR.to_vec())).unwrap();
    let rows = data.read_rows_dmatrix([2usize, 0].into_iter()).unwrap();
    for (i, &r) in [2usize, 0].iter().enumerate() {
        for g in 0..7 {
            assert_eq!(rows[(i, g)], raw[g][r] / FACTOR[g], "row {r}, column {g}");
        }
    }
}

#[test]
fn masking_columns_keeps_each_factor_with_its_column() {
    let dir = tempfile::tempdir().unwrap();
    let (mut data, raw) = two_backends(&dir, true);
    data.set_column_scales(Some(FACTOR.to_vec())).unwrap();
    let keep = [true, false, true, false, true, true, true];
    data.mask_columns(&keep).unwrap();
    let kept: Vec<usize> = (0..7).filter(|&g| keep[g]).collect();
    let want = expected(&raw, &kept);
    let x = csc_dense(&data.read_columns_csc(0..kept.len()).unwrap());
    assert_eq!(x, want);
    let kept_factors: Vec<f32> = kept.iter().map(|&g| FACTOR[g]).collect();
    assert_eq!(data.column_scales().unwrap(), kept_factors.as_slice());
    let rows = data.read_rows_dmatrix([1usize].into_iter()).unwrap();
    for (k, &g) in kept.iter().enumerate() {
        assert_eq!(rows[(0, k)], raw[g][1] / FACTOR[g]);
    }
}

#[test]
fn a_cell_two_backends_share_is_divided_by_its_one_factor() {
    let dir = tempfile::tempdir().unwrap();
    // `x` holds cells c0, c1; `y` holds c1, c2 on other rows: c1 is shared.
    let x = backend(
        &dir,
        "x",
        &["P0", "P1"],
        2,
        &[(0, 0, 3.0), (1, 1, 6.0)],
        true,
    );
    let y = backend(&dir, "y", &["Q0"], 2, &[(0, 0, 9.0), (0, 1, 5.0)], true);
    // Name the columns alike so `Union` glues c1.
    let mut data = SparseIoVec::new()
        .with_column_alignment(ColumnAlignment::Union)
        .unwrap()
        .with_column_canonicalizer(|s: &str| -> Box<str> {
            match s {
                "x_0" => "c0".into(),
                "x_1" | "y_0" => "c1".into(),
                _ => "c2".into(),
            }
        })
        .unwrap();
    data.push(x, None).unwrap();
    data.push(y, None).unwrap();
    assert_eq!(data.num_columns(), 3);
    data.set_column_scales(Some(vec![1.0, 3.0, 5.0])).unwrap();
    let m = csc_dense(&data.read_columns_csc(0..3).unwrap());
    let names = data.row_names().unwrap();
    let row = |n: &str| names.iter().position(|r| &**r == n).unwrap();
    assert_eq!(m[1][row("P1")], 2.0);
    assert_eq!(m[1][row("Q0")], 3.0);
    assert_eq!(m[2][row("Q0")], 1.0);
}

#[test]
fn scales_must_match_the_columns_and_be_positive_and_finite() {
    let dir = tempfile::tempdir().unwrap();
    let (mut data, _) = two_backends(&dir, true);
    assert!(data.set_column_scales(Some(vec![1.0; 6])).is_err());
    let mut bad = FACTOR.to_vec();
    for v in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        bad[3] = v;
        assert!(data.set_column_scales(Some(bad.clone())).is_err(), "{v}");
    }
    assert!(data.column_scales().is_none());
}

#[test]
fn a_backend_pushed_after_the_scales_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let (mut data, _) = two_backends(&dir, true);
    data.set_column_scales(Some(FACTOR.to_vec())).unwrap();
    let c = backend(
        &dir,
        "c",
        &["G0", "G1", "G2", "G3"],
        1,
        &[(0, 0, 1.0)],
        true,
    );
    assert!(data.push(c, None).is_err());
}

/////////////////////////////
// depth factors from data //
/////////////////////////////

/// Rows: 0 broad (1 count per cell in `a`, 20·d in `b`); 1 only in `a` and
/// 2 only in `b` (3·d each); 3.. on a sparse pattern at 2·d. Cell depth
/// `d` cycles through 1, 2, 4. With 6 rows each cell holds one pattern row,
/// so off the broad row a cell's count is 5·d,
/// proportional to its `d`.
const PER_BATCH: usize = 6;
const N_ROWS: usize = 6;

fn tissue(dir: &tempfile::TempDir) -> (SparseIoVec, Vec<f32>) {
    let (per_batch, n_rows) = (PER_BATCH, N_ROWS);
    let rows: Vec<String> = (0..n_rows).map(|g| format!("GENE{g}")).collect();
    let rows: Vec<&str> = rows.iter().map(String::as_str).collect();
    let mut depth = Vec::new();
    let mut data = SparseIoVec::new();
    for (tag, broad, own) in [("a", 1.0f32, 1u64), ("b", 20.0, 2)] {
        let mut t: Triplets = Vec::new();
        for c in 0..per_batch as u64 {
            let d = [1.0f32, 2.0, 4.0][(c % 3) as usize];
            t.push((0, c, if tag == "a" { broad } else { broad * d }));
            t.push((own, c, 3.0 * d));
            for g in 3..n_rows as u64 {
                if (c + g) % 3 == 0 {
                    t.push((g, c, 2.0 * d));
                }
            }
            depth.push(d);
        }
        data.push(backend(dir, tag, &rows, per_batch, &t, true), None)
            .unwrap();
    }
    (data, depth)
}

/// Rows detected in more than `share` of the cells of both batches.
fn broad(share: f64) -> RowFilter {
    RowFilter::Broad {
        share,
        groups: batches(PER_BATCH),
    }
}

/// A mask leaving out row `row` of `n`.
fn mask_of(n: usize, row: usize) -> RowFilter {
    RowFilter::Mask((0..n).map(|r| r == row).collect())
}

fn batches(per_batch: usize) -> Vec<Box<str>> {
    (0..2 * per_batch)
        .map(|i| {
            if i < per_batch {
                "A".into()
            } else {
                "B".into()
            }
        })
        .collect()
}

#[test]
fn a_row_every_group_detects_in_most_cells_is_broad() {
    let dir = tempfile::tempdir().unwrap();
    let (data, _) = tissue(&dir);
    let s = data.depth_scales(&[broad(0.5)], None).unwrap();
    assert!(s.ignored_rows[0], "detected in every cell of both groups");
    assert!(!s.ignored_rows[1], "only in group A");
    assert!(!s.ignored_rows[2], "only in group B");
    assert!(s.ignored_rows[3..].iter().all(|&b| !b), "a third of cells");
}

#[test]
fn a_broad_row_does_not_move_the_depth_factors() {
    let dir = tempfile::tempdir().unwrap();
    let (data, depth) = tissue(&dir);
    let s = data.depth_scales(&[broad(0.5)], None).unwrap();
    // Off the broad row every cell's count is proportional to its depth,
    // so the factors are the depths over their median (2), in both groups.
    let want: Vec<f32> = depth.iter().map(|d| d / 2.0).collect();
    for (k, (&got, &w)) in s.factor.iter().zip(&want).enumerate() {
        assert!((got - w).abs() < 1e-6, "cell {k}: {got} vs {w}");
    }
}

#[test]
fn with_no_filter_every_row_counts() {
    let dir = tempfile::tempdir().unwrap();
    let (data, _) = tissue(&dir);
    let s = data.depth_scales(&[], Some(4)).unwrap();
    assert!(s.ignored_rows.iter().all(|&b| !b));
    // The broad row's counts differ between the groups, so the cells of
    // equal depth in A and B now get different factors.
    assert!((s.factor[0] - s.factor[6]).abs() > 0.1);
}

#[test]
fn rows_named_by_a_pattern_or_a_mask_are_left_out() {
    let dir = tempfile::tempdir().unwrap();
    let (data, depth) = tissue(&dir);
    let by_name = data
        .depth_scales(
            &[RowFilter::Names(regex::Regex::new("^GENE0$").unwrap())],
            None,
        )
        .unwrap();
    let by_mask = data.depth_scales(&[mask_of(N_ROWS, 0)], None).unwrap();
    for s in [&by_name, &by_mask] {
        assert_eq!(s.ignored_rows, [true, false, false, false, false, false]);
        for (&got, d) in s.factor.iter().zip(&depth) {
            assert!((got - d / 2.0).abs() < 1e-6);
        }
    }
}

#[test]
fn filters_combine_by_union() {
    let dir = tempfile::tempdir().unwrap();
    let (data, _) = tissue(&dir);
    let mask = mask_of(N_ROWS, 1);
    let names = RowFilter::Names(regex::Regex::new("^GENE2$").unwrap());
    let broad = broad(0.5);
    // Each filter comes last in one order, so one that overwrote the rows
    // the others left out, rather than adding to them, would lose them.
    for order in [
        [names.clone(), broad.clone(), mask.clone()],
        [mask, broad, names],
    ] {
        let s = data.depth_scales(&order, None).unwrap();
        assert_eq!(&s.ignored_rows[..4], &[true, true, true, false]);
    }
}

#[test]
fn depth_factors_are_refused_once_scales_are_set() {
    let dir = tempfile::tempdir().unwrap();
    let (mut data, _) = tissue(&dir);
    let first = data.depth_scales(&[], None).unwrap();
    data.set_column_scales(Some(first.factor)).unwrap();
    assert!(data.depth_scales(&[], None).is_err());
}

#[test]
fn a_cell_of_ignored_rows_alone_is_read_as_stored() {
    let dir = tempfile::tempdir().unwrap();
    let rows = ["GENE0", "GENE1"];
    let t = [(0, 0, 5.0), (1, 1, 2.0), (1, 2, 4.0), (1, 3, 8.0)];
    let mut data = SparseIoVec::new();
    data.push(backend(&dir, "a", &rows, 4, &t, true), None)
        .unwrap();
    let s = data.depth_scales(&[mask_of(2, 0)], None).unwrap();
    assert_eq!(s.factor, [1.0, 0.5, 1.0, 2.0]);
}

#[test]
fn a_filter_of_the_wrong_length_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let (data, _) = tissue(&dir);
    assert!(data
        .depth_scales(&[RowFilter::Mask(vec![false; 5])], None)
        .is_err());
    assert!(data
        .depth_scales(
            &[RowFilter::Broad {
                share: 0.5,
                groups: batches(5)
            }],
            None
        )
        .is_err());
}

#[test]
fn rows_a_canonicalizer_merges_are_summed_then_divided() {
    let dir = tempfile::tempdir().unwrap();
    // `G0.1` and `G0.2` are one row once the suffix is cut.
    let b = backend(
        &dir,
        "a",
        &["G0.1", "G0.2", "G1"],
        2,
        &[(0, 0, 2.0), (1, 0, 4.0), (2, 0, 3.0), (1, 1, 8.0)],
        true,
    );
    let mut data = SparseIoVec::new()
        .with_row_canonicalizer(|s: &str| -> Box<str> { s.split('.').next().unwrap().into() })
        .unwrap();
    data.push(b, None).unwrap();
    data.set_column_scales(Some(vec![2.0, 4.0])).unwrap();
    let names = data.row_names().unwrap();
    let g0 = names.iter().position(|r| &**r == "G0").unwrap();
    let mut streamed = vec![vec![0.0f32; data.num_rows()]; 2];
    data.for_each_triplet([0usize, 1].into_iter(), 1, |r, c, v| {
        streamed[c as usize][r as usize] += v
    })
    .unwrap();
    assert_eq!(streamed[0][g0], 3.0);
    assert_eq!(streamed[1][g0], 2.0);
    let csc = csc_dense(&data.read_columns_csc(0..2).unwrap());
    assert_eq!(csc, streamed);
}

#[test]
fn an_even_count_of_depths_takes_the_mean_of_the_middle_two() {
    let dir = tempfile::tempdir().unwrap();
    let t = [(0, 0, 2.0), (0, 1, 4.0), (0, 2, 6.0), (0, 3, 8.0)];
    let mut data = SparseIoVec::new();
    data.push(backend(&dir, "a", &["GENE0"], 4, &t, true), None)
        .unwrap();
    let s = data.depth_scales(&[], None).unwrap();
    assert_eq!(s.factor, [0.4, 0.8, 1.2, 1.6]);
}

#[test]
fn a_broad_share_outside_zero_to_one_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let (data, _) = tissue(&dir);
    for share in [-0.1, 1.0, f64::NAN] {
        assert!(data.depth_scales(&[broad(share)], None).is_err(), "{share}");
    }
}
