use super::*;
use crate::sparse_io::{create_sparse_from_triplets, SparseIoBackend};

/// `rows × 4` planted counts; row `r` of the returned backend holds
/// `triplets` entries whose row index is `rows_kept[r]` in the full layout, so
/// the two constructions below hold the SAME numbers in the same row order.
fn backend(rows: usize, rows_kept: &[usize]) -> SparseIoVec {
    let value = |r: usize, c: usize| (1 + (r * 7 + c * 3) % 11) as f32;
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for (new_r, &old_r) in rows_kept.iter().enumerate() {
        for c in 0..4usize {
            triplets.push((new_r as u64, c as u64, value(old_r, c)));
        }
    }
    let shape = (rows, 4usize, triplets.len());
    let mut b = create_sparse_from_triplets(&triplets, shape, None, Some(&SparseIoBackend::Zarr))
        .expect("backend");
    b.register_row_names_vec(
        &(0..rows)
            .map(|r| format!("r{r}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    b.register_column_names_vec(
        &(0..4)
            .map(|c| format!("c{c}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    let mut v = SparseIoVec::new();
    v.push(std::sync::Arc::from(b), None).expect("push");
    v
}

/// Six feature rows: rows 0..3 are the ones the sketch reads (on a multi-track
/// axis, the base track's).
fn first_three_of_six() -> Vec<bool> {
    vec![true, true, true, false, false, false]
}

#[test]
fn the_row_mask_keeps_the_marked_features_backend_rows() {
    // A feature axis narrowed by an earlier pass: unified feature `i` lives on
    // backend row `2 * i`.
    let f2b: Vec<usize> = (0..6).map(|i| i * 2).collect();
    let keep = projection_row_mask(&first_three_of_six(), &f2b, 12).expect("one per feature");
    let kept: Vec<usize> = keep
        .iter()
        .enumerate()
        .filter(|&(_, &k)| k)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(kept, vec![0, 2, 4]);
    assert!(projection_row_mask(&[true; 5], &f2b, 12).is_err());
}

#[test]
fn rates_are_gathered_onto_the_compact_axis() {
    // compact feature i is backend row 2 - i
    let backend = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = gather_to_unified_axis(&backend, 2, &[2, 0]);
    assert_eq!(out, DMatrix::from_row_slice(2, 2, &[5.0, 6.0, 1.0, 2.0]));
}

#[test]
fn weights_follow_the_rows_they_belong_to() {
    let w = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let keep = vec![false, true, true, false, true];
    assert_eq!(subset_kept(&w, &keep), vec![2.0, 3.0, 5.0]);
}

/// The projection that seeds the collapse must see the base track alone, so a
/// row-masked clone of a two-track backend has to sketch exactly like a backend
/// that only ever held those rows.
#[test]
fn the_masked_sketch_equals_a_base_only_backends_sketch() {
    let full = backend(6, &[0, 1, 2, 3, 4, 5]);
    let base_only = backend(3, &[0, 1, 2]);
    let keep = projection_row_mask(&first_three_of_six(), &(0..6).collect::<Vec<_>>(), 6)
        .expect("one per feature");
    let mut view = full.clone_for_collapse();
    view.mask_rows(&keep).expect("mask");
    assert_eq!(view.num_rows(), 3);

    let none: Option<&[Box<str>]> = None;
    let masked = project_backend(&view, 3, None, none, None, 1234).expect("masked sketch");
    let direct = project_backend(&base_only, 3, None, none, None, 1234).expect("base sketch");
    assert_eq!(masked.proj.shape(), direct.proj.shape());
    for (a, b) in masked.proj.iter().zip(direct.proj.iter()) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }

    // The same, weighted: the weights are subset to the kept rows in order.
    let w_full = vec![1.0f32, 0.0, 2.0, 9.0, 9.0, 9.0];
    let w_base = subset_kept(&w_full, &keep);
    assert_eq!(w_base, vec![1.0, 0.0, 2.0]);
    let masked_w = project_backend(&view, 3, None, none, Some(&w_base), 1234).expect("masked");
    let direct_w = project_backend(&base_only, 3, None, none, Some(&w_base), 1234).expect("base");
    for (a, b) in masked_w.proj.iter().zip(direct_w.proj.iter()) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }
    // …and the weights actually bite: dropping a row changes the sketch.
    assert!(masked_w
        .proj
        .iter()
        .zip(masked.proj.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6));
}

/// `n_cells` columns over six rows in two planted states (even / odd columns), with a
/// small per-column wiggle so no two columns are identical.
fn two_state_backend(n_cells: usize) -> SparseIoVec {
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for c in 0..n_cells {
        let wiggle = 1.0 + 0.05 * ((c % 5) as f32 - 2.0);
        for r in 0..6usize {
            let v = if c % 2 == 0 {
                10.0 + 3.0 * (r % 2) as f32
            } else {
                4.0 + 2.0 * ((r + 1) % 2) as f32
            };
            triplets.push((r as u64, c as u64, v * wiggle));
        }
    }
    let shape = (6usize, n_cells, triplets.len());
    let mut b = create_sparse_from_triplets(&triplets, shape, None, Some(&SparseIoBackend::Zarr))
        .expect("backend");
    b.register_row_names_vec(
        &(0..6)
            .map(|r| format!("r{r}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    b.register_column_names_vec(
        &(0..n_cells)
            .map(|c| format!("c{c}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    let mut v = SparseIoVec::new();
    v.push(std::sync::Arc::from(b), None).expect("push");
    v
}

/// The two labelings induce the same equivalence on cells.
fn same_partition(a: &[usize], b: &[usize]) -> bool {
    let mut ab = rustc_hash::FxHashMap::default();
    let mut ba = rustc_hash::FxHashMap::default();
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(&x, &y)| *ab.entry(x).or_insert(y) == y && *ba.entry(y).or_insert(x) == x)
}

/// A given partition is collapsed on as is: its groups ignore the planted states, so a
/// refined hash partition could not land on them by chance.
#[test]
fn a_given_partition_is_the_one_collapsed_on() {
    const N: usize = 60;
    let mut v = two_state_backend(N);
    let labels: Vec<Box<str>> = vec!["b0".into(); N];
    let none: Option<&[Box<str>]> = None;
    let proj = project_backend(&v, 3, None, none, None, 7)
        .expect("proj")
        .proj;
    let finest: Vec<usize> = (0..N).map(|c| (c * 7919) % 5).collect();
    let coarse: Vec<usize> = finest.iter().map(|&g| usize::from(g >= 3)).collect();
    let partition = vec![finest.clone(), coarse.clone()];
    let mut params = MultilevelParams::new(4);
    params.num_levels = partition.len();
    // The collapse spreads its levels between `sort_dim` and a fixed coarsest depth
    // and refuses a level count it cannot spread, so two levels need a deep finest.
    params.sort_dim = 8;
    let out = collapse(&mut v, &proj, &labels, &params, Some(&partition)).expect("collapse");
    assert_eq!(out.cell_to_pb_per_level.len(), 2);
    assert!(same_partition(&finest, &out.cell_to_pb_per_level[0]));
    assert!(same_partition(&coarse, &out.cell_to_pb_per_level[1]));
    assert_eq!(out.levels[0].mu_observed.posterior_mean().ncols(), 5);
    assert_eq!(out.levels[1].mu_observed.posterior_mean().ncols(), 2);
}

#[test]
fn a_given_partition_must_map_every_backend_column() {
    assert!(check_partition(&[vec![0; 4], vec![0; 4]], 4).is_ok());
    assert!(check_partition(&[], 4).is_err());
    assert!(check_partition(&[vec![0; 4], vec![0; 3]], 4).is_err());
}

/// The unified data streams its counts on the compact feature axis: after a
/// subset, rows outside it are skipped and the kept ones are renumbered.
#[test]
fn unified_counts_stream_on_the_compact_axis() {
    use legume_numeric::matrix::cell_count_stream::CellCountStream;
    let mut unified = UnifiedData::from_sparse_io(
        backend(6, &[0, 1, 2, 3, 4, 5]),
        &vec!["b0".into(); 4],
        None,
        false,
        None,
    )
    .expect("unified");
    let mut all = Vec::new();
    unified
        .for_each_count(1..3, &mut |f, c, v| all.push((f, c, v)))
        .expect("stream");
    assert_eq!(all.len(), 12, "6 rows × 2 cells");

    unified.subset_features(&[4, 1]);
    assert_eq!(CellCountStream::n_features(&unified), 2);
    let mut kept = Vec::new();
    unified
        .for_each_count(1..3, &mut |f, c, v| kept.push((f, c, v)))
        .expect("stream");
    let value = |r: usize, c: usize| (1 + (r * 7 + c * 3) % 11) as f32;
    kept.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    assert_eq!(
        kept,
        vec![
            (0, 0, value(4, 1)),
            (0, 1, value(4, 2)),
            (1, 0, value(1, 1)),
            (1, 1, value(1, 2)),
        ]
    );
}
