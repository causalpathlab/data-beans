use super::{
    build_pb_samples, knn_distinct_pbsamples_in_batch, per_batch_sc_neighbors, MatchablePbSamples,
    PbSampleLayout,
};
use crate::sparse_io_vector::SparseIoVec;
use legume_numeric::matrix::knn_match::ColumnDict;
use nalgebra::DMatrix;
use std::collections::HashSet;

/// Build a 1-D batch where the nearest cells collapse into only 3
/// pb-samples, but 12 more pb-samples sit farther out. The adaptive search
/// must still recover `knn` distinct pb-samples — the old fixed `4·knn+1`
/// (41-cell) single shot would have seen only 3.
#[test]
fn adaptive_recovers_knn_distinct_pbsamples() {
    let knn = 10usize;
    let mut feats: Vec<f32> = Vec::new();
    let mut cell_to_pbsamp: Vec<usize> = Vec::new();
    // pb 0,1,2: 20 cells each at x=0,1,2  → 60 dense near cells / 3 pb-samples
    for pb in 0..3 {
        for _ in 0..20 {
            feats.push(pb as f32);
            cell_to_pbsamp.push(pb);
        }
    }
    // pb 3..=14: 3 cells each at x=3..=14 → 12 sparse far pb-samples
    for pb in 3..15 {
        for _ in 0..3 {
            feats.push(pb as f32);
            cell_to_pbsamp.push(pb);
        }
    }
    let n = feats.len();
    let names: Vec<usize> = (0..n).collect(); // name == global cell index
    let mat = DMatrix::<f32>::from_row_slice(1, n, &feats);
    let bknn = ColumnDict::<usize>::from_dmatrix(mat, names);

    let query = vec![0.0f32];
    let hits = knn_distinct_pbsamples_in_batch(
        &bknn,
        &query,
        knn,
        15,
        &cell_to_pbsamp,
        usize::MAX - 1,
        None,
    )
    .unwrap();

    let distinct: HashSet<usize> = hits.iter().map(|&(p, _)| p).collect();
    assert_eq!(
        distinct.len(),
        knn,
        "should recover exactly knn distinct pb-samples, got {distinct:?}"
    );
    // Growth happened: at least one recovered pb-sample lies past the dense
    // near cluster (pb id ≥ 3), unreachable by a single 41-cell fetch.
    assert!(
        distinct.iter().any(|&p| p >= 3),
        "must reach pb-samples beyond the dense near cluster"
    );
    // Distances are sorted ascending and truncated to knn (closest set).
    assert!(hits.windows(2).all(|w| w[0].1 <= w[1].1));
}

/// Fewer pb-samples than `knn` → return all of them (exhaustion), no hang.
#[test]
fn adaptive_returns_all_when_fewer_than_knn() {
    let knn = 10usize;
    let mut feats: Vec<f32> = Vec::new();
    let mut cell_to_pbsamp: Vec<usize> = Vec::new();
    for pb in 0..3 {
        for _ in 0..5 {
            feats.push(pb as f32);
            cell_to_pbsamp.push(pb);
        }
    }
    let n = feats.len();
    let names: Vec<usize> = (0..n).collect();
    let mat = DMatrix::<f32>::from_row_slice(1, n, &feats);
    let bknn = ColumnDict::<usize>::from_dmatrix(mat, names);
    let query = vec![0.0f32];
    // The answer is the same whether the caller knows the batch holds three
    // pb-samples or only that it holds no more than `knn`.
    for available in [3, knn] {
        let hits = knn_distinct_pbsamples_in_batch(
            &bknn,
            &query,
            knn,
            available,
            &cell_to_pbsamp,
            usize::MAX - 1,
            None,
        )
        .unwrap();
        let ids: Vec<usize> = hits.iter().map(|&(p, _)| p).collect();
        assert_eq!(ids, vec![0, 1, 2]);
        let dists: Vec<f32> = hits.iter().map(|&(_, d)| d).collect();
        assert_eq!(dists, vec![0.0, 1.0, 2.0]);
    }
}

/// One batch on a line: 20 pb-samples of stratum 0 (three cells each) near
/// the origin, then two pb-samples of stratum 1 far out. Returns the cell
/// index, the cell → pb-sample map and the per-pb-sample strata.
fn rare_stratum_batch() -> (ColumnDict<usize>, Vec<usize>, Vec<usize>) {
    let mut feats: Vec<f32> = Vec::new();
    let mut cell_to_pbsamp: Vec<usize> = Vec::new();
    for pb in 0..22 {
        for i in 0..3 {
            feats.push(pb as f32 + 0.1 * i as f32);
            cell_to_pbsamp.push(pb);
        }
    }
    let strata: Vec<usize> = (0..22).map(|pb| usize::from(pb >= 20)).collect();
    let n = feats.len();
    let mat = DMatrix::<f32>::from_row_slice(1, n, &feats);
    let bknn = ColumnDict::<usize>::from_dmatrix(mat, (0..n).collect());
    (bknn, cell_to_pbsamp, strata)
}

/// A stratum with fewer than `knn` pb-samples in the batch: the search
/// returns exactly those, at their nearest cells, and nothing for a stratum
/// the batch does not hold.
#[test]
fn rare_stratum_returns_its_few_pbsamples() {
    let (bknn, cell_to_pbsamp, strata) = rare_stratum_batch();
    let query = vec![0.0f32];
    let search = |available: usize, own_stratum: usize| {
        knn_distinct_pbsamples_in_batch(
            &bknn,
            &query,
            10,
            available,
            &cell_to_pbsamp,
            usize::MAX - 1,
            Some((&strata, own_stratum)),
        )
        .unwrap()
    };
    assert_eq!(search(2, 1), vec![(20, 20.0), (21, 21.0)]);
    // An overcount only costs the early stop.
    assert_eq!(search(10, 1), vec![(20, 20.0), (21, 21.0)]);
    assert!(search(0, 2).is_empty());
}

/// The census behind `available`, and the match table built with it: a
/// pb-sample of the rare stratum matches the other batch's rare pb-samples
/// only, one of a stratum the other batch lacks matches nothing.
#[test]
fn match_table_respects_the_stratum_census() {
    let (bknn_b, mut cell_to_pbsamp, mut strata) = rare_stratum_batch();
    // Batch 0 holds two single-cell pb-samples, appended after batch 1's 22:
    // pb 22 of stratum 1 and pb 23 of stratum 5.
    let n_b = cell_to_pbsamp.len();
    cell_to_pbsamp.extend([22, 23]);
    strata.extend([1, 5]);
    let bknn_a = ColumnDict::<usize>::from_dmatrix(
        DMatrix::<f32>::from_row_slice(1, 2, &[0.0, 0.5]),
        vec![n_b, n_b + 1],
    );
    let num_pb = 24;
    let mut centroids = DMatrix::<f32>::zeros(1, num_pb);
    for pb in 0..22 {
        centroids[(0, pb)] = pb as f32 + 0.1;
    }
    centroids[(0, 23)] = 0.5;
    let mut pb_sample_to_batch = vec![1usize; 22];
    pb_sample_to_batch.extend([0, 0]);
    let layout = PbSampleLayout {
        centroids,
        cell_counts: vec![1.0; num_pb],
        pb_sample_to_batch,
        pb_sample_to_group: (0..num_pb).collect(),
        bg_to_pbsamp: Default::default(),
        cell_to_pbsamp,
        singleton_col: vec![None; num_pb],
        bulk_batches: Vec::new(),
        pb_sample_to_stratum: Some(strata),
    };
    let lookups = [bknn_a, bknn_b];

    let census = MatchablePbSamples::new(&layout, &lookups);
    assert_eq!(census.count(0, None), 2);
    assert_eq!(census.count(1, None), 22);
    assert_eq!(census.count(1, Some(0)), 20);
    assert_eq!(census.count(1, Some(1)), 2);
    assert_eq!(census.count(1, Some(5)), 0);
    assert_eq!(census.count(0, Some(5)), 1);

    let table = per_batch_sc_neighbors(&layout, &lookups, 10, None).unwrap();
    assert_eq!(table[22], vec![(20, 20.0), (21, 21.0)]);
    assert!(table[23].is_empty());
    // A stratum-0 pb-sample of batch 1 finds no stratum-0 partner in batch 0.
    assert!(table[0].is_empty());
    // The rare pb-samples of batch 1 match batch 0's one stratum-1 pb-sample.
    assert_eq!(table[20].iter().map(|m| m.0).collect::<Vec<_>>(), vec![22]);
}

/// Gene sums per pb-sample against a column-by-column accumulation, with
/// column multiplicities and both ordinary and singleton pb-samples.
#[test]
fn gene_sums_are_the_weighted_column_sums_of_each_pbsample() {
    use crate::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
    const D: usize = 7;
    const N: usize = 30;
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("sums.zarr");
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for j in 0..N {
        for g in 0..D {
            let v = (g * 5 + j * 3) % 4;
            if v > 0 {
                triplets.push((g as u64, j as u64, v as f32));
            }
        }
    }
    let mut backend = create_sparse_from_triplets(
        &triplets,
        (D, N, triplets.len()),
        Some(path.to_str().expect("utf8")),
        Some(&SparseIoBackend::Zarr),
    )
    .expect("backend");
    let genes: Vec<Box<str>> = (0..D).map(|g| format!("g{g}").into()).collect();
    let cols: Vec<Box<str>> = (0..N).map(|c| format!("c{c}").into()).collect();
    backend.register_row_names_vec(&genes);
    backend.register_column_names_vec(&cols);
    let mut data = SparseIoVec::new();
    data.push(std::sync::Arc::from(backend), None)
        .expect("push");

    let batches: Vec<usize> = (0..N).map(|c| c % 3).collect();
    data.register_batch_membership(&batches);
    let multiplicity: Vec<f32> = (0..N).map(|c| 1.0 + (c % 2) as f32).collect();
    data.register_column_multiplicity(&multiplicity).unwrap();
    let groups: Vec<String> = (0..N).map(|c| format!("{}", (c * 7) % 4)).collect();
    data.assign_groups(&groups, None);
    let proj = DMatrix::<f32>::from_fn(2, N, |d, c| ((c + d) % 5) as f32);
    let batch_of = data.get_batch_membership(0..N);

    // Batch `batch_of[0]` is a summary batch: its columns are singletons.
    let pbs = build_pb_samples(&data, &proj, D, &[batch_of[0]], &[], None).unwrap();
    let num_pb = pbs.layout.cell_counts.len();
    assert!(pbs.layout.singleton_col.iter().any(Option::is_some));
    assert!(pbs.layout.singleton_col.iter().any(Option::is_none));

    let mut want = vec![vec![0f32; D]; num_pb];
    let mut present = vec![vec![false; D]; num_pb];
    for (c, (&p, &w)) in pbs
        .layout
        .cell_to_pbsamp
        .iter()
        .zip(&multiplicity)
        .enumerate()
    {
        let y = data.read_columns_csc(std::iter::once(c)).unwrap();
        let col = y.col(0);
        for (&g, &v) in col.row_indices().iter().zip(col.values()) {
            want[p][g] += v * w;
            present[p][g] = true;
        }
    }
    assert_eq!(pbs.gene_sums.len(), num_pb);
    for p in 0..num_pb {
        let expected: Vec<(usize, f32)> = (0..D)
            .filter(|&g| present[p][g])
            .map(|g| (g, want[p][g]))
            .collect();
        assert!(!expected.is_empty());
        assert_eq!(pbs.gene_sums[p], expected, "pb-sample {p}");
    }
}
