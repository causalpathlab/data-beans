use super::*;
use crate::alg::cell_pairs::split_reads_over_pair_units;
use crate::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use std::sync::Arc;

fn data_from(dir: &std::path::Path, tag: &str, x: &DMatrix<f32>) -> SparseIoVec {
    let mut trip = Vec::new();
    for c in 0..x.ncols() {
        for g in 0..x.nrows() {
            if x[(g, c)] != 0.0 {
                trip.push((g as u64, c as u64, x[(g, c)]));
            }
        }
    }
    let path = dir.join(format!("{tag}.zarr"));
    let mut b = create_sparse_from_triplets(
        &trip,
        (x.nrows(), x.ncols(), trip.len()),
        Some(path.to_str().expect("utf8")),
        Some(&SparseIoBackend::Zarr),
    )
    .expect("backend");
    b.register_row_names_vec(
        &(0..x.nrows())
            .map(|g| format!("g{g}").into())
            .collect::<Vec<_>>(),
    );
    b.register_column_names_vec(
        &(0..x.ncols())
            .map(|c| format!("c{c}").into())
            .collect::<Vec<_>>(),
    );
    let mut v = SparseIoVec::new();
    v.push(Arc::from(b), None).expect("push");
    v
}

fn rule(label: Vec<usize>, n_labels: usize, min_spans: usize, min_units: usize) -> SpanRule {
    SpanRule {
        label,
        n_labels,
        min_spans,
        min_units,
    }
}

/// Two tight, far-apart clumps: Ward's root splits them.
#[test]
fn ward_root_splits_two_clumps() {
    let pts = DMatrix::from_row_slice(5, 1, &[0.0, 0.1, 0.2, 10.0, 10.1]);
    let t = CoarseTree::ward(&pts);
    let (a, b) = t.children[t.root].expect("root splits");
    let mut sides = [t.leaves(a), t.leaves(b)];
    sides.iter_mut().for_each(|s| s.sort_unstable());
    sides.sort();
    assert_eq!(sides, [vec![0, 1, 2], vec![3, 4]]);
    assert_eq!(t.parent[t.root], t.root);
}

/// A node is split only if both children span; a root that does not span
/// leaves every group without a node.
#[test]
fn cut_splits_only_where_both_children_span() {
    let pts = DMatrix::from_row_slice(4, 1, &[0.0, 0.1, 10.0, 10.1]);
    let t = CoarseTree::ward(&pts);
    // groups 0,1 hold both batches; groups 2,3 only batch 0
    let both = vec![vec![5, 5], vec![5, 5], vec![5, 0], vec![5, 0]];
    let r = rule(vec![], 2, 2, 3);
    let cut = t.cut(&both, &r);
    assert!(
        cut.iter().all(|n| *n == Some(t.root)),
        "one side fails: no split"
    );

    // each group alone is too thin, each clump of two spans
    let all = vec![vec![2, 2]; 4];
    let cut = t.cut(&all, &r);
    assert_eq!(cut[0], cut[1]);
    assert_eq!(cut[2], cut[3]);
    assert_ne!(cut[0], cut[2]);

    let thin = vec![vec![1, 0]; 4]; // one batch only, even at the root
    assert!(t.cut(&thin, &r).iter().all(Option::is_none));
}

/// A fold planted per (gene, node, batch) on noiseless data is recovered
/// (as the n/(n−1) contrast),
/// and a node seen in one batch keeps δ exactly.
#[test]
fn matched_fold_recovers_a_planted_fold() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (g, per) = (3usize, 4usize);
    let mu = [[10.0f32, 20.0, 30.0], [30.0, 5.0, 25.0], [8.0, 8.0, 8.0]];
    // fold[node][batch][gene]: geometric mean 1 over batches; equal totals per batch
    let f = [
        [[2.0f32, 0.5, 1.0], [0.5, 2.0, 1.0]],
        [[1.5, 1.0, 1.0 / 1.5], [1.0 / 1.5, 1.0, 1.5]],
    ];
    let mut cols = Vec::new();
    let (mut batch, mut node) = (Vec::new(), Vec::new());
    for (k, fk) in f.iter().enumerate() {
        for (b, fkb) in fk.iter().enumerate() {
            for _ in 0..per {
                cols.push((0..g).map(|j| mu[k][j] * fkb[j]).collect::<Vec<_>>());
                batch.push(b);
                node.push(Some(k));
            }
        }
    }
    for _ in 0..per {
        cols.push(mu[2].to_vec());
        batch.push(0);
        node.push(Some(2));
    }
    let n = cols.len();
    let x = DMatrix::from_fn(g, n, |r, c| cols[c][r]);
    let coords = DMatrix::from_fn(1, n, |_, c| node[c].unwrap_or(0) as f32);
    let log_delta = DMatrix::<f32>::from_row_slice(g, 2, &[0.1, -0.1, 0.0, 0.0, 0.2, -0.2]);
    let x_raw = DMatrix::from_fn(g, n, |r, c| x[(r, c)] * log_delta[(r, batch[c])].exp());
    let data_raw = data_from(dir.path(), "planted_raw", &x_raw);
    let params = GroupFoldParams {
        pseudo_count: 0.0,
        min_units: 1,
        ..GroupFoldParams::default()
    };
    let gf = matched_fold(&data_raw, &batch, &coords, &node, 3, &log_delta, &params).expect("fold");
    // a fold is a rate ratio up to the column's scale (the division trait
    // matches it): compare across genes, relative to gene 0
    for (k, fk) in f.iter().enumerate() {
        for (b, fkb) in fk.iter().enumerate() {
            let col = gf.log_fold.column(k * 2 + b);
            for j in 1..g {
                // two batches: the contrast is n/(n−1) = 2 × log f
                let want = (log_delta[(j, b)] + 2.0 * fkb[j].ln())
                    - (log_delta[(0, b)] + 2.0 * fkb[0].ln());
                let got = col[j] - col[0];
                assert!(
                    (got - want).abs() < 1e-4,
                    "node {k} batch {b} gene {j}: {got} vs {want}"
                );
            }
        }
    }
    for b in 0..2 {
        assert_eq!(
            gf.log_fold_at(Some(2), b),
            log_delta.column(b),
            "one-batch node keeps δ"
        );
    }
    assert_eq!(gf.log_fold_at(None, 1), log_delta.column(1));
}

/// Every read lands in exactly one unit; a self-loop holds its own cell's
/// reads only; a pair holds no more than its two endpoints.
#[test]
fn split_conserves_reads_and_keeps_loops_apart() {
    let dir = tempfile::tempdir().expect("tempdir");
    let x = DMatrix::from_row_slice(2, 3, &[5.0, 0.0, 7.0, 3.0, 9.0, 1.0]);
    let pairs = vec![(0, 1), (1, 2), (0, 2)];
    let data = data_from(dir.path(), "split", &x);
    let trip = split_reads_over_pair_units(&data, &pairs, None).expect("split");
    let mut units = DMatrix::<f32>::zeros(2, pairs.len() + 3);
    for &(g, u, v) in &trip {
        units[(g as usize, u as usize)] += v;
    }
    for g in 0..2 {
        assert_eq!(units.row(g).sum(), x.row(g).sum(), "gene {g} conserved");
        for c in 0..3 {
            assert!(units[(g, pairs.len() + c)] <= x[(g, c)]);
        }
        for (k, &(u, v)) in pairs.iter().enumerate() {
            assert!(units[(g, k)] <= x[(g, u)] + x[(g, v)]);
        }
    }
    let mut keys: Vec<(u64, u64)> = trip.iter().map(|&(g, u, _)| (g, u)).collect();
    keys.dedup();
    assert_eq!(keys.len(), trip.len(), "one triplet per (gene, unit)");
}

/// A pair key that does not span climbs both endpoints; one that never
/// spans, not even at the root, keeps δ.
#[test]
fn pair_nodes_climb_until_they_span() {
    // tree: ((0,1),(2,3))
    let pts = DMatrix::from_row_slice(4, 1, &[0.0, 0.1, 10.0, 10.1]);
    let t = CoarseTree::ward(&pts);
    let p01 = t.parent[0];
    // 4 pairs (0,0) in batches 0/1 → spans at (0,0); 2 pairs (0,1) in batch 0
    // and 2 pairs (1,1) in batch 1 → each fails, both climb to (p01,p01) and span
    let at = vec![
        Some((0, 0)),
        Some((0, 0)),
        Some((0, 0)),
        Some((0, 0)),
        Some((0, 1)),
        Some((1, 0)),
        Some((1, 1)),
        Some((1, 1)),
        None,
    ];
    let label = vec![0, 1, 0, 1, 0, 0, 1, 1, 0];
    let r = rule(vec![], 2, 2, 2);
    let (node, keys, _) = settle_pair_nodes(&t, at, &label, &r);
    assert_eq!(node[0], Some(keys[&(0, 0)]));
    assert!(node[..4].iter().all(|n| *n == node[0]));
    let up = Some(keys[&(p01, p01)]);
    assert!(node[4..8].iter().all(|n| *n == up));
    assert_eq!(node[8], None);

    // a lone pair never spans: it climbs to the root and keeps δ
    let (node, keys, _) = settle_pair_nodes(&t, vec![Some((2, 3))], &[0], &r);
    assert_eq!(node, vec![None]);
    assert!(keys.is_empty());
}

/// The nearest-neighbour chain builds the same tree as greedy Ward: the same
/// clusters (leaf sets) at every internal node.
#[test]
fn nn_chain_ward_matches_greedy_ward() {
    let (k, dim) = (40usize, 3usize);
    let mut s = 12345u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 33) as f32 / (1u64 << 31) as f32
    };
    let pts = DMatrix::from_fn(k, dim, |_, _| next());

    // greedy Ward, O(K³), as the reference
    let mut members: Vec<Option<Vec<usize>>> = (0..k).map(|i| Some(vec![i])).collect();
    let mut want: Vec<Vec<usize>> = Vec::new();
    let cost = |a: &[usize], b: &[usize]| {
        let mean = |s: &[usize]| {
            (0..dim)
                .map(|c| s.iter().map(|&i| f64::from(pts[(i, c)])).sum::<f64>() / s.len() as f64)
                .collect::<Vec<_>>()
        };
        let (ma, mb) = (mean(a), mean(b));
        let d2: f64 = ma.iter().zip(&mb).map(|(x, y)| (x - y).powi(2)).sum();
        2.0 * (a.len() * b.len()) as f64 / (a.len() + b.len()) as f64 * d2
    };
    for _ in 0..k - 1 {
        let live: Vec<usize> = (0..k).filter(|&i| members[i].is_some()).collect();
        let (mut bi, mut bj, mut best) = (0, 0, f64::INFINITY);
        for (x, &i) in live.iter().enumerate() {
            for &j in &live[x + 1..] {
                let c = cost(members[i].as_ref().unwrap(), members[j].as_ref().unwrap());
                if c < best {
                    (bi, bj, best) = (i, j, c);
                }
            }
        }
        let mut m = members[bi].take().unwrap();
        m.extend(members[bj].take().unwrap());
        m.sort_unstable();
        want.push(m.clone());
        members[bi] = Some(m);
    }
    want.sort();

    let t = CoarseTree::ward(&pts);
    let mut got: Vec<Vec<usize>> = (k..t.parent.len())
        .map(|nd| {
            let mut l = t.leaves(nd);
            l.sort_unstable();
            l
        })
        .collect();
    got.sort();
    assert_eq!(got, want);
}
