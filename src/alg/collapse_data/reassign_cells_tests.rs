use super::*;
use nalgebra_sparse::CooMatrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Poisson};

/// `n_lin` lineages with private high-mass gene blocks, `per` cells each;
/// batch 1 (if any) carries a per-gene platform factor.
fn lineages(
    n_lin: usize,
    per: usize,
    ngenes: usize,
    batches: usize,
    seed: u64,
) -> (CscMatrix<f32>, Vec<usize>, Vec<usize>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let block = ngenes / (n_lin + 1);
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    let (mut lineage, mut batch) = (Vec::new(), Vec::new());
    let mut c = 0usize;
    for l in 0..n_lin {
        for i in 0..per {
            let b = i % batches;
            for g in 0..ngenes {
                let mut rate = 1.0 + (g % 5) as f64 / 2.0;
                let blk = g / block;
                if blk == l {
                    rate *= 8.0;
                } else if blk < n_lin {
                    rate *= 0.2;
                }
                if b == 1 {
                    rate *= if g % 2 == 0 { 2.0 } else { 0.5 };
                }
                let x: f64 = Poisson::new(rate).unwrap().sample(&mut rng);
                if x > 0.0 {
                    rows.push(g);
                    cols.push(c);
                    vals.push(x as f32);
                }
            }
            lineage.push(l);
            batch.push(b);
            c += 1;
        }
    }
    let coo = CooMatrix::try_from_triplets(ngenes, c, rows, cols, vals).unwrap();
    (CscMatrix::from(&coo), lineage, batch)
}

fn corrupt(labels: &[usize], k: usize, frac: f64, seed: u64) -> Vec<usize> {
    let mut rng = SmallRng::seed_from_u64(seed);
    labels
        .iter()
        .map(|&l| {
            if rng.random::<f64>() < frac {
                (l + 1 + rng.random_range(0..k - 1)) % k
            } else {
                l
            }
        })
        .collect()
}

/// Fraction of cells whose node's majority lineage is their own lineage.
fn purity(node: &[usize], lineage: &[usize]) -> f64 {
    let k = node.iter().max().map_or(0, |m| m + 1);
    let nl = lineage.iter().max().map_or(0, |m| m + 1);
    let mut counts = vec![vec![0usize; nl]; k];
    for (&n, &l) in node.iter().zip(lineage) {
        counts[n][l] += 1;
    }
    let maj: Vec<usize> = counts
        .iter()
        .map(|c| (0..nl).max_by_key(|&l| c[l]).unwrap_or(0))
        .collect();
    node.iter()
        .zip(lineage)
        .filter(|(&n, &l)| maj[n] == l)
        .count() as f64
        / node.len() as f64
}

#[test]
fn corrupted_nodes_are_repaired() {
    let (csc, lineage, batch) = lineages(4, 60, 200, 1, 3);
    let n = csc.ncols();
    let mut node = corrupt(&lineage, 4, 0.3, 5);
    let before = purity(&node, &lineage);
    assert!(before < 0.85, "fixture should start corrupted ({before})");
    let active = vec![true; n];
    let moves = reassign_cells_to_nodes(
        &csc,
        &batch,
        1,
        &active,
        &mut node,
        &ReassignCellsParams::default(),
    );
    let after = purity(&node, &lineage);
    assert!(after >= 0.95, "purity after {after}, moves {moves}");
    assert!(
        moves >= n / 5,
        "expected the corrupted cells to move, got {moves}"
    );
}

#[test]
fn clean_nodes_are_stable() {
    let (csc, lineage, batch) = lineages(4, 60, 200, 1, 7);
    let n = csc.ncols();
    let mut node = lineage.clone();
    let active = vec![true; n];
    let moves = reassign_cells_to_nodes(
        &csc,
        &batch,
        1,
        &active,
        &mut node,
        &ReassignCellsParams::default(),
    );
    assert!(
        moves <= n / 20,
        "clean labels should barely move, got {moves}"
    );
    assert!(purity(&node, &lineage) >= 0.95);
}

#[test]
fn repair_holds_within_batches() {
    let (csc, lineage, batch) = lineages(3, 80, 150, 2, 11);
    let n = csc.ncols();
    let mut node = corrupt(&lineage, 3, 0.3, 13);
    let active = vec![true; n];
    reassign_cells_to_nodes(
        &csc,
        &batch,
        2,
        &active,
        &mut node,
        &ReassignCellsParams::default(),
    );
    for b in 0..2 {
        let idx: Vec<usize> = (0..n).filter(|&i| batch[i] == b).collect();
        let nd: Vec<usize> = idx.iter().map(|&i| node[i]).collect();
        let ln: Vec<usize> = idx.iter().map(|&i| lineage[i]).collect();
        let p = purity(&nd, &ln);
        assert!(p >= 0.95, "batch {b}: purity {p}");
    }
}

#[test]
fn inactive_cells_keep_their_node() {
    let (csc, lineage, batch) = lineages(3, 40, 120, 1, 17);
    let n = csc.ncols();
    let mut node = corrupt(&lineage, 3, 0.5, 19);
    let mut active = vec![true; n];
    active[0] = false;
    active[1] = false;
    let frozen = (node[0], node[1]);
    reassign_cells_to_nodes(
        &csc,
        &batch,
        1,
        &active,
        &mut node,
        &ReassignCellsParams::default(),
    );
    assert_eq!((node[0], node[1]), frozen);
}

fn node_sizes(node: &[usize]) -> Vec<usize> {
    let k = node.iter().max().map_or(0, |m| m + 1);
    let mut n = vec![0usize; k];
    node.iter().for_each(|&x| n[x] += 1);
    n
}

#[test]
fn nodes_of_one_lineage_are_not_merged() {
    // Each lineage starts as three clean nodes of 10, 30 and 50 cells.
    // Nothing separates the nodes of a lineage, so the small ones must not
    // drain into the large one.
    let (csc, lineage, batch) = lineages(2, 90, 200, 1, 23);
    let n = csc.ncols();
    let start: Vec<usize> = (0..n)
        .map(|c| {
            let i = c % 90;
            lineage[c] * 3 + usize::from(i >= 10) + usize::from(i >= 40)
        })
        .collect();
    assert_eq!(node_sizes(&start), vec![10, 30, 50, 10, 30, 50]);
    let mut node = start.clone();
    reassign_cells_to_nodes(
        &csc,
        &batch,
        1,
        &vec![true; n],
        &mut node,
        &ReassignCellsParams::default(),
    );
    assert!(purity(&node, &lineage) >= 0.999);
    let sizes = node_sizes(&node);
    assert_eq!(sizes.len(), 6);
    for (b, (&now, &was)) in sizes.iter().zip(&[10, 30, 50, 10, 30, 50]).enumerate() {
        assert!(2 * now >= was, "node {b}: {was} -> {now} cells ({sizes:?})");
    }
    let stayed = node.iter().zip(&start).filter(|(a, b)| a == b).count();
    assert!(stayed * 10 >= n * 8, "{stayed} of {n} cells stayed");
}

#[test]
fn a_mixed_node_gives_its_cells_to_the_clean_ones() {
    // Two clean nodes and one that is half of each lineage: its cells go
    // to the nodes of their lineages, and the clean nodes keep theirs.
    let (csc, lineage, batch) = lineages(2, 60, 200, 1, 29);
    let n = csc.ncols();
    let mut node: Vec<usize> = (0..n)
        .map(|c| if c % 60 < 40 { lineage[c] } else { 2 })
        .collect();
    assert_eq!(node_sizes(&node), vec![40, 40, 40]);
    reassign_cells_to_nodes(
        &csc,
        &batch,
        1,
        &vec![true; n],
        &mut node,
        &ReassignCellsParams::default(),
    );
    assert!(purity(&node, &lineage) >= 0.98);
    for (c, &nd) in node.iter().enumerate() {
        if c % 60 < 40 {
            assert_eq!(nd, lineage[c], "cell {c} left its clean node");
        }
    }
}
