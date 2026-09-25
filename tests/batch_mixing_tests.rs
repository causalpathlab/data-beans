//! Batches spread across cell states.
//!
//! Batch correction of a projection must remove what a batch does to every
//! one of its cells (a shift), not the batch's cell-state composition.
//! Subtracting each batch's plain mean removes both: a batch rich in one
//! state has that state pulled to the origin, and cells of one state no
//! longer line up across batches. Pseudobulks built on such a projection
//! mix batches poorly; the tree merge then makes each one hold enough
//! batches.
#![cfg(feature = "alg")]

use data_beans::alg::batch_mixing::*;
use data_beans::alg::random_projection::RandProjOps;
use data_beans::sparse_io::create_sparse_from_triplets;
use data_beans::sparse_io_vector::SparseIoVec;
use nalgebra::{DMatrix, DVector};
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal, Poisson};

/// Largest spread across batches of the per-(state, batch) mean of any
/// coordinate.
fn misalignment(proj: &DMatrix<f32>, batch: &[usize], state: &[usize]) -> f32 {
    let n_batch = batch.iter().max().unwrap() + 1;
    let n_state = state.iter().max().unwrap() + 1;
    let mut worst = 0f32;
    for s in 0..n_state {
        for r in 0..proj.nrows() {
            let means: Vec<f32> = (0..n_batch)
                .filter_map(|b| {
                    let v: Vec<f32> = (0..proj.ncols())
                        .filter(|&j| batch[j] == b && state[j] == s)
                        .map(|j| proj[(r, j)])
                        .collect();
                    (!v.is_empty()).then(|| v.iter().sum::<f32>() / v.len() as f32)
                })
                .collect();
            let m = means.iter().sum::<f32>() / means.len() as f32;
            let sd =
                (means.iter().map(|x| (x - m).powi(2)).sum::<f32>() / means.len() as f32).sqrt();
            worst = worst.max(sd);
        }
    }
    worst
}

/// Two states on the first coordinate, twenty batches of very different
/// compositions, and a batch shift on the second coordinate.
fn two_states(seed: u64) -> (DMatrix<f32>, Vec<usize>, Vec<usize>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let noise = Normal::new(0f32, 0.3).unwrap();
    let (n_batch, per_batch, dim) = (20, 100, 4);
    let n = n_batch * per_batch;
    let mut proj = DMatrix::<f32>::zeros(dim, n);
    let (mut batch, mut state) = (vec![0; n], vec![0; n]);
    for b in 0..n_batch {
        let frac_a = 0.1 + 0.8 * b as f32 / (n_batch - 1) as f32;
        let shift = if b % 2 == 0 { 2.0 } else { -2.0 };
        for c in 0..per_batch {
            let j = b * per_batch + c;
            let s = usize::from(rng.random::<f32>() >= frac_a);
            batch[j] = b;
            state[j] = s;
            for r in 0..dim {
                proj[(r, j)] = noise.sample(&mut rng);
            }
            proj[(0, j)] += if s == 0 { 1.5 } else { -1.5 };
            proj[(1, j)] += shift;
        }
    }
    (proj, batch, state)
}

#[test]
fn centring_within_state_aligns_batches_of_any_composition() {
    let (proj, batch, state) = two_states(7);
    let centred = centre_batches_within_state(&proj, &batch, 2).unwrap();

    let mut plain = proj.clone();
    for b in 0..20 {
        let cols: Vec<usize> = (0..plain.ncols()).filter(|&j| batch[j] == b).collect();
        let mean = cols
            .iter()
            .map(|&j| plain.column(j).into_owned())
            .sum::<DVector<f32>>()
            / cols.len() as f32;
        for &j in &cols {
            let mut col = plain.column_mut(j);
            col -= &mean;
        }
    }

    let within = misalignment(&centred, &batch, &state);
    let naive = misalignment(&plain, &batch, &state);
    assert!(within < 0.15, "within-state misalignment {within}");
    assert!(
        naive > 4.0 * within,
        "plain {naive} vs within-state {within}"
    );
}

/// The batch-corrected projection of count data lines up each cell state
/// across two batches of opposite composition.
#[test]
fn batch_corrected_projection_keeps_composition() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(11);
    let (n_genes, per_batch) = (20usize, 400usize);
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    let (mut batch, mut state) = (Vec::new(), Vec::new());
    for b in 0..2 {
        let frac_a = if b == 0 { 0.9 } else { 0.1 };
        for _ in 0..per_batch {
            let j = batch.len();
            let s = usize::from(rng.random::<f32>() >= frac_a);
            for g in 0..n_genes {
                let marker = (g < n_genes / 2) == (s == 0);
                let mut rate = if marker { 5.0 } else { 0.5 };
                if b == 1 && (10..15).contains(&g) {
                    rate *= 3.0;
                }
                let y = Poisson::new(rate).unwrap().sample(&mut rng) as f32;
                if y > 0.0 {
                    triplets.push((g as u64, j as u64, y));
                }
            }
            batch.push(b);
            state.push(s);
        }
    }
    let n = batch.len();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mix.zarr");
    let mut backend = create_sparse_from_triplets(
        &triplets,
        (n_genes, n, triplets.len()),
        Some(path.to_str().unwrap()),
        Some(&data_beans::sparse_io::SparseIoBackend::Zarr),
    )
    .unwrap();
    backend.register_row_names_vec(
        &(0..n_genes)
            .map(|g| format!("g{g}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    backend.register_column_names_vec(
        &(0..n)
            .map(|c| format!("c{c}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    let mut data = SparseIoVec::new();
    data.push(std::sync::Arc::from(backend), None).unwrap();

    let proj = data
        .project_columns_with_batch_correction(5, None, Some(&batch))
        .unwrap()
        .proj;

    // gap between the two states' means, for scale
    let state_mean = |s: usize| -> DVector<f32> {
        let cols: Vec<usize> = (0..n).filter(|&j| state[j] == s).collect();
        cols.iter()
            .map(|&j| proj.column(j).into_owned())
            .sum::<DVector<f32>>()
            / cols.len() as f32
    };
    let gap = (state_mean(0) - state_mean(1)).norm();
    let within = misalignment(&proj, &batch, &state);
    assert!(
        within < 0.1 * gap,
        "misalignment {within} vs state gap {gap}"
    );
}

/// A bin with too few batches merges with its sibling into the parent;
/// well-mixed bins elsewhere keep their fine codes.
#[test]
fn poorly_mixed_bins_merge_up_the_code_tree() {
    let mut codes = Vec::new();
    let mut batch = Vec::new();
    for b in 0..4 {
        for code in [0b001, 0b010, 0b110] {
            codes.push(code);
            batch.push(b);
        }
    }
    codes.push(0b101);
    batch.push(9);

    let labels = merge_poorly_mixed_bins(&codes, &batch, 3, 8, 3);
    let label_of = |code: usize| labels[codes.iter().position(|&c| c == code).unwrap()];
    assert_eq!(label_of(0b101), label_of(0b001));
    assert_ne!(label_of(0b010), label_of(0b110));
    assert_ne!(label_of(0b010), label_of(0b001));
}

/// Merging stops at the root or after `levels` levels; a bin still poorly
/// mixed stays.
#[test]
fn merging_is_limited_by_levels() {
    let codes = vec![0b111, 0b111, 0b000, 0b000];
    let batch = vec![0, 0, 1, 2];
    let one_level = merge_poorly_mixed_bins(&codes, &batch, 3, 1, 3);
    assert_ne!(one_level[0], one_level[2]);
    let to_root = merge_poorly_mixed_bins(&codes, &batch, 3, 8, 3);
    assert!(to_root.iter().all(|&l| l == to_root[0]));
}

/// The batch-aware partition gives every group at least the required number
/// of batches when the tree allows it.
#[test]
fn mixed_partition_holds_enough_batches() {
    let (proj, batch, _) = two_states(3);
    let centred = centre_batches_within_state(&proj, &batch, 2).unwrap();
    let labels = {
        let codes = data_beans::alg::random_projection::binary_sort_columns(&centred, 4).unwrap();
        merge_poorly_mixed_bins(&codes, &batch, 4, 8, 3)
    };
    let mut per_group: std::collections::HashMap<usize, std::collections::HashSet<usize>> =
        Default::default();
    for (&l, &b) in labels.iter().zip(&batch) {
        per_group.entry(l).or_default().insert(b);
    }
    assert!(per_group.values().all(|bs| bs.len() >= 3));
}
