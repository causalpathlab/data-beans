//! The multilevel collapse, end to end, against what its outputs are defined
//! to be: at every level the observed counts of a pseudobulk are the summed
//! counts of the cells the returned membership puts in it, and each level
//! nests in the next coarser one.

#![cfg(feature = "alg")]

use data_beans::alg::collapse_data::{
    collapse_columns_multilevel_with_hierarchy, collapse_columns_multilevel_with_partition,
    collapse_stack_multilevel_with_hierarchy, CollapsedOut, MultilevelParams,
};
use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use data_beans::sparse_io_stack::SparseIoStack;
use data_beans::sparse_io_vector::SparseIoVec;
use legume_numeric::param::traits::Inference;
use nalgebra::DMatrix;

const D: usize = 8;
const PER_BATCH: usize = 60;
const N: usize = 2 * PER_BATCH;

/// Counts of column `j`: two cell types alternating by column parity, the
/// second batch (columns `PER_BATCH..`) carrying a per-gene shift. `layer`
/// varies the values so stacked layers differ.
fn column(j: usize, layer: usize) -> Vec<f32> {
    (0..D)
        .map(|g| {
            let base = if j.is_multiple_of(2) {
                10 + 3 * (g % 2)
            } else {
                4 + 2 * ((g + 1) % 2)
            };
            let shift = if j >= PER_BATCH && g < D / 2 { 2 } else { 1 };
            let v = base * shift + (j * (g + 1 + layer)) % 3 + layer * (g % 3);
            // Leave some entries out so the columns are sparse.
            if (j + 2 * g + layer).is_multiple_of(7) {
                0.0
            } else {
                v as f32
            }
        })
        .collect()
}

fn data(dir: &std::path::Path, layer: usize) -> SparseIoVec {
    let path = dir.join(format!("layer{layer}.zarr"));
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for j in 0..N {
        for (g, v) in column(j, layer).into_iter().enumerate() {
            if v > 0.0 {
                triplets.push((g as u64, j as u64, v));
            }
        }
    }
    let mut b = create_sparse_from_triplets(
        &triplets,
        (D, N, triplets.len()),
        Some(path.to_str().expect("utf8")),
        Some(&SparseIoBackend::Zarr),
    )
    .expect("backend");
    b.register_row_names_vec(
        &(0..D)
            .map(|g| format!("g{g}").into())
            .collect::<Vec<Box<str>>>(),
    );
    b.register_column_names_vec(
        &(0..N)
            .map(|c| format!("c{c}").into())
            .collect::<Vec<Box<str>>>(),
    );
    let mut v = SparseIoVec::new();
    v.push(std::sync::Arc::from(b), None).expect("push");
    v
}

fn batches() -> Vec<&'static str> {
    (0..N)
        .map(|j| if j < PER_BATCH { "one" } else { "two" })
        .collect()
}

/// A projection separating the two types on its first axis and three
/// sub-states on its second, with every column at a distinct point.
fn projection() -> DMatrix<f32> {
    DMatrix::from_fn(9, N, |d, j| {
        let jitter = ((j * (2 * d + 3) + d * d) % 13) as f32 * 0.05 - 0.3 + 0.001 * j as f32;
        match d {
            0 => (if j.is_multiple_of(2) { 3.0 } else { -3.0 }) + jitter,
            1 => ((j / 2) % 3) as f32 * 2.0 - 2.0 + jitter,
            _ => jitter,
        }
    })
}

/// `(num_levels, sort_dim)` of the two regimes: one level of a few large
/// groups, whose pb-samples hold several cells each, and two levels of a fine
/// partition (a second level exists only above seven sort dimensions).
const REGIMES: [(usize, usize); 2] = [(1, 3), (2, 8)];

fn params(num_levels: usize, sort_dim: usize) -> MultilevelParams {
    let mut p = MultilevelParams::new(9);
    p.knn_pb_samples = 3;
    p.num_levels = num_levels;
    p.sort_dim = sort_dim;
    p.num_opt_iter = 10;
    p
}

/// `level`'s observed counts are the member cells' summed counts, and its
/// sizes their number.
fn assert_observed_counts(level: &CollapsedOut, cell_to_pb: &[usize], layer: usize, tag: &str) {
    assert_eq!(cell_to_pb.len(), N);
    let (counts, sizes) = level.observed_counts(cell_to_pb).expect("kept stats");
    let n_pb = counts.ncols();
    assert_eq!(n_pb, cell_to_pb.iter().max().expect("cells") + 1, "{tag}");
    let mut want = DMatrix::<f32>::zeros(D, n_pb);
    let mut want_sizes = vec![0f32; n_pb];
    for (j, &pb) in cell_to_pb.iter().enumerate() {
        for (g, v) in column(j, layer).into_iter().enumerate() {
            want[(g, pb)] += v;
        }
        want_sizes[pb] += 1.0;
    }
    assert_eq!(sizes, want_sizes, "{tag}");
    for (got, want) in counts.iter().zip(want.iter()) {
        assert!(
            (got - want).abs() <= 1e-3 * (1.0 + want.abs()),
            "{tag}: observed count {got}, cells sum to {want}"
        );
    }
}

/// Every pseudobulk of `fine` lies within one pseudobulk of `coarse`.
fn assert_nested(fine: &[usize], coarse: &[usize], tag: &str) {
    let mut parent = vec![usize::MAX; fine.iter().max().expect("cells") + 1];
    for (&f, &c) in fine.iter().zip(coarse) {
        if parent[f] == usize::MAX {
            parent[f] = c;
        }
        assert_eq!(
            parent[f], c,
            "{tag}: pseudobulk {f} straddles coarse groups"
        );
    }
}

fn assert_batch_model_is_finite(level: &CollapsedOut, tag: &str) {
    for (name, param) in [
        ("mu_adjusted", level.mu_adjusted.as_ref()),
        ("delta", level.delta.as_ref()),
        ("gamma", level.gamma.as_ref()),
    ] {
        let mean = param
            .unwrap_or_else(|| panic!("{tag}: {name} on two batches"))
            .posterior_mean();
        assert!(
            mean.iter().all(|v| v.is_finite() && *v >= 0.0),
            "{tag}: {name} {mean}"
        );
    }
}

#[test]
fn single_layer_levels_hold_their_cells_counts() {
    for (num_levels, sort_dim) in REGIMES {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut v = data(dir.path(), 0);
        let out = collapse_columns_multilevel_with_hierarchy(
            &mut v,
            &projection(),
            &batches(),
            &params(num_levels, sort_dim),
        )
        .expect("collapse");
        assert_eq!(out.levels.len(), num_levels);
        assert_eq!(out.cell_to_pb_per_level.len(), num_levels);
        for (l, (level, cell_to_pb)) in out.levels.iter().zip(&out.cell_to_pb_per_level).enumerate()
        {
            let tag = format!("sort dim {sort_dim} level {l}");
            assert_observed_counts(level, cell_to_pb, 0, &tag);
            assert_batch_model_is_finite(level, &tag);
        }
        for pair in out.cell_to_pb_per_level.windows(2) {
            assert_nested(&pair[0], &pair[1], "refined");
        }
    }
}

#[test]
fn stack_levels_hold_each_layers_counts() {
    for (num_levels, sort_dim) in REGIMES {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut stack = SparseIoStack::new();
        for layer in 0..2 {
            stack.push(data(dir.path(), layer)).expect("push layer");
        }
        let out = collapse_stack_multilevel_with_hierarchy(
            &mut stack,
            &projection(),
            &batches(),
            &params(num_levels, sort_dim),
        )
        .expect("collapse");
        assert_eq!(out.levels.len(), num_levels);
        for (l, (layers, cell_to_pb)) in
            out.levels.iter().zip(&out.cell_to_pb_per_level).enumerate()
        {
            assert_eq!(layers.len(), 2);
            for (layer, level) in layers.iter().enumerate() {
                let tag = format!("sort dim {sort_dim} level {l} layer {layer}");
                assert_observed_counts(level, cell_to_pb, layer, &tag);
                assert_batch_model_is_finite(level, &tag);
            }
        }
        for pair in out.cell_to_pb_per_level.windows(2) {
            assert_nested(&pair[0], &pair[1], "stack");
        }
    }
}

/// An inherited hierarchy whose finest groups straddle its coarse groups
/// across batches: the collapse still returns nested levels whose statistics
/// agree with the membership it reports.
#[test]
fn inherited_non_nested_hierarchy_collapses_to_nested_levels() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut v = data(dir.path(), 0);
    // Six finest groups spanning both batches; the coarse level splits by
    // batch for the first three and by group for the rest.
    let finest: Vec<usize> = (0..N).map(|j| (j * 7) % 6).collect();
    let coarse: Vec<usize> = (0..N)
        .map(|j| {
            let f = (j * 7) % 6;
            if f < 3 {
                usize::from(j >= PER_BATCH)
            } else {
                f % 2
            }
        })
        .collect();
    let out = collapse_columns_multilevel_with_partition(
        &mut v,
        &projection(),
        &batches(),
        &params(2, 8),
        &[finest.clone(), coarse],
    )
    .expect("collapse");
    assert_eq!(out.levels.len(), 2);
    for (l, (level, cell_to_pb)) in out.levels.iter().zip(&out.cell_to_pb_per_level).enumerate() {
        let tag = format!("inherited level {l}");
        assert_observed_counts(level, cell_to_pb, 0, &tag);
        assert_batch_model_is_finite(level, &tag);
    }
    // The finest partition is the inherited one.
    assert_nested(&finest, &out.cell_to_pb_per_level[0], "inherited finest");
    assert_nested(&out.cell_to_pb_per_level[0], &finest, "inherited finest");
    assert_nested(
        &out.cell_to_pb_per_level[0],
        &out.cell_to_pb_per_level[1],
        "inherited",
    );
}
