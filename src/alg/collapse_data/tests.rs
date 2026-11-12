use super::*;
use crate::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use crate::sparse_io_vector::RowAlignment;

/// An inherited hierarchy that is not nested: finest group 1 has its batch-A
/// pb-sample under coarse group 0 and its batch-B pb-sample under coarse
/// group 1. The synthesized levels must nest anyway — one coarse label per
/// finest group — or `fine_to_coarse_from_refined` reads a label that only
/// some of the group's pb-samples carry.
#[test]
fn inherited_levels_nest_when_the_source_hierarchy_did_not() {
    // pb-samples: (A, fine 0), (A, fine 1), (B, fine 1), (B, fine 2)
    let pb_sample_to_cells = vec![vec![0, 1], vec![2, 3, 4], vec![5, 6], vec![7, 8]];
    let finest = vec![0, 0, 1, 1, 1, 1, 1, 2, 2];
    let coarse = vec![0, 0, 0, 0, 0, 1, 1, 1, 1];

    let (levels, ks) = nested_levels_by_vote(&pb_sample_to_cells, &[finest, coarse]);
    assert_eq!(ks, vec![3, 2]);
    assert_eq!(levels[0], vec![0, 1, 1, 2]);
    // Fine group 1 holds three cells voting coarse 0 and two voting coarse 1.
    assert_eq!(levels[1], vec![0, 0, 0, 1]);

    let fine_to_coarse = fine_to_coarse_from_refined(&levels[0], &levels[1], ks[0]);
    assert_eq!(fine_to_coarse, vec![0, 0, 1]);
}

/// A nested inherited hierarchy comes back as it was given.
#[test]
fn inherited_levels_keep_a_nested_hierarchy() {
    let pb_sample_to_cells = vec![vec![0], vec![1, 2], vec![3], vec![4, 5]];
    let finest = vec![4, 7, 7, 7, 9, 9];
    let coarse = vec![1, 0, 0, 0, 1, 1];

    let (levels, ks) = nested_levels_by_vote(&pb_sample_to_cells, &[finest, coarse]);
    assert_eq!(ks, vec![3, 2]);
    assert_eq!(levels[0], vec![0, 1, 1, 2]);
    assert_eq!(levels[1], vec![0, 1, 1, 0]);
}

fn backend_over(
    dir: &std::path::Path,
    tag: &str,
    genes: &[usize],
    n_cells: usize,
) -> Arc<dyn crate::sparse_io::SparseIo<IndexIter = Vec<usize>>> {
    let path = dir.join(format!("{tag}.zarr"));
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for c in 0..n_cells {
        for local in 0..genes.len() {
            triplets.push((local as u64, c as u64, 1.0 + ((c + local) % 3) as f32));
        }
    }
    let mut b = create_sparse_from_triplets(
        &triplets,
        (genes.len(), n_cells, triplets.len()),
        Some(path.to_str().expect("utf8")),
        Some(&SparseIoBackend::Zarr),
    )
    .expect("create backend");
    b.register_row_names_vec(
        &genes
            .iter()
            .map(|g| format!("g{g}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    b.register_column_names_vec(
        &(0..n_cells)
            .map(|c| format!("{tag}_c{c}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    Arc::from(b)
}

/// `size_ds[g, s]` is the mass of sample `s` whose backend measures gene `g`,
/// and the δ mask is 1 where a batch has a backend measuring the gene.
#[test]
fn observability_counts_the_measuring_mass_per_gene_and_sample() {
    const D: usize = 5;
    let dir = tempfile::tempdir().expect("tempdir");
    // Three panels: all genes, all but gene 4, and genes 0..=2 only.
    let panels: [Vec<usize>; 3] = [(0..D).collect(), (0..4).collect(), (0..3).collect()];
    let cells_per_backend = [7usize, 5, 6];

    let mut data = SparseIoVec::new()
        .with_row_alignment(RowAlignment::Union)
        .expect("union rows");
    for (i, (panel, &n)) in panels.iter().zip(&cells_per_backend).enumerate() {
        data.push(backend_over(dir.path(), &format!("p{i}"), panel, n), None)
            .expect("push");
    }
    let n: usize = cells_per_backend.iter().sum();
    assert_eq!(data.num_rows(), D);

    // Backends 0 and 1 share batch "x"; backend 2 is batch "y".
    let source_of = |c: usize| {
        if c < 7 {
            0
        } else if c < 12 {
            1
        } else {
            2
        }
    };
    let batches: Vec<&str> = (0..n)
        .map(|c| if source_of(c) < 2 { "x" } else { "y" })
        .collect();
    data.register_batch_membership(&batches);
    let batch_of = data.get_batch_membership(0..n);
    let num_groups = 4;
    let cell_to_group: Vec<usize> = (0..n).map(|c| (c * 5) % num_groups).collect();
    data.assign_groups(&pad_numeric_labels(&cell_to_group, num_groups), None);

    let mut stat = CollapsedStat::new(D, num_groups, 2);
    for &g in &cell_to_group {
        stat.size_s[g] += 1.0;
    }
    attach_observability(&mut stat, &data).expect("observability");

    let mut want_size = DMatrix::<f32>::zeros(D, num_groups);
    let mut want_mask = DMatrix::<f32>::zeros(D, 2);
    for c in 0..n {
        for &g in &panels[source_of(c)] {
            want_size[(g, cell_to_group[c])] += 1.0;
            want_mask[(g, batch_of[c])] = 1.0;
        }
    }
    assert_eq!(stat.size_ds.as_ref().expect("size_ds"), &want_size);
    assert_eq!(stat.obs_mask_db.as_ref().expect("mask"), &want_mask);
}
