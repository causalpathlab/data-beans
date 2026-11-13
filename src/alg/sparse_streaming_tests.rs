//! The streaming statistics must come out the same bits on any number of
//! threads: sums of non-integer values depend on how they are grouped, and
//! the grouping may not follow the scheduler.

use super::*;
use crate::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use legume_numeric::matrix::traits::RunningStatOps;
use std::sync::Arc;

/// Run `f` on a rayon pool of `threads` workers.
fn on_threads<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool")
        .install(f)
}

/// `d × n` matrix of non-integer values whose partial sums round differently
/// under different groupings.
fn fractional_data(dir: &tempfile::TempDir, d: usize, n: usize) -> anyhow::Result<SparseIoVec> {
    let path = dir.path().join("frac.zarr");
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for i in 0..d {
        for j in 0..n {
            if (i * 5 + j * 3) % 4 != 0 {
                let v = 0.1 + ((i * 131 + j * 17) % 97) as f32 * 0.37;
                triplets.push((i as u64, j as u64, v));
            }
        }
    }
    let mut backend = create_sparse_from_triplets(
        &triplets,
        (d, n, triplets.len()),
        Some(path.to_str().unwrap()),
        Some(&SparseIoBackend::Zarr),
    )?;
    let rows: Vec<Box<str>> = (0..d).map(|i| format!("g{i}").into()).collect();
    let cols: Vec<Box<str>> = (0..n).map(|j| format!("c{j}").into()).collect();
    backend.register_row_names_vec(&rows);
    backend.register_column_names_vec(&cols);
    let mut data_vec = SparseIoVec::new();
    data_vec.push(Arc::from(backend), None)?;
    Ok(data_vec)
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn fixed_order_fold_groups_by_position_alone() -> anyhow::Result<()> {
    let items: Vec<f32> = (0..1000).map(|i| 0.1 + (i % 89) as f32 * 0.37).collect();
    let sum = |threads: usize, leaf: usize| -> anyhow::Result<f32> {
        on_threads(threads, || {
            fold_in_fixed_order(
                &items,
                leaf,
                &|| 0f32,
                &|acc, &x| {
                    *acc += x;
                    Ok(())
                },
                &|a, b| *a += b,
            )
        })
    };
    let one = sum(1, 3)?;
    for threads in [2, 4, 7] {
        assert_eq!(sum(threads, 3)?.to_bits(), one.to_bits());
    }
    // One leaf is the plain left-to-right sum; no item at all is the seed.
    let plain = items.iter().fold(0f32, |a, &x| a + x);
    assert_eq!(sum(4, items.len())?.to_bits(), plain.to_bits());
    let nothing: [f32; 0] = [];
    let empty = fold_in_fixed_order(&nothing, 3, &|| 7f32, &|_, _| Ok(()), &|_, _| {})?;
    assert_eq!(empty, 7.0);
    Ok(())
}

#[test]
fn fixed_order_fold_passes_an_error_on() {
    let items: Vec<usize> = (0..100).collect();
    let out = fold_in_fixed_order(
        &items,
        4,
        &|| 0usize,
        &|acc, &x| {
            anyhow::ensure!(x != 57, "item {x}");
            *acc += x;
            Ok(())
        },
        &|a, b| *a += b,
    );
    assert!(out.is_err());
}

#[test]
fn streaming_stats_do_not_depend_on_the_thread_count() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let (d, n) = (8usize, 600usize);
    let data_vec = fractional_data(&dir, d, n)?;
    let row_to_gene: Vec<u32> = (0..d as u32).map(|r| r % 3).collect();

    let run = |threads: usize| {
        on_threads(threads, || -> anyhow::Result<_> {
            let plain = streaming_sparse_running_stats(&data_vec, Some(5), "test")?;
            let (rows, folded) =
                streaming_sparse_running_stats_folded(&data_vec, Some(5), "test", &row_to_gene, 3)?;
            Ok((plain, rows, folded))
        })
    };
    let (plain_1, rows_1, folded_1) = run(1)?;
    assert_eq!(plain_1.ncols_processed(), n);
    for threads in [3, 6] {
        let (plain, rows, folded) = run(threads)?;
        for (got, want) in [(&plain, &plain_1), (&rows, &rows_1), (&folded, &folded_1)] {
            assert_eq!(bits(&got.sum()), bits(&want.sum()), "{threads} threads");
            assert_eq!(
                bits(&got.variance()),
                bits(&want.variance()),
                "{threads} threads"
            );
        }
    }
    // The row statistics of the folded pass are the plain pass's.
    assert_eq!(bits(&rows_1.sum()), bits(&plain_1.sum()));
    Ok(())
}
