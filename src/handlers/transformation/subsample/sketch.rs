//! The sketch `subsample --sketch` draws across: each cell's bin, from
//! senna's own random projection (`--proj-dim`) and sign-bit binning
//! (`--sort-dim`), in `data_beans::alg`.

use rand::rngs::SmallRng;
use rand::seq::index::sample;
use std::collections::BTreeMap;

/// How the cells are projected and binned.
#[derive(Clone, Copy, Debug)]
pub(super) struct Sketch {
    /// Dimensions of the random projection.
    pub dim: usize,
    /// Top components whose signs make a bin: at most `2^bits` bins.
    pub bits: usize,
}

/// Each cell of the backend at `data_file` in a bin: log1p counts, each
/// cell scaled to unit length, times a Gaussian `dim × genes` matrix,
/// standardised; then a bit per sign of its loading on each of the top
/// `bits` singular vectors.
#[cfg(feature = "alg")]
pub(super) fn bins(data_file: &str, sketch: Sketch, seed: u64) -> anyhow::Result<Vec<usize>> {
    use data_beans::alg::random_projection::{binary_sort_columns, RandProjOps};
    use std::sync::Arc;
    // The library's own reader: `alg` works on its types, not the CLI's.
    let (backend, file) = data_beans::hdf5_io::resolve_backend_file(data_file, None)?;
    let mut data = data_beans::sparse_io_vector::SparseIoVec::new();
    data.push(
        Arc::from(data_beans::sparse_io::open_sparse_matrix(&file, &backend)?),
        None,
    )?;
    let ncol = data.num_columns();
    if ncol < 2 {
        return Ok(vec![0; ncol]);
    }
    let proj = data
        .project_columns_with_batch_correction_seeded::<usize>(sketch.dim, None, None, seed)?
        .proj;
    binary_sort_columns(&proj, sketch.bits.min(sketch.dim).min(ncol))
}

/// Without `alg` there is no projection to bin by.
#[cfg(not(feature = "alg"))]
pub(super) fn bins(_: &str, _: Sketch, _: u64) -> anyhow::Result<Vec<usize>> {
    anyhow::bail!("--sketch needs data-beans built with the `alg` feature (on by default)")
}

/// `k` of `cells` at random, as evenly over their bins as the bins' sizes
/// allow (all `cells` when `k` covers them).
pub(super) fn draw_across_bins(
    rng: &mut SmallRng,
    cells: &[usize],
    bin_of: &[usize],
    k: usize,
) -> Vec<usize> {
    let mut by_bin: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &c in cells {
        by_bin.entry(bin_of[c]).or_default().push(c);
    }
    let bins: Vec<Vec<usize>> = by_bin.into_values().collect();
    let sizes: Vec<usize> = bins.iter().map(Vec::len).collect();
    let shares = even_shares(rng, &sizes, k);
    bins.iter()
        .zip(shares)
        .flat_map(|(bin, n)| {
            sample(rng, bin.len(), n)
                .into_iter()
                .map(|i| bin[i])
                .collect::<Vec<_>>()
        })
        .collect()
}

/// `n` split over bins of `sizes`, as evenly as they allow: every bin
/// gives the same share or, when smaller, all it has; the shares add up to
/// `n` (to every cell, if `n` is more). What does not divide evenly goes
/// one each to bins chosen at random.
pub(super) fn even_shares(rng: &mut SmallRng, sizes: &[usize], n: usize) -> Vec<usize> {
    let mut share = vec![0; sizes.len()];
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by_key(|&b| sizes[b]);
    let mut left = n.min(sizes.iter().sum());
    let mut open = &order[..];
    while !open.is_empty() && left > 0 {
        let level = left / open.len();
        // The bins no bigger than the even share give all they have.
        let small = open.partition_point(|&b| sizes[b] <= level);
        if small > 0 {
            for &b in &open[..small] {
                share[b] = sizes[b];
                left -= sizes[b];
            }
            open = &open[small..];
            continue;
        }
        // Every open bin is bigger than `level`, so each has room for one more.
        for &b in open {
            share[b] = level;
        }
        for i in sample(rng, open.len(), left - level * open.len()) {
            share[open[i]] += 1;
        }
        break;
    }
    share
}
