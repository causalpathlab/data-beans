#[cfg(feature = "ndarray")]
use ndarray::prelude::*;
use rustc_hash::FxHashMap as HashMap;

use super::DMatrix;

pub fn build_name2index_map(_names: &[Box<str>]) -> HashMap<Box<str>, usize> {
    _names
        .iter()
        .enumerate()
        .map(|(r, name)| (name.clone(), r))
        .collect()
}

pub fn take_subset_indices_names(
    new_indices: &[usize],
    ntot: usize,
    old_names: Vec<Box<str>>,
) -> (HashMap<u64, u64>, Vec<Box<str>>) {
    let mut old2new: HashMap<u64, u64> = Default::default();
    let mut new2old = vec![];
    debug_assert!(ntot == old_names.len());
    let mut k = 0_u64;
    for idx in new_indices.iter() {
        if *idx < ntot {
            old2new.insert(*idx as u64, k);
            new2old.push(*idx);
            k += 1;
        }
    }

    let new_names = new2old
        .iter()
        .map(|&i| old_names[i].clone())
        .collect::<Vec<Box<str>>>();

    (old2new, new_names)
}

pub fn take_subset_indices_names_if_needed(
    new_indices: Option<&Vec<usize>>,
    ntot: Option<usize>,
    old_names: Vec<Box<str>>,
) -> (HashMap<u64, u64>, Vec<Box<str>>) {
    let ntot = ntot.unwrap_or(old_names.len());
    if let Some(new_indices) = new_indices {
        take_subset_indices_names(new_indices, ntot, old_names)
    } else {
        let names = old_names;
        let identity = (0..(ntot as u64))
            .zip(0..(ntot as u64))
            .collect::<HashMap<u64, u64>>();
        (identity, names)
    }
}

#[cfg(feature = "ndarray")]
pub fn ndarray_to_triplets(array: &Array2<f32>) -> Vec<(u64, u64, f32)> {
    let eps = 1e-6;
    array
        .indexed_iter()
        .filter(|(_, &elem)| elem.abs() > eps)
        .map(|((row, col), &value)| (row as u64, col as u64, value))
        .collect::<Vec<(u64, u64, f32)>>()
}

pub fn dmatrix_to_triplets(matrix: &DMatrix<f32>) -> Vec<(u64, u64, f32)> {
    let (nrow, _) = matrix.shape();
    let eps = 1e-6;
    matrix
        .iter() // column-major
        .enumerate()
        .filter(|(_, &elem)| elem.abs() > eps)
        .map(|(idx, &value)| {
            let row = idx % nrow;
            let col = idx / nrow;
            (row as u64, col as u64, value)
        })
        .collect::<Vec<(u64, u64, f32)>>()
}

/// Remove a backend at `path`, whether it is a file (`.h5`, `.zarr.zip`) or a
/// directory (`.zarr`). Nothing happens when the path does not exist.
pub fn remove_backend_path(path: &str) -> anyhow::Result<()> {
    let p = std::path::Path::new(path);
    if p.exists() {
        if p.is_file() {
            std::fs::remove_file(p)?;
        } else {
            std::fs::remove_dir_all(p)?;
        }
    }
    Ok(())
}

/// Bytes reserved by live preloads in this process; see [`reserve_preload`].
static RESERVED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Preloading costs 12 bytes per non-zero (a `u64` index and an `f32` value).
const PRELOAD_BYTES_PER_NNZ: usize = 12;

/// A preload's share of the process-wide preload budget, given back when it
/// is dropped. Keep it next to the preloaded arrays and drop it with them,
/// so a later preload in the same process sees the memory as free again.
///
/// Cloning reserves the same amount again: a clone carries its own copy of
/// the arrays.
#[derive(Debug)]
pub struct PreloadReservation {
    bytes: usize,
    counter: &'static std::sync::atomic::AtomicUsize,
}

impl PreloadReservation {
    /// Bytes this reservation holds against the budget (0 under a
    /// `LEGUME_PRELOAD_BUDGET_BYTES` override, which is per preload).
    #[allow(dead_code)] // library API; the bin has no caller
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Clone for PreloadReservation {
    fn clone(&self) -> Self {
        self.counter
            .fetch_add(self.bytes, std::sync::atomic::Ordering::Relaxed);
        Self {
            bytes: self.bytes,
            counter: self.counter,
        }
    }
}

impl Drop for PreloadReservation {
    fn drop(&mut self) {
        self.counter
            .fetch_sub(self.bytes, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Bytes currently held by live [`PreloadReservation`]s in this process.
#[allow(dead_code)] // library API; the bin has no caller
pub fn preload_reserved_bytes() -> usize {
    RESERVED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Reserve room for a preload of `nnz` entries, or `None` when it does not
/// fit the budget.
///
/// An unchecked preload at imaging scale is an OOM order, not a request. The
/// budget turns that into a logged skip: every read path already handles the
/// not-preloaded state, it is just slower.
///
/// The budget is half the memory available when the process first asks,
/// shared by every live preload of the process (8 GiB when the system does
/// not say). The reservation is given back when the returned guard drops.
/// `LEGUME_PRELOAD_BUDGET_BYTES` overrides it per preload, following the
/// `LEGUME_ZARR_CACHE_CAP` precedent for memory knobs.
pub fn reserve_preload(nnz: usize, what: &str) -> Option<PreloadReservation> {
    let cost = nnz.saturating_mul(PRELOAD_BYTES_PER_NNZ);
    if let Some(budget) = std::env::var("LEGUME_PRELOAD_BUDGET_BYTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        return fits_or_warn(cost, budget, nnz, what).then_some(PreloadReservation {
            bytes: 0,
            counter: &RESERVED,
        });
    }
    reserve_against(&RESERVED, cost, auto_preload_budget_bytes(), nnz, what)
}

fn reserve_against(
    counter: &'static std::sync::atomic::AtomicUsize,
    cost: usize,
    budget: usize,
    nnz: usize,
    what: &str,
) -> Option<PreloadReservation> {
    use std::sync::atomic::Ordering;
    let reserved = counter.fetch_add(cost, Ordering::Relaxed);
    if fits_or_warn(reserved.saturating_add(cost), budget, nnz, what) {
        Some(PreloadReservation {
            bytes: cost,
            counter,
        })
    } else {
        counter.fetch_sub(cost, Ordering::Relaxed);
        None
    }
}

/// Whether a preload of `nnz` entries fits the budget. Unlike
/// [`reserve_preload`], the reservation is held for the life of the process;
/// prefer [`reserve_preload`] and keep its guard with the preloaded arrays.
#[allow(dead_code)] // library API; the bin has no caller
pub fn preload_within_budget(nnz: usize, what: &str) -> bool {
    reserve_preload(nnz, what).map(std::mem::forget).is_some()
}

fn fits_or_warn(cost: usize, budget: usize, nnz: usize, what: &str) -> bool {
    if cost > budget {
        log::warn!(
            "skipping {what} preload of {nnz} nnz: {cost} bytes would exceed the \
             {budget}-byte budget (LEGUME_PRELOAD_BUDGET_BYTES to raise); reads stay on the \
             streaming path"
        );
        false
    } else {
        true
    }
}

/// Half the memory available at the first call, or 8 GiB when unknown.
fn auto_preload_budget_bytes() -> usize {
    static BUDGET: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        const FALLBACK_BYTES: usize = 8 << 30;
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        match usize::try_from(sys.available_memory()) {
            Ok(avail) if avail > 0 => avail / 2,
            _ => FALLBACK_BYTES,
        }
    })
}

/// Whether loaders preload columns without being asked. On unless
/// `LEGUME_AUTO_PRELOAD` is `0`, `false` or `off`; the preload itself still
/// answers to [`preload_within_budget`], so data that does not fit in memory
/// stays on the streaming path.
#[allow(dead_code)] // called by library loaders; the bin has none
pub fn auto_preload_enabled() -> bool {
    !matches!(
        std::env::var("LEGUME_AUTO_PRELOAD").as_deref(),
        Ok("0") | Ok("false") | Ok("off")
    )
}

/// Bytes of `(u64, u64, f32)` triplets a streaming-write slab may hold, and the
/// padded size of one such triplet. One definition, because the two streaming
/// pipelines (the subset trait method and the handlers' column-selection
/// writer) each carried their own copy and two memory ceilings drift apart.
pub const SLAB_BUDGET_BYTES: usize = 256 << 20;
/// `(u64, u64, f32)` padded.
pub const TRIPLET_BYTES: usize = 24;

#[cfg(test)]
mod preload_reservation_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn dropping_a_reservation_gives_it_back() {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let a = reserve_against(&COUNTER, 60, 100, 5, "test").expect("fits");
        assert_eq!(COUNTER.load(Ordering::Relaxed), 60);
        // A second preload that would overflow the budget is refused and
        // leaves nothing behind.
        assert!(reserve_against(&COUNTER, 60, 100, 5, "test").is_none());
        assert_eq!(COUNTER.load(Ordering::Relaxed), 60);
        drop(a);
        assert_eq!(COUNTER.load(Ordering::Relaxed), 0);
        // With the first one gone, the same preload fits again.
        let b = reserve_against(&COUNTER, 60, 100, 5, "test").expect("fits after release");
        assert_eq!(b.bytes(), 60);
        drop(b);
        assert_eq!(COUNTER.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn cloning_a_reservation_reserves_again() {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let a = reserve_against(&COUNTER, 30, 100, 5, "test").expect("fits");
        let b = a.clone();
        assert_eq!(COUNTER.load(Ordering::Relaxed), 60);
        drop(a);
        assert_eq!(COUNTER.load(Ordering::Relaxed), 30);
        drop(b);
        assert_eq!(COUNTER.load(Ordering::Relaxed), 0);
    }
}
