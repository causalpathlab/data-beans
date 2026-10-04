//! Index audits for compressed sparse arrays.
//!
//! An index that is out of range does not always fail loudly: a stored row
//! index past `nrow` panics far from where it went bad, or, worse, lands on a
//! valid slot of a lookup table and silently reroutes a value. These checks
//! run where indices enter memory (preload) and again where they are used, so
//! a bad value surfaces as an error that names the array, the position and
//! the value, close to where it appeared.

use rayon::prelude::*;

/// Check a compressed-sparse pointer array: `n_major + 1` entries, starting
/// at 0, never decreasing, and ending at `nnz`.
pub(crate) fn check_indptr(
    label: &str,
    indptr: &[u64],
    n_major: usize,
    nnz: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        indptr.len() == n_major + 1,
        "{label}: indptr has {} entries, expected {}",
        indptr.len(),
        n_major + 1
    );
    anyhow::ensure!(
        indptr[0] == 0,
        "{label}: indptr[0] = {}, expected 0",
        indptr[0]
    );
    if let Some(w) = indptr.windows(2).position(|w| w[0] > w[1]) {
        anyhow::bail!(
            "{label}: indptr decreases at {w}: {} > {}",
            indptr[w],
            indptr[w + 1]
        );
    }
    anyhow::ensure!(
        indptr[n_major] == nnz as u64,
        "{label}: indptr ends at {}, but the arrays hold {nnz} entries",
        indptr[n_major]
    );
    Ok(())
}

/// Check that every stored inner index (row for CSC, column for CSR) is
/// below `bound`. Parallel: one compare per entry.
pub(crate) fn check_inner_indices(
    label: &str,
    indices: &[u64],
    bound: usize,
) -> anyhow::Result<()> {
    let bound = bound as u64;
    match indices.par_iter().position_first(|&i| i >= bound) {
        None => Ok(()),
        Some(k) => Err(inner_out_of_range(label, k, indices[k], bound as usize)),
    }
}

/// Check one major slot's pointer range `[start, end)` before slicing with it.
pub(crate) fn check_slot(
    label: &str,
    slot: usize,
    start: u64,
    end: u64,
    nnz: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        start <= end && end <= nnz as u64,
        "{label}: slot {slot} points at [{start}, {end}) in an array of {nnz} entries"
    );
    Ok(())
}

/// Check that the resident `indptr` covers every major slot a read is about
/// to index. Only the requested slots are checked, so a read that touches no
/// slot (an empty request) succeeds as it always has, even on a backend that
/// carries no index for this orientation.
pub(crate) fn check_requested_slots(
    label: &str,
    requested: impl IntoIterator<Item = usize>,
    indptr_len: usize,
) -> anyhow::Result<()> {
    if let Some(slot) = requested.into_iter().find(|&s| s + 1 >= indptr_len) {
        if indptr_len == 0 {
            anyhow::bail!(
                "{label}: slot {slot} requested, but this backend has no {label} index \
                 (its indptr array is missing)"
            );
        }
        anyhow::bail!(
            "{label}: slot {slot} requested, but the indptr has only {indptr_len} entries"
        );
    }
    Ok(())
}

/// The error for an inner index found out of range at a use site.
pub(crate) fn inner_out_of_range(
    label: &str,
    pos: usize,
    value: u64,
    bound: usize,
) -> anyhow::Error {
    anyhow::anyhow!(
        "{label}: stored index {value} (0x{value:016x}) at position {pos} is outside 0..{bound}"
    )
}

/// Check a whole preloaded compressed-sparse structure.
pub(crate) fn check_compressed(
    label: &str,
    indptr: &[u64],
    indices: &[u64],
    n_values: usize,
    n_major: usize,
    n_inner: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        indices.len() == n_values,
        "{label}: {} indices vs {n_values} values",
        indices.len()
    );
    check_indptr(label, indptr, n_major, indices.len())?;
    check_inner_indices(label, indices, n_inner)
}

/// Which axis a compressed structure is major in.
#[derive(Clone, Copy)]
pub(crate) enum Major {
    Column,
    Row,
}

/// Audit freshly preloaded arrays against the backend's shape, so a bad
/// stored index is reported with its position before the arrays become the
/// fast path. Skipped while the shape is unknown.
pub(crate) fn check_preload(
    major: Major,
    nrow: Option<usize>,
    ncol: Option<usize>,
    indptr: &[u64],
    indices: &[u64],
    n_values: usize,
) -> anyhow::Result<()> {
    let (Some(nrow), Some(ncol)) = (nrow, ncol) else {
        return Ok(());
    };
    match major {
        Major::Column => check_compressed("preload_columns", indptr, indices, n_values, ncol, nrow),
        Major::Row => check_compressed("preload_rows", indptr, indices, n_values, nrow, ncol),
    }
}

/// Append the entries of one major `slot` of preloaded arrays to `out`, as
/// `make(inner, value)`. The slot's pointer range and every inner index are
/// checked first: the preload audit ran once, and this catches a value that
/// changed after it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_slot<T>(
    label: &str,
    slot: usize,
    indptr: &[u64],
    indices: &[u64],
    data: &[f32],
    inner_bound: usize,
    out: &mut Vec<T>,
    make: impl Fn(u64, f32) -> T,
) -> anyhow::Result<()> {
    let (start, end) = (indptr[slot], indptr[slot + 1]);
    check_slot(label, slot, start, end, indices.len().min(data.len()))?;
    let (start, end) = (start as usize, end as usize);
    let idx = &indices[start..end];
    if let Some(k) = idx.iter().position(|&i| i >= inner_bound as u64) {
        return Err(inner_out_of_range(label, start + k, idx[k], inner_bound));
    }
    out.extend(idx.iter().zip(&data[start..end]).map(|(&i, &v)| make(i, v)));
    Ok(())
}

#[cfg(test)]
#[path = "index_audit_tests.rs"]
mod tests;
