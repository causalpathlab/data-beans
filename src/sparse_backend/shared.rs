//! Storage-agnostic helpers shared by the `zarr` and `hdf5` sparse backends.
//!
//! These capture the *algorithmic* duplication between the two `SparseIo`
//! impls (range coalescing and the MTX export) without touching the
//! *structural* divergences (zarr's decoded-chunk caches and `Element` string
//! storage vs hdf5's datasets and `VarLenUnicode`). Each backend supplies its
//! own retrieval primitive via a closure or keeps its own dataset-write tail;
//! only the format-agnostic math lives here. (Name-file parsing is shared too,
//! but lives in `utilities::io_helpers::parse_name_file` since it is general
//! I/O, not backend-specific.)

use legume_numeric::matrix::common_io::open_buf_writer;
use std::io::Write;

/// Coalesce abutting/overlapping ranges in `tagged` (already sorted by
/// `start`), retrieve each merged span once through `retrieve`, and emit
/// triplets via `make_triplet(out_tag, inner_idx, value)`.
///
/// Used by both CSC (`out_tag` = output column, `inner_idx` = row) and CSR
/// (`out_tag` = output row, `inner_idx` = column) read paths. `retrieve`
/// returns the `(data, indices)` slabs covering `[start, end)`; abstracting it
/// keeps each backend's cache / dataset handle backend-private (zarr fetches
/// through its decoded-chunk cache, hdf5 through `read_slice_1d`). The buffers
/// are only indexed, so any `Index<usize>`-able owner works (`Vec`,
/// `ndarray::Array1`, …).
///
/// **Contract:** `retrieve(start, end)` must return buffers of length exactly
/// `end - start`, indexed base-0 at `start` (so element `start + k` lives at
/// buffer offset `k`). The `Index<usize>` bound can't express this; a caller
/// returning a short or differently-based buffer would index out of bounds or
/// silently misalign values against indices. Emission is in `tagged`-order
/// (i.e. sorted by `start`), not request order — fine for triplet consumers
/// that rebuild a matrix, which is order-independent.
pub(crate) fn coalesce_and_emit<R, F, D, I>(
    tagged: &[(u64, u64, u64)],
    inner_bound: usize,
    make_triplet: F,
    mut retrieve: R,
) -> anyhow::Result<Vec<(u64, u64, f32)>>
where
    R: FnMut(u64, u64) -> anyhow::Result<(D, I)>,
    D: std::ops::Index<usize, Output = f32>,
    I: std::ops::Index<usize, Output = u64>,
    F: Fn(u64, u64, f32) -> (u64, u64, f32),
{
    let total: u64 = tagged.iter().map(|&(_, s, e)| e - s).sum();
    let mut ret: Vec<(u64, u64, f32)> = Vec::with_capacity(total as usize);

    // Fuse across SMALL gaps, not only across abutting ranges. A column subset
    // that keeps every other column produces ranges that never abut, and
    // abutting-only merging degenerated into one retrieve per range — several
    // hundred thousand cache round-trips (each with subset math and two Vec
    // allocations) where a few dozen block reads carry the same bytes. Reading
    // a skipped gap costs its decode; a separate retrieve costs far more, so
    // gaps up to `GAP_FUSE` fuse. `MERGED_SPAN_CAP` bounds the retrieve buffer
    // so fusion cannot grow one read without limit (~48 MB across both arrays).
    const GAP_FUSE: u64 = 65_536;
    const MERGED_SPAN_CAP: u64 = 4_000_000;

    let mut i = 0;
    while i < tagged.len() {
        let merged_start = tagged[i].1;
        let mut merged_end = tagged[i].2;
        let mut j = i + 1;
        while j < tagged.len()
            && tagged[j].1 <= merged_end.saturating_add(GAP_FUSE)
            && tagged[j].2.max(merged_end) - merged_start <= MERGED_SPAN_CAP
        {
            merged_end = merged_end.max(tagged[j].2);
            j += 1;
        }

        let (data_buf, indices_buf) = retrieve(merged_start, merged_end)?;

        for &(tag, start, end) in &tagged[i..j] {
            let off = (start - merged_start) as usize;
            let len = (end - start) as usize;
            for k in 0..len {
                let inner = indices_buf[off + k];
                let val = data_buf[off + k];
                if inner >= inner_bound as u64 {
                    return Err(crate::sparse_io::index_audit::inner_out_of_range(
                        "sparse read",
                        (start + k as u64) as usize,
                        inner,
                        inner_bound,
                    ));
                }
                ret.push(make_triplet(tag, inner, val));
            }
        }

        i = j;
    }
    Ok(ret)
}

/// Block size (in elements) for streaming the CSC value/row-index arrays when
/// exporting to Matrix Market. ~1M elements ≈ 4 MiB f32 + 8 MiB u64 transient
/// per block — large enough to amortize per-retrieve overhead, small enough to
/// bound memory regardless of matrix size.
pub(crate) const MTX_STREAM_BLOCK: u64 = 1 << 20;

/// Write the MatrixMarket coordinate header: the format magic line followed by
/// the `rows  cols  nnz` size line.
pub(crate) fn write_mtx_header<W: Write>(
    buf: &mut W,
    nrow: usize,
    ncol: usize,
    nnz: usize,
) -> anyhow::Result<()> {
    writeln!(buf, "%%MatrixMarket matrix coordinate real general")?;
    writeln!(buf, "{}\t{}\t{}", nrow, ncol, nnz)?;
    Ok(())
}

/// Write a CSC matrix of `shape` = `(nrow, ncol, nnz)` as a MatrixMarket
/// coordinate file: the header, then one 1-based `row  col  value` line per
/// stored entry.
///
/// The value/row-index arrays are streamed in sequential blocks of `block`
/// elements (instead of two retrieves per column) and `indptr` is walked to
/// map each stored nonzero back to its column. CSC values are laid out
/// column-major in column order, so emitting them in storage order reproduces
/// the same column-by-column output as a per-column scan; empty columns are
/// skipped by advancing the column pointer.
///
/// `indptr` is audited against the shape before anything is written, so a
/// short or non-monotone pointer array, or one that disagrees with the
/// declared `nnz`, is an error rather than a file whose header and body
/// differ. A stored row index outside `nrow` is an error too.
///
/// `retrieve(start, end)` follows the same contract as in
/// [`coalesce_and_emit`]: `(data, indices)` buffers of length `end - start`,
/// indexed base-0 at `start`.
pub(crate) fn write_mtx<W, R, D, I>(
    buf: &mut W,
    shape: (usize, usize, usize),
    indptr: &[u64],
    block: u64,
    mut retrieve: R,
) -> anyhow::Result<()>
where
    W: Write,
    R: FnMut(u64, u64) -> anyhow::Result<(D, I)>,
    D: std::ops::Index<usize, Output = f32>,
    I: std::ops::Index<usize, Output = u64>,
{
    let (nrow, ncol, nnz) = shape;
    crate::sparse_io::index_audit::check_indptr("to_mtx_file", indptr, ncol, nnz)?;
    write_mtx_header(buf, nrow, ncol, nnz)?;

    let total_nnz = indptr[ncol];
    let block = block.max(1);
    let mut jj = 0usize; // column owning the running nnz position
    let mut pos = 0u64;
    while pos < total_nnz {
        let end = (pos + block).min(total_nnz);
        let (data_block, indices_block) = retrieve(pos, end)?;

        for k in 0..(end - pos) as usize {
            let global = pos + k as u64;
            // advance to the column owning this nonzero (skips empties)
            while indptr[jj + 1] <= global {
                jj += 1;
            }
            let ii = indices_block[k];
            if ii >= nrow as u64 {
                return Err(crate::sparse_io::index_audit::inner_out_of_range(
                    "to_mtx_file",
                    global as usize,
                    ii,
                    nrow,
                ));
            }
            // 1-based indices
            writeln!(buf, "{}\t{}\t{}", ii + 1, jj + 1, data_block[k])?;
        }
        pos = end;
    }
    Ok(())
}

/// Create `mtx_file`, let `write` fill it and flush it. A file left
/// incomplete by a failed `write` or flush is removed, so an export that
/// returns an error leaves nothing a reader could take for the matrix.
///
/// `mtx_file` follows `open_buf_writer`: `stdout` and `stderr` name the
/// standard streams, which are never removed.
pub(crate) fn write_mtx_file<F>(mtx_file: &str, write: F) -> anyhow::Result<()>
where
    F: FnOnce(&mut Box<dyn Write>) -> anyhow::Result<()>,
{
    let mut buf = open_buf_writer(mtx_file)?;
    let written = write(&mut buf).and_then(|()| Ok(buf.flush()?));
    // Only a regular file is a partial output to take away: a stream
    // (`stdout`, a device, a pipe) holds nothing to remove.
    if written.is_err() && std::fs::metadata(mtx_file).is_ok_and(|m| m.is_file()) {
        // close the file before unlinking it
        drop(buf);
        let _ = std::fs::remove_file(mtx_file);
    }
    written
}

#[cfg(test)]
#[path = "shared_tests.rs"]
mod tests;
