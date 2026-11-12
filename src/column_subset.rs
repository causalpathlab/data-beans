//! Stream a column selection (and an optional row selection or renumbering)
//! of a sparse backend into a fresh backend without materialising the
//! survivors. Shared by the `split` / `subsample` / `subset-*` CLI handlers,
//! by the in-place subset and reorder of [`SparseIo`], and by downstream
//! crates (faba's `qc`) that filter a matrix by cells and features.

use crate::sparse_io::SparseIo;
use rayon::prelude::*;

/// Where each row of the input lands in the output: `to[old]` is its new
/// index, `None` when the row is dropped; `nrow` is the output's row count.
/// Rows may land in any order, and an output row may be fed by no input row
/// (it is then empty), but never by two.
pub struct RowMap {
    pub to: Vec<Option<u64>>,
    pub nrow: usize,
}

impl RowMap {
    /// The map that keeps the rows `keep` of `nrow_full`, in `keep`'s order.
    pub fn keeping(keep: &[usize], nrow_full: usize) -> anyhow::Result<Self> {
        let mut to = vec![None; nrow_full];
        for (new, &old) in keep.iter().enumerate() {
            let slot = to
                .get_mut(old)
                .ok_or_else(|| anyhow::anyhow!("row {old} is past the {nrow_full} rows"))?;
            anyhow::ensure!(slot.is_none(), "the row selection repeats row {old}");
            *slot = Some(new as u64);
        }
        Ok(Self {
            to,
            nrow: keep.len(),
        })
    }

    /// Whether the rows kept stay in their order, so that a column's rows
    /// still ascend once renumbered and need no sort.
    fn is_monotone(&self) -> bool {
        self.to.iter().flatten().is_sorted_by(|a, b| a < b)
    }

    /// Each output row is in range and fed by at most one of the `nrow_full`
    /// input rows.
    fn check(&self, nrow_full: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.to.len() == nrow_full,
            "the row map covers {} rows, the data has {nrow_full}",
            self.to.len()
        );
        let mut fed = vec![false; self.nrow];
        for &new in self.to.iter().flatten() {
            let slot = fed.get_mut(new as usize).ok_or_else(|| {
                anyhow::anyhow!("row {new} is past the {} output rows", self.nrow)
            })?;
            anyhow::ensure!(
                !std::mem::replace(slot, true),
                "two rows of the data land on output row {new}"
            );
        }
        Ok(())
    }
}

/// Number of surviving entries when every column of `src` is kept, read off
/// the resident row index without touching the data. `None` when that index
/// is absent or does not describe `nrow` rows.
fn nnz_of_kept_rows<S>(src: &S, rows: &RowMap) -> Option<u64>
where
    S: SparseIo + ?Sized,
{
    let indptr = src.row_indptr();
    if indptr.len() != rows.to.len() + 1 {
        return None;
    }
    rows.to
        .iter()
        .zip(indptr.windows(2))
        .filter(|(kept, _)| kept.is_some())
        .try_fold(0u64, |acc, (_, w)| Some(acc + w[1].checked_sub(w[0])?))
}

/// True when `cols` names each of the `ncol` columns exactly once.
fn is_permutation(cols: &[usize], ncol: usize) -> bool {
    let mut seen = vec![false; ncol];
    cols.len() == ncol
        && cols
            .iter()
            .all(|&c| c < ncol && !std::mem::replace(&mut seen[c], true))
}

/// Write columns `old_cols` of `src`, in that order and with rows renumbered
/// by `rows` (`None` keeps every row), as the whole content of `out`: the
/// column index in byte-budgeted slabs, then the row index transposed from
/// it. `out` is an empty streaming backend; names and metadata are left to
/// the caller. Returns the output's `(rows, columns, non-zeros)`.
///
/// The output's nnz comes from the resident indptr when every row survives,
/// or when every column does (the by-row indptr then counts the rows kept);
/// otherwise from one counting pass that keeps counts, never entries. A row
/// map that reorders costs one small sort per column.
pub(crate) fn stream_csc_selection<S, O>(
    src: &S,
    old_cols: &[usize],
    rows: Option<&RowMap>,
    out: &mut O,
) -> anyhow::Result<(usize, usize, usize)>
where
    S: SparseIo + Sync + ?Sized,
    O: SparseIo + ?Sized,
{
    let nrow_src = src
        .num_rows()
        .ok_or_else(|| anyhow::anyhow!("backend has no `nrow`"))?;
    if let Some(rows) = rows {
        rows.check(nrow_src)?;
    }
    let monotone = rows.is_none_or(RowMap::is_monotone);
    let out_nrow = rows.map_or(nrow_src, |r| r.nrow);
    let ncol_src = src
        .num_columns()
        .ok_or_else(|| anyhow::anyhow!("backend has no `ncol`"))?;
    let out_ncol = old_cols.len();

    // Stored nnz of each selected column, off the resident indptr: exact when
    // every row survives, an upper bound under a row filter.
    let stored_nnz: Vec<u64> = old_cols
        .iter()
        .map(|&c| {
            src.column_nnz(c)
                .ok_or_else(|| anyhow::anyhow!("no indptr entry for column {c}"))
        })
        .collect::<anyhow::Result<_>>()?;

    // The writer needs the exact total up front. Without a row filter it is
    // the stored total. With one, and every column kept, the resident row
    // index gives it for free. Otherwise one counting pass — block reads,
    // several at once, keeping counts and never entries.
    let (nnz, slab_nnz): (u64, Vec<u64>) = match rows {
        None => (stored_nnz.iter().sum(), stored_nnz),
        Some(remap) => {
            let from_row_index = is_permutation(old_cols, ncol_src)
                .then(|| nnz_of_kept_rows(src, remap))
                .flatten();
            match from_row_index {
                Some(total) => (total, stored_nnz),
                None => {
                    let coarse = legume_numeric::matrix::utils::generate_minibatch_intervals(
                        out_ncol,
                        0,
                        Some(8192),
                    );
                    let counts: Vec<u64> = coarse
                        .par_iter()
                        .map(|&(lb, ub)| -> anyhow::Result<Vec<u64>> {
                            let mut counts = vec![0u64; ub - lb];
                            let (_, _, triplets) = src.read_triplets_by_columns(
                                old_cols[lb..ub].iter().copied().collect(),
                            )?;
                            for (r, c_local, _) in triplets {
                                if remap.to[r as usize].is_some() {
                                    counts[c_local as usize] += 1;
                                }
                            }
                            Ok(counts)
                        })
                        .collect::<anyhow::Result<Vec<_>>>()?
                        .into_iter()
                        .flatten()
                        .collect();
                    (counts.iter().sum(), counts)
                }
            }
        }
    };

    out.begin_streaming_csc((out_nrow, out_ncol, nnz as usize))?;

    // Blocks bounded by bytes of triplets read, not by a fixed column count.
    let blocks = legume_numeric::matrix::utils::byte_budget_intervals(
        &slab_nnz,
        crate::sparse_io::SLAB_BUDGET_BYTES,
        crate::sparse_io::TRIPLET_BYTES,
    );

    let mut nnz_offset = 0u64;
    let mut column_scratch: Vec<(u64, f32)> = Vec::new();
    for (lb, ub) in blocks {
        // ONE block read per slab, never a read per column: a single-column
        // read pays the cached-subset machinery per call, and at hundreds of
        // thousands of columns that made this path slower than the memory wall
        // it replaced. The block read returns LOCAL column ids for the
        // requested set, rows ascending within each column.
        let (_, _, triplets) =
            src.read_triplets_by_columns(old_cols[lb..ub].iter().copied().collect())?;
        let new_row = |r: u64| match rows {
            None => Some(r),
            Some(remap) => remap.to[r as usize],
        };

        // Counting-sort scatter: size every column from the survivors, then
        // drop each entry at its column's cursor. Entries of one column keep
        // their arrival order, whatever order the columns arrive in.
        let n_block = ub - lb;
        let mut colptr = vec![0u64; n_block + 1];
        for &(r, c_local, _) in &triplets {
            if new_row(r).is_some() {
                colptr[c_local as usize + 1] += 1;
            }
        }
        for c in 0..n_block {
            colptr[c + 1] += colptr[c];
        }
        let slab_len = colptr[n_block] as usize;
        let mut row_indices = vec![0u64; slab_len];
        let mut values = vec![0f32; slab_len];
        let mut cursor = colptr[..n_block].to_vec();
        for (r, c_local, x) in triplets {
            if let Some(new_r) = new_row(r) {
                let at = &mut cursor[c_local as usize];
                row_indices[*at as usize] = new_r;
                values[*at as usize] = x;
                *at += 1;
            }
        }

        if !monotone {
            // Renumbering scrambled the order within each column; restore the
            // ascending-rows invariant the writer enforces.
            for w in colptr.windows(2) {
                let span = w[0] as usize..w[1] as usize;
                column_scratch.clear();
                column_scratch.extend(
                    row_indices[span.clone()]
                        .iter()
                        .copied()
                        .zip(values[span.clone()].iter().copied()),
                );
                column_scratch.sort_unstable_by_key(|&(r, _)| r);
                for (k, &(r, x)) in span.zip(&column_scratch) {
                    row_indices[k] = r;
                    values[k] = x;
                }
            }
        }

        out.append_csc_slab(
            lb as u64,
            nnz_offset,
            &colptr[..n_block],
            &row_indices,
            &values,
        )?;
        nnz_offset += slab_len as u64;
    }

    out.finalize_streaming_csc()?;
    out.build_csr_from_csc_streaming()?;
    Ok((out_nrow, out_ncol, nnz as usize))
}

/// Stream a column selection (with an optional row filter) into a fresh
/// backend, without ever materialising the survivors.
///
/// The shape `split` and `subsample` share: pick columns, optionally keep a
/// subset of rows, write a new file. This is [`stream_mapped_selection`] over
/// the rows `row_filter` keeps, in its order.
pub fn stream_column_selection(
    data: &dyn SparseIo<IndexIter = Vec<usize>>,
    selected_columns: &[usize],
    row_filter: Option<&[usize]>,
    out_row_names: &[Box<str>],
    out_col_names: &[Box<str>],
    file_out: &str,
    backend_out: &crate::sparse_io::SparseIoBackend,
) -> anyhow::Result<(usize, usize, usize)> {
    let nrow_full = data
        .num_rows()
        .ok_or_else(|| anyhow::anyhow!("backend has no `nrow`"))?;
    let rows = row_filter
        .map(|keep| RowMap::keeping(keep, nrow_full))
        .transpose()?;
    stream_mapped_selection(
        data,
        selected_columns,
        rows.as_ref(),
        out_row_names,
        out_col_names,
        file_out,
        backend_out,
    )
}

/// Stream the columns `selected_columns` of `data`, in that order, into a
/// fresh backend at `file_out`, its rows renumbered (dropped, reordered) by
/// `rows` when given, with the given names and `data`'s metadata. Returns the
/// output's `(rows, columns, non-zeros)`.
///
/// The survivors are never materialised: they flow through the validated
/// streaming writer in byte-budgeted slabs, cut from the resident indptr.
pub fn stream_mapped_selection(
    data: &dyn SparseIo<IndexIter = Vec<usize>>,
    selected_columns: &[usize],
    rows: Option<&RowMap>,
    out_row_names: &[Box<str>],
    out_col_names: &[Box<str>],
    file_out: &str,
    backend_out: &crate::sparse_io::SparseIoBackend,
) -> anyhow::Result<(usize, usize, usize)> {
    let mut out =
        crate::sparse_io::create_sparse_streaming_empty(Some(file_out), Some(backend_out))?;

    let t_stream = std::time::Instant::now();
    let shape = stream_csc_selection(data, selected_columns, rows, &mut *out)?;
    log::info!(
        "streamed {} entries in {} columns ({:.1}s)",
        shape.2,
        shape.1,
        t_stream.elapsed().as_secs_f32()
    );
    out.register_row_names_vec(out_row_names);
    out.register_column_names_vec(out_col_names);
    out.set_metadata(&data.metadata())?;
    Ok(shape)
}
