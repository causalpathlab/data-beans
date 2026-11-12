//! Stream a column selection (and an optional ascending row selection) of a
//! sparse backend into a fresh backend without materialising the survivors.
//! Shared by the `split` / `subsample` CLI handlers and by downstream crates
//! (faba's `qc`) that filter a matrix by cells and features.

use crate::sparse_io::SparseIo;

/// Where each row of the source lands in a streamed selection.
pub(crate) struct RowRemap {
    /// Old row → new row; `None` drops the row.
    map: Vec<Option<u64>>,
    /// True when the surviving rows keep their relative order, so a column's
    /// entries stay ascending after renumbering and need no sort.
    monotone: bool,
}

impl RowRemap {
    /// From a dense old → new map over the source's rows.
    pub(crate) fn new(map: Vec<Option<u64>>) -> Self {
        let monotone = map.iter().flatten().is_sorted_by(|a, b| a < b);
        Self { map, monotone }
    }

    /// Keep `rows` of a `nrow`-row source, renumbered in the order given.
    pub(crate) fn from_selection(rows: &[usize], nrow: usize) -> anyhow::Result<Self> {
        let mut map = vec![None; nrow];
        for (new, &old) in rows.iter().enumerate() {
            let slot = map.get_mut(old).ok_or_else(|| {
                anyhow::anyhow!("row index {old} is outside the {nrow}-row matrix")
            })?;
            *slot = Some(new as u64);
        }
        Ok(Self::new(map))
    }
}

/// Number of surviving entries when every column of `src` is kept, read off
/// the resident row index without touching the data. `None` when that index
/// is absent or does not describe `nrow` rows.
fn nnz_of_kept_rows<S>(src: &S, rows: &RowRemap) -> Option<u64>
where
    S: SparseIo + ?Sized,
{
    let indptr = src.row_indptr();
    if indptr.len() != rows.map.len() + 1 {
        return None;
    }
    rows.map
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
/// the caller. Returns the number of entries written.
pub(crate) fn stream_csc_selection<S, O>(
    src: &S,
    old_cols: &[usize],
    rows: Option<&RowRemap>,
    out_nrow: usize,
    out: &mut O,
) -> anyhow::Result<usize>
where
    S: SparseIo + ?Sized,
    O: SparseIo + ?Sized,
{
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
    // never one column at a time, keeping counts and never entries.
    let (nnz, slab_nnz): (u64, Vec<u64>) = match rows {
        None => (stored_nnz.iter().sum(), stored_nnz),
        Some(remap) => {
            let from_row_index = is_permutation(old_cols, ncol_src)
                .then(|| nnz_of_kept_rows(src, remap))
                .flatten();
            match from_row_index {
                Some(total) => (total, stored_nnz),
                None => {
                    let mut counts = vec![0u64; out_ncol];
                    let coarse = legume_numeric::matrix::utils::generate_minibatch_intervals(
                        out_ncol,
                        0,
                        Some(8192),
                    );
                    for (lb, ub) in coarse {
                        let (_, _, triplets) = src
                            .read_triplets_by_columns(old_cols[lb..ub].iter().copied().collect())?;
                        for (r, c_local, _) in triplets {
                            if remap.map[r as usize].is_some() {
                                counts[lb + c_local as usize] += 1;
                            }
                        }
                    }
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
            Some(remap) => remap.map[r as usize],
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

        if rows.is_some_and(|remap| !remap.monotone) {
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
    Ok(nnz as usize)
}

/// Stream a column selection (with an optional ASCENDING row filter) into a
/// fresh backend, without ever materialising the survivors.
///
/// The shape `split` and `subsample` share: pick columns, optionally keep an
/// ascending subset of rows, write a new file. Both used to read every
/// surviving triplet into one `Vec` and hand it to the sorting triplet writer —
/// 24 B/nnz of residency that OOMs at imaging scale. Here the exact output nnz
/// comes from the resident indptr (or one counting pass when rows and columns
/// are both filtered), and survivors flow through the validated streaming
/// writer in byte-budgeted slabs.
///
/// `row_filter`, when given, must be ascending: the renumbering is then
/// monotone, so within-column row order survives and no per-column sort is
/// needed. Both callers sample or select ascending; the debug assert keeps the
/// next caller honest.
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

    // Old-row → new-row, monotone by construction.
    let row_map = row_filter
        .map(|keep| {
            debug_assert!(
                keep.windows(2).all(|w| w[0] < w[1]),
                "row filter must ascend"
            );
            RowRemap::from_selection(keep, nrow_full)
        })
        .transpose()?;
    let out_nrow = row_filter.map_or(nrow_full, <[usize]>::len);
    let out_ncol = selected_columns.len();

    let mut out =
        crate::sparse_io::create_sparse_streaming_empty(Some(file_out), Some(backend_out))?;

    let t_stream = std::time::Instant::now();
    let nnz = stream_csc_selection(
        data,
        selected_columns,
        row_map.as_ref(),
        out_nrow,
        &mut *out,
    )?;
    log::info!(
        "streamed {nnz} entries in {out_ncol} columns ({:.1}s)",
        t_stream.elapsed().as_secs_f32()
    );
    out.register_row_names_vec(out_row_names);
    out.register_column_names_vec(out_col_names);
    out.set_metadata(&data.metadata())?;
    Ok((out_nrow, out_ncol, nnz))
}
