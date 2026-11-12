//! Stream a column selection (and an optional row selection or renumbering)
//! of a sparse backend into a fresh backend without materialising the
//! survivors. Shared by the `split` / `subsample` / `subset-*` CLI handlers,
//! by the in-place subset and reorder of [`SparseIo`], and by downstream
//! crates (faba's `qc`) that filter a matrix by cells and features.

use crate::sparse_io::SparseIo;
use rayon::prelude::*;

/// Every index of `selection` is below `n` and listed once; `what` names the
/// side ("row", "column") in the error.
pub(crate) fn distinct_within(selection: &[usize], n: usize, what: &str) -> anyhow::Result<()> {
    let mut seen = vec![false; n];
    for &i in selection {
        let slot = seen
            .get_mut(i)
            .ok_or_else(|| anyhow::anyhow!("{what} {i} is past the {n} {what}s"))?;
        anyhow::ensure!(
            !std::mem::replace(slot, true),
            "the {what} selection repeats {what} {i}"
        );
    }
    Ok(())
}

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
    ///
    /// # Errors
    /// If `keep` lists a row past `nrow_full`, or one row twice.
    pub fn keeping(keep: &[usize], nrow_full: usize) -> anyhow::Result<Self> {
        distinct_within(keep, nrow_full, "row")?;
        let mut to = vec![None; nrow_full];
        for (new, &old) in keep.iter().enumerate() {
            to[old] = Some(new as u64);
        }
        Ok(Self {
            to,
            nrow: keep.len(),
        })
    }

    /// The map that puts the rows named `old_names` in the order of `order`:
    /// output row `i` is the input row named `order[i]`, empty when no input
    /// row has that name; an input row whose name `order` does not list is
    /// dropped.
    ///
    /// # Errors
    /// If `order` lists a name twice (its second position would stay empty),
    /// or two input rows carry one listed name (they would share a position).
    pub fn by_names(old_names: &[Box<str>], order: &[Box<str>]) -> anyhow::Result<Self> {
        let place = crate::sparse_io::build_name2index_map(order);
        if place.len() != order.len() {
            // The map keeps the last position of a repeated name.
            let (first, name) = order
                .iter()
                .enumerate()
                .find(|&(i, name)| place[name] != i)
                .expect("a shorter map means a repeated name");
            anyhow::bail!(
                "the row order lists `{name}` twice (positions {first} and {})",
                place[name]
            );
        }
        // `claimed[new]` is the input row that took the output position.
        let mut claimed: Vec<Option<usize>> = vec![None; order.len()];
        let to = old_names
            .iter()
            .enumerate()
            .map(|(old, name)| {
                let Some(&new) = place.get(name) else {
                    return Ok(None);
                };
                if let Some(first) = claimed[new].replace(old) {
                    anyhow::bail!(
                        "rows {first} and {old} are both named `{name}`; \
                         they cannot share one position"
                    );
                }
                Ok(Some(new as u64))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self {
            to,
            nrow: order.len(),
        })
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

/// Lay the triplets of one block read out as a CSC slab of `ncol` columns:
/// `(colptr, row indices, values)`, where `colptr[j]` is the slab-local
/// offset of column `j`'s first entry (the sentinel after the last column is
/// the writer's to add).
///
/// The triplets carry slab-local column ids. With `rows`, each row is
/// renumbered by `rows[old]` and an entry whose row maps to `None` is
/// dropped. Entries are counted and placed by column, so those of one column
/// keep their arrival order whatever order the columns arrive in; a column
/// whose rows then do not ascend is sorted by row, stably, so entries sharing
/// a row stay in arrival order too.
pub(crate) fn csc_slab(
    triplets: Vec<(u64, u64, f32)>,
    ncol: usize,
    rows: Option<&[Option<u64>]>,
) -> (Vec<u64>, Vec<u64>, Vec<f32>) {
    let new_row = |r: u64| match rows {
        None => Some(r),
        Some(to) => to[r as usize],
    };

    let mut colptr = vec![0u64; ncol + 1];
    for &(r, c_local, _) in &triplets {
        if new_row(r).is_some() {
            colptr[c_local as usize + 1] += 1;
        }
    }
    for c in 0..ncol {
        colptr[c + 1] += colptr[c];
    }
    let slab_len = colptr[ncol] as usize;
    let mut row_indices = vec![0u64; slab_len];
    let mut values = vec![0f32; slab_len];
    let mut cursor = colptr[..ncol].to_vec();
    for (r, c_local, x) in triplets {
        if let Some(new_r) = new_row(r) {
            let at = &mut cursor[c_local as usize];
            row_indices[*at as usize] = new_r;
            values[*at as usize] = x;
            *at += 1;
        }
    }

    let mut column: Vec<(u64, f32)> = Vec::new();
    for w in colptr.windows(2) {
        let span = w[0] as usize..w[1] as usize;
        if row_indices[span.clone()].is_sorted() {
            continue;
        }
        column.clear();
        column.extend(
            row_indices[span.clone()]
                .iter()
                .copied()
                .zip(values[span.clone()].iter().copied()),
        );
        column.sort_by_key(|&(r, _)| r);
        for (k, &(r, x)) in span.zip(&column) {
            row_indices[k] = r;
            values[k] = x;
        }
    }

    colptr.truncate(ncol);
    (colptr, row_indices, values)
}

/// What a selection writes, settled before any output exists: the output's
/// shape and the column blocks its slabs are read in.
pub(crate) struct SelectionPlan {
    nrow: usize,
    nnz: u64,
    /// `[lb, ub)` positions in the column selection, one slab each.
    slabs: Vec<(usize, usize)>,
}

/// Check a selection of `src` (columns `old_cols`, rows renumbered by `rows`)
/// and size its output, reading nothing but counts.
///
/// The output's nnz comes from the resident indptr when every row survives,
/// or when every column does (the by-row indptr then counts the rows kept);
/// otherwise from one counting pass that keeps counts, never entries.
///
/// # Errors
/// If a column of `old_cols` is past the data's columns, or `rows` does not
/// map the data's rows onto distinct output rows.
pub(crate) fn plan_csc_selection<S>(
    src: &S,
    old_cols: &[usize],
    rows: Option<&RowMap>,
) -> anyhow::Result<SelectionPlan>
where
    S: SparseIo + Sync + ?Sized,
{
    plan_within(src, old_cols, rows, crate::sparse_io::SLAB_BUDGET_BYTES)
}

/// [`plan_csc_selection`] with `slab_budget` bytes of triplets to a slab.
fn plan_within<S>(
    src: &S,
    old_cols: &[usize],
    rows: Option<&RowMap>,
    slab_budget: usize,
) -> anyhow::Result<SelectionPlan>
where
    S: SparseIo + Sync + ?Sized,
{
    use crate::sparse_io::TRIPLET_BYTES;
    use legume_numeric::matrix::utils::byte_budget_intervals;

    let nrow_src = src
        .num_rows()
        .ok_or_else(|| anyhow::anyhow!("backend has no `nrow`"))?;
    if let Some(rows) = rows {
        rows.check(nrow_src)?;
    }
    let ncol_src = src
        .num_columns()
        .ok_or_else(|| anyhow::anyhow!("backend has no `ncol`"))?;

    // Stored nnz of each selected column, off the resident indptr: exact when
    // every row survives, an upper bound under a row filter.
    let stored_nnz: Vec<u64> = old_cols
        .iter()
        .map(|&c| {
            anyhow::ensure!(c < ncol_src, "column {c} is past the {ncol_src} columns");
            src.column_nnz(c)
                .ok_or_else(|| anyhow::anyhow!("no indptr entry for column {c}"))
        })
        .collect::<anyhow::Result<_>>()?;

    // The writer needs the exact total up front. Without a row filter it is
    // the stored total. With one, and every column kept, the resident row
    // index gives it for free. Otherwise one counting pass.
    let every_column_once =
        old_cols.len() == ncol_src && distinct_within(old_cols, ncol_src, "column").is_ok();
    let (nnz, slab_nnz): (u64, Vec<u64>) = match rows {
        None => (stored_nnz.iter().sum(), stored_nnz),
        Some(remap) => match every_column_once
            .then(|| nnz_of_kept_rows(src, remap))
            .flatten()
        {
            Some(total) => (total, stored_nnz),
            None => {
                // Blocks are read all at once, one to a thread, so each takes
                // a thread's share of the slab budget (never under a triplet,
                // so a block always holds a column).
                let share = (slab_budget / rayon::current_num_threads().max(1)).max(TRIPLET_BYTES);
                let counts: Vec<u64> = byte_budget_intervals(&stored_nnz, share, TRIPLET_BYTES)
                    .par_iter()
                    .map(|&(lb, ub)| -> anyhow::Result<Vec<u64>> {
                        let mut counts = vec![0u64; ub - lb];
                        let (_, _, triplets) = src
                            .read_triplets_by_columns(old_cols[lb..ub].iter().copied().collect())?;
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
        },
    };

    Ok(SelectionPlan {
        nrow: rows.map_or(nrow_src, |r| r.nrow),
        nnz,
        // Slabs bounded by bytes of triplets read, not by a column count.
        slabs: byte_budget_intervals(&slab_nnz, slab_budget, TRIPLET_BYTES),
    })
}

/// Write the selection `plan` was made for as the whole content of `out`, an
/// empty streaming backend: the column index slab by slab, then the row
/// index transposed from it. Names and metadata are left to the caller.
/// Returns the output's `(rows, columns, non-zeros)`.
pub(crate) fn write_csc_selection<S, O>(
    src: &S,
    old_cols: &[usize],
    rows: Option<&RowMap>,
    plan: &SelectionPlan,
    out: &mut O,
) -> anyhow::Result<(usize, usize, usize)>
where
    S: SparseIo + Sync + ?Sized,
    O: SparseIo + ?Sized,
{
    let shape = (plan.nrow, old_cols.len(), plan.nnz as usize);
    out.begin_streaming_csc(shape)?;

    let mut nnz_offset = 0u64;
    for &(lb, ub) in &plan.slabs {
        // One block read per slab, never a read per column: a single-column
        // read pays the cached-subset machinery per call, which dominates at
        // hundreds of thousands of columns. The block read numbers its
        // columns from the block's first, rows ascending within each; a row
        // map that reorders is sorted for within each column.
        let (_, _, triplets) =
            src.read_triplets_by_columns(old_cols[lb..ub].iter().copied().collect())?;
        let (colptr, row_indices, values) =
            csc_slab(triplets, ub - lb, rows.map(|r| r.to.as_slice()));
        out.append_csc_slab(lb as u64, nnz_offset, &colptr, &row_indices, &values)?;
        nnz_offset += values.len() as u64;
    }

    out.finalize_streaming_csc()?;
    out.build_csr_from_csc_streaming()?;
    Ok(shape)
}

/// Write columns `old_cols` of `src`, in that order and with rows renumbered
/// by `rows` (`None` keeps every row), as the whole content of `out`:
/// [`plan_csc_selection`], then [`write_csc_selection`]. A caller that
/// creates `out` for the selection plans first, so a refused selection
/// leaves nothing behind.
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
    let plan = plan_csc_selection(src, old_cols, rows)?;
    write_csc_selection(src, old_cols, rows, &plan, out)
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
///
/// # Errors
/// If a selected column is past the data's columns, or `rows` does not map
/// the data's rows onto distinct output rows; `file_out` is then not created.
pub fn stream_mapped_selection(
    data: &dyn SparseIo<IndexIter = Vec<usize>>,
    selected_columns: &[usize],
    rows: Option<&RowMap>,
    out_row_names: &[Box<str>],
    out_col_names: &[Box<str>],
    file_out: &str,
    backend_out: &crate::sparse_io::SparseIoBackend,
) -> anyhow::Result<(usize, usize, usize)> {
    let plan = plan_csc_selection(data, selected_columns, rows)?;
    let mut out =
        crate::sparse_io::create_sparse_streaming_empty(Some(file_out), Some(backend_out))?;

    let t_stream = std::time::Instant::now();
    let shape = write_csc_selection(data, selected_columns, rows, &plan, &mut *out)?;
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

#[cfg(test)]
#[path = "column_subset_tests.rs"]
mod tests;
