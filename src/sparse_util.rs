use clap::ValueEnum;

pub struct ValuesIndicesPointers<'a> {
    pub values: &'a [f32],
    pub indices: &'a [u64],
    pub indptr: &'a [u64],
}

pub struct CooTripletsShape {
    pub triplets: Vec<(u64, u64, f32)>,
    pub shape: TripletsShape,
}

pub struct TripletsShape {
    pub nrows: usize,
    pub ncols: usize,
    pub nnz: usize,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum IndexPointerType {
    Column,
    Row,
}

pub trait SparseTripletsTraits {
    /// convert sparse pointers into (row, column, value) triplets
    fn to_coo(&self, pointer_type: IndexPointerType) -> anyhow::Result<CooTripletsShape>;
}

/////////////////////
// implementations //
/////////////////////

impl<'a> SparseTripletsTraits for ValuesIndicesPointers<'a> {
    fn to_coo(&self, pointer_type: IndexPointerType) -> anyhow::Result<CooTripletsShape> {
        use rayon::prelude::*;

        let indices = self.indices;
        let indptr = self.indptr;
        let values = self.values;

        let nelem = values.len();
        if nelem != indices.len() {
            return Err(anyhow::anyhow!(
                "`values` and `indices` have different sizes"
            ));
        }
        if indptr.is_empty() {
            return Err(anyhow::anyhow!("`indptr` is empty"));
        }
        let nvectors = indptr.len() - 1;

        // Each compressed vector's entries, clamped to the arrays.
        let spans: Vec<(usize, usize)> = indptr
            .windows(2)
            .map(|w| {
                let end = (w[1] as usize).min(nelem);
                ((w[0] as usize).min(end), end)
            })
            .collect();

        // The spans say where every triplet lands, so the output is allocated
        // once and carved into one slot per vector; workers fill the slots in
        // place, with no per-task buffers to stitch together afterwards.
        let total: usize = spans.iter().map(|&(start, end)| end - start).sum();
        let mut triplets: Vec<(u64, u64, f32)> = vec![(0, 0, 0.0); total];
        let mut slots: Vec<&mut [(u64, u64, f32)]> = Vec::with_capacity(nvectors);
        let mut rest = triplets.as_mut_slice();
        for &(start, end) in &spans {
            let (slot, tail) = rest.split_at_mut(end - start);
            slots.push(slot);
            rest = tail;
        }
        slots
            .into_par_iter()
            .zip(spans.par_iter())
            .enumerate()
            .for_each(|(idx, (slot, &(start, end)))| {
                let j = idx as u64;
                let entries = indices[start..end].iter().zip(&values[start..end]);
                for (out, (&i, &x_ij)) in slot.iter_mut().zip(entries) {
                    *out = match pointer_type {
                        IndexPointerType::Column => (i, j, x_ij),
                        IndexPointerType::Row => (j, i, x_ij),
                    };
                }
            });

        let nnz = triplets.len();

        // Shape is derivable from the compressed layout directly — no need to
        // scan the whole triplets Vec again.
        let max_idx_index = indices.par_iter().copied().max().unwrap_or(0) as usize + 1;
        let (nrows, ncols) = match pointer_type {
            IndexPointerType::Column => (max_idx_index, nvectors),
            IndexPointerType::Row => (nvectors, max_idx_index),
        };

        Ok(CooTripletsShape {
            triplets,
            shape: TripletsShape { nrows, ncols, nnz },
        })
    }
}

/// Sort `triplets` by (column, row), the order the backend writer sorts by
/// first, and sum the values of the entries that share both, leaving one
/// entry per coordinate. The sort is stable, so a coordinate's values are
/// summed in the order they arrived, as the dense readers accumulate them:
/// float sums depend on the order.
pub(crate) fn sum_duplicate_triplets(triplets: &mut Vec<(u64, u64, f32)>) {
    use rayon::prelude::*;
    triplets.par_sort_by_key(|&(r, c, _)| (c, r));
    triplets.dedup_by(|later, kept| {
        let same = (later.0, later.1) == (kept.0, kept.1);
        if same {
            kept.2 += later.2;
        }
        same
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three values on each coordinate whose sum depends on the order:
    /// `(1e8 + 1) - 1e8` is 0 in `f32`, `(1e8 - 1e8) + 1` is 1. Enough
    /// coordinates that a sort has to move entries past their equals.
    #[test]
    fn values_on_one_coordinate_are_summed_in_the_order_they_arrived() {
        let n = 5_000u64;
        let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
        for value in [1e8_f32, 1.0, -1e8] {
            // Descending, so sorting reverses the coordinates.
            triplets.extend((0..n).rev().map(|k| (k % 50, k / 50, value)));
        }
        sum_duplicate_triplets(&mut triplets);
        assert_eq!(triplets.len(), n as usize);
        assert!(triplets
            .windows(2)
            .all(|w| (w[0].1, w[0].0) < (w[1].1, w[1].0)));
        assert!(
            triplets.iter().all(|&(_, _, x)| x == 0.0),
            "summed out of arrival order"
        );
    }

    /// A coordinate listed once keeps its value; one listed several times
    /// holds their sum; the result is ordered by column, then row.
    #[test]
    fn duplicates_are_summed_and_singletons_kept() {
        let (a, b, c) = (0.1_f32, 0.2, 0.3);
        let mut triplets = vec![(1, 0, a), (0, 1, 2.0), (1, 0, b), (0, 0, 5.0), (1, 0, c)];
        sum_duplicate_triplets(&mut triplets);
        assert_eq!(triplets, [(0, 0, 5.0), (1, 0, (a + b) + c), (0, 1, 2.0)]);
    }

    /// The triplets a sequential walk of the compressed vectors gives.
    fn walk(
        values: &[f32],
        indices: &[u64],
        indptr: &[u64],
        pointer_type: IndexPointerType,
    ) -> Vec<(u64, u64, f32)> {
        let mut out = Vec::new();
        for (j, w) in indptr.windows(2).enumerate() {
            let end = (w[1] as usize).min(values.len());
            for k in (w[0] as usize).min(end)..end {
                out.push(match pointer_type {
                    IndexPointerType::Column => (indices[k], j as u64, values[k]),
                    IndexPointerType::Row => (j as u64, indices[k], values[k]),
                });
            }
        }
        out
    }

    #[test]
    fn to_coo_keeps_the_order_of_the_compressed_vectors() {
        // 5,000 vectors of 0..=6 entries, some empty.
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for j in 0..5_000u64 {
            for k in 0..(j * 7 + 3) % 7 {
                indices.push((j + 11 * k) % 97);
                values.push((j * 31 + k) as f32);
            }
            indptr.push(indices.len() as u64);
        }
        for pointer_type in [IndexPointerType::Column, IndexPointerType::Row] {
            let coo = ValuesIndicesPointers {
                values: &values,
                indices: &indices,
                indptr: &indptr,
            }
            .to_coo(pointer_type)
            .unwrap();
            assert_eq!(coo.triplets, walk(&values, &indices, &indptr, pointer_type));
            assert_eq!(coo.shape.nnz, values.len());
            let (major, minor) = match pointer_type {
                IndexPointerType::Column => (coo.shape.ncols, coo.shape.nrows),
                IndexPointerType::Row => (coo.shape.nrows, coo.shape.ncols),
            };
            assert_eq!((major, minor), (5_000, 97));
        }
    }

    #[test]
    fn to_coo_clamps_pointers_past_the_arrays() {
        let values = [1.0f32, 2.0, 3.0];
        let indices = [0u64, 1, 2];
        // The last vector claims two entries the arrays do not hold.
        let indptr = [0u64, 2, 5];
        let coo = ValuesIndicesPointers {
            values: &values,
            indices: &indices,
            indptr: &indptr,
        }
        .to_coo(IndexPointerType::Column)
        .unwrap();
        assert_eq!(coo.triplets, vec![(0, 0, 1.0), (1, 0, 2.0), (2, 1, 3.0)]);
    }
}
