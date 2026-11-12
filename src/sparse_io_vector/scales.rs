//! Read-time column scales, and the depth factors they come from.
//!
//! A column's depth is read off its total count, but a few rows detected in
//! nearly every column can take a large share of the counts, and that share
//! can differ from one group of columns to the next. Divided by a total that
//! holds them, every other row of a group whose broad rows take more would
//! read lower: a group effect the normalization itself makes. So
//! [`SparseIoVec::depth_scales`] reads each column's count over the rows no
//! [`RowFilter`] leaves out, over the median of that count across columns.
//!
//! The filters only choose which rows count toward the depth. With
//! [`SparseIoVec::set_column_scales`], every read divides each column by its
//! factor, the left-out rows included; without it, reads return the counts
//! as stored.

use super::*;

/// Rows a column's depth leaves out. Several filters combine by union.
#[derive(Clone, Debug)]
pub enum RowFilter {
    /// One flag per row, `true` to leave the row out.
    Mask(Vec<bool>),
    /// Rows whose names match.
    Names(regex::Regex),
    /// Rows detected in more than `share` of the columns of every group,
    /// `groups` giving each column's group. A row common in one group alone
    /// is kept, since that may be what tells the groups apart.
    Broad { share: f64, groups: Vec<Box<str>> },
}

/// Each column's depth factor, and the rows its depth left out.
#[derive(Clone, Debug)]
pub struct ColumnScales {
    /// Per column: its depth over the median of the positive depths. A
    /// column with no count off the left-out rows gets 1, so its reads stay
    /// as stored rather than blown up.
    pub factor: Vec<f32>,
    /// Per row: `true` when some filter left it out of the depth.
    pub ignored_rows: Vec<bool>,
}

impl SparseIoVec {
    /// The depth factors of the columns, read off the counts as stored, in
    /// blocks of `block_size` columns; refused once scales are set, since
    /// the reads would then be scaled. A [`RowFilter::Broad`] filter takes
    /// one pass over the columns to find its rows before the pass that reads
    /// the depths.
    pub fn depth_scales(
        &self,
        ignore: &[RowFilter],
        block_size: Option<usize>,
    ) -> anyhow::Result<ColumnScales> {
        anyhow::ensure!(
            self.column_scales.is_none(),
            "depth_scales on a scaled vector: read the depths before set_column_scales"
        );
        let (n_rows, n_cols) = (self.num_rows(), self.num_columns());
        let blocks = generate_minibatch_intervals(n_cols, n_rows, block_size);

        let mut ignored = vec![false; n_rows];
        for filter in ignore {
            match filter {
                RowFilter::Mask(mask) => {
                    anyhow::ensure!(
                        mask.len() == n_rows,
                        "row mask has {} flags for {n_rows} rows",
                        mask.len()
                    );
                    ignored.iter_mut().zip(mask).for_each(|(i, &o)| *i |= o);
                }
                RowFilter::Names(re) => {
                    let names = self.row_names()?;
                    ignored
                        .iter_mut()
                        .zip(&names)
                        .for_each(|(i, n)| *i |= re.is_match(n));
                }
                RowFilter::Broad { share, groups } => {
                    let broad = self.broad_rows(*share, groups, &blocks)?;
                    ignored.iter_mut().zip(broad).for_each(|(i, o)| *i |= o);
                }
            }
        }

        let kept: Vec<usize> = (0..n_rows).filter(|&r| !ignored[r]).collect();
        let (_, depth, _, _) =
            crate::qc::collect_column_stat_across_vec(self, Some(&kept), block_size)?.to_f32_vecs();

        Ok(ColumnScales {
            factor: factors(&depth),
            ignored_rows: ignored,
        })
    }

    /// Rows detected in more than `share` of the columns of every group.
    fn broad_rows(
        &self,
        share: f64,
        groups: &[Box<str>],
        blocks: &[(usize, usize)],
    ) -> anyhow::Result<Vec<bool>> {
        let (n_rows, n_cols) = (self.num_rows(), self.num_columns());
        anyhow::ensure!(
            share.is_finite() && (0.0..1.0).contains(&share),
            "broad share {share} is not in [0, 1)"
        );
        anyhow::ensure!(
            groups.len() == n_cols,
            "{} group labels for {n_cols} columns",
            groups.len()
        );
        let mut index: HashMap<&str, usize> = HashMap::default();
        let g: Vec<usize> = groups
            .iter()
            .map(|l| {
                let next = index.len();
                *index.entry(l).or_insert(next)
            })
            .collect();
        let n_groups = index.len();
        let mut size = vec![0usize; n_groups];
        g.iter().for_each(|&k| size[k] += 1);

        // `detected[k * n_rows + r]`: the columns of group `k` holding row `r`.
        let zero = || vec![0u32; n_groups * n_rows];
        let detected = blocks
            .par_iter()
            .try_fold(zero, |mut d, &(lb, ub)| -> anyhow::Result<Vec<u32>> {
                let x = self.read_columns_csc(lb..ub)?;
                for (k, col) in x.col_iter().enumerate() {
                    let dk = &mut d[g[lb + k] * n_rows..][..n_rows];
                    for (&r, &v) in col.row_indices().iter().zip(col.values()) {
                        if v > 0.0 {
                            dk[r] += 1;
                        }
                    }
                }
                Ok(d)
            })
            .try_reduce(zero, |mut a, b| {
                a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
                Ok(a)
            })?;

        Ok((0..n_rows)
            .map(|r| {
                (0..n_groups).all(|k| detected[k * n_rows + r] as f64 > share * size[k] as f64)
            })
            .collect())
    }

    /// Divide every read of column `c` by `factor[c]` from now on; `None`
    /// reads the counts as stored again. Each factor must be positive and
    /// finite, one per column. Pushing another backend afterwards is
    /// refused, since its columns would have none.
    ///
    /// Every read through this vector is scaled, collapse and QC included:
    /// keep a clone without scales for whatever must see the stored counts.
    pub fn set_column_scales(&mut self, factor: Option<Vec<f32>>) -> anyhow::Result<()> {
        if let Some(f) = &factor {
            anyhow::ensure!(
                f.len() == self.num_columns(),
                "{} column scales for {} columns",
                f.len(),
                self.num_columns()
            );
            if let Some((c, v)) = f
                .iter()
                .enumerate()
                .find(|(_, v)| !(v.is_finite() && **v > 0.0))
            {
                anyhow::bail!("column {c} has scale {v}; scales must be positive and finite");
            }
        }
        self.column_scales = factor;
        Ok(())
    }

    /// The factors every read divides by, one per column, when set.
    pub fn column_scales(&self) -> Option<&[f32]> {
        self.column_scales.as_deref()
    }
}

/// Each depth over the median of the positive ones; a column with none gets
/// 1. All ones when no depth is positive.
fn factors(depth: &[f32]) -> Vec<f32> {
    let positive: Vec<f32> = depth.iter().copied().filter(|&d| d > 0.0).collect();
    if positive.is_empty() {
        return vec![1.0; depth.len()];
    }
    let median = median(&positive);
    depth
        .iter()
        .map(|&d| if d > 0.0 { d / median } else { 1.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_io::{create_sparse_from_triplets, SparseIoBackend};

    /// The matched and neighbour reads go through `read_column_offset`.
    #[test]
    fn a_single_column_read_divides_by_its_factor() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.zarr");
        let t = [(0, 0, 4.0), (1, 1, 6.0), (0, 2, 9.0)];
        let mut b = create_sparse_from_triplets(
            &t,
            (2, 3, t.len()),
            Some(path.to_str().unwrap()),
            Some(&SparseIoBackend::Zarr),
        )
        .unwrap();
        b.register_row_names_vec(&["R0".into(), "R1".into()]);
        b.register_column_names_vec(&["c0".into(), "c1".into(), "c2".into()]);
        let mut data = SparseIoVec::new();
        data.push(Arc::from(b), None).unwrap();
        data.set_column_scales(Some(vec![2.0, 3.0, 9.0])).unwrap();
        let (mut off, mut out) = (0, Vec::new());
        for g in [2, 1] {
            data.read_column_offset(g, &mut off, &mut out).unwrap();
        }
        assert_eq!(out, [(0, 0, 1.0), (1, 1, 2.0)]);
    }
}
