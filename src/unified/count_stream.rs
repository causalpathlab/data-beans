//! [`CellCountStream`] for data-beans' count matrices, so a graph embedding
//! reads their counts without depending on data-beans.

use super::UnifiedData;
use crate::sparse_io_vector::SparseIoVec;
use legume_cell_pseudobulk::CellCountStream;
use std::ops::Range;

/// The backend's rows as they are.
impl CellCountStream for SparseIoVec {
    fn n_cells(&self) -> usize {
        self.num_columns()
    }

    fn n_features(&self) -> usize {
        self.num_rows()
    }

    fn nnz_hint(&self) -> Option<usize> {
        self.num_non_zeros().ok().filter(|&n| n > 0)
    }

    fn for_each_count(
        &self,
        cells: Range<usize>,
        f: &mut dyn FnMut(u32, u32, f32),
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            cells.end <= self.num_columns(),
            "cells {cells:?} past the backend's {} columns",
            self.num_columns()
        );
        let slab = cells.len();
        self.for_each_triplet(cells, slab, |row, local, v| {
            f(row as u32, local as u32, v);
        })
        .map(|_| ())
    }
}

/// On the compact feature axis: backend rows outside it are skipped.
impl CellCountStream for UnifiedData {
    fn n_cells(&self) -> usize {
        UnifiedData::n_cells(self)
    }

    fn n_features(&self) -> usize {
        UnifiedData::n_features(self)
    }

    fn nnz_hint(&self) -> Option<usize> {
        self.count_backend().nnz_hint()
    }

    fn for_each_count(
        &self,
        cells: Range<usize>,
        f: &mut dyn FnMut(u32, u32, f32),
    ) -> anyhow::Result<()> {
        let backend = self.count_backend();
        let identity = self.feature_to_backend_row.len() == backend.num_rows()
            && self
                .feature_to_backend_row
                .iter()
                .enumerate()
                .all(|(i, &r)| i == r);
        if identity {
            return backend.for_each_count(cells, f);
        }
        let mut feature_of_row = vec![u32::MAX; backend.num_rows()];
        for (feature, &row) in self.feature_to_backend_row.iter().enumerate() {
            if row < feature_of_row.len() {
                feature_of_row[row] = feature as u32;
            }
        }
        backend.for_each_count(cells, &mut |row, local, v| {
            let feature = feature_of_row[row as usize];
            if feature != u32::MAX {
                f(feature, local, v);
            }
        })
    }
}
