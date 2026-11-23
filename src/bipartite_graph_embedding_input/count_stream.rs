//! The bipartite embedding cells' counts streamed slab by slab on their compact
//! feature axis, for a trainer that reads counts once (graph-embedding
//! samplers, graph builders) without holding the matrix.

use super::BipartiteEmbeddingCells;
use std::ops::Range;

impl BipartiteEmbeddingCells {
    /// Nonzero counts in the backend, or `None` when it cannot say.
    #[must_use]
    pub fn nnz_hint(&self) -> Option<usize> {
        self.count_backend().num_non_zeros().ok().filter(|&n| n > 0)
    }

    /// Call `f(feature, cell - cells.start, count)` for every nonzero count of
    /// cells `cells`, `feature` on the compact axis: backend rows outside it
    /// are skipped.
    pub fn for_each_count(
        &self,
        cells: Range<usize>,
        f: &mut dyn FnMut(u32, u32, f32),
    ) -> anyhow::Result<()> {
        let backend = self.count_backend();
        anyhow::ensure!(
            cells.end <= backend.num_columns(),
            "cells {cells:?} past the backend's {} columns",
            backend.num_columns()
        );
        let identity = self.feature_to_backend_row.len() == backend.num_rows()
            && self
                .feature_to_backend_row
                .iter()
                .enumerate()
                .all(|(i, &r)| i == r);
        let feature_of_row = (!identity).then(|| {
            let mut feature_of_row = vec![u32::MAX; backend.num_rows()];
            for (feature, &row) in self.feature_to_backend_row.iter().enumerate() {
                if row < feature_of_row.len() {
                    feature_of_row[row] = feature as u32;
                }
            }
            feature_of_row
        });
        let slab = cells.len();
        backend
            .for_each_triplet(cells, slab, |row, local, v| {
                let feature = match &feature_of_row {
                    None => row as u32,
                    Some(map) => map[row as usize],
                };
                if feature != u32::MAX {
                    f(feature, local as u32, v);
                }
            })
            .map(|_| ())
    }
}
