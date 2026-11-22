//! The data read once: each pseudobulk's raw counts summed, with its batch,
//! span label, coarsest group, size and mean matching coordinates. Every step
//! of the fold after this runs on these sums, so nodes, tree depths and
//! swaps never read the data again.

use crate::sparse_io_vector::SparseIoVec;
use nalgebra::DMatrix;
use rayon::prelude::*;

/// Per pseudobulk: what it is and what it holds.
pub(crate) struct PseudobulkSums {
    /// Its raw counts, `(gene, sum)` sorted by gene.
    pub counts: Vec<Vec<(usize, f32)>>,
    pub batch: Vec<usize>,
    /// Its span label (its units' donor, or batch).
    pub span_label: Vec<usize>,
    /// Its units' coarsest collapse group (the first unit's).
    pub coarsest: Vec<usize>,
    /// Its units' mean matching coordinates, components × pseudobulks.
    pub coords: DMatrix<f32>,
    /// Each unit's pseudobulk (`None`: the unit is in none).
    pub of_unit: Vec<Option<usize>>,
    pub n_genes: usize,
}

/// What [`sum_by_pseudobulk`] reads, per unit.
pub(crate) struct PseudobulkKeys<'a> {
    /// Each unit's pseudobulk key (`None`: leave the unit out). Units with
    /// one key form one pseudobulk.
    pub key: &'a [Option<(usize, usize, usize)>],
    pub batch: &'a [usize],
    pub span_label: &'a [usize],
    pub coarsest: &'a [usize],
    /// Matching coordinates, components × units.
    pub coords: &'a DMatrix<f32>,
}

/// Pseudobulks summed per job.
const PSEUDOBULKS_PER_JOB: usize = 64;

/// The pseudobulks of `keys`, summed from `data` in one pass: jobs own whole
/// pseudobulks, so each reads only its units and owns what it writes.
pub(crate) fn sum_by_pseudobulk(
    data: &SparseIoVec,
    keys: &PseudobulkKeys<'_>,
) -> anyhow::Result<PseudobulkSums> {
    let n_genes = data.num_rows();
    let (of_unit, distinct) = super::compact_present(keys.key);
    let n_pb = distinct.len();
    let mut units_of: Vec<Vec<usize>> = vec![Vec::new(); n_pb];
    for (u, pb) in of_unit.iter().enumerate() {
        if let Some(pb) = *pb {
            units_of[pb].push(u);
        }
    }
    let counts: Vec<Vec<(usize, f32)>> = units_of
        .par_chunks(PSEUDOBULKS_PER_JOB)
        .map(|chunk| -> anyhow::Result<Vec<Vec<(usize, f32)>>> {
            let cols: Vec<usize> = chunk.iter().flatten().copied().collect();
            let yy = data.read_columns_csc(cols.iter().copied())?;
            let mut out = Vec::with_capacity(chunk.len());
            let mut dense = vec![0f32; n_genes];
            let mut at = 0;
            for units in chunk {
                for y in yy.col_iter().skip(at).take(units.len()) {
                    for (&g, &x) in y.row_indices().iter().zip(y.values()) {
                        dense[g] += x;
                    }
                }
                at += units.len();
                let nz: Vec<(usize, f32)> = dense
                    .iter()
                    .enumerate()
                    .filter(|(_, &v)| v != 0.0)
                    .map(|(g, &v)| (g, v))
                    .collect();
                dense.iter_mut().for_each(|v| *v = 0.0);
                out.push(nz);
            }
            Ok(out)
        })
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();

    let dim = keys.coords.nrows();
    let mut coords = DMatrix::<f32>::zeros(dim, n_pb);
    for (pb, units) in units_of.iter().enumerate() {
        for &u in units {
            coords.column_mut(pb).add_assign(&keys.coords.column(u));
        }
        coords
            .column_mut(pb)
            .scale_mut(1.0 / units.len().max(1) as f32);
    }
    let first = |pb: usize| units_of[pb][0];
    Ok(PseudobulkSums {
        counts,
        batch: (0..n_pb).map(|pb| keys.batch[first(pb)]).collect(),
        span_label: (0..n_pb).map(|pb| keys.span_label[first(pb)]).collect(),
        coarsest: (0..n_pb).map(|pb| keys.coarsest[first(pb)]).collect(),
        coords,
        of_unit,
        n_genes,
    })
}

impl PseudobulkSums {
    pub(crate) fn len(&self) -> usize {
        self.counts.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }
}

use std::ops::AddAssign;
