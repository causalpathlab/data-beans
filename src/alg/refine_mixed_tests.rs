use super::*;
use std::collections::HashSet;

/// Counts of one cell: `(gene, count)` pairs.
type Cell = Vec<(usize, f32)>;

/// genes x cells from per-cell `(gene, count)` lists.
fn csc_of(cells: &[Cell], ngenes: usize) -> CscMatrix<f32> {
    let mut offsets = vec![0usize];
    let mut rows = Vec::new();
    let mut vals = Vec::new();
    for cell in cells {
        let mut sorted = cell.clone();
        sorted.sort_by_key(|&(g, _)| g);
        for (g, v) in sorted {
            rows.push(g);
            vals.push(v);
        }
        offsets.push(rows.len());
    }
    CscMatrix::try_from_csc_data(ngenes, cells.len(), offsets, rows, vals).unwrap()
}

/// State markers: state `s` expresses genes `4s..4s+4` at 10. Individual
/// `i` expresses its own gene `8 + i` at `offset`. Every cell has gene 12
/// at 5.
fn cell(state: usize, indv: usize, offset: f32) -> Cell {
    let mut c: Cell = (0..4).map(|j| (4 * state + j, 10.0)).collect();
    if offset > 0.0 {
        c.push((8 + indv, offset));
    }
    c.push((12, 5.0));
    c
}

const NGENES: usize = 13;

fn params() -> MixedRefineParams {
    MixedRefineParams {
        max_sweeps: 10,
        ..MixedRefineParams::default()
    }
}

/// One-bit neighbours map through merged labels; unoccupied codes drop out.
#[test]
fn code_neighbours_follow_merged_labels() {
    // codes 0 and 1 merged into one label, code 2 on its own, code 3 empty
    let codes = vec![0, 1, 2, 2];
    let labels = vec![0, 0, 1, 1];
    let nb = CodeNeighbours::new(codes, &labels, 2);
    assert_eq!(nb.candidates(0), &[0, 1]);
    assert_eq!(nb.candidates(2), &[0, 1]);
}

/// Two states across four individuals, a quarter of each group swapped:
/// the groups become state-pure and keep all individuals.
#[test]
fn mixed_groups_become_state_pure() {
    let mut cells = Vec::new();
    let mut batch = Vec::new();
    let mut state = Vec::new();
    for s in 0..2 {
        for i in 0..4 {
            for _ in 0..4 {
                cells.push(cell(s, i, 30.0));
                batch.push(i);
                state.push(s);
            }
        }
    }
    // initial group = state, except the first cell of every (state, indv)
    let mut labels: Vec<usize> = state.clone();
    for c in (0..cells.len()).step_by(4) {
        labels[c] = 1 - labels[c];
    }
    let codes = labels.clone();
    let nb = CodeNeighbours::new(codes, &labels, 1);
    let csc = csc_of(&cells, NGENES);

    let moves = refine_mixed_labels(&csc, &batch, &mut labels, &nb, 2, &params()).unwrap();

    assert_eq!(moves, 8);
    for c in 0..cells.len() {
        assert_eq!(labels[c] == labels[0], state[c] == state[0], "cell {c}");
    }
    for g in 0..2 {
        let indvs: HashSet<usize> = (0..cells.len())
            .filter(|&c| labels[c] == g)
            .map(|c| batch[c])
            .collect();
        assert_eq!(indvs.len(), 4, "group {g} keeps every individual");
    }
}

/// Two groups of one state with opposite individual imbalance and strong
/// individual genes: the offsets are not state, so nothing moves. A score
/// without individual offsets would pull cells to their own individual.
#[test]
fn individual_offsets_do_not_move_cells() {
    let mut cells = Vec::new();
    let mut batch = Vec::new();
    let mut labels = Vec::new();
    for (g, per_indv) in [(0usize, [3usize, 1]), (1, [1, 3])] {
        for (i, &n) in per_indv.iter().enumerate() {
            for _ in 0..n {
                cells.push(cell(0, i, 40.0));
                batch.push(i);
                labels.push(g);
            }
        }
    }
    let before = labels.clone();
    let nb = CodeNeighbours::new(labels.clone(), &labels, 1);
    let csc = csc_of(&cells, NGENES);

    let moves = refine_mixed_labels(&csc, &batch, &mut labels, &nb, 1, &params()).unwrap();

    assert_eq!(moves, 0);
    assert_eq!(labels, before);
}

/// A move that would leave its group with fewer than `min_batches`
/// individuals is vetoed.
#[test]
fn guard_keeps_min_batches() {
    // group 0: state 0 from individuals 0 and 1, state 1 from individual 2
    // group 1: state 1 from individuals 0, 1 and 2
    let spec = [
        (0, 0, 0),
        (0, 1, 0),
        (1, 2, 0),
        (1, 0, 1),
        (1, 1, 1),
        (1, 2, 1),
    ];
    let mut cells = Vec::new();
    let mut batch = Vec::new();
    let mut labels = Vec::new();
    for &(s, i, g) in &spec {
        for _ in 0..3 {
            cells.push(cell(s, i, 0.0));
            batch.push(i);
            labels.push(g);
        }
    }
    let csc = csc_of(&cells, NGENES);
    let start = labels.clone();

    let mut guarded = start.clone();
    let nb = CodeNeighbours::new(start.clone(), &start, 1);
    let moves = refine_mixed_labels(&csc, &batch, &mut guarded, &nb, 3, &params()).unwrap();
    assert_eq!(moves, 2, "two of three leave; the last keeps group 0 at 3");
    let indvs: HashSet<usize> = (0..cells.len())
        .filter(|&c| guarded[c] == 0)
        .map(|c| batch[c])
        .collect();
    assert_eq!(indvs.len(), 3);

    let mut free = start.clone();
    let moves = refine_mixed_labels(&csc, &batch, &mut free, &nb, 2, &params()).unwrap();
    assert_eq!(moves, 3, "with two individuals required, all three leave");
}

/// No sweeps, no moves.
#[test]
fn zero_sweeps_is_identity() {
    let cells: Vec<Cell> = (0..4).map(|c| cell(c % 2, c / 2, 0.0)).collect();
    let batch = vec![0, 0, 1, 1];
    let mut labels = vec![0, 0, 1, 1];
    let nb = CodeNeighbours::new(labels.clone(), &labels, 1);
    let p = MixedRefineParams {
        max_sweeps: 0,
        ..MixedRefineParams::default()
    };
    let moves =
        refine_mixed_labels(&csc_of(&cells, NGENES), &batch, &mut labels, &nb, 1, &p).unwrap();
    assert_eq!(moves, 0);
    assert_eq!(labels, vec![0, 0, 1, 1]);
}

/// Two states across four individuals, written to a small Zarr.
fn two_state_data(dir: &tempfile::TempDir) -> anyhow::Result<(SparseIoVec, Vec<usize>)> {
    use crate::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    let mut batch = Vec::new();
    let mut j = 0u64;
    for s in 0..2 {
        for i in 0..4 {
            for r in 0..6 {
                // vary depth so cells are not identical
                for (g, v) in cell(s, i, 30.0) {
                    triplets.push((g as u64, j, v + (r % 3) as f32));
                }
                batch.push(i);
                j += 1;
            }
        }
    }
    let n = j as usize;
    let path = dir.path().join("mixed.zarr");
    let mut backend = create_sparse_from_triplets(
        &triplets,
        (NGENES, n, triplets.len()),
        Some(path.to_str().unwrap()),
        Some(&SparseIoBackend::Zarr),
    )?;
    let genes: Vec<Box<str>> = (0..NGENES).map(|g| format!("g{g}").into()).collect();
    let cols: Vec<Box<str>> = (0..n).map(|c| format!("c{c}").into()).collect();
    backend.register_row_names_vec(&genes);
    backend.register_column_names_vec(&cols);
    let mut data = SparseIoVec::new();
    data.push(std::sync::Arc::from(backend), None)?;
    Ok((data, batch))
}

/// Groups as sorted column sets, independent of group ids.
fn partition_of(data: &SparseIoVec) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = data
        .take_grouped_columns()
        .expect("groups assigned")
        .iter()
        .filter(|g| !g.is_empty())
        .map(|g| {
            let mut g = g.clone();
            g.sort_unstable();
            g
        })
        .collect();
    groups.sort();
    groups
}

/// Through the data: deterministic, and without sweeps equal to the plain
/// mixed partition.
#[test]
fn refined_partition_through_data() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let (mut data, batch) = two_state_data(&dir)?;
    let proj = data
        .project_columns_with_batch_correction(4, None, Some(&batch))?
        .proj;

    data.partition_columns_to_mixed_groups(&proj, None, &batch, 2, 8)?;
    let plain = partition_of(&data);

    let off = MixedRefineParams {
        max_sweeps: 0,
        ..MixedRefineParams::default()
    };
    data.partition_columns_to_refined_mixed_groups(&proj, None, &batch, 2, 8, &off)?;
    assert_eq!(partition_of(&data), plain);

    data.partition_columns_to_refined_mixed_groups(&proj, None, &batch, 2, 8, &params())?;
    let first = partition_of(&data);
    data.partition_columns_to_refined_mixed_groups(&proj, None, &batch, 2, 8, &params())?;
    assert_eq!(partition_of(&data), first);
    Ok(())
}
