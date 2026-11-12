//! The coalescer's job is to keep the RETRIEVE COUNT proportional to the data
//! span, not to the number of requested ranges. A subsample keeps every other
//! column, so no two ranges abut — and abutting-only merging degenerated into
//! one retrieve per column, several hundred thousand cache round-trips where a
//! few dozen block reads carry the same bytes.

use super::*;
use std::cell::Cell;

/// A fake backing array 0..n as both values and indices, with a counter on the
/// retrieve closure — the quantity under test.
fn run(tagged: &[(u64, u64, u64)], n: u64) -> (Vec<(u64, u64, f32)>, usize) {
    let calls = Cell::new(0usize);
    let out = coalesce_and_emit(
        tagged,
        n as usize,
        |tag, inner, val| (inner, tag, val),
        |s, e| {
            calls.set(calls.get() + 1);
            let data: Vec<f32> = (s..e).map(|v| v as f32).collect();
            let indices: Vec<u64> = (s..e).collect();
            Ok((data, indices))
        },
    )
    .expect("emit");
    (out, calls.get())
}

#[test]
fn small_gaps_fuse_into_one_retrieve() {
    // 1000 ranges of 60 entries with 60-entry gaps: the alternating-columns
    // shape a subsample produces. Fused, this is a handful of bounded reads;
    // unfused it is 1000.
    let tagged: Vec<(u64, u64, u64)> = (0..1000).map(|k| (k, k * 120, k * 120 + 60)).collect();
    let (out, calls) = run(&tagged, 200_000);
    assert_eq!(out.len(), 60_000, "every requested entry is emitted");
    assert!(
        calls <= 8,
        "1000 near-adjacent ranges must fuse into a few reads, got {calls}"
    );
    // and nothing from the gaps leaks in
    assert!(out.iter().all(|&(inner, _, _)| inner % 120 < 60));
}

#[test]
fn a_huge_gap_still_splits_the_read() {
    // Two ranges a hundred million entries apart must NOT be fused into one
    // retrieve spanning the void.
    let tagged = vec![(0u64, 0u64, 10u64), (1, 100_000_000, 100_000_010)];
    let (out, calls) = run(&tagged, 200_000_000);
    assert_eq!(out.len(), 20);
    assert_eq!(calls, 2, "fusing across the void would read 100M entries");
}

#[test]
fn merged_spans_stay_bounded() {
    // Many small gaps in a long run: fusion must cap the merged span rather
    // than growing one retrieve without limit.
    let tagged: Vec<(u64, u64, u64)> = (0..100_000).map(|k| (k, k * 120, k * 120 + 60)).collect();
    let (out, calls) = run(&tagged, 100_000 * 120);
    assert_eq!(out.len(), 6_000_000);
    assert!(calls >= 2, "a 12M-entry span must not become one retrieve");
    assert!(
        calls <= 64,
        "but it must stay a bounded handful, got {calls}"
    );
}

/// Export a CSC matrix held in memory in `block`-sized reads; the text
/// written and the number of reads made.
fn mtx(
    shape: (usize, usize, usize),
    indptr: &[u64],
    indices: &[u64],
    values: &[f32],
    block: u64,
) -> anyhow::Result<(String, usize)> {
    let calls = Cell::new(0usize);
    let mut out = Vec::new();
    write_mtx(&mut out, shape, indptr, block, |s, e| {
        calls.set(calls.get() + 1);
        let (s, e) = (s as usize, e as usize);
        Ok((values[s..e].to_vec(), indices[s..e].to_vec()))
    })?;
    Ok((String::from_utf8(out)?, calls.get()))
}

const MTX_MAGIC: &str = "%%MatrixMarket matrix coordinate real general\n";

#[test]
fn mtx_blocks_cut_through_columns_and_skip_empty_ones() {
    // 4 x 6; columns 0, 2 and 5 are empty, column 3 holds three entries.
    let indptr = [0u64, 0, 2, 2, 5, 6, 6];
    let indices = [0u64, 3, 0, 1, 2, 3];
    let values = [1.5f32, 2.0, 3.0, 4.0, 5.0, -6.25];
    let body = "1\t2\t1.5\n4\t2\t2\n1\t4\t3\n2\t4\t4\n3\t4\t5\n4\t5\t-6.25\n";
    let expected = format!("{MTX_MAGIC}4\t6\t6\n{body}");
    // whatever the block size, the same text; a block boundary inside a
    // column and one landing exactly on a column's end are both covered
    for (block, reads) in [(1u64, 6usize), (2, 3), (4, 2), (5, 2), (6, 1), (1 << 20, 1)] {
        let (text, calls) = mtx((4, 6, 6), &indptr, &indices, &values, block).unwrap();
        assert_eq!(text, expected, "block {block}");
        assert_eq!(
            calls, reads,
            "block {block}: reads follow the data, not the columns"
        );
    }
}

#[test]
fn mtx_of_an_empty_matrix_is_the_header() {
    let (text, calls) = mtx((3, 2, 0), &[0, 0, 0], &[], &[], 4).unwrap();
    assert_eq!(text, format!("{MTX_MAGIC}3\t2\t0\n"));
    assert_eq!(calls, 0);
}

#[test]
fn mtx_refuses_index_pointers_that_do_not_describe_the_matrix() {
    let indices = [0u64, 1, 2];
    let values = [1.0f32, 2.0, 3.0];
    for (what, indptr, nnz) in [
        ("shorter than ncol + 1", vec![0u64, 3], 3usize),
        ("decreasing", vec![0, 2, 1, 3], 3),
        ("ending short of the declared nnz", vec![0, 1, 2, 2], 3),
        ("ending past the declared nnz", vec![0, 1, 2, 3], 2),
    ] {
        let err = mtx((3, 3, nnz), &indptr, &indices, &values, 2);
        let msg = format!("{:#}", err.expect_err(what));
        assert!(msg.contains("to_mtx_file"), "{what}: {msg}");
    }
}

#[test]
fn mtx_refuses_a_row_index_outside_the_matrix() {
    let err = mtx((2, 1, 2), &[0, 2], &[0, 2], &[1.0, 2.0], 8);
    let msg = format!("{:#}", err.expect_err("row 2 of a 2-row matrix"));
    assert!(msg.contains("to_mtx_file"), "{msg}");
}
