//! A stored or in-flight index that is out of range must fail where it is
//! found, naming the value, rather than panic later or drop an entry. The
//! corrupt values below carry one high bit on top of a valid index, the shape
//! a single flipped bit takes.

use data_beans::sparse_io::*;

const HIGH_BIT: u64 = 1 << 62;

type Triplets = Vec<(u64, u64, f32)>;
type Backend = Box<dyn SparseIo<IndexIter = Vec<usize>>>;

/// 4 x 5 with an empty interior column.
fn reference() -> (Triplets, (usize, usize, usize)) {
    let triplets = vec![
        (0u64, 0u64, 1.0f32),
        (2, 0, 2.0),
        (1, 1, 3.0),
        (0, 3, 4.0),
        (1, 3, 5.0),
        (3, 3, 6.0),
        (2, 4, 7.0),
    ];
    (triplets, (4, 5, 7))
}

fn build(
    dir: &tempfile::TempDir,
    name: &str,
    triplets: &[(u64, u64, f32)],
    shape: (usize, usize, usize),
) -> anyhow::Result<(String, Backend)> {
    let path = dir.path().join(name).to_str().expect("utf8").to_string();
    let backend =
        create_sparse_from_triplets(triplets, shape, Some(&path), Some(&SparseIoBackend::Zarr))?;
    Ok((path, backend))
}

#[test]
fn a_triplet_column_past_ncol_is_named_not_dropped() -> anyhow::Result<()> {
    let (mut triplets, shape) = reference();
    triplets[3].1 += HIGH_BIT;
    let dir = tempfile::tempdir()?;
    let err = build(&dir, "c.zarr", &triplets, shape)
        .err()
        .expect("must fail");
    let msg = format!("{err:#}");
    assert!(msg.contains("0x4000000000000003"), "{msg}");
    assert!(msg.contains("outside the 4 x 5 matrix"), "{msg}");
    Ok(())
}

#[test]
fn a_triplet_row_past_nrow_is_named_not_dropped() -> anyhow::Result<()> {
    let (mut triplets, shape) = reference();
    triplets[5].0 += HIGH_BIT;
    let dir = tempfile::tempdir()?;
    let err = build(&dir, "r.zarr", &triplets, shape)
        .err()
        .expect("must fail");
    let msg = format!("{err:#}");
    assert!(msg.contains("0x4000000000000003"), "{msg}");
    Ok(())
}

/// Write a clean matrix, then overwrite one stored CSC row index.
fn corrupt_csc(dir: &tempfile::TempDir) -> anyhow::Result<String> {
    let (triplets, shape) = reference();
    let (path, mut backend) = build(dir, "s.zarr", &triplets, shape)?;
    // position 2 holds row 1 of column 1
    backend.cs_write_u64(CsKey::CscIndices, 2, &[1 + HIGH_BIT])?;
    drop(backend);
    Ok(path)
}

#[test]
fn a_bad_stored_index_fails_the_preload() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = corrupt_csc(&dir)?;
    let mut data = open_sparse_matrix(&path, &SparseIoBackend::Zarr)?;
    let msg = format!("{:#}", data.preload_columns().unwrap_err());
    assert!(msg.contains("preload_columns"), "{msg}");
    assert!(msg.contains("position 2"), "{msg}");
    assert!(msg.contains("0x4000000000000001"), "{msg}");
    Ok(())
}

#[test]
fn a_bad_stored_index_fails_a_cold_read() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = corrupt_csc(&dir)?;
    let data = open_sparse_matrix(&path, &SparseIoBackend::Zarr)?;
    // the clean column still reads
    let (_, _, t) = data.read_triplets_by_columns(vec![0])?;
    assert_eq!(t.len(), 2);
    for read in [
        data.read_triplets_by_columns(vec![1]).map(|_| ()),
        data.read_triplets_by_columns((0..5).collect()).map(|_| ()),
        data.read_triplets_by_single_column(1).map(|_| ()),
    ] {
        let msg = format!("{:#}", read.unwrap_err());
        assert!(msg.contains("0x4000000000000001"), "{msg}");
    }
    Ok(())
}

#[test]
fn out_of_range_column_requests_keep_their_behaviour() -> anyhow::Result<()> {
    let (triplets, shape) = reference();
    let dir = tempfile::tempdir()?;
    let (path, _) = build(&dir, "o.zarr", &triplets, shape)?;
    let mut data = open_sparse_matrix(&path, &SparseIoBackend::Zarr)?;
    // the cold multi-column read has always skipped a column past the end
    let (_, ncol_out, t) = data.read_triplets_by_columns(vec![0, 5])?;
    assert_eq!((ncol_out, t.len()), (2, 2));
    // these used to panic on the indptr lookup; now they are errors
    assert!(data.read_triplets_by_single_column(5).is_err());
    data.preload_columns()?;
    assert!(data.read_triplets_by_columns(vec![0, 5]).is_err());
    Ok(())
}

/// A backend with CSC arrays only, the shape some older files have.
fn csc_only(dir: &tempfile::TempDir) -> anyhow::Result<String> {
    let (triplets, shape) = reference();
    let path = dir
        .path()
        .join("csc.zarr")
        .to_str()
        .expect("utf8")
        .to_string();
    let mut out = create_sparse_streaming_empty(Some(&path), Some(&SparseIoBackend::Zarr))?;
    out.begin_streaming_csc(shape)?;
    let mut colptr = Vec::new();
    let (mut rows, mut vals) = (Vec::new(), Vec::new());
    for c in 0..shape.1 as u64 {
        colptr.push(rows.len() as u64);
        for &(i, j, x) in &triplets {
            if j == c {
                rows.push(i);
                vals.push(x);
            }
        }
    }
    out.append_csc_slab(0, 0, &colptr, &rows, &vals)?;
    out.finalize_streaming_csc()?;
    Ok(path)
}

#[test]
fn row_reads_without_a_row_index() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = csc_only(&dir)?;
    let data = open_sparse_matrix(&path, &SparseIoBackend::Zarr)?;
    // reads that touch no row slot fail as they always have, on the missing
    // row arrays, not on the new index checks
    for req in [vec![], vec![4, 9]] {
        let msg = format!("{:#}", data.read_triplets_by_rows(req).unwrap_err());
        assert!(msg.contains("array metadata is missing"), "{msg}");
    }
    // a real row read used to panic on the empty indptr; now it says why
    let msg = format!("{:#}", data.read_triplets_by_rows(vec![0]).unwrap_err());
    assert!(msg.contains("no row index"), "{msg}");
    // columns are unaffected
    let (_, _, t) = data.read_triplets_by_columns((0..5).collect())?;
    assert_eq!(t.len(), 7);
    Ok(())
}
