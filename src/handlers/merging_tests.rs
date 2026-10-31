//! merge-backend's first tests: a clean merge round-trips, a zipped merge
//! ends as the archive asked for, backends with other rows are refused with
//! an error, and a corrupted input FAILS rather than shipping a silently
//! short matrix.
//!
//! The failure test has two layers of protection to pin. The block-read loop
//! used to swallow a failed read (`.ok()` in a filter_map), dropping the block;
//! since the output nnz is precomputed from file headers, the drop left a gap
//! of fill values in the CSC arrays — corruption, not truncation. The streaming
//! writer's finalize audit now catches that as appended != declared, so the
//! run errors either way; propagating the read's own error on top of it turns
//! a confusing post-hoc audit failure into the actual cause.

use super::*;

fn write_fixture(path: &str, cols: usize, seed_val: f32) -> anyhow::Result<()> {
    let mut arr = ndarray::Array2::<f32>::zeros((3, cols));
    for r in 0..3 {
        for c in 0..cols {
            arr[(r, c)] = seed_val + (r * cols + c) as f32;
        }
    }
    let mut data = create_sparse_from_ndarray(&arr, Some(path), Some(&SparseIoBackend::Zarr))?;
    let rows: Vec<Box<str>> = (0..3).map(|r| format!("g{r}").into()).collect();
    let names: Vec<Box<str>> = (0..cols)
        .map(|c| format!("{seed_val}c{c}").into())
        .collect();
    data.register_row_names_vec(&rows);
    data.register_column_names_vec(&names);
    Ok(())
}

/// Two inputs in a fresh directory: `a.zarr` (4 columns) and `b.zarr` (3),
/// both over rows g0..g2.
fn two_fixtures() -> anyhow::Result<(tempfile::TempDir, String, String)> {
    let dir = tempfile::tempdir()?;
    let path = |name: &str| dir.path().join(name).to_string_lossy().into_owned();
    let (a, b) = (path("a.zarr"), path("b.zarr"));
    write_fixture(&a, 4, 100.0)?;
    write_fixture(&b, 3, 900.0)?;
    Ok((dir, a, b))
}

fn merge_args(files: &[&str], output: &str) -> MergeBackendArgs {
    MergeBackendArgs {
        data_files: files.iter().map(|f| Box::from(*f)).collect(),
        backend: SparseIoBackend::Zarr,
        output: output.into(),
        zip: false,
        common_rows: false,
        do_squeeze: false,
        row_nnz_cutoff: 0,
        column_nnz_cutoff: 0,
        block_size: None,
    }
}

#[test]
fn a_clean_merge_round_trips() -> anyhow::Result<()> {
    let (dir, a, b) = two_fixtures()?;
    let out = dir.path().join("m");
    run_merge_backend(&merge_args(&[&a, &b], out.to_str().expect("utf8")))?;
    let merged = open_sparse_matrix(
        &format!("{}.zarr", out.to_str().expect("utf8")),
        &SparseIoBackend::Zarr,
    )?;
    assert_eq!(merged.num_rows(), Some(3));
    assert_eq!(merged.num_columns(), Some(7));
    assert_eq!(merged.num_non_zeros(), Some(21));
    Ok(())
}

#[test]
fn a_zipped_merge_ends_as_the_archive_with_its_batches_beside() -> anyhow::Result<()> {
    let (dir, a, b) = two_fixtures()?;
    let out = dir.path().join("m");
    let args = MergeBackendArgs {
        zip: true,
        ..merge_args(&[&a, &b], out.to_str().expect("utf8"))
    };
    run_merge_backend(&args)?;
    // `-o m` asks for m.zarr.zip; the working m.zarr used to be left instead.
    assert!(dir.path().join("m.zarr.zip").is_file());
    assert!(!dir.path().join("m.zarr").exists());
    assert!(dir.path().join("m.batch.gz").is_file());

    // A suffixed output keeps its batches beside it under the bare name.
    let suffixed = dir.path().join("n.zarr.zip");
    run_merge_backend(&MergeBackendArgs {
        output: suffixed.to_str().expect("utf8").into(),
        ..args
    })?;
    assert!(dir.path().join("n.zarr.zip").is_file());
    assert!(!dir.path().join("n.zarr").exists());
    assert!(dir.path().join("n.batch.gz").is_file());

    // Each column labelled with the input it came from, in order.
    let labels = read_lines(dir.path().join("m.batch.gz").to_str().expect("utf8"))?;
    let want: Vec<Box<str>> = ["a"; 4]
        .into_iter()
        .chain(["b"; 3])
        .map(Box::from)
        .collect();
    assert_eq!(labels, want);
    Ok(())
}

#[test]
fn an_output_that_is_an_input_is_refused_before_anything_is_cleared() -> anyhow::Result<()> {
    let (dir, a, b) = two_fixtures()?;
    // `-o a --no-zip` is a.zarr itself, which used to be deleted first.
    let out = dir.path().join("a");
    let err = run_merge_backend(&merge_args(&[&a, &b], out.to_str().expect("utf8"))).unwrap_err();
    assert!(format!("{err:#}").contains("is the input"), "{err:#}");
    assert!(open_sparse_matrix(&a, &SparseIoBackend::Zarr)?.num_columns() == Some(4));
    Ok(())
}

#[test]
fn backends_with_other_rows_are_refused_not_panicked_on() -> anyhow::Result<()> {
    let (dir, a, b) = two_fixtures()?;
    let mut other = open_sparse_matrix(&b, &SparseIoBackend::Zarr)?;
    let rows: Vec<Box<str>> = ["x0", "x1", "x2"].map(Box::from).into();
    other.register_row_names_vec(&rows);
    drop(other);
    let out = dir.path().join("m");
    let err = run_merge_backend(&merge_args(&[&a, &b], out.to_str().expect("utf8"))).unwrap_err();
    assert!(format!("{err:#}").contains("other row names"), "{err:#}");
    assert!(!dir.path().join("m.zarr").exists());
    Ok(())
}

#[test]
fn a_merge_keeps_the_metadata_its_inputs_share() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let (a, b) = (dir.path().join("a.zarr"), dir.path().join("b.zarr"));
    for (path, sample) in [(&a, "s1"), (&b, "s2")] {
        let path = path.to_str().expect("utf8");
        write_fixture(path, 2, 1.0)?;
        let mut data = open_sparse_matrix(path, &SparseIoBackend::Zarr)?;
        data.set_meta(meta::SAMPLE, sample)?;
        data.set_meta(meta::PRODUCER, "p1")?;
    }
    let out = dir.path().join("m");
    run_merge_backend(&merge_args(
        &[a.to_str().expect("utf8"), b.to_str().expect("utf8")],
        out.to_str().expect("utf8"),
    ))?;
    let merged = open_sparse_matrix(
        &format!("{}.zarr", out.to_str().expect("utf8")),
        &SparseIoBackend::Zarr,
    )?;
    // Samples differ, so `sample` goes; the producer is the same.
    assert_eq!(merged.meta(meta::SAMPLE), None);
    assert_eq!(merged.meta(meta::PRODUCER).as_deref(), Some("p1"));
    Ok(())
}

#[test]
fn a_corrupted_input_fails_instead_of_shipping_a_short_matrix() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let a = dir.path().join("a.zarr");
    let b = dir.path().join("b.zarr");
    write_fixture(a.to_str().expect("utf8"), 4, 100.0)?;
    write_fixture(b.to_str().expect("utf8"), 3, 900.0)?;

    // Garbage over one data chunk of the second input: the read of that block
    // fails to decode, which used to be swallowed.
    let mut corrupted = false;
    for entry in walk(&b.join("by_column").join("data"))? {
        if entry.file_name().is_some_and(|n| n != "zarr.json") && entry.is_file() {
            std::fs::write(&entry, b"not a chunk")?;
            corrupted = true;
            break;
        }
    }
    assert!(
        corrupted,
        "fixture layout changed; no chunk found to corrupt"
    );

    let out = dir.path().join("m");
    let result = run_merge_backend(&merge_args(
        &[a.to_str().expect("utf8"), b.to_str().expect("utf8")],
        out.to_str().expect("utf8"),
    ));
    assert!(
        result.is_err(),
        "a merge that cannot read an input must fail, not write fewer entries \
         than its own header declares"
    );
    Ok(())
}

fn walk(root: &std::path::Path) -> anyhow::Result<Vec<std::path::PathBuf>> {
    let mut out = Vec::new();
    if root.is_dir() {
        for e in std::fs::read_dir(root)? {
            let p = e?.path();
            if p.is_dir() {
                out.extend(walk(&p)?);
            } else {
                out.push(p);
            }
        }
    }
    Ok(out)
}
