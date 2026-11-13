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
        view: PlanArgs::batch(),
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

/// A backend's dense matrix, row names and column names.
type Merged = (nalgebra::DMatrix<f32>, Vec<Box<str>>, Vec<Box<str>>);

/// The merged backend `-o out --no-zip` left.
fn read_merged(out: &std::path::Path) -> anyhow::Result<Merged> {
    let merged = open_sparse_matrix(
        &format!("{}.zarr", out.to_str().expect("utf8")),
        &SparseIoBackend::Zarr,
    )?;
    let ncol = merged.num_columns().expect("ncol");
    Ok((
        merged.read_columns_dmatrix((0..ncol).collect())?,
        merged.row_names()?,
        merged.column_names()?,
    ))
}

#[test]
fn a_merge_keeps_every_value_under_its_row_and_column() -> anyhow::Result<()> {
    let (dir, a, b) = two_fixtures()?;
    // The second input holds its rows in another order, and one of its own.
    let mut other = open_sparse_matrix(&b, &SparseIoBackend::Zarr)?;
    let rows: Vec<Box<str>> = ["g2", "x", "g0"].map(Box::from).into();
    other.register_row_names_vec(&rows);
    drop(other);

    // Rows g0 and g2 of the first (values 100 + 4r + c), then rows 2 and 0
    // of the second (900 + 3r + c).
    let want = |r: usize, c: usize| -> f32 {
        match (r, c < 4) {
            (0, true) => 100.0 + c as f32,
            (_, true) => 108.0 + c as f32,
            (0, false) => 906.0 + (c - 4) as f32,
            (_, false) => 900.0 + (c - 4) as f32,
        }
    };
    // Two threads, whatever the machine has, so the windows are of two
    // slabs each. In one slab to a batch (the whole budget); in slabs of one
    // column, which the budget of one triplet to a thread cuts: two windows
    // for the first batch's four columns; and in slabs of two columns.
    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build()?;
    for (name, budget, block_size) in [
        ("whole", SLAB_BUDGET_BYTES, None),
        ("cut", 2 * TRIPLET_BYTES, None),
        ("two", SLAB_BUDGET_BYTES, Some(2)),
    ] {
        let out = dir.path().join(name);
        let args = MergeBackendArgs {
            common_rows: true,
            block_size,
            ..merge_args(&[&a, &b], out.to_str().expect("utf8"))
        };
        pool.install(|| merge_backend_within(&args, true, false, budget))?;
        let (dense, rows, columns) = read_merged(&out)?;
        assert_eq!(rows, ["g0", "g2"].map(Box::from));
        assert_eq!(columns.len(), 7);
        assert_eq!(columns[0].as_ref(), "100c0@a");
        assert_eq!(columns[6].as_ref(), "900c2@b");
        for r in 0..2 {
            for c in 0..7 {
                assert_eq!(dense[(r, c)], want(r, c), "{name}: ({r}, {c})");
            }
        }
    }
    Ok(())
}

/// A 10x-style file set in `dir`: `matrix.mtx`, `genes.tsv`, `barcodes.tsv`,
/// the entries `(row, column, value)` counted from 0.
fn write_mtx_set(
    dir: &std::path::Path,
    genes: &[&str],
    barcodes: &[&str],
    entries: &[(usize, usize, f32)],
) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut mtx = format!(
        "%%MatrixMarket matrix coordinate real general\n{} {} {}\n",
        genes.len(),
        barcodes.len(),
        entries.len()
    );
    for (r, c, x) in entries {
        mtx.push_str(&format!("{} {} {}\n", r + 1, c + 1, x));
    }
    std::fs::write(dir.join("matrix.mtx"), mtx)?;
    std::fs::write(dir.join("genes.tsv"), genes.join("\n") + "\n")?;
    std::fs::write(dir.join("barcodes.tsv"), barcodes.join("\n") + "\n")?;
    Ok(())
}

fn mtx_args(dirs: &[&std::path::Path], output: &std::path::Path) -> MergeMtxArgs {
    MergeMtxArgs {
        data_directories: dirs
            .iter()
            .map(|d| d.to_str().expect("utf8").into())
            .collect(),
        backend: SparseIoBackend::Zarr,
        output: output.to_str().expect("utf8").into(),
        zip: false,
        mtx_file_name: "matrix.mtx".into(),
        feature_file_name: "genes.tsv".into(),
        num_feature_name_words: 1,
        barcode_file_name: "barcodes.tsv".into(),
        num_barcode_name_words: 1,
        do_squeeze: false,
        row_nnz_cutoff: 1,
        column_nnz_cutoff: 1,
        block_size: None,
        view: PlanArgs::batch(),
    }
}

#[test]
fn sets_in_directories_of_one_name_are_batches_of_their_own() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    // 10x's layout: the same barcode under the same directory name twice.
    let (a, b) = (dir.path().join("sampleA"), dir.path().join("sampleB"));
    for (sample, x) in [(&a, 1.0), (&b, 2.0)] {
        write_mtx_set(
            &sample.join("filtered_feature_bc_matrix"),
            &["G1"],
            &["ACGT-1"],
            &[(0, 0, x)],
        )?;
    }
    let out = dir.path().join("m");
    run_merge_mtx(&mtx_args(&[&a, &b], &out))?;
    let (dense, _, columns) = read_merged(&out)?;
    assert_eq!(
        columns,
        [
            "ACGT-1@sampleA_filtered_feature_bc_matrix",
            "ACGT-1@sampleB_filtered_feature_bc_matrix"
        ]
        .map(Box::from)
    );
    assert_eq!((dense[(0, 0)], dense[(0, 1)]), (1.0, 2.0));
    let labels = read_lines(dir.path().join("m.batch.gz").to_str().expect("utf8"))?;
    assert_eq!(
        labels,
        [
            "sampleA_filtered_feature_bc_matrix",
            "sampleB_filtered_feature_bc_matrix"
        ]
        .map(Box::from)
    );
    Ok(())
}

#[test]
fn a_batch_is_named_by_its_directory_dots_and_all() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    for run in ["run.1", "run.2"] {
        write_mtx_set(&dir.path().join(run), &["G1"], &["ACGT-1"], &[(0, 0, 1.0)])?;
    }
    let sets = find_mtx_sets(&mtx_args(&[dir.path()], &dir.path().join("m")))?;
    assert_eq!(sets.batch_names, ["run.1", "run.2"]);
    Ok(())
}

#[test]
fn the_rows_kept_are_those_every_set_names_each_once() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let set = |name: &str| dir.path().join(name);
    // X: twice in the first set, once in the second, not in the third.
    // A: in all three, twice in the first.
    write_mtx_set(
        &set("s1"),
        &["X", "X", "A", "A"],
        &["c1"],
        &[(0, 0, 1.0), (1, 0, 2.0), (2, 0, 5.0), (3, 0, 7.0)],
    )?;
    write_mtx_set(
        &set("s2"),
        &["X", "A"],
        &["c2"],
        &[(0, 0, 3.0), (1, 0, 4.0)],
    )?;
    write_mtx_set(
        &set("s3"),
        &["A", "B"],
        &["c3"],
        &[(0, 0, 6.0), (1, 0, 8.0)],
    )?;
    let out = dir.path().join("m");
    run_merge_mtx(&mtx_args(&[dir.path()], &out))?;
    let (dense, rows, columns) = read_merged(&out)?;
    assert_eq!(rows, ["A"].map(Box::from));
    assert_eq!(columns, ["c1@s1", "c2@s2", "c3@s3"].map(Box::from));
    // The first set's A is its first row of that name.
    assert_eq!(
        (dense[(0, 0)], dense[(0, 1)], dense[(0, 2)]),
        (5.0, 4.0, 6.0)
    );
    Ok(())
}

#[test]
fn a_matrix_larger_than_its_name_files_is_refused_by_name() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let set = dir.path().join("s1");
    write_mtx_set(&set, &["G1", "G2"], &["c1", "c2"], &[(1, 1, 1.0)])?;
    write_mtx_set(
        &dir.path().join("s2"),
        &["G1", "G2"],
        &["c3"],
        &[(0, 0, 1.0)],
    )?;
    let args = mtx_args(&[dir.path()], &dir.path().join("m"));

    std::fs::write(set.join("barcodes.tsv"), "c1\n")?;
    let err = format!("{:#}", run_merge_mtx(&args).unwrap_err());
    assert!(
        err.contains("has 2 columns") && err.contains("barcodes.tsv"),
        "{err}"
    );

    std::fs::write(set.join("barcodes.tsv"), "c1\nc2\n")?;
    std::fs::write(set.join("genes.tsv"), "G1\n")?;
    let err = format!("{:#}", run_merge_mtx(&args).unwrap_err());
    assert!(
        err.contains("has 2 rows") && err.contains("genes.tsv"),
        "{err}"
    );
    Ok(())
}

/// The names files of a set end in blank lines often enough: the matrix
/// header says how many names there are, and blank lines past them are not
/// names.
#[test]
fn blank_lines_after_the_names_a_matrix_declares_are_ignored() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let entries = [(0, 0, 1.0), (1, 1, 2.0), (1, 0, 3.0)];
    let merged = |name: &str, genes: &str, barcodes: &str| -> anyhow::Result<Merged> {
        let root = dir.path().join(name);
        let set = root.join("s1");
        write_mtx_set(&set, &["G1", "G2"], &["c1", "c2"], &entries)?;
        std::fs::write(set.join("genes.tsv"), genes)?;
        std::fs::write(set.join("barcodes.tsv"), barcodes)?;
        let out = root.join("m");
        run_merge_mtx(&mtx_args(&[&root], &out))?;
        read_merged(&out)
    };

    let plain = merged("plain", "G1\nG2\n", "c1\nc2\n")?;
    assert_eq!(plain.1, ["G1", "G2"].map(Box::from));
    assert_eq!(plain.2, ["c1@s1", "c2@s1"].map(Box::from));
    // One blank line after the names, and two.
    assert_eq!(merged("one", "G1\nG2\n\n", "c1\nc2\n\n")?, plain);
    assert_eq!(merged("two", "G1\nG2\n\n\n", "c1\nc2\n\n\n")?, plain);
    Ok(())
}

#[test]
fn names_that_do_not_fit_the_matrix_header_are_refused() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let set = dir.path().join("s1");
    write_mtx_set(&set, &["G1", "G2"], &["c1", "c2"], &[(1, 1, 1.0)])?;
    let args = mtx_args(&[dir.path()], &dir.path().join("m"));
    let refused = |genes: &str, barcodes: &str| -> anyhow::Result<String> {
        std::fs::write(set.join("genes.tsv"), genes)?;
        std::fs::write(set.join("barcodes.tsv"), barcodes)?;
        Ok(format!("{:#}", run_merge_mtx(&args).unwrap_err()))
    };

    // A name past those the matrix declares.
    let err = refused("G1\nG2\nG3\n", "c1\nc2\n")?;
    assert!(
        err.contains("has 2 rows") && err.contains("genes.tsv names 3"),
        "{err}"
    );
    // A name after a blank line is one too.
    let err = refused("G1\nG2\n", "c1\nc2\n\nc3\n")?;
    assert!(
        err.contains("has 2 columns") && err.contains("barcodes.tsv names 4"),
        "{err}"
    );
    // A blank line among the names leaves too few of them before the end.
    let err = refused("G1\n\nG2\n", "c1\nc2\n")?;
    assert!(
        err.contains("has 2 rows") && err.contains("genes.tsv names 3"),
        "{err}"
    );
    // Too few names.
    let err = refused("G1\n", "c1\nc2\n")?;
    assert!(
        err.contains("has 2 rows") && err.contains("genes.tsv names 1"),
        "{err}"
    );
    assert!(!dir.path().join("m.zarr").exists(), "nothing is written");
    Ok(())
}

/// As many names as the matrix declares are its names, an empty last one
/// included.
#[test]
fn an_empty_name_among_those_the_matrix_declares_is_a_name() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let set = dir.path().join("s1");
    write_mtx_set(&set, &["G1", "G2"], &["c1", "c2"], &[(1, 1, 5.0)])?;
    std::fs::write(set.join("genes.tsv"), "G1\n\n")?;
    let out = dir.path().join("m");
    run_merge_mtx(&mtx_args(&[dir.path()], &out))?;
    let (dense, rows, _) = read_merged(&out)?;
    assert_eq!(rows, ["", "G1"].map(Box::from));
    assert_eq!(dense[(0, 1)], 5.0);
    Ok(())
}

#[test]
fn a_file_name_that_fits_several_files_takes_the_one_called_that() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let set = dir.path().join("s1");
    write_mtx_set(&set, &["G1"], &["c1"], &[(0, 0, 1.0)])?;
    std::fs::copy(set.join("matrix.mtx"), set.join("raw_matrix.mtx"))?;
    let args = mtx_args(&[dir.path()], &dir.path().join("m"));
    let sets = find_mtx_sets(&args)?;
    assert!(
        sets.mtx_files[0].ends_with("/matrix.mtx"),
        "{:?}",
        sets.mtx_files
    );

    // None called that: which of the two was meant is not determined.
    std::fs::rename(set.join("matrix.mtx"), set.join("filtered_matrix.mtx"))?;
    let err = format!("{:#}", find_mtx_sets(&args).err().expect("ambiguous"));
    assert!(err.contains("both end with"), "{err}");

    // One file ending with the name is the one.
    std::fs::remove_file(set.join("raw_matrix.mtx"))?;
    let sets = find_mtx_sets(&args)?;
    assert!(sets.mtx_files[0].ends_with("filtered_matrix.mtx"));
    Ok(())
}

fn align_args(files: &[&str], types: usize, out: &std::path::Path) -> AlignDataArgs {
    AlignDataArgs {
        data_files: files.iter().map(|f| Box::from(*f)).collect(),
        num_data_types: types,
        output_directory: out.to_str().expect("utf8").into(),
        view: PlanArgs::batch(),
    }
}

#[test]
fn aligning_over_no_data_type_is_an_error() -> anyhow::Result<()> {
    let (dir, a, b) = two_fixtures()?;
    let err = align_backends(&align_args(&[&a, &b], 0, &dir.path().join("out"))).unwrap_err();
    assert!(format!("{err:#}").contains("at least 1"), "{err:#}");
    Ok(())
}

#[test]
fn files_align_on_the_names_each_has_however_often() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = |name: &str| dir.path().join(name).to_string_lossy().into_owned();
    let (a, b) = (path("a.zarr"), path("b.zarr"));
    // g0 is on two rows of the first file and none of the second.
    for (file, rows) in [(&a, ["g0", "g0", "g1"]), (&b, ["g1", "g2", "g3"])] {
        write_fixture(file, 3, 1.0)?;
        let mut data = open_sparse_matrix(file, &SparseIoBackend::Zarr)?;
        data.register_row_names_vec(&rows.map(Box::from));
        data.register_column_names_vec(&["c0", "c1", "c2"].map(Box::from));
    }
    let out = dir.path().join("out");
    align_backends(&align_args(&[&a, &b], 1, &out))?;
    for name in ["1_1_a.zarr", "1_2_b.zarr"] {
        let aligned = open_sparse_matrix(
            out.join(name).to_str().expect("utf8"),
            &SparseIoBackend::Zarr,
        )?;
        assert_eq!(aligned.row_names()?, ["g1"].map(Box::from), "{name}");
        assert_eq!(aligned.num_columns(), Some(3), "{name}");
    }
    Ok(())
}
