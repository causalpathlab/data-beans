//! Every subcommand that writes a backend produces the same thing by default.
//!
//! The four handlers here stage a working `.zarr` directory and re-zip it at the
//! end, so before this they silently left a directory where every `from-*`
//! subcommand leaves a `.zarr.zip`. A pipeline that feeds one subcommand's
//! output to the next then breaks on the path alone.

use super::super::{ReorderRowsArgs, SubsetColumnsArgs, SubsetRowsArgs};
use crate::handlers::cell_qc::CellQcArgs;
use clap::Parser;

#[derive(Parser)]
struct Cols {
    #[command(flatten)]
    args: SubsetColumnsArgs,
}
#[derive(Parser)]
struct Rows {
    #[command(flatten)]
    args: SubsetRowsArgs,
}
#[derive(Parser)]
struct Reorder {
    #[command(flatten)]
    args: ReorderRowsArgs,
}
#[derive(Parser)]
struct Qc {
    #[command(flatten)]
    args: CellQcArgs,
}

#[test]
fn an_edit_handler_zips_its_output_like_every_builder_does() {
    assert!(Cols::parse_from(["x", "in.zarr.zip", "-o", "out"]).args.zip);
    assert!(Rows::parse_from(["x", "in.zarr.zip", "-o", "out"]).args.zip);
    assert!(
        Reorder::parse_from(["x", "in.zarr.zip", "-r", "rows.tsv", "-o", "out"])
            .args
            .zip
    );
    assert!(Qc::parse_from(["x", "in.zarr.zip", "-o", "out"]).args.zip);
}

#[test]
fn no_zip_still_asks_for_a_directory() {
    assert!(
        !Cols::parse_from(["x", "in.zarr.zip", "-o", "out", "--no-zip"])
            .args
            .zip
    );
    assert!(
        !Rows::parse_from(["x", "in.zarr.zip", "-o", "out", "--no-zip"])
            .args
            .zip
    );
}

#[test]
fn interactive_output_keeps_the_input_container() {
    use super::zip_like_input;
    assert!(
        !zip_like_input("in.zarr", "out"),
        "a directory stays a directory"
    );
    assert!(zip_like_input("in.zarr.zip", "out"), "a zip stays a zip");
    assert!(!zip_like_input("in.h5", "out"), "HDF5 is never zipped");
    assert!(
        !zip_like_input("in.zarr.zip", "out.zarr"),
        "an explicit .zarr output wins"
    );
}

/// A subset written as `.zarr.zip` leaves the archive alone in its folder: the
/// working `.zarr` (and subset's `.subset_tmp`) are built in a hidden stage,
/// marked for Dropbox to ignore, so a sync client watching the output folder
/// does not see them come and go and put them back empty.
#[test]
fn a_zipped_subset_leaves_only_the_archive_in_its_folder() {
    use crate::sparse_io::open_sparse_matrix_by_path;
    let (out_dir, left) = subset_into("out.zarr.zip");
    let out = out_dir.path().join("results/out.zarr.zip");
    let out = out.to_str().unwrap();
    assert_eq!(left, ["out.zarr.zip"]);
    // Never as a plain sibling, even for a moment: built in the stage.
    let (_, working) =
        crate::hdf5_io::resolve_backend_file(out, Some(crate::sparse_io::SparseIoBackend::Zarr))
            .unwrap();
    let stage = std::path::Path::new(working.as_ref()).parent().unwrap();
    assert_eq!(stage, out_dir.path().join("results/.data-beans-stage"));
    assert!(!stage.exists(), "{} left behind", stage.display());
    assert_eq!(
        open_sparse_matrix_by_path(out).unwrap().num_columns(),
        Some(3)
    );
}

/// A subset into a `.zarr` directory leaves the directory alone too: its
/// `.subset_tmp` is built in the hidden stage, which goes once it is empty.
#[test]
fn a_directory_subset_leaves_only_the_directory_in_its_folder() {
    let (_dir, left) = subset_into("out.zarr");
    assert_eq!(left, ["out.zarr"]);
}

/// Columns 1-3 of a small backend subset into `results/{out}` of a new
/// folder: that folder, and the names left in `results/` after.
fn subset_into(out: &str) -> (tempfile::TempDir, Vec<String>) {
    use crate::sparse_io::create_sparse_from_triplets;
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.zarr");
    let triplets = [(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0), (0, 3, 4.0)];
    let mut data = create_sparse_from_triplets(
        &triplets,
        (3, 4, triplets.len()),
        Some(input.to_str().unwrap()),
        Some(&crate::sparse_io::SparseIoBackend::Zarr),
    )
    .unwrap();
    data.register_row_names_vec(&["g0".into(), "g1".into(), "g2".into()]);
    data.register_column_names_vec(&["c0".into(), "c1".into(), "c2".into(), "c3".into()]);
    drop(data);

    let out = dir.path().join("results").join(out);
    let mut argv = vec![
        "x",
        input.to_str().unwrap(),
        "-i",
        "1-3",
        "-o",
        out.to_str().unwrap(),
        "--batch-process",
    ];
    if !out.to_str().unwrap().ends_with(".zip") {
        argv.push("--no-zip");
    }
    let cols = Cols::parse_from(argv);
    super::subset_columns(&cols.args).unwrap();

    let left = std::fs::read_dir(dir.path().join("results"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    (dir, left)
}
