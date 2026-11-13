//! Every subcommand that writes a backend produces the same thing by default.
//!
//! The four handlers here stage a working `.zarr` directory and re-zip it at the
//! end, so before this they silently left a directory where every `from-*`
//! subcommand leaves a `.zarr.zip`. A pipeline that feeds one subcommand's
//! output to the next then breaks on the path alone.

use super::super::tests::input_in;
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
    let dir = tempfile::tempdir().unwrap();
    let input = input_in(dir.path());

    let out = dir.path().join("results").join(out);
    let mut argv = vec![
        "x",
        input.as_str(),
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

/// The backend at `path`: its row names, and its rows' values.
fn rows_of(path: &str) -> (Vec<String>, Vec<Vec<f32>>) {
    let data = crate::sparse_io::open_sparse_matrix_by_path(path).unwrap();
    let (nrow, ncol) = (data.num_rows().unwrap(), data.num_columns().unwrap());
    let dense = data.read_columns_dmatrix((0..ncol).collect()).unwrap();
    let names = data.row_names().unwrap();
    (
        names.iter().map(|n| n.to_string()).collect(),
        (0..nrow)
            .map(|r| (0..ncol).map(|c| dense[(r, c)]).collect())
            .collect(),
    )
}

/// What [`input_in`] wrote, to see it is still there after a command.
fn input_rows() -> (Vec<String>, Vec<Vec<f32>>) {
    (
        ["g0", "g1", "g2"].map(String::from).to_vec(),
        vec![
            vec![1.0, 0.0, 0.0, 4.0],
            vec![0.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 3.0, 0.0],
        ],
    )
}

#[test]
fn rows_are_subset_in_the_order_asked_and_the_input_is_only_read() {
    let dir = tempfile::tempdir().unwrap();
    let input = input_in(dir.path());
    let out = dir.path().join("out.zarr");
    let out = out.to_str().unwrap();
    let rows = Rows::parse_from([
        "x",
        input.as_str(),
        "-i",
        "2,0",
        "-o",
        out,
        "--batch-process",
        "--no-zip",
    ]);
    super::subset_rows(&rows.args).unwrap();
    let (names, values) = rows_of(out);
    assert_eq!(names, ["g2", "g0"]);
    assert_eq!(values, [vec![0.0, 0.0, 3.0, 0.0], vec![1.0, 0.0, 0.0, 4.0]]);
    assert_eq!(rows_of(&input), input_rows());
}

#[test]
fn rows_are_reordered_as_the_file_names_them() {
    let dir = tempfile::tempdir().unwrap();
    let input = input_in(dir.path());
    // g1 is not named and goes; `other` is not in the data and is empty.
    let order = dir.path().join("rows.tsv");
    std::fs::write(&order, "g2\nother\ng0\n").unwrap();
    let out = dir.path().join("out.zarr");
    let out = out.to_str().unwrap();
    let reorder = Reorder::parse_from([
        "x",
        input.as_str(),
        "-r",
        order.to_str().unwrap(),
        "-o",
        out,
        "--no-zip",
    ]);
    super::reorder_rows(&reorder.args).unwrap();
    let (names, values) = rows_of(out);
    assert_eq!(names, ["g2", "other", "g0"]);
    assert_eq!(
        values,
        [
            vec![0.0, 0.0, 3.0, 0.0],
            vec![0.0; 4],
            vec![1.0, 0.0, 0.0, 4.0]
        ]
    );
    let by_row = crate::sparse_io::open_sparse_matrix_by_path(out)
        .unwrap()
        .read_rows_dmatrix(vec![2])
        .unwrap();
    assert_eq!(by_row.iter().copied().sum::<f32>(), 5.0);
    assert_eq!(rows_of(&input), input_rows());
}

#[test]
fn an_output_that_is_the_input_is_refused_and_the_input_kept() {
    let dir = tempfile::tempdir().unwrap();
    let input = input_in(dir.path());
    let bare = input.strip_suffix(".zarr").unwrap();
    // `-o in.zarr`, and `-o in`, which --no-zip resolves to it.
    for out in [input.as_str(), bare] {
        let argv = [
            "x",
            input.as_str(),
            "-i",
            "0-1",
            "-o",
            out,
            "--batch-process",
            "--no-zip",
        ];
        let err = super::subset_columns(&Cols::parse_from(argv).args).unwrap_err();
        assert!(format!("{err:#}").contains("is the input"), "{err:#}");
        let err = super::subset_rows(&Rows::parse_from(argv).args).unwrap_err();
        assert!(format!("{err:#}").contains("is the input"), "{err:#}");
        assert_eq!(rows_of(&input), input_rows());
    }
}

/// A zipped input is no more its own output than a directory is: `-o in`
/// resolves to `in.zarr.zip`, and is refused with the archive untouched.
#[test]
fn an_output_that_is_the_zipped_input_is_refused_and_the_input_kept() {
    let dir = tempfile::tempdir().unwrap();
    let plain = input_in(dir.path());
    let input = format!("{plain}.zip");
    let input = input.as_str();
    crate::zarr_io::finalize_zarr_output(&plain, input).unwrap();
    assert_eq!(rows_of(input), input_rows());
    let bytes = std::fs::read(input).unwrap();
    let bare = input.strip_suffix(".zarr.zip").unwrap();
    let order = dir.path().join("rows.tsv");
    std::fs::write(&order, "g2\ng0\n").unwrap();

    for out in [input, bare] {
        let argv = ["x", input, "-i", "0-1", "-o", out, "--batch-process"];
        let err = super::subset_columns(&Cols::parse_from(argv).args).unwrap_err();
        assert!(format!("{err:#}").contains("is the input"), "{err:#}");
        let err = super::subset_rows(&Rows::parse_from(argv).args).unwrap_err();
        assert!(format!("{err:#}").contains("is the input"), "{err:#}");
        let argv = ["x", input, "-r", order.to_str().unwrap(), "-o", out];
        let err = super::reorder_rows(&Reorder::parse_from(argv).args).unwrap_err();
        assert!(format!("{err:#}").contains("is the input"), "{err:#}");
        assert_eq!(std::fs::read(input).unwrap(), bytes, "the archive changed");
    }
    let left: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name != "rows.tsv")
        .collect();
    assert_eq!(left, ["in.zarr.zip"]);
}

/// A name the order lists twice would leave its second row empty; a name
/// two rows of the data carry would fold them into one. Both are refused
/// and nothing is written.
#[test]
fn an_order_that_places_a_row_twice_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let input = input_in(dir.path());
    let order = dir.path().join("rows.tsv");
    let out = dir.path().join("out.zarr");
    let argv = [
        "x",
        input.as_str(),
        "-r",
        order.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
        "--no-zip",
    ];

    std::fs::write(&order, "g2\ng0\ng2\n").unwrap();
    let err = super::reorder_rows(&Reorder::parse_from(argv).args).unwrap_err();
    assert!(format!("{err:#}").contains("lists `g2` twice"), "{err:#}");
    assert!(!out.exists());

    let mut data = crate::sparse_io::open_sparse_matrix_by_path(&input).unwrap();
    data.register_row_names_vec(&["g0".into(), "g1".into(), "g0".into()]);
    drop(data);
    std::fs::write(&order, "g1\ng0\n").unwrap();
    let err = super::reorder_rows(&Reorder::parse_from(argv).args).unwrap_err();
    assert!(
        format!("{err:#}").contains("rows 0 and 2 are both named `g0`"),
        "{err:#}"
    );
    assert!(!out.exists());
}

/// Blank lines at the end of the order file name no row: the result is the
/// one the file without them gives.
#[test]
fn blank_lines_ending_an_order_are_not_rows() {
    let dir = tempfile::tempdir().unwrap();
    let input = input_in(dir.path());
    let order = dir.path().join("rows.tsv");
    let reordered = |text: &str, name: &str| {
        std::fs::write(&order, text).unwrap();
        let out = dir.path().join(name);
        let out = out.to_str().unwrap().to_string();
        let argv = [
            "x",
            input.as_str(),
            "-r",
            order.to_str().unwrap(),
            "-o",
            out.as_str(),
            "--no-zip",
        ];
        super::reorder_rows(&Reorder::parse_from(argv).args).unwrap();
        let data = crate::sparse_io::open_sparse_matrix_by_path(&out).unwrap();
        let ncol = data.num_columns().unwrap();
        (
            data.row_names().unwrap(),
            data.read_triplets_by_columns((0..ncol).collect()).unwrap(),
        )
    };
    let plain = reordered("g2\ng0\n", "plain.zarr");
    assert_eq!(plain.0.len(), 2);
    assert_eq!(reordered("g2\ng0\n\n\n", "blank.zarr"), plain);
}

/// An output named for the other format is written in that format.
#[cfg(feature = "hdf5")]
#[test]
fn an_output_named_h5_is_an_hdf5_file() {
    use crate::sparse_io::{open_sparse_matrix, SparseIoBackend};
    let dir = tempfile::tempdir().unwrap();
    let input = input_in(dir.path());
    let out = dir.path().join("out.h5");
    let out = out.to_str().unwrap();
    let cols = Cols::parse_from([
        "x",
        input.as_str(),
        "-i",
        "1-3",
        "-o",
        out,
        "--batch-process",
    ]);
    super::subset_columns(&cols.args).unwrap();
    assert!(std::path::Path::new(out).is_file());
    let data = open_sparse_matrix(out, &SparseIoBackend::HDF5).unwrap();
    assert_eq!(data.num_columns(), Some(3));
    assert_eq!(
        rows_of(out).1,
        [
            vec![0.0, 0.0, 4.0],
            vec![2.0, 0.0, 0.0],
            vec![0.0, 3.0, 0.0]
        ]
    );
}
