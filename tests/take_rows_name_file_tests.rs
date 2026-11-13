#![cfg(feature = "tui")]

//! `data-beans rows --name-file`: a line of the name file with an id and a
//! name is one row name, joined as the backend joins them.

use data_beans::sparse_io::*;
use std::process::Command;

const EXE: &str = env!("CARGO_BIN_EXE_data-beans");

#[test]
fn a_two_word_line_of_the_name_file_names_an_id_name_row() {
    let dir = tempfile::tempdir().unwrap();
    let path = |name: &str| dir.path().join(name).to_string_lossy().into_owned();

    let data = path("data.zarr");
    let m = nalgebra::DMatrix::<f32>::from_fn(3, 2, |i, j| (1 + i + 3 * j) as f32);
    let mut backend =
        create_sparse_from_dmatrix(&m, Some(&data), Some(&SparseIoBackend::Zarr)).unwrap();
    let rows: Vec<Box<str>> = ["ENSG1_CD8A", "ENSG2_CD4", "ENSG3_MS4A1"]
        .map(Box::from)
        .to_vec();
    let cols: Vec<Box<str>> = ["c0", "c1"].map(Box::from).to_vec();
    backend.register_row_names_vec(&rows);
    backend.register_column_names_vec(&cols);
    drop(backend);

    let name_file = path("rows.tsv");
    std::fs::write(&name_file, "ENSG1\tCD8A\nENSG3\tMS4A1\n").unwrap();
    let output = path("out.tsv");

    let run = Command::new(EXE)
        .args(["rows", &data, "--name-file", &name_file, "-o", &output])
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );

    let table = std::fs::read_to_string(&output).unwrap();
    let header = table.lines().next().unwrap();
    assert!(header.contains("ENSG1_CD8A"), "{table}");
    assert!(header.contains("ENSG3_MS4A1"), "{table}");
    assert!(!table.contains("ENSG2_CD4"), "{table}");
    // One line per column after the header.
    assert_eq!(table.lines().count(), 3, "{table}");
}
