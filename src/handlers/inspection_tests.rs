//! `info -o PREFIX` writes `PREFIX.rows.gz` and `PREFIX.columns.gz`: the
//! prefix is a file's, so its folder is made and nothing of its own name.

use super::*;

#[test]
fn info_writes_the_names_beside_the_prefix_and_no_directory_of_its_name() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let input = dir.path().join("in.zarr");
    let triplets = [(0, 0, 1.0), (1, 1, 2.0)];
    let mut data = create_sparse_from_triplets(
        &triplets,
        (2, 2, triplets.len()),
        Some(input.to_str().expect("utf8")),
        Some(&SparseIoBackend::Zarr),
    )?;
    data.register_row_names_vec(&["g0".into(), "g1".into()]);
    data.register_column_names_vec(&["c0".into(), "c1".into()]);
    drop(data);

    // A regular file of the prefix's name is no obstacle either.
    let prefix = dir.path().join("out").join("x");
    std::fs::create_dir_all(dir.path().join("out"))?;
    std::fs::write(&prefix, "kept")?;
    show_info(&InfoArgs {
        data_file: input.to_str().expect("utf8").into(),
        output: prefix.to_str().expect("utf8").into(),
    })?;
    assert_eq!(std::fs::read_to_string(&prefix)?, "kept");
    let rows = read_lines(dir.path().join("out/x.rows.gz").to_str().expect("utf8"))?;
    assert_eq!(rows, ["g0", "g1"].map(Box::from));

    // A folder that is not there yet is made; the prefix is not.
    let prefix = dir.path().join("new").join("y");
    show_info(&InfoArgs {
        data_file: input.to_str().expect("utf8").into(),
        output: prefix.to_str().expect("utf8").into(),
    })?;
    assert!(!prefix.exists());
    let columns = read_lines(dir.path().join("new/y.columns.gz").to_str().expect("utf8"))?;
    assert_eq!(columns, ["c0", "c1"].map(Box::from));
    Ok(())
}
