use super::*;
use crate::export::h5ad::{vlu, write_str_attr};
use crate::hdf5_io::read_h5ad_column;
use crate::sparse_io::open_sparse_matrix;
use hdf5::types::VarLenUnicode;
use std::path::Path;

fn write_strings(group: &hdf5::Group, name: &str, values: &[&str]) {
    let values: Vec<VarLenUnicode> = values.iter().map(|v| vlu(v).unwrap()).collect();
    group
        .new_dataset::<VarLenUnicode>()
        .shape(values.len())
        .create(name)
        .unwrap()
        .write_raw(&values)
        .unwrap();
}

/// An AnnData dataframe group: its index stored under `index.0`, as the
/// `_index` attribute says, and string columns.
fn write_dataframe(
    file: &hdf5::File,
    name: &str,
    index: (&str, &[&str]),
    columns: &[(&str, &[&str])],
) {
    let group = file.create_group(name).unwrap();
    write_str_attr(&group, "_index", index.0).unwrap();
    write_strings(&group, index.0, index.1);
    let order: Vec<VarLenUnicode> = columns.iter().map(|(c, _)| vlu(c).unwrap()).collect();
    group
        .new_attr::<VarLenUnicode>()
        .shape([order.len()])
        .create("column-order")
        .unwrap()
        .write_raw(&order)
        .unwrap();
    for (column, values) in columns {
        write_strings(&group, column, values);
    }
}

/// The sparse group `X`, its format and shape under the attributes
/// named.
fn write_x(
    file: &hdf5::File,
    format: Option<(&str, &str)>,
    shape: Option<(&str, [i64; 2])>,
    data: &[f32],
    indices: &[i32],
    indptr: &[i32],
) {
    let x = file.create_group("X").unwrap();
    if let Some((name, value)) = format {
        write_str_attr(&x, name, value).unwrap();
    }
    if let Some((name, value)) = shape {
        x.new_attr::<i64>()
            .shape([2])
            .create(name)
            .unwrap()
            .write_raw(&value)
            .unwrap();
    }
    let ds = |name: &str, n: usize| x.new_dataset::<i32>().shape(n).create(name).unwrap();
    x.new_dataset::<f32>()
        .shape(data.len())
        .create("data")
        .unwrap()
        .write_raw(data)
        .unwrap();
    ds("indices", indices.len()).write_raw(indices).unwrap();
    ds("indptr", indptr.len()).write_raw(indptr).unwrap();
}

// 3 cells x 4 genes, CSR over cells; the last gene has no count:
//   BC1 [1 0 2 0], BC2 [0 3 0 0], BC3 [4 0 0 0]
const DATA: [f32; 4] = [1., 2., 3., 4.];
const INDICES: [i32; 4] = [0, 2, 1, 0];
const INDPTR: [i32; 4] = [0, 2, 3, 4];
const CELLS: [&str; 3] = ["BC1", "BC2", "BC3"];
const GENES: [&str; 4] = ["G1", "G2", "G3", "G4"];

fn args(h5ad: &Path, output: &Path) -> FromH5adArgs {
    let fields = |xs: [&str; 2]| xs.iter().map(|&x| Box::from(x)).collect();
    FromH5adArgs {
        h5ad_file: h5ad.to_str().unwrap().into(),
        backend: SparseIoBackend::Zarr,
        output: output.to_str().unwrap().into(),
        zip: false,
        row_id_field: fields(["_index", "gene_id"]),
        row_name_field: fields(["feature_name", "gene_name"]),
        col_name_field: fields(["_index", "cell"]),
        select_row_type: "".into(),
        remove_row_type: "".into(),
        do_squeeze: false,
        row_nnz_cutoff: 1,
        column_nnz_cutoff: 1,
        block_size: None,
        view: crate::handlers::explore::PlanArgs::batch(),
    }
}

type Built = (Vec<Box<str>>, Vec<Box<str>>, Vec<Vec<f32>>);

/// Build `h5ad` and read the backend back: its row names, column names
/// and the dense (features x cells) matrix, row by row.
fn build(h5ad: &Path) -> anyhow::Result<Built> {
    let output = h5ad.with_extension("");
    run_build_from_h5ad(&args(h5ad, &output))?;
    let data = open_sparse_matrix(
        &format!("{}.zarr", output.to_str().unwrap()),
        &SparseIoBackend::Zarr,
    )?;
    let cols = data.column_names()?;
    let m = data.read_columns_dmatrix((0..cols.len()).collect())?;
    let rows = m.row_iter().map(|r| r.iter().copied().collect()).collect();
    Ok((data.row_names()?, cols, rows))
}

fn genes_by_cells() -> Vec<Vec<f32>> {
    vec![
        vec![1., 0., 4.],
        vec![0., 3., 0.],
        vec![2., 0., 0.],
        vec![0., 0., 0.],
    ]
}

#[test]
fn a_named_index_names_the_cells_and_features() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("named.h5ad");
    let file = hdf5::File::create(&h5ad).unwrap();
    let csr = Some(("encoding-type", "csr_matrix"));
    write_x(
        &file,
        csr,
        Some(("shape", [3, 4])),
        &DATA,
        &INDICES,
        &INDPTR,
    );
    write_dataframe(&file, "obs", ("barcode", &CELLS), &[]);
    write_dataframe(&file, "var", ("gene_ids", &GENES), &[]);
    drop(file);

    let (rows, cols, m) = build(&h5ad).unwrap();
    assert_eq!(cols, CELLS.map(Box::from));
    assert_eq!(rows, GENES.map(Box::from));
    assert_eq!(m, genes_by_cells());
}

#[test]
fn features_and_cells_past_the_last_count_are_kept() {
    let dir = tempfile::tempdir().unwrap();
    // No shape on X: var/ and obs/ say how large the matrix is.
    let h5ad = dir.path().join("csr.h5ad");
    let file = hdf5::File::create(&h5ad).unwrap();
    let csr = Some(("encoding-type", "csr_matrix"));
    write_x(&file, csr, None, &DATA, &INDICES, &INDPTR);
    write_dataframe(&file, "obs", ("_index", &CELLS), &[]);
    write_dataframe(&file, "var", ("_index", &GENES), &[]);
    drop(file);
    let (rows, _, m) = build(&h5ad).unwrap();
    assert_eq!(rows, GENES.map(Box::from));
    assert_eq!(m, genes_by_cells());

    // The same arrays as CSC over genes: 4 cells x 3 genes, the last
    // cell empty.
    let h5ad = dir.path().join("csc.h5ad");
    let file = hdf5::File::create(&h5ad).unwrap();
    let csc = Some(("encoding-type", "csc_matrix"));
    write_x(
        &file,
        csc,
        Some(("shape", [4, 3])),
        &DATA,
        &INDICES,
        &INDPTR,
    );
    write_dataframe(&file, "obs", ("_index", &GENES), &[]);
    write_dataframe(&file, "var", ("_index", &CELLS), &[]);
    drop(file);
    let (rows, cols, m) = build(&h5ad).unwrap();
    assert_eq!(rows, CELLS.map(Box::from));
    assert_eq!(cols, GENES.map(Box::from));
    assert_eq!(
        m,
        [[1., 0., 2., 0.], [0., 3., 0., 0.], [4., 0., 0., 0.]].map(Vec::from)
    );
}

#[test]
fn data_reaching_past_the_names_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("short.h5ad");
    let file = hdf5::File::create(&h5ad).unwrap();
    let csr = Some(("encoding-type", "csr_matrix"));
    write_x(&file, csr, None, &DATA, &INDICES, &INDPTR);
    write_dataframe(&file, "obs", ("_index", &CELLS), &[]);
    write_dataframe(&file, "var", ("_index", &GENES[..2]), &[]);
    drop(file);
    let e = build(&h5ad).unwrap_err().to_string();
    assert!(e.contains("2 features"), "{e}");
}

#[test]
fn the_sparse_format_is_read_where_it_is_named() {
    let dir = tempfile::tempdir().unwrap();
    let frames = |file: &hdf5::File| {
        write_dataframe(file, "obs", ("_index", &CELLS), &[]);
        write_dataframe(file, "var", ("_index", &GENES), &[]);
    };
    // As anndata before 0.7 wrote a CSR matrix.
    let h5ad = dir.path().join("legacy.h5ad");
    let file = hdf5::File::create(&h5ad).unwrap();
    let legacy = Some(("h5sparse_format", "csr"));
    let shape = Some(("h5sparse_shape", [3, 4]));
    write_x(&file, legacy, shape, &DATA, &INDICES, &INDPTR);
    frames(&file);
    drop(file);
    let (rows, cols, m) = build(&h5ad).unwrap();
    assert_eq!(rows, GENES.map(Box::from));
    assert_eq!(cols, CELLS.map(Box::from));
    assert_eq!(m, genes_by_cells());

    let h5ad = dir.path().join("coo.h5ad");
    let file = hdf5::File::create(&h5ad).unwrap();
    let coo = Some(("encoding-type", "coo_matrix"));
    write_x(&file, coo, None, &DATA, &INDICES, &INDPTR);
    frames(&file);
    drop(file);
    let e = build(&h5ad).unwrap_err().to_string();
    assert!(e.contains("CSR"), "{e}");
}

#[test]
fn an_unnamed_sparse_format_is_told_by_the_pointer_length() {
    let dir = tempfile::tempdir().unwrap();
    // Three vectors, 3 cells and 4 genes: CSR over cells.
    let h5ad = dir.path().join("csr.h5ad");
    let file = hdf5::File::create(&h5ad).unwrap();
    write_x(&file, None, None, &DATA, &INDICES, &INDPTR);
    write_dataframe(&file, "obs", ("_index", &CELLS), &[]);
    write_dataframe(&file, "var", ("_index", &GENES), &[]);
    drop(file);
    let (rows, cols, m) = build(&h5ad).unwrap();
    assert_eq!(rows, GENES.map(Box::from));
    assert_eq!(cols, CELLS.map(Box::from));
    assert_eq!(m, genes_by_cells());

    // Three vectors, 4 cells and 3 genes: CSC over genes.
    let h5ad = dir.path().join("csc.h5ad");
    let file = hdf5::File::create(&h5ad).unwrap();
    let shape = Some(("shape", [4, 3]));
    write_x(&file, None, shape, &DATA, &INDICES, &INDPTR);
    write_dataframe(&file, "obs", ("_index", &GENES), &[]);
    write_dataframe(&file, "var", ("_index", &CELLS), &[]);
    drop(file);
    let (rows, cols, m) = build(&h5ad).unwrap();
    assert_eq!(rows, CELLS.map(Box::from));
    assert_eq!(cols, GENES.map(Box::from));
    assert_eq!(
        m,
        [[1., 0., 2., 0.], [0., 3., 0., 0.], [4., 0., 0., 0.]].map(Vec::from)
    );

    // Three vectors fit 3 cells and 3 genes alike, and neither of 4
    // cells and 4 genes.
    for (name, n) in [("square.h5ad", 3), ("neither.h5ad", 4)] {
        let h5ad = dir.path().join(name);
        let file = hdf5::File::create(&h5ad).unwrap();
        write_x(&file, None, None, &DATA, &INDICES, &INDPTR);
        write_dataframe(&file, "obs", ("_index", &GENES[..n]), &[]);
        write_dataframe(&file, "var", ("_index", &GENES[..n]), &[]);
        drop(file);
        let e = build(&h5ad).unwrap_err().to_string();
        assert!(e.contains("CSR from CSC"), "{e}");
    }
}

#[test]
fn the_declared_shape_sizes_an_axis_without_names() {
    let dir = tempfile::tempdir().unwrap();
    // var/ has no index, and the counts reach 3 genes of the 4 that
    // the shape declares.
    let h5ad = dir.path().join("unnamed.h5ad");
    let file = hdf5::File::create(&h5ad).unwrap();
    let csr = Some(("encoding-type", "csr_matrix"));
    let shape = Some(("shape", [3, 4]));
    write_x(&file, csr, shape, &DATA, &INDICES, &INDPTR);
    write_dataframe(&file, "obs", ("_index", &CELLS), &[]);
    file.create_group("var").unwrap();
    drop(file);
    let (rows, cols, m) = build(&h5ad).unwrap();
    assert_eq!(rows, ["0", "1", "2", "3"].map(Box::from));
    assert_eq!(cols, CELLS.map(Box::from));
    assert_eq!(m, genes_by_cells());
}

#[test]
fn names_that_disagree_with_the_declared_shape_are_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("disagree.h5ad");
    let file = hdf5::File::create(&h5ad).unwrap();
    let csr = Some(("encoding-type", "csr_matrix"));
    let shape = Some(("shape", [3, 4]));
    write_x(&file, csr, shape, &DATA, &INDICES, &INDPTR);
    write_dataframe(&file, "obs", ("_index", &CELLS), &[]);
    let genes = ["G1", "G2", "G3", "G4", "G5"];
    write_dataframe(&file, "var", ("_index", &genes), &[]);
    drop(file);
    let e = build(&h5ad).unwrap_err().to_string();
    assert!(e.contains("lists 5 feature IDs but X has 4"), "{e}");
}

#[test]
fn selecting_no_feature_is_an_error_before_anything_is_written() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("typed.h5ad");
    let file = hdf5::File::create(&h5ad).unwrap();
    let csr = Some(("encoding-type", "csr_matrix"));
    let shape = Some(("shape", [3, 4]));
    write_x(&file, csr, shape, &DATA, &INDICES, &INDPTR);
    write_dataframe(&file, "obs", ("_index", &CELLS), &[]);
    let types: [(&str, &[&str]); 1] = [("feature_types", &["Gene Expression"; 4])];
    write_dataframe(&file, "var", ("_index", &GENES), &types);
    drop(file);

    let output = dir.path().join("typed");
    let mut args = args(&h5ad, &output);
    args.select_row_type = "peak".into();
    let e = run_build_from_h5ad(&args).unwrap_err().to_string();
    assert!(e.contains("none of the 4 rows is kept"), "{e}");
    assert!(e.contains("Gene Expression") && e.contains("`peak`"), "{e}");
    assert!(!output.with_extension("zarr").exists());
}

/// Two donors share a barcode. The cell metadata `from-h5ad` writes,
/// given back to `to-h5ad`, annotates each cell with its own donor's
/// row.
#[test]
fn cell_metadata_round_trips_when_donors_share_a_barcode() {
    use crate::handlers::exporters::{run_export_to_h5ad, ToH5adArgs};
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("donors.h5ad");
    let file = hdf5::File::create(&h5ad).unwrap();
    let csr = Some(("encoding-type", "csr_matrix"));
    write_x(
        &file,
        csr,
        Some(("shape", [3, 4])),
        &DATA,
        &INDICES,
        &INDPTR,
    );
    let obs: [(&str, &[&str]); 2] = [
        ("donor_id", &["donorA", "donorB", "donorA"]),
        ("cell_type", &["T", "B", "NK"]),
    ];
    let barcodes = ["AAAC-1", "AAAC-1", "GGGT-1"];
    write_dataframe(&file, "obs", ("_index", &barcodes), &obs);
    write_dataframe(&file, "var", ("_index", &GENES), &[]);
    drop(file);

    let (_, cols, _) = build(&h5ad).unwrap();
    let cells = ["AAAC-1@donorA", "AAAC-1@donorB", "GGGT-1@donorA"].map(Box::from);
    assert_eq!(cols, cells);

    let stem = dir.path().join("donors");
    let stem = stem.to_str().unwrap();
    let back = dir.path().join("back.h5ad");
    run_export_to_h5ad(&ToH5adArgs {
        data_file: format!("{stem}.zarr").into(),
        output: Some(back.to_str().unwrap().into()),
        obs: Some(format!("{stem}.cell_metadata.tsv.gz").into()),
        var: None,
    })
    .unwrap();
    let obs = hdf5::File::open(&back).unwrap().group("obs").unwrap();
    assert_eq!(read_h5ad_column(&obs, "_index").unwrap(), cells);
    assert_eq!(
        read_h5ad_column(&obs, "cell_type").unwrap(),
        ["T", "B", "NK"].map(Box::from)
    );
}
