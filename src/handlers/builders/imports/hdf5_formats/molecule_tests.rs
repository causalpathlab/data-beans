use super::*;
use crate::sparse_io::open_sparse_matrix;
use crate::zarr_io::apply_zip_flag;
use hdf5::types::VarLenUnicode;

fn strings(g: &hdf5::Group, name: &str, xs: &[&str]) {
    let v: Vec<VarLenUnicode> = xs.iter().map(|s| s.parse().unwrap()).collect();
    g.new_dataset_builder().with_data(&v).create(name).unwrap();
}

/// A `molecule_info.h5` as Cell Ranger writes it: three features (the
/// last without a molecule), a library of each type, one of them with a
/// string id, molecules left out by `umi_type`, library type and cell
/// calling, and a called cell without a molecule. With `probe_set`, the
/// expression library is probe-based, targeting the first and last
/// features.
fn molecule_info(path: &std::path::Path, probe_set: Option<&str>) {
    let f = hdf5::File::create(path).unwrap();
    let features = f.create_group("features").unwrap();
    strings(&features, "id", &["FID1", "FID2", "FID3"]);
    strings(&features, "name", &["GENE1", "GENE2", "GENE3"]);
    strings(&features, "feature_type", &["Gene Expression"; 3]);
    strings(&f, "barcodes", &["AAAC", "AAAG", "AAAT", "AACA"]);
    let target = match probe_set {
        Some(name) => {
            features
                .create_group("target_sets")
                .unwrap()
                .new_dataset_builder()
                .with_data(&[0u32, 2])
                .create(name)
                .unwrap();
            format!(r#", "target_set_name": "{name}""#)
        }
        None => String::new(),
    };
    let libraries = format!(
        r#"[{{"gem_group": 1, "library_id": "0", "library_type": "Gene Expression"{target}}},
            {{"gem_group": 1, "library_id": 1, "library_type": "Antibody Capture"}}]"#
    );
    strings(&f, "library_info", &[&libraries]);
    let column = |name: &str, v: &[u64]| {
        f.new_dataset_builder().with_data(v).create(name).unwrap();
    };
    // Molecules: barcode, feature, reads, library, counted.
    let m: [[u64; 5]; 6] = [
        [0, 0, 3, 0, 1],
        [0, 0, 2, 0, 1],
        [0, 1, 4, 0, 0], // not counted (umi_type 0)
        [1, 1, 1, 0, 1],
        [1, 0, 7, 1, 1], // another library type
        [2, 0, 1, 0, 1], // not a cell
    ];
    let col = |j: usize| m.iter().map(|r| r[j]).collect::<Vec<_>>();
    column("barcode_idx", &col(0));
    column("feature_idx", &col(1));
    column("count", &col(2));
    column("library_idx", &col(3));
    column("umi_type", &col(4));
    column("gem_group", &[1; 6]);
    // Cells: [barcode, library, genome]; the last has no molecule.
    let cells = ndarray::arr2(&[[0u64, 0, 0], [1, 0, 0], [1, 1, 0], [3, 0, 0]]);
    f.create_group("barcode_info")
        .unwrap()
        .new_dataset_builder()
        .with_data(&cells)
        .create("pass_filter")
        .unwrap();
}

/// What the reader wrote: row and column names, values row by row.
struct Written {
    rows: Vec<Box<str>>,
    columns: Vec<Box<str>>,
    values: Vec<Vec<f32>>,
}

fn read(h5: &std::path::Path, out: &std::path::Path, sum_reads: bool) -> Written {
    let args = From10xMoleculeArgs {
        h5_file: h5.to_str().unwrap().into(),
        backend: SparseIoBackend::Zarr,
        output: out.to_str().unwrap().into(),
        zip: true,
        library_type: "Gene Expression".into(),
        select_row_type: "".into(),
        remove_row_type: "".into(),
        no_pass_filter: false,
        sum_reads,
        do_squeeze: false,
        row_nnz_cutoff: 1,
        column_nnz_cutoff: 1,
        block_size: None,
        view: crate::handlers::explore::PlanArgs::batch(),
    };
    run_build_from_10x_molecule(&args).unwrap();
    let data = open_sparse_matrix(
        &apply_zip_flag(&args.output, true, &SparseIoBackend::Zarr),
        &SparseIoBackend::Zarr,
    )
    .unwrap();
    let m = data
        .read_columns_dmatrix((0..data.num_columns().unwrap()).collect())
        .unwrap();
    Written {
        rows: data.row_names().unwrap(),
        columns: data.column_names().unwrap(),
        values: m.row_iter().map(|r| r.iter().copied().collect()).collect(),
    }
}

#[test]
fn molecules_count_as_cell_ranger_counts_them() {
    let dir = tempfile::tempdir().unwrap();
    let h5 = dir.path().join("molecule_info.h5");
    molecule_info(&h5, None);

    let umis = read(&h5, &dir.path().join("umis"), false);
    // Every feature and every called cell, with or without molecules.
    assert_eq!(
        umis.rows,
        ["FID1_GENE1", "FID2_GENE2", "FID3_GENE3"].map(Box::from)
    );
    assert_eq!(umis.columns, ["AAAC-1", "AAAG-1", "AACA-1"].map(Box::from));
    assert_eq!(umis.values, [[2., 0., 0.], [0., 1., 0.], [0., 0., 0.]]);

    let reads = read(&h5, &dir.path().join("reads"), true);
    assert_eq!(reads.values, [[5., 0., 0.], [0., 1., 0.], [0., 0., 0.]]);
}

#[test]
fn a_probe_based_library_counts_the_genes_its_probes_target() {
    let dir = tempfile::tempdir().unwrap();
    let h5 = dir.path().join("molecule_info.h5");
    molecule_info(&h5, Some("SET1"));

    let umis = read(&h5, &dir.path().join("umis"), false);
    assert_eq!(umis.rows, ["FID1_GENE1", "FID3_GENE3"].map(Box::from));
    assert_eq!(umis.columns, ["AAAC-1", "AAAG-1", "AACA-1"].map(Box::from));
    assert_eq!(umis.values, [[2., 0., 0.], [0., 0., 0.]]);
}
