//! A backend written out by `data_beans::export` and read back by
//! `data_beans::import` is the backend it was: counts, feature and cell
//! names, for each interchange format.

use super::*;
use crate::export::{self, MtxExport};
use crate::sparse_io::{create_sparse_from_triplets, open_sparse_matrix_by_path};

/// 3 features x 4 cells, a cell without a count, features named `id_name`.
fn genes_by_cells() -> Vec<Vec<f32>> {
    vec![
        vec![1.0, 0.0, 5.0, 0.0],
        vec![0.0, 2.0, 0.0, 0.0],
        vec![3.0, 4.0, 7.0, 0.0],
    ]
}

fn names(xs: &[&str]) -> Vec<Box<str>> {
    xs.iter().map(|&x| Box::from(x)).collect()
}

/// A backend at `dir/source.zarr` holding [`genes_by_cells`].
fn source(dir: &std::path::Path) -> Box<str> {
    let m = genes_by_cells();
    let triplets: Vec<(u64, u64, f32)> = m
        .iter()
        .enumerate()
        .flat_map(|(i, row)| {
            row.iter()
                .enumerate()
                .filter(|(_, &v)| v != 0.0)
                .map(move |(j, &v)| (i as u64, j as u64, v))
        })
        .collect();
    let file = dir.join("source.zarr").to_string_lossy().into_owned();
    let shape = (m.len(), m[0].len(), triplets.len());
    let mut data = create_sparse_from_triplets(&triplets, shape, Some(&file), None).unwrap();
    data.register_row_names_vec(&names(&["G1_A", "G2_B", "G3_C"]));
    data.register_column_names_vec(&names(&["AAAC-1", "AAAG-1", "AACT-1", "AAGT-1"]));
    file.into()
}

/// A backend's counts row by row, and its row and column names.
struct Contents {
    counts: Vec<Vec<f32>>,
    rows: Vec<Box<str>>,
    columns: Vec<Box<str>>,
}

/// The backend at `file`.
fn contents(file: &str) -> Contents {
    let data = open_sparse_matrix_by_path(file).unwrap();
    let ncol = data.num_columns().unwrap();
    let dense = data.read_columns_dmatrix((0..ncol).collect()).unwrap();
    let rows = dense
        .row_iter()
        .map(|r| r.iter().copied().collect())
        .collect();
    Contents {
        counts: rows,
        rows: data.row_names().unwrap(),
        columns: data.column_names().unwrap(),
    }
}

#[test]
fn a_backend_round_trips_through_mtx_gzipped_or_not() {
    let dir = tempfile::tempdir().unwrap();
    let data = open_sparse_matrix_by_path(&source(dir.path())).unwrap();
    for gzip in [true, false] {
        let out_dir = dir.path().join(format!("mtx_{gzip}"));
        let out_dir = out_dir.to_str().unwrap();
        let opts = MtxExport {
            gzip,
            ..MtxExport::default()
        };
        assert_eq!(
            &*export::to_mtx(data.as_ref(), out_dir, &opts).unwrap(),
            out_dir
        );

        let gz = if gzip { ".gz" } else { "" };
        let mtx = MtxImport {
            row: Some(format!("{out_dir}/features.tsv{gz}").into()),
            col: Some(format!("{out_dir}/barcodes.tsv{gz}").into()),
            ..MtxImport::new(&format!("{out_dir}/matrix.mtx{gz}"))
        };
        let output = dir.path().join(format!("back_{gzip}"));
        let back = from_mtx(&mtx, &Output::new(output.to_str().unwrap())).unwrap();
        assert!(back.ends_with(".zarr.zip"), "{back}");

        let Contents {
            counts,
            rows,
            columns,
        } = contents(&back);
        assert_eq!(counts, genes_by_cells());
        assert_eq!(rows, names(&["G1_A", "G2_B", "G3_C"]));
        assert_eq!(columns, data.column_names().unwrap());
    }
}

/// A squeeze asked of the import drops the cell without a count.
#[test]
fn an_import_squeezes_as_its_output_asks() {
    let dir = tempfile::tempdir().unwrap();
    let data = open_sparse_matrix_by_path(&source(dir.path())).unwrap();
    let out_dir = dir.path().join("mtx");
    let out_dir = out_dir.to_str().unwrap();
    export::to_mtx(data.as_ref(), out_dir, &MtxExport::default()).unwrap();

    let mtx = MtxImport {
        col: Some(format!("{out_dir}/barcodes.tsv.gz").into()),
        ..MtxImport::new(&format!("{out_dir}/matrix.mtx.gz"))
    };
    let out = Output {
        zip: false,
        squeeze: Some(Squeeze::default()),
        ..Output::new(dir.path().join("squeezed").to_str().unwrap())
    };
    let back = from_mtx(&mtx, &out).unwrap();
    assert!(back.ends_with("squeezed.zarr"), "{back}");
    let Contents {
        counts,
        rows,
        columns,
    } = contents(&back);
    assert_eq!(counts.iter().map(Vec::len).max(), Some(3));
    assert_eq!(rows, names(&["1", "2", "3"]));
    assert_eq!(columns, names(&["AAAC-1", "AAAG-1", "AACT-1"]));
}

#[cfg(feature = "hdf5")]
#[test]
fn a_backend_round_trips_through_h5ad() {
    let dir = tempfile::tempdir().unwrap();
    let file = source(dir.path());
    let mut data = open_sparse_matrix_by_path(&file).unwrap();
    let h5ad = export::h5ad_path(&file, None);
    assert_eq!(h5ad, dir.path().join("source.h5ad").to_str().unwrap());
    export::to_h5ad(data.as_mut(), &h5ad, &export::H5adExport::default()).unwrap();

    let output = dir.path().join("back");
    let back = from_h5ad(
        &H5adImport::new(&h5ad),
        &Output::new(output.to_str().unwrap()),
    )
    .unwrap();
    let Contents {
        counts,
        rows,
        columns,
    } = contents(&back);
    assert_eq!(counts, genes_by_cells());
    // `var/_index` holds the full name; with no symbol column it is the id.
    assert_eq!(rows, names(&["G1_A", "G2_B", "G3_C"]));
    assert_eq!(columns, data.column_names().unwrap());
    assert!(dir.path().join("back.cell_metadata.tsv.gz").exists());
}

/// Cells read from several files are named as `SparseIoVec` names them.
#[test]
fn stacked_cells_are_named_by_their_file_when_there_are_several() {
    use crate::sparse_io_vector::{data_name_of, stacked_column_name, SparseIoVec};
    use std::sync::Arc;

    let dir = tempfile::tempdir().unwrap();
    let file = source(dir.path());
    assert_eq!(&*stacked_column_name("AAAC-1", &file, 1).unwrap(), "AAAC-1");
    assert_eq!(
        &*stacked_column_name("AAAC-1", "x/pbmc.zarr.zip", 2).unwrap(),
        "AAAC-1@pbmc"
    );

    let mut stacked = SparseIoVec::new();
    for _ in 0..2 {
        let data = open_sparse_matrix_by_path(&file).unwrap();
        stacked
            .push(Arc::from(data), data_name_of(&file, 2).unwrap())
            .unwrap();
    }
    let expected: Vec<Box<str>> = (0..2)
        .flat_map(|_| ["AAAC-1", "AAAG-1", "AACT-1", "AAGT-1"])
        .map(|c| stacked_column_name(c, &file, 2).unwrap())
        .collect();
    assert_eq!(stacked.column_names().unwrap(), expected);
}
