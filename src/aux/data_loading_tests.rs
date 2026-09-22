//! `read_data_on_shared_rows` under the default (auto) naming rule, with
//! files that spell their rows differently.

use super::{read_data_on_shared_rows, ReadSharedRowsArgs};
use crate::sparse_io::SparseIoBackend;
use nalgebra::DMatrix;

/// A tiny zarr backend whose rows are `rows` and whose single column holds
/// a count for every row, so no row is dropped as empty.
fn backend(dir: &std::path::Path, name: &str, rows: &[&str]) -> Box<str> {
    let path = dir.join(format!("{name}.zarr"));
    let path: Box<str> = path.to_string_lossy().into_owned().into();
    let m = DMatrix::<f32>::from_fn(rows.len(), 2, |i, j| (i + j + 1) as f32);
    let mut b =
        crate::sparse_io::create_sparse_from_dmatrix(&m, Some(&path), Some(&SparseIoBackend::Zarr))
            .expect("backend");
    let rows: Vec<Box<str>> = rows.iter().map(|r| (*r).into()).collect();
    b.register_row_names_vec(&rows);
    let cols: Vec<Box<str>> = (0..2).map(|j| format!("{name}_c{j}").into()).collect();
    b.register_column_names_vec(&cols);
    path
}

fn load_rows(files: Vec<Box<str>>) -> usize {
    let loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: files,
        ..Default::default()
    })
    .expect("load");
    loaded.data.num_rows()
}

#[test]
fn a_raw_cohort_and_its_canonical_reference_share_one_axis() {
    // Pooled: 2 gene-like of 5 names = 40% < 50%, which used to sniff Exact
    // and give 5 rows. Per file: Gene + Exact -> Gene -> 3 rows.
    let dir = tempfile::tempdir().unwrap();
    let raw = backend(dir.path(), "raw", &["ENSG1_A", "ENSG2_B"]);
    let canon = backend(dir.path(), "canon", &["A", "B", "C"]);
    assert_eq!(load_rows(vec![raw, canon]), 3);
}

#[test]
fn overlapping_loci_across_files_still_merge() {
    // The locus-overlap map needs every file's names at once; per-file
    // detection must not take that pool away.
    let dir = tempfile::tempdir().unwrap();
    let a = backend(dir.path(), "a", &["chr1:1-20", "chr2:1-10"]);
    let b = backend(dir.path(), "b", &["chr1:15-30", "chr3:1-10"]);
    assert_eq!(load_rows(vec![a, b]), 3);
}

////////////////////////////////////////////////////////////////////
// empty-barcode gate                                              //
////////////////////////////////////////////////////////////////////

/// A zarr backend whose column `j` holds `nnz[j]` ones.
fn backend_nnz(
    dir: &std::path::Path,
    name: &str,
    rows: &[Box<str>],
    cols: &[(String, usize)],
) -> Box<str> {
    let path = dir.join(format!("{name}.zarr"));
    let path: Box<str> = path.to_string_lossy().into_owned().into();
    let m = DMatrix::<f32>::from_fn(rows.len(), cols.len(), |i, j| {
        if i < cols[j].1 {
            1.0
        } else {
            0.0
        }
    });
    let mut b =
        crate::sparse_io::create_sparse_from_dmatrix(&m, Some(&path), Some(&SparseIoBackend::Zarr))
            .expect("backend");
    b.register_row_names_vec(rows);
    let names: Vec<Box<str>> = cols.iter().map(|(c, _)| c.as_str().into()).collect();
    b.register_column_names_vec(&names);
    path
}

/// RNA: 40 called cells. ATAC: the same 40 barcodes (cell `c0` barely
/// seen in ATAC) plus 400 ambient barcodes with 1–4 peaks.
fn multiome_pair(dir: &std::path::Path) -> (Box<str>, Box<str>) {
    let genes: Vec<Box<str>> = (0..100).map(|i| format!("GENE{i}").into()).collect();
    let peaks: Vec<Box<str>> = (0..200)
        .map(|i| format!("chr1:{}-{}", i * 100, i * 100 + 50).into())
        .collect();
    let cells: Vec<(String, usize)> = (0..40).map(|j| (format!("c{j}"), 80)).collect();
    let rna = backend_nnz(dir, "rna", &genes, &cells);
    let mut atac_cols: Vec<(String, usize)> = (0..40)
        .map(|j| (format!("c{j}"), if j == 0 { 2 } else { 120 + j }))
        .collect();
    atac_cols.extend((0..400).map(|j| (format!("a{j}"), 1 + j % 4)));
    let atac = backend_nnz(dir, "atac", &peaks, &atac_cols);
    (rna, atac)
}

fn load_union(files: Vec<Box<str>>, keep_empty_barcodes: bool) -> super::SparseDataWithBatch {
    read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: files,
        column_alignment: crate::sparse_io_vector::ColumnAlignment::Union,
        keep_empty_barcodes,
        ..Default::default()
    })
    .expect("load")
}

#[test]
fn empty_barcodes_seen_only_in_atac_are_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let (rna, atac) = multiome_pair(dir.path());
    let loaded = load_union(vec![rna, atac], false);
    assert_eq!(loaded.data.num_columns(), 40);
    assert_eq!(loaded.batch.len(), 40, "batch labels filtered in lockstep");
    let names = loaded.data.column_names().unwrap();
    assert!(
        names.iter().all(|n| n.starts_with('c')),
        "only cells remain"
    );
    assert!(
        names.iter().any(|n| n.as_ref() == "c0"),
        "a cell weak in ATAC survives on its RNA"
    );
}

#[test]
fn keep_empty_barcodes_opts_out() {
    let dir = tempfile::tempdir().unwrap();
    let (rna, atac) = multiome_pair(dir.path());
    let loaded = load_union(vec![rna, atac], true);
    assert_eq!(loaded.data.num_columns(), 440);
}

#[test]
fn a_single_unfiltered_file_is_cell_called() {
    let dir = tempfile::tempdir().unwrap();
    let (_, atac) = multiome_pair(dir.path());
    let loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: vec![atac],
        ..Default::default()
    })
    .expect("load");
    // c0 has no other modality to vouch for it.
    assert_eq!(loaded.data.num_columns(), 39);
    assert_eq!(loaded.batch.len(), 39);
}

#[test]
fn called_cells_are_left_alone() {
    let dir = tempfile::tempdir().unwrap();
    let (rna, _) = multiome_pair(dir.path());
    let loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: vec![rna],
        ..Default::default()
    })
    .expect("load");
    assert_eq!(loaded.data.num_columns(), 40);
}
