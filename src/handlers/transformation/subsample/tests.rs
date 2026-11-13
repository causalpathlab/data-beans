//! Round-trips for the subsample write path — its first tests, written against
//! the materialising implementation to pin behaviour before the streaming
//! rewrite. What must survive: which cells and genes are kept (seeded, so the
//! selection is reproducible), the ascending renumbering, names, and values.

use super::*;

/// `arr` as a backend at `path`, rows `g{r}` and columns `cell{c}`.
fn store(arr: &ndarray::Array2<f32>, path: &str) -> anyhow::Result<()> {
    let mut data = create_sparse_from_ndarray(arr, Some(path), Some(&SparseIoBackend::Zarr))?;
    let rows: Vec<Box<str>> = (0..arr.nrows()).map(|r| format!("g{r}").into()).collect();
    let cols: Vec<Box<str>> = (0..arr.ncols())
        .map(|c| format!("cell{c}").into())
        .collect();
    data.register_row_names_vec(&rows);
    data.register_column_names_vec(&cols);
    Ok(())
}

fn fixture_at(path: &str) -> anyhow::Result<()> {
    // 4 genes x 6 cells, all values distinct so aliasing cannot hide.
    let mut arr = ndarray::Array2::<f32>::zeros((4, 6));
    for r in 0..4 {
        for c in 0..6 {
            arr[(r, c)] = (r * 10 + c + 1) as f32;
        }
    }
    store(&arr, path)
}

/// The output `run_subsample` wrote for `-o output --no-zip`.
fn open_out(output: &std::path::Path) -> anyhow::Result<Box<dyn SparseIo<IndexIter = Vec<usize>>>> {
    open_sparse_matrix(
        &format!("{}.zarr", output.to_str().expect("utf8")),
        &SparseIoBackend::Zarr,
    )
}

/// The indices of the cells an output kept, from their `cell{c}` names.
fn kept_cells(output: &std::path::Path) -> anyhow::Result<Vec<usize>> {
    Ok(open_out(output)?
        .column_names()?
        .iter()
        .map(|n| n.trim_start_matches("cell").parse().expect("index"))
        .collect())
}

fn args(input: &str, output: &str) -> SubsampleArgs {
    SubsampleArgs {
        data_file: input.into(),
        cells: None,
        cell_frac: None,
        genes: None,
        gene_frac: None,
        groups: None,
        per_group: None,
        per_group_frac: None,
        sketch: false,
        sketch_dim: 50,
        sketch_bits: 10,
        seed: 11,
        backend: SparseIoBackend::Zarr,
        output: output.into(),
        zip: false,
        view: crate::handlers::explore::PlanArgs::batch(),
    }
}

#[test]
fn an_output_that_is_the_input_is_refused_and_the_input_kept() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let input = dir.path().join("in.zarr");
    let input = input.to_str().expect("utf8");
    fixture_at(input)?;
    // `-o in --no-zip` is in.zarr itself, cleared before it is read.
    let mut a = args(input, dir.path().join("in").to_str().expect("utf8"));
    a.cells = Some(3);
    let err = run_subsample(&a).unwrap_err();
    assert!(format!("{err:#}").contains("is the input"), "{err:#}");
    let kept = open_sparse_matrix(input, &SparseIoBackend::Zarr)?;
    assert_eq!(kept.num_columns(), Some(6));
    assert_eq!(kept.num_non_zeros(), Some(24));
    Ok(())
}

#[test]
fn a_cell_subsample_keeps_whole_columns_untouched() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let input = dir.path().join("in.zarr");
    fixture_at(input.to_str().expect("utf8"))?;
    let output = dir.path().join("out");
    let mut a = args(
        input.to_str().expect("utf8"),
        output.to_str().expect("utf8"),
    );
    a.cells = Some(3);
    run_subsample(&a)?;

    let out = open_out(&output)?;
    assert_eq!(out.num_rows(), Some(4), "no gene filter: full row axis");
    assert_eq!(out.num_columns(), Some(3));
    // Every kept column must be the original column for its name, bit for bit.
    let src = open_sparse_matrix(input.to_str().expect("utf8"), &SparseIoBackend::Zarr)?;
    let src_names = src.column_names()?;
    let src_dense = src.read_columns_dmatrix((0..6).collect())?;
    let out_dense = out.read_columns_dmatrix((0..3).collect())?;
    for (k, name) in out.column_names()?.iter().enumerate() {
        let orig = src_names
            .iter()
            .position(|n| n == name)
            .expect("kept cell exists in the source");
        for r in 0..4 {
            assert_eq!(
                out_dense[(r, k)],
                src_dense[(r, orig)],
                "cell {name} row {r}"
            );
        }
    }
    Ok(())
}

#[test]
fn a_gene_and_cell_subsample_renumbers_ascending() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let input = dir.path().join("in.zarr");
    fixture_at(input.to_str().expect("utf8"))?;
    let output = dir.path().join("out");
    let mut a = args(
        input.to_str().expect("utf8"),
        output.to_str().expect("utf8"),
    );
    a.cells = Some(4);
    a.genes = Some(2);
    run_subsample(&a)?;

    let out = open_out(&output)?;
    assert_eq!(out.num_rows(), Some(2));
    assert_eq!(out.num_columns(), Some(4));
    // The kept gene names identify the original rows; values must follow them.
    let src = open_sparse_matrix(input.to_str().expect("utf8"), &SparseIoBackend::Zarr)?;
    let src_dense = src.read_columns_dmatrix((0..6).collect())?;
    let src_rows = src.row_names()?;
    let src_cols = src.column_names()?;
    let out_dense = out.read_columns_dmatrix((0..4).collect())?;
    for (rk, rname) in out.row_names()?.iter().enumerate() {
        let ro = src_rows
            .iter()
            .position(|n| n == rname)
            .expect("gene exists");
        for (ck, cname) in out.column_names()?.iter().enumerate() {
            let co = src_cols
                .iter()
                .position(|n| n == cname)
                .expect("cell exists");
            assert_eq!(out_dense[(rk, ck)], src_dense[(ro, co)], "{rname}/{cname}");
        }
    }
    // Ascending selection: kept names appear in their original relative order.
    let kept: Vec<usize> = out
        .row_names()?
        .iter()
        .map(|n| src_rows.iter().position(|m| m == n).expect("gene"))
        .collect();
    assert!(
        kept.windows(2).all(|w| w[0] < w[1]),
        "gene order preserved: {kept:?}"
    );
    Ok(())
}

fn group_of(c: usize) -> &'static str {
    match c {
        0..20 => "A",
        20..35 => "B",
        _ => "C",
    }
}

/// 40 cells over 3 groups: A has 20, B 15, C 5. Values encode the cell.
fn grouped_fixture(dir: &std::path::Path) -> anyhow::Result<(String, String)> {
    let (nrow, ncol) = (6, 40);
    let mut arr = ndarray::Array2::<f32>::zeros((nrow, ncol));
    for c in 0..ncol {
        arr[(c % nrow, c)] = (c + 1) as f32;
    }
    let path = dir.join("g.zarr").to_str().expect("utf8").to_string();
    store(&arr, &path)?;
    let table = dir.join("groups.tsv");
    let body: String = (0..ncol)
        .map(|c| format!("cell{c}\t{}\n", group_of(c)))
        .collect();
    std::fs::write(&table, body)?;
    Ok((path, table.to_str().expect("utf8").to_string()))
}

#[test]
fn a_per_group_quota_keeps_n_of_each_and_small_groups_whole() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let (input, table) = grouped_fixture(dir.path())?;
    let output = dir.path().join("out");
    let mut a = args(&input, output.to_str().expect("utf8"));
    a.groups = Some(table.into());
    a.per_group = Some(8);
    run_subsample(&a)?;

    let kept = kept_cells(&output)?;
    assert_eq!(kept.len(), 8 + 8 + 5, "C has only 5, kept whole");
    assert!(kept.windows(2).all(|w| w[0] < w[1]), "column order kept");
    for g in ["A", "B", "C"] {
        let n = kept.iter().filter(|&&c| group_of(c) == g).count();
        assert_eq!(n, if g == "C" { 5 } else { 8 }, "group {g}");
    }
    // The labels beside it, in the output's column order.
    let labels = legume_numeric::matrix::common_io::read_lines(&format!(
        "{}.batch.gz",
        output.to_str().expect("utf8")
    ))?;
    let want: Vec<&str> = kept.iter().map(|&c| group_of(c)).collect();
    let got: Vec<&str> = labels.iter().map(AsRef::as_ref).collect();
    assert_eq!(got, want);
    Ok(())
}

#[test]
fn a_per_group_fraction_and_the_batch_file_form_agree_with_the_table() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let (input, _) = grouped_fixture(dir.path())?;
    // One label per line, in column order: qc's {out}.batch.gz.
    let batch = dir.path().join("in.batch");
    let body: String = (0..40).map(|c| format!("{}\n", group_of(c))).collect();
    std::fs::write(&batch, body)?;
    let output = dir.path().join("frac");
    let mut a = args(&input, output.to_str().expect("utf8"));
    a.groups = Some(batch.to_str().expect("utf8").into());
    a.per_group_frac = Some(0.5);
    run_subsample(&a)?;
    let out = open_out(&output)?;
    // Half of 20, 15 and 5, rounded half away from zero: 10 + 8 + 3.
    assert_eq!(out.num_columns(), Some(21));
    Ok(())
}

#[test]
fn groups_refuse_a_pool_wide_count() {
    use clap::Parser;
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        s: SubsampleArgs,
    }
    let parsed = Cli::try_parse_from([
        "x", "in.zarr", "-o", "o", "--groups", "g.tsv", "--cells", "5",
    ]);
    assert!(
        parsed.is_err(),
        "--cells is pool-wide; a quota is per group"
    );
    let parsed = Cli::try_parse_from(["x", "in.zarr", "-o", "o", "--per-group", "5"]);
    assert!(parsed.is_err(), "--per-group needs --groups");
}

#[test]
fn even_shares_fill_small_bins_and_split_the_rest() {
    let mut rng = SmallRng::seed_from_u64(1);
    let shares = sketch::even_shares(&mut rng, &[2, 100, 50, 1], 31);
    assert_eq!(shares.iter().sum::<usize>(), 31);
    assert_eq!((shares[0], shares[3]), (2, 1), "small bins whole");
    // 28 left over two big bins: 14 each.
    assert_eq!((shares[1], shares[2]), (14, 14));
    // More than there is: everything.
    let all = sketch::even_shares(&mut rng, &[3, 4], 100);
    assert_eq!(all, vec![3, 4]);
    // An uneven remainder goes one each, never past a bin's size.
    let odd = sketch::even_shares(&mut rng, &[5, 5, 5], 7);
    assert_eq!(odd.iter().sum::<usize>(), 7);
    assert!(odd.iter().all(|&s| (2..=3).contains(&s)));
}

/// 200 cells: 190 of one state (genes 0-9), 10 of a rare one (genes 10-19).
/// A uniform draw of 20 expects one rare cell; the sketch keeps many more.
#[cfg(feature = "alg")]
#[test]
fn the_sketch_keeps_a_rare_state() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let (nrow, ncol, rare) = (20, 200, 10);
    let mut arr = ndarray::Array2::<f32>::zeros((nrow, ncol));
    for c in 0..ncol {
        let base = if c < rare { 10 } else { 0 };
        for g in 0..10 {
            arr[(base + g, c)] = 1.0 + ((c * 7 + g * 3) % 5) as f32;
        }
    }
    let input = dir.path().join("s.zarr");
    let path = input.to_str().expect("utf8");
    store(&arr, path)?;

    let output = dir.path().join("sk");
    let mut a = args(path, output.to_str().expect("utf8"));
    a.cells = Some(20);
    a.sketch = true;
    a.sketch_dim = 10;
    a.sketch_bits = 2;
    run_subsample(&a)?;
    let kept = kept_cells(&output)?;
    assert_eq!(kept.len(), 20);
    let n_rare = kept.iter().filter(|&&c| c < rare).count();
    assert!(n_rare >= 5, "the rare state is up-weighted: {n_rare} of 20");
    Ok(())
}
