use crate::handlers::transformation::squeeze_in_place;
use crate::hdf5_io::*;
use crate::sparse_data_visitors::styled_progress_bar;
use crate::sparse_io::*;
use crate::utilities::io_helpers::{read_col_names, read_row_names};

use crate::zarr_io::{apply_zip_flag, finalize_zarr_output, materialize_writable_backend};
use clap::Args;
use data_beans::sparse_data_visitors::create_jobs;
use legume_numeric::matrix::common_io::*;
use legume_numeric::matrix::mtx_io;
use legume_numeric::matrix::utils::byte_budget_intervals;
use log::info;
use rayon::prelude::*;

use rustc_hash::FxHashMap as HashMap;

#[path = "merge_plan.rs"]
mod plan;
pub use crate::handlers::explore::PlanArgs;

#[derive(Args, Debug)]
pub struct AlignDataArgs {
    /// data file -- either `.zarr` or `.h5`
    #[arg(required = true, value_delimiter = ',')]
    pub data_files: Vec<Box<str>>,

    /// Data types (treating them as different rows)
    #[arg(short = 'r', long, required = true)]
    pub num_data_types: usize,

    /// output directory
    #[arg(short, long, required = true)]
    pub output_directory: Box<str>,

    #[command(flatten)]
    pub view: PlanArgs,
}

#[derive(Args, Debug)]
pub struct MergeBackendArgs {
    #[arg(
        value_delimiter = ',',
        help = "Input data files",
        long_help = "Data files to be merged into a single backend.\n\
                     Provide one or more files in supported formats."
    )]
    pub data_files: Vec<Box<str>>,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Backend format",
        long_help = "Specify the backend format to use for the merged data.\n\
                     Supported formats include 'zarr', 'h5', etc."
    )]
    pub backend: SparseIoBackend,

    #[arg(
        short,
        long,
        required = true,
        help = "Output file header",
        long_help = "Output file header: {output}.{backend} and {output}.batch.gz,\n\
                     {output} without a backend suffix it was given.\n\
                     The zarr backend produces {output}.zarr.zip by default;\n\
                     pass --no-zip to keep a {output}.zarr directory instead.\n\
                     The backend will contain everything.\n\
                     Batch assignment information goes to a separate file.\n\
                     Later embedding steps need it."
    )]
    pub output: Box<str>,

    /// keep a `.zarr` directory instead of producing a `.zarr.zip` archive
    #[arg(long = "no-zip", default_value_t = true, action = clap::ArgAction::SetFalse)]
    pub zip: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Keep only the rows every backend has",
        long_help = "Keep only the rows (features) every backend has, matched by name,\n\
                     in the first backend's order. Without it, the backends must have\n\
                     the same rows in the same order."
    )]
    pub common_rows: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Squeeze sparse rows/columns",
        long_help = "Enable squeezing to remove rows and columns with too few non-zeros.\n\
                     This can help reduce file size and improve performance."
    )]
    pub do_squeeze: bool,

    #[arg(
        long,
        default_value_t = 1,
        help = "Row non-zero cutoff",
        long_help = "Minimum number of non-zero elements required for rows.\n\
                     Rows with fewer non-zeros will be removed if squeezing is enabled."
    )]
    pub row_nnz_cutoff: usize,

    #[arg(
        long,
        default_value_t = 1,
        help = "Column non-zero cutoff",
        long_help = "Minimum number of non-zero elements required for columns.\n\
                     Columns with fewer non-zeros will be removed if squeezing is enabled."
    )]
    pub column_nnz_cutoff: usize,

    #[arg(
        long,
        help = "Cells per rayon job (omit for auto-scaling by feature count)"
    )]
    pub block_size: Option<usize>,

    #[command(flatten)]
    pub view: PlanArgs,
}

#[derive(clap::Args, Debug)]
pub struct MergeMtxArgs {
    #[arg(
        value_delimiter = ',',
        required = true,
        help = "Input data directories",
        long_help = "Each directory and its sub-directories are searched.\n\
                     The program looks for files named by: (1) `mtx_file_name`,\n\
                     (2) `feature_file_name`, (3) `barcode_file_name`.\n\
                     Those are merged into one backend file."
    )]
    pub data_directories: Vec<Box<str>>,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Backend format",
        long_help = "Specify the backend format for the merged data.\n\
                     Supported formats include 'zarr', 'h5', etc."
    )]
    pub backend: SparseIoBackend,

    #[arg(
        short,
        long,
        required = true,
        help = "Output file header",
        long_help = "Output file header: {output}.{backend} and {output}.batch.gz,\n\
                     {output} without a backend suffix it was given.\n\
                     The zarr backend produces {output}.zarr.zip by default;\n\
                     pass --no-zip to keep a {output}.zarr directory instead.\n\
                     The backend will contain all merged data.\n\
                     Batch assignment information goes to a separate file.\n\
                     Later embedding steps need it."
    )]
    pub output: Box<str>,

    /// keep a `.zarr` directory instead of producing a `.zarr.zip` archive
    #[arg(long = "no-zip", default_value_t = true, action = clap::ArgAction::SetFalse)]
    pub zip: bool,

    #[arg(
        short,
        long,
        default_value = "matrix.mtx",
        help = "Matrix file name",
        long_help = "Name of the matrix file to search for in each directory.\n\
                     The default for 10x data is 'matrix.mtx'."
    )]
    pub mtx_file_name: Box<str>,

    #[arg(
        short,
        long,
        default_value = "genes.tsv.gz",
        help = "Feature/row file name",
        long_help = "Name of the feature (row) file to search for in each directory.\n\
                     The default is 'genes.tsv.gz'."
    )]
    pub feature_file_name: Box<str>,

    #[arg(
        long,
        default_value_t = 2,
        help = "Number of words for feature names",
        long_help = "Number of words to use when parsing feature names from the feature file.\n\
                     Adjust this to match your data format."
    )]
    pub num_feature_name_words: usize,

    #[arg(
        short,
        long,
        default_value = "barcodes.tsv.gz",
        help = "Barcode/column file name",
        long_help = "Name of the barcode (column) file to search for in each directory.\n\
                     The default is 'barcodes.tsv.gz'."
    )]
    pub barcode_file_name: Box<str>,

    #[arg(
        long,
        default_value_t = 5,
        help = "Number of words for barcode names",
        long_help = "Number of words to use when parsing barcode names from the barcode file.\n\
                     Adjust this to match your data format."
    )]
    pub num_barcode_name_words: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Squeeze sparse rows/columns",
        long_help = "Enable squeezing to remove rows and columns with too few non-zeros.\n\
                     This can help reduce file size and improve performance."
    )]
    pub do_squeeze: bool,

    #[arg(
        long,
        default_value_t = 1,
        help = "Row non-zero cutoff",
        long_help = "Minimum number of non-zero elements required for rows.\n\
                     Rows with fewer non-zeros will be removed if squeezing is enabled."
    )]
    pub row_nnz_cutoff: usize,

    #[arg(
        long,
        default_value_t = 1,
        help = "Column non-zero cutoff",
        long_help = "Minimum number of non-zero elements required for columns.\n\
                     Columns with fewer non-zeros will be removed if squeezing is enabled."
    )]
    pub column_nnz_cutoff: usize,

    #[arg(
        long,
        help = "Cells per rayon job for the triplet read and squeeze pass",
        long_help = "Cells per rayon job for the triplet read and squeeze pass.\n\
                     Omit it for auto-scaling by feature count."
    )]
    pub block_size: Option<usize>,

    #[command(flatten)]
    pub view: PlanArgs,
}

/// Lay triplets out as a CSC slab of `ncol` columns: (colptr, row indices,
/// values), each column's rows ascending and entries of one cell in the
/// order given.
///
/// `colptr[j]` is the local offset of column `j`'s first entry; the trailing
/// sentinel is written by `finalize_streaming_csc`. Entries are counted and
/// placed by column; only a column whose rows do not already ascend is
/// sorted.
fn build_csc_slab(triplets: Vec<(u64, u64, f32)>, ncol: usize) -> (Vec<u64>, Vec<u64>, Vec<f32>) {
    let mut colptr = vec![0u64; ncol + 1];
    for &(_, col, _) in &triplets {
        colptr[col as usize + 1] += 1;
    }
    for j in 0..ncol {
        colptr[j + 1] += colptr[j];
    }
    debug_assert_eq!(colptr[ncol] as usize, triplets.len());

    let mut next = colptr[..ncol].to_vec();
    let mut row_indices = vec![0u64; triplets.len()];
    let mut values = vec![0f32; triplets.len()];
    for (row, col, value) in triplets {
        let at = &mut next[col as usize];
        row_indices[*at as usize] = row;
        values[*at as usize] = value;
        *at += 1;
    }

    for j in 0..ncol {
        let (lo, hi) = (colptr[j] as usize, colptr[j + 1] as usize);
        if row_indices[lo..hi].is_sorted() {
            continue;
        }
        let mut entries: Vec<(u64, f32)> = row_indices[lo..hi]
            .iter()
            .copied()
            .zip(values[lo..hi].iter().copied())
            .collect();
        entries.sort_by_key(|&(row, _)| row);
        for (k, (row, value)) in entries.into_iter().enumerate() {
            row_indices[lo + k] = row;
            values[lo + k] = value;
        }
    }

    colptr.truncate(ncol);
    (colptr, row_indices, values)
}

/// Where each of `rows` lands among the `n` common rows that `place`
/// indexes by name: a name outside them nowhere, and a name `rows` repeats
/// on its first row only.
fn common_row_map(rows: &[Box<str>], place: &HashMap<&str, u64>, n: usize) -> Vec<Option<u64>> {
    let mut seen = vec![false; n];
    rows.iter()
        .map(|name| {
            let i = *place.get(name.as_ref())?;
            (!std::mem::replace(&mut seen[i as usize], true)).then_some(i)
        })
        .collect()
}

pub fn run_merge_backend(args: &MergeBackendArgs) -> anyhow::Result<()> {
    if args.data_files.len() <= 1 {
        info!("no need to merge one file");
        return Ok(());
    }
    plan::merge_backend(args)
}

/// `merge-backend` with the rows and squeeze decided: `common_rows` keeps
/// the rows every file has, `squeeze` drops sparse rows and columns after.
fn merge_backend(args: &MergeBackendArgs, common_rows: bool, squeeze: bool) -> anyhow::Result<()> {
    let num_batches = args.data_files.len();
    info!("merging over {} batches ...", num_batches);

    let effective_output = apply_zip_flag(&args.output, args.zip, &args.backend);
    let (backend, backend_file) =
        resolve_backend_file(&effective_output, Some(args.backend.clone()))?;
    // The output is cleared before the inputs are read and zipped over its
    // name after, so it must not be one of them.
    for input in &args.data_files {
        for out in [&*effective_output, &*backend_file] {
            anyhow::ensure!(
                !same_file(input, out),
                "the output {out} is the input {input}: give -o another name"
            );
        }
    }

    let batch_names = generate_unique_batch_names(&args.data_files)?;
    info!("Batch names: {:?}", batch_names);

    struct BatchHandle {
        path: Box<str>,
        backend: SparseIoBackend,
        ncol: usize,
        nnz: u64,
        col_offset: u64,
        /// Under `--common-rows`, each row's place among the common rows.
        row_map: Option<Vec<Option<u64>>>,
    }

    let mut batches: Vec<BatchHandle> = Vec::with_capacity(num_batches);
    let mut row_names: Vec<Box<str>> = vec![];
    let mut column_names: Vec<Box<str>> = vec![];
    let mut column_batch_names: Vec<Box<str>> = vec![];
    // Under `--common-rows`: each batch's row names, and its row index to
    // count the common rows' entries.
    let mut batch_rows: Vec<Vec<Box<str>>> = vec![];
    let mut batch_indptr: Vec<Vec<u64>> = vec![];
    let mut col_offset: u64 = 0;
    let mut metas: Vec<Metadata> = Vec::with_capacity(num_batches);

    for (batch_idx, data_file) in args.data_files.iter().enumerate() {
        info!("inventorying data file: {}", data_file);

        let backend = match file_ext(data_file)?.to_string().as_str() {
            "h5" => SparseIoBackend::HDF5,
            "zarr" => SparseIoBackend::Zarr,
            _ => SparseIoBackend::Zarr,
        };

        let data = open_sparse_matrix(data_file, &backend)?;
        metas.push(data.metadata());
        let ncol = data
            .num_columns()
            .ok_or_else(|| anyhow::anyhow!("missing ncol in {}", data_file))?;
        let nnz = data
            .num_non_zeros()
            .ok_or_else(|| anyhow::anyhow!("missing nnz in {}", data_file))?;

        if common_rows {
            batch_rows.push(data.row_names()?);
            batch_indptr.push(data.row_indptr().to_vec());
        } else if batch_idx == 0 {
            row_names = data.row_names()?;
        } else {
            info!("checking if the row names are consistent");
            anyhow::ensure!(
                row_names == data.row_names()?,
                "{} has other row names than {}: merge-backend needs the same rows, in order; \
                 subset them to common rows first, or pass --common-rows",
                data_file,
                args.data_files[0]
            );
        }

        let _names = data.column_names()?;
        let batch_name = &batch_names[batch_idx];
        column_names.extend(
            _names
                .into_iter()
                .map(|x| format!("{}{}{}", x, COLUMN_SEP, batch_name).into_boxed_str()),
        );
        column_batch_names.extend(vec![batch_name.clone(); ncol]);

        batches.push(BatchHandle {
            path: data_file.clone(),
            backend,
            ncol,
            nnz: nnz as u64,
            col_offset,
            row_map: None,
        });

        col_offset += ncol as u64;
    }

    if common_rows {
        row_names = common_names(&batch_rows);
        anyhow::ensure!(!row_names.is_empty(), "the backends share no row names");
        info!("{} rows common to all backends", row_names.len());
        let place: HashMap<&str, u64> = row_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_ref(), i as u64))
            .collect();
        for ((h, rows), indptr) in batches.iter_mut().zip(&batch_rows).zip(&batch_indptr) {
            // A name a backend repeats keeps its first row only.
            let map = common_row_map(rows, &place, row_names.len());
            if map.iter().enumerate().all(|(r, m)| *m == Some(r as u64)) {
                continue; // these are the common rows, in order
            }
            // The rows kept hold this many entries: the row index says.
            anyhow::ensure!(
                indptr.len() == rows.len() + 1,
                "{} has no by-row index to count its common rows' entries",
                h.path
            );
            h.nnz = map
                .iter()
                .enumerate()
                .filter(|(_, m)| m.is_some())
                .map(|(r, _)| indptr[r + 1] - indptr[r])
                .sum();
            h.row_map = Some(map);
        }
    }
    let total_ncol = col_offset as usize;
    let total_nnz = batches.iter().map(|h| h.nnz).sum::<u64>() as usize;
    let total_nrow = row_names.len();
    info!(
        "Found {} columns/barcodes across {} batches (total nnz = {})",
        total_ncol, num_batches, total_nnz
    );

    if std::path::Path::new(backend_file.as_ref()).exists() {
        info!(
            "This existing backend file '{}' will be deleted",
            backend_file
        );
        remove_file(&backend_file)?;
    }

    let mut out = create_sparse_streaming_empty(Some(&backend_file), Some(&backend))?;
    out.begin_streaming_csc((total_nrow, total_ncol, total_nnz))?;

    // Each batch goes through in slabs of columns, as many read at once as
    // there are threads and appended in order. The threads share one slab
    // budget, so what is resident does not grow with the batch.
    let threads = rayon::current_num_threads().max(1);
    let prog_bar = styled_progress_bar(num_batches as u64, "batches");
    let mut nnz_offset: u64 = 0;
    for h in &batches {
        let src = open_sparse_matrix(&h.path, &h.backend)?;
        let slabs = match args.block_size {
            Some(_) => create_jobs(h.ncol, total_nrow, args.block_size),
            None => {
                let column_nnz = (0..h.ncol)
                    .map(|c| {
                        src.column_nnz(c).ok_or_else(|| {
                            anyhow::anyhow!("{} has no index entry for column {c}", h.path)
                        })
                    })
                    .collect::<anyhow::Result<Vec<u64>>>()?;
                byte_budget_intervals(&column_nnz, SLAB_BUDGET_BYTES / threads, TRIPLET_BYTES)
            }
        };
        let row_map = h.row_map.as_deref();
        let mut written: u64 = 0;
        for window in slabs.chunks(threads) {
            // A block read returns its columns in order, numbered from the
            // block's first, rows ascending; a row map that reorders is
            // sorted for within each column. A failed read stops the merge:
            // the output's nnz is declared up front, so a block left out
            // would leave a gap of fill values behind.
            let built = window
                .par_iter()
                .map(|&(lb, ub)| -> anyhow::Result<_> {
                    let (_, _, mut triplets) = src.read_triplets_by_columns((lb..ub).collect())?;
                    if let Some(map) = row_map {
                        triplets = triplets
                            .into_iter()
                            .filter_map(|(i, j, x)| Some((map[i as usize]?, j, x)))
                            .collect();
                    }
                    Ok(build_csc_slab(triplets, ub - lb))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            for (&(lb, _), (local_colptr, row_indices, values)) in window.iter().zip(&built) {
                out.append_csc_slab(
                    h.col_offset + lb as u64,
                    nnz_offset + written,
                    local_colptr,
                    row_indices,
                    values,
                )?;
                written += values.len() as u64;
            }
        }
        anyhow::ensure!(
            written == h.nnz,
            "{} holds {written} entries in the rows kept, where its index says {}",
            h.path,
            h.nnz
        );
        nnz_offset += h.nnz;

        prog_bar.inc(1);
    }
    prog_bar.finish_and_clear();

    out.finalize_streaming_csc()?;
    info!("transposing CSC → CSR on disk");
    out.build_csr_from_csc_streaming()?;

    out.register_row_names_vec(&row_names);
    out.register_column_names_vec(&column_names);
    // What every batch says alike: a merge of samples keeps no `sample`.
    out.set_metadata(&meta::common_metadata(&metas))?;
    drop(out);

    info!(
        "Successfully created a sparse backend file: {}",
        backend_file
    );

    let batch_map = column_names
        .into_iter()
        .zip(column_batch_names)
        .collect::<HashMap<_, _>>();

    if squeeze {
        squeeze_in_place(
            &backend_file,
            args.row_nnz_cutoff,
            args.column_nnz_cutoff,
            args.block_size,
        )?;
    }

    write_batches_and_finalize(&effective_output, &backend_file, &backend, &batch_map)?;
    info!("done");
    Ok(())
}

/// A merge's batch labels: `{output}.batch.gz`, the output without its
/// backend suffix.
pub(crate) fn batch_file(effective_output: &str) -> String {
    format!("{}.batch.gz", strip_backend_suffix(effective_output))
}

/// Whether paths `a` and `b` name the same file: compared resolved when
/// they exist, as written (made absolute) when not.
pub(crate) fn same_file(a: &str, b: &str) -> bool {
    let resolve = |p: &str| std::fs::canonicalize(p).or_else(|_| std::path::absolute(p));
    matches!((resolve(a), resolve(b)), (Ok(x), Ok(y)) if x == y)
}

/// The end of a merge: label each column of the merged backend at
/// `backend_file` with its batch (from `batch_map`; a column it misses takes
/// the output's own name) in `{output}.batch.gz`, the output's name without
/// its backend suffix; then finalize the backend as `effective_output`, the
/// output resolved (a `.zarr.zip`, unless `--no-zip`).
pub(crate) fn write_batches_and_finalize(
    effective_output: &str,
    backend_file: &str,
    backend: &SparseIoBackend,
    batch_map: &HashMap<Box<str>, Box<str>>,
) -> anyhow::Result<()> {
    let stem = strip_backend_suffix(effective_output);
    // The output's name itself, dots and all (`sample.v2`), not cut at its
    // last dot.
    let default_batch: Box<str> = std::path::Path::new(stem)
        .file_name()
        .map_or_else(|| stem.into(), |n| n.to_string_lossy().into());
    let labels: Vec<Box<str>> = open_sparse_matrix(backend_file, backend)?
        .column_names()?
        .iter()
        .map(|k| batch_map.get(k).unwrap_or(&default_batch).clone())
        .collect();
    write_lines(&labels, &batch_file(effective_output))?;
    finalize_zarr_output(backend_file, effective_output)
}

/// The names in every list, in the first list's order, each once.
fn common_names(lists: &[Vec<Box<str>>]) -> Vec<Box<str>> {
    let Some((first, rest)) = lists.split_first() else {
        return vec![];
    };
    let others: Vec<rustc_hash::FxHashSet<&str>> = rest
        .iter()
        .map(|l| l.iter().map(|n| n.as_ref()).collect())
        .collect();
    let mut seen = rustc_hash::FxHashSet::default();
    first
        .iter()
        .filter(|&n| others.iter().all(|o| o.contains(n.as_ref())) && seen.insert(n.as_ref()))
        .cloned()
        .collect()
}

pub fn run_merge_mtx(args: &MergeMtxArgs) -> anyhow::Result<()> {
    plan::merge_mtx(args)
}

/// The MTX file sets under `merge-mtx`'s directories, one batch each.
pub(crate) struct MtxSets {
    pub(crate) batch_names: Vec<String>,
    pub(crate) mtx_files: Vec<Box<str>>,
    pub(crate) row_files: Vec<Box<str>>,
    pub(crate) col_files: Vec<Box<str>>,
}

/// The file among `files` (a directory's) that `pattern` names: the one
/// called exactly that, else the one whose path ends with it
/// (`GSM1_matrix.mtx` for `matrix.mtx`). Several ending with it and none
/// called it is an error: which was meant is not determined.
fn file_named(files: &[String], pattern: &str) -> anyhow::Result<Option<Box<str>>> {
    let exact = files.iter().find(|f| {
        std::path::Path::new(f)
            .file_name()
            .is_some_and(|n| n == pattern)
    });
    if let Some(file) = exact {
        return Ok(Some(file.as_str().into()));
    }
    let mut ending = files.iter().filter(|f| f.ends_with(pattern));
    match (ending.next(), ending.next()) {
        (Some(a), Some(b)) => anyhow::bail!(
            "{a} and {b} both end with `{pattern}`: name the one to merge exactly, \
             or keep one set of files to a directory"
        ),
        (file, _) => Ok(file.map(|f| f.as_str().into())),
    }
}

/// The matrix, feature and barcode files directly in `dir`, when it holds
/// all three; `None` too for what is not a directory.
fn mtx_set_in(dir: &std::path::Path, args: &MergeMtxArgs) -> anyhow::Result<Option<[Box<str>; 3]>> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(None);
    };
    let mut files: Vec<String> = Vec::new();
    for entry in entries {
        if let Some(path) = entry?.path().to_str() {
            files.push(path.to_string());
        }
    }
    files.sort();
    let patterns = [
        &args.mtx_file_name,
        &args.feature_file_name,
        &args.barcode_file_name,
    ];
    // Not a set unless every kind of file is there: only then does a name
    // that fits several files matter.
    if !patterns
        .iter()
        .all(|p| files.iter().any(|f| f.ends_with(p.as_ref())))
    {
        return Ok(None);
    }
    let [Some(mtx), Some(row), Some(col)] = [
        file_named(&files, patterns[0])?,
        file_named(&files, patterns[1])?,
        file_named(&files, patterns[2])?,
    ] else {
        return Ok(None);
    };
    Ok(Some([mtx, row, col]))
}

/// `names` with each one that repeats numbered (`a_0`, `a_1`); the others
/// as they are.
fn number_repeated_names(names: &[Box<str>]) -> Vec<Box<str>> {
    let mut counts: HashMap<&str, usize> = Default::default();
    for name in names {
        *counts.entry(name.as_ref()).or_insert(0) += 1;
    }
    let mut numbered: HashMap<&str, usize> = Default::default();
    names
        .iter()
        .map(|name| {
            if counts[name.as_ref()] == 1 {
                return name.clone();
            }
            let k = numbered.entry(name.as_ref()).or_insert(0);
            *k += 1;
            format!("{}_{}", name, *k - 1).into_boxed_str()
        })
        .collect()
}

/// A batch name for each directory holding a file set: the directory's own
/// name, dots and all. Directories of one name (10x's
/// `sampleA/filtered_feature_bc_matrix`, `sampleB/filtered_feature_bc_matrix`)
/// take their parent's in front, and a number after when that is not enough,
/// so no two batches share a name.
fn mtx_batch_names(dirs: &[std::path::PathBuf]) -> anyhow::Result<Vec<String>> {
    let name_of = |p: &std::path::Path| -> Option<String> {
        Some(p.file_name()?.to_string_lossy().into_owned())
    };
    let own: Vec<String> = dirs
        .iter()
        .map(|d| name_of(d).ok_or_else(|| anyhow::anyhow!("{} has no name", d.display())))
        .collect::<anyhow::Result<_>>()?;
    let mut times: HashMap<&str, usize> = Default::default();
    for name in &own {
        *times.entry(name.as_str()).or_insert(0) += 1;
    }
    let named: Vec<Box<str>> = dirs
        .iter()
        .zip(&own)
        .map(|(dir, name)| match dir.parent().and_then(name_of) {
            Some(parent) if times[name.as_str()] > 1 => format!("{parent}_{name}").into(),
            _ => name.as_str().into(),
        })
        .collect();
    let names = number_repeated_names(&named);
    let distinct: rustc_hash::FxHashSet<&str> = names.iter().map(|n| n.as_ref()).collect();
    anyhow::ensure!(
        distinct.len() == names.len(),
        "the directories do not give each batch its own name: {names:?}"
    );
    Ok(names.into_iter().map(String::from).collect())
}

/// Find each directory's (and subdirectory's) matrix, feature and barcode
/// files.
fn find_mtx_sets(args: &MergeMtxArgs) -> anyhow::Result<MtxSets> {
    let mut mtx_files = vec![];
    let mut row_files = vec![];
    let mut col_files = vec![];
    let mut set_dirs = vec![];

    for dir in args.data_directories.iter() {
        let dir = std::path::Path::new(dir.as_ref());

        // The directory itself, when it has a name to give its batch, then
        // its subdirectories by name.
        let mut candidates = vec![];
        if dir.file_name().is_some() {
            candidates.push(dir.to_path_buf());
        }
        info!("Searching subdir within: {}", dir.display());
        let mut subs = std::fs::read_dir(dir)?
            .filter_map(Result::ok)
            .collect::<Vec<_>>();
        subs.sort_by_key(|entry| entry.file_name());
        candidates.extend(subs.into_iter().map(|sub| sub.path()));

        for candidate in candidates {
            if candidate.to_str().is_none() || candidate.file_name().is_none() {
                continue;
            }
            if let Some([m, r, c]) = mtx_set_in(&candidate, args)? {
                info!("Build a batch from {}, {}, {} ", m, r, c);
                mtx_files.push(m);
                row_files.push(r);
                col_files.push(c);
                set_dirs.push(candidate);
            }
        }
    }

    Ok(MtxSets {
        batch_names: mtx_batch_names(&set_dirs)?,
        mtx_files,
        row_files,
        col_files,
    })
}

/// `merge-mtx` over the file sets found, squeezed after when `squeeze`.
fn merge_mtx(args: &MergeMtxArgs, sets: MtxSets, squeeze: bool) -> anyhow::Result<()> {
    let MtxSets {
        batch_names,
        mtx_files,
        row_files,
        col_files,
    } = sets;
    let num_batches = batch_names.len();

    info!("merging over {} batches ...", num_batches);

    debug_assert_eq!(num_batches, mtx_files.len());
    debug_assert_eq!(num_batches, row_files.len());
    debug_assert_eq!(num_batches, col_files.len());

    if num_batches == 0 {
        return Err(anyhow::anyhow!("No relevant files found"));
    }

    info!("Finding common rows/features ...");

    let batch_rows = row_files
        .iter()
        .map(|f| read_row_names(f.clone(), args.num_feature_name_words))
        .collect::<anyhow::Result<Vec<_>>>()?;

    // The names every set has, sorted.
    let mut common_rows = common_names(&batch_rows);
    common_rows.sort();
    anyhow::ensure!(
        !common_rows.is_empty(),
        "the sets share no row/feature name"
    );

    // Each set's rows among the common ones; a name a set repeats keeps its
    // first row only.
    let row_maps: Vec<Vec<Option<u64>>> = {
        let place: HashMap<&str, u64> = common_rows
            .iter()
            .enumerate()
            .map(|(i, name)| (name.as_ref(), i as u64))
            .collect();
        batch_rows
            .iter()
            .map(|rows| common_row_map(rows, &place, common_rows.len()))
            .collect()
    };
    drop(batch_rows);

    info!(
        "Found {} common row/feature names across {} file sets",
        common_rows.len(),
        num_batches
    );

    info!("Elongating column/barcode names ...");

    let mut column_names = vec![];
    let mut column_batch_names = vec![];
    let mut batch_ncols: Vec<usize> = Vec::with_capacity(num_batches);

    for (col_file, batch_name) in col_files.iter().zip(batch_names.iter()) {
        let _names = read_col_names(col_file.clone(), args.num_barcode_name_words)?;
        let nn = _names.len();
        batch_ncols.push(nn);
        column_names.extend(
            _names
                .into_iter()
                .map(|x| format!("{}{}{}", x, COLUMN_SEP, batch_name).into_boxed_str())
                .collect::<Vec<_>>(),
        );

        column_batch_names.extend(vec![batch_name.clone().into_boxed_str(); nn]);
    }

    info!("Found {} columns/barcodes ...", column_names.len());

    info!("Sizing per-batch nnz ...");
    let mut batch_nnz: Vec<usize> = Vec::with_capacity(num_batches);
    let pb1 = styled_progress_bar(num_batches as u64, "batches");
    for b in 0..num_batches {
        let row_map = &row_maps[b];
        let (nrow, ncol, header_nnz) = mtx_io::read_mtx_header(&mtx_files[b])?;
        // The matrix indexes the lines of its feature and barcode files.
        anyhow::ensure!(
            nrow == row_map.len(),
            "{} has {nrow} rows, but {} names {}",
            mtx_files[b],
            row_files[b],
            row_map.len()
        );
        anyhow::ensure!(
            ncol == batch_ncols[b],
            "{} has {ncol} columns, but {} names {}",
            mtx_files[b],
            col_files[b],
            batch_ncols[b]
        );

        // Skip the parse if every row in this batch is kept — header nnz is exact.
        let no_filter = row_map.iter().all(Option::is_some);
        let count = if no_filter {
            header_nnz
        } else {
            let (triplets, _shape) = mtx_io::read_mtx_triplets(&mtx_files[b])?;
            triplets
                .iter()
                .filter(|(batch_i, _, _)| row_map[*batch_i as usize].is_some())
                .count()
        };
        batch_nnz.push(count);
        pb1.inc(1);
    }
    pb1.finish_and_clear();

    let total_nnz: usize = batch_nnz.iter().sum();
    let total_ncol: usize = batch_ncols.iter().sum();
    let total_nrow = common_rows.len();

    info!(
        "Total filtered nnz = {} across {} batches",
        total_nnz, num_batches
    );

    let backend = args.backend.clone();
    let effective_output = apply_zip_flag(&args.output, args.zip, &backend);

    // The backend the output's name resolves to (an `.h5` name, or a zarr
    // fallback for hdf5 without the feature), not the flag's.
    let (backend, backend_file) = resolve_backend_file(&effective_output, Some(backend))?;
    let backend_file: String = backend_file.to_string();

    if std::path::Path::new(&backend_file).exists() {
        info!(
            "This existing backend file '{}' will be deleted",
            backend_file
        );
        remove_file(&backend_file)?;
    }

    let mut out = create_sparse_streaming_empty(Some(&backend_file), Some(&backend))?;
    out.begin_streaming_csc((total_nrow, total_ncol, total_nnz))?;

    let mut col_offset: u64 = 0;
    let mut nnz_offset: u64 = 0;
    let pb2 = styled_progress_bar(num_batches as u64, "batches");
    for b in 0..num_batches {
        let row_map = &row_maps[b];
        let (triplets, _shape) = mtx_io::read_mtx_triplets(&mtx_files[b])?;

        let triplets_batch: Vec<(u64, u64, f32)> = triplets
            .into_iter()
            .filter_map(|(batch_i, batch_j, x_ij)| {
                Some((row_map[batch_i as usize]?, batch_j, x_ij))
            })
            .collect();

        debug_assert_eq!(triplets_batch.len(), batch_nnz[b]);

        let ncol_b = batch_ncols[b];
        let (local_colptr, row_indices, values) = build_csc_slab(triplets_batch, ncol_b);
        out.append_csc_slab(col_offset, nnz_offset, &local_colptr, &row_indices, &values)?;

        col_offset += ncol_b as u64;
        nnz_offset += batch_nnz[b] as u64;
        pb2.inc(1);
    }
    pb2.finish_and_clear();

    out.finalize_streaming_csc()?;
    info!("transposing CSC → CSR on disk");
    out.build_csr_from_csc_streaming()?;

    out.register_row_names_vec(&common_rows);
    out.register_column_names_vec(&column_names);
    drop(out);

    info!(
        "Successfully created a sparse backend file: {}",
        backend_file
    );

    let batch_map = column_names
        .into_iter()
        .zip(column_batch_names)
        .collect::<HashMap<_, _>>();

    if squeeze {
        squeeze_in_place(
            &backend_file,
            args.row_nnz_cutoff,
            args.column_nnz_cutoff,
            args.block_size,
        )?;
    }

    write_batches_and_finalize(&effective_output, &backend_file, &backend, &batch_map)?;
    info!("done");
    Ok(())
}

/// Generate unique batch names from file paths
/// If basenames are unique, use them as-is
/// If duplicates exist, add numeric suffixes
pub fn generate_unique_batch_names(files: &[Box<str>]) -> anyhow::Result<Vec<Box<str>>> {
    // Extract basenames and strip backend extensions (.zarr.zip, .zarr, .h5)
    let basenames: Vec<Box<str>> = files
        .iter()
        .map(|f| {
            basename(f).map(|b| {
                let stripped = strip_backend_suffix(&b);
                if stripped.len() == b.len() {
                    b // No suffix stripped, reuse original
                } else {
                    stripped.into() // Suffix stripped, allocate once
                }
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    Ok(number_repeated_names(&basenames))
}

pub fn align_backends(args: &AlignDataArgs) -> anyhow::Result<()> {
    plan::align_data(args)
}

/// `align-data` over `num_types` data types.
fn align_data(args: &AlignDataArgs, num_types: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        num_types >= 1,
        "the data types (-r) are a whole number, at least 1"
    );
    let n_data = args.data_files.len();
    let n_data_columns = n_data.div_ceil(num_types);
    let n_expected = n_data_columns * num_types;

    if n_expected != n_data {
        return Err(anyhow::anyhow!(format!(
            "Should be multiple of {}: actual data files {} < expected {}",
            num_types, n_data, n_expected
        )));
    }

    let n_data_rows = n_expected.div_ceil(n_data_columns);

    let mut full_data_vec = args
        .data_files
        .iter()
        .enumerate()
        .map(|(i, a_file)| -> anyhow::Result<_> {
            // Use a writable extension (`.zarr`/`.h5`) regardless of whether
            // the input is `.zarr.zip`, since `subset_columns_rows` below
            // mutates the backend and read-only zip stores cannot absorb writes.
            let dst_ext = if a_file.ends_with(".zarr.zip") || a_file.ends_with(".zarr") {
                "zarr"
            } else {
                "h5"
            };
            let base = basename(a_file)?;

            let data_col_id = i % n_data_columns + 1;
            let data_row_id = i / n_data_columns + 1;
            let dst_path = format!(
                "{}/{}_{}_{}.{}",
                args.output_directory, data_row_id, data_col_id, base, dst_ext
            );
            info!("staging aligned copy: {}", dst_path);
            materialize_writable_backend(a_file, &dst_path)?;
            let (backend, a_copied_file) = resolve_backend_file(&dst_path, None)?;
            open_sparse_matrix(&a_copied_file, &backend)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // identify common rows
    let mut shared_rows = vec![];
    for r in 0..n_data_rows {
        let data_set_idx: Vec<usize> = (r * n_data_columns..(r + 1) * n_data_columns).collect();

        let mut shared = common_names(
            &data_set_idx
                .iter()
                .map(|&di| full_data_vec[di].row_names())
                .collect::<anyhow::Result<Vec<_>>>()?,
        );

        info!(
            "found {} shared rows/features on the data type {}",
            shared.len(),
            r
        );

        if shared.is_empty() {
            return Err(anyhow::anyhow!("no features are shared"));
        }

        shared.par_sort();
        shared_rows.push(shared);
    }

    for data_col in 0..n_data_columns {
        info!("aligning on the data column {}", data_col);

        // 0. subset of data sets
        let data_set_idx: Vec<usize> = (data_col..n_data).step_by(n_data_columns).collect();

        let subset_data_vec = data_set_idx
            .iter()
            .map(|&di| &full_data_vec[di])
            .collect::<Vec<_>>();

        // 1. figure out shared names for each column
        let mut shared = common_names(
            &subset_data_vec
                .iter()
                .map(|d| d.column_names())
                .collect::<anyhow::Result<Vec<_>>>()?,
        );

        info!("found {} shared columns", shared.len());

        if shared.is_empty() {
            for &di in data_set_idx.iter() {
                let data = &mut full_data_vec[di];
                info!(
                    "let's remove this unaligned backend: {}",
                    data.get_backend_file_name()
                );
                data.remove_backend_file()?;
            }
            continue;
        }

        shared.par_sort();

        // 2. subset columns
        for (r, &di) in data_set_idx.iter().enumerate() {
            let data = &mut full_data_vec[di];

            let pos: HashMap<Box<str>, usize> = data
                .column_names()?
                .into_iter()
                .enumerate()
                .map(|(i, x)| (x, i))
                .collect();
            let columns: Vec<usize> = shared.iter().map(|x| pos[x]).collect();

            let pos: HashMap<Box<str>, usize> = data
                .row_names()?
                .into_iter()
                .enumerate()
                .map(|(i, x)| (x, i))
                .collect();
            let rows: Vec<usize> = shared_rows[r].iter().map(|x| pos[x]).collect();
            data.subset_columns_rows(Some(&columns), Some(&rows))?;
        }

        info!(
            "done on the data column {}/{}",
            data_col + 1,
            n_data_columns
        );
    }

    Ok(())
}

#[cfg(all(test, feature = "ndarray"))]
#[path = "merging_tests.rs"]
mod tests;
