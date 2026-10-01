use super::run_squeeze_if_needed;
use crate::hdf5_io::*;
use crate::sparse_io::*;
use crate::utilities::name_matching::{
    compose_id_name, filter_row_indices_by_type, make_names_unique,
};
use crate::zarr_io::*;

use clap::Args;
use legume_numeric::matrix::common_io::*;
use log::info;

#[derive(Args, Debug)]
pub struct From10xMoleculeArgs {
    #[arg(
        help = "Input 10X molecule_info.h5 file",
        long_help = "Specify the molecule_info.h5 file from Cell Ranger count/multi.\n\
                     Contains per-molecule data: barcode_idx, feature_idx, count, gem_group,\n\
                     etc."
    )]
    pub h5_file: Box<str>,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Backend format for output",
        long_help = "Choose the backend format for the output file."
    )]
    pub backend: SparseIoBackend,

    #[arg(
        short,
        long,
        help = "Output file header or name",
        long_help = "Specify the output file header.\n\
                     The zarr backend produces {output}.zarr.zip by default;\n\
                     pass --no-zip to keep a {output}.zarr directory instead."
    )]
    pub output: Box<str>,

    /// keep a `.zarr` directory instead of producing a `.zarr.zip` archive
    #[arg(long = "no-zip", default_value_t = true, action = clap::ArgAction::SetFalse)]
    pub zip: bool,

    #[arg(
        long,
        default_value = "Gene Expression",
        help = "Library type to include",
        long_help = "Filter molecules to only those from libraries of this type. Common types:\n\
                     'Gene Expression', 'Antibody Capture', 'CRISPR Guide Capture'.\n\
                     Reads library_info JSON to determine which library indices match."
    )]
    pub library_type: Box<str>,

    #[arg(
        long,
        default_value = "",
        help = "Select row type (feature_type)",
        long_help = "Filter features by type.\n\
                     Rows are included if their type contains this value.\n\
                     Empty (default) keeps all features. 10X uses 'Gene Expression',\n\
                     'Antibody Capture', etc."
    )]
    pub select_row_type: Box<str>,

    #[arg(
        long,
        default_value = "",
        help = "Remove row type",
        long_help = "Remove rows if their type contains this value.\n\
                     Empty (default) removes nothing."
    )]
    pub remove_row_type: Box<str>,

    #[arg(
        long,
        default_value_t = false,
        help = "Skip pass_filter and include all barcodes",
        long_help = "By default,\n\
                     only barcodes that passed Cell Ranger cell calling are included.\n\
                     Set this flag to include ALL barcodes with at least one molecule."
    )]
    pub no_pass_filter: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Sum read counts instead of counting molecules (UMIs)",
        long_help = "By default each entry is the number of molecules (UMIs) of a\n\
                     feature in a barcode, as in Cell Ranger's feature-barcode\n\
                     matrices. With this flag it is the sum of their read counts."
    )]
    pub sum_reads: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Squeeze sparse rows or columns",
        long_help = "Enable squeezing to remove rows and columns with too few non-zeros."
    )]
    pub do_squeeze: bool,

    #[arg(
        long,
        default_value_t = 1,
        help = "Row non-zero cutoff",
        long_help = "Minimum number of non-zero elements required for rows."
    )]
    pub row_nnz_cutoff: usize,

    #[arg(
        long,
        default_value_t = 1,
        help = "Column non-zero cutoff",
        long_help = "Minimum number of non-zero elements required for columns."
    )]
    pub column_nnz_cutoff: usize,

    #[arg(
        long,
        help = "Cells per rayon job for the post-build squeeze pass",
        long_help = "Cells per rayon job for the post-build squeeze pass.\n\
                     Omit it for auto-scaling by feature count."
    )]
    pub block_size: Option<usize>,
}
pub fn run_build_from_10x_molecule(args: &From10xMoleculeArgs) -> anyhow::Result<()> {
    let file = hdf5::File::open(args.h5_file.to_string())?;
    info!("Opened molecule_info.h5: {}", args.h5_file);

    let effective_output = apply_zip_flag(&args.output, args.zip, &args.backend);
    let (backend, backend_file) =
        resolve_backend_file(&effective_output, Some(args.backend.clone()))?;

    if std::path::Path::new(backend_file.as_ref()).exists() {
        info!("Removing existing backend file: {}", &backend_file);
        remove_file(&backend_file)?;
    }

    // 1. Read per-molecule arrays. Keep ndarray `Array1` owners — don't
    // `.to_vec()` them, as that doubles peak memory on large molecule files.
    let barcode_idx = file.dataset("barcode_idx")?.read_1d::<u64>()?;
    let feature_idx = file.dataset("feature_idx")?.read_1d::<u32>()?;
    let count = file.dataset("count")?.read_1d::<u32>()?;
    let gem_group = file.dataset("gem_group")?.read_1d::<u16>()?;
    let library_idx = file.dataset("library_idx")?.read_1d::<u16>()?;
    // Cell Ranger 7+: whether a molecule counts towards the feature-barcode
    // matrix (1) or not (0). Older files have no such field: all count.
    let umi_type = file
        .dataset("umi_type")
        .ok()
        .map(|d| d.read_1d::<u32>())
        .transpose()?;
    let n_molecules = barcode_idx.len();
    info!("Read {} molecules", n_molecules);

    // 2. Read lookup tables
    let barcodes = read_hdf5_strings(file.dataset("barcodes")?)?;
    let feature_group = file.group("features")?;
    let mut row_ids: Vec<Box<str>> = read_hdf5_strings(feature_group.dataset("id")?)?;
    let mut row_names: Vec<Box<str>> = read_hdf5_strings(feature_group.dataset("name")?)?;
    let mut row_types: Vec<Box<str>> = read_hdf5_strings(feature_group.dataset("feature_type")?)?;

    let n_features = row_ids.len();
    info!("Read {} barcodes, {} features", barcodes.len(), n_features);

    // 3. Parse library_info and filter by library type: each library kept,
    //    with its gem group when the file gives one.
    let valid_libraries: rustc_hash::FxHashMap<u16, Option<u16>> = {
        let lib_info_ds = file.dataset("library_info")?;
        let lib_info_raw = read_hdf5_strings(lib_info_ds)?;
        let lib_info_json: String = lib_info_raw.iter().map(|s| s.as_ref()).collect();
        let lib_entries: Vec<serde_json::Value> = serde_json::from_str(&lib_info_json)?;

        let mut valid = rustc_hash::FxHashMap::default();
        let mut types_seen = Vec::new();
        for entry in &lib_entries {
            // Cell Ranger writes `library_id` as a number or, in older files,
            // as a numeric string ("0").
            let lib_id = entry.get("library_id").and_then(|v| {
                v.as_u64()
                    .or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
            });
            let lib_type = entry.get("library_type").and_then(|v| v.as_str());
            let gem_group = entry.get("gem_group").and_then(|v| v.as_u64());
            if let Some(t) = lib_type {
                types_seen.push(t.to_string());
            }
            if let (Some(lib_id), Some(lib_type)) = (lib_id, lib_type) {
                if lib_type.contains(args.library_type.as_ref()) {
                    valid.insert(lib_id as u16, gem_group.map(|g| g as u16));
                }
            }
        }
        info!(
            "Library type '{}': {} of {} libraries match",
            args.library_type,
            valid.len(),
            lib_entries.len()
        );
        anyhow::ensure!(
            !valid.is_empty(),
            "no library of type '{}' in {} (library types: {})",
            args.library_type,
            args.h5_file,
            types_seen.join(", ")
        );
        valid
    };

    // 4. Read pass_filter if needed
    //    Column key = (barcode_idx, gem_group)
    use rustc_hash::FxHashMap as HashMap;
    use std::collections::BTreeSet;

    let mut col_keys = BTreeSet::new();
    let valid_cells: Option<rustc_hash::FxHashSet<(u64, u16)>> = if !args.no_pass_filter {
        // Rows of [barcode_idx, library_idx, genome_idx].
        let pf = file.dataset("barcode_info/pass_filter")?.read_2d::<u64>()?;
        let mut cells = rustc_hash::FxHashSet::default();
        for row in pf.rows() {
            let bc_idx = row[0];
            let lib_idx = row[1] as u16;
            if let Some(gem_group) = valid_libraries.get(&lib_idx) {
                cells.insert((bc_idx, lib_idx));
                // Every cell is a column, as in Cell Ranger's filtered
                // matrix, even one without a molecule of this library type.
                if let Some(gem_group) = gem_group {
                    col_keys.insert((bc_idx, *gem_group));
                }
            }
        }
        info!("pass_filter: {} valid cells", cells.len());
        Some(cells)
    } else {
        info!("Skipping pass_filter (--no-pass-filter)");
        None
    };

    // 5. Filter molecules and aggregate into triplets
    let mut triplet_map: HashMap<(u64, u64), f32> = Default::default();

    {
        // Borrow raw arrays as slices once; avoids repeated bounds-checked
        // indexing through `Array1::Index` in the hot loop.
        let barcode_idx_s = barcode_idx.as_slice().expect("barcode_idx not contiguous");
        let feature_idx_s = feature_idx.as_slice().expect("feature_idx not contiguous");
        let count_s = count.as_slice().expect("count not contiguous");
        let gem_group_s = gem_group.as_slice().expect("gem_group not contiguous");
        let library_idx_s = library_idx.as_slice().expect("library_idx not contiguous");

        for i in 0..n_molecules {
            // Filter by library type
            if !valid_libraries.contains_key(&library_idx_s[i]) {
                continue;
            }

            // Molecules Cell Ranger leaves out of its matrices
            if umi_type.as_ref().is_some_and(|t| t[i] != 1) {
                continue;
            }

            // Filter by pass_filter
            if let Some(ref cells) = valid_cells {
                if !cells.contains(&(barcode_idx_s[i], library_idx_s[i])) {
                    continue;
                }
            }

            let col_key = (barcode_idx_s[i], gem_group_s[i]);
            col_keys.insert(col_key);

            // Will remap column index after collecting all keys
            let row = feature_idx_s[i] as u64;
            *triplet_map
                .entry((row, barcode_idx_s[i] * 65536 + gem_group_s[i] as u64))
                .or_insert(0.0) += if args.sum_reads {
                count_s[i] as f32
            } else {
                1.0
            };
        }
    }

    // Free the raw per-molecule arrays before we materialize the (huge)
    // triplets Vec. These can easily be multiple GB on 10X Aggr outputs.
    drop(barcode_idx);
    drop(feature_idx);
    drop(count);
    drop(gem_group);
    drop(library_idx);
    drop(umi_type);
    drop(valid_cells);

    // Build dense column index mapping
    let col_keys_vec: Vec<(u64, u16)> = col_keys.into_iter().collect();
    let col_key_to_idx: HashMap<(u64, u16), u64> = col_keys_vec
        .iter()
        .enumerate()
        .map(|(idx, &key)| (key, idx as u64))
        .collect();
    let ncols = col_keys_vec.len();

    // Build column names: SEQUENCE-GEMGROUP
    let column_names: Vec<Box<str>> = col_keys_vec
        .iter()
        .map(|&(bc_idx, gg)| {
            let bc = barcodes[bc_idx as usize].as_ref();
            format!("{}-{}", bc, gg).into_boxed_str()
        })
        .collect();

    info!("Aggregated into {} columns (cells)", ncols);

    // Convert triplet_map to proper triplets with remapped column indices
    let triplets: Vec<(u64, u64, f32)> = triplet_map
        .into_iter()
        .map(|((row, packed_col), val)| {
            let bc_idx = packed_col / 65536;
            let gg = (packed_col % 65536) as u16;
            let col = col_key_to_idx[&(bc_idx, gg)];
            (row, col, val)
        })
        .collect();

    let nrows = n_features;
    let nnz = triplets.len();
    info!("Built {} triplets in {} x {} matrix", nnz, nrows, ncols);

    // 6. Build backend
    let mut out = create_sparse_from_triplets_owned(
        triplets,
        (nrows, ncols, nnz),
        Some(&backend_file),
        Some(&backend),
    )?;
    info!("Created sparse matrix: {}", backend_file);

    // 7. Register names
    // Composite row names: id_name
    if nrows < row_ids.len() {
        row_ids.truncate(nrows);
    }
    if nrows < row_names.len() {
        row_names.truncate(nrows);
    }

    let mut row_id_names = compose_id_name(row_ids, row_names);
    make_names_unique(&mut row_id_names);

    out.register_row_names_vec(&row_id_names);
    out.register_column_names_vec(&column_names);

    // 8. Filter by feature type
    if nrows < row_types.len() {
        row_types.truncate(nrows);
    }

    let select_rows =
        filter_row_indices_by_type(&row_types, &args.select_row_type, &args.remove_row_type);

    if select_rows.len() < nrows {
        info!(
            "Filtering features: {} -> {} of '{}' type",
            nrows,
            select_rows.len(),
            args.select_row_type
        );
        out.subset_columns_rows(None, Some(&select_rows))?;
    }

    // 9. Squeeze if needed
    run_squeeze_if_needed(
        args.do_squeeze,
        args.row_nnz_cutoff,
        args.column_nnz_cutoff,
        args.block_size,
        &backend_file,
    )?;
    finalize_zarr_output(&backend_file, &effective_output)?;
    info!("done");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hdf5::types::VarLenUnicode;

    fn strings(g: &hdf5::Group, name: &str, xs: &[&str]) {
        let v: Vec<VarLenUnicode> = xs.iter().map(|s| s.parse().unwrap()).collect();
        g.new_dataset_builder().with_data(&v).create(name).unwrap();
    }

    /// A `molecule_info.h5` as Cell Ranger writes it: three features (the
    /// last without a molecule), a library of each type, one of them with a
    /// string id, molecules left out by `umi_type`, library type and cell
    /// calling, and a called cell without a molecule.
    fn molecule_info(path: &std::path::Path) {
        let f = hdf5::File::create(path).unwrap();
        let features = f.create_group("features").unwrap();
        strings(&features, "id", &["FID1", "FID2", "FID3"]);
        strings(&features, "name", &["GENE1", "GENE2", "GENE3"]);
        strings(&features, "feature_type", &["Gene Expression"; 3]);
        strings(&f, "barcodes", &["AAAC", "AAAG", "AAAT", "AACA"]);
        strings(
            &f,
            "library_info",
            &[
                r#"[{"gem_group": 1, "library_id": "0", "library_type": "Gene Expression"},
                  {"gem_group": 1, "library_id": 1, "library_type": "Antibody Capture"}]"#,
            ],
        );
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
        };
        run_build_from_10x_molecule(&args).unwrap();
        let data = open_sparse_matrix(&format!("{}.zarr.zip", args.output), &SparseIoBackend::Zarr)
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
        molecule_info(&h5);

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
}
