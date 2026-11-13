use super::{Output, Staged};
use crate::convert::{write_10x_matrix, TenxMatrix};
use crate::hdf5_io::*;

use log::info;
use rustc_hash::{FxHashMap, FxHashSet};

/// A Cell Ranger `molecule_info.h5`, and what [`from_10x_molecule`] counts
/// of it.
#[derive(Clone, Debug)]
pub struct TenxMoleculeImport {
    pub h5_file: Box<str>,
    /// Count the molecules of libraries whose `library_type` contains this:
    /// `Gene Expression`, `Antibody Capture`, `CRISPR Guide Capture`, ...
    pub library_type: Box<str>,
    /// Keep features whose type contains this; empty keeps all.
    pub select_row_type: Box<str>,
    /// Drop features whose type contains this; empty drops none.
    pub remove_row_type: Box<str>,
    /// Only the barcodes Cell Ranger called cells (`pass_filter`), every
    /// one of them, as in its filtered matrix; else every barcode with a
    /// molecule.
    pub pass_filter: bool,
    /// Sum the molecules' read counts instead of counting molecules (UMIs).
    pub sum_reads: bool,
}

impl TenxMoleculeImport {
    /// `h5_file` counted as Cell Ranger's filtered matrix: UMIs of the
    /// `Gene Expression` libraries in the called cells.
    pub fn new(h5_file: &str) -> Self {
        Self {
            h5_file: h5_file.into(),
            library_type: "Gene Expression".into(),
            select_row_type: "".into(),
            remove_row_type: "".into(),
            pass_filter: true,
            sum_reads: false,
        }
    }
}

/// A library `library_type` keeps.
struct Library {
    gem_group: Option<u16>,
    /// The probe set a probe-based (Flex) library targets.
    target_set: Option<Box<str>>,
}

/// Build the backend `out` names from the molecules in `h5`, a column per
/// (barcode, GEM group) named `SEQUENCE-GEMGROUP`; the path written.
pub fn from_10x_molecule(h5: &TenxMoleculeImport, out: &Output) -> anyhow::Result<Box<str>> {
    stage_10x_molecule(h5, out)?.finish()
}

/// [`from_10x_molecule`] short of squeezing and putting the backend in place.
pub fn stage_10x_molecule(args: &TenxMoleculeImport, out: &Output) -> anyhow::Result<Staged> {
    let file = hdf5::File::open(args.h5_file.to_string())?;
    info!("Opened molecule_info.h5: {}", args.h5_file);

    let (staged, backend) = out.stage()?;

    // 1. Read per-molecule arrays. Keep ndarray `Array1` owners — don't
    // `.to_vec()` them, as that doubles peak memory on large molecule files.
    let barcode_idx = file.dataset("barcode_idx")?.read_1d::<u64>()?;
    let feature_idx = file.dataset("feature_idx")?.read_1d::<u32>()?;
    let gem_group = file.dataset("gem_group")?.read_1d::<u16>()?;
    let library_idx = file.dataset("library_idx")?.read_1d::<u16>()?;
    // Reads per molecule, needed only to sum them.
    let count = if args.sum_reads {
        Some(file.dataset("count")?.read_1d::<u32>()?)
    } else {
        None
    };
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
    let row_ids: Vec<Box<str>> = read_hdf5_strings(feature_group.dataset("id")?)?;
    let row_names: Vec<Box<str>> = read_hdf5_strings(feature_group.dataset("name")?)?;
    let row_types: Vec<Box<str>> = read_hdf5_strings(feature_group.dataset("feature_type")?)?;

    let n_features = row_ids.len();
    info!("Read {} barcodes, {} features", barcodes.len(), n_features);

    // 3. Parse library_info and keep the libraries of the requested type
    let libraries: FxHashMap<u16, Library> = {
        let lib_info_raw = read_hdf5_strings(file.dataset("library_info")?)?;
        let lib_info_json: String = lib_info_raw.iter().map(|s| s.as_ref()).collect();
        let lib_entries: Vec<serde_json::Value> = serde_json::from_str(&lib_info_json)?;

        let mut kept = FxHashMap::default();
        for entry in &lib_entries {
            // Cell Ranger writes `library_id` as a number or, in older files,
            // as a numeric string ("0").
            let lib_id = entry.get("library_id").and_then(|v| {
                v.as_u64()
                    .or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
            });
            let lib_type = entry.get("library_type").and_then(|v| v.as_str());
            if let (Some(lib_id), Some(lib_type)) = (lib_id, lib_type) {
                if lib_type.contains(args.library_type.as_ref()) {
                    let library = Library {
                        gem_group: entry
                            .get("gem_group")
                            .and_then(|v| v.as_u64())
                            .map(|g| g as u16),
                        target_set: entry
                            .get("target_set_name")
                            .and_then(|v| v.as_str())
                            .map(Box::from),
                    };
                    kept.insert(lib_id as u16, library);
                }
            }
        }
        info!(
            "Library type '{}': {} of {} libraries match",
            args.library_type,
            kept.len(),
            lib_entries.len()
        );
        anyhow::ensure!(
            !kept.is_empty(),
            "no library of type '{}' in {} (library types: {})",
            args.library_type,
            args.h5_file,
            lib_entries
                .iter()
                .filter_map(|e| e.get("library_type")?.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        kept
    };

    // A probe-based library (Flex) counts the genes its probe set targets:
    // Cell Ranger's matrices have those rows only, leaving out molecules of
    // other genes, from probes it excludes. Libraries without a probe set
    // keep every feature.
    let targeted: Option<Vec<bool>> = libraries
        .values()
        .map(|l| l.target_set.as_deref())
        .collect::<Option<Vec<_>>>()
        .map(|names| -> anyhow::Result<Vec<bool>> {
            let mut keep = vec![false; n_features];
            for name in names {
                let rows: Vec<u32> = file
                    .dataset(&format!("features/target_sets/{name}"))
                    .map_err(|_| anyhow::anyhow!("no probe set '{name}' in {}", args.h5_file))?
                    .read_raw()?;
                for i in rows {
                    *keep.get_mut(i as usize).ok_or_else(|| {
                        anyhow::anyhow!("probe set '{name}' names feature {i} of {n_features}")
                    })? = true;
                }
            }
            Ok(keep)
        })
        .transpose()?;

    // 4. Columns, (barcode, gem group) → index. With cell calling, every
    //    called cell is one, as in Cell Ranger's filtered matrix, even one
    //    without a molecule of these libraries.
    let mut columns: FxHashMap<(u64, u16), u64> = FxHashMap::default();
    let called: Option<FxHashSet<(u64, u16)>> = if args.pass_filter {
        // Rows of [barcode_idx, library_idx, genome_idx].
        let pf = file.dataset("barcode_info/pass_filter")?.read_2d::<u64>()?;
        let mut cells = FxHashSet::default();
        for row in pf.rows() {
            let (bc, lib) = (row[0], row[1] as u16);
            if let Some(library) = libraries.get(&lib) {
                cells.insert((bc, lib));
                if let Some(gg) = library.gem_group {
                    let next = columns.len() as u64;
                    columns.entry((bc, gg)).or_insert(next);
                }
            }
        }
        info!("pass_filter: {} called cells", cells.len());
        Some(cells)
    } else {
        info!("Skipping pass_filter: every barcode with a molecule is a column");
        None
    };

    // 5. Filter molecules and aggregate them by (feature, column)
    let mut entries: FxHashMap<(u64, u64), f32> = FxHashMap::default();
    {
        // Borrow raw arrays as slices once; avoids repeated bounds-checked
        // indexing through `Array1::Index` in the hot loop.
        fn slice<T>(a: &ndarray::Array1<T>) -> &[T] {
            a.as_slice().expect("molecule array not contiguous")
        }
        let barcode_idx_s: &[u64] = slice(&barcode_idx);
        let feature_idx_s: &[u32] = slice(&feature_idx);
        let gem_group_s: &[u16] = slice(&gem_group);
        let library_idx_s: &[u16] = slice(&library_idx);
        let count_s: Option<&[u32]> = count.as_ref().map(slice);
        let umi_type_s: Option<&[u32]> = umi_type.as_ref().map(slice);

        for i in 0..n_molecules {
            let (bc, lib) = (barcode_idx_s[i], library_idx_s[i]);
            if !libraries.contains_key(&lib)
                // Molecules Cell Ranger leaves out of its matrices
                || umi_type_s.is_some_and(|t| t[i] != 1)
                || called.as_ref().is_some_and(|c| !c.contains(&(bc, lib)))
            {
                continue;
            }
            let next = columns.len() as u64;
            let col = *columns.entry((bc, gem_group_s[i])).or_insert(next);
            *entries.entry((feature_idx_s[i] as u64, col)).or_insert(0.0) +=
                count_s.map_or(1.0, |c| c[i] as f32);
        }
    }

    // Free the raw per-molecule arrays before we materialize the (huge)
    // triplets Vec. These can easily be multiple GB on 10X Aggr outputs.
    drop((
        barcode_idx,
        feature_idx,
        gem_group,
        library_idx,
        count,
        umi_type,
    ));
    drop(called);

    // Columns in barcode order, named SEQUENCE-GEMGROUP.
    let mut columns: Vec<((u64, u16), u64)> = columns.into_iter().collect();
    columns.sort_unstable();
    let mut new_col = vec![0u64; columns.len()];
    for (new, &(_, old)) in columns.iter().enumerate() {
        new_col[old as usize] = new as u64;
    }
    let column_names: Vec<Box<str>> = columns
        .iter()
        .map(|&((bc, gg), _)| format!("{}-{gg}", barcodes[bc as usize]).into_boxed_str())
        .collect();
    info!("Aggregated into {} columns (cells)", column_names.len());

    let triplets: Vec<(u64, u64, f32)> = entries
        .into_iter()
        .map(|((row, col), val)| (row, new_col[col as usize], val))
        .collect();

    // 6. Name and keep rows, and write the backend
    write_10x_matrix(
        TenxMatrix {
            triplets,
            reach: (n_features, column_names.len()),
            row_ids: Some(row_ids),
            row_names: Some(row_names),
            row_types: Some(row_types),
            column_names: Some(column_names),
            keep_rows: targeted,
        },
        &args.select_row_type,
        &args.remove_row_type,
        staged.working(),
        &backend,
    )?;

    Ok(staged)
}
