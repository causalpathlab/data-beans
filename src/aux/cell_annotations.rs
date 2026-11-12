use rustc_hash::FxHashMap as HashMap;

use anyhow::{bail, Result};
use log::{info, warn};
use nalgebra::DMatrix;

use legume_numeric::matrix::common_io::{read_lines, unquote_field};
use legume_numeric::matrix::traits::IoOps;

/// Parsed cell annotations: maps each cell to an individual (and optionally a cell type).
#[derive(Debug, Clone)]
pub struct CellAnnotations {
    /// cell_id -> individual_idx
    pub cell_to_individual: HashMap<Box<str>, usize>,
    /// Ordered individual IDs (index corresponds to individual_idx)
    pub individual_ids: Vec<Box<str>>,
}

impl CellAnnotations {
    /// Convert to a flat vector aligned to the given column names.
    ///
    /// Each entry is the individual ID for the corresponding column,
    /// or `missing` for unmatched cells.
    pub fn to_column_aligned_vec(&self, column_names: &[Box<str>], missing: &str) -> Vec<Box<str>> {
        column_names
            .iter()
            .map(|name| {
                self.cell_to_individual
                    .get(name)
                    .map(|&idx| self.individual_ids[idx].clone())
                    .unwrap_or_else(|| Box::from(missing))
            })
            .collect()
    }
}

/// Cell-type membership matrix with cell type names.
///
/// Note: this is different from `legume_numeric::matrix::membership::Membership`
/// which is a key→group string map. This holds a dense probability matrix.
pub struct CellTypeMembership {
    /// Membership matrix: cells × cell_types (aligned to SC backend column order)
    pub matrix: DMatrix<f32>,
    /// Ordered cell type names (columns of the membership matrix)
    pub cell_type_names: Vec<Box<str>>,
}

/// The data rows of an annotation table (TSV, CSV, or whitespace-separated,
/// plain or gzipped), each split into its fields.
///
/// The header line names the delimiter for the whole file: tab when it holds
/// a tab, else comma when it holds a comma, else runs of whitespace. A tab-
/// or comma-separated field therefore keeps its spaces (`CD4 T cell`), and is
/// trimmed and unquoted, while a comma inside a label of a
/// whitespace-separated table (`B,naive`) is part of the label. Comment lines
/// (`#`, `%`) and blank lines are skipped; the first remaining line is the
/// header and is dropped.
fn read_annotation_rows(path: &str) -> Result<Vec<Vec<Box<str>>>> {
    let lines: Vec<Box<str>> = read_lines(path)?
        .into_iter()
        .filter(|line| !line.trim().is_empty() && !line.starts_with(['#', '%']))
        .collect();
    let delim = lines
        .first()
        .and_then(|header| ['\t', ','].into_iter().find(|&d| header.contains(d)));
    Ok(lines
        .iter()
        .skip(1)
        .map(|line| match delim {
            Some(d) => line.split(d).map(|x| unquote_field(x).into()).collect(),
            None => line.split_whitespace().map(Box::from).collect(),
        })
        .collect())
}

/// Each row's cell id and its label in field `label_idx`, in file order.
/// A row too short to hold the label is skipped; so is a row whose cell id or
/// label is empty. Both are counted in a warning, and a file that yields no
/// row at all is an error: its rows do not follow the header's delimiter, or
/// it has no such column.
fn read_cell_labels(path: &str, label_idx: usize) -> Result<Vec<(Box<str>, Box<str>)>> {
    let mut n_short = 0usize;
    let mut n_empty = 0usize;
    let mut out = Vec::new();
    for mut words in read_annotation_rows(path)? {
        if words.len() <= label_idx {
            n_short += 1;
            continue;
        }
        if words[0].is_empty() || words[label_idx].is_empty() {
            n_empty += 1;
            continue;
        }
        let label = words.swap_remove(label_idx);
        out.push((words.swap_remove(0), label));
    }
    let n_fields = label_idx + 1;
    if out.is_empty() {
        bail!(
            "{path}: no row holds a cell id and a label in field {n_fields} \
             ({n_short} rows with fewer than {n_fields} fields, {n_empty} rows \
             with an empty cell id or label); the header line sets the delimiter"
        );
    }
    if n_short > 0 || n_empty > 0 {
        warn!(
            "{path}: skipped {n_short} rows with fewer than {n_fields} fields \
             and {n_empty} rows with an empty cell id or label"
        );
    }
    Ok(out)
}

/// Read cell annotations from a delimited file (TSV, CSV, or space-separated).
///
/// Supports gzip-compressed files (.gz). The file is split on one delimiter
/// (tab, else comma, else whitespace; see [`read_annotation_rows`]), so a
/// field of a TSV or CSV may hold spaces.
/// First line is treated as a header and skipped.
///
/// Requires at least 2 columns: `cell_id individual_id`.
/// A 3rd column (`cell_type`) is accepted but ignored here;
/// use [`build_onehot_membership`] to convert it into a membership matrix.
pub fn read_cell_annotations(path: &str) -> Result<CellAnnotations> {
    info!("Reading cell annotations from {}", path);

    let mut individual_to_idx: HashMap<Box<str>, usize> = Default::default();
    let mut individual_ids: Vec<Box<str>> = Vec::new();
    let mut cell_to_individual: HashMap<Box<str>, usize> = Default::default();

    for (cell_id, ind_id) in read_cell_labels(path, 1)? {
        let ind_idx = *individual_to_idx.entry(ind_id.clone()).or_insert_with(|| {
            let idx = individual_ids.len();
            individual_ids.push(ind_id);
            idx
        });

        cell_to_individual.insert(cell_id, ind_idx);
    }

    info!(
        "Loaded {} cell annotations: {} individuals",
        cell_to_individual.len(),
        individual_ids.len(),
    );

    Ok(CellAnnotations {
        cell_to_individual,
        individual_ids,
    })
}

/// Infer cell annotations from cell names by splitting on `@`.
///
/// Cell names like `ACGT@IND_A` map to individual `IND_A`.
/// Names without `@` are assigned to a single individual `"all"`.
pub fn infer_cell_annotations(column_names: &[Box<str>]) -> CellAnnotations {
    info!("Inferring individuals from cell names (barcode@indiv)");

    let mut individual_to_idx: HashMap<Box<str>, usize> = Default::default();
    let mut individual_ids: Vec<Box<str>> = Vec::new();
    let mut cell_to_individual: HashMap<Box<str>, usize> = Default::default();

    for cell_name in column_names {
        let indiv: Box<str> = if let Some(pos) = cell_name.rfind('@') {
            Box::from(&cell_name[pos + 1..])
        } else {
            Box::from("all")
        };
        let idx = *individual_to_idx.entry(indiv.clone()).or_insert_with(|| {
            let i = individual_ids.len();
            individual_ids.push(indiv);
            i
        });
        cell_to_individual.insert(cell_name.clone(), idx);
    }

    info!(
        "Inferred {} individuals from cell names",
        individual_ids.len()
    );

    CellAnnotations {
        cell_to_individual,
        individual_ids,
    }
}

/// Build a one-hot membership matrix from hard cell-type annotations.
///
/// Reads a delimited file with 3+ columns: `cell_id individual_id cell_type`,
/// split as [`read_cell_annotations`] splits it.
/// The resulting matrix has one-hot rows aligned to the SC backend `column_names`.
pub fn build_onehot_membership(
    path: &str,
    column_names: &[Box<str>],
) -> Result<CellTypeMembership> {
    info!("Building one-hot membership from {}", path);

    let mut celltype_to_idx: HashMap<Box<str>, usize> = Default::default();
    let mut cell_type_names: Vec<Box<str>> = Vec::new();
    let mut cell_to_ct: HashMap<Box<str>, usize> = Default::default();

    for (cell_id, ct_name) in read_cell_labels(path, 2)? {
        let ct_idx = *celltype_to_idx.entry(ct_name.clone()).or_insert_with(|| {
            let idx = cell_type_names.len();
            cell_type_names.push(ct_name);
            idx
        });

        cell_to_ct.insert(cell_id, ct_idx);
    }

    let n_cells = column_names.len();
    let n_ct = cell_type_names.len();
    let mut matrix = DMatrix::<f32>::zeros(n_cells, n_ct);
    let mut matched = 0usize;

    for (cell_idx, cell_name) in column_names.iter().enumerate() {
        if let Some(&ct_idx) = cell_to_ct.get(cell_name) {
            matrix[(cell_idx, ct_idx)] = 1.0;
            matched += 1;
        }
    }

    info!(
        "One-hot membership: {}/{} cells matched, {} cell types",
        matched, n_cells, n_ct
    );

    if matched == 0 {
        bail!("No cells matched between SC backend and annotation file");
    }

    Ok(CellTypeMembership {
        matrix,
        cell_type_names,
    })
}

/// Read soft cell-type membership proportions from a parquet file.
///
/// The parquet should have cell IDs as row names and cell type names as
/// column headers.  Values are probabilities (rows sum to ~1).
/// Cells are matched to the SC backend by `column_names`.
pub fn read_membership_proportions(
    file_path: &str,
    column_names: &[Box<str>],
) -> Result<CellTypeMembership> {
    info!("Reading membership proportions from {}", file_path);

    let mat_with_names = DMatrix::<f32>::from_parquet(file_path)?;
    let cell_type_names = mat_with_names.cols;
    let n_cell_types = cell_type_names.len();
    let n_cells = column_names.len();

    // Build lookup: parquet row name -> parquet row index
    let parquet_cell_lookup: HashMap<&str, usize> = mat_with_names
        .rows
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_ref(), i))
        .collect();

    // Build membership matrix aligned to column_names order
    let mut matrix = DMatrix::<f32>::zeros(n_cells, n_cell_types);
    let mut matched = 0usize;

    for (cell_idx, cell_name) in column_names.iter().enumerate() {
        if let Some(&pq_row) = parquet_cell_lookup.get(cell_name.as_ref()) {
            for k in 0..n_cell_types {
                matrix[(cell_idx, k)] = mat_with_names.mat[(pq_row, k)];
            }
            matched += 1;
        }
    }

    info!(
        "Matched {}/{} cells to membership parquet ({} cell types)",
        matched, n_cells, n_cell_types
    );

    if matched == 0 {
        bail!("No cells matched between SC backend and membership parquet");
    }

    Ok(CellTypeMembership {
        matrix,
        cell_type_names,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_cell_annotations_roundtrip() -> Result<()> {
        use std::io::Write;

        let dir = tempfile::tempdir()?;
        let path = dir.path().join("cells.tsv");
        let path_str = path.to_str().unwrap();

        let mut f = std::fs::File::create(&path)?;
        writeln!(f, "cell_id\tindividual_id\tcell_type")?;
        writeln!(f, "cell_0\tIND_A\tT_cell")?;
        writeln!(f, "cell_1\tIND_A\tB_cell")?;
        writeln!(f, "cell_2\tIND_B\tT_cell")?;
        writeln!(f, "cell_3\tIND_B\tT_cell")?;
        f.flush()?;

        let anno = read_cell_annotations(path_str)?;
        assert_eq!(anno.individual_ids.len(), 2);
        assert_eq!(anno.cell_to_individual.len(), 4);

        let &ind_idx = anno.cell_to_individual.get(&Box::from("cell_0")).unwrap();
        assert_eq!(anno.individual_ids[ind_idx].as_ref(), "IND_A");

        Ok(())
    }

    fn write_table(dir: &tempfile::TempDir, name: &str, lines: &[&str]) -> String {
        let path = dir.path().join(name);
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();
        path.to_str().unwrap().to_string()
    }

    fn individual_of<'a>(anno: &'a CellAnnotations, cell: &str) -> &'a str {
        let &idx = anno.cell_to_individual.get(cell).unwrap();
        anno.individual_ids[idx].as_ref()
    }

    #[test]
    fn fields_of_a_tsv_or_csv_keep_their_spaces() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let tsv = write_table(
            &dir,
            "cells.tsv",
            &[
                "cell_id\tindividual_id\tcell_type",
                "cell_0\tdonor 1\tCD4 T cell",
                "cell_1\tdonor 2\tCD4 T cell",
                "cell_2\tdonor 2\tB cell, naive",
            ],
        );
        let csv = write_table(
            &dir,
            "cells.csv",
            &[
                "cell_id, individual_id, cell_type",
                "cell_0, donor 1, CD4 T cell",
                "cell_1, \"donor 2\", CD4 T cell",
                "cell_2, donor 2, B cell",
            ],
        );
        let columns: Vec<Box<str>> = vec!["cell_0".into(), "cell_1".into(), "cell_2".into()];
        for (path, b_cell) in [(&tsv, "B cell, naive"), (&csv, "B cell")] {
            let anno = read_cell_annotations(path)?;
            assert_eq!(
                anno.individual_ids,
                vec![Box::<str>::from("donor 1"), "donor 2".into()],
                "{path}"
            );
            assert_eq!(individual_of(&anno, "cell_0"), "donor 1");
            assert_eq!(individual_of(&anno, "cell_2"), "donor 2");

            let membership = build_onehot_membership(path, &columns)?;
            assert_eq!(
                membership.cell_type_names,
                vec![Box::<str>::from("CD4 T cell"), b_cell.into()],
                "{path}"
            );
            assert_eq!(membership.matrix[(1, 0)], 1.0);
            assert_eq!(membership.matrix[(2, 1)], 1.0);
        }
        Ok(())
    }

    #[test]
    fn whitespace_tables_and_rows_with_an_empty_field() -> Result<()> {
        let dir = tempfile::tempdir()?;
        // Aligned with runs of spaces: no field is empty.
        let spaced = write_table(
            &dir,
            "cells.txt",
            &[
                "cell  indiv  type",
                "cell_0   IND_A  T",
                "cell_1   IND_B  B",
            ],
        );
        let anno = read_cell_annotations(&spaced)?;
        assert_eq!(individual_of(&anno, "cell_0"), "IND_A");
        assert_eq!(individual_of(&anno, "cell_1"), "IND_B");

        // A row without a cell id or without an individual names no one.
        let gapped = write_table(
            &dir,
            "gapped.tsv",
            &[
                "cell\tindiv",
                "cell_0\tIND_A",
                "cell_1\t",
                "\tIND_B",
                "",
                "cell_2\tIND_A",
            ],
        );
        let anno = read_cell_annotations(&gapped)?;
        assert_eq!(anno.individual_ids, vec![Box::<str>::from("IND_A")]);
        assert_eq!(anno.cell_to_individual.len(), 2);
        Ok(())
    }

    #[test]
    fn the_header_line_sets_the_delimiter() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let columns: Vec<Box<str>> = vec!["cell_0".into(), "cell_1".into(), "cell_2".into()];

        // Whitespace-separated: a comma is part of the label it sits in.
        let spaced = write_table(
            &dir,
            "cells.txt",
            &[
                "cell indiv type",
                "cell_0 IND_A T",
                "cell_1 IND_B B,naive",
                "cell_2 IND_B T",
            ],
        );
        let anno = read_cell_annotations(&spaced)?;
        assert_eq!(anno.cell_to_individual.len(), 3);
        assert_eq!(individual_of(&anno, "cell_1"), "IND_B");
        let membership = build_onehot_membership(&spaced, &columns)?;
        assert_eq!(
            membership.cell_type_names,
            vec![Box::<str>::from("T"), "B,naive".into()]
        );
        assert_eq!(membership.matrix[(1, 1)], 1.0);
        assert_eq!(membership.matrix[(2, 0)], 1.0);

        let tsv = write_table(
            &dir,
            "cells.tsv",
            &[
                "cell\tindiv\ttype",
                "cell_0\tIND_A\tT",
                "cell_1\tIND_B\tB",
                "cell_2\tIND_B\tT",
            ],
        );
        let anno = read_cell_annotations(&tsv)?;
        assert_eq!(anno.cell_to_individual.len(), 3);
        assert_eq!(individual_of(&anno, "cell_0"), "IND_A");
        assert_eq!(individual_of(&anno, "cell_2"), "IND_B");
        assert_eq!(
            build_onehot_membership(&tsv, &columns)?.cell_type_names,
            vec![Box::<str>::from("T"), "B".into()]
        );
        Ok(())
    }

    #[test]
    fn a_table_where_no_row_parses_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        // The rows do not use the header's delimiter: each is one field.
        let mixed = write_table(
            &dir,
            "mixed.tsv",
            &["cell\tindiv", "cell_0 IND_A", "cell_1 IND_B"],
        );
        let err = read_cell_annotations(&mixed).unwrap_err().to_string();
        assert!(err.contains("2 rows with fewer than 2 fields"), "{err}");

        // Two columns hold no cell type.
        let two = write_table(&dir, "two.tsv", &["cell\tindiv", "cell_0\tIND_A"]);
        assert!(read_cell_annotations(&two).is_ok());
        let columns: Vec<Box<str>> = vec!["cell_0".into()];
        assert!(build_onehot_membership(&two, &columns).is_err());

        let header_only = write_table(&dir, "empty.tsv", &["cell\tindiv"]);
        assert!(read_cell_annotations(&header_only).is_err());
    }

    #[test]
    fn test_infer_cell_annotations() {
        let names: Vec<Box<str>> = vec![
            "ACGT@IND_A".into(),
            "TGCA@IND_A".into(),
            "GGCC@IND_B".into(),
        ];
        let anno = infer_cell_annotations(&names);
        assert_eq!(anno.individual_ids.len(), 2);
        assert_eq!(anno.cell_to_individual.len(), 3);
    }

    #[test]
    fn test_to_column_aligned_vec() {
        let names: Vec<Box<str>> = vec!["ACGT@IND_A".into(), "TGCA@IND_B".into(), "XXXX".into()];
        let anno = infer_cell_annotations(&names);
        let aligned = anno.to_column_aligned_vec(&names, "missing");
        assert_eq!(aligned[0].as_ref(), "IND_A");
        assert_eq!(aligned[1].as_ref(), "IND_B");
        assert_eq!(aligned[2].as_ref(), "all");
    }
}
