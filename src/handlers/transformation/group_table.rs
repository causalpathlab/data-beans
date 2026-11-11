//! Group labels for a backend's cells, from the table `split --groups` and
//! `subsample --groups` read.
//!
//! Two forms, both `.gz` or not:
//!
//! - a two-column table (cell name, group label), `.parquet` or delimited
//!   text, matched to the cells by name, EXACTLY;
//! - one label per line, as many lines as cells, in the backend's column
//!   order: the `{out}.batch.gz` that `qc` and `merge-backend` write beside
//!   a pooled output, and that senna's `-b` reads.
//!
//! Matching stays exact. `Membership` offers base-key and prefix fallbacks,
//! which are right for joining annotations and wrong here: a fuzzy match puts
//! a cell in the wrong half or the wrong quota, and nothing downstream can
//! detect it.

use rustc_hash::FxHashMap as HashMap;

/// Each cell's group label, in column order. A cell the table does not name
/// is an error rather than a group of its own: a partial grouping would
/// treat part of the data by group and the rest by cell, and the result is
/// neither.
pub(super) fn column_labels(
    path: &str,
    column_names: &[Box<str>],
) -> anyhow::Result<Vec<Box<str>>> {
    match read_table(path)? {
        Table::InOrder(labels) => in_order(path, labels, column_names),
        Table::Named(names, labels) => by_name(path, &names, &labels, column_names),
    }
}

/// Labels one per line, checked to be this backend's: as many as cells,
/// and not a bare cell list.
fn in_order(
    path: &str,
    labels: Vec<Box<str>>,
    column_names: &[Box<str>],
) -> anyhow::Result<Vec<Box<str>>> {
    let ncol = column_names.len();
    anyhow::ensure!(
        labels.len() == ncol,
        "{path}: one label per line, but {} lines for {ncol} cells; \
         give a two-column table (cell name, group label) instead",
        labels.len()
    );
    // A bare cell list has one field per line too, and read as labels it
    // puts every cell in a group of its own (or nearly, where barcodes
    // repeat across a pool): a plain random draw wearing the flag's name.
    let distinct: rustc_hash::FxHashSet<&str> = labels.iter().map(AsRef::as_ref).collect();
    let cell_list = labels == column_names || (ncol >= 2 && distinct.len() == ncol);
    anyhow::ensure!(
        !cell_list,
        "{path}: one field per line that reads as a list of cells, \
         not group labels; give a two-column table (cell name, group label)"
    );
    Ok(labels)
}

/// Each cell's label, looked up by its exact name.
fn by_name(
    path: &str,
    names: &[Box<str>],
    labels: &[Box<str>],
    column_names: &[Box<str>],
) -> anyhow::Result<Vec<Box<str>>> {
    anyhow::ensure!(
        names.len() == labels.len(),
        "{path}: {} names but {} group labels",
        names.len(),
        labels.len()
    );
    let label_of: HashMap<&str, &Box<str>> = names.iter().map(AsRef::as_ref).zip(labels).collect();
    let out: Vec<Box<str>> = column_names
        .iter()
        .filter_map(|name| label_of.get(name.as_ref()).map(|&l| l.clone()))
        .collect();
    let n_missing = column_names.len() - out.len();
    anyhow::ensure!(
        n_missing == 0,
        "{path}: {n_missing} of {} cells have no group label",
        column_names.len()
    );
    Ok(out)
}

/// Each column's group, each group's columns, and the groups' labels;
/// groups in order of first appearance.
pub(super) fn groups_of(labels: &[Box<str>]) -> (Vec<usize>, Vec<Vec<usize>>, Vec<Box<str>>) {
    let mut group_of: HashMap<&str, usize> = HashMap::default();
    let mut of_column = Vec::with_capacity(labels.len());
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut names: Vec<Box<str>> = Vec::new();
    for (column, label) in labels.iter().enumerate() {
        let next = groups.len();
        let g = *group_of.entry(label.as_ref()).or_insert(next);
        if g == next {
            groups.push(Vec::new());
            names.push(label.clone());
        }
        groups[g].push(column);
        of_column.push(g);
    }
    (of_column, groups, names)
}

enum Table {
    /// `(name, label)` columns.
    Named(Vec<Box<str>>, Vec<Box<str>>),
    /// One label per line, in column order.
    InOrder(Vec<Box<str>>),
}

/// The table at `path`, from parquet or delimited text.
///
/// Both, because the flag says `TABLE` and the help says "two-column table":
/// reading only parquet meant the TSV a user would naturally write failed with
/// a parquet decode error. The text path goes through the workspace reader, so
/// it sniffs the delimiter and handles `.gz` like every other list this tool
/// takes. A text file with one field on every line is labels in column order.
fn read_table(path: &str) -> anyhow::Result<Table> {
    let is_parquet = std::path::Path::new(path)
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .is_some_and(|e| e.eq_ignore_ascii_case("parquet"));
    if is_parquet {
        return Ok(Table::Named(
            legume_numeric::matrix::parquet::read_parquet_string_column(path, 0)?,
            legume_numeric::matrix::parquet::read_parquet_string_column(path, 1)?,
        ));
    }

    let delim = legume_numeric::matrix::membership::detect_delimiter(path);
    let out = legume_numeric::matrix::common_io::read_lines_of_words_delim(path, delim, -1)?;
    anyhow::ensure!(!out.lines.is_empty(), "{path}: no group labels");
    if out.lines.iter().all(|f| f.len() == 1) {
        return Ok(Table::InOrder(
            out.lines
                .into_iter()
                .map(|mut f| f.swap_remove(0))
                .collect(),
        ));
    }
    let mut names = Vec::with_capacity(out.lines.len());
    let mut labels = Vec::with_capacity(out.lines.len());
    for (row, fields) in out.lines.iter().enumerate() {
        anyhow::ensure!(
            fields.len() >= 2,
            "{path}:{}: {} column(s), need a cell name and a group label",
            row + 1,
            fields.len()
        );
        names.push(fields[0].clone());
        labels.push(fields[1].clone());
    }
    Ok(Table::Named(names, labels))
}
