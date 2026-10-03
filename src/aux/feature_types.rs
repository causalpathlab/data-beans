//! The typed feature table that rides beside a mixed-type embedding:
//! `{prefix}.feature_types.parquet`, string columns `feature` and `type`, one
//! row per embedding row in the same order. `senna fne` and `gene-text` write
//! it; a consumer that wants only one type of row (`senna bge` pinning gene
//! rows, say) reads it. The type names are the shared vocabulary.

use legume_numeric::matrix::parquet::{
    read_parquet_string_columns_by_name, write_named_table, Column,
};
use std::path::Path;

/// Nodes whose names are canonicalised as gene symbols.
pub const GENE_TYPE: &str = "gene";
/// Ontology terms and gene sets.
pub const TERM_TYPE: &str = "term";
/// Fixed genomic windows.
pub const REGION_TYPE: &str = "region";
/// Vocabulary words of a text relation.
pub const WORD_TYPE: &str = "word";

/// Whether rows of type `t` name a data feature — a gene, or a genomic window
/// a peak can match — rather than a term, word or cell type.
#[must_use]
pub fn is_data_feature_type(t: &str) -> bool {
    matches!(t, GENE_TYPE | REGION_TYPE)
}

pub fn feature_types_path(prefix: &str) -> String {
    format!("{prefix}.feature_types.parquet")
}

/// Write the table for `names[i]` of type `types[i]`.
pub fn write_feature_types(
    prefix: &str,
    names: &[Box<str>],
    types: &[Box<str>],
) -> anyhow::Result<()> {
    anyhow::ensure!(
        names.len() == types.len(),
        "feature types: {} names for {} types",
        names.len(),
        types.len()
    );
    write_named_table(
        &feature_types_path(prefix),
        "feature",
        names,
        &[(Box::from("type"), Column::Str(types))],
    )
}

/// One flag per row of a table (`names`, in order): true for a gene or a
/// genomic window, by the table's types table `types`. `None` when `types`
/// does not list `names` in order — written for another table under the
/// same prefix. For [`crate::aux::frozen_features::load_frozen_feature_host_matching`].
#[must_use]
pub fn feature_rows(types: &[FeatureType], names: &[Box<str>]) -> Option<Vec<bool>> {
    if !types.iter().map(|(n, _)| n).eq(names.iter()) {
        return None;
    }
    Some(types.iter().map(|(_, t)| is_data_feature_type(t)).collect())
}

/// One row of the table: the feature's name and its type.
pub type FeatureType = (Box<str>, Box<str>);

/// The run's rows; `None` when the run wrote no table, which a caller reads as
/// "every row is of the one type it expects".
pub fn read_feature_types(prefix: &str) -> anyhow::Result<Option<Vec<FeatureType>>> {
    let path = feature_types_path(prefix);
    if !Path::new(&path).exists() {
        return Ok(None);
    }
    let mut cols = read_parquet_string_columns_by_name(&path, &["feature", "type"])?;
    let types = cols.pop().expect("two columns requested");
    let names = cols.pop().expect("two columns requested");
    Ok(Some(names.into_iter().zip(types).collect()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_rows_are_marked_by_position_for_the_table_listed() {
        let t = |n: &str, ty: &str| -> FeatureType { (n.into(), ty.into()) };
        let types = [
            t("CD4", "cell_type"),
            t("CD4", "gene"),
            t("chr1:0-5000", "region"),
        ];
        let names: Vec<Box<str>> = vec!["CD4".into(), "CD4".into(), "chr1:0-5000".into()];
        assert_eq!(feature_rows(&types, &names), Some(vec![false, true, true]));
        assert_eq!(feature_rows(&types, &names[..2]), None);
        let other: Vec<Box<str>> = vec!["CD4".into(), "MYC".into(), "chr1:0-5000".into()];
        assert_eq!(feature_rows(&types, &other), None);
        assert!(is_data_feature_type(GENE_TYPE) && !is_data_feature_type(WORD_TYPE));
    }

    #[test]
    fn round_trip_and_absence() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("run").to_string_lossy().into_owned();
        assert!(read_feature_types(&prefix).unwrap().is_none());
        let names: Vec<Box<str>> = vec!["TP53".into(), "GO:1".into()];
        let types: Vec<Box<str>> = vec![GENE_TYPE.into(), TERM_TYPE.into()];
        write_feature_types(&prefix, &names, &types).unwrap();
        let rows = read_feature_types(&prefix).unwrap().unwrap();
        assert_eq!(
            rows,
            vec![
                ("TP53".into(), GENE_TYPE.into()),
                ("GO:1".into(), TERM_TYPE.into())
            ]
        );
        assert!(write_feature_types(&prefix, &names, &types[..1]).is_err());
    }
}
