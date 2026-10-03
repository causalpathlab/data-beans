//! Load a pre-trained per-gene embedding (and optional per-gene bias)
//! from parquet and strictly intersect its row axis against a caller's
//! target feature axis.
//!
//! Used by `senna gbe / topic / cell-embedded-topic` to freeze the
//! gene-side parameter table (`E_feat` in gbe, ρ in the ETM topic models)
//! so cells train on a shared, pre-fit gene-relation space.
//!
//! Source formats supported via [`FrozenLoadArgs`]:
//! - **gbe**: `{prefix}.dictionary.parquet` + `{prefix}.feature_bias.parquet`
//!   (gene × H plus gene × 1 bias).
//! - **topic / cell-embedded-topic**: `{prefix}.feature_embedding.parquet`
//!   alone — bias defaults to zeros, which is what the topic models use
//!   internally (no per-gene additive bias on ρ).
//!
//! Name resolution goes through [`FeatureNameKind`] so `TGFB1` and
//! `ENSG00000105329_TGFB1` resolve to the same row.

use crate::aux::feature_names::FeatureNameKind;
use legume_numeric::matrix::traits::IoOps;
use nalgebra::DMatrix;
use rustc_hash::{FxHashMap, FxHashSet};

/// Loaded + aligned frozen feature side ready to hand off to a candle
/// engine (`graph-embedding-util` or one of the topic-model encoders).
///
/// Rows are reordered to follow the *target* axis. `keep_target_indices`
/// records which positions in the caller's target feature axis survived
/// the intersection — the caller MUST restrict its data (triplets,
/// encoder D, decoder β) to these indices, otherwise the row order
/// disagrees with the embedding rows.
pub struct FrozenFeatureHost {
    /// `[|keep|, H]`, rows in the same order as `keep_target_indices`.
    pub e_feat: DMatrix<f32>,
    /// `[|keep|]`. Zeros when no `bias_path` was given.
    pub b_feat: Vec<f32>,
    /// Indices into the *target* feature axis that matched a source row.
    /// Length equals `e_feat.nrows()`.
    pub keep_target_indices: Vec<usize>,
    /// The *source* (dictionary) row each kept target row came from, parallel to
    /// `keep_target_indices` — what a caller needs to look up any other table
    /// keyed on the dictionary's rows (a module membership, say).
    pub keep_src_indices: Vec<usize>,
    /// The whole source table `[n_src, H]` and its row names, as read — so a
    /// caller that needs the unmatched rows too (to place a gene by its
    /// neighbours' rows) does not decode the file a second time.
    pub src_e_feat: DMatrix<f32>,
    pub src_names: Vec<Box<str>>,
    /// Rows in the dictionary file, i.e. how many features the MODEL has.
    ///
    /// The only field that survives the intersection unfiltered, and the reason
    /// it exists: `e_feat` and `keep_target_indices` are both already restricted
    /// to the matched features, so a coverage fraction built from them is
    /// identically 1 and tells a caller nothing. This is the denominator.
    pub n_src: usize,
    pub h: usize,
}

/// A rename of source row names, see [`FrozenLoadArgs::source_name_map`].
pub type SourceNameMap<'a> = &'a dyn Fn(&str) -> Box<str>;

pub struct FrozenLoadArgs<'a> {
    /// Path to the `[D_src, H]` parquet (gbe `dictionary.parquet` or
    /// topic `feature_embedding.parquet`). Row column 0 is the gene name.
    pub dictionary_path: &'a str,
    /// Optional path to a `[D_src, 1]` per-gene bias parquet (gbe
    /// `feature_bias.parquet`). `None` → bias filled with zeros, which
    /// matches the topic models' implicit "no per-gene bias on ρ".
    pub bias_path: Option<&'a str>,
    /// Caller's feature axis (e.g. `unified.feature_names` for gbe;
    /// the topic models' `gene_names`). Output rows follow this order
    /// after dropping unmatched entries.
    pub target_feature_names: &'a [Box<str>],
    /// Per-name canonicalization rule applied to both source and target
    /// names before intersection. [`FeatureNameKind::Exact`] for strict
    /// matching; [`FeatureNameKind::Gene { delim: '_' }`] is the typical
    /// choice for scRNA gene IDs.
    pub name_kind: FeatureNameKind,
    /// Applied to each SOURCE row name that may match (every row, unless
    /// [`load_frozen_feature_host_matching`] marks fewer) before
    /// canonicalization, and kept as
    /// the host's `src_names`: how a caller whose axis carries a row grammar
    /// (`{gene}/count/spliced`) reads a plain gene table, lifting each bare
    /// name into the grammar first. `None` = the names as read.
    pub source_name_map: Option<SourceNameMap<'a>>,
}

/// Load the dictionary and match its rows to the target axis by canonical
/// name. Among source rows with one canonical name the first wins.
pub fn load_frozen_feature_host(args: FrozenLoadArgs) -> anyhow::Result<FrozenFeatureHost> {
    load_frozen_feature_host_matching(args, |names| Ok(vec![true; names.len()]))
}

/// [`load_frozen_feature_host`], matching only the source rows `matchable`
/// marks. It is handed the dictionary's row names as read (before
/// [`FrozenLoadArgs::source_name_map`]) and returns one flag per row, BY
/// POSITION: two rows may share a name (a cell type `CD4` beside the gene
/// `CD4`), so marking by name would mark both and the first would still win.
/// [`crate::aux::feature_types::feature_rows`] marks a mixed-type table's
/// gene and region rows from its types table. A row left unmarked is never
/// matched or renamed, yet stays in `src_names` / `src_e_feat`.
pub fn load_frozen_feature_host_matching(
    args: FrozenLoadArgs,
    matchable: impl FnOnce(&[Box<str>]) -> anyhow::Result<Vec<bool>>,
) -> anyhow::Result<FrozenFeatureHost> {
    let dict = <DMatrix<f32> as IoOps>::from_parquet(args.dictionary_path)?;
    let n_src = dict.rows.len();
    let h = dict.mat.ncols();
    anyhow::ensure!(
        h > 0 && dict.mat.nrows() == n_src,
        "{}: malformed dictionary (rows={}, mat dims={}x{})",
        args.dictionary_path,
        n_src,
        dict.mat.nrows(),
        h
    );

    let src_bias: Vec<f32> = match args.bias_path {
        None => vec![0.0; n_src],
        Some(p) => {
            let bias = <DMatrix<f32> as IoOps>::from_parquet(p)?;
            anyhow::ensure!(
                bias.rows == dict.rows,
                "{} row names disagree with {} (both files must come from the same training run)",
                p,
                args.dictionary_path
            );
            anyhow::ensure!(
                bias.mat.ncols() == 1,
                "{}: expected 1 data column (bias), got {}",
                p,
                bias.mat.ncols()
            );
            (0..n_src).map(|i| bias.mat[(i, 0)]).collect()
        }
    };

    let matchable = matchable(&dict.rows)
        .map_err(|e| anyhow::anyhow!("{}: marking its rows: {e}", args.dictionary_path))?;
    anyhow::ensure!(
        matchable.len() == n_src,
        "{}: {} row flags for {} rows",
        args.dictionary_path,
        matchable.len(),
        n_src
    );
    let n_matchable = matchable.iter().filter(|&&m| m).count();
    anyhow::ensure!(
        n_src == 0 || n_matchable > 0,
        "{}: none of its {} rows is marked as one that may match a feature",
        args.dictionary_path,
        n_src
    );
    // Only a row that may match is renamed: an unmarked one keeps its name.
    let src_names: Vec<Box<str>> = match args.source_name_map {
        Some(f) => dict
            .rows
            .iter()
            .zip(&matchable)
            .map(|(n, &m)| if m { f(n) } else { n.clone() })
            .collect(),
        None => dict.rows,
    };
    let mut src_by_canon: FxHashMap<Box<str>, usize> = FxHashMap::default();
    let mut src_dupes = 0usize;
    for (i, name) in src_names.iter().enumerate() {
        if !matchable[i] {
            continue;
        }
        let canon = args.name_kind.canonicalize(name);
        // First occurrence wins (see `load_frozen_feature_host`); `insert`
        // would keep the last.
        if let std::collections::hash_map::Entry::Vacant(e) = src_by_canon.entry(canon) {
            e.insert(i);
        } else {
            src_dupes += 1;
        }
    }
    if src_dupes > 0 {
        log::warn!(
            "{}: {} source rows had duplicate canonical names — kept first occurrence",
            args.dictionary_path,
            src_dupes
        );
    }
    let shadowing = src_names
        .iter()
        .zip(&matchable)
        .filter(|(n, &m)| !m && src_by_canon.contains_key(&args.name_kind.canonicalize(n)))
        .count();
    if shadowing > 0 {
        log::info!(
            "{}: {} unmarked rows share a name with a matchable row and were passed over",
            args.dictionary_path,
            shadowing
        );
    }

    let mut keep_target_indices = Vec::new();
    let mut keep_src_indices = Vec::new();
    for (target_i, name) in args.target_feature_names.iter().enumerate() {
        let canon = args.name_kind.canonicalize(name);
        if let Some(&src_i) = src_by_canon.get(&canon) {
            keep_target_indices.push(target_i);
            keep_src_indices.push(src_i);
        }
    }
    anyhow::ensure!(
        !keep_target_indices.is_empty(),
        "No feature names matched between {} (n={}) and target axis (n={}) under {:?} \
         — check the gene-name kind (Exact / Gene / Locus / Mixed) and source axis",
        args.dictionary_path,
        n_src,
        args.target_feature_names.len(),
        args.name_kind
    );

    let unique_src_used: FxHashSet<usize> = keep_src_indices.iter().copied().collect();
    // A dictionary is a plain gene table. Source rows carrying the channelized
    // row grammar ({gene}/{modality}/... ) that matched nothing usually mean
    // the caller fed a channelized or co-embedding artifact; a PARTIAL match
    // would otherwise proceed silently on the plain-name subset.
    let channelized_unmatched = src_names
        .iter()
        .enumerate()
        .filter(|(i, r)| {
            matchable[*i]
                && !unique_src_used.contains(i)
                && crate::aux::feature_rows::parse_feature_row(r).is_some()
        })
        .count();
    if channelized_unmatched > 0 {
        log::warn!(
            "{}: {} unmatched source rows carry the channelized row grammar — is this a raw gene dictionary, or a channelized/co-embedding output?",
            args.dictionary_path,
            channelized_unmatched
        );
    }
    let matchable_note = if n_matchable < n_src {
        format!("; {n_matchable} source rows may match")
    } else {
        String::new()
    };
    log::info!(
        "Frozen feature side from {}: {}/{} target features matched (H={}, {} of {} source rows reused{}, kind={:?})",
        args.dictionary_path,
        keep_target_indices.len(),
        args.target_feature_names.len(),
        h,
        unique_src_used.len(),
        n_src,
        matchable_note,
        args.name_kind
    );

    let k = keep_target_indices.len();
    let mut e_feat = DMatrix::<f32>::zeros(k, h);
    let mut b_feat = Vec::with_capacity(k);
    for (out_i, &src_i) in keep_src_indices.iter().enumerate() {
        for j in 0..h {
            e_feat[(out_i, j)] = dict.mat[(src_i, j)];
        }
        b_feat.push(src_bias[src_i]);
    }

    Ok(FrozenFeatureHost {
        e_feat,
        b_feat,
        keep_target_indices,
        keep_src_indices,
        src_e_feat: dict.mat,
        src_names,
        n_src,
        h,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use legume_numeric::matrix::traits::IoOps;

    fn write_test_parquet(
        path: &str,
        rows: &[&str],
        row_axis: &str,
        cols: &[&str],
        data: &DMatrix<f32>,
    ) {
        let row_names: Vec<Box<str>> = rows.iter().map(|s| (*s).into()).collect();
        let col_names: Vec<Box<str>> = cols.iter().map(|s| (*s).into()).collect();
        data.to_parquet_with_names(path, (Some(&row_names), Some(row_axis)), Some(&col_names))
            .unwrap();
    }

    #[test]
    fn strict_intersection_drops_unmatched_and_preserves_target_order() {
        let dir = tempfile::tempdir().unwrap();
        let dict_path = dir.path().join("d.parquet").to_str().unwrap().to_string();

        // Source: 4 genes × H=3. Source row "ENSG_DROP" has no target match.
        let src = DMatrix::<f32>::from_row_slice(
            4,
            3,
            &[
                1.0, 2.0, 3.0, // TGFB1
                4.0, 5.0, 6.0, // MYC
                7.0, 8.0, 9.0, // ENSG_DROP (unmatched)
                10.0, 11.0, 12.0, // TP53
            ],
        );
        write_test_parquet(
            &dict_path,
            &["TGFB1", "MYC", "ENSG_DROP", "TP53"],
            "gene",
            &["h0", "h1", "h2"],
            &src,
        );

        // Target: 5 genes; "FOO" and "BAR" don't appear in source.
        let target: Vec<Box<str>> = ["FOO", "TP53", "TGFB1", "BAR", "MYC"]
            .iter()
            .map(|s| (*s).into())
            .collect();

        let host = load_frozen_feature_host(FrozenLoadArgs {
            dictionary_path: &dict_path,
            bias_path: None,
            target_feature_names: &target,
            name_kind: FeatureNameKind::Exact,
            source_name_map: None,
        })
        .unwrap();

        // Kept target indices = positions of TP53, TGFB1, MYC in target order.
        assert_eq!(host.keep_target_indices, vec![1, 2, 4]);
        assert_eq!(host.h, 3);
        assert_eq!(host.e_feat.nrows(), 3);
        assert_eq!(host.b_feat, vec![0.0, 0.0, 0.0]);

        // Row 0 of e_feat should be source row for TP53 (= source row 3).
        assert_eq!(host.e_feat[(0, 0)], 10.0);
        assert_eq!(host.e_feat[(0, 2)], 12.0);
        // Row 1: TGFB1 → source row 0.
        assert_eq!(host.e_feat[(1, 0)], 1.0);
        // Row 2: MYC → source row 1.
        assert_eq!(host.e_feat[(2, 1)], 5.0);
    }

    /// A source of bare gene names read onto an axis that carries the row
    /// grammar: the map lifts each source name into the grammar before the
    /// canonical match, and the host reports the lifted names.
    #[test]
    fn a_source_name_map_is_applied_before_matching_and_kept_in_src_names() {
        let dir = tempfile::tempdir().unwrap();
        let dict_path = dir.path().join("d.parquet").to_str().unwrap().to_string();
        let src = DMatrix::<f32>::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        write_test_parquet(
            &dict_path,
            &["TGFB1", "MYC/count/unspliced"],
            "gene",
            &["h0", "h1"],
            &src,
        );
        let target: Vec<Box<str>> = [
            "ENSG_TGFB1/count/spliced",
            "ENSG_MYC/count/spliced",
            "ENSG_MYC/count/unspliced",
        ]
        .iter()
        .map(|s| (*s).into())
        .collect();
        let lift = |n: &str| -> Box<str> {
            if n.contains('/') {
                n.into()
            } else {
                format!("{n}/count/spliced").into()
            }
        };
        let host = load_frozen_feature_host(FrozenLoadArgs {
            dictionary_path: &dict_path,
            bias_path: None,
            target_feature_names: &target,
            name_kind: FeatureNameKind::Gene { delim: '_' },
            source_name_map: Some(&lift),
        })
        .unwrap();
        assert_eq!(host.keep_target_indices, vec![0, 2]);
        assert_eq!(host.keep_src_indices, vec![0, 1]);
        assert_eq!(
            host.src_names,
            vec![
                Box::<str>::from("TGFB1/count/spliced"),
                Box::<str>::from("MYC/count/unspliced")
            ]
        );
        assert_eq!(host.e_feat[(0, 0)], 1.0);
        assert_eq!(host.e_feat[(1, 1)], 4.0);
    }

    #[test]
    fn gene_canon_matches_across_delim_variants() {
        let dir = tempfile::tempdir().unwrap();
        let dict_path = dir.path().join("d.parquet").to_str().unwrap().to_string();

        // Source uses ENSG-prefixed; target uses bare gene symbols.
        let src = DMatrix::<f32>::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        write_test_parquet(
            &dict_path,
            &["ENSG00000105329_TGFB1", "ENSG00000141510_TP53"],
            "gene",
            &["h0", "h1"],
            &src,
        );
        let target: Vec<Box<str>> = ["TP53", "TGFB1"].iter().map(|s| (*s).into()).collect();

        let host = load_frozen_feature_host(FrozenLoadArgs {
            dictionary_path: &dict_path,
            bias_path: None,
            target_feature_names: &target,
            name_kind: FeatureNameKind::Gene { delim: '_' },
            source_name_map: None,
        })
        .unwrap();

        assert_eq!(host.keep_target_indices, vec![0, 1]);
        // Row 0 (target TP53) ← source row 1.
        assert_eq!(host.e_feat[(0, 0)], 3.0);
        // Row 1 (target TGFB1) ← source row 0.
        assert_eq!(host.e_feat[(1, 0)], 1.0);
    }

    /// An unmarked row is never matched, though it comes first and shares
    /// the gene's name, and stays in the source table.
    #[test]
    fn only_the_marked_rows_match() {
        let dir = tempfile::tempdir().unwrap();
        let dict_path = dir.path().join("d.parquet").to_str().unwrap().to_string();
        let src = DMatrix::<f32>::from_row_slice(3, 2, &[9.0, 9.0, 1.0, 2.0, 3.0, 4.0]);
        write_test_parquet(
            &dict_path,
            &["CD4", "CD4", "MYC"],
            "feature",
            &["h0", "h1"],
            &src,
        );
        let target: Vec<Box<str>> = ["MYC", "CD4"].iter().map(|s| (*s).into()).collect();
        let args = || FrozenLoadArgs {
            dictionary_path: &dict_path,
            bias_path: None,
            target_feature_names: &target,
            name_kind: FeatureNameKind::Exact,
            source_name_map: None,
        };
        let host =
            load_frozen_feature_host_matching(args(), |_| Ok(vec![false, true, true])).unwrap();
        assert_eq!(host.keep_target_indices, vec![0, 1]);
        assert_eq!(host.keep_src_indices, vec![2, 1]);
        assert_eq!(
            host.e_feat.row(1).iter().copied().collect::<Vec<_>>(),
            [1.0, 2.0]
        );
        assert_eq!(host.src_names.len(), 3);

        // Every row unmarked, or the wrong count: refused.
        let err = |m: Vec<bool>| {
            load_frozen_feature_host_matching(args(), |_| Ok(m))
                .err()
                .unwrap()
                .to_string()
        };
        assert!(err(vec![false; 3]).contains("none of its 3 rows is marked"));
        assert!(err(vec![true]).contains("1 row flags for 3 rows"));

        // The marks are asked of the names as read, before any rename.
        let lift = |n: &str| -> Box<str> { format!("{n}/count/spliced").into() };
        let mut seen: Vec<Box<str>> = Vec::new();
        let renamed = FrozenLoadArgs {
            source_name_map: Some(&lift),
            ..args()
        };
        assert!(load_frozen_feature_host_matching(renamed, |names| {
            seen = names.to_vec();
            Ok(vec![false; names.len()])
        })
        .is_err());
        let read: Vec<Box<str>> = vec!["CD4".into(), "CD4".into(), "MYC".into()];
        assert_eq!(seen, read);

        // ...and only a marked row is renamed.
        let target: Vec<Box<str>> = vec!["CD4/count/spliced".into()];
        let host = load_frozen_feature_host_matching(
            FrozenLoadArgs {
                target_feature_names: &target,
                source_name_map: Some(&lift),
                ..args()
            },
            |_| Ok(vec![false, true, true]),
        )
        .unwrap();
        assert_eq!(&*host.src_names[0], "CD4");
        assert_eq!(&*host.src_names[1], "CD4/count/spliced");
        assert_eq!(host.keep_src_indices, vec![1]);
    }

    #[test]
    fn empty_intersection_errors() {
        let dir = tempfile::tempdir().unwrap();
        let dict_path = dir.path().join("d.parquet").to_str().unwrap().to_string();
        let src = DMatrix::<f32>::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        write_test_parquet(&dict_path, &["A", "B"], "gene", &["h0", "h1"], &src);
        let target: Vec<Box<str>> = ["C", "D"].iter().map(|s| (*s).into()).collect();
        let result = load_frozen_feature_host(FrozenLoadArgs {
            dictionary_path: &dict_path,
            bias_path: None,
            target_feature_names: &target,
            name_kind: FeatureNameKind::Exact,
            source_name_map: None,
        });
        let err = match result {
            Ok(_) => panic!("expected empty-intersection error"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("No feature names matched"));
    }

    #[test]
    fn bias_loaded_when_provided() {
        let dir = tempfile::tempdir().unwrap();
        let dict_path = dir.path().join("d.parquet").to_str().unwrap().to_string();
        let bias_path = dir.path().join("b.parquet").to_str().unwrap().to_string();
        let src = DMatrix::<f32>::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        write_test_parquet(&dict_path, &["A", "B"], "gene", &["h0", "h1"], &src);
        let bias = DMatrix::<f32>::from_row_slice(2, 1, &[0.5, -0.3]);
        write_test_parquet(&bias_path, &["A", "B"], "gene", &["bias"], &bias);

        let target: Vec<Box<str>> = ["B", "A"].iter().map(|s| (*s).into()).collect();
        let host = load_frozen_feature_host(FrozenLoadArgs {
            dictionary_path: &dict_path,
            bias_path: Some(&bias_path),
            target_feature_names: &target,
            name_kind: FeatureNameKind::Exact,
            source_name_map: None,
        })
        .unwrap();
        // Row 0 of output = target "B" = source row 1.
        assert_eq!(host.b_feat, vec![-0.3, 0.5]);
    }
}
