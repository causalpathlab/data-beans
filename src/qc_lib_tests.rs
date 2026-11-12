use super::*;

/// Build a `QcArgs` with clap's declared defaults, without going through a
/// parser — mirrors what every subcommand gets when the user passes no QC flags.
fn default_args() -> QcArgs {
    QcArgs {
        no_qc: false,
        qc_mads: 5.0,
        qc_min_cell_nnz: 2,
        qc_min_counts: 0.0,
        qc_mito_pattern: None,
        qc_mito_max_frac: None,
        qc_ribo_pattern: None,
        qc_ribo_max_frac: None,
        qc_feature_min_cells: 0,
        qc_report: None,
        qc_histogram: false,
        qc_mad_on_genes: true,
        qc_mad_on_counts: true,
        qc_mad_on_mito: true,
        qc_auto_cutoff: false,
    }
}

/// The MAD gates must REACH `QcConfig`.
///
/// They were hardcoded `false` in `to_config` with no CLI path, which also made
/// `--qc-mads` inert: nothing except a MAD gate reads `n_mads`, so the flag did
/// nothing at all unless `--qc-mito-pattern` happened to be set. This test is
/// the regression guard for that — it fails if either gate is pinned again.
#[test]
fn mad_gates_reach_the_config_and_are_on_by_default() {
    let cfg = default_args().to_config().expect("QC is on by default");
    assert!(cfg.mad_on_n_genes, "--qc-mad-on-genes must reach QcConfig");
    assert!(cfg.mad_on_counts, "--qc-mad-on-counts must reach QcConfig");
    assert_eq!(cfg.n_mads, 5.0, "--qc-mads must reach QcConfig");

    let mut off = default_args();
    off.qc_mad_on_genes = false;
    off.qc_mad_on_counts = false;
    let cfg = off.to_config().unwrap();
    assert!(
        !cfg.mad_on_n_genes && !cfg.mad_on_counts,
        "the gates must be settable OFF"
    );
}

/// `--qc-auto-cutoff` must reach `QcConfig` and stay off by default.
///
/// It was documented in chickpea's help before it existed as a flag, and
/// `auto_cell_cutoff` was hardcoded `false`.
#[test]
fn auto_cutoff_is_reachable_but_off_by_default() {
    assert!(!default_args().to_config().unwrap().auto_cell_cutoff);

    let mut on = default_args();
    on.qc_auto_cutoff = true;
    assert!(on.to_config().unwrap().auto_cell_cutoff);
}

/// Mito stays implicit: it can only run when a pattern selects rows to measure.
/// Without one there is nothing to compute a fraction over, so the gate must
/// stay off rather than silently reporting 0.0 for every cell.
#[test]
fn mito_gate_follows_the_pattern_not_a_flag() {
    assert!(!default_args().to_config().unwrap().mad_on_mito);

    let mut with = default_args();
    with.qc_mito_pattern = Some("(?i)^MT-".to_string());
    assert!(with.to_config().unwrap().mad_on_mito);
}

/// `--no-qc` still short-circuits everything.
#[test]
fn no_qc_disables_the_whole_config() {
    let mut off = default_args();
    off.no_qc = true;
    assert!(off.to_config().is_none());
}

/// `n` values spread evenly over `centre` times 0.8 to 1.2.
fn around(centre: f32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (centre * (0.8 + 0.4 * i as f32 / n as f32)).round())
        .collect()
}

fn metrics_of(n_genes: Vec<f32>) -> QcMetrics {
    QcMetrics {
        total_counts: n_genes.iter().map(|g| 3.0 * g).collect(),
        n_genes,
        mito_frac: None,
        ribo_frac: None,
        feature_n_cells: None,
        n_rows: 0,
    }
}

fn auto_cutoff_alone() -> QcConfig {
    QcConfig {
        auto_cell_cutoff: true,
        mad_on_n_genes: false,
        mad_on_counts: false,
        ..QcConfig::default()
    }
}

/// Carried pseudobulks, far deeper than the cells, make a second peak; the
/// trough before it is not a cell cutoff.
#[test]
fn exempt_columns_do_not_shape_the_auto_cutoff() {
    let (cells, carried) = (400, 200);
    let mut n_genes = around(300.0, cells);
    n_genes.extend(around(8000.0, carried));
    let exempt: Vec<bool> = (0..cells + carried).map(|c| c >= cells).collect();
    let cut = crate::qc::suggest_nnz_cutoff(&n_genes).expect("two peaks over every column");
    assert!(cut > 400 && cut < 6000);

    let v = cell_verdicts(&metrics_of(n_genes), &auto_cutoff_alone(), Some(&exempt));
    assert!(v.train_keep.iter().all(|&k| k));
    assert_eq!(v.n_cells_dropped, 0);
    assert!(!v.guarded);
}

/// An exempt column below the cells' trough is not dropped with the ambient.
#[test]
fn exempt_columns_do_not_fall_to_the_auto_cutoff() {
    let (ambient, cells, carried) = (300, 300, 10);
    let mut n_genes = around(20.0, ambient);
    n_genes.extend(around(3000.0, cells));
    n_genes.extend(around(5.0, carried));
    let n = n_genes.len();
    let exempt: Vec<bool> = (0..n).map(|c| c >= ambient + cells).collect();

    let v = cell_verdicts(&metrics_of(n_genes), &auto_cutoff_alone(), Some(&exempt));
    assert_eq!(v.n_cells_dropped, ambient);
    assert!((0..n).all(|c| v.train_keep[c] == (c >= ambient)));
    assert!((ambient + cells..n).all(|c| !v.near_empty[c]));
}

/// Rules that fail every cell are held back although exempt columns, which
/// are never dropped, would remain.
#[test]
fn the_guardrail_counts_cells_not_exempt_columns() {
    let (cells, carried) = (100, 5);
    let exempt: Vec<bool> = (0..cells + carried).map(|c| c >= cells).collect();
    let cfg = QcConfig {
        min_counts_per_cell: 1e6,
        ..QcConfig::default()
    };
    let v = cell_verdicts(
        &metrics_of(around(300.0, cells + carried)),
        &cfg,
        Some(&exempt),
    );
    assert!(v.guarded);
    assert!(v.train_keep.iter().all(|&k| k));
    assert_eq!(v.n_cells_dropped, 0);
}

/// The metrics of one pass are those of a pass per metric.
#[test]
fn metrics_match_a_pass_per_metric() {
    use crate::qc::collect_row_stat_across_vec;
    use crate::sparse_io::create_sparse_from_triplets;
    let (nrows, ncols) = (30usize, 120usize);
    let mut triplets = vec![];
    for c in 0..ncols as u64 {
        for r in 0..nrows as u64 {
            let h = (r * 7919 + c * 104_729 + r * c) % 89;
            if h < 25 && c % 40 != 7 {
                triplets.push((r, c, 1.0 + (h % 6) as f32));
            }
        }
    }
    let shape = (nrows, ncols, triplets.len());
    let mut data = create_sparse_from_triplets(&triplets, shape, None, None).unwrap();
    let rows: Vec<Box<str>> = (0..nrows)
        .map(|r| match r % 10 {
            0 => format!("MT-G{r}").into(),
            3 => format!("RPS{r}").into(),
            _ => format!("G{r}").into(),
        })
        .collect();
    data.register_row_names_vec(&rows);
    let cols: Vec<Box<str>> = (0..ncols).map(|c| format!("c{c}").into()).collect();
    data.register_column_names_vec(&cols);
    let mut vec = SparseIoVec::new();
    vec.push(std::sync::Arc::from(data), None).unwrap();

    let cfg = QcConfig {
        mito_pattern: Some("^MT-".into()),
        ribo_pattern: Some("^RPS".into()),
        ..QcConfig::default()
    };
    let block = Some(16);
    let m = compute_qc_metrics(&vec, &cfg, block, true).unwrap();

    let all = collect_column_stat_across_vec(&vec, None, block).unwrap();
    assert_eq!(m.n_genes, all.count_positives());
    assert_eq!(m.total_counts, all.sum());
    assert!(m.total_counts.contains(&0.0) && m.total_counts.iter().any(|&t| t > 0.0));
    let frac = |pattern: &str| -> Vec<f32> {
        let picked = resolve_rows_by_regex(&rows, pattern).unwrap();
        assert!(!picked.is_empty());
        collect_column_stat_across_vec(&vec, Some(&picked), block)
            .unwrap()
            .sum()
            .iter()
            .zip(&m.total_counts)
            .map(|(&s, &t)| if t > 0.0 { s / t } else { 0.0 })
            .collect()
    };
    let (mito, ribo) = (frac("^MT-"), frac("^RPS"));
    assert_ne!(mito, ribo);
    assert_eq!(m.mito_frac.as_deref(), Some(&mito[..]));
    assert_eq!(m.ribo_frac.as_deref(), Some(&ribo[..]));
    assert_eq!(
        m.feature_n_cells,
        Some(
            collect_row_stat_across_vec(&vec, block)
                .unwrap()
                .count_positives()
        )
    );

    // A pattern that matches nothing leaves its metric out and the other in place.
    let cfg = QcConfig {
        mito_pattern: Some("^NONE".into()),
        ribo_pattern: Some("^RPS".into()),
        ..QcConfig::default()
    };
    let m = compute_qc_metrics(&vec, &cfg, block, false).unwrap();
    assert!(m.mito_frac.is_none() && m.feature_n_cells.is_none());
    assert_eq!(m.ribo_frac.as_deref(), Some(&ribo[..]));

    let m = compute_qc_metrics(&vec, &QcConfig::default(), block, false).unwrap();
    assert!(m.mito_frac.is_none() && m.ribo_frac.is_none());
    assert_eq!(m.total_counts, all.sum());
}
