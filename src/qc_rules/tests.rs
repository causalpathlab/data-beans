use super::*;
use crate::qc_lib::{cell_verdicts, QcArgs};
use clap::{CommandFactory, FromArgMatches, Parser};

/// 1000 cells: most around 2000 counts over 800 genes, 60 near-ambient,
/// 15 with a high mito fraction; 300 genes detected in 1 to 300 cells.
fn metrics() -> QcMetrics {
    let mut rng = 7u64;
    let mut next = move || {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng >> 33) as f32 / (1u64 << 31) as f32
    };
    let (mut n_genes, mut total, mut mito) = (vec![], vec![], vec![]);
    for c in 0..1000 {
        let ambient = c % 17 == 0;
        let depth = if ambient { 60.0 } else { 2000.0 } * (0.6 + 0.8 * next());
        total.push(depth.round());
        n_genes.push((depth * 0.4).round().max(1.0));
        let m = if c % 67 == 0 {
            0.3 + 0.2 * next()
        } else {
            0.02 + 0.03 * next()
        };
        mito.push(m);
    }
    QcMetrics {
        n_genes,
        total_counts: total,
        mito_frac: Some(mito),
        ribo_frac: None,
        feature_n_cells: Some((0..300).map(|g| (1 + g) as f32).collect()),
        n_rows: 300,
    }
}

fn config() -> QcConfig {
    QcConfig {
        mito_pattern: Some("(?i)^MT-".into()),
        mad_on_mito: true,
        ..QcConfig::default()
    }
}

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    qc: QcArgs,
}

/// The config `flags` stand for, read as `data-beans qc` reads them: from
/// its own mito pattern, (?i)^MT-, unless one is given.
fn read_back(flags: &[String]) -> QcConfig {
    let argv = std::iter::once("qc".to_string()).chain(flags.iter().cloned());
    let cmd = Cli::command().mut_arg("qc_mito_pattern", |a| a.default_value("(?i)^MT-"));
    let matches = cmd.try_get_matches_from(argv).unwrap();
    Cli::from_arg_matches(&matches)
        .unwrap()
        .qc
        .to_config()
        .unwrap()
}

fn assert_flags_decide_the_same(rules: &CellRules) {
    let cfg = read_back(&rules.flags());
    assert_eq!(
        &cell_verdicts(rules.metrics(), &cfg, None),
        rules.verdicts()
    );
}

#[test]
fn untouched_rules_decide_as_qc_lib() {
    let rules = CellRules::new(metrics(), config());
    assert_eq!(
        rules.verdicts(),
        &cell_verdicts(&metrics(), &config(), None)
    );
    assert_eq!(
        rules.rules(),
        &[Rule::Genes, Rule::Counts, Rule::Mito, Rule::GeneCells]
    );
    assert_flags_decide_the_same(&rules);
}

#[test]
fn moved_rules_print_flags_that_decide_the_same() {
    let mut rules = CellRules::new(metrics(), config());
    rules.set_mads(3.0);
    assert_flags_decide_the_same(&rules);
    rules.set_value(Rule::Counts, 900.0);
    assert_flags_decide_the_same(&rules);
    rules.toggle_mode(Rule::Mito);
    assert_eq!(rules.mode(Rule::Mito), Mode::Value);
    assert_flags_decide_the_same(&rules);
    rules.set_value(Rule::Mito, 0.1);
    rules.set_value(Rule::GeneCells, 20.0);
    assert_flags_decide_the_same(&rules);
    rules.set_on(Rule::Genes, false);
    assert_flags_decide_the_same(&rules);
    assert_eq!(rules.kept_genes().len(), 300 - 19);
}

#[test]
fn counts_add_up() {
    let rules = CellRules::new(metrics(), config());
    let removed = rules.verdicts().train_keep.len() - rules.kept_cells().len();
    assert!(removed > 0);
    for &r in rules.rules() {
        let c = rules.count(r);
        assert!(c.only <= c.removes, "{r:?}");
    }
    // The ambient cells fail both the genes and the counts band, so neither
    // removes them alone; the mito cells only mito removes.
    assert_eq!(rules.count(Rule::Genes).only, 0);
    assert_eq!(rules.count(Rule::Counts).only, 0);
    assert!(rules.count(Rule::Mito).only >= 14);
    let groups = rules.overlap();
    assert_eq!(groups.iter().map(|g| g.1).sum::<usize>(), removed);
    assert_eq!(groups[0].0, vec![Rule::Genes, Rule::Counts]);
}

#[test]
fn edges_follow_the_settings() {
    let mut rules = CellRules::new(metrics(), config());
    let wide = rules.edge(Rule::Counts).unwrap();
    rules.set_mads(2.0);
    assert!(rules.edge(Rule::Counts).unwrap() > wide);
    rules.toggle_mode(Rule::Counts);
    assert_eq!(rules.mode(Rule::Counts), Mode::Value);
    let held = rules.edge(Rule::Counts).unwrap();
    rules.set_mads(8.0);
    assert_eq!(rules.edge(Rule::Counts).unwrap(), held);
    rules.toggle_mode(Rule::Counts);
    assert_eq!(rules.mode(Rule::Counts), Mode::Mad);
    assert_eq!(rules.edge(Rule::GeneCells), None);
}

#[test]
fn a_rule_turned_off_and_on_comes_back() {
    let mut rules = CellRules::new(metrics(), config());
    let before = rules.verdicts().clone();
    rules.set_on(Rule::Mito, false);
    assert!(!rules.is_on(Rule::Mito));
    assert_eq!(rules.count(Rule::Mito).removes, 0);
    assert!(rules.kept_cells().len() > before.train_keep.iter().filter(|&&k| k).count() - 60);
    rules.set_on(Rule::Mito, true);
    assert_eq!(rules.verdicts(), &before);
    rules.set_value(Rule::Genes, 5.0);
    rules.reset();
    assert_eq!(rules.verdicts(), &before);
}

#[test]
fn mito_turned_off_stays_off_in_the_flags() {
    // Without a pattern; read back, `qc`'s own (?i)^MT- must not return.
    let mut m = metrics();
    m.mito_frac = None;
    let cfg = QcConfig {
        mito_pattern: None,
        mad_on_mito: false,
        ..QcConfig::default()
    };
    let rules = CellRules::new(m, cfg);
    let flags = rules.flags();
    let at = flags.iter().position(|f| f == "--qc-mito-pattern").unwrap();
    assert_eq!(flags[at + 1], "");
    let back = read_back(&flags);
    assert_eq!(back.mito_pattern, None);
    assert!(!back.mad_on_mito);
}

#[test]
fn a_gene_rule_turned_off_under_auto_cutoff_stays_off() {
    let cfg = QcConfig {
        auto_cell_cutoff: true,
        feature_min_cells: 7,
        ..config()
    };
    let mut rules = CellRules::new(metrics(), cfg);
    rules.set_on(Rule::GeneCells, false);
    let flags = rules.flags();
    let at = flags
        .iter()
        .position(|f| f == "--qc-feature-min-cells")
        .unwrap();
    assert_eq!(flags[at + 1], "0", "{flags:?}");
    assert_flags_decide_the_same(&rules);
}

#[test]
fn a_rule_with_no_spread_switched_to_a_value_removes_nothing() {
    // Most cells without a mito read: no MAD, so no band and no edge.
    let mut m = metrics();
    m.mito_frac = Some(
        (0..1000)
            .map(|c| if c % 10 == 0 { 0.05 } else { 0.0 })
            .collect(),
    );
    let mut rules = CellRules::new(m, config());
    assert_eq!(rules.edge(Rule::Mito), None);
    let before = rules.kept_cells().len();
    rules.toggle_mode(Rule::Mito);
    assert_eq!(rules.mode(Rule::Mito), Mode::Value);
    assert_eq!(rules.count(Rule::Mito).removes, 0);
    assert_eq!(rules.kept_cells().len(), before);
}
