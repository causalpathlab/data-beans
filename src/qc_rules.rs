//! Cell QC as rules one can move: the settings of [`crate::qc_lib`] seen
//! rule by rule, each with the edge it holds a metric to, how many cells it
//! fails on its own, and how many it alone removes.
//!
//! Nothing here decides anything [`cell_verdicts`] does not: a [`CellRules`]
//! holds a [`QcConfig`] and the [`QcMetrics`] measured once, and every count
//! is that function run again under the config with a rule turned off or
//! left alone. Moving a rule changes the config, so [`CellRules::flags`]
//! gives `qc` flags that decide the same cells. What a frame draws (the
//! bands, edges, removed masks and their overlap) is worked out once per
//! change, in [`CellRules::recount`], not per frame.

use crate::qc_lib::{cell_verdicts_quiet, robust_band, QcConfig, QcMetrics, RobustBand, Verdicts};
use rayon::prelude::*;

/// One rule of cell QC, and the gene rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rule {
    /// Detected genes per cell: the near-empty floor, the MAD band's lower
    /// edge, and the automatic cutoff.
    Genes,
    /// Total counts per cell: the MAD band's lower edge and the floor.
    Counts,
    /// Mitochondrial fraction: the MAD band's upper edge and the cap.
    Mito,
    /// Ribosomal fraction: the cap.
    Ribo,
    /// Cells detecting each gene (the feature axis).
    GeneCells,
}

impl Rule {
    /// What a table or figure calls it.
    pub fn label(self) -> &'static str {
        match self {
            Rule::Genes => "detected genes",
            Rule::Counts => "total counts",
            Rule::Mito => "mito fraction",
            Rule::Ribo => "ribo fraction",
            Rule::GeneCells => "cells per gene",
        }
    }

    /// A short name, for the overlap of rules a cell fails.
    pub fn short(self) -> &'static str {
        match self {
            Rule::Genes => "genes",
            Rule::Counts => "counts",
            Rule::Mito => "mito",
            Rule::Ribo => "ribo",
            Rule::GeneCells => "gene cells",
        }
    }

    /// Whether it decides cells (else genes).
    pub fn on_cells(self) -> bool {
        self != Rule::GeneCells
    }

    /// Whether it keeps what is at or above its edge (else at or below).
    pub fn keeps_above(self) -> bool {
        matches!(self, Rule::Genes | Rule::Counts | Rule::GeneCells)
    }

    /// Whether its metric is a count, read on a log axis (else a fraction).
    pub fn count(self) -> bool {
        self.keeps_above()
    }
}

/// How a rule's edge is set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// By the MAD band (`--qc-mads`, shared by every banded rule).
    Mad,
    /// By a value of its own.
    Value,
}

/// What one rule does under the current settings.
#[derive(Clone, Debug, Default)]
pub struct RuleCount {
    /// Cells (genes, for [`Rule::GeneCells`]) it fails on its own.
    pub removes: usize,
    /// Of those, the ones no other rule removes.
    pub only: usize,
    /// Which pass every other rule.
    pub pass_others: Vec<bool>,
    /// Which it fails on its own.
    pub fails: Vec<bool>,
    /// Which only it removes: removed, and passing every other rule.
    pub only_mask: Vec<bool>,
}

/// Cell QC as rules, with what each does.
pub struct CellRules {
    metrics: QcMetrics,
    cfg: QcConfig,
    /// The settings it started from, which a rule turned back on returns to.
    base: QcConfig,
    rules: Vec<Rule>,
    verdicts: Verdicts,
    counts: Vec<RuleCount>,
    /// Per rule, in table order: its metric's robust band, raw, and the
    /// edge it holds the metric to.
    bands: Vec<Option<RobustBand>>,
    edges: Vec<Option<f32>>,
    /// Whether each cell, and each gene, is removed.
    removed_cells: Vec<bool>,
    removed_genes: Vec<bool>,
    /// The removed cells by the rules they fail; see [`CellRules::overlap`].
    groups: Vec<(Vec<Rule>, usize)>,
}

impl CellRules {
    /// Rules over `metrics` (with its per-gene cell counts measured) under
    /// `cfg`. The histogram printing of `cfg` is left to `compute_qc`.
    pub fn new(metrics: QcMetrics, mut cfg: QcConfig) -> Self {
        cfg.qc_histogram = false;
        let mut rules = vec![Rule::Genes, Rule::Counts];
        if metrics.mito_frac.is_some() {
            rules.push(Rule::Mito);
        }
        if metrics.ribo_frac.is_some() {
            rules.push(Rule::Ribo);
        }
        if metrics.feature_n_cells.is_some() {
            rules.push(Rule::GeneCells);
        }
        let verdicts = cell_verdicts_quiet(&metrics, &cfg);
        let mut out = Self {
            metrics,
            base: cfg.clone(),
            cfg,
            rules,
            verdicts,
            counts: vec![],
            bands: vec![],
            edges: vec![],
            removed_cells: vec![],
            removed_genes: vec![],
            groups: vec![],
        };
        out.recount();
        out
    }

    pub fn metrics(&self) -> &QcMetrics {
        &self.metrics
    }

    pub fn config(&self) -> &QcConfig {
        &self.cfg
    }

    /// The rules the data has, in table order.
    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }

    /// The verdicts under the current settings.
    pub fn verdicts(&self) -> &Verdicts {
        &self.verdicts
    }

    fn index(&self, rule: Rule) -> usize {
        self.rules
            .iter()
            .position(|&r| r == rule)
            .expect("a rule of the data")
    }

    /// What `rule` does now.
    pub fn count(&self, rule: Rule) -> &RuleCount {
        &self.counts[self.index(rule)]
    }

    /// Whether each cell (each gene, for [`Rule::GeneCells`]) is removed now.
    pub fn removed(&self, rule: Rule) -> &[bool] {
        match rule.on_cells() {
            true => &self.removed_cells,
            false => &self.removed_genes,
        }
    }

    /// The robust band of `rule`'s metric on its own scale, as qc_lib fits
    /// it (over the cells above the near-empty floor), whether or not the
    /// rule holds to it; `None` for a rule with no band, or no spread.
    pub fn band(&self, rule: Rule) -> Option<RobustBand> {
        self.bands[self.index(rule)]
    }

    /// The cells QC keeps (in the output), and how many there are.
    pub fn kept_cells(&self) -> Vec<usize> {
        (0..self.verdicts.train_keep.len())
            .filter(|&c| self.verdicts.emitted(c))
            .collect()
    }

    /// The genes QC keeps.
    pub fn kept_genes(&self) -> Vec<usize> {
        (0..self.verdicts.feature_keep.len())
            .filter(|&r| self.verdicts.feature_keep[r])
            .collect()
    }

    /// The values `rule` is held to: one per cell, or per gene.
    pub fn values(&self, rule: Rule) -> &[f32] {
        let m = &self.metrics;
        match rule {
            Rule::Genes => &m.n_genes,
            Rule::Counts => &m.total_counts,
            Rule::Mito => m.mito_frac.as_deref().unwrap_or(&[]),
            Rule::Ribo => m.ribo_frac.as_deref().unwrap_or(&[]),
            Rule::GeneCells => m.feature_n_cells.as_deref().unwrap_or(&[]),
        }
    }

    /// Whether `rule` decides anything now.
    pub fn is_on(&self, rule: Rule) -> bool {
        is_on(&self.cfg, rule)
    }

    /// How `rule`'s edge is set now.
    pub fn mode(&self, rule: Rule) -> Mode {
        let c = &self.cfg;
        let mad = match rule {
            Rule::Genes => c.mad_on_n_genes,
            Rule::Counts => c.mad_on_counts,
            Rule::Mito => c.mad_on_mito,
            Rule::Ribo | Rule::GeneCells => false,
        };
        if mad {
            Mode::Mad
        } else {
            Mode::Value
        }
    }

    /// Whether `rule` can be set by the MAD band.
    pub fn banded(rule: Rule) -> bool {
        matches!(rule, Rule::Genes | Rule::Counts | Rule::Mito)
    }

    /// The edge `rule` holds its metric to now, on the metric's own scale:
    /// the tightest of its band edge and its own value; `None` when it is
    /// off or its band has no spread.
    pub fn edge(&self, rule: Rule) -> Option<f32> {
        self.edges[self.index(rule)]
    }

    /// The band of `rule`'s metric fitted now, for [`CellRules::recount`].
    fn fit_band(&self, rule: Rule, consider: &[bool]) -> Option<RobustBand> {
        let (values, log1p) = match rule {
            Rule::Genes => (&self.metrics.n_genes[..], true),
            Rule::Counts => (&self.metrics.total_counts[..], true),
            Rule::Mito => (self.metrics.mito_frac.as_deref()?, false),
            Rule::Ribo => (self.metrics.ribo_frac.as_deref()?, false),
            Rule::GeneCells => return None,
        };
        robust_band(values, self.cfg.n_mads, log1p, Some(consider)).map(|b| b.raw(log1p))
    }

    /// The edge of `rule` under the current settings, from its fitted band.
    fn fit_edge(&self, rule: Rule, band: Option<RobustBand>) -> Option<f32> {
        let c = &self.cfg;
        let tightest = |parts: [Option<f32>; 3], above: bool| {
            parts
                .into_iter()
                .flatten()
                .reduce(|a, b| if above { a.max(b) } else { a.min(b) })
        };
        match rule {
            Rule::Genes => tightest(
                [
                    (c.min_cell_nnz > 0).then_some(c.min_cell_nnz as f32),
                    c.mad_on_n_genes.then(|| band.map(|b| b.lo)).flatten(),
                    c.auto_cell_cutoff
                        .then(|| {
                            crate::qc::suggest_nnz_cutoff(&self.metrics.n_genes).map(|k| k as f32)
                        })
                        .flatten(),
                ],
                true,
            ),
            Rule::Counts => tightest(
                [
                    (c.min_counts_per_cell > 0.0).then_some(c.min_counts_per_cell),
                    c.mad_on_counts.then(|| band.map(|b| b.lo)).flatten(),
                    None,
                ],
                true,
            ),
            Rule::Mito => tightest(
                [
                    c.mito_max_frac,
                    c.mad_on_mito.then(|| band.map(|b| b.hi)).flatten(),
                    None,
                ],
                false,
            ),
            Rule::Ribo => c.ribo_max_frac,
            Rule::GeneCells => (c.feature_min_cells > 0).then_some(c.feature_min_cells as f32),
        }
    }

    /// Turn `rule` off, or back on as the starting settings had it (by the
    /// MAD band where it has one; a cap where it has none, at the largest
    /// value, keeping everything until it is moved).
    pub fn set_on(&mut self, rule: Rule, on: bool) {
        if on == self.is_on(rule) {
            return;
        }
        if on {
            restore(&mut self.cfg, &self.base, rule);
            if !is_on(&self.cfg, rule) {
                let max = self.values(rule).iter().copied().fold(0.0f32, f32::max);
                match rule {
                    Rule::Genes => self.cfg.mad_on_n_genes = true,
                    Rule::Counts => self.cfg.mad_on_counts = true,
                    Rule::Mito => self.cfg.mad_on_mito = true,
                    Rule::Ribo => self.cfg.ribo_max_frac = Some(max),
                    Rule::GeneCells => self.cfg.feature_min_cells = 1,
                }
            }
        } else {
            turn_off(&mut self.cfg, rule);
        }
        self.recount();
    }

    /// Hold `rule` to `value` of its own, off the MAD band.
    pub fn set_value(&mut self, rule: Rule, value: f32) {
        let c = &mut self.cfg;
        let value = value.max(0.0);
        match rule {
            Rule::Genes => {
                c.mad_on_n_genes = false;
                c.auto_cell_cutoff = false;
                c.min_cell_nnz = value.round() as usize;
            }
            Rule::Counts => {
                c.mad_on_counts = false;
                c.min_counts_per_cell = value;
            }
            Rule::Mito => {
                c.mad_on_mito = false;
                c.mito_max_frac = Some(value.min(1.0));
            }
            Rule::Ribo => c.ribo_max_frac = Some(value.min(1.0)),
            Rule::GeneCells => c.feature_min_cells = value.round() as usize,
        }
        self.recount();
    }

    /// Set the MAD band's width, which every banded rule shares.
    pub fn set_mads(&mut self, n_mads: f32) {
        self.cfg.n_mads = n_mads.clamp(0.5, 50.0);
        self.recount();
    }

    /// Switch `rule` between the MAD band and a value of its own, the value
    /// starting where the edge is now. Back on the band, its own value
    /// returns to the starting one.
    pub fn toggle_mode(&mut self, rule: Rule) {
        if !Self::banded(rule) {
            return;
        }
        match self.mode(rule) {
            Mode::Mad => {
                // From where the edge is; with none (a band with no spread),
                // from a value that keeps everything.
                let edge = self.edge(rule).unwrap_or_else(|| self.keeps_all(rule));
                self.set_value(rule, edge);
            }
            Mode::Value => {
                let (c, b) = (&mut self.cfg, &self.base);
                match rule {
                    Rule::Genes => {
                        c.mad_on_n_genes = true;
                        c.min_cell_nnz = b.min_cell_nnz;
                        c.auto_cell_cutoff = b.auto_cell_cutoff;
                    }
                    Rule::Counts => {
                        c.mad_on_counts = true;
                        c.min_counts_per_cell = b.min_counts_per_cell;
                    }
                    Rule::Mito => {
                        c.mad_on_mito = true;
                        c.mito_max_frac = b.mito_max_frac;
                    }
                    Rule::Ribo | Rule::GeneCells => {}
                }
                self.recount();
            }
        }
    }

    /// A value of `rule`'s own that removes nothing: 0 for a floor, the
    /// largest value for a cap.
    fn keeps_all(&self, rule: Rule) -> f32 {
        match rule.keeps_above() {
            true => 0.0,
            false => self.values(rule).iter().copied().fold(0.0f32, f32::max),
        }
    }

    /// Back to the starting settings.
    pub fn reset(&mut self) {
        self.cfg = self.base.clone();
        self.recount();
    }

    /// The removed cells grouped by the cell rules each fails on its own,
    /// largest group first. A cell removed only as the rules act together
    /// (a band refitted without a floor, say) has an empty set.
    pub fn overlap(&self) -> &[(Vec<Rule>, usize)] {
        &self.groups
    }

    fn group(&self) -> Vec<(Vec<Rule>, usize)> {
        let cell_rules: Vec<(usize, Rule)> = self
            .rules
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, r)| r.on_cells())
            .collect();
        let mut groups: Vec<(Vec<Rule>, usize)> = vec![];
        for c in (0..self.removed_cells.len()).filter(|&c| self.removed_cells[c]) {
            let set: Vec<Rule> = cell_rules
                .iter()
                .filter(|(i, _)| self.counts[*i].fails[c])
                .map(|&(_, r)| r)
                .collect();
            match groups.iter_mut().find(|(s, _)| *s == set) {
                Some(g) => g.1 += 1,
                None => groups.push((set, 1)),
            }
        }
        groups.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.len().cmp(&b.0.len())));
        groups
    }

    /// `qc` flags that decide what these settings do, past `qc`'s defaults.
    /// The mito pattern is always given, an empty one turning the metric off
    /// (`qc` starts from a pattern of its own); under `--qc-auto-cutoff` the
    /// gene rule is always given too, 0 for off, as `qc` puts it at the
    /// trough only when none is given.
    pub fn flags(&self) -> Vec<String> {
        let (c, d) = (&self.cfg, &QcConfig::default());
        let mut f = vec![];
        let mut pair = |k: &str, v: String| {
            f.push(k.to_string());
            f.push(v);
        };
        if (c.mad_on_n_genes || c.mad_on_counts || c.mad_on_mito) && c.n_mads != d.n_mads {
            pair("--qc-mads", fmt_value(c.n_mads));
        }
        if c.min_cell_nnz != d.min_cell_nnz {
            pair("--qc-min-cell-nnz", c.min_cell_nnz.to_string());
        }
        if c.min_counts_per_cell != d.min_counts_per_cell {
            pair("--qc-min-counts", fmt_value(c.min_counts_per_cell));
        }
        match &c.mito_pattern {
            Some(p) => pair("--qc-mito-pattern", p.clone()),
            None => pair("--qc-mito-pattern", String::new()),
        }
        if let Some(v) = c.mito_max_frac {
            pair("--qc-mito-max-frac", fmt_value(v));
        }
        if let Some(p) = &c.ribo_pattern {
            pair("--qc-ribo-pattern", p.clone());
        }
        if let Some(v) = c.ribo_max_frac {
            pair("--qc-ribo-max-frac", fmt_value(v));
        }
        if c.feature_min_cells != d.feature_min_cells || c.auto_cell_cutoff {
            pair("--qc-feature-min-cells", c.feature_min_cells.to_string());
        }
        if !c.mad_on_n_genes {
            f.push("--qc-mad-on-genes=false".into());
        }
        if !c.mad_on_counts {
            f.push("--qc-mad-on-counts=false".into());
        }
        if c.mito_pattern.is_some() && !c.mad_on_mito {
            f.push("--qc-mad-on-mito=false".into());
        }
        if c.auto_cell_cutoff {
            f.push("--qc-auto-cutoff".into());
        }
        f
    }

    /// Decide again under the current settings, what each rule does (the
    /// config with it off and with it alone, decided in parallel), and what
    /// a frame draws from them.
    fn recount(&mut self) {
        self.verdicts = cell_verdicts_quiet(&self.metrics, &self.cfg);
        let all = &self.verdicts;
        let n = all.train_keep.len();
        self.removed_cells = (0..n).map(|c| !all.emitted(c)).collect();
        self.removed_genes = all.feature_keep.iter().map(|&k| !k).collect();
        let (metrics, cfg, rules) = (&self.metrics, &self.cfg, &self.rules);
        let (removed_cells, removed_genes) = (&self.removed_cells, &self.removed_genes);
        self.counts = rules
            .par_iter()
            .map(|&rule| {
                let mut without = cfg.clone();
                turn_off(&mut without, rule);
                let mut alone = cfg.clone();
                for &other in rules.iter().filter(|&&o| o != rule) {
                    turn_off(&mut alone, other);
                }
                let (w, a) = rayon::join(
                    || cell_verdicts_quiet(metrics, &without),
                    || cell_verdicts_quiet(metrics, &alone),
                );
                let (pass_others, fails, removed): (Vec<bool>, Vec<bool>, &[bool]) =
                    if rule.on_cells() {
                        (
                            (0..n).map(|c| w.emitted(c)).collect(),
                            (0..n).map(|c| is_on(cfg, rule) && !a.emitted(c)).collect(),
                            removed_cells,
                        )
                    } else {
                        (
                            w.feature_keep.clone(),
                            a.feature_keep.iter().map(|&k| !k).collect(),
                            removed_genes,
                        )
                    };
                let only_mask: Vec<bool> = (0..removed.len())
                    .map(|i| removed[i] && pass_others[i])
                    .collect();
                RuleCount {
                    removes: fails.iter().filter(|&&f| f).count(),
                    only: only_mask.iter().filter(|&&o| o).count(),
                    pass_others,
                    fails,
                    only_mask,
                }
            })
            .collect();
        // Bands over the cells above the near-empty floor, as qc_lib fits them.
        let consider: Vec<bool> = self
            .metrics
            .n_genes
            .iter()
            .map(|&g| (g as usize) >= self.cfg.min_cell_nnz)
            .collect();
        self.bands = self
            .rules
            .iter()
            .map(|&r| self.fit_band(r, &consider))
            .collect();
        self.edges = (0..self.rules.len())
            .map(|i| self.fit_edge(self.rules[i], self.bands[i]))
            .collect();
        self.groups = self.group();
    }
}

fn is_on(c: &QcConfig, rule: Rule) -> bool {
    match rule {
        Rule::Genes => c.mad_on_n_genes || c.min_cell_nnz > 0 || c.auto_cell_cutoff,
        Rule::Counts => c.mad_on_counts || c.min_counts_per_cell > 0.0,
        Rule::Mito => c.mad_on_mito || c.mito_max_frac.is_some(),
        Rule::Ribo => c.ribo_max_frac.is_some(),
        Rule::GeneCells => c.feature_min_cells > 0,
    }
}

fn turn_off(c: &mut QcConfig, rule: Rule) {
    match rule {
        Rule::Genes => {
            c.mad_on_n_genes = false;
            c.min_cell_nnz = 0;
            c.auto_cell_cutoff = false;
        }
        Rule::Counts => {
            c.mad_on_counts = false;
            c.min_counts_per_cell = 0.0;
        }
        Rule::Mito => {
            c.mad_on_mito = false;
            c.mito_max_frac = None;
        }
        Rule::Ribo => c.ribo_max_frac = None,
        Rule::GeneCells => c.feature_min_cells = 0,
    }
}

fn restore(c: &mut QcConfig, b: &QcConfig, rule: Rule) {
    match rule {
        Rule::Genes => {
            c.mad_on_n_genes = b.mad_on_n_genes;
            c.min_cell_nnz = b.min_cell_nnz;
            c.auto_cell_cutoff = b.auto_cell_cutoff;
        }
        Rule::Counts => {
            c.mad_on_counts = b.mad_on_counts;
            c.min_counts_per_cell = b.min_counts_per_cell;
        }
        Rule::Mito => {
            c.mad_on_mito = b.mad_on_mito;
            c.mito_max_frac = b.mito_max_frac;
        }
        Rule::Ribo => c.ribo_max_frac = b.ribo_max_frac,
        Rule::GeneCells => c.feature_min_cells = b.feature_min_cells,
    }
}

/// A flag's value: as short as reads back to the same `f32`.
fn fmt_value(v: f32) -> String {
    format!("{v}")
}

#[cfg(test)]
#[path = "qc_rules/tests.rs"]
mod tests;
