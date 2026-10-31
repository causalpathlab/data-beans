//! The cell QC figure: for each rule, the histogram of its metric in three
//! layers (every cell, the cells passing the other rules, the cells only it
//! removes) with its edge; total counts against detected genes, coloured
//! by the mito fraction with removed cells ringed; cells per gene with the
//! gene rule; and the removed cells by the rules they fail. The view draws
//! the same panels ([`panel`]) as images.

use super::*;
use crate::qc_rules::{CellRules, Mode, Rule};

/// One panel of the figure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Panel {
    Rule(Rule),
    Joint,
    Overlap,
}

/// The panels the figure has for `rules`, in order.
pub fn panels(rules: &CellRules) -> Vec<Panel> {
    let mut p: Vec<Panel> = rules
        .rules()
        .iter()
        .filter(|r| r.on_cells())
        .map(|&r| Panel::Rule(r))
        .collect();
    p.push(Panel::Joint);
    if rules.rules().contains(&Rule::GeneCells) {
        p.push(Panel::Rule(Rule::GeneCells));
    }
    p.push(Panel::Overlap);
    p
}

/// An edge as `qc` prints it: counts to a tenth, fractions to four places.
pub fn shown(rule: Rule, v: f32) -> String {
    if rule.count() {
        format!("{v:.1}")
    } else {
        format!("{v:.4}")
    }
}

/// `≥ 92.9 (5 MADs)`, or `off`.
pub fn edge_text(rules: &CellRules, rule: Rule) -> String {
    let Some(edge) = rules.edge(rule) else {
        return if rules.is_on(rule) {
            "no spread".into()
        } else {
            "off".into()
        };
    };
    let side = if rule.keeps_above() { "≥" } else { "≤" };
    let how = match rules.mode(rule) {
        Mode::Mad => format!(" ({} MADs)", short(rules.config().n_mads as f64)),
        Mode::Value => String::new(),
    };
    format!("{side} {}{how}", shown(rule, edge))
}

fn rule_panel(
    rules: &CellRules,
    rule: Rule,
    cursor: Option<f64>,
    c: &mut Canvas,
    (x, y, w, h): (f64, f64, f64, f64),
) {
    let count = rules.count(rule);
    let values = rules.values(rule);
    let mut axis = Axis::fit(values, rule.count());
    if let Some(e) = rules.edge(rule) {
        axis = axis.including(e as f64);
    }
    let what = if rule.on_cells() { "" } else { " genes" };
    let subtitle = match rules.is_on(rule) {
        true => format!(
            "{} · removes {}{what}, {} only by this",
            edge_text(rules, rule),
            count.removes,
            count.only
        ),
        false => "off".into(),
    };
    let (title, x_title) = match rule {
        Rule::Genes => ("Detected genes", "genes per cell"),
        Rule::Counts => ("Total counts", "counts per cell"),
        Rule::Mito => ("Mitochondrial fraction", "fraction of counts"),
        Rule::Ribo => ("Ribosomal fraction", "fraction of counts"),
        Rule::GeneCells => ("Cells detecting each gene", "cells per gene"),
    };
    Hist {
        title: title.into(),
        subtitle,
        x_title: x_title.into(),
        axis,
        values,
        layers: vec![
            (None, FAINT),
            (Some(&count.pass_others), BAR),
            (Some(&count.only_mask), ACCENT),
        ],
        edge: rules.edge(rule).map(|e| (e as f64, shown(rule, e))),
        cursor,
    }
    .draw(c, x, y, w, h);
}

fn joint_panel(rules: &CellRules, c: &mut Canvas, (x, y, w, h): (f64, f64, f64, f64)) {
    let m = rules.metrics();
    let v = rules.verdicts();
    let mito = m.mito_frac.as_deref();
    // Darkest at twice the mito edge, or at the largest fraction.
    let top = rules
        .edge(Rule::Mito)
        .map(|e| 2.0 * e as f64)
        .or_else(|| mito.map(|f| f.iter().copied().fold(0.0f32, f32::max) as f64))
        .filter(|t| *t > 0.0)
        .unwrap_or(1.0);
    let points: Vec<Point> = (0..m.n_genes.len())
        .map(|i| Point {
            x: m.total_counts[i],
            y: m.n_genes[i],
            fill: mito.map_or(BAR, |f| ramp(f[i] as f64 / top)),
            ringed: !v.emitted(i),
        })
        .collect();
    let subtitle = match mito {
        Some(_) => "colour: mito fraction, darker higher · ring: removed",
        None => "ring: removed",
    };
    let mut x_axis = Axis::fit(&m.total_counts, true);
    let mut y_axis = Axis::fit(&m.n_genes, true);
    if let Some(e) = rules.edge(Rule::Counts) {
        x_axis = x_axis.including(e as f64);
    }
    if let Some(e) = rules.edge(Rule::Genes) {
        y_axis = y_axis.including(e as f64);
    }
    Scatter {
        title: "Counts against genes".into(),
        subtitle: subtitle.into(),
        x_title: "counts per cell".into(),
        y_title: "genes per cell".into(),
        x: x_axis,
        y: y_axis,
        points: &points,
        x_edge: rules.edge(Rule::Counts).map(|e| e as f64),
        y_edge: rules.edge(Rule::Genes).map(|e| e as f64),
        key: vec![],
    }
    .draw(c, x, y, w, h);
}

fn overlap_panel(rules: &CellRules, c: &mut Canvas, (x, y, w, _h): (f64, f64, f64, f64)) {
    let groups = rules.overlap();
    let removed: usize = groups.iter().map(|g| g.1).sum();
    c.bold(
        x + 44.0,
        y + 12.0,
        "Why cells were removed",
        11.0,
        Anchor::Start,
        INK,
    );
    c.text(
        x + 44.0,
        y + 25.0,
        &format!("{removed} cells, by the rules each fails on its own"),
        9.0,
        Anchor::Start,
        MUTED,
    );
    let cell_rules: Vec<Rule> = rules
        .rules()
        .iter()
        .copied()
        .filter(|r| r.on_cells())
        .collect();
    let (dx, x0) = (16.0, x + 50.0);
    for (j, r) in cell_rules.iter().enumerate() {
        c.turned(
            x0 + j as f64 * dx + 3.0,
            y + 58.0,
            r.short(),
            8.0,
            -50.0,
            Anchor::Start,
            MUTED,
        );
    }
    let bars_x = x0 + cell_rules.len() as f64 * dx + 8.0;
    let room = (x + w - 40.0 - bars_x).max(10.0);
    let most = groups.first().map_or(1, |g| g.1).max(1) as f64;
    for (i, (set, n)) in groups.iter().take(8).enumerate() {
        let gy = y + 74.0 + i as f64 * 18.0;
        let on: Vec<usize> = (0..cell_rules.len())
            .filter(|&j| set.contains(&cell_rules[j]))
            .collect();
        if let (Some(&a), Some(&b)) = (on.first(), on.last()) {
            c.line(x0 + a as f64 * dx, gy, x0 + b as f64 * dx, gy, INK, 1.5);
        }
        for j in 0..cell_rules.len() {
            let colour = if on.contains(&j) { INK } else { GRID };
            c.dot(x0 + j as f64 * dx, gy, 3.6, colour, 1.0);
        }
        let bw = *n as f64 / most * room;
        c.rect(bars_x, gy - 5.0, bw, 10.0, ACCENT);
        c.text(
            bars_x + bw + 4.0,
            gy + 3.5,
            &n.to_string(),
            9.0,
            Anchor::Start,
            INK,
        );
    }
    if groups.is_empty() {
        c.text(
            x0,
            y + 74.0,
            "no cell is removed",
            9.0,
            Anchor::Start,
            MUTED,
        );
    } else if groups.len() > 8 {
        let rest: usize = groups[8..].iter().map(|g| g.1).sum();
        c.text(
            x0,
            y + 74.0 + 8.0 * 18.0,
            &format!("and {rest} more in {} other sets", groups.len() - 8),
            8.5,
            Anchor::Start,
            MUTED,
        );
    }
}

/// Draw `panel` into the box `(x, y, w, h)` of `c`; `cursor` marks a value
/// under a rule's histogram.
pub fn draw_panel(
    rules: &CellRules,
    panel: Panel,
    cursor: Option<f64>,
    c: &mut Canvas,
    area: (f64, f64, f64, f64),
) {
    match panel {
        Panel::Rule(rule) => rule_panel(rules, rule, cursor, c, area),
        Panel::Joint => joint_panel(rules, c, area),
        Panel::Overlap => overlap_panel(rules, c, area),
    }
}

/// `panel` alone as SVG, `w` x `h` points: what a view shows as an image.
pub fn panel(rules: &CellRules, panel: Panel, cursor: Option<f64>, w: f64, h: f64) -> String {
    let mut c = Canvas::new(w, h);
    draw_panel(rules, panel, cursor, &mut c, (0.0, 4.0, w, h - 4.0));
    c.finish()
}

/// The whole figure, its title naming the data.
pub fn figure(rules: &CellRules, title: &str) -> String {
    let panels = panels(rules);
    let (cols, pw, ph) = (3usize, 320.0, 230.0);
    let rows = panels.len().div_ceil(cols);
    let (left, top) = (16.0, 92.0);
    let (w, h) = (left * 2.0 + cols as f64 * pw, top + rows as f64 * ph + 12.0);
    let mut c = Canvas::new(w, h);
    let v = rules.verdicts();
    let cells = v.train_keep.len();
    let kept = rules.kept_cells().len();
    let genes = v.feature_keep.len();
    let kept_genes = rules.kept_genes().len();
    c.bold(
        left + 8.0,
        28.0,
        &format!("{title} · cell QC"),
        15.0,
        Anchor::Start,
        INK,
    );
    let mut sub = format!(
        "{cells} cells → {kept} kept ({:.1}%) · {genes} genes → {kept_genes} kept",
        100.0 * kept as f64 / cells.max(1) as f64
    );
    if let Some(p) = &rules.config().mito_pattern {
        sub.push_str(&format!(" · mito '{p}'"));
    }
    c.text(left + 8.0, 46.0, &sub, 10.0, Anchor::Start, MUTED);
    c.legend(
        left + 8.0,
        70.0,
        &[
            (FAINT, "all"),
            (BAR, "pass the other rules"),
            (ACCENT, "removed only by this rule"),
        ],
    );
    for (k, &p) in panels.iter().enumerate() {
        let (col, row) = (k % cols, k / cols);
        let area = (
            left + col as f64 * pw,
            top + row as f64 * ph,
            pw - 12.0,
            ph - 14.0,
        );
        draw_panel(rules, p, None, &mut c, area);
    }
    c.finish()
}
