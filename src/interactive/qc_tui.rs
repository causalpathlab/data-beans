//! `data-beans qc`'s full-screen view: the cell QC rules in a table, each
//! with its edge, how many cells it fails and how many it alone removes;
//! the selected rule's histogram in three layers (every cell, the cells
//! passing the other rules, the cells only it removes); total counts
//! against detected genes; and the removed cells by the rules they fail.
//!
//! The view moves a [`CellRules`], so every count is `qc_lib`'s own
//! decision and `p` shows `qc` flags that decide the same cells. Where the
//! terminal has an image protocol the plots are the figure's own panels
//! ([`crate::figure::qc::panel`]); `s` saves the whole figure as PDF and
//! PNG.

use ratatui::buffer::Buffer;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};
use ratatui::Frame;
use ratatui_image::picker::Picker;

use super::ui::{
    header, help_line, input_line, panel, title_bar, Screen, Tui, ACCENT, DIM, HIGHLIGHT, PLAIN,
};
use crate::figure::qc::{edge_text, panel as panel_svg, Panel};
use crate::figure::term::PlotImage;
use crate::figure::{free_name, save, taken, Axis};
use crate::qc_rules::{CellRules, Mode, Rule};

/// What the view ended with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Enter: go on with these settings.
    Apply,
    /// q, Esc, Ctrl-C or a `kill`: write nothing.
    Cancel,
}

enum Typing {
    /// A value for the selected rule's edge.
    Edge(String),
    /// The name to save the figure under.
    Save(String),
    /// The output to write the kept cells to.
    Output(String),
}

/// The view over `rules`.
pub struct QcView {
    rules: CellRules,
    /// The data's name, for the title and the figure.
    name: String,
    /// Where Enter writes the kept cells; asked for when none was given.
    output: Option<String>,
    /// The output name offered when one is asked for.
    suggest: String,
    focus: usize,
    typing: Option<Typing>,
    decision: Option<Decision>,
    /// A line for the footer until the next key: a save, the flags.
    note: Option<Line<'static>>,
    picker: Option<Picker>,
    picked_at: Option<u64>,
    images: bool,
    detecting: bool,
    plots: [PlotImage; 2],
}

/// Colours of the text histogram's layers, back to front.
const LAYERS: [Color; 3] = [Color::DarkGray, Color::Gray, ACCENT];

impl QcView {
    /// The view of `rules` for data `name`. Enter writes the kept cells to
    /// `output`, or, without one, asks where (offering a free name after
    /// `suggest`, say `pbmc.qc`).
    pub fn new(rules: CellRules, name: &str, output: Option<String>, suggest: &str) -> Self {
        Self {
            rules,
            name: name.into(),
            output,
            suggest: suggest.into(),
            focus: 0,
            typing: None,
            decision: None,
            note: None,
            picker: None,
            picked_at: None,
            images: true,
            detecting: false,
            plots: Default::default(),
        }
    }

    /// Ask the attached terminal for images, as each attach brings one.
    pub fn detect_images(mut self) -> Self {
        self.detecting = true;
        self
    }

    pub fn rules(&self) -> &CellRules {
        &self.rules
    }

    pub fn into_rules(self) -> CellRules {
        self.rules
    }

    pub fn decision(&self) -> Option<Decision> {
        self.decision
    }

    /// Where the kept cells go, once Enter has settled it.
    pub fn output(&self) -> Option<&str> {
        self.output.as_deref()
    }

    fn rule(&self) -> Rule {
        self.rules.rules()[self.focus]
    }

    fn changed(&mut self) {
        for p in &mut self.plots {
            p.invalidate();
        }
    }

    /// Ask the terminal attached now for images, once per attach.
    fn refresh_picker(&mut self) {
        if !self.detecting {
            return;
        }
        let attach = legume_tui::attaches();
        if self.picked_at == Some(attach) {
            return;
        }
        self.picked_at = Some(attach);
        self.picker = legume_tui::current().and_then(crate::figure::term::picker);
        self.changed();
    }

    /// Move the selected rule's edge `steps` to the right on its axis.
    fn nudge(&mut self, steps: i32) {
        let rule = self.rule();
        if !self.rules.is_on(rule) {
            self.rules.set_on(rule, true);
            self.changed();
            return;
        }
        match self.rules.mode(rule) {
            Mode::Mad => {
                // A wider band moves a lower edge left and an upper one right.
                let sign = if rule.keeps_above() { -1.0 } else { 1.0 };
                let k = self.rules.config().n_mads + sign * 0.25 * steps as f32;
                self.rules.set_mads(k);
            }
            Mode::Value => {
                let values = self.rules.values(rule);
                let axis = Axis::fit(values, rule.count());
                let edge = self.rules.edge(rule).unwrap_or(axis.lo as f32) as f64;
                let mut next = axis.at(axis.frac(edge) + steps as f64 / 80.0);
                if matches!(rule, Rule::Genes | Rule::GeneCells) {
                    next = next.round();
                    if next == edge {
                        next = edge + steps.signum() as f64;
                    }
                }
                self.rules.set_value(rule, next.clamp(0.0, axis.hi) as f32);
            }
        }
        self.changed();
    }

    fn save_figure(&mut self, name: &str) {
        let svg = crate::figure::qc::figure(&self.rules, &self.name);
        self.note = Some(match save(&svg, name) {
            Ok(paths) => Line::from(Span::styled(format!(" saved {}", paths.join(", ")), DIM)),
            Err(e) => Line::from(Span::styled(format!(" save failed: {e}"), HIGHLIGHT)),
        });
    }

    /// The flags that decide what the view shows, as shell words; empty for
    /// the defaults.
    fn flags_text(&self) -> String {
        self.rules
            .flags()
            .iter()
            .map(|f| shell_word(f))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn flags_line(&self) -> Line<'static> {
        let text = match self.flags_text() {
            t if t.is_empty() => "the defaults: no flags needed".to_string(),
            t => t,
        };
        Line::from(vec![
            Span::styled(" flags: ", DIM),
            Span::styled(text, PLAIN),
        ])
    }

    fn handle_typing(&mut self, key: KeyEvent) {
        self.note = None;
        let Some(typing) = self.typing.as_mut() else {
            return;
        };
        let buf = match typing {
            Typing::Edge(b) | Typing::Save(b) | Typing::Output(b) => b,
        };
        match key.code {
            KeyCode::Esc => self.typing = None,
            KeyCode::Backspace => {
                buf.pop();
            }
            KeyCode::Enter => match self.typing.take() {
                Some(Typing::Edge(text)) => match text.trim().parse::<f32>() {
                    Ok(v) => {
                        let rule = self.rule();
                        self.rules.set_value(rule, v);
                        self.changed();
                    }
                    Err(_) => {
                        self.note = Some(Line::from(Span::styled(
                            format!(" not a number: {text}"),
                            HIGHLIGHT,
                        )))
                    }
                },
                Some(Typing::Save(name)) if !name.trim().is_empty() => {
                    let name = name.trim();
                    if taken(name) {
                        // Never over a file: the next free name, offered.
                        self.typing = Some(Typing::Save(free_name(name)));
                        self.note = Some(Line::from(Span::styled(
                            format!(" {name} is already there; not replacing it"),
                            HIGHLIGHT,
                        )));
                    } else {
                        self.save_figure(name)
                    }
                }
                Some(Typing::Output(name)) => {
                    let name = name.trim();
                    if name.is_empty() {
                        self.note = Some(Line::from(Span::styled(
                            " no output named: Enter again to name one, q to leave",
                            DIM,
                        )));
                    } else if output_taken(name) {
                        self.typing = Some(Typing::Output(free_output(name)));
                        self.note = Some(Line::from(Span::styled(
                            format!(" {name} is already there; not replacing it"),
                            HIGHLIGHT,
                        )));
                    } else {
                        self.output = Some(name.to_string());
                        self.decision = Some(Decision::Apply);
                    }
                }
                _ => {}
            },
            KeyCode::Char(c) if buf.len() < 512 => buf.push(c),
            _ => {}
        }
    }

    fn render_rules(&self, frame: &mut Frame, area: Rect) {
        let block = panel(" cell QC rules ".into(), true);
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let mut lines = vec![Line::from(Span::styled(
            format!(
                "  {:<16}{:>22}{:>9}{:>6}",
                "rule", "edge", "removes", "only"
            ),
            DIM,
        ))];
        for (i, &rule) in self.rules.rules().iter().enumerate() {
            if rule == Rule::GeneCells {
                lines.push(Line::from(Span::styled("  genes", DIM)));
            }
            let sel = i == self.focus;
            let on = self.rules.is_on(rule);
            let c = self.rules.count(rule);
            let style = if sel {
                HIGHLIGHT
            } else if on {
                PLAIN
            } else {
                DIM
            };
            let num = |n: usize, w: usize| {
                Span::styled(
                    format!("{n:>w$}"),
                    if n > 0 { Style::new().fg(ACCENT) } else { DIM },
                )
            };
            lines.push(Line::from(vec![
                Span::styled(if sel { "▸ " } else { "  " }, HIGHLIGHT),
                Span::styled(format!("{:<16}", rule.label()), style),
                Span::styled(format!("{:>22}", edge_text(&self.rules, rule)), style),
                num(c.removes, 9),
                num(c.only, 6),
            ]));
        }
        let v = self.rules.verdicts();
        let (cells, kept) = (v.train_keep.len(), self.rules.kept_cells().len());
        let (genes, kept_genes) = (v.feature_keep.len(), self.rules.kept_genes().len());
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::raw("  kept "),
            Span::styled(kept.to_string(), HIGHLIGHT),
            Span::raw(format!(
                " / {cells} cells ({:.1}%)",
                100.0 * kept as f64 / cells.max(1) as f64
            )),
        ]));
        lines.push(Line::from(format!("       {kept_genes} / {genes} genes")));
        if v.guarded {
            lines.push(Line::from(Span::styled(
                "  every cell (or gene) fails: QC keeps them all instead",
                HIGHLIGHT,
            )));
        }
        // Held to a value, the genes rule is the near-empty floor, which
        // the other bands are fitted above.
        if self.rules.mode(Rule::Genes) == Mode::Value && self.rules.is_on(Rule::Genes) {
            lines.push(Line::from(Span::styled(
                "  the bands are fitted on the cells above the genes floor",
                DIM,
            )));
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  removes: fails it on its own; only: no other rule removes",
            DIM,
        )));
        frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
    }

    fn render_overlap(&self, frame: &mut Frame, area: Rect) {
        let block = panel(" removed cells, by the rules they fail ".into(), false);
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let groups = self.rules.overlap();
        let most = groups.first().map_or(1, |g| g.1).max(1);
        let label_w = 26usize;
        let room = (inner.width as usize).saturating_sub(label_w + 9).max(4);
        let mut lines = vec![];
        for (set, n) in groups.iter().take(inner.height as usize) {
            let name = if set.is_empty() {
                "(the rules together)".to_string()
            } else {
                set.iter()
                    .map(|r| r.short())
                    .collect::<Vec<_>>()
                    .join(" + ")
            };
            let eighths = (n * room * 8).div_ceil(most);
            let bar =
                "█".repeat(eighths / 8) + ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"][eighths % 8];
            lines.push(Line::from(vec![
                Span::raw(format!("  {name:<label_w$}")),
                Span::raw(format!("{n:>6} ")),
                Span::styled(bar, Style::new().fg(ACCENT)),
            ]));
        }
        if groups.is_empty() {
            lines.push(Line::from(Span::styled("  no cell is removed", DIM)));
        }
        frame.render_widget(Paragraph::new(lines), inner);
    }

    fn render_focus(&mut self, frame: &mut Frame, area: Rect) {
        let rule = self.rule();
        let title = format!(
            " {} · {} per bin ",
            rule.label(),
            if rule.on_cells() { "cells" } else { "genes" }
        );
        let block = panel(title, true);
        let inner = block.inner(area);
        frame.render_widget(block, area);
        if let Some(picker) = self.picker.as_ref().filter(|_| self.images) {
            let rules = &self.rules;
            if self.plots[0].render(frame, inner, picker, |w, h| {
                panel_svg(rules, Panel::Rule(rule), None, w, h)
            }) {
                return;
            }
        }
        let [legend, plot] =
            Layout::vertical([Constraint::Length(1), Constraint::Fill(1)]).areas(inner);
        let count = self.rules.count(rule);
        let values = self.rules.values(rule);
        let mut axis = Axis::fit(values, rule.count());
        let edge = self.rules.edge(rule);
        if let Some(e) = edge {
            axis = axis.including(e as f64);
        }
        layered_text(
            frame.buffer_mut(),
            plot,
            values,
            &axis,
            &[None, Some(&count.pass_others), Some(&count.only_mask)],
            edge.map(f64::from),
        );
        let pass = count.pass_others.iter().filter(|&&p| p).count();
        frame.render_widget(
            Line::from(vec![
                Span::styled(" █", Style::new().fg(LAYERS[0])),
                Span::raw(format!(" all {}  ", values.len())),
                Span::styled("█", Style::new().fg(LAYERS[1])),
                Span::raw(format!(" pass the other rules {pass}  ")),
                Span::styled("█", Style::new().fg(LAYERS[2])),
                Span::raw(format!(" removed only by this {}", count.only)),
            ]),
            legend,
        );
    }

    fn render_joint(&mut self, frame: &mut Frame, area: Rect) {
        let block = panel(" total counts × detected genes · log ".into(), false);
        let inner = block.inner(area);
        frame.render_widget(block, area);
        if let Some(picker) = self.picker.as_ref().filter(|_| self.images) {
            let rules = &self.rules;
            if self.plots[1].render(frame, inner, picker, |w, h| {
                panel_svg(rules, Panel::Joint, None, w, h)
            }) {
                return;
            }
        }
        braille_scatter(frame.buffer_mut(), inner, &self.rules);
    }
}

/// `s` as one shell word.
pub fn shell_word(s: &str) -> String {
    if !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_=./,:+".contains(c))
    {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', r"'\''"))
    }
}

const EIGHTHS: [&str; 9] = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];

/// A histogram of `values` as text, layers (masks; `None` for all) drawn
/// back to front in [`LAYERS`], two columns a bin, the y axis in a 6-wide
/// gutter, x ticks below, and the edge as a dotted rule with ▲.
fn layered_text(
    buf: &mut Buffer,
    area: Rect,
    values: &[f32],
    axis: &Axis,
    layers: &[Option<&[bool]>],
    edge: Option<f64>,
) {
    if area.height < 4 || area.width < 12 {
        return;
    }
    let gutter = 6u16;
    let (cx, cw) = (area.x + gutter + 1, area.width - gutter - 1);
    let ph = area.height - 2;
    let n = (cw / 2).max(1) as usize;
    let counts: Vec<Vec<usize>> = layers
        .iter()
        .map(|m| axis.counts(values, n, |i| m.is_none_or(|m| m[i])))
        .collect();
    let top = counts.iter().flatten().copied().max().unwrap_or(0).max(1);
    let style = |fg: Color, bg: Option<Color>| {
        let s = Style::reset().fg(fg);
        match bg {
            Some(b) => s.bg(b),
            None => s,
        }
    };
    for b in 0..n {
        let hs: Vec<usize> = counts
            .iter()
            .map(|c| (c[b] * ph as usize * 8).div_ceil(top))
            .collect();
        for k in 0..ph as usize {
            let at = k * 8;
            let Some(front) = (0..hs.len()).rev().find(|&l| hs[l] > at) else {
                continue;
            };
            let frac = (hs[front] - at).min(8);
            let back = (0..front)
                .rev()
                .find(|&l| hs[l] > at + frac)
                .map(|l| LAYERS[l]);
            let y = area.y + ph - 1 - k as u16;
            for dx in 0..2 {
                let x = cx + 2 * b as u16 + dx;
                if x < area.right() {
                    buf[(x, y)]
                        .set_symbol(EIGHTHS[frac])
                        .set_style(style(LAYERS[front], back));
                }
            }
        }
    }
    let dim = Style::reset().add_modifier(ratatui::style::Modifier::DIM);
    for y in area.y..area.y + ph {
        buf[(area.x + gutter, y)].set_symbol("│").set_style(dim);
    }
    buf.set_string(area.x, area.y, format!("{top:>5}"), dim);
    buf.set_string(area.x, area.y + ph - 1, format!("{:>5}", 0), dim);
    let axis_y = area.y + ph;
    buf.set_string(
        area.x + gutter,
        axis_y,
        format!("└{}", "─".repeat(cw as usize)),
        dim,
    );
    for t in axis.ticks() {
        let label = crate::figure::short(t);
        let x = cx + (axis.frac(t) * (cw.saturating_sub(1)) as f64).round() as u16;
        let x = x
            .saturating_sub(label.len() as u16 / 2)
            .max(cx)
            .min(area.right().saturating_sub(label.len() as u16));
        buf.set_string(x, axis_y + 1, &label, dim);
    }
    if let Some(e) = edge {
        let x = cx + (axis.frac(e).clamp(0.0, 1.0) * (cw.saturating_sub(1)) as f64).round() as u16;
        for y in area.y..area.y + ph {
            if buf[(x, y)].symbol() == " " {
                buf[(x, y)]
                    .set_symbol("┊")
                    .set_style(Style::reset().fg(ACCENT));
            }
        }
        buf[(x, axis_y)]
            .set_symbol("▲")
            .set_style(Style::reset().fg(ACCENT));
    }
}

/// Total counts against detected genes in braille dots, log on both;
/// a cell holding any removed cell in the accent, both edges dotted.
fn braille_scatter(buf: &mut Buffer, area: Rect, rules: &CellRules) {
    if area.height < 4 || area.width < 16 {
        return;
    }
    let m = rules.metrics();
    let v = rules.verdicts();
    let gutter = 6u16;
    let (sx, sw, sh) = (
        area.x + gutter + 1,
        area.width - gutter - 1,
        area.height - 2,
    );
    let mut xa = Axis::fit(&m.total_counts, true);
    let mut ya = Axis::fit(&m.n_genes, true);
    let (ce, ge) = (rules.edge(Rule::Counts), rules.edge(Rule::Genes));
    if let Some(e) = ce {
        xa = xa.including(e as f64);
    }
    if let Some(e) = ge {
        ya = ya.including(e as f64);
    }
    let accent = Style::reset().fg(ACCENT);
    let dim = Style::reset().add_modifier(ratatui::style::Modifier::DIM);
    // Edges first; dots drawn over them.
    if let Some(e) = ce {
        let x = sx + (xa.frac(e as f64) * (sw - 1) as f64).round() as u16;
        for y in area.y..area.y + sh {
            buf[(x, y)].set_symbol("┊").set_style(accent);
        }
    }
    if let Some(e) = ge {
        let y = area.y + sh - 1 - (ya.frac(e as f64) * (sh - 1) as f64).round() as u16;
        for x in sx..sx + sw {
            buf[(x, y)].set_symbol("┄").set_style(accent);
        }
    }
    let (w, h) = (sw as usize, sh as usize);
    let mut dots = vec![(0u8, false); w * h];
    for c in 0..m.n_genes.len() {
        let (fx, fy) = (
            xa.frac(m.total_counts[c] as f64),
            ya.frac(m.n_genes[c] as f64),
        );
        if !(fx.is_finite() && fy.is_finite()) {
            continue;
        }
        let px = ((fx * (2 * w - 1) as f64).round() as usize).min(2 * w - 1);
        let py = ((fy * (4 * h - 1) as f64).round() as usize).min(4 * h - 1);
        let (col, row) = (px / 2, h - 1 - py / 4);
        let bit = [[0x40u8, 0x80], [0x04, 0x20], [0x02, 0x10], [0x01, 0x08]][py % 4][px % 2];
        let d = &mut dots[row * w + col];
        d.0 |= bit;
        d.1 |= !v.emitted(c);
    }
    for (k, &(bits, gone)) in dots.iter().enumerate() {
        if bits == 0 {
            continue;
        }
        let (x, y) = (sx + (k % w) as u16, area.y + (k / w) as u16);
        let ch = char::from_u32(0x2800 + bits as u32).unwrap_or('⣿');
        buf[(x, y)].set_symbol(&ch.to_string()).set_style(if gone {
            accent
        } else {
            Style::reset()
        });
    }
    for y in area.y..area.y + sh {
        buf[(area.x + gutter, y)].set_symbol("│").set_style(dim);
    }
    buf.set_string(
        area.x,
        area.y,
        format!("{:>5}", crate::figure::short(ya.hi)),
        dim,
    );
    buf.set_string(
        area.x,
        area.y + sh - 1,
        format!("{:>5}", crate::figure::short(ya.lo)),
        dim,
    );
    buf.set_string(
        area.x + gutter,
        area.y + sh,
        format!("└{}", "─".repeat(sw as usize)),
        dim,
    );
    for t in xa.ticks() {
        let label = crate::figure::short(t);
        let x = sx + (xa.frac(t) * (sw - 1) as f64).round() as u16;
        let x = x
            .saturating_sub(label.len() as u16 / 2)
            .max(sx)
            .min(area.right().saturating_sub(label.len() as u16));
        buf.set_string(x, area.y + sh + 1, &label, dim);
    }
    // Top left is empty: no cell detects more genes than it counts.
    let key = "⣿ kept  ";
    buf.set_string(sx + 1, area.y, key, Style::reset());
    buf.set_string(
        sx + 1 + key.chars().count() as u16,
        area.y,
        "⣿ removed",
        accent,
    );
}

impl Screen for QcView {
    fn render(&mut self, frame: &mut Frame) {
        self.refresh_picker();
        let [top, body, footer] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Fill(1),
            Constraint::Length(1),
        ])
        .areas(frame.area());
        let v = self.rules.verdicts();
        let extra = format!(
            "{} cells × {} genes{}",
            v.train_keep.len(),
            v.feature_keep.len(),
            match (&self.picker, self.images) {
                (None, _) => "",
                (Some(_), true) => " · image",
                (Some(_), false) => " · text",
            }
        );
        title_bar(
            frame,
            top,
            header("qc", &self.name, &extra),
            &super::session_keys(),
        );
        let left_w = (body.width * 2 / 5).clamp(58, 70).min(body.width / 2);
        let [left, right] =
            Layout::horizontal([Constraint::Length(left_w), Constraint::Fill(1)]).areas(body);
        let rows = self.rules.rules().len() as u16 + 12;
        let [rules_area, overlap] =
            Layout::vertical([Constraint::Length(rows), Constraint::Fill(1)]).areas(left);
        let [focus, joint] =
            Layout::vertical([Constraint::Percentage(50), Constraint::Fill(1)]).areas(right);
        self.render_rules(frame, rules_area);
        self.render_overlap(frame, overlap);
        self.render_focus(frame, focus);
        self.render_joint(frame, joint);
        let line = match &self.typing {
            Some(Typing::Edge(b)) => input_line(
                &format!("{} edge: ", self.rule().label()),
                b,
                &[("Enter", "set"), ("Esc", "back")],
            ),
            Some(Typing::Save(b)) => match &self.note {
                Some(note) => note.clone(),
                None => input_line(
                    "save the figure as (.pdf + .png): ",
                    b,
                    &[("Enter", "save"), ("Esc", "back")],
                ),
            },
            Some(Typing::Output(b)) => match &self.note {
                Some(note) => note.clone(),
                None => input_line(
                    "write the kept cells to: ",
                    b,
                    &[("Enter", "write"), ("Esc", "back")],
                ),
            },
            None => match &self.note {
                Some(note) => note.clone(),
                None => {
                    let mut keys = vec![
                        ("↑/↓", "rule"),
                        ("←/→", "move edge"),
                        ("m", "MAD ↔ value"),
                        ("e", "type edge"),
                        ("o", "on/off"),
                        ("r", "reset"),
                        ("s", "save figure"),
                        ("p", "flags"),
                        ("c", "copy flags"),
                    ];
                    if self.picker.is_some() {
                        keys.push(("i", "image/text"));
                    }
                    let enter = match &self.output {
                        Some(out) => format!("write {}", stem(out)),
                        None => "write…".to_string(),
                    };
                    keys.push(("Enter", enter.as_str()));
                    keys.push(("q", "quit"));
                    help_line(&keys)
                }
            },
        };
        frame.render_widget(line, footer);
    }

    fn handle_key(&mut self, key: KeyEvent) {
        if self.typing.is_some() {
            self.handle_typing(key);
            return;
        }
        self.note = None;
        let n = self.rules.rules().len();
        let step = if key.modifiers.contains(KeyModifiers::SHIFT) {
            5
        } else {
            1
        };
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.focus = (self.focus + n - 1) % n;
                self.changed();
            }
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Tab => {
                self.focus = (self.focus + 1) % n;
                self.changed();
            }
            KeyCode::Left | KeyCode::Char('h') => self.nudge(-step),
            KeyCode::Right | KeyCode::Char('l') => self.nudge(step),
            KeyCode::Char('H') => self.nudge(-5),
            KeyCode::Char('L') => self.nudge(5),
            KeyCode::Char('m') => {
                let rule = self.rule();
                if CellRules::banded(rule) {
                    self.rules.toggle_mode(rule);
                    self.changed();
                } else {
                    self.note = Some(Line::from(Span::styled(
                        format!(" {} has no MAD band; it takes a value", rule.label()),
                        DIM,
                    )));
                }
            }
            KeyCode::Char('o') => {
                let rule = self.rule();
                let on = self.rules.is_on(rule);
                self.rules.set_on(rule, !on);
                self.changed();
            }
            KeyCode::Char('e') => {
                let now = self
                    .rules
                    .edge(self.rule())
                    .map(|e| e.to_string())
                    .unwrap_or_default();
                self.typing = Some(Typing::Edge(now));
            }
            KeyCode::Char('r') => {
                self.rules.reset();
                self.changed();
            }
            KeyCode::Char('s') => {
                self.typing = Some(Typing::Save(free_name(&format!("{}_qc", stem(&self.name)))));
            }
            KeyCode::Char('p') => self.note = Some(self.flags_line()),
            KeyCode::Char('c') => {
                let flags = self.flags_text();
                super::ui::copy(&flags);
                let said = match flags.is_empty() {
                    true => " copied nothing: these are the defaults".to_string(),
                    false => format!(" copied to the clipboard: {flags}"),
                };
                self.note = Some(Line::from(Span::styled(said, DIM)));
            }
            KeyCode::Char('i') if self.picker.is_some() => {
                self.images = !self.images;
                self.changed();
            }
            KeyCode::Enter => match self.output {
                Some(_) => self.decision = Some(Decision::Apply),
                None => self.typing = Some(Typing::Output(free_output(&self.suggest))),
            },
            KeyCode::Char('q') | KeyCode::Esc => self.decision = Some(Decision::Cancel),
            _ => {}
        }
    }

    fn interrupt(&mut self) {
        self.decision = Some(Decision::Cancel);
    }

    fn done(&self) -> bool {
        self.decision.is_some()
    }

    fn status(&self) -> Option<String> {
        Some(format!(
            "qc {}: keeps {} of {} cells",
            self.name,
            self.rules.kept_cells().len(),
            self.rules.verdicts().train_keep.len()
        ))
    }
}

/// A backend's name without its extension: `pbmc` of `pbmc.zarr.zip`.
pub fn stem(name: &str) -> &str {
    let base = name.rsplit('/').next().unwrap_or(name);
    [".zarr.zip", ".zarr", ".h5ad", ".h5"]
        .iter()
        .find_map(|e| base.strip_suffix(e))
        .unwrap_or(base)
}

/// Whether an output named `name` would replace something: the name as
/// given, or the backend a write under it makes.
pub fn output_taken(name: &str) -> bool {
    let base = name.trim_end_matches(".zip").trim_end_matches(".zarr");
    [
        name.to_string(),
        format!("{base}.zarr.zip"),
        format!("{base}.zarr"),
        format!("{base}.h5"),
    ]
    .iter()
    .any(|p| std::path::Path::new(p).exists())
}

/// `name`, or failing that the first of `{name}-2`, `{name}-3`, … that
/// [`output_taken`] lets through.
pub fn free_output(name: &str) -> String {
    (1..)
        .map(|k| match k {
            1 => name.to_string(),
            k => format!("{name}-{k}"),
        })
        .find(|n| !output_taken(n))
        .expect("some suffix is free")
}

/// Run the view of `rules` in `tui` until Enter or quit: the rules as left,
/// how it ended, and where the kept cells go (`output`, or the name Enter
/// asked for, offering `suggest`).
pub fn choose_rules(
    tui: &mut Tui<'_>,
    rules: CellRules,
    name: &str,
    output: Option<String>,
    suggest: &str,
) -> anyhow::Result<(CellRules, Decision, Option<String>)> {
    let mut view = QcView::new(rules, name, output, suggest).detect_images();
    tui.run(&mut view)?;
    let decision = view.decision().unwrap_or(Decision::Cancel);
    let output = view.output().map(str::to_string);
    Ok((view.into_rules(), decision, output))
}

#[cfg(test)]
#[path = "tests/qc_tui.rs"]
mod tests;
