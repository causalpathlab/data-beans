//! `data-beans-sim view`: a reference-mode simulation against its
//! reference, from its `{out}.fidelity.json`
//! ([`crate::sim::copula::fidelity`]).
//!
//! Panels, one at a time (`←`/`→`, Tab, or `1`–`7`):
//!
//! 1. a summary table of the reference and simulated statistics;
//! 2. library size and 3. detected genes per cell, as histograms of both,
//!    as shares of cells, the simulation in front of the reference;
//! 4. gene mean, 5. gene variance and 6. zero fraction, one point per gene,
//!    simulated against reference, beside the identity line;
//! 7. the gene-gene correlation of the densest copula genes, one point per
//!    pair.

use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Axis, Cell, Chart, Dataset, GraphType, Paragraph, Row, Table};
use ratatui::Frame;

use super::ui::{
    compact, header, help_line, panel, run_screen, title_bar, Binned, HistPlot, Scale, Screen,
    ACCENTED, DIM, HIGHLIGHT,
};
use crate::sim::copula::fidelity::{FidelityReport, StatKind};

const PANELS: [&str; 7] = [
    "summary",
    "library size",
    "detected genes",
    "gene mean",
    "gene variance",
    "zero fraction",
    "correlation",
];

/// Points drawn per scatter at most; the rest are thinned evenly.
const MAX_POINTS: usize = 20_000;

/// Open the view of the report at `path` until `q`.
pub fn view_fidelity(path: &str) -> anyhow::Result<()> {
    let report = FidelityReport::read(path)?;
    let mut view = FidelityView::new(report, path);
    run_screen(&mut view)
}

/// One row of the summary: a label and the reference and simulated values.
struct SummaryRow {
    label: &'static str,
    reference: String,
    simulated: String,
}

/// A histogram of one per-cell statistic of both sides, as shares of cells.
struct Overlay {
    hist: Binned,
    reference: Vec<f64>,
    simulated: Vec<f64>,
}

impl Overlay {
    fn new(reference: &[f32], simulated: &[f32]) -> Self {
        let mut all: Vec<f32> = reference
            .iter()
            .chain(simulated)
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        all.sort_by(f32::total_cmp);
        let hist = Binned::new(&all, Scale::Log);
        let share = |values: &[f32]| -> Vec<f64> {
            let n = values.len().max(1) as f64;
            hist.count(values.iter().copied())
                .into_iter()
                .map(|c| c as f64 / n)
                .collect()
        };
        Self {
            reference: share(reference),
            simulated: share(simulated),
            hist,
        }
    }
}

/// Points of a scatter, simulated against reference, and the axis range.
struct Scatter {
    points: Vec<(f64, f64)>,
    lo: f64,
    hi: f64,
    /// Axis values are `log10` of the statistic.
    log: bool,
}

impl Scatter {
    fn new(reference: &[f32], simulated: &[f32], log: bool) -> Self {
        let step = (reference.len() / MAX_POINTS).max(1);
        let points: Vec<(f64, f64)> = reference
            .iter()
            .zip(simulated)
            .step_by(step)
            .filter(|(r, s)| !log || (**r > 0.0 && **s > 0.0))
            .map(|(&r, &s)| {
                let f = |v: f32| if log { (v as f64).log10() } else { v as f64 };
                (f(r), f(s))
            })
            .filter(|(r, s)| r.is_finite() && s.is_finite())
            .collect();
        let (lo, hi) = points
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &(r, s)| {
                (lo.min(r).min(s), hi.max(r).max(s))
            });
        let (lo, hi) = if lo.is_finite() && hi > lo {
            (lo, hi)
        } else {
            (0.0, 1.0)
        };
        Self {
            points,
            lo,
            hi,
            log,
        }
    }

    fn label(&self, v: f64) -> String {
        if !self.log {
            return format!("{v:.2}");
        }
        let x = 10f64.powf(v);
        // `compact` rounds values below 0.01 to 0.
        if x < 0.01 {
            format!("{x:.0e}")
        } else {
            compact(x)
        }
    }
}

pub struct FidelityView {
    title: String,
    tab: usize,
    y_scale: Scale,
    quit: bool,
    summary: Vec<SummaryRow>,
    library: Overlay,
    detected: Overlay,
    scatters: [Scatter; 4],
}

impl FidelityView {
    pub fn new(report: FidelityReport, path: &str) -> Self {
        let r = &report;
        let scatters = [
            Scatter::new(
                &r.profiles.reference.moments.mean,
                &r.profiles.simulated.moments.mean,
                true,
            ),
            Scatter::new(
                &r.profiles.reference.moments.var,
                &r.profiles.simulated.moments.var,
                true,
            ),
            Scatter::new(
                &r.profiles.reference.moments.zero,
                &r.profiles.simulated.moments.zero,
                false,
            ),
            Scatter::new(
                &r.profiles.reference.correlation,
                &r.profiles.simulated.correlation,
                false,
            ),
        ];
        let title = if report.output.is_empty() {
            path.to_string()
        } else {
            format!("{} vs {}", report.output, report.reference)
        };
        Self {
            title,
            tab: 0,
            y_scale: Scale::Linear,
            quit: false,
            summary: summarize(r),
            library: Overlay::new(&r.profiles.reference.library, &r.profiles.simulated.library),
            detected: Overlay::new(
                &r.profiles.reference.detected,
                &r.profiles.simulated.detected,
            ),
            scatters,
        }
    }

    fn render_summary(&self, frame: &mut Frame, area: Rect) {
        let rows: Vec<Row> = self
            .summary
            .iter()
            .map(|row| {
                Row::new(vec![
                    Cell::from(Span::styled(row.label, DIM)),
                    Cell::from(row.reference.clone()),
                    Cell::from(Span::styled(row.simulated.clone(), ACCENTED)),
                ])
            })
            .collect();
        let table = Table::new(
            rows,
            [
                Constraint::Length(34),
                Constraint::Length(14),
                Constraint::Length(14),
            ],
        )
        .header(Row::new(vec![
            Cell::from(""),
            Cell::from(Span::styled("reference", HIGHLIGHT)),
            Cell::from(Span::styled("simulated", HIGHLIGHT)),
        ]))
        .block(panel(" summary ".to_string(), true));
        frame.render_widget(table, area);
    }

    fn render_overlay(&self, frame: &mut Frame, area: Rect, overlay: &Overlay, what: &str) {
        let block = panel(format!(" {what} per cell · share of cells "), true);
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let [legend, plot] =
            Layout::vertical([Constraint::Length(1), Constraint::Min(5)]).areas(inner);
        frame.render_widget(
            Line::from(vec![
                Span::styled("█ simulated", ACCENTED),
                Span::raw("   "),
                Span::styled("█ reference (behind)", DIM),
            ]),
            legend,
        );
        let y_max = overlay
            .reference
            .iter()
            .chain(&overlay.simulated)
            .fold(0.0_f64, |m, &v| m.max(v));
        HistPlot {
            bins: overlay.hist.bins,
            kmin: overlay.hist.kmin,
            counts: &overlay.reference,
            style: &|_| ACCENTED,
            subset: Some(&overlay.simulated),
            y_scale: self.y_scale,
            y_max: Some(y_max),
            pointer: None,
            marks: Vec::new(),
            x_label: None,
            tick_every: None,
        }
        .render(frame.buffer_mut(), plot);
    }

    fn render_scatter(&self, frame: &mut Frame, area: Rect, scatter: &Scatter, what: &str) {
        let identity = [(scatter.lo, scatter.lo), (scatter.hi, scatter.hi)];
        let axis = |name: &'static str| {
            let mid = 0.5 * (scatter.lo + scatter.hi);
            Axis::default()
                .title(Span::styled(name, DIM))
                .bounds([scatter.lo, scatter.hi])
                .labels([
                    scatter.label(scatter.lo),
                    scatter.label(mid),
                    scatter.label(scatter.hi),
                ])
                .style(DIM)
        };
        let chart = Chart::new(vec![
            Dataset::default()
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(DIM)
                .data(&identity),
            Dataset::default()
                .marker(Marker::Braille)
                .graph_type(GraphType::Scatter)
                .style(ACCENTED)
                .data(&scatter.points),
        ])
        .block(panel(
            format!(" {what} · simulated against reference "),
            true,
        ))
        .x_axis(axis("reference"))
        .y_axis(axis("simulated"));
        frame.render_widget(chart, area);
    }

    fn help(&self) -> Line<'static> {
        let mut pairs = vec![("←→", "panel"), ("1-7", "jump")];
        if matches!(self.tab, 1 | 2) {
            pairs.push(("y", "y scale"));
        }
        pairs.push(("q", "quit"));
        help_line(&pairs)
    }
}

impl Screen for FidelityView {
    fn render(&mut self, frame: &mut Frame) {
        let [top, tabs, body, footer] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Fill(1),
            Constraint::Length(1),
        ])
        .areas(frame.area());
        title_bar(frame, top, header("sim", &self.title, "fidelity"), &[]);
        let tab_line: Vec<Span> = PANELS
            .iter()
            .enumerate()
            .flat_map(|(i, name)| {
                let style = if i == self.tab { HIGHLIGHT } else { DIM };
                [
                    Span::styled(format!(" {} {name} ", i + 1), style),
                    Span::raw(" "),
                ]
            })
            .collect();
        frame.render_widget(Paragraph::new(Line::from(tab_line)), tabs);
        match self.tab {
            0 => self.render_summary(frame, body),
            1 => self.render_overlay(frame, body, &self.library, "library size"),
            2 => self.render_overlay(frame, body, &self.detected, "detected genes"),
            t => {
                let what = PANELS[t];
                self.render_scatter(frame, body, &self.scatters[t - 3], what);
            }
        }
        frame.render_widget(self.help(), footer);
    }

    fn handle_key(&mut self, key: KeyEvent) {
        let n = PANELS.len();
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => self.quit = true,
            KeyCode::Right | KeyCode::Tab => self.tab = (self.tab + 1) % n,
            KeyCode::Left | KeyCode::BackTab => self.tab = (self.tab + n - 1) % n,
            KeyCode::Char(c @ '1'..='7') => self.tab = c as usize - '1' as usize,
            KeyCode::Char('y') => self.y_scale = self.y_scale.next(),
            _ => {}
        }
    }

    fn interrupt(&mut self) {
        self.quit = true;
    }

    fn done(&self) -> bool {
        self.quit
    }

    fn status(&self) -> Option<String> {
        Some(format!("simulation fidelity: {}", PANELS[self.tab]))
    }
}

/// The summary table of a report: counts compact, ratios to three places.
fn summarize(r: &FidelityReport) -> Vec<SummaryRow> {
    let show = |v: f64, kind: StatKind| match kind {
        StatKind::Count => compact(v),
        StatKind::Ratio => format!("{v:.3}"),
    };
    r.summary()
        .into_iter()
        .map(|s| SummaryRow {
            label: s.label,
            reference: s
                .reference
                .map_or_else(|| "–".to_string(), |v| show(v, s.kind)),
            simulated: show(s.simulated, s.kind),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::copula::fidelity::{GeneMoments, Paired, Profile};
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn report() -> FidelityReport {
        let moments = GeneMoments {
            mean: vec![0.5, 2.0, 0.01],
            var: vec![1.0, 6.0, 0.02],
            zero: vec![0.7, 0.2, 0.99],
        };
        FidelityReport {
            reference: "ref.zarr".into(),
            output: "sim".into(),
            gene_names: vec!["a".into(), "b".into(), "c".into()],
            correlation_genes: vec![0, 1],
            profiles: Paired {
                reference: Profile {
                    library: vec![100.0, 300.0, 2000.0],
                    detected: vec![50.0, 80.0, 400.0],
                    moments: moments.clone(),
                    correlation: vec![0.3],
                },
                simulated: Profile {
                    library: vec![150.0, 250.0, 1800.0, 90.0],
                    detected: vec![60.0, 70.0, 380.0, 40.0],
                    moments,
                    correlation: vec![0.25],
                },
            },
        }
    }

    #[test]
    fn every_panel_draws() {
        let mut view = FidelityView::new(report(), "sim.fidelity.json");
        let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
        for tab in 0..PANELS.len() {
            view.tab = tab;
            terminal.draw(|f| view.render(f)).unwrap();
        }
        view.handle_key(KeyEvent::from(KeyCode::Char('q')));
        assert!(view.done());
    }

    #[test]
    fn summary_reads_the_report() {
        let rows = summarize(&report());
        let get = |label: &str| rows.iter().find(|r| r.label == label).unwrap();
        assert_eq!(get("cells").reference, "3");
        assert_eq!(get("cells").simulated, "4");
        assert_eq!(get("zero fraction: correlation").simulated, "1.000");
    }
}
