//! Full-screen nnz cutoff picker for `squeeze --interactive`.
//!
//! Shows the row and column nnz histograms with the current cutoff, and lets
//! the user move each cutoff with the keyboard while the drop counts update
//! live. Either histogram axis can be on a log, sqrt, or linear scale. On the
//! log scale the bins are the printed histogram's (`qc`), and drop counts
//! always use the squeeze's own rule, so all views agree.

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Flex, Layout, Rect};
use ratatui::style::Stylize;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Clear, LineGauge, Paragraph, Wrap};
use ratatui::{DefaultTerminal, Frame};

use super::ui::{
    header, help_line, title, Binning, HistPlot, Layer, Scale, ACCENTED, DIM, HIGHLIGHT, PLAIN,
};
use crate::qc::below_nnz_cutoff;

/// One axis (rows or columns): the sorted nnz counts, their histogram, and
/// the cutoff being edited.
pub struct AxisView {
    label: String,
    sorted: Vec<f32>,
    bins: Binning,
    /// Counts per bin key, indexed from `kmin`.
    counts: Vec<usize>,
    kmin: i32,
    /// Cutoffs the bin steps visit, ascending: 0, the first count of every
    /// bin past the lowest (the cutoff that drops the bars left of it), and
    /// one past the max (drops everything).
    stops: Vec<usize>,
    cutoff: usize,
    initial: usize,
    suggest: Option<usize>,
}

impl AxisView {
    pub fn new(label: &str, nnz: &[f32], cutoff: usize, suggest: Option<usize>) -> Self {
        let mut sorted = nnz.to_vec();
        sorted.sort_unstable_by(f32::total_cmp);
        let mut view = Self {
            label: label.to_string(),
            bins: Binning::new(Scale::Log, 0.0, true),
            sorted,
            counts: Vec::new(),
            kmin: 0,
            stops: Vec::new(),
            cutoff,
            initial: cutoff,
            suggest,
        };
        view.set_scale(Scale::Log);
        view
    }

    /// Re-bin the histogram on `scale`.
    fn set_scale(&mut self, scale: Scale) {
        let min = self.sorted.first().map_or(0, |&x| x as usize);
        let max = self.max_value();
        self.bins = Binning::new(scale, max as f64, true);
        let (kmin, kmax) = (self.bins.key(min as f64), self.bins.key(max as f64));
        self.kmin = kmin;
        self.stops = std::iter::once(0)
            .chain((kmin + 1..=kmax).map(|k| self.bins.lower_edge(k)))
            .chain(std::iter::once(max + 1))
            .collect();
        self.stops.dedup();
        // nnz are whole counts, so bin `k` holds exactly the values between
        // its lower edge and the next one: two binary searches per bin.
        self.counts = (kmin..=kmax)
            .map(|k| {
                self.removed_at(self.bins.lower_edge(k + 1))
                    - self.removed_at(self.bins.lower_edge(k))
            })
            .collect();
    }

    fn total(&self) -> usize {
        self.sorted.len()
    }

    /// Number of entries `cutoff` drops, by the rule the squeeze applies.
    fn removed_at(&self, cutoff: usize) -> usize {
        self.sorted
            .partition_point(|&x| below_nnz_cutoff(x, cutoff))
    }

    fn removed(&self) -> usize {
        self.removed_at(self.cutoff)
    }

    fn max_value(&self) -> usize {
        self.sorted.last().map_or(0, |&x| x as usize)
    }

    fn median(&self) -> f32 {
        let n = self.sorted.len();
        match n {
            0 => 0.0,
            _ if n.is_multiple_of(2) => (self.sorted[n / 2 - 1] + self.sorted[n / 2]) / 2.0,
            _ => self.sorted[n / 2],
        }
    }

    /// Move the cutoff to the next (`dir > 0`) or previous stop, so one
    /// keypress moves the marker by exactly one bar.
    fn step_bin(&mut self, dir: i32) {
        self.cutoff = if dir > 0 {
            let i = self.stops.partition_point(|&s| s <= self.cutoff);
            self.stops.get(i).copied().unwrap_or(self.cutoff)
        } else {
            let i = self.stops.partition_point(|&s| s < self.cutoff);
            i.checked_sub(1).map_or(0, |i| self.stops[i])
        };
    }

    /// Nudge the cutoff by an exact amount.
    fn nudge(&mut self, delta: i64) {
        let cap = self.max_value() as i64 + 1;
        self.cutoff = (self.cutoff as i64 + delta).clamp(0, cap) as usize;
    }

    fn snap_to_suggestion(&mut self) {
        if let Some(s) = self.suggest {
            self.cutoff = s;
        }
    }

    fn reset(&mut self) {
        self.cutoff = self.initial;
    }

    fn pct(&self, removed: usize) -> f64 {
        match self.total() {
            0 => 0.0,
            n => 100.0 * removed as f64 / n as f64,
        }
    }

    fn stats_lines(&self) -> Vec<Line<'static>> {
        let dim = |t: &str| Span::styled(t.to_string(), DIM);
        let removed = self.removed();
        let suggestion = match self.suggest {
            Some(s) => dim(&format!(
                "   ◆ suggested {} (drops {:.2}%)",
                s,
                self.pct(self.removed_at(s))
            )),
            None => dim("   no trough suggestion"),
        };
        vec![
            Line::from(vec![
                dim("n "),
                Span::raw(self.total().to_string()),
                dim("   min "),
                Span::raw(self.sorted.first().map_or(0, |&x| x as usize).to_string()),
                dim("   median "),
                Span::raw(self.median().to_string()),
                dim("   max "),
                Span::raw(self.max_value().to_string()),
            ]),
            Line::from(vec![
                dim("cutoff "),
                Span::styled(self.cutoff.to_string(), HIGHLIGHT),
                dim("   drops "),
                Span::styled(
                    format!("{} / {} ({:.2}%)", removed, self.total(), self.pct(removed)),
                    ACCENTED,
                ),
                suggestion,
            ]),
        ]
    }

    fn render(&self, frame: &mut Frame, area: Rect, focused: bool, y_scale: Scale) {
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(if focused { PLAIN } else { DIM })
            .title(title(
                format!(" {} nnz ", self.label),
                if focused { HIGHLIGHT } else { DIM },
            ));
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let [stats, gauge, plot] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Length(1),
            Constraint::Min(5),
        ])
        .areas(inner);
        frame.render_widget(Paragraph::new(self.stats_lines()), stats);

        let kept = self.total() - self.removed();
        let ratio = match self.total() {
            0 => 1.0,
            n => kept as f64 / n as f64,
        };
        frame.render_widget(
            LineGauge::default()
                .ratio(ratio)
                .label(Line::from(format!("keeps {:>6.2}% ", 100.0 * ratio)))
                .filled_symbol("━")
                .unfilled_symbol("━")
                .filled_style(PLAIN)
                .unfilled_style(ACCENTED),
            gauge,
        );

        // Bars left of the cutoff's bin are what it drops.
        let cut_key = (self.cutoff > 0).then(|| self.bins.key(self.cutoff as f64));
        let style = |k: i32| match cut_key {
            Some(c) if k < c => ACCENTED,
            _ => PLAIN,
        };
        let mut marks = Vec::new();
        if let Some(s) = self.suggest {
            marks.push((self.bins.key(s as f64), "◆", PLAIN.bold()));
        }
        if let Some(c) = cut_key {
            marks.push((c, "▲", HIGHLIGHT));
        }
        HistPlot {
            bins: self.bins,
            kmin: self.kmin,
            layers: vec![Layer {
                counts: &self.counts,
                style: &style,
            }],
            y_scale,
            rule: cut_key.map(|c| (c, ACCENTED)),
            marks,
        }
        .render(frame.buffer_mut(), plot);
    }
}

/// What the user decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Proceed { row: usize, column: usize },
    Cancel,
}

enum Mode {
    Browse,
    /// Typing an exact cutoff for the focused axis.
    Edit(String),
    /// Asking before an in-place write.
    ConfirmInPlace,
}

/// State of the picker, independent of the terminal so it can be tested.
pub struct CutoffPicker {
    title: String,
    axes: [AxisView; 2],
    focus: usize,
    x_scale: Scale,
    y_scale: Scale,
    /// When set, Enter asks before squeezing this file in place.
    in_place_target: Option<String>,
    mode: Mode,
    decision: Option<Decision>,
}

impl CutoffPicker {
    pub fn new(
        title: &str,
        row: AxisView,
        column: AxisView,
        in_place_target: Option<&str>,
    ) -> Self {
        Self {
            title: title.to_string(),
            axes: [row, column],
            focus: 0,
            x_scale: Scale::Log,
            y_scale: Scale::Log,
            in_place_target: in_place_target.map(str::to_string),
            mode: Mode::Browse,
            decision: None,
        }
    }

    fn proceed(&mut self) {
        self.decision = Some(Decision::Proceed {
            row: self.axes[0].cutoff,
            column: self.axes[1].cutoff,
        });
    }

    fn handle_key(&mut self, key: KeyEvent) {
        if key.kind != KeyEventKind::Press {
            return;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            self.decision = Some(Decision::Cancel);
            return;
        }
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        let f = self.focus;
        match &mut self.mode {
            Mode::Edit(buf) => match key.code {
                KeyCode::Char(c) if c.is_ascii_digit() && buf.len() < 12 => buf.push(c),
                KeyCode::Backspace => {
                    buf.pop();
                }
                KeyCode::Enter => {
                    if let Ok(v) = buf.parse::<usize>() {
                        self.axes[f].cutoff = v;
                    }
                    self.mode = Mode::Browse;
                }
                KeyCode::Esc => self.mode = Mode::Browse,
                _ => {}
            },
            Mode::ConfirmInPlace => match key.code {
                KeyCode::Char('y' | 'Y') => self.proceed(),
                KeyCode::Char('n' | 'N') | KeyCode::Esc => self.mode = Mode::Browse,
                _ => {}
            },
            Mode::Browse => match key.code {
                KeyCode::Tab
                | KeyCode::BackTab
                | KeyCode::Up
                | KeyCode::Down
                | KeyCode::Char('k' | 'j') => self.focus = 1 - self.focus,
                KeyCode::Left if shift => self.axes[f].nudge(-1),
                KeyCode::Right if shift => self.axes[f].nudge(1),
                KeyCode::Left | KeyCode::Char('h') => self.axes[f].step_bin(-1),
                KeyCode::Right | KeyCode::Char('l') => self.axes[f].step_bin(1),
                KeyCode::Char('-' | ',') => self.axes[f].nudge(-1),
                KeyCode::Char('+' | '=' | '.') => self.axes[f].nudge(1),
                KeyCode::Char('s') => self.axes[f].snap_to_suggestion(),
                KeyCode::Char('x') => {
                    self.x_scale = self.x_scale.next();
                    for axis in &mut self.axes {
                        axis.set_scale(self.x_scale);
                    }
                }
                KeyCode::Char('y') => self.y_scale = self.y_scale.next(),
                KeyCode::Char('r') => self.axes[f].reset(),
                KeyCode::Char('e') => self.mode = Mode::Edit(String::new()),
                KeyCode::Char(c) if c.is_ascii_digit() => self.mode = Mode::Edit(c.to_string()),
                KeyCode::Enter => {
                    if self.in_place_target.is_some() {
                        self.mode = Mode::ConfirmInPlace;
                    } else {
                        self.proceed();
                    }
                }
                KeyCode::Char('q') | KeyCode::Esc => self.decision = Some(Decision::Cancel),
                _ => {}
            },
        }
    }

    fn render(&self, frame: &mut Frame) {
        let [top, row_area, col_area, footer] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Fill(1),
            Constraint::Fill(1),
            Constraint::Length(1),
        ])
        .areas(frame.area());

        let scales = format!("x {} · y {}", self.x_scale.name(), self.y_scale.name());
        frame.render_widget(header("squeeze", &self.title, &scales), top);
        self.axes[0].render(frame, row_area, self.focus == 0, self.y_scale);
        self.axes[1].render(frame, col_area, self.focus == 1, self.y_scale);

        let help = match &self.mode {
            Mode::Edit(buf) => {
                let mut spans = vec![
                    Span::raw(format!(" {} cutoff: ", self.axes[self.focus].label)),
                    Span::styled(format!("{}▏", buf), HIGHLIGHT),
                    Span::raw("  "),
                ];
                spans.extend(help_line(&[("Enter", "set"), ("Esc", "back")]).spans);
                Line::from(spans)
            }
            _ => help_line(&[
                ("←/→", "bin"),
                ("-/+", "±1"),
                ("0-9", "type"),
                ("s", "suggested"),
                ("r", "reset"),
                ("x/y", "scale"),
                ("Tab", "rows/cols"),
                ("Enter", "squeeze"),
                ("q", "cancel"),
            ]),
        };
        frame.render_widget(help, footer);

        if let (Mode::ConfirmInPlace, Some(target)) = (&self.mode, &self.in_place_target) {
            self.render_confirm(frame, target);
        }
    }

    fn render_confirm(&self, frame: &mut Frame, target: &str) {
        let area = frame.area();
        let width = (target.len() as u16 + 6).max(44).min(area.width);
        // Long paths wrap onto extra lines rather than getting cut off.
        let path_lines = (target.len() as u16).div_ceil(width.saturating_sub(2).max(1));
        let [popup] = Layout::horizontal([Constraint::Length(width)])
            .flex(Flex::Center)
            .areas(area);
        let [popup] = Layout::vertical([Constraint::Length(6 + path_lines)])
            .flex(Flex::Center)
            .areas(popup);
        let text = vec![
            Line::from("Squeeze in place? This permanently alters"),
            Line::from(target.to_string()).bold(),
            Line::from(format!(
                "row cutoff {}, column cutoff {}",
                self.axes[0].cutoff, self.axes[1].cutoff
            ))
            .style(DIM),
            Line::from(vec![
                Span::styled("y", HIGHLIGHT),
                Span::raw(" yes   "),
                Span::styled("n", HIGHLIGHT),
                Span::raw(" back"),
            ]),
        ];
        frame.render_widget(Clear, popup);
        frame.render_widget(
            Paragraph::new(text)
                .centered()
                .wrap(Wrap { trim: false })
                .block(
                    Block::bordered()
                        .border_type(BorderType::Double)
                        .border_style(ACCENTED)
                        .title(Line::from(" confirm ").style(HIGHLIGHT).centered()),
                ),
            popup,
        );
    }

    fn run(&mut self, terminal: &mut DefaultTerminal) -> anyhow::Result<Decision> {
        loop {
            terminal.draw(|f| self.render(f))?;
            if let Event::Key(key) = event::read()? {
                self.handle_key(key);
            }
            if let Some(d) = self.decision.take() {
                return Ok(d);
            }
        }
    }
}

/// Run the picker full screen until the user proceeds or cancels. The
/// terminal is restored on return and on panic.
pub fn choose_cutoffs(mut picker: CutoffPicker) -> anyhow::Result<Decision> {
    ratatui::run(|terminal| picker.run(terminal))
}

#[cfg(test)]
#[path = "tests/cutoff_tui.rs"]
mod tests;
