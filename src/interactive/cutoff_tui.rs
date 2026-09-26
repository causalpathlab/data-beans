//! Full-screen nnz cutoff picker for `squeeze --interactive`.
//!
//! Shows the row and column nnz histograms with the current cutoff, and lets
//! the user move each cutoff with the keyboard while the drop counts update
//! live. Either histogram axis can be on a log, sqrt, or linear scale. On the
//! log scale the bins are the printed histogram's (`qc`), and drop counts
//! always use the squeeze's own rule, so all views agree.

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Flex, Layout, Rect};
use ratatui::style::Stylize;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Clear, LineGauge, Paragraph, Wrap};
use ratatui::Frame;

use super::ui::{
    header, help_line, input_line, median, panel, run_screen, Binned, HistPlot, Scale, Screen,
    ACCENTED, DIM, HIGHLIGHT, PLAIN,
};
use crate::qc::{below_nnz_cutoff, pct};

/// One axis (rows or columns): the sorted nnz counts, their histogram, and
/// the cutoff being edited.
pub struct AxisView {
    label: String,
    sorted: Vec<f32>,
    hist: Binned,
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
        let hist = Binned::new(&sorted, Scale::Log);
        let stops = Self::stops(&hist, &sorted);
        Self {
            label: label.to_string(),
            sorted,
            hist,
            stops,
            cutoff,
            initial: cutoff,
            suggest,
        }
    }

    fn stops(hist: &Binned, sorted: &[f32]) -> Vec<usize> {
        let max = sorted.last().map_or(0, |&x| x as usize);
        let mut stops: Vec<usize> = std::iter::once(0)
            .chain((hist.kmin + 1..=hist.kmax()).map(|k| hist.bins.lower_edge(k)))
            .chain(std::iter::once(max + 1))
            .collect();
        stops.dedup();
        stops
    }

    /// Re-bin the histogram on `scale`.
    fn set_scale(&mut self, scale: Scale) {
        self.hist = Binned::new(&self.sorted, scale);
        self.stops = Self::stops(&self.hist, &self.sorted);
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

    fn stats_lines(&self) -> Vec<Line<'static>> {
        let dim = |t: &str| Span::styled(t.to_string(), DIM);
        let removed = self.removed();
        let suggestion = match self.suggest {
            Some(s) => dim(&format!(
                "   ◆ suggested {} (drops {:.2}%)",
                s,
                pct(self.removed_at(s), self.total())
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
                Span::raw(median(&self.sorted).to_string()),
                dim("   max "),
                Span::raw(self.max_value().to_string()),
            ]),
            Line::from(vec![
                dim("cutoff "),
                Span::styled(self.cutoff.to_string(), HIGHLIGHT),
                dim("   drops "),
                Span::styled(
                    format!(
                        "{} / {} ({:.2}%)",
                        removed,
                        self.total(),
                        pct(removed, self.total())
                    ),
                    ACCENTED,
                ),
                suggestion,
            ]),
        ]
    }

    fn render(&self, frame: &mut Frame, area: Rect, focused: bool, y_scale: Scale) {
        let block = panel(format!(" {} nnz ", self.label), focused);
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let [stats, gauge, plot] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Length(1),
            Constraint::Min(5),
        ])
        .areas(inner);
        frame.render_widget(Paragraph::new(self.stats_lines()), stats);

        let ratio = 1.0 - pct(self.removed(), self.total()) / 100.0;
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
        let bins = self.hist.bins;
        let cut_key = (self.cutoff > 0).then(|| bins.key(self.cutoff as f64));
        let style = |k: i32| match cut_key {
            Some(c) if k < c => ACCENTED,
            _ => PLAIN,
        };
        HistPlot {
            bins,
            kmin: self.hist.kmin,
            counts: &self.hist.counts,
            style: &style,
            subset: None,
            y_scale,
            pointer: cut_key,
            marks: self
                .suggest
                .map(|s| (bins.key(s as f64), "◆", PLAIN.bold()))
                .into_iter()
                .collect(),
        }
        .render(frame.buffer_mut(), plot);
    }
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
    /// Set once the user is done: the row and column cutoffs, or `None` to
    /// cancel.
    decision: Option<Option<(usize, usize)>>,
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
        self.decision = Some(Some((self.axes[0].cutoff, self.axes[1].cutoff)));
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
            help_line(&[("y", "yes"), ("n", "back")]),
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
}

impl Screen for CutoffPicker {
    fn done(&self) -> bool {
        self.decision.is_some()
    }

    fn interrupt(&mut self) {
        self.decision = Some(None);
    }

    fn handle_key(&mut self, key: KeyEvent) {
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
                KeyCode::Char('q') | KeyCode::Esc => self.decision = Some(None),
                _ => {}
            },
        }
    }

    fn render(&mut self, frame: &mut Frame) {
        let [top, row_area, col_area, footer] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Fill(1),
            Constraint::Fill(1),
            Constraint::Length(1),
        ])
        .areas(frame.area());

        let scales = format!("x {} · y {}", self.x_scale.name(), self.y_scale.name());
        frame.render_widget(header("squeeze", &self.title, &scales), top);
        for (i, area) in [row_area, col_area].into_iter().enumerate() {
            self.axes[i].render(frame, area, self.focus == i, self.y_scale);
        }

        let help = match &self.mode {
            Mode::Edit(buf) => input_line(
                &format!("{} cutoff: ", self.axes[self.focus].label),
                buf,
                &[("Enter", "set"), ("Esc", "back")],
            ),
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
}

/// Run the picker full screen until the user proceeds (the row and column
/// cutoffs) or cancels (`None`).
pub fn choose_cutoffs(mut picker: CutoffPicker) -> anyhow::Result<Option<(usize, usize)>> {
    run_screen(&mut picker)?;
    Ok(picker.decision.flatten())
}

#[cfg(test)]
#[path = "tests/cutoff_tui.rs"]
mod tests;
