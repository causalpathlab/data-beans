//! What a command is about to write, before it writes it: a table of the
//! inputs or outputs (rows, columns, what is shared), notes, and the
//! command's settings, each on a key. A flag flips on its key; a number
//! asks for a new value. Every change redraws the plan; Enter writes with
//! the settings shown, `q` writes nothing. A plan that cannot be written
//! says why, and Enter waits for a setting that can.

use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Cell, Paragraph, Row, Table};
use ratatui::Frame;

use super::ui::{header, help_line, input_line, panel, title_bar, Screen, Tui, DIM, HIGHLIGHT};

/// A setting's value.
#[derive(Debug, Clone, PartialEq)]
pub enum Setting {
    Flag(bool),
    /// A number, with the value it is reset to if cleared (`None`: unset).
    Number(Option<f64>),
}

impl Setting {
    pub fn flag(&self) -> bool {
        matches!(self, Setting::Flag(true))
    }

    pub fn number(&self) -> Option<f64> {
        match self {
            Setting::Number(v) => *v,
            Setting::Flag(_) => None,
        }
    }

    fn shown(&self) -> String {
        match self {
            Setting::Flag(true) => "on".into(),
            Setting::Flag(false) => "off".into(),
            Setting::Number(Some(v)) => short(*v),
            Setting::Number(None) => "unset".into(),
        }
    }
}

/// `v` without a trailing `.0`.
fn short(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

/// One setting on a key.
#[derive(Debug, Clone)]
pub struct Control {
    pub key: char,
    pub label: &'static str,
    pub value: Setting,
}

impl Control {
    pub fn flag(key: char, label: &'static str, on: bool) -> Self {
        Self {
            key,
            label,
            value: Setting::Flag(on),
        }
    }

    pub fn number(key: char, label: &'static str, value: Option<f64>) -> Self {
        Self {
            key,
            label,
            value: Setting::Number(value),
        }
    }
}

/// What a command would write with some settings.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Plan {
    /// The table's heading, then its rows.
    pub columns: Vec<String>,
    pub rows: Vec<Vec<String>>,
    /// Lines under the table: the output, its size.
    pub notes: Vec<String>,
    /// Why it cannot be written so; Enter waits.
    pub blocked: Option<String>,
}

/// Computes the plan for the settings, in control order.
pub type Planner<'a> = Box<dyn Fn(&[Setting]) -> anyhow::Result<Plan> + 'a>;

#[derive(Debug, Clone, PartialEq)]
enum Mode {
    Browse,
    /// Typing a new value for the control at this index.
    Edit(usize, String),
}

/// The view, independent of the terminal so it can be tested.
pub struct PlanView<'a> {
    badge: &'static str,
    title: String,
    /// What Enter does, in a word or two (`merge`, `write`), and `q`.
    verb: &'static str,
    cancel: &'static str,
    controls: Vec<Control>,
    planner: Planner<'a>,
    plan: Plan,
    mode: Mode,
    status: Option<String>,
    /// Some(true) on Enter, Some(false) on `q`.
    decision: Option<bool>,
    offset: usize,
    session_keys: Vec<(String, String)>,
}

impl<'a> PlanView<'a> {
    pub fn new(
        badge: &'static str,
        title: &str,
        verb: &'static str,
        controls: Vec<Control>,
        planner: Planner<'a>,
    ) -> Self {
        debug_assert!(
            controls.iter().all(|c| c.key != 'q'),
            "q leaves the plan; no setting takes it"
        );
        let mut view = Self {
            badge,
            title: title.to_string(),
            verb,
            cancel: "cancel",
            controls,
            planner,
            plan: Plan::default(),
            mode: Mode::Browse,
            status: None,
            decision: None,
            offset: 0,
            session_keys: super::session_keys(),
        };
        view.replan();
        view
    }

    /// What `q` is said to do (`cancel` unless told).
    pub fn cancel_label(mut self, label: &'static str) -> Self {
        self.cancel = label;
        self
    }

    fn settings(&self) -> Vec<Setting> {
        self.controls.iter().map(|c| c.value.clone()).collect()
    }

    fn replan(&mut self) {
        self.plan = match (self.planner)(&self.settings()) {
            Ok(plan) => plan,
            Err(e) => Plan {
                blocked: Some(format!("{e:#}")),
                ..Plan::default()
            },
        };
        self.offset = 0;
    }

    /// The settings Enter wrote with, or `None` when left with `q`.
    pub fn decided(&self) -> Option<Vec<Setting>> {
        (self.decision == Some(true)).then(|| self.settings())
    }

    fn browse_key(&mut self, key: KeyEvent) {
        self.status = None;
        // A setting's key first: it may be any letter but q.
        if let KeyCode::Char(c) = key.code {
            if let Some(i) = self.controls.iter().position(|k| k.key == c) {
                match &self.controls[i].value {
                    Setting::Flag(on) => {
                        self.controls[i].value = Setting::Flag(!on);
                        self.replan();
                    }
                    Setting::Number(v) => {
                        let text = v.map(short).unwrap_or_default();
                        self.mode = Mode::Edit(i, text);
                    }
                }
                return;
            }
        }
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => self.decision = Some(false),
            KeyCode::Enter => match &self.plan.blocked {
                Some(why) => self.status = Some(format!("not yet: {why}")),
                None => self.decision = Some(true),
            },
            KeyCode::Down => {
                self.offset = (self.offset + 1).min(self.plan.rows.len().saturating_sub(1))
            }
            KeyCode::Up => self.offset = self.offset.saturating_sub(1),
            _ => {}
        }
    }

    fn edit_key(&mut self, key: KeyEvent) {
        let Mode::Edit(i, text) = &mut self.mode else {
            return;
        };
        let i = *i;
        match key.code {
            KeyCode::Esc => self.mode = Mode::Browse,
            KeyCode::Enter => {
                let text = text.trim().to_string();
                if text.is_empty() {
                    self.controls[i].value = Setting::Number(None);
                } else {
                    match text.parse::<f64>() {
                        Ok(v) if v.is_finite() => self.controls[i].value = Setting::Number(Some(v)),
                        _ => {
                            self.status = Some(format!("{text}: not a number"));
                            return;
                        }
                    }
                }
                self.mode = Mode::Browse;
                self.replan();
            }
            KeyCode::Backspace => {
                text.pop();
            }
            KeyCode::Char(c) if c.is_ascii_digit() || matches!(c, '.' | '-' | 'e') => text.push(c),
            _ => {}
        }
    }

    fn table(&self) -> Table<'static> {
        let n = self.plan.columns.len();
        let mut widths = vec![0usize; n];
        for row in std::iter::once(&self.plan.columns).chain(&self.plan.rows) {
            for (w, cell) in widths.iter_mut().zip(row) {
                *w = (*w).max(cell.chars().count());
            }
        }
        let head = Row::new(
            self.plan
                .columns
                .iter()
                .map(|c| Cell::from(c.clone()).style(DIM)),
        );
        let rows: Vec<Row> = self
            .plan
            .rows
            .iter()
            .skip(self.offset)
            .map(|r| Row::new(r.iter().map(|c| Cell::from(c.clone()))))
            .collect();
        let widths: Vec<Constraint> = widths
            .iter()
            .map(|&w| Constraint::Length(w.min(48) as u16))
            .collect();
        Table::new(rows, widths).header(head).column_spacing(2)
    }

    fn footer(&self) -> Line<'static> {
        if let Mode::Edit(i, text) = &self.mode {
            let label = self.controls[*i].label;
            return input_line(
                &format!("{label}: "),
                text,
                &[("enter", "set"), ("esc", "back")],
            );
        }
        if let Some(s) = &self.status {
            return Line::from(Span::styled(format!(" {s}"), HIGHLIGHT));
        }
        let mut keys: Vec<(String, String)> = vec![("enter".into(), self.verb.into())];
        for c in &self.controls {
            keys.push((c.key.to_string(), c.label.into()));
        }
        keys.push(("q".into(), self.cancel.into()));
        let pairs: Vec<(&str, &str)> = keys.iter().map(|(k, w)| (k.as_str(), w.as_str())).collect();
        help_line(&pairs)
    }
}

impl Screen for PlanView<'_> {
    fn render(&mut self, frame: &mut Frame) {
        let settings = self.controls.len().div_ceil(3) as u16;
        let notes = self.plan.notes.len() as u16 + u16::from(self.plan.blocked.is_some());
        let [top, table, below, controls, foot] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Fill(1),
            Constraint::Length(notes),
            Constraint::Length(if settings > 0 { settings + 2 } else { 0 }),
            Constraint::Length(1),
        ])
        .areas(frame.area());
        let extra = match &self.plan.blocked {
            Some(_) => "cannot be written yet".to_string(),
            None => format!("Enter to {}", self.verb),
        };
        title_bar(
            frame,
            top,
            header(self.badge, &self.title, &extra),
            &self.session_keys,
        );
        let block = panel("plan".into(), true);
        let inner = block.inner(table);
        frame.render_widget(block, table);
        frame.render_widget(self.table(), inner);
        let mut lines: Vec<Line> = self
            .plan
            .notes
            .iter()
            .map(|n| Line::from(format!(" {n}")))
            .collect();
        if let Some(why) = &self.plan.blocked {
            lines.push(Line::from(Span::styled(
                format!(" ✗ {why}"),
                HIGHLIGHT.add_modifier(Modifier::BOLD),
            )));
        }
        frame.render_widget(Paragraph::new(lines), below);
        if settings > 0 {
            let block = panel("settings".into(), false);
            let inner = block.inner(controls);
            frame.render_widget(block, controls);
            let lines: Vec<Line> = self
                .controls
                .chunks(3)
                .map(|chunk| {
                    let mut spans = Vec::new();
                    for c in chunk {
                        spans.push(Span::styled(format!(" {} ", c.key), HIGHLIGHT));
                        spans.push(Span::styled(format!("{} ", c.label), DIM));
                        spans.push(Span::raw(format!("{:<12}", c.value.shown())));
                    }
                    Line::from(spans)
                })
                .collect();
            frame.render_widget(Paragraph::new(lines), inner);
        }
        frame.render_widget(self.footer(), foot);
    }

    fn handle_key(&mut self, key: KeyEvent) {
        match self.mode {
            Mode::Browse => self.browse_key(key),
            Mode::Edit(..) => self.edit_key(key),
        }
    }

    fn interrupt(&mut self) {
        self.decision = Some(false);
    }

    fn done(&self) -> bool {
        self.decision.is_some()
    }

    fn status(&self) -> Option<String> {
        Some(format!("planning {}", self.title))
    }
}

/// Show `view` on the screen of `tui`; the settings it was written with,
/// or `None` when cancelled.
pub fn plan_in(tui: &mut Tui<'_>, mut view: PlanView<'_>) -> anyhow::Result<Option<Vec<Setting>>> {
    tui.run(&mut view)?;
    Ok(view.decided())
}

#[cfg(test)]
#[path = "tests/plan_tui.rs"]
mod tests;
