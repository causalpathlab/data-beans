//! Full-screen explorer for `stat --interactive`.
//!
//! A table of every row (or column) with its nnz, sum, mean, and sd, sortable
//! and filterable by name, beside a histogram of the chosen statistic. A name
//! filter draws its subset in front of the whole distribution, and the
//! selected entry is marked on the histogram with its rank. Tab switches
//! between rows and columns, computing the other side the first time.

use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Cell, Paragraph, Row, Table, TableState};
use ratatui::Frame;
use regex::{Regex, RegexBuilder};

use super::ui::{
    header, help_line, input_line, median, panel, run_screen, Binned, HistPlot, Scale, Screen,
    ACCENTED, DIM, HIGHLIGHT, PLAIN,
};
use crate::qc::fmt_stat;

/// Statistics in table order.
const STATS: [&str; 4] = ["nnz", "sum", "mean", "sd"];

/// Rows the table moves on PageUp / PageDown.
const PAGE: usize = 20;

/// Decimals for fractional values in the table and summary.
const DECIMALS: usize = 3;

/// Which margin the explorer shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Rows,
    Columns,
}

impl Side {
    fn other(self) -> Self {
        match self {
            Side::Rows => Side::Columns,
            Side::Columns => Side::Rows,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Side::Rows => "rows",
            Side::Columns => "columns",
        }
    }
}

/// One side's entries: names, and nnz/sum/mean/sd per entry.
pub struct Dataset {
    pub names: Vec<Box<str>>,
    pub values: [Vec<f32>; 4],
}

/// Computes a side's statistics the first time it is shown.
pub type Loader<'a> = Box<dyn FnMut(Side) -> anyhow::Result<Dataset> + 'a>;

/// The side not on screen.
enum Other<'a> {
    /// No other side (Tab does nothing).
    Unavailable,
    /// Computed by the loader on first use.
    Lazy(Loader<'a>),
    Ready(Dataset),
}

enum Mode {
    Browse,
    /// Typing a name filter (a case-insensitive regex).
    Filter,
}

/// State of the explorer, independent of the terminal so it can be tested.
pub struct StatExplorer<'a> {
    title: String,
    /// The side on screen; its entries are `names` and `values`.
    side: Side,
    names: Vec<Box<str>>,
    /// Per statistic in [`STATS`] order, one value per entry.
    values: [Vec<f32>; 4],
    other: Other<'a>,
    /// A Tab is waiting for the other side to be computed.
    loading: bool,
    /// Why the last switch failed, shown in the footer.
    status: Option<String>,
    /// Statistic on the histogram, and the sort key unless sorting by name.
    stat: usize,
    by_name: bool,
    descending: bool,
    /// Every entry in sort order; the filter only picks from it.
    order: Vec<usize>,
    filter: String,
    /// Entries passing the filter, in sort order.
    view: Vec<usize>,
    /// Position of the selection in `view`, and the first row on screen.
    cursor: usize,
    offset: usize,
    /// The statistic, sorted, for ranks and the summary line.
    sorted: Vec<f32>,
    x_scale: Scale,
    y_scale: Scale,
    hist: Binned,
    /// Histogram of the filtered entries, when a filter hides some.
    shown: Option<Vec<usize>>,
    mode: Mode,
    quit: bool,
}

impl<'a> StatExplorer<'a> {
    /// Show `data` for `side`; `loader`, if any, computes the other side on
    /// the first Tab.
    pub fn new(title: &str, side: Side, data: Dataset, loader: Option<Loader<'a>>) -> Self {
        let sorted = sorted_copy(&data.values[0]);
        let mut explorer = Self {
            title: title.to_string(),
            side,
            hist: Binned::new(&sorted, Scale::Log),
            sorted,
            names: data.names,
            values: data.values,
            other: loader.map_or(Other::Unavailable, Other::Lazy),
            loading: false,
            status: None,
            stat: 0,
            by_name: false,
            descending: true,
            order: Vec::new(),
            filter: String::new(),
            view: Vec::new(),
            cursor: 0,
            offset: 0,
            x_scale: Scale::Log,
            y_scale: Scale::Log,
            shown: None,
            mode: Mode::Browse,
            quit: false,
        };
        explorer.reorder();
        explorer.refilter(None);
        explorer
    }

    fn selected(&self) -> Option<usize> {
        self.view.get(self.cursor).copied()
    }

    /// The filter as a case-insensitive regex; text that is not a valid
    /// regex matches literally.
    fn filter_regex(&self) -> Option<Regex> {
        if self.filter.is_empty() {
            return None;
        }
        let build = |p: &str| RegexBuilder::new(p).case_insensitive(true).build();
        build(&self.filter)
            .or_else(|_| build(&regex::escape(&self.filter)))
            .ok()
    }

    /// Sort every entry by the current key and direction.
    fn reorder(&mut self) {
        let (names, vals) = (&self.names, &self.values[self.stat]);
        let mut order: Vec<usize> = (0..names.len()).collect();
        if self.by_name {
            order.sort_unstable_by(|&a, &b| names[a].cmp(&names[b]));
        } else {
            order.sort_unstable_by(|&a, &b| {
                vals[a].total_cmp(&vals[b]).then(names[a].cmp(&names[b]))
            });
        }
        if self.descending {
            order.reverse();
        }
        self.order = order;
    }

    /// Pick the entries passing the filter from the sorted order, keeping
    /// `keep` selected when it is still shown.
    fn refilter(&mut self, keep: Option<usize>) {
        let re = self.filter_regex();
        self.view = match &re {
            None => self.order.clone(),
            Some(re) => self
                .order
                .iter()
                .copied()
                .filter(|&i| re.is_match(&self.names[i]))
                .collect(),
        };
        self.cursor = keep
            .and_then(|k| self.view.iter().position(|&i| i == k))
            .unwrap_or(0);
        self.rebin_shown();
    }

    /// Sort, sorted values, and histogram for a new statistic or side.
    fn restat(&mut self) {
        self.sorted = sorted_copy(&self.values[self.stat]);
        self.hist = Binned::new(&self.sorted, self.x_scale);
    }

    fn rebin_shown(&mut self) {
        let vals = &self.values[self.stat];
        self.shown = (self.view.len() < self.names.len())
            .then(|| self.hist.count(self.view.iter().map(|&i| vals[i])));
    }

    /// Reverse the order in place, keeping the same entry selected.
    fn flip(&mut self) {
        self.descending = !self.descending;
        self.order.reverse();
        self.view.reverse();
        self.cursor = self.view.len().saturating_sub(1 + self.cursor);
    }

    /// Sort and plot statistic `s`; pressing the current one flips the order.
    fn choose_stat(&mut self, s: usize) {
        if !self.by_name && self.stat == s {
            return self.flip();
        }
        let keep = self.selected();
        self.by_name = false;
        self.descending = true;
        if self.stat != s {
            self.stat = s;
            self.restat();
        }
        self.reorder();
        self.refilter(keep);
    }

    fn sort_by_name(&mut self) {
        if self.by_name {
            return self.flip();
        }
        let keep = self.selected();
        self.by_name = true;
        self.descending = false;
        self.reorder();
        self.refilter(keep);
    }

    fn set_filter(&mut self, filter: String) {
        let keep = self.selected();
        self.filter = filter;
        self.refilter(keep);
    }

    /// Show the other side now if it is computed, else ask for it.
    fn request_switch(&mut self) {
        match self.other {
            Other::Ready(_) => self.switch(),
            Other::Lazy(_) => self.loading = true,
            Other::Unavailable => {}
        }
    }

    /// Compute the other side (blocking), then show it.
    fn finish_loading(&mut self) {
        self.loading = false;
        let Other::Lazy(loader) = &mut self.other else {
            return;
        };
        match loader(self.side.other()) {
            Ok(data) => {
                self.other = Other::Ready(data);
                self.switch();
            }
            Err(e) => {
                self.status = Some(format!(
                    "could not compute {}: {e}",
                    self.side.other().name()
                ))
            }
        }
    }

    /// Swap in the other side, keeping the sort, filter, and scales.
    fn switch(&mut self) {
        let Other::Ready(data) = std::mem::replace(&mut self.other, Other::Unavailable) else {
            return;
        };
        let names = std::mem::replace(&mut self.names, data.names);
        let values = std::mem::replace(&mut self.values, data.values);
        self.other = Other::Ready(Dataset { names, values });
        self.side = self.side.other();
        self.status = None;
        self.offset = 0;
        self.restat();
        self.reorder();
        self.refilter(None);
    }

    fn step(&mut self, delta: isize) {
        let last = self.view.len().saturating_sub(1) as isize;
        self.cursor = (self.cursor as isize + delta).clamp(0, last) as usize;
    }

    fn render_table(&mut self, frame: &mut Frame, area: Rect) {
        let arrow = if self.descending { " ▼" } else { " ▲" };
        let head = |col: Option<usize>, name: &str| {
            let sorted = match col {
                Some(s) => !self.by_name && self.stat == s,
                None => self.by_name,
            };
            let line = Line::from(if sorted {
                format!("{name}{arrow}")
            } else {
                name.to_string()
            });
            let line = if col.is_some() {
                line.right_aligned()
            } else {
                line
            };
            Cell::from(line).style(if sorted { HIGHLIGHT } else { DIM })
        };
        let mut header_cells = vec![head(None, "name")];
        header_cells.extend((0..4).map(|s| head(Some(s), STATS[s])));

        // Build only the rows on screen (there can be millions), scrolling
        // just enough to keep the selection in view.
        let height = area.height.saturating_sub(3).max(1) as usize;
        if self.cursor < self.offset {
            self.offset = self.cursor;
        } else if self.cursor >= self.offset + height {
            self.offset = self.cursor + 1 - height;
        }
        let rows = self.view.iter().skip(self.offset).take(height).map(|&i| {
            let mut cells = vec![Cell::from(self.names[i].to_string())];
            cells.extend((0..4).map(|s| {
                let cell =
                    Cell::from(Line::from(fmt_stat(self.values[s][i], DECIMALS)).right_aligned());
                if !self.by_name && self.stat == s {
                    cell
                } else {
                    cell.style(DIM)
                }
            }));
            Row::new(cells)
        });

        let title = if self.filter.is_empty() {
            format!(" {} ", self.side.name())
        } else {
            format!(
                " {} · {} of {} match /{}/ ",
                self.side.name(),
                self.view.len(),
                self.names.len(),
                self.filter
            )
        };
        let widths = [
            Constraint::Fill(1),
            Constraint::Length(9),
            Constraint::Length(11),
            Constraint::Length(9),
            Constraint::Length(9),
        ];
        let table = Table::new(rows, widths)
            .header(Row::new(header_cells))
            .row_highlight_style(PLAIN.add_modifier(Modifier::REVERSED))
            .highlight_symbol(Line::from("▶ ").style(ACCENTED))
            .block(panel(title, false));
        // The table sees only the visible slice, so select relative to it.
        let mut state = TableState::default()
            .with_selected((!self.view.is_empty()).then(|| self.cursor - self.offset));
        frame.render_stateful_widget(table, area, &mut state);
    }

    fn render_hist(&self, frame: &mut Frame, area: Rect) {
        let stat = STATS[self.stat];
        let block = panel(format!(" {stat} "), false);
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let [summary, plot] =
            Layout::vertical([Constraint::Length(2), Constraint::Min(5)]).areas(inner);

        let n = self.sorted.len();
        let fmt = |v: f32| fmt_stat(v, DECIMALS);
        let dim = |t: &str| Span::styled(t.to_string(), DIM);
        let mut lines = vec![Line::from(vec![
            dim("min "),
            Span::raw(fmt(self.sorted.first().copied().unwrap_or(0.0))),
            dim("   median "),
            Span::raw(fmt(median(&self.sorted))),
            dim("   max "),
            Span::raw(fmt(self.sorted.last().copied().unwrap_or(0.0))),
        ])];
        let selected = self.selected();
        if let Some(i) = selected {
            let v = self.values[self.stat][i];
            let above = n - self.sorted.partition_point(|&x| x <= v);
            lines.push(Line::from(vec![
                Span::styled(format!("▲ {}", self.names[i]), HIGHLIGHT),
                dim(&format!(" {stat} ")),
                Span::raw(fmt(v)),
                dim(&format!("   rank {} of {}", above + 1, n)),
            ]));
        }
        frame.render_widget(Paragraph::new(lines), summary);

        let bins = self.hist.bins;
        HistPlot {
            bins,
            kmin: self.hist.kmin,
            counts: &self.hist.counts,
            style: &|_| PLAIN,
            subset: self.shown.as_deref(),
            y_scale: self.y_scale,
            pointer: selected.map(|i| bins.key(self.values[self.stat][i] as f64)),
            marks: Vec::new(),
        }
        .render(frame.buffer_mut(), plot);
    }

    fn help(&self) -> Line<'static> {
        if let Mode::Filter = self.mode {
            return input_line(
                "filter /",
                &self.filter,
                &[("Enter", "keep"), ("Esc", "clear")],
            );
        }
        let other = self.side.other().name();
        let lazy = format!("{other} (computed on first use)");
        let mut keys = vec![
            ("↑/↓", "move"),
            ("1-4", "nnz/sum/mean/sd"),
            ("0", "name"),
            ("/", "filter"),
            ("x/y", "scale"),
        ];
        match self.other {
            Other::Ready(_) => keys.push(("Tab", other)),
            Other::Lazy(_) => keys.push(("Tab", &lazy)),
            Other::Unavailable => {}
        }
        keys.push(("q", "quit"));
        let mut line = help_line(&keys);
        if let Some(status) = &self.status {
            line.spans.push(Span::styled(status.clone(), ACCENTED));
        }
        line
    }
}

impl Screen for StatExplorer<'_> {
    fn done(&self) -> bool {
        self.quit
    }

    fn interrupt(&mut self) {
        self.quit = true;
    }

    fn pending_work(&self) -> Option<String> {
        self.loading
            .then(|| format!("computing {} statistics ...", self.side.other().name()))
    }

    fn do_work(&mut self) {
        self.finish_loading();
    }

    fn handle_key(&mut self, key: KeyEvent) {
        match self.mode {
            Mode::Filter => match key.code {
                KeyCode::Enter => self.mode = Mode::Browse,
                KeyCode::Esc => {
                    self.set_filter(String::new());
                    self.mode = Mode::Browse;
                }
                KeyCode::Backspace => {
                    let mut filter = self.filter.clone();
                    filter.pop();
                    self.set_filter(filter);
                }
                KeyCode::Char(c) => self.set_filter(format!("{}{c}", self.filter)),
                _ => {}
            },
            Mode::Browse => match key.code {
                KeyCode::Down | KeyCode::Char('j') => self.step(1),
                KeyCode::Up | KeyCode::Char('k') => self.step(-1),
                KeyCode::PageDown => self.step(PAGE as isize),
                KeyCode::PageUp => self.step(-(PAGE as isize)),
                KeyCode::Home | KeyCode::Char('g') => self.cursor = 0,
                KeyCode::End | KeyCode::Char('G') => {
                    self.cursor = self.view.len().saturating_sub(1)
                }
                KeyCode::Char(c @ '1'..='4') => self.choose_stat(c as usize - '1' as usize),
                KeyCode::Char('0' | 'n') => self.sort_by_name(),
                KeyCode::Char('/') => self.mode = Mode::Filter,
                KeyCode::Tab | KeyCode::BackTab => self.request_switch(),
                KeyCode::Char('x') => {
                    self.x_scale = self.x_scale.next();
                    self.hist = Binned::new(&self.sorted, self.x_scale);
                    self.rebin_shown();
                }
                KeyCode::Char('y') => self.y_scale = self.y_scale.next(),
                KeyCode::Esc if !self.filter.is_empty() => self.set_filter(String::new()),
                KeyCode::Char('q') | KeyCode::Esc => self.quit = true,
                _ => {}
            },
        }
    }

    fn render(&mut self, frame: &mut Frame) {
        let [top, body, footer] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Fill(1),
            Constraint::Length(1),
        ])
        .areas(frame.area());

        let extra = format!(
            "{} {} · x {} · y {}",
            self.names.len(),
            self.side.name(),
            self.x_scale.name(),
            self.y_scale.name()
        );
        frame.render_widget(header("stat", &self.title, &extra), top);

        // Side by side when there is room, else the table above the plot.
        let [left, right] = if body.width >= 110 {
            Layout::horizontal([Constraint::Percentage(48), Constraint::Percentage(52)]).areas(body)
        } else {
            Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(body)
        };
        self.render_table(frame, left);
        self.render_hist(frame, right);
        frame.render_widget(self.help(), footer);
    }
}

fn sorted_copy(values: &[f32]) -> Vec<f32> {
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f32::total_cmp);
    sorted
}

/// Run the explorer full screen until the user quits.
pub fn explore(mut explorer: StatExplorer<'_>) -> anyhow::Result<()> {
    run_screen(&mut explorer)
}

#[cfg(test)]
#[path = "tests/stat_tui.rs"]
mod tests;
