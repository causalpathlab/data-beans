//! Full-screen explorer for per-row and per-column statistics.
//!
//! A table of every row (or column) with its nnz, sum, mean, and sd, sortable
//! and filterable by name, beside a histogram of the chosen statistic. A name
//! filter draws its subset in front of the whole distribution, and the
//! selected entry is marked on the histogram with its rank.
//!
//! Entries can be marked: `v` shows the marked entries' values against the
//! other side, `w` saves the marked names, and when picking (for
//! `subset-*` and `rows`/`columns`) Enter hands the marked entries back.
//! While exploring, Tab switches between rows and columns, computing the
//! other side the first time.

use legume_numeric::matrix::common_io::write_lines;
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

/// Decimals for fractional values in the tables and summary.
const DECIMALS: usize = 3;

/// Width range of a value column in the values view: wide enough for its
/// label (and a sort arrow) where that fits.
const VALUE_WIDTH: (u16, u16) = (9, 24);

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

/// The values of some entries of one side against every entry of the other.
pub struct Values {
    /// The other side's names, one per value row.
    pub names: Vec<Box<str>>,
    /// One column of values per requested entry, in request order.
    pub columns: Vec<Vec<f32>>,
}

/// Computes a side's statistics the first time it is shown.
pub type Loader<'a> = Box<dyn FnMut(Side) -> anyhow::Result<Dataset> + 'a>;

/// Reads the values of the given entries of a side.
pub type ValuesReader<'a> = Box<dyn FnMut(Side, &[usize]) -> anyhow::Result<Values> + 'a>;

/// What the explorer is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purpose {
    /// Look around (`stat --interactive`); Tab switches sides.
    Explore,
    /// Mark entries for a command; Enter returns them, labelled `verb`.
    Pick { verb: &'static str },
}

/// The side not on screen.
enum Other<'a> {
    /// No other side (Tab does nothing).
    Unavailable,
    /// Computed by the loader on first use.
    Lazy(Loader<'a>),
    /// Computed, with its marks.
    Ready(Dataset, Vec<bool>),
}

/// Blocking work a key asked for, run between frames.
enum Work {
    OtherSide,
    Values(Vec<usize>),
}

/// Where typed text goes.
enum Input {
    Filter,
    SaveNames,
    SaveValues,
}

enum Mode {
    Browse,
    Typing(Input, String),
    Values,
}

/// Marked entries against the other side, sortable by any of them.
struct ValuesView {
    /// Names of the marked entries (the value columns).
    labels: Vec<Box<str>>,
    /// The other side's names (the value rows).
    names: Vec<Box<str>>,
    columns: Vec<Vec<f32>>,
    /// Value rows in display order.
    order: Vec<usize>,
    /// Sorted column and whether descending.
    sort: Option<(usize, bool)>,
    /// Selected column, first column on screen, cursor row, first row on
    /// screen.
    col: usize,
    hscroll: usize,
    cursor: usize,
    offset: usize,
}

impl ValuesView {
    fn sort_by_selected(&mut self) {
        let descending = match self.sort {
            Some((c, d)) if c == self.col => !d,
            _ => true,
        };
        let vals = &self.columns[self.col];
        self.order
            .sort_unstable_by(|&a, &b| vals[a].total_cmp(&vals[b]).then(a.cmp(&b)));
        if descending {
            self.order.reverse();
        }
        self.sort = Some((self.col, descending));
        self.cursor = 0;
    }

    /// Tab-separated lines: a header, then one line per value row in order.
    fn lines(&self, side: Side) -> Vec<Box<str>> {
        let mut head = side.other().name().trim_end_matches('s').to_string();
        for label in &self.labels {
            head.push('\t');
            head.push_str(label);
        }
        std::iter::once(head.into_boxed_str())
            .chain(self.order.iter().map(|&r| {
                let mut line = self.names[r].to_string();
                for col in &self.columns {
                    line.push('\t');
                    line.push_str(&col[r].to_string());
                }
                line.into_boxed_str()
            }))
            .collect()
    }
}

/// State of the explorer, independent of the terminal so it can be tested.
pub struct StatExplorer<'a> {
    title: String,
    purpose: Purpose,
    /// The side on screen; its entries are `names` and `values`.
    side: Side,
    names: Vec<Box<str>>,
    /// Per statistic in [`STATS`] order, one value per entry.
    values: [Vec<f32>; 4],
    marked: Vec<bool>,
    other: Other<'a>,
    reader: Option<ValuesReader<'a>>,
    pending: Option<Work>,
    /// The last action's outcome or failure, shown in the footer.
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
    values_view: Option<ValuesView>,
    mode: Mode,
    quit: bool,
    /// Marked entries handed back by Enter when picking.
    picked: Option<Vec<usize>>,
}

impl<'a> StatExplorer<'a> {
    /// Show `data` for `side`. `other`, if any, computes the other side on
    /// the first Tab (exploring only); `reader`, if any, backs the values
    /// view.
    pub fn new(
        title: &str,
        side: Side,
        data: Dataset,
        other: Option<Loader<'a>>,
        reader: Option<ValuesReader<'a>>,
        purpose: Purpose,
    ) -> Self {
        let sorted = sorted_copy(&data.values[0]);
        let other = match (purpose, other) {
            (Purpose::Explore, Some(loader)) => Other::Lazy(loader),
            _ => Other::Unavailable,
        };
        let mut explorer = Self {
            title: title.to_string(),
            purpose,
            side,
            hist: Binned::new(&sorted, Scale::Log),
            sorted,
            marked: vec![false; data.names.len()],
            names: data.names,
            values: data.values,
            other,
            reader,
            pending: None,
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
            values_view: None,
            mode: Mode::Browse,
            quit: false,
            picked: None,
        };
        explorer.reorder();
        explorer.refilter(None);
        explorer
    }

    fn selected(&self) -> Option<usize> {
        self.view.get(self.cursor).copied()
    }

    /// Marked entries in their original order.
    fn marked_entries(&self) -> Vec<usize> {
        (0..self.names.len()).filter(|&i| self.marked[i]).collect()
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

    /// Sorted values and histogram for a new statistic or side.
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

    /// Mark or unmark the selected entry, then move down.
    fn toggle_mark(&mut self) {
        if let Some(i) = self.selected() {
            self.marked[i] = !self.marked[i];
            self.step(1);
        }
    }

    /// Mark every shown entry, or unmark them if all are marked.
    fn toggle_shown(&mut self) {
        let mark = !self.view.iter().all(|&i| self.marked[i]);
        for &i in &self.view {
            self.marked[i] = mark;
        }
    }

    /// Show the other side now if it is computed, else ask for it.
    fn request_switch(&mut self) {
        match self.other {
            Other::Ready(..) => self.switch(),
            Other::Lazy(_) => self.pending = Some(Work::OtherSide),
            Other::Unavailable => {}
        }
    }

    /// Swap in the other side, keeping the sort, filter, and scales.
    fn switch(&mut self) {
        let Other::Ready(data, marked) = std::mem::replace(&mut self.other, Other::Unavailable)
        else {
            return;
        };
        let names = std::mem::replace(&mut self.names, data.names);
        let values = std::mem::replace(&mut self.values, data.values);
        let marks = std::mem::replace(&mut self.marked, marked);
        self.other = Other::Ready(Dataset { names, values }, marks);
        self.side = self.side.other();
        self.status = None;
        self.offset = 0;
        self.restat();
        self.reorder();
        self.refilter(None);
    }

    /// Ask for the values view of the marked entries (or the selected one).
    fn request_values(&mut self) {
        if self.reader.is_none() {
            self.status = Some("no values view here".into());
            return;
        }
        let entries = match self.marked_entries() {
            m if !m.is_empty() => m,
            _ => self.selected().into_iter().collect(),
        };
        if !entries.is_empty() {
            self.pending = Some(Work::Values(entries));
        }
    }

    fn save_names(&mut self, path: &str) {
        let names: Vec<Box<str>> = self
            .marked_entries()
            .into_iter()
            .map(|i| self.names[i].clone())
            .collect();
        self.status = Some(if names.is_empty() {
            "mark entries first (Space)".into()
        } else {
            match write_lines(&names, path) {
                Ok(()) => format!("wrote {} names to {path}", names.len()),
                Err(e) => format!("could not write {path}: {e}"),
            }
        });
    }

    fn save_values(&mut self, path: &str) {
        let Some(view) = &self.values_view else {
            return;
        };
        let lines = view.lines(self.side);
        self.status = Some(match write_lines(&lines, path) {
            Ok(()) => format!("wrote {} lines to {path}", lines.len()),
            Err(e) => format!("could not write {path}: {e}"),
        });
    }

    fn step(&mut self, delta: isize) {
        let last = self.view.len().saturating_sub(1) as isize;
        self.cursor = (self.cursor as isize + delta).clamp(0, last) as usize;
    }

    fn finish_pick(&mut self) {
        let marked = self.marked_entries();
        if marked.is_empty() {
            self.status = Some("mark entries first (Space)".into());
        } else {
            self.picked = Some(marked);
            self.quit = true;
        }
    }

    fn handle_typing(&mut self, input: Input, mut text: String, key: KeyEvent) {
        match key.code {
            KeyCode::Enter => {
                match input {
                    Input::Filter => {}
                    Input::SaveNames => self.save_names(&text),
                    Input::SaveValues => self.save_values(&text),
                }
                self.mode = self.resting_mode(&input);
                return;
            }
            KeyCode::Esc => {
                if let Input::Filter = input {
                    self.set_filter(String::new());
                }
                self.mode = self.resting_mode(&input);
                return;
            }
            KeyCode::Backspace => {
                text.pop();
            }
            KeyCode::Char(c) => text.push(c),
            _ => {}
        }
        if let Input::Filter = input {
            self.set_filter(text.clone());
        }
        self.mode = Mode::Typing(input, text);
    }

    /// The mode to return to after typing.
    fn resting_mode(&self, input: &Input) -> Mode {
        match input {
            Input::SaveValues => Mode::Values,
            _ => Mode::Browse,
        }
    }

    fn handle_browse(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => self.step(1),
            KeyCode::Up | KeyCode::Char('k') => self.step(-1),
            KeyCode::PageDown => self.step(PAGE as isize),
            KeyCode::PageUp => self.step(-(PAGE as isize)),
            KeyCode::Home | KeyCode::Char('g') => self.cursor = 0,
            KeyCode::End | KeyCode::Char('G') => self.cursor = self.view.len().saturating_sub(1),
            KeyCode::Char(c @ '1'..='4') => self.choose_stat(c as usize - '1' as usize),
            KeyCode::Char('0' | 'n') => self.sort_by_name(),
            KeyCode::Char('/') => self.mode = Mode::Typing(Input::Filter, self.filter.clone()),
            KeyCode::Char(' ') => self.toggle_mark(),
            KeyCode::Char('a') => self.toggle_shown(),
            KeyCode::Char('u') => self.marked.fill(false),
            KeyCode::Char('v') => self.request_values(),
            KeyCode::Char('w') => {
                let path = format!("{}.txt", self.side.name());
                self.mode = Mode::Typing(Input::SaveNames, path);
            }
            KeyCode::Tab | KeyCode::BackTab => self.request_switch(),
            KeyCode::Char('x') => {
                self.x_scale = self.x_scale.next();
                self.hist = Binned::new(&self.sorted, self.x_scale);
                self.rebin_shown();
            }
            KeyCode::Char('y') => self.y_scale = self.y_scale.next(),
            KeyCode::Enter => {
                if let Purpose::Pick { .. } = self.purpose {
                    self.finish_pick();
                }
            }
            KeyCode::Esc if !self.filter.is_empty() => self.set_filter(String::new()),
            KeyCode::Char('q') | KeyCode::Esc => self.quit = true,
            _ => {}
        }
    }

    fn handle_values(&mut self, key: KeyEvent) {
        let Some(view) = self.values_view.as_mut() else {
            self.mode = Mode::Browse;
            return;
        };
        let last_row = view.order.len().saturating_sub(1);
        let last_col = view.labels.len().saturating_sub(1);
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => view.cursor = (view.cursor + 1).min(last_row),
            KeyCode::Up | KeyCode::Char('k') => view.cursor = view.cursor.saturating_sub(1),
            KeyCode::PageDown => view.cursor = (view.cursor + PAGE).min(last_row),
            KeyCode::PageUp => view.cursor = view.cursor.saturating_sub(PAGE),
            KeyCode::Home | KeyCode::Char('g') => view.cursor = 0,
            KeyCode::End | KeyCode::Char('G') => view.cursor = last_row,
            KeyCode::Right | KeyCode::Char('l') => view.col = (view.col + 1).min(last_col),
            KeyCode::Left | KeyCode::Char('h') => view.col = view.col.saturating_sub(1),
            KeyCode::Char('s') | KeyCode::Enter => view.sort_by_selected(),
            KeyCode::Char('w') => {
                self.mode = Mode::Typing(Input::SaveValues, "values.tsv".into());
            }
            KeyCode::Esc | KeyCode::Char('v' | 'q') => self.mode = Mode::Browse,
            _ => {}
        }
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
        let mut header_cells = vec![head(None, "  name")];
        header_cells.extend((0..4).map(|s| head(Some(s), STATS[s])));

        // Build only the rows on screen (there can be millions), scrolling
        // just enough to keep the selection in view.
        let height = area.height.saturating_sub(3).max(1) as usize;
        (self.cursor, self.offset) = scroll(self.cursor, self.offset, height);
        let rows = self.view.iter().skip(self.offset).take(height).map(|&i| {
            let name = Line::from(vec![
                Span::styled(if self.marked[i] { "● " } else { "  " }, ACCENTED),
                Span::raw(self.names[i].to_string()),
            ]);
            let mut cells = vec![Cell::from(name)];
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

        let mut title = format!(" {}", self.side.name());
        let n_marked = self.marked.iter().filter(|&&m| m).count();
        if n_marked > 0 {
            title += &format!(" · {n_marked} marked");
        }
        if !self.filter.is_empty() {
            title += &format!(
                " · {} of {} match /{}/",
                self.view.len(),
                self.names.len(),
                self.filter
            );
        }
        title.push(' ');
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
            .highlight_symbol(Line::from("▶").style(ACCENTED))
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

    fn render_values(&mut self, frame: &mut Frame, area: Rect) {
        let side = self.side;
        let Some(view) = self.values_view.as_mut() else {
            return;
        };
        let title = format!(
            " values · {} {} × {} {} ",
            view.labels.len(),
            side.name(),
            view.names.len(),
            side.other().name()
        );
        let block = panel(title, true);
        let inner = block.inner(area);
        frame.render_widget(block, area);

        // Keep the selected column and the cursor row on screen.
        let name_width = 24u16.min(inner.width / 2);
        let longest = view
            .labels
            .iter()
            .map(|l| l.chars().count())
            .max()
            .unwrap_or(0);
        let value_width = (longest as u16 + 3).clamp(VALUE_WIDTH.0, VALUE_WIDTH.1);
        let fit =
            ((inner.width.saturating_sub(name_width + 2)) / (value_width + 1)).max(1) as usize;
        (view.col, view.hscroll) = scroll(view.col, view.hscroll, fit);
        let height = inner.height.saturating_sub(1).max(1) as usize;
        (view.cursor, view.offset) = scroll(view.cursor, view.offset, height);
        let shown_cols: Vec<usize> = (view.hscroll..view.labels.len()).take(fit).collect();

        let mut header_cells = vec![Cell::from(side.other().name()).style(DIM)];
        header_cells.extend(shown_cols.iter().map(|&c| {
            let arrow = match view.sort {
                Some((s, true)) if s == c => " ▼",
                Some((s, false)) if s == c => " ▲",
                _ => "",
            };
            let label = format!("{}{arrow}", view.labels[c]);
            let cell = Cell::from(Line::from(label).right_aligned());
            if c == view.col {
                cell.style(HIGHLIGHT)
            } else {
                cell.style(DIM)
            }
        }));
        let rows = view.order.iter().skip(view.offset).take(height).map(|&r| {
            let mut cells = vec![Cell::from(view.names[r].to_string())];
            cells.extend(shown_cols.iter().map(|&c| {
                let v = view.columns[c][r];
                let cell = Cell::from(Line::from(fmt_stat(v, DECIMALS)).right_aligned());
                if v == 0.0 {
                    cell.style(DIM)
                } else {
                    cell
                }
            }));
            Row::new(cells)
        });
        let mut widths = vec![Constraint::Length(name_width)];
        widths.extend(shown_cols.iter().map(|_| Constraint::Length(value_width)));
        let table = Table::new(rows, widths)
            .header(Row::new(header_cells))
            .row_highlight_style(PLAIN.add_modifier(Modifier::REVERSED));
        let mut state = TableState::default()
            .with_selected((!view.order.is_empty()).then(|| view.cursor - view.offset));
        frame.render_stateful_widget(table, inner, &mut state);
    }

    fn help(&self) -> Line<'static> {
        let mut line = match &self.mode {
            Mode::Typing(Input::Filter, text) => {
                input_line("filter /", text, &[("Enter", "keep"), ("Esc", "clear")])
            }
            Mode::Typing(_, text) => {
                input_line("save to ", text, &[("Enter", "write"), ("Esc", "back")])
            }
            Mode::Values => help_line(&[
                ("↑/↓", "move"),
                ("←/→", "column"),
                ("s", "sort"),
                ("w", "save tsv"),
                ("Esc", "back"),
            ]),
            Mode::Browse => {
                let other = self.side.other().name();
                let lazy = format!("{other} (computed on first use)");
                let n_marked = self.marked.iter().filter(|&&m| m).count();
                let finish = match self.purpose {
                    Purpose::Pick { verb } => format!("{verb} {n_marked} marked"),
                    Purpose::Explore => String::new(),
                };
                let mut keys = vec![
                    ("↑/↓", "move"),
                    ("1-4", "sort"),
                    ("0", "name"),
                    ("/", "filter"),
                    ("Space", "mark"),
                    ("a", "mark shown"),
                ];
                if self.reader.is_some() {
                    keys.push(("v", "values"));
                }
                keys.push(("w", "save names"));
                keys.push(("x/y", "scale"));
                match self.other {
                    Other::Ready(..) => keys.push(("Tab", other)),
                    Other::Lazy(_) => keys.push(("Tab", &lazy)),
                    Other::Unavailable => {}
                }
                if let Purpose::Pick { .. } = self.purpose {
                    keys.push(("Enter", &finish));
                }
                keys.push(("q", "quit"));
                help_line(&keys)
            }
        };
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
        self.picked = None;
        self.quit = true;
    }

    fn pending_work(&self) -> Option<String> {
        Some(match self.pending.as_ref()? {
            Work::OtherSide => format!("computing {} statistics ...", self.side.other().name()),
            Work::Values(entries) => {
                format!("reading {} {} ...", entries.len(), self.side.name())
            }
        })
    }

    fn do_work(&mut self) {
        match self.pending.take() {
            Some(Work::OtherSide) => {
                let Other::Lazy(loader) = &mut self.other else {
                    return;
                };
                match loader(self.side.other()) {
                    Ok(data) => {
                        let marks = vec![false; data.names.len()];
                        self.other = Other::Ready(data, marks);
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
            Some(Work::Values(entries)) => {
                let Some(reader) = self.reader.as_mut() else {
                    return;
                };
                match reader(self.side, &entries) {
                    Ok(values) => {
                        let mut view = ValuesView {
                            labels: entries.iter().map(|&i| self.names[i].clone()).collect(),
                            order: (0..values.names.len()).collect(),
                            names: values.names,
                            columns: values.columns,
                            sort: None,
                            col: 0,
                            hscroll: 0,
                            cursor: 0,
                            offset: 0,
                        };
                        view.sort_by_selected();
                        self.values_view = Some(view);
                        self.mode = Mode::Values;
                    }
                    Err(e) => self.status = Some(format!("could not read values: {e}")),
                }
            }
            None => {}
        }
    }

    fn handle_key(&mut self, key: KeyEvent) {
        self.status = None;
        match std::mem::replace(&mut self.mode, Mode::Browse) {
            Mode::Typing(input, text) => self.handle_typing(input, text, key),
            Mode::Values => {
                self.mode = Mode::Values;
                self.handle_values(key);
            }
            Mode::Browse => self.handle_browse(key),
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
        let badge = match self.purpose {
            Purpose::Explore => "stat",
            Purpose::Pick { verb } => verb,
        };
        frame.render_widget(header(badge, &self.title, &extra), top);

        let in_values = matches!(self.mode, Mode::Values | Mode::Typing(Input::SaveValues, _));
        if in_values {
            self.render_values(frame, body);
        } else {
            // Side by side when there is room, else the table above the plot.
            let [left, right] = if body.width >= 110 {
                Layout::horizontal([Constraint::Percentage(48), Constraint::Percentage(52)])
                    .areas(body)
            } else {
                Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .areas(body)
            };
            self.render_table(frame, left);
            self.render_hist(frame, right);
        }
        frame.render_widget(self.help(), footer);
    }
}

/// Keep `cursor` within the `height` rows shown from `offset`: returns the
/// cursor and the offset scrolled just enough.
fn scroll(cursor: usize, offset: usize, height: usize) -> (usize, usize) {
    let offset = if cursor < offset {
        cursor
    } else if cursor >= offset + height {
        cursor + 1 - height
    } else {
        offset
    };
    (cursor, offset)
}

fn sorted_copy(values: &[f32]) -> Vec<f32> {
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f32::total_cmp);
    sorted
}

/// Run the explorer full screen until the user quits. When picking, returns
/// the marked entries if the user finished with Enter.
pub fn explore(mut explorer: StatExplorer<'_>) -> anyhow::Result<Option<Vec<usize>>> {
    run_screen(&mut explorer)?;
    Ok(explorer.picked)
}

#[cfg(test)]
#[path = "tests/stat_tui.rs"]
mod tests;
