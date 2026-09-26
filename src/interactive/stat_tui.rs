//! Full-screen explorer for `stat --interactive`.
//!
//! A table of every row (or column) with its nnz, sum, mean, and sd, sortable
//! and filterable by name, beside a histogram of the chosen statistic. A name
//! filter draws its subset in front of the whole distribution, and the
//! selected entry is marked on the histogram with its rank. Tab switches
//! between rows and columns, computing the other side the first time.

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Cell, Paragraph, Row, Table, TableState};
use ratatui::{DefaultTerminal, Frame};
use regex::{Regex, RegexBuilder};

use super::ui::{
    bin_counts, header, help_line, title as ui_title, Binning, HistPlot, Layer, Scale, ACCENTED,
    DIM, HIGHLIGHT, PLAIN,
};

/// Statistics in table order.
const STATS: [&str; 4] = ["nnz", "sum", "mean", "sd"];

/// Rows the table moves on PageUp / PageDown.
const PAGE: usize = 20;

/// Histogram of one statistic: the whole population, and the filtered subset
/// when a filter is on.
struct Hist {
    bins: Binning,
    kmin: i32,
    all: Vec<usize>,
    shown: Option<Vec<usize>>,
    /// The statistic, sorted, for ranks and the summary line.
    sorted: Vec<f32>,
}

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
    /// The other side, once computed.
    parked: Option<Dataset>,
    loader: Option<Loader<'a>>,
    /// A switch is waiting for the other side to be computed.
    loading: bool,
    /// Why the last switch failed, shown in the footer.
    status: Option<String>,
    /// Statistic on the histogram, and the sort key unless sorting by name.
    stat: usize,
    by_name: bool,
    descending: bool,
    filter: String,
    /// Entries passing the filter, in display order.
    view: Vec<usize>,
    table: TableState,
    x_scale: Scale,
    y_scale: Scale,
    hist: Hist,
    mode: Mode,
    quit: bool,
}

impl<'a> StatExplorer<'a> {
    /// Show `data` for `side`; `loader`, if any, computes the other side on
    /// the first Tab.
    pub fn new(title: &str, side: Side, data: Dataset, loader: Option<Loader<'a>>) -> Self {
        let mut explorer = Self {
            title: title.to_string(),
            side,
            view: (0..data.names.len()).collect(),
            names: data.names,
            values: data.values,
            parked: None,
            loader,
            loading: false,
            status: None,
            stat: 0,
            by_name: false,
            descending: true,
            filter: String::new(),
            table: TableState::default().with_selected(Some(0)),
            x_scale: Scale::Log,
            y_scale: Scale::Log,
            hist: Hist {
                bins: Binning::new(Scale::Log, 0.0, true),
                kmin: 0,
                all: Vec::new(),
                shown: None,
                sorted: Vec::new(),
            },
            mode: Mode::Browse,
            quit: false,
        };
        explorer.resort();
        explorer.table.select(Some(0));
        explorer.rebin();
        explorer
    }

    fn selected(&self) -> Option<usize> {
        self.table
            .selected()
            .and_then(|i| self.view.get(i).copied())
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

    fn refilter(&mut self) {
        let re = self.filter_regex();
        self.view = (0..self.names.len())
            .filter(|&i| re.as_ref().is_none_or(|re| re.is_match(&self.names[i])))
            .collect();
        self.resort();
        self.rebin_shown();
    }

    fn resort(&mut self) {
        let keep = self.selected();
        let (names, vals) = (&self.names, &self.values[self.stat]);
        if self.by_name {
            self.view.sort_by(|&a, &b| names[a].cmp(&names[b]));
        } else {
            self.view
                .sort_by(|&a, &b| vals[a].total_cmp(&vals[b]).then(names[a].cmp(&names[b])));
        }
        if self.descending {
            self.view.reverse();
        }
        // Keep the same entry selected when it is still shown.
        let at = keep.and_then(|k| self.view.iter().position(|&i| i == k));
        self.table
            .select(at.or((!self.view.is_empty()).then_some(0)));
    }

    fn rebin(&mut self) {
        let vals = &self.values[self.stat];
        let mut sorted = vals.clone();
        sorted.sort_unstable_by(f32::total_cmp);
        let (min, max) = match (sorted.first(), sorted.last()) {
            (Some(&lo), Some(&hi)) => (lo as f64, hi as f64),
            _ => (0.0, 0.0),
        };
        let bins = Binning::new(self.x_scale, max, self.stat == 0);
        let kmin = bins.key(min);
        let nbins = (bins.key(max) - kmin + 1).max(1) as usize;
        self.hist = Hist {
            bins,
            kmin,
            all: bin_counts(vals.iter().copied(), &bins, kmin, nbins),
            shown: None,
            sorted,
        };
        self.rebin_shown();
    }

    fn rebin_shown(&mut self) {
        let h = &self.hist;
        self.hist.shown = (self.view.len() < self.names.len()).then(|| {
            let vals = &self.values[self.stat];
            bin_counts(
                self.view.iter().map(|&i| vals[i]),
                &h.bins,
                h.kmin,
                h.all.len(),
            )
        });
    }

    /// Sort and plot statistic `s`; pressing the current one flips the order.
    fn choose_stat(&mut self, s: usize) {
        if !self.by_name && self.stat == s {
            self.descending = !self.descending;
        } else {
            self.descending = true;
        }
        self.by_name = false;
        let replot = self.stat != s;
        self.stat = s;
        self.resort();
        if replot {
            self.rebin();
        }
    }

    fn sort_by_name(&mut self) {
        self.descending = self.by_name && !self.descending;
        self.by_name = true;
        self.resort();
    }

    fn can_switch(&self) -> bool {
        self.parked.is_some() || self.loader.is_some()
    }

    /// Show the other side now if it is computed, else ask for it.
    fn request_switch(&mut self) {
        if self.parked.is_some() {
            self.switch();
        } else if self.loader.is_some() {
            self.loading = true;
        }
    }

    /// Compute the other side (blocking), then show it.
    fn finish_loading(&mut self) {
        self.loading = false;
        let Some(loader) = self.loader.as_mut() else {
            return;
        };
        match loader(self.side.other()) {
            Ok(data) => {
                self.parked = Some(data);
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

    /// Swap in the parked side, keeping the sort, filter, and scales.
    fn switch(&mut self) {
        let Some(data) = self.parked.take() else {
            return;
        };
        let names = std::mem::replace(&mut self.names, data.names);
        let values = std::mem::replace(&mut self.values, data.values);
        self.parked = Some(Dataset { names, values });
        self.side = self.side.other();
        self.status = None;
        self.table.select(None);
        *self.table.offset_mut() = 0;
        self.refilter();
        self.table.select((!self.view.is_empty()).then_some(0));
        self.rebin();
    }

    fn step(&mut self, delta: isize) {
        if self.view.is_empty() {
            return;
        }
        let at = self.table.selected().unwrap_or(0) as isize + delta;
        self.table
            .select(Some(at.clamp(0, self.view.len() as isize - 1) as usize));
    }

    fn handle_key(&mut self, key: KeyEvent) {
        if key.kind != KeyEventKind::Press {
            return;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            self.quit = true;
            return;
        }
        match self.mode {
            Mode::Filter => match key.code {
                KeyCode::Enter => self.mode = Mode::Browse,
                KeyCode::Esc => {
                    self.filter.clear();
                    self.refilter();
                    self.mode = Mode::Browse;
                }
                KeyCode::Backspace => {
                    self.filter.pop();
                    self.refilter();
                }
                KeyCode::Char(c) => {
                    self.filter.push(c);
                    self.refilter();
                }
                _ => {}
            },
            Mode::Browse => match key.code {
                KeyCode::Down | KeyCode::Char('j') => self.step(1),
                KeyCode::Up | KeyCode::Char('k') => self.step(-1),
                KeyCode::PageDown => self.step(PAGE as isize),
                KeyCode::PageUp => self.step(-(PAGE as isize)),
                KeyCode::Home | KeyCode::Char('g') => self.step(isize::MIN / 2),
                KeyCode::End | KeyCode::Char('G') => self.step(isize::MAX / 2),
                KeyCode::Char(c @ '1'..='4') => self.choose_stat(c as usize - '1' as usize),
                KeyCode::Char('0' | 'n') => self.sort_by_name(),
                KeyCode::Char('/') => self.mode = Mode::Filter,
                KeyCode::Tab | KeyCode::BackTab => self.request_switch(),
                KeyCode::Char('x') => {
                    self.x_scale = self.x_scale.next();
                    self.rebin();
                }
                KeyCode::Char('y') => self.y_scale = self.y_scale.next(),
                KeyCode::Esc if !self.filter.is_empty() => {
                    self.filter.clear();
                    self.refilter();
                }
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

        let help = match self.mode {
            Mode::Filter => {
                let mut spans = vec![
                    Span::raw(" filter /"),
                    Span::styled(format!("{}▏", self.filter), HIGHLIGHT),
                    Span::raw("  "),
                ];
                spans.extend(help_line(&[("Enter", "keep"), ("Esc", "clear")]).spans);
                Line::from(spans)
            }
            Mode::Browse => {
                let mut keys = vec![
                    ("↑/↓", "move"),
                    ("1-4", "nnz/sum/mean/sd"),
                    ("0", "name"),
                    ("/", "filter"),
                    ("x/y", "scale"),
                ];
                let other = format!("{} (computed on first use)", self.side.other().name());
                if self.can_switch() {
                    keys.push((
                        "Tab",
                        if self.parked.is_some() {
                            self.side.other().name()
                        } else {
                            &other
                        },
                    ));
                }
                keys.push(("q", "quit"));
                let mut line = help_line(&keys);
                if let Some(status) = &self.status {
                    line.spans.push(Span::styled(status.clone(), ACCENTED));
                }
                line
            }
        };
        frame.render_widget(help, footer);
    }

    fn render_table(&mut self, frame: &mut Frame, area: Rect) {
        let arrow = if self.descending { " ▼" } else { " ▲" };
        let head = |i: Option<usize>, name: &str| {
            let sorted = match i {
                Some(s) => !self.by_name && self.stat == s,
                None => self.by_name,
            };
            let text = if sorted {
                format!("{name}{arrow}")
            } else {
                name.to_string()
            };
            let line = Line::from(text);
            let cell = Cell::from(if i.is_some() {
                line.right_aligned()
            } else {
                line
            });
            if sorted {
                cell.style(HIGHLIGHT)
            } else {
                cell.style(DIM)
            }
        };
        let mut header_cells = vec![head(None, "name")];
        header_cells.extend((0..4).map(|s| head(Some(s), STATS[s])));

        // Build only the rows on screen (there can be millions), scrolling
        // just enough to keep the selection in view.
        let height = area.height.saturating_sub(3).max(1) as usize;
        let selected = self.table.selected().unwrap_or(0);
        let mut offset = self.table.offset();
        if selected < offset {
            offset = selected;
        } else if selected >= offset + height {
            offset = selected + 1 - height;
        }
        *self.table.offset_mut() = offset;
        let rows = self.view.iter().skip(offset).take(height).map(|&i| {
            let mut cells = vec![Cell::from(self.names[i].to_string())];
            cells.extend((0..4).map(|s| {
                let text = fmt_value(self.values[s][i]);
                let cell = Cell::from(Line::from(text).right_aligned());
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
            .block(
                Block::bordered()
                    .border_type(BorderType::Rounded)
                    .border_style(DIM)
                    .title(ui_title(title, HIGHLIGHT)),
            );

        // The table sees only the visible slice, so select relative to it.
        let mut state =
            TableState::default().with_selected(self.table.selected().map(|s| s - offset));
        frame.render_stateful_widget(table, area, &mut state);
    }

    fn render_hist(&self, frame: &mut Frame, area: Rect) {
        let stat = STATS[self.stat];
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(DIM)
            .title(ui_title(format!(" {stat} "), HIGHLIGHT));
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let [summary, plot] =
            Layout::vertical([Constraint::Length(2), Constraint::Min(5)]).areas(inner);

        let h = &self.hist;
        let n = h.sorted.len();
        let dim = |t: &str| Span::styled(t.to_string(), DIM);
        let median = match n {
            0 => 0.0,
            _ if n.is_multiple_of(2) => (h.sorted[n / 2 - 1] + h.sorted[n / 2]) / 2.0,
            _ => h.sorted[n / 2],
        };
        let mut lines = vec![Line::from(vec![
            dim("min "),
            Span::raw(fmt_value(h.sorted.first().copied().unwrap_or(0.0))),
            dim("   median "),
            Span::raw(fmt_value(median)),
            dim("   max "),
            Span::raw(fmt_value(h.sorted.last().copied().unwrap_or(0.0))),
        ])];
        let selected = self.selected();
        if let Some(i) = selected {
            let v = self.values[self.stat][i];
            let above = n - h.sorted.partition_point(|&x| x <= v);
            lines.push(Line::from(vec![
                Span::styled(format!("▲ {}", self.names[i]), HIGHLIGHT),
                dim(&format!(" {stat} ")),
                Span::raw(fmt_value(v)),
                dim(&format!("   rank {} of {}", above + 1, n)),
            ]));
        }
        frame.render_widget(Paragraph::new(lines), summary);

        let all_style = |_: i32| if h.shown.is_some() { DIM } else { PLAIN };
        let shown_style = |_: i32| PLAIN;
        let mut layers = vec![Layer {
            counts: &h.all,
            style: &all_style,
        }];
        if let Some(shown) = &h.shown {
            layers.push(Layer {
                counts: shown,
                style: &shown_style,
            });
        }
        let pick = selected.map(|i| h.bins.key(self.values[self.stat][i] as f64));
        HistPlot {
            bins: h.bins,
            kmin: h.kmin,
            layers,
            y_scale: self.y_scale,
            rule: pick.map(|k| (k, ACCENTED)),
            marks: pick.map(|k| (k, "▲", HIGHLIGHT)).into_iter().collect(),
        }
        .render(frame.buffer_mut(), plot);
    }

    fn run(&mut self, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
        while !self.quit {
            terminal.draw(|f| self.render(f))?;
            if let Event::Key(key) = event::read()? {
                self.handle_key(key);
            }
            if self.loading {
                // Compute on the normal screen, where the progress bars
                // belong, then come back.
                ratatui::restore();
                eprintln!("computing {} statistics ...", self.side.other().name());
                self.finish_loading();
                *terminal = ratatui::try_init()?;
            }
        }
        Ok(())
    }
}

/// Whole numbers print as integers; fractions keep three decimals.
fn fmt_value(v: f32) -> String {
    if v.fract() == 0.0 {
        format!("{}", v as i64)
    } else {
        format!("{:.3}", v)
    }
}

/// Run the explorer full screen until the user quits. The terminal is
/// restored on return and on panic.
pub fn explore(mut explorer: StatExplorer<'_>) -> anyhow::Result<()> {
    ratatui::run(|terminal| explorer.run(terminal))
}

#[cfg(test)]
#[path = "tests/stat_tui.rs"]
mod tests;
