//! A list to browse, flat or a tree: the names of a matrix's rows or
//! columns (`row-names`, `column-names`), or the hierarchy of a file
//! (`list-h5`, `list-zarr`).
//!
//! `/` filters (case-insensitive, a regular expression when the text is
//! one), Space marks, `a` marks every entry shown and `u` clears the marks,
//! `w` writes the marked entries (or, with none marked, those shown) to a
//! file, one per line, and `y` copies them (or the entry under the cursor)
//! to the clipboard. In a tree, Enter or →/← open and close a group; an
//! entry's name is then its path from the root.

use legume_numeric::matrix::common_io::write_lines;
use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;
use regex::{Regex, RegexBuilder};

use super::ui::{copy, header, help_line, input_line, title_bar, Screen, Tui, DIM, HIGHLIGHT};

/// One entry: its name, how deep it sits (0 at the top), and a note shown
/// beside it (an array's shape and type, say).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Item {
    pub name: String,
    pub depth: usize,
    pub detail: String,
}

impl Item {
    pub fn new(name: impl Into<String>, depth: usize, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            depth,
            detail: detail.into(),
        }
    }
}

/// What the footer is doing.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Mode {
    Browse,
    Filter,
    /// Asking where to write, with the text so far.
    Write(String),
}

/// The browser, independent of the terminal so it can be tested.
pub struct ListView {
    badge: &'static str,
    title: String,
    items: Vec<Item>,
    /// Each item's path from the root, `/`-joined: what is written.
    paths: Vec<String>,
    /// Each item's parent, and whether it has children.
    parent: Vec<Option<usize>>,
    has_children: Vec<bool>,
    folded: Vec<bool>,
    marked: Vec<bool>,
    filter: String,
    /// Items on screen, in order.
    view: Vec<usize>,
    cursor: usize,
    offset: usize,
    /// Rows the list had when last drawn, for paging.
    page: usize,
    /// Lines shown under the list (a summary the command printed).
    notes: Vec<String>,
    /// Where `w` writes unless told otherwise.
    write_to: String,
    mode: Mode,
    status: Option<String>,
    quit: bool,
    session_keys: Vec<(String, String)>,
}

impl ListView {
    /// `items` in display order, a child right after its parent at one
    /// level deeper. `badge` names the view; `write_to` is where `w` offers
    /// to write.
    pub fn new(badge: &'static str, title: &str, items: Vec<Item>, write_to: &str) -> Self {
        let n = items.len();
        let mut parent = vec![None; n];
        let mut has_children = vec![false; n];
        let mut paths = Vec::with_capacity(n);
        // The open ancestors of the current item, by depth.
        let mut stack: Vec<usize> = Vec::new();
        for (i, it) in items.iter().enumerate() {
            stack.truncate(it.depth);
            if let Some(&p) = stack.last() {
                parent[i] = Some(p);
                has_children[p] = true;
            }
            paths.push(match parent[i] {
                Some(p) => format!("{}/{}", paths[p], it.name.trim_end_matches('/')),
                None => it.name.trim_end_matches('/').to_string(),
            });
            stack.push(i);
        }
        let mut view = Self {
            badge,
            title: title.to_string(),
            items,
            paths,
            parent,
            has_children,
            folded: vec![false; n],
            marked: vec![false; n],
            filter: String::new(),
            view: Vec::new(),
            cursor: 0,
            offset: 0,
            page: 20,
            notes: Vec::new(),
            write_to: write_to.to_string(),
            mode: Mode::Browse,
            status: None,
            quit: false,
            session_keys: super::session_keys(),
        };
        view.refilter();
        view
    }

    /// Flat names, one entry each.
    pub fn names(badge: &'static str, title: &str, names: &[Box<str>], write_to: &str) -> Self {
        let items = names.iter().map(|n| Item::new(&**n, 0, "")).collect();
        Self::new(badge, title, items, write_to)
    }

    /// Lines to show under the list.
    pub fn with_notes(mut self, notes: Vec<String>) -> Self {
        self.notes = notes;
        self
    }

    fn is_tree(&self) -> bool {
        self.has_children.iter().any(|&c| c)
    }

    fn pattern(&self) -> Option<Regex> {
        if self.filter.is_empty() {
            return None;
        }
        RegexBuilder::new(&self.filter)
            .case_insensitive(true)
            .build()
            .or_else(|_| {
                RegexBuilder::new(&regex::escape(&self.filter))
                    .case_insensitive(true)
                    .build()
            })
            .ok()
    }

    /// Recompute the items shown: under a filter, the matches and their
    /// ancestors, folds aside; else every item no folded group hides.
    fn refilter(&mut self) {
        let keep = self.view.get(self.cursor).copied();
        self.view = match self.pattern() {
            Some(re) => {
                let mut shown = vec![false; self.items.len()];
                for i in 0..self.items.len() {
                    if re.is_match(&self.items[i].name) {
                        let mut at = Some(i);
                        while let Some(j) = at.filter(|&j| !shown[j]) {
                            shown[j] = true;
                            at = self.parent[j];
                        }
                    }
                }
                (0..self.items.len()).filter(|&i| shown[i]).collect()
            }
            None => (0..self.items.len())
                .filter(|&i| !self.hidden_by_fold(i))
                .collect(),
        };
        self.cursor = keep
            .and_then(|k| self.view.iter().position(|&i| i == k))
            .unwrap_or(0);
        self.offset = self.offset.min(self.cursor);
    }

    fn hidden_by_fold(&self, i: usize) -> bool {
        let mut at = self.parent[i];
        while let Some(p) = at {
            if self.folded[p] {
                return true;
            }
            at = self.parent[p];
        }
        false
    }

    fn selected(&self) -> Option<usize> {
        self.view.get(self.cursor).copied()
    }

    fn move_by(&mut self, delta: isize) {
        let last = self.view.len().saturating_sub(1) as isize;
        self.cursor = (self.cursor as isize + delta).clamp(0, last) as usize;
    }

    fn marked_count(&self) -> usize {
        self.marked.iter().filter(|&&m| m).count()
    }

    /// What `w` and `y` take: the marked entries, else those shown.
    fn chosen(&self) -> Vec<Box<str>> {
        let marked: Vec<usize> = (0..self.items.len()).filter(|&i| self.marked[i]).collect();
        let from = if marked.is_empty() {
            &self.view
        } else {
            &marked
        };
        from.iter()
            .map(|&i| self.paths[i].as_str().into())
            .collect()
    }

    fn set_fold(&mut self, folded: bool) {
        if let Some(i) = self.selected().filter(|&i| self.has_children[i]) {
            if self.folded[i] != folded {
                self.folded[i] = folded;
                self.refilter();
            }
        } else if folded {
            // ← on a leaf goes to its group.
            if let Some(p) = self.selected().and_then(|i| self.parent[i]) {
                if let Some(at) = self.view.iter().position(|&j| j == p) {
                    self.cursor = at;
                }
            }
        }
    }

    fn write(&mut self, path: &str) {
        let lines = self.chosen();
        self.status = Some(match write_lines(&lines, path) {
            Ok(()) => format!("wrote {} entries to {path}", lines.len()),
            Err(e) => format!("not written: {e}"),
        });
    }

    fn browse_key(&mut self, key: KeyEvent) {
        self.status = None;
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => self.quit = true,
            KeyCode::Down | KeyCode::Char('j') => self.move_by(1),
            KeyCode::Up | KeyCode::Char('k') => self.move_by(-1),
            KeyCode::PageDown => self.move_by(self.page as isize),
            KeyCode::PageUp => self.move_by(-(self.page as isize)),
            KeyCode::Home | KeyCode::Char('g') => self.cursor = 0,
            KeyCode::End | KeyCode::Char('G') => self.move_by(isize::MAX / 2),
            KeyCode::Char('/') => self.mode = Mode::Filter,
            KeyCode::Char(' ') => {
                if let Some(i) = self.selected() {
                    self.marked[i] = !self.marked[i];
                    self.move_by(1);
                }
            }
            KeyCode::Char('a') => {
                for &i in &self.view {
                    self.marked[i] = true;
                }
            }
            KeyCode::Char('u') => self.marked.fill(false),
            KeyCode::Enter => {
                if let Some(i) = self.selected().filter(|&i| self.has_children[i]) {
                    self.folded[i] = !self.folded[i];
                    self.refilter();
                }
            }
            KeyCode::Right | KeyCode::Char('l') => self.set_fold(false),
            KeyCode::Left | KeyCode::Char('h') => self.set_fold(true),
            KeyCode::Char('w') => self.mode = Mode::Write(self.write_to.clone()),
            KeyCode::Char('y') => {
                let text = if self.marked_count() > 0 {
                    self.chosen().join("\n")
                } else {
                    self.selected()
                        .map(|i| self.paths[i].clone())
                        .unwrap_or_default()
                };
                let n = text.lines().count();
                copy(&text);
                self.status = Some(format!("copied {n} to the clipboard"));
            }
            _ => {}
        }
    }

    fn filter_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Enter => self.mode = Mode::Browse,
            KeyCode::Esc => {
                self.filter.clear();
                self.mode = Mode::Browse;
                self.refilter();
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
        }
    }

    fn write_key(&mut self, key: KeyEvent) {
        let Mode::Write(text) = &mut self.mode else {
            return;
        };
        match key.code {
            KeyCode::Esc => self.mode = Mode::Browse,
            KeyCode::Enter => {
                let path = text.trim().to_string();
                self.mode = Mode::Browse;
                if !path.is_empty() {
                    self.write_to.clone_from(&path);
                    self.write(&path);
                }
            }
            KeyCode::Backspace => {
                text.pop();
            }
            KeyCode::Char(c) => text.push(c),
            _ => {}
        }
    }

    fn row(&self, i: usize, at_cursor: bool) -> Line<'static> {
        let it = &self.items[i];
        let mark = if self.marked[i] { "● " } else { "  " };
        // A filter shows what matches inside a closed group too.
        let closed = self.folded[i] && self.filter.is_empty();
        let fold = match (self.has_children[i], closed) {
            (true, true) => "▸ ",
            (true, false) => "▾ ",
            (false, _) if self.is_tree() => "  ",
            _ => "",
        };
        let name_style = if at_cursor {
            HIGHLIGHT.add_modifier(Modifier::REVERSED)
        } else if self.marked[i] {
            HIGHLIGHT
        } else {
            ratatui::style::Style::new()
        };
        let mut spans = vec![
            Span::styled(mark, HIGHLIGHT),
            Span::raw(format!("{}{fold}", "  ".repeat(it.depth))),
            Span::styled(it.name.clone(), name_style),
        ];
        if !it.detail.is_empty() {
            spans.push(Span::styled(format!("  {}", it.detail), DIM));
        }
        Line::from(spans)
    }

    fn footer(&self) -> Line<'static> {
        match &self.mode {
            Mode::Filter => input_line(
                "filter: ",
                &self.filter,
                &[("enter", "keep"), ("esc", "clear")],
            ),
            Mode::Write(text) => {
                input_line("write to: ", text, &[("enter", "write"), ("esc", "cancel")])
            }
            Mode::Browse => match &self.status {
                Some(s) => Line::from(Span::styled(format!(" {s}"), HIGHLIGHT)),
                None => {
                    let mut keys = vec![("/", "filter"), ("space", "mark"), ("a", "all")];
                    keys.push(("u", "unmark"));
                    if self.is_tree() {
                        keys.push(("enter", "fold"));
                    }
                    keys.extend([("w", "write"), ("y", "copy"), ("q", "quit")]);
                    help_line(&keys)
                }
            },
        }
    }
}

impl Screen for ListView {
    fn render(&mut self, frame: &mut Frame) {
        let notes = self.notes.len().min(8) as u16;
        let [top, list, below, foot] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Fill(1),
            Constraint::Length(if notes > 0 { notes + 1 } else { 0 }),
            Constraint::Length(1),
        ])
        .areas(frame.area());
        let mut extra = format!("{} entries", self.items.len());
        if self.view.len() < self.items.len() {
            extra.push_str(&format!(" · {} shown", self.view.len()));
        }
        let marked = self.marked_count();
        if marked > 0 {
            extra.push_str(&format!(" · {marked} marked"));
        }
        if !self.filter.is_empty() {
            extra.push_str(&format!(" · /{}", self.filter));
        }
        title_bar(
            frame,
            top,
            header(self.badge, &self.title, &extra),
            &self.session_keys,
        );
        self.page = list.height.max(1) as usize;
        if self.cursor < self.offset {
            self.offset = self.cursor;
        } else if self.cursor >= self.offset + self.page {
            self.offset = self.cursor + 1 - self.page;
        }
        let lines: Vec<Line> = self
            .view
            .iter()
            .enumerate()
            .skip(self.offset)
            .take(self.page)
            .map(|(k, &i)| self.row(i, k == self.cursor))
            .collect();
        let lines = if lines.is_empty() {
            vec![Line::from(Span::styled("  nothing matches", DIM))]
        } else {
            lines
        };
        frame.render_widget(Paragraph::new(lines), list);
        if notes > 0 {
            let lines: Vec<Line> = std::iter::once(Line::from(""))
                .chain(
                    self.notes
                        .iter()
                        .take(8)
                        .map(|n| Line::from(Span::styled(format!(" {n}"), DIM))),
                )
                .collect();
            frame.render_widget(Paragraph::new(lines), below);
        }
        frame.render_widget(self.footer(), foot);
    }

    fn handle_key(&mut self, key: KeyEvent) {
        match self.mode {
            Mode::Browse => self.browse_key(key),
            Mode::Filter => self.filter_key(key),
            Mode::Write(_) => self.write_key(key),
        }
    }

    fn interrupt(&mut self) {
        self.quit = true;
    }

    fn done(&self) -> bool {
        self.quit
    }

    fn status(&self) -> Option<String> {
        Some(format!("browsing {}", self.title))
    }
}

/// Browse `view` on the screen of `tui` until `q`.
pub fn browse_in(tui: &mut Tui<'_>, mut view: ListView) -> anyhow::Result<()> {
    tui.run(&mut view)
}

#[cfg(test)]
#[path = "tests/list_tui.rs"]
mod tests;
