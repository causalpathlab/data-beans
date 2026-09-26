//! Shared pieces of the full-screen views: the palette, panels, header and
//! help lines, the event loop, histogram scales and binning, and a histogram
//! plot with a y gutter, an x axis, and markers.

use ratatui::buffer::Buffer;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType};
use ratatui::Frame;

use crate::qc::log_bin_key;

// Palette: the terminal's own foreground (so light and dark backgrounds both
// work) for nearly everything, and one accent for what needs the eye: what a
// cutoff drops, the selection, and key hints.
pub const ACCENT: Color = Color::Rgb(217, 119, 87);

/// Plain text and bars in the terminal's foreground.
pub const PLAIN: Style = Style::new();
/// Secondary text: labels, axes, units.
pub const DIM: Style = Style::new().add_modifier(Modifier::DIM);
/// Accent bars and marks.
pub const ACCENTED: Style = Style::new().fg(ACCENT);
/// Key names in help lines, typed values, and the value in focus.
pub const HIGHLIGHT: Style = Style::new().fg(ACCENT).add_modifier(Modifier::BOLD);

/// A full-screen view driven by [`run_screen`].
pub trait Screen {
    fn render(&mut self, frame: &mut Frame);
    /// A key press (Ctrl-C goes to [`Screen::interrupt`] instead).
    fn handle_key(&mut self, key: KeyEvent);
    fn interrupt(&mut self);
    fn done(&self) -> bool;
    /// Blocking work the last key asked for, as a line to print while it
    /// runs; [`Screen::do_work`] then does it.
    fn pending_work(&self) -> Option<String> {
        None
    }
    fn do_work(&mut self) {}
}

/// Run `screen` full screen until it is done. The terminal is restored on
/// return and on panic. Blocking work runs on the normal screen, where its
/// own progress output belongs, and the view comes back after it.
pub fn run_screen(screen: &mut impl Screen) -> anyhow::Result<()> {
    ratatui::run(|terminal| -> anyhow::Result<()> {
        while !screen.done() {
            terminal.draw(|f| screen.render(f))?;
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                    screen.interrupt();
                } else {
                    screen.handle_key(key);
                }
            }
            if let Some(message) = screen.pending_work() {
                ratatui::restore();
                eprintln!("{message}");
                screen.do_work();
                *terminal = ratatui::try_init()?;
            }
        }
        Ok(())
    })
}

/// Title bar: a reverse-video badge naming the view, then plain text.
pub fn header(badge: &str, title: &str, extra: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!(" {badge} "),
            HIGHLIGHT.add_modifier(Modifier::REVERSED),
        ),
        Span::raw(format!(" {title}")),
        Span::styled(format!("   {extra}"), DIM),
    ])
}

/// Rounded panel titled `title`: plain border and accent title when
/// `focused`, dim border otherwise.
pub fn panel(title: String, focused: bool) -> Block<'static> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(if focused { PLAIN } else { DIM })
        // Titles inherit the border style; start clear of it.
        .title(Line::from(title).style(Style::reset().patch(HIGHLIGHT)))
}

/// Help line from `(key, what it does)` pairs.
pub fn help_line(pairs: &[(&str, &str)]) -> Line<'static> {
    let mut spans = vec![Span::raw(" ")];
    for (key, what) in pairs {
        spans.push(Span::styled(key.to_string(), HIGHLIGHT));
        spans.push(Span::styled(format!(" {what}  "), DIM));
    }
    Line::from(spans)
}

/// Footer while typing: `prompt`, the text so far with a cursor, then keys.
pub fn input_line(prompt: &str, text: &str, keys: &[(&str, &str)]) -> Line<'static> {
    let mut spans = vec![
        Span::raw(format!(" {prompt}")),
        Span::styled(format!("{text}▏"), HIGHLIGHT),
        Span::raw("  "),
    ];
    spans.extend(help_line(keys).spans);
    Line::from(spans)
}

/// Set a cell outright, rather than layering `style` over what was there.
fn put(buf: &mut Buffer, x: u16, y: u16, symbol: &str, style: Style) {
    buf[(x, y)]
        .set_symbol(symbol)
        .set_style(Style::reset().patch(style));
}

/// Width of the y-axis gutter left of each histogram.
const GUTTER: u16 = 6;

/// Bins on the sqrt and linear scales (the log scale uses tenth-decade bins).
const TARGET_BINS: f64 = 50.0;

/// How a histogram axis is drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scale {
    Log,
    Sqrt,
    Linear,
}

impl Scale {
    pub fn next(self) -> Self {
        match self {
            Scale::Log => Scale::Sqrt,
            Scale::Sqrt => Scale::Linear,
            Scale::Linear => Scale::Log,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Scale::Log => "log",
            Scale::Sqrt => "sqrt",
            Scale::Linear => "linear",
        }
    }

    fn apply(self, v: f64) -> f64 {
        match self {
            Scale::Log => (v + 1.0).log10(),
            Scale::Sqrt => v.max(0.0).sqrt(),
            Scale::Linear => v,
        }
    }

    fn invert(self, t: f64) -> f64 {
        match self {
            Scale::Log => 10f64.powf(t) - 1.0,
            Scale::Sqrt => t * t,
            Scale::Linear => t,
        }
    }
}

/// Equal-width bins on a scale, keyed by integers. On the log scale these are
/// the printed histogram's tenth-decade bins, keyed by rounding.
#[derive(Debug, Clone, Copy)]
pub struct Binning {
    pub scale: Scale,
    /// Bin width on the scaled axis.
    width: f64,
}

impl Binning {
    /// Bins spanning `0..=max`. Whole counts (`integer`) get linear bins at
    /// least one count wide, so no bin falls between two integers.
    pub fn new(scale: Scale, max: f64, integer: bool) -> Self {
        let span = if integer { max + 1.0 } else { max };
        let width = match scale {
            Scale::Log => 0.1,
            Scale::Linear if integer => (span / TARGET_BINS).ceil().max(1.0),
            _ => scale.apply(span) / TARGET_BINS,
        };
        Self {
            scale,
            width: width.max(f64::MIN_POSITIVE),
        }
    }

    pub fn key(&self, x: f64) -> i32 {
        match self.scale {
            Scale::Log => log_bin_key(x),
            _ => (self.scale.apply(x) / self.width).floor() as i32,
        }
    }

    /// Scaled position where bin `k` starts: log keys round, so their bins
    /// start half a bin early.
    fn start(&self, k: i32) -> f64 {
        match self.scale {
            Scale::Log => (k as f64 - 0.5) * self.width,
            _ => k as f64 * self.width,
        }
    }

    /// Smallest whole count in bin `k` or above: the cutoff that drops every
    /// bin left of `k`.
    pub fn lower_edge(&self, k: i32) -> usize {
        if k <= 0 {
            return 0;
        }
        // Start from the exact inverse, then settle rounding either way.
        let mut x = self.scale.invert(self.start(k)).ceil().max(0.0) as usize;
        while self.key(x as f64) < k {
            x += 1;
        }
        while x > 0 && self.key((x - 1) as f64) >= k {
            x -= 1;
        }
        x
    }

    /// Label for the tick at bin `k` (the bin centre on the log scale, as the
    /// printed histogram labels it; the bin start otherwise).
    fn tick_value(&self, k: i32) -> f64 {
        self.scale.invert(k as f64 * self.width)
    }

    /// Ticks every half decade on the log scale, about six otherwise.
    fn tick_every(&self, nbins: usize) -> i32 {
        match self.scale {
            Scale::Log => 5,
            _ => (nbins as i32 / 6).max(1),
        }
    }
}

/// A sorted statistic binned on a scale: the bins and their counts.
pub struct Binned {
    pub bins: Binning,
    pub kmin: i32,
    pub counts: Vec<usize>,
}

impl Binned {
    /// Bin `sorted` (ascending). All-whole data gets whole-count bins.
    pub fn new(sorted: &[f32], scale: Scale) -> Self {
        let (min, max) = match (sorted.first(), sorted.last()) {
            (Some(&lo), Some(&hi)) => (lo as f64, hi as f64),
            _ => (0.0, 0.0),
        };
        let integer = sorted.iter().all(|v| v.fract() == 0.0);
        let bins = Binning::new(scale, max, integer);
        let kmin = bins.key(min);
        let nbins = (bins.key(max) - kmin + 1).max(1) as usize;
        let counts = count(&bins, kmin, nbins, sorted.iter().copied());
        Self { bins, kmin, counts }
    }

    /// Counts of `values` in these bins.
    pub fn count(&self, values: impl Iterator<Item = f32>) -> Vec<usize> {
        count(&self.bins, self.kmin, self.counts.len(), values)
    }

    pub fn kmax(&self) -> i32 {
        self.kmin + self.counts.len() as i32 - 1
    }
}

/// Counts of `values` per bin, from `kmin` to `kmin + nbins - 1` (values
/// outside land in the end bins).
fn count(bins: &Binning, kmin: i32, nbins: usize, values: impl Iterator<Item = f32>) -> Vec<usize> {
    let mut counts = vec![0; nbins];
    for v in values {
        let i = (bins.key(v as f64) - kmin).clamp(0, nbins as i32 - 1);
        counts[i as usize] += 1;
    }
    counts
}

/// Median of an ascending slice (0 when empty).
pub fn median(sorted: &[f32]) -> f32 {
    crate::qc::median_of_sorted(sorted)
}

/// Compact number for axis labels: 950, 1.2k, 35k, 1.1M; small fractions
/// keep two significant digits.
pub fn compact(v: f64) -> String {
    if v != 0.0 && v.abs() < 10.0 && v.fract() != 0.0 {
        format!("{:.2}", v)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    } else if v < 1e3 {
        format!("{}", v.round() as i64)
    } else if v < 1e4 {
        format!("{:.1}k", v / 1e3)
    } else if v < 1e6 {
        format!("{}k", (v / 1e3).round() as u64)
    } else if v < 1e9 {
        format!("{:.1}M", v / 1e6)
    } else {
        format!("{:.1}G", v / 1e9)
    }
}

/// A histogram of `counts` over bins `kmin..`, scaled to them.
pub struct HistPlot<'a> {
    pub bins: Binning,
    pub kmin: i32,
    pub counts: &'a [usize],
    /// Style of each bin's bar, by key.
    pub style: &'a dyn Fn(i32) -> Style,
    /// A subset drawn in front, in the bar style; `counts` then draw dimmed
    /// behind it.
    pub subset: Option<&'a [usize]>,
    pub y_scale: Scale,
    /// Bin under the accent rule and ▲.
    pub pointer: Option<i32>,
    /// Other symbols on the x axis, by key.
    pub marks: Vec<(i32, &'static str, Style)>,
}

const EIGHTHS: [&str; 8] = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];

impl HistPlot<'_> {
    /// Draw into `area`: bars over the rows above the last two, which hold
    /// the x axis and its labels; the left [`GUTTER`] columns hold the y axis.
    pub fn render(&self, buf: &mut Buffer, area: Rect) {
        let [plot, axis, labels] = Layout::vertical([
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .areas(area);
        let [gutter, chart] =
            Layout::horizontal([Constraint::Length(GUTTER), Constraint::Min(1)]).areas(plot);
        if chart.width == 0 || chart.height == 0 {
            return;
        }
        let nbins = self.counts.len();
        let bw = (chart.width / nbins.max(1) as u16).clamp(1, 4);
        let x_of = |k: i32| -> Option<u16> {
            let i = k - self.kmin;
            (i >= 0 && (i as usize) < nbins)
                .then(|| chart.x + i as u16 * bw)
                .filter(|&x| x < chart.right())
        };

        let height = |c: usize| self.y_scale.apply(c as f64);
        let max_h = self.counts.iter().map(|&c| height(c)).fold(0.0, f64::max);
        let cells = chart.height as usize * 8;
        let eighths = |c: usize| {
            if c == 0 || max_h <= 0.0 {
                0
            } else {
                ((height(c) / max_h * cells as f64).round() as usize).clamp(1, cells)
            }
        };

        if let Some(x) = self.pointer.and_then(x_of) {
            for y in chart.top()..chart.bottom() {
                put(buf, x, y, "┊", ACCENTED);
            }
        }

        let mut bars = |counts: &[usize], behind: Option<&[usize]>, dim: bool| {
            for (i, &c) in counts.iter().enumerate() {
                let x0 = chart.x + i as u16 * bw;
                if x0 >= chart.right() {
                    break;
                }
                let style = if dim {
                    DIM
                } else {
                    (self.style)(self.kmin + i as i32)
                };
                let (top, under) = (eighths(c), behind.map_or(0, |b| eighths(b[i])));
                for (j, y) in (chart.top()..chart.bottom()).rev().enumerate() {
                    let mut fill = top.saturating_sub(j * 8).min(8);
                    if fill == 0 {
                        break;
                    }
                    // A partial top in front of a taller bar would show a gap
                    // above it (the terminal's foreground cannot be a
                    // background), so it rounds up to a whole cell.
                    if under >= (j + 1) * 8 {
                        fill = 8;
                    }
                    for x in x0..(x0 + bw).min(chart.right()) {
                        put(buf, x, y, EIGHTHS[fill - 1], style);
                    }
                }
            }
        };
        bars(self.counts, None, self.subset.is_some());
        if let Some(subset) = self.subset {
            bars(subset, Some(self.counts), false);
        }

        // y axis: count at the top and at half height on the y scale.
        let gx = gutter.right() - 1;
        for y in gutter.top()..gutter.bottom() {
            put(buf, gx, y, "│", DIM);
        }
        let mut ylabel = |y: u16, v: f64| {
            let s = compact(v);
            let x = gx.saturating_sub(1 + s.len() as u16).max(gutter.x);
            buf.set_string(x, y, &s, DIM);
            put(buf, gx, y, "┤", DIM);
        };
        if max_h > 0.0 {
            ylabel(gutter.top(), self.y_scale.invert(max_h));
            if gutter.height >= 6 {
                ylabel(
                    gutter.top() + gutter.height / 2,
                    self.y_scale.invert(max_h / 2.0),
                );
            }
        }

        // x axis: baseline with ticks, labels below, then the marks.
        for x in axis.left()..axis.right() {
            let sym = match x.cmp(&gx) {
                std::cmp::Ordering::Less => " ",
                std::cmp::Ordering::Equal => "└",
                std::cmp::Ordering::Greater => "─",
            };
            put(buf, x, axis.y, sym, DIM);
        }
        let every = self.bins.tick_every(nbins);
        let mut next_free = labels.x;
        let kmax = self.kmin + nbins as i32 - 1;
        for k in (self.kmin..=kmax).filter(|k| k % every == 0) {
            let Some(x) = x_of(k) else { continue };
            put(buf, x, axis.y, "┴", DIM);
            let s = compact(self.bins.tick_value(k));
            if x >= next_free && x + (s.len() as u16) <= labels.right() {
                buf.set_string(x, labels.y, &s, DIM);
                next_free = x + s.len() as u16 + 1;
            }
        }
        let pointer = self.pointer.map(|k| (k, "▲", HIGHLIGHT));
        for &(k, sym, style) in self.marks.iter().chain(pointer.iter()) {
            if let Some(x) = x_of(k) {
                put(buf, x, axis.y, sym, style);
            }
        }
    }
}

#[cfg(test)]
#[path = "tests/ui.rs"]
mod tests;
