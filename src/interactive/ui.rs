//! Shared pieces of the full-screen views: the palette, the header and help
//! lines, histogram scales and binning, and a histogram plot with a y gutter,
//! an x axis, and markers.

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use crate::qc::{log_bin_key, log_bin_lower_edge};

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

/// Panel title in `style`, clear of the (often dim) border style it sits on.
pub fn title(text: String, style: Style) -> Line<'static> {
    Line::from(text).style(Style::reset().patch(style))
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
/// the printed histogram's tenth-decade bins.
#[derive(Debug, Clone, Copy)]
pub struct Binning {
    pub scale: Scale,
    /// Bin width on the scaled axis (sqrt and linear only).
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

    /// Smallest whole count in bin `k` or above: the cutoff that drops every
    /// bin left of `k`.
    pub fn lower_edge(&self, k: i32) -> usize {
        if self.scale == Scale::Log {
            return log_bin_lower_edge(k);
        }
        if k <= 0 {
            return 0;
        }
        // Start from the exact inverse, then settle rounding either way.
        let mut x = self.scale.invert(k as f64 * self.width).ceil().max(0.0) as usize;
        while self.key(x as f64) < k {
            x += 1;
        }
        while x > 0 && self.key((x - 1) as f64) >= k {
            x -= 1;
        }
        x
    }

    /// Label for the tick at bin `k`.
    fn tick_value(&self, k: i32) -> f64 {
        match self.scale {
            // Bin centre, as the printed histogram labels it.
            Scale::Log => 10f64.powf(k as f64 / 10.0) - 1.0,
            _ => self.scale.invert(k as f64 * self.width),
        }
    }

    /// Ticks every half decade on the log scale, about six otherwise.
    fn tick_every(&self, nbins: usize) -> i32 {
        match self.scale {
            Scale::Log => 5,
            _ => (nbins as i32 / 6).max(1),
        }
    }
}

/// Counts of `values` per bin key, from `kmin` to `kmin + nbins - 1`.
pub fn bin_counts(
    values: impl Iterator<Item = f32>,
    bins: &Binning,
    kmin: i32,
    nbins: usize,
) -> Vec<usize> {
    let mut counts = vec![0; nbins];
    for v in values {
        let i = (bins.key(v as f64) - kmin).clamp(0, nbins as i32 - 1);
        counts[i as usize] += 1;
    }
    counts
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

/// One series of bars. Later layers draw over earlier ones, so a subset can
/// sit in front of the whole distribution.
pub struct Layer<'a> {
    pub counts: &'a [usize],
    pub style: &'a dyn Fn(i32) -> Style,
}

/// A histogram over bins `kmin..`, scaled to the first layer.
pub struct HistPlot<'a> {
    pub bins: Binning,
    pub kmin: i32,
    pub layers: Vec<Layer<'a>>,
    pub y_scale: Scale,
    /// Dotted vertical rule at a bin, drawn behind the bars.
    pub rule: Option<(i32, Style)>,
    /// Symbols on the x axis at a bin.
    pub marks: Vec<(i32, &'static str, Style)>,
}

const EIGHTHS: [&str; 8] = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];

impl HistPlot<'_> {
    fn nbins(&self) -> usize {
        self.layers.first().map_or(0, |l| l.counts.len())
    }

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
        let nbins = self.nbins().max(1);
        let bw = (chart.width / nbins as u16).clamp(1, 4);
        let x_of = |k: i32| -> Option<u16> {
            let i = k - self.kmin;
            (i >= 0 && (i as usize) < self.nbins())
                .then(|| chart.x + i as u16 * bw)
                .filter(|&x| x < chart.right())
        };

        let height = |c: usize| self.y_scale.apply(c as f64);
        let max_h = self.layers.first().map_or(0.0, |l| {
            l.counts.iter().map(|&c| height(c)).fold(0.0, f64::max)
        });

        if let Some((x, style)) = self.rule.and_then(|(k, s)| Some((x_of(k)?, s))) {
            for y in chart.top()..chart.bottom() {
                buf[(x, y)].set_symbol("┊").set_style(style);
            }
        }

        let cells = chart.height as usize * 8;
        let mut below: Vec<(usize, Style)> = vec![(0, PLAIN); self.nbins()];
        for layer in &self.layers {
            for (i, &c) in layer.counts.iter().enumerate() {
                let x0 = chart.x + i as u16 * bw;
                if x0 >= chart.right() {
                    break;
                }
                let style = (layer.style)(self.kmin + i as i32);
                let eighths = if c == 0 || max_h <= 0.0 {
                    0
                } else {
                    ((height(c) / max_h * cells as f64).round() as usize).clamp(1, cells)
                };
                for (j, y) in (chart.top()..chart.bottom()).rev().enumerate() {
                    let mut fill = eighths.saturating_sub(j * 8).min(8);
                    if fill == 0 {
                        break;
                    }
                    // A partial top over an earlier layer shows that layer
                    // behind it as the background. The terminal's own
                    // foreground cannot be a background, so over such a
                    // layer the top rounds up to a whole cell instead.
                    let (under, under_style) = below[i];
                    let mut bg = Color::Reset;
                    if fill < 8 && under >= (j + 1) * 8 {
                        match under_style.fg {
                            Some(c) => bg = c,
                            None => fill = 8,
                        }
                    }
                    for x in x0..(x0 + bw).min(chart.right()) {
                        buf[(x, y)]
                            .set_symbol(EIGHTHS[fill - 1])
                            .set_style(Style::reset().patch(style).bg(bg));
                    }
                }
                below[i] = (eighths, style);
            }
        }

        // y axis: count at the top and at half height on the y scale.
        let dim = DIM;
        let gx = gutter.right() - 1;
        for y in gutter.top()..gutter.bottom() {
            buf[(gx, y)].set_symbol("│").set_style(dim);
        }
        let mut ylabel = |y: u16, v: f64| {
            let s = compact(v);
            let x = gx.saturating_sub(1 + s.len() as u16).max(gutter.x);
            buf.set_string(x, y, &s, dim);
            buf[(gx, y)].set_symbol("┤");
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
            buf[(x, axis.y)].set_symbol(sym).set_style(dim);
        }
        let every = self.bins.tick_every(self.nbins());
        let mut next_free = labels.x;
        let kmax = self.kmin + self.nbins() as i32 - 1;
        for k in (self.kmin..=kmax).filter(|k| k % every == 0) {
            let Some(x) = x_of(k) else { continue };
            buf[(x, axis.y)].set_symbol("┴");
            let s = compact(self.bins.tick_value(k));
            if x >= next_free && x + (s.len() as u16) <= labels.right() {
                buf.set_string(x, labels.y, &s, dim);
                next_free = x + s.len() as u16 + 1;
            }
        }
        for &(k, sym, style) in &self.marks {
            if let Some(x) = x_of(k) {
                // Replace the baseline's dim style rather than adding to it.
                buf[(x, axis.y)]
                    .set_symbol(sym)
                    .set_style(Style::reset().patch(style));
            }
        }
    }
}

#[cfg(test)]
#[path = "tests/ui.rs"]
mod tests;
