//! Shared pieces of the full-screen views: the palette, panels, header and
//! help lines, the event loop, histogram scales and binning, and a histogram
//! plot with a y gutter, an x axis, and markers.

use ratatui::buffer::Buffer;
use ratatui::crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers, KeyboardEnhancementFlags, MouseButton, MouseEvent, MouseEventKind,
    PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
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
    /// Called about every [`TICK`] while no key comes: true redraws, for a
    /// view waiting on work in the background.
    fn tick(&mut self) -> bool {
        false
    }
    /// Whether the view takes the mouse: only then is the terminal asked for
    /// it, since a view holding it keeps text from being selected.
    fn takes_mouse(&self) -> bool {
        false
    }
    /// A left click or a turn of the wheel, at a cell of the screen.
    fn mouse(&mut self, _event: MouseEvent) {}
    /// Whether the terminal is asked to report keys held with Shift, Alt or
    /// Ctrl apart (the kitty keyboard protocol; others ignore the request),
    /// so that Shift+Enter, say, is not taken for Enter.
    fn reports_chords(&self) -> bool {
        false
    }
    /// Something to tell whoever is away, once: [`run_screen`] rings the
    /// terminal's bell and asks it for a desktop notification saying it.
    fn take_notice(&mut self) -> Option<String> {
        None
    }
}

/// The terminal modes a screen asked for, taken back while its screen is
/// still up: terminals keep each screen's modes apart.
#[derive(Default)]
struct Modes {
    mouse: bool,
    chords: bool,
}

impl Modes {
    fn ask(&mut self, screen: &impl Screen) {
        let mut out = std::io::stdout();
        if screen.takes_mouse() && !self.mouse {
            self.mouse = ratatui::crossterm::execute!(out, EnableMouseCapture).is_ok();
        }
        if screen.reports_chords() && !self.chords {
            let flags = KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES;
            self.chords =
                ratatui::crossterm::execute!(out, PushKeyboardEnhancementFlags(flags)).is_ok();
        }
    }

    fn release(&mut self) {
        let mut out = std::io::stdout();
        if std::mem::take(&mut self.mouse) {
            let _ = ratatui::crossterm::execute!(out, DisableMouseCapture);
        }
        if std::mem::take(&mut self.chords) {
            let _ = ratatui::crossterm::execute!(out, PopKeyboardEnhancementFlags);
        }
    }
}

/// How long [`run_screen`] waits for a key before asking [`Screen::tick`].
pub const TICK: std::time::Duration = std::time::Duration::from_millis(200);

/// Holds log records back while alive, writing them when dropped.
struct HeldLogs;

impl HeldLogs {
    fn new() -> Self {
        crate::aux::logging::hold_logs(true);
        HeldLogs
    }
}

impl Drop for HeldLogs {
    fn drop(&mut self) {
        crate::aux::logging::hold_logs(false);
    }
}

/// Run `screen` full screen until it is done. The terminal is restored on
/// return and on panic. Log records raised meanwhile are held back and
/// written once the normal screen is back. Blocking work runs on the normal
/// screen, where its own progress output belongs, and the view comes back
/// after it. The mouse and keys held with modifiers are reported only to a
/// screen that asks for them, and events already queued (a turn of the wheel
/// is several) are taken before the screen is drawn again.
pub fn run_screen(screen: &mut impl Screen) -> anyhow::Result<()> {
    ratatui::run(|terminal| -> anyhow::Result<()> {
        let held = HeldLogs::new();
        let mut modes = Modes::default();
        let mut redraw = true;
        while !screen.done() {
            if redraw {
                terminal.draw(|f| screen.render(f))?;
                // Asked once the screen is up: its modes are its own.
                modes.ask(screen);
            }
            if let Some(notice) = screen.take_notice() {
                // A bell, and a desktop notification where the terminal shows
                // one (OSC 9); terminals without either ignore them.
                let mut out = std::io::stdout();
                let _ = std::io::Write::write_all(
                    &mut out,
                    format!("\x07\x1b]9;{notice}\x07").as_bytes(),
                );
                let _ = std::io::Write::flush(&mut out);
            }
            if !event::poll(TICK)? {
                redraw = screen.tick();
                continue;
            }
            redraw = false;
            loop {
                redraw |= match event::read()? {
                    Event::Key(key) if key.kind != KeyEventKind::Press => false,
                    Event::Key(key)
                        if key.modifiers.contains(KeyModifiers::CONTROL)
                            && key.code == KeyCode::Char('c') =>
                    {
                        screen.interrupt();
                        true
                    }
                    Event::Key(key) => {
                        screen.handle_key(key);
                        true
                    }
                    Event::Mouse(m)
                        if matches!(
                            m.kind,
                            MouseEventKind::Down(MouseButton::Left)
                                | MouseEventKind::ScrollUp
                                | MouseEventKind::ScrollDown
                        ) =>
                    {
                        screen.mouse(m);
                        true
                    }
                    Event::Resize(..) => true,
                    _ => false,
                };
                let more = !screen.done() && screen.pending_work().is_none();
                if !more || !event::poll(std::time::Duration::ZERO)? {
                    break;
                }
            }
            if let Some(message) = screen.pending_work() {
                modes.release();
                ratatui::restore();
                crate::aux::logging::hold_logs(false);
                eprintln!("{message}");
                screen.do_work();
                crate::aux::logging::hold_logs(true);
                *terminal = ratatui::try_init()?;
                redraw = true;
            }
        }
        modes.release();
        // Restore before writing what was held, not after.
        ratatui::restore();
        drop(held);
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

    /// Bins of an explicit `width` on the scaled axis. On the linear scale
    /// with width 1, bin `k` holds exactly the value `k`, so a histogram of
    /// bin indices draws one bar per category.
    pub fn with_width(scale: Scale, width: f64) -> Self {
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
/// A bar height [`HistPlot`] can draw: whole counts, or any non-negative
/// real value (a summed signal, a log statistic).
pub trait BarValue: Copy {
    fn bar(self) -> f64;
}

impl BarValue for usize {
    fn bar(self) -> f64 {
        self as f64
    }
}

impl BarValue for f64 {
    fn bar(self) -> f64 {
        self
    }
}

pub struct HistPlot<'a, T: BarValue = usize> {
    pub bins: Binning,
    pub kmin: i32,
    pub counts: &'a [T],
    /// Style of each bin's bar, by key.
    pub style: &'a dyn Fn(i32) -> Style,
    /// A subset drawn in front, in the bar style; `counts` then draw dimmed
    /// behind it.
    pub subset: Option<&'a [T]>,
    pub y_scale: Scale,
    /// Top of the y axis in count units; `None` scales to the tallest bar.
    /// Set it to put several plots on one scale.
    pub y_max: Option<f64>,
    /// Bin under the accent rule and ▲.
    pub pointer: Option<i32>,
    /// Other symbols on the x axis, by key.
    pub marks: Vec<(i32, &'static str, Style)>,
    /// Tick label at bin `k` in place of the bin's value; `None` from it
    /// drops that tick, so the labels decide where ticks go (e.g. at
    /// category boundaries with `tick_every: Some(1)`). Unset, every tick
    /// shows its value.
    pub x_label: Option<&'a dyn Fn(i32) -> Option<String>>,
    /// Ticks every this many bins, in place of the scale's default.
    pub tick_every: Option<i32>,
}

const EIGHTHS: [&str; 8] = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];

impl<T: BarValue> HistPlot<'_, T> {
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

        let height = |c: T| self.y_scale.apply(c.bar().max(0.0));
        let tallest = self.counts.iter().map(|&c| height(c)).fold(0.0, f64::max);
        let max_h = self
            .y_max
            .map_or(tallest, |m| self.y_scale.apply(m.max(0.0)).max(tallest));
        let cells = chart.height as usize * 8;
        let eighths = |c: T| {
            if c.bar() <= 0.0 || max_h <= 0.0 {
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

        let mut bars = |counts: &[T], behind: Option<&[T]>, dim: bool| {
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
        let every = self
            .tick_every
            .unwrap_or_else(|| self.bins.tick_every(nbins))
            .max(1);
        let mut next_free = labels.x;
        let kmax = self.kmin + nbins as i32 - 1;
        for k in (self.kmin..=kmax).filter(|k| k % every == 0) {
            let Some(x) = x_of(k) else { continue };
            let s = match self.x_label {
                Some(label) => match label(k) {
                    Some(s) => s,
                    None => continue,
                },
                None => compact(self.bins.tick_value(k)),
            };
            put(buf, x, axis.y, "┴", DIM);
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

/// One side of a [`MirrorPlot`].
pub struct MirrorSide<'a, T: BarValue = f64> {
    pub counts: &'a [T],
    /// A subset drawn in front, in `style`; `counts` then draw dimmed
    /// behind it.
    pub subset: Option<&'a [T]>,
    pub style: Style,
    /// Named in the side's outer corner; empty for none.
    pub name: &'a str,
}

/// Two bar series on one scale around a zero line, one growing up and the
/// other down, as a Miami plot, in half cells. Signed values are a mirror of
/// their positive and negative parts. Axes as [`HistPlot`]'s, one column
/// per bar.
pub struct MirrorPlot<'a, T: BarValue = f64> {
    pub up: MirrorSide<'a, T>,
    pub down: MirrorSide<'a, T>,
    pub y_scale: Scale,
    /// Top of either side in count units; `None` scales to the tallest bar.
    pub y_max: Option<f64>,
    /// Labels at the top, the zero line and the bottom, in place of the
    /// scale's own.
    pub y_labels: Option<[String; 3]>,
    /// Bar under the accent rule and ▲.
    pub pointer: Option<usize>,
    /// Tick label at bar `i`; `None` from it drops that tick. Unset, there
    /// are no ticks.
    pub x_label: Option<&'a dyn Fn(usize) -> Option<String>>,
}

impl<T: BarValue> MirrorPlot<'_, T> {
    /// Draw into `area`: the halves over the rows above the last two, which
    /// hold the x axis and its labels; the left [`GUTTER`] columns hold the
    /// y axis.
    pub fn render(&self, buf: &mut Buffer, area: Rect) {
        let [plot, axis, labels] = Layout::vertical([
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .areas(area);
        let [gutter, chart] =
            Layout::horizontal([Constraint::Length(GUTTER), Constraint::Min(1)]).areas(plot);
        if chart.width == 0 || chart.height < 3 {
            return;
        }
        let half = (chart.height - 1) / 2;
        let zero = chart.top() + half;
        let x_of = |i: usize| Some(chart.x + i as u16).filter(|&x| x < chart.right());

        let height = |c: T| self.y_scale.apply(c.bar().max(0.0));
        let all = self.up.counts.iter().chain(self.down.counts);
        let tallest = all.map(|&c| height(c)).fold(0.0, f64::max);
        let max_h = self
            .y_max
            .map_or(tallest, |m| self.y_scale.apply(m.max(0.0)).max(tallest));
        let cells = half as usize * 2;
        let halves = |c: T| {
            if c.bar() <= 0.0 || max_h <= 0.0 {
                0
            } else {
                ((height(c) / max_h * cells as f64).round() as usize).clamp(1, cells)
            }
        };

        if let Some(x) = self.pointer.and_then(x_of) {
            for y in chart.top()..chart.top() + 2 * half + 1 {
                put(buf, x, y, "┊", ACCENTED);
            }
        }
        for x in chart.left()..chart.right() {
            put(buf, x, zero, "─", DIM);
        }
        for (side, up) in [(&self.up, true), (&self.down, false)] {
            let (whole, part) = if up { ("█", "▄") } else { ("█", "▀") };
            let mut bars = |counts: &[T], behind: Option<&[T]>, style: Style| {
                for (i, &c) in counts.iter().enumerate() {
                    let Some(x) = x_of(i) else { break };
                    let (top, under) = (halves(c), behind.map_or(0, |b| halves(b[i])));
                    for k in 0..top.div_ceil(2) {
                        let y = if up {
                            zero - 1 - k as u16
                        } else {
                            zero + 1 + k as u16
                        };
                        // As in HistPlot, a partial end in front of a longer
                        // bar rounds up to a whole cell.
                        let full = 2 * k + 2 <= top || under >= 2 * k + 2;
                        put(buf, x, y, if full { whole } else { part }, style);
                    }
                }
            };
            match side.subset {
                Some(subset) => {
                    bars(side.counts, None, DIM);
                    bars(subset, Some(side.counts), side.style);
                }
                None => bars(side.counts, None, side.style),
            }
        }
        buf.set_string(chart.x, chart.top(), self.up.name, DIM);
        buf.set_string(chart.x, chart.top() + 2 * half, self.down.name, DIM);

        // y axis: the scale's top on both sides of the zero line.
        let gx = gutter.right() - 1;
        for y in gutter.top()..gutter.bottom() {
            put(buf, gx, y, "│", DIM);
        }
        let own = || {
            let top = compact(self.y_scale.invert(max_h));
            [top.clone(), "0".to_string(), top]
        };
        let ys = [chart.top(), zero, chart.top() + 2 * half];
        for (y, s) in ys
            .into_iter()
            .zip(self.y_labels.clone().unwrap_or_else(own))
        {
            let x = gx.saturating_sub(1 + s.len() as u16).max(gutter.x);
            buf.set_string(x, y, &s, DIM);
            put(buf, gx, y, "┤", DIM);
        }

        // x axis: baseline with ticks, labels below, then the pointer.
        for x in axis.left()..axis.right() {
            let sym = match x.cmp(&gx) {
                std::cmp::Ordering::Less => " ",
                std::cmp::Ordering::Equal => "└",
                std::cmp::Ordering::Greater => "─",
            };
            put(buf, x, axis.y, sym, DIM);
        }
        let n = self.up.counts.len().max(self.down.counts.len());
        let mut next_free = labels.x;
        for i in 0..n {
            let Some(x) = x_of(i) else { break };
            let Some(s) = self.x_label.and_then(|label| label(i)) else {
                continue;
            };
            put(buf, x, axis.y, "┴", DIM);
            if x >= next_free && x + (s.len() as u16) <= labels.right() {
                buf.set_string(x, labels.y, &s, DIM);
                next_free = x + s.len() as u16 + 1;
            }
        }
        if let Some(x) = self.pointer.and_then(x_of) {
            put(buf, x, axis.y, "▲", HIGHLIGHT);
        }
    }
}

#[cfg(test)]
#[path = "tests/ui.rs"]
mod tests;
