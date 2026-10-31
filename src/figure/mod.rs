//! Figures (feature `figure`): each one drawn once as SVG, written as a
//! vector PDF and a PNG by [`save`], returned as PNG bytes by [`png`] for an
//! agent, and shown as an image inside the full-screen views where the
//! terminal has an image protocol ([`term`], feature `tui`). What is saved
//! is what was on screen.
//!
//! The drawing is faba's (`src/figure/`): a [`Canvas`] in points on white,
//! greys for context and one accent for what a threshold removes. Panels
//! here read values on an [`Axis`] (linear, or log on `ln(shift + x)`), the
//! axis the printed histograms and the views' histograms bin on too.
//!
//! Text is set in the system's Helvetica, Arial or DejaVu Sans; a host with
//! none of them installed writes figures without their labels.

pub mod qc;
pub mod stat;
#[cfg(feature = "tui")]
pub mod term;

use std::fmt::Write as _;
use std::sync::{Arc, OnceLock};

/// Text and axes.
pub const INK: &str = "#1a1a1a";
/// Bars.
pub const BAR: &str = "#4d4d4d";
/// Secondary text and axis labels.
pub const MUTED: &str = "#6b6b6b";
/// Bars behind a front subset.
pub const FAINT: &str = "#cfcfcf";
/// Gridlines.
pub const GRID: &str = "#ececec";
/// The one accent, as on screen: what a threshold removes, the cursor.
pub const ACCENT: &str = "#d97757";
const FONT: &str = "Helvetica, Arial, 'DejaVu Sans', sans-serif";

/// Escape text for SVG content and attributes.
pub fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

#[derive(Debug, Clone, Copy)]
pub enum Anchor {
    Start,
    Middle,
    End,
}

impl Anchor {
    fn svg(self) -> &'static str {
        match self {
            Anchor::Start => "start",
            Anchor::Middle => "middle",
            Anchor::End => "end",
        }
    }
}

/// An SVG drawing in points, white background.
pub struct Canvas {
    width: f64,
    height: f64,
    body: String,
}

impl Canvas {
    pub fn new(width: f64, height: f64) -> Self {
        let mut c = Self {
            width,
            height,
            body: String::new(),
        };
        c.rect(0.0, 0.0, width, height, "#ffffff");
        c
    }

    /// A filled rectangle; empty ones (zero or negative size) are skipped,
    /// as SVG renderers reject them.
    pub fn rect(&mut self, x: f64, y: f64, w: f64, h: f64, fill: &str) {
        // Written to two decimals: what would read as zero is skipped too.
        if !(w >= 0.005 && h >= 0.005 && w.is_finite() && h.is_finite()) {
            return;
        }
        let _ = writeln!(
            self.body,
            r#"<rect x="{x:.2}" y="{y:.2}" width="{w:.2}" height="{h:.2}" fill="{fill}"/>"#
        );
    }

    /// A rectangle at `opacity`.
    pub fn shade(&mut self, x: f64, y: f64, w: f64, h: f64, fill: &str, opacity: f64) {
        if !(w >= 0.005 && h >= 0.005 && w.is_finite() && h.is_finite()) {
            return;
        }
        let _ = writeln!(
            self.body,
            r#"<rect x="{x:.2}" y="{y:.2}" width="{w:.2}" height="{h:.2}" fill="{fill}" fill-opacity="{opacity:.3}"/>"#
        );
    }

    pub fn line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, stroke: &str, width: f64) {
        let _ = writeln!(
            self.body,
            r#"<line x1="{x1:.2}" y1="{y1:.2}" x2="{x2:.2}" y2="{y2:.2}" stroke="{stroke}" stroke-width="{width}"/>"#
        );
    }

    pub fn dashed(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, stroke: &str, width: f64) {
        let _ = writeln!(
            self.body,
            r#"<line x1="{x1:.2}" y1="{y1:.2}" x2="{x2:.2}" y2="{y2:.2}" stroke="{stroke}" stroke-width="{width}" stroke-dasharray="3 2"/>"#
        );
    }

    /// A filled dot.
    pub fn dot(&mut self, x: f64, y: f64, r: f64, fill: &str, opacity: f64) {
        let _ = writeln!(
            self.body,
            r#"<circle cx="{x:.2}" cy="{y:.2}" r="{r:.2}" fill="{fill}" fill-opacity="{opacity:.2}"/>"#
        );
    }

    /// A dot with a ring around it.
    pub fn ringed(&mut self, x: f64, y: f64, r: f64, fill: &str, ring: &str) {
        let _ = writeln!(
            self.body,
            r#"<circle cx="{x:.2}" cy="{y:.2}" r="{r:.2}" fill="{fill}" stroke="{ring}" stroke-width="0.9"/>"#
        );
    }

    /// A triangle pointing up, its tip at `(x, y)`.
    pub fn pointer(&mut self, x: f64, y: f64, size: f64, fill: &str) {
        let _ = writeln!(
            self.body,
            r#"<path d="M{:.2},{:.2} L{:.2},{:.2} L{x:.2},{y:.2} Z" fill="{fill}"/>"#,
            x - size,
            y + 1.6 * size,
            x + size,
            y + 1.6 * size
        );
    }

    pub fn text(&mut self, x: f64, y: f64, s: &str, size: f64, anchor: Anchor, fill: &str) {
        let _ = writeln!(
            self.body,
            r#"<text x="{x:.2}" y="{y:.2}" font-family="{FONT}" font-size="{size}" text-anchor="{}" fill="{fill}">{}</text>"#,
            anchor.svg(),
            esc(s)
        );
    }

    pub fn bold(&mut self, x: f64, y: f64, s: &str, size: f64, anchor: Anchor, fill: &str) {
        let _ = writeln!(
            self.body,
            r#"<text x="{x:.2}" y="{y:.2}" font-family="{FONT}" font-size="{size}" font-weight="bold" text-anchor="{}" fill="{fill}">{}</text>"#,
            anchor.svg(),
            esc(s)
        );
    }

    /// Text turned to read bottom to top, centred on `(x, y)`.
    pub fn vtext(&mut self, x: f64, y: f64, s: &str, size: f64, fill: &str) {
        self.turned(x, y, s, size, -90.0, Anchor::Middle, fill);
    }

    /// Text turned by `degrees`, anchored at `(x, y)`.
    #[allow(clippy::too_many_arguments)]
    pub fn turned(
        &mut self,
        x: f64,
        y: f64,
        s: &str,
        size: f64,
        degrees: f64,
        anchor: Anchor,
        fill: &str,
    ) {
        let _ = writeln!(
            self.body,
            r#"<text transform="translate({x:.2} {y:.2}) rotate({degrees})" font-family="{FONT}" font-size="{size}" text-anchor="{}" fill="{fill}">{}</text>"#,
            anchor.svg(),
            esc(s)
        );
    }

    /// An open line through `points`, dashed when asked.
    pub fn polyline(&mut self, points: &[(f64, f64)], stroke: &str, width: f64, dashed: bool) {
        let pts: Vec<String> = points
            .iter()
            .map(|(x, y)| format!("{x:.2},{y:.2}"))
            .collect();
        let dash = if dashed {
            r#" stroke-dasharray="4 3""#
        } else {
            ""
        };
        let _ = writeln!(
            self.body,
            r#"<polyline points="{}" fill="none" stroke="{stroke}" stroke-width="{width}"{dash}/>"#,
            pts.join(" ")
        );
    }

    /// Swatches and their names in a row from `(x, y)`; the x after the last.
    pub fn legend(&mut self, mut x: f64, y: f64, items: &[(&str, &str)]) -> f64 {
        for (colour, name) in items {
            self.rect(x, y - 8.0, 9.0, 9.0, colour);
            self.text(x + 13.0, y, name, 9.5, Anchor::Start, INK);
            x += 26.0 + text_width(name, 9.5);
        }
        x
    }

    pub fn finish(self) -> String {
        format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
{body}</svg>
"#,
            w = self.width,
            h = self.height,
            body = self.body
        )
    }
}

/// About how wide `s` sets at `size` points, for laying out legends.
pub fn text_width(s: &str, size: f64) -> f64 {
    s.chars().count() as f64 * size * 0.55
}

/// A number as a short label: `1.2k`, `35`, `0.05`.
pub fn short(v: f64) -> String {
    let a = v.abs();
    let trim = |s: String| {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    };
    if a >= 1e6 {
        format!("{}M", trim(format!("{:.1}", v / 1e6)))
    } else if a >= 1e4 {
        format!("{}k", trim(format!("{:.0}", v / 1e3)))
    } else if a >= 1e3 {
        format!("{}k", trim(format!("{:.1}", v / 1e3)))
    } else if a >= 10.0 || a == 0.0 {
        format!("{v:.0}")
    } else if a >= 1.0 {
        trim(format!("{v:.1}"))
    } else {
        trim(format!("{v:.3}"))
    }
}

pub use crate::qc::axis::Axis;

/// Bins across a panel: about one per 6 points, from 20 to 60.
pub fn bins_for(width: f64) -> usize {
    ((width / 6.0) as usize).clamp(20, 60)
}

/// The box a panel draws its data in, inside `(x, y, w, h)`: room on the
/// left for y labels, above for the title and subtitle, below for ticks
/// and the x title.
pub fn plot_area(x: f64, y: f64, w: f64, h: f64) -> (f64, f64, f64, f64) {
    let (left, right, top, bottom) = (44.0, 10.0, 34.0, 32.0);
    (
        x + left,
        y + top,
        (w - left - right).max(1.0),
        (h - top - bottom).max(1.0),
    )
}

/// A panel's title and subtitle.
fn heading(c: &mut Canvas, x: f64, y: f64, title: &str, subtitle: &str) {
    c.bold(x, y + 12.0, title, 11.0, Anchor::Start, INK);
    c.text(x, y + 25.0, subtitle, 9.0, Anchor::Start, MUTED);
}

/// The x axis under a plot area: baseline, tick stubs and labels, title.
fn x_axis(c: &mut Canvas, axis: &Axis, (px, py, pw, ph): (f64, f64, f64, f64), title: &str) {
    c.line(px, py + ph, px + pw, py + ph, INK, 0.6);
    for t in axis.ticks() {
        let tx = px + axis.frac(t) * pw;
        c.line(tx, py + ph, tx, py + ph + 3.0, INK, 0.6);
        c.text(tx, py + ph + 12.0, &short(t), 8.0, Anchor::Middle, MUTED);
    }
    c.text(
        px + pw / 2.0,
        py + ph + 25.0,
        title,
        8.5,
        Anchor::Middle,
        MUTED,
    );
}

/// A histogram of `values` in layers drawn back to front, each the values
/// its mask takes (all of them for `None`) in its colour; with an edge, a
/// dashed line and its label; with a cursor, a pointer under the axis.
pub struct Hist<'a> {
    pub title: String,
    pub subtitle: String,
    pub x_title: String,
    pub axis: Axis,
    pub values: &'a [f32],
    pub layers: Vec<(Option<&'a [bool]>, &'static str)>,
    pub edge: Option<(f64, String)>,
    pub cursor: Option<f64>,
}

impl Hist<'_> {
    pub fn draw(&self, c: &mut Canvas, x: f64, y: f64, w: f64, h: f64) {
        heading(c, x, y, &self.title, &self.subtitle);
        let (px, py, pw, ph) = plot_area(x, y, w, h);
        let n = bins_for(pw);
        let layers: Vec<(Vec<usize>, &str)> = self
            .layers
            .iter()
            .map(|(mask, colour)| {
                let counts = self
                    .axis
                    .counts(self.values, n, |i| mask.is_none_or(|m| m[i]));
                (counts, *colour)
            })
            .collect();
        let top = layers
            .iter()
            .flat_map(|(c, _)| c.iter().copied())
            .max()
            .unwrap_or(0)
            .max(1) as f64;
        let y_axis = Axis::linear(0.0, top * 1.08);
        for t in y_axis.ticks() {
            let gy = py + ph - y_axis.frac(t) * ph;
            if t > 0.0 {
                c.line(px, gy, px + pw, gy, GRID, 0.6);
            }
            c.text(px - 4.0, gy + 3.0, &short(t), 8.0, Anchor::End, MUTED);
        }
        let bw = pw / n as f64;
        let gap = if bw > 3.0 { 0.12 * bw } else { 0.0 };
        for (counts, colour) in &layers {
            for (i, &k) in counts.iter().enumerate() {
                let bh = y_axis.frac(k as f64) * ph;
                c.rect(
                    px + i as f64 * bw + gap / 2.0,
                    py + ph - bh,
                    bw - gap,
                    bh,
                    colour,
                );
            }
        }
        x_axis(c, &self.axis, (px, py, pw, ph), &self.x_title);
        if let Some((v, label)) = &self.edge {
            let ex = px + self.axis.frac(*v).clamp(0.0, 1.0) * pw;
            c.dashed(ex, py - 2.0, ex, py + ph, ACCENT, 1.1);
            let (anchor, dx) = if ex > px + 0.8 * pw {
                (Anchor::End, -3.0)
            } else {
                (Anchor::Start, 3.0)
            };
            c.text(ex + dx, py + 7.0, label, 8.5, anchor, ACCENT);
        }
        if let Some(v) = self.cursor {
            let cx = px + self.axis.frac(v).clamp(0.0, 1.0) * pw;
            c.pointer(cx, py + ph + 1.0, 3.5, ACCENT);
        }
    }
}

/// One point of a [`Scatter`]: where, its fill, and whether it is ringed.
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub fill: &'static str,
    pub ringed: bool,
}

/// Points over two axes, with an edge on either; past [`Scatter::DOTS`]
/// points, a density grid in grey with the ringed points' cells over it in
/// the accent.
pub struct Scatter<'a> {
    pub title: String,
    pub subtitle: String,
    pub x_title: String,
    pub y_title: String,
    pub x: Axis,
    pub y: Axis,
    pub points: &'a [Point],
    pub x_edge: Option<f64>,
    pub y_edge: Option<f64>,
    /// Swatches in the corner: colour and name.
    pub key: Vec<(&'static str, String)>,
}

impl Scatter<'_> {
    /// More points than this draw as a density grid.
    pub const DOTS: usize = 20_000;

    pub fn draw(&self, c: &mut Canvas, x: f64, y: f64, w: f64, h: f64) {
        heading(c, x, y, &self.title, &self.subtitle);
        let (px, py, pw, ph) = plot_area(x, y, w, h);
        let at = |p: &Point| {
            (
                px + self.x.frac(p.x as f64) * pw,
                py + ph - self.y.frac(p.y as f64) * ph,
            )
        };
        for t in self.y.ticks() {
            let gy = py + ph - self.y.frac(t) * ph;
            c.line(px, gy, px + pw, gy, GRID, 0.6);
            c.text(px - 4.0, gy + 3.0, &short(t), 8.0, Anchor::End, MUTED);
        }
        for t in self.x.ticks() {
            let gx = px + self.x.frac(t) * pw;
            c.line(gx, py, gx, py + ph, GRID, 0.6);
        }
        c.vtext(x + 9.0, py + ph / 2.0, &self.y_title, 8.5, MUTED);
        if self.points.len() <= Self::DOTS {
            for p in self.points.iter().filter(|p| !p.ringed) {
                let (sx, sy) = at(p);
                c.dot(sx, sy, 1.6, p.fill, 0.85);
            }
            for p in self.points.iter().filter(|p| p.ringed) {
                let (sx, sy) = at(p);
                c.ringed(sx, sy, 2.3, p.fill, ACCENT);
            }
        } else {
            let (gw, gh) = (((pw / 3.0) as usize).max(1), ((ph / 3.0) as usize).max(1));
            let mut all = vec![0usize; gw * gh];
            let mut ringed = vec![0usize; gw * gh];
            for p in self.points {
                let (fx, fy) = (self.x.frac(p.x as f64), self.y.frac(p.y as f64));
                if !(fx.is_finite() && fy.is_finite()) {
                    continue;
                }
                let gx = ((fx * gw as f64) as usize).min(gw - 1);
                let gy = ((fy * gh as f64) as usize).min(gh - 1);
                all[gy * gw + gx] += 1;
                if p.ringed {
                    ringed[gy * gw + gx] += 1;
                }
            }
            let most = all.iter().copied().max().unwrap_or(1).max(1) as f64;
            let (cw, ch) = (pw / gw as f64, ph / gh as f64);
            for (k, (&a, &r)) in all.iter().zip(&ringed).enumerate() {
                if a == 0 {
                    continue;
                }
                let (gx, gy) = ((k % gw) as f64, (k / gw) as f64);
                let opacity = 0.2 + 0.8 * (a as f64).ln_1p() / most.ln_1p();
                let colour = if r * 2 >= a { ACCENT } else { BAR };
                c.shade(
                    px + gx * cw,
                    py + ph - (gy + 1.0) * ch,
                    cw,
                    ch,
                    colour,
                    opacity,
                );
            }
        }
        if let Some(v) = self.x_edge {
            let ex = px + self.x.frac(v).clamp(0.0, 1.0) * pw;
            c.dashed(ex, py, ex, py + ph, ACCENT, 1.1);
        }
        if let Some(v) = self.y_edge {
            let ey = py + ph - self.y.frac(v).clamp(0.0, 1.0) * ph;
            c.dashed(px, ey, px + pw, ey, ACCENT, 1.1);
        }
        x_axis(c, &self.x, (px, py, pw, ph), &self.x_title);
        let mut ky = py + ph - 6.0 - 12.0 * self.key.len().saturating_sub(1) as f64;
        for (colour, name) in &self.key {
            let kx = px + pw - 8.0 - text_width(name, 8.0) - 12.0;
            c.rect(kx, ky - 7.0, 8.0, 8.0, colour);
            c.text(kx + 11.0, ky, name, 8.0, Anchor::Start, MUTED);
            ky += 12.0;
        }
    }
}

/// The light-to-dark ramp a value from 0 to 1 takes, in five steps.
pub fn ramp(f: f64) -> &'static str {
    const STEPS: [&str; 5] = ["#bdcbdb", "#93a8bf", "#6984a2", "#3f5f84", "#18365c"];
    STEPS[((f.clamp(0.0, 1.0) * 5.0) as usize).min(4)]
}

fn fontdb() -> Arc<usvg::fontdb::Database> {
    static DB: OnceLock<Arc<usvg::fontdb::Database>> = OnceLock::new();
    DB.get_or_init(|| {
        let mut db = usvg::fontdb::Database::new();
        db.load_system_fonts();
        Arc::new(db)
    })
    .clone()
}

/// SVG parsing options, with the system fonts loaded once.
fn options() -> usvg::Options<'static> {
    usvg::Options {
        fontdb: fontdb(),
        ..usvg::Options::default()
    }
}

/// `tree` rendered at `scale` pixels per point, on white.
fn pixmap(tree: &usvg::Tree, scale: f32) -> anyhow::Result<resvg::tiny_skia::Pixmap> {
    let size = tree
        .size()
        .to_int_size()
        .scale_by(scale)
        .ok_or_else(|| anyhow::anyhow!("figure has no size"))?;
    let mut pixmap = resvg::tiny_skia::Pixmap::new(size.width(), size.height())
        .ok_or_else(|| anyhow::anyhow!("figure too large to rasterise"))?;
    pixmap.fill(resvg::tiny_skia::Color::WHITE);
    resvg::render(
        tree,
        resvg::tiny_skia::Transform::from_scale(scale, scale),
        &mut pixmap.as_mut(),
    );
    Ok(pixmap)
}

/// Pixels per point in a saved PNG.
const PNG_SCALE: f32 = 2.0;

/// `svg` as a PNG, `scale` pixels per point.
pub fn png(svg: &str, scale: f32) -> anyhow::Result<Vec<u8>> {
    let tree = usvg::Tree::from_str(svg, &options())?;
    Ok(pixmap(&tree, scale)?.encode_png()?)
}

/// `svg` as an RGBA image, `scale` pixels per point.
pub fn image(svg: &str, scale: f32) -> anyhow::Result<image::RgbaImage> {
    let tree = usvg::Tree::from_str(svg, &options())?;
    let pixmap = pixmap(&tree, scale)?;
    let (w, h) = (pixmap.width(), pixmap.height());
    image::RgbaImage::from_raw(w, h, pixmap.take())
        .ok_or_else(|| anyhow::anyhow!("pixel buffer size mismatch"))
}

/// The files a figure saved under `name` is written as: `name` itself when
/// it ends in `.pdf`, `.png` or `.svg`, else `name.pdf` and `name.png`.
pub fn saved_paths(name: &str) -> Vec<String> {
    if [".pdf", ".png", ".svg"].iter().any(|e| name.ends_with(e)) {
        vec![name.to_string()]
    } else {
        vec![format!("{name}.pdf"), format!("{name}.png")]
    }
}

/// Write `svg` under `name` (see [`saved_paths`]); the paths written. A
/// file already there is never replaced: the save is refused instead.
pub fn save(svg: &str, name: &str) -> anyhow::Result<Vec<String>> {
    let paths = saved_paths(name);
    if let Some(there) = paths.iter().find(|p| std::path::Path::new(p).exists()) {
        anyhow::bail!("{there} is already there; not replacing it");
    }
    let tree = usvg::Tree::from_str(svg, &options())?;
    for path in &paths {
        if let Some(dir) = std::path::Path::new(path)
            .parent()
            .filter(|d| !d.as_os_str().is_empty())
        {
            std::fs::create_dir_all(dir)?;
        }
        if path.ends_with(".pdf") {
            let pdf = svg2pdf::to_pdf(&tree, Default::default(), Default::default())
                .map_err(|e| anyhow::anyhow!("PDF conversion failed: {e:?}"))?;
            std::fs::write(path, pdf)?;
        } else if path.ends_with(".png") {
            std::fs::write(path, pixmap(&tree, PNG_SCALE)?.encode_png()?)?;
        } else {
            std::fs::write(path, svg)?;
        }
    }
    Ok(paths)
}

/// Whether a save under `name` would replace a file already there.
pub fn taken(name: &str) -> bool {
    saved_paths(name)
        .iter()
        .any(|p| std::path::Path::new(p).exists())
}

/// `name`, or failing that the first of `{name}_2`, `{name}_3`, … that no
/// save would replace a file under.
pub fn free_name(name: &str) -> String {
    let (stem, ext) = [".pdf", ".png", ".svg"]
        .iter()
        .find_map(|e| name.strip_suffix(e).map(|s| (s, *e)))
        .unwrap_or((name, ""));
    (1..)
        .map(|n| match n {
            1 => format!("{stem}{ext}"),
            n => format!("{stem}_{n}{ext}"),
        })
        .find(|p| !taken(p))
        .expect("some suffix is free")
}

#[cfg(test)]
mod tests;
