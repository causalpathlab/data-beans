//! Panels for papers: a histogram and a 2D density, each laid out inside
//! exactly the box given, all text at one size ([`Look::font`]), on the
//! same [`Axis`] the other figures read values on.

use super::*;

/// How a panel is drawn: text size, what is drawn besides the plot, and
/// the colours of kept and removed values.
#[derive(Debug, Clone, PartialEq)]
pub struct Look {
    /// Text size in points, as printed: ticks, axis titles and the key.
    pub font: f64,
    /// Only the plot, its axes, ticks and axis titles: no title or key.
    pub panel_only: bool,
    /// Kept values at full depth, as `#rrggbb`.
    pub kept: String,
    /// Removed values at full depth, as `#rrggbb`.
    pub removed: String,
}

impl Default for Look {
    fn default() -> Self {
        Self {
            font: 8.0,
            panel_only: false,
            kept: BAR.into(),
            removed: ACCENT.into(),
        }
    }
}

impl Look {
    /// Kept dark grey, removed light grey.
    pub fn gray() -> Self {
        Self {
            kept: "#404040".into(),
            removed: "#bfbfbf".into(),
            ..Self::default()
        }
    }

    /// The size a title is set at: a little larger than the rest.
    pub fn title_size(&self) -> f64 {
        1.25 * self.font
    }

    /// The height of the row a title and key take, none when panel only
    /// or there is neither.
    fn head(&self, title: &str, key: bool) -> f64 {
        if self.panel_only || (title.is_empty() && !key) {
            0.0
        } else {
            1.4 * self.title_size()
        }
    }

    /// The title at the left of the row from `(x, y)` and, with `key`, the
    /// kept and removed swatches at its right, `w` across.
    fn heading(&self, c: &mut Canvas, (x, y, w): (f64, f64, f64), title: &str, key: bool) {
        if self.head(title, key) == 0.0 {
            return;
        }
        let s = self.font;
        let base = y + 0.1 * s + 0.75 * self.title_size();
        if !title.is_empty() {
            c.bold(
                x + 0.1 * s,
                base,
                title,
                self.title_size(),
                Anchor::Start,
                INK,
            );
        }
        if key {
            let items = [(&self.kept, "kept"), (&self.removed, "removed")];
            let sw = 0.8 * s;
            let wide: f64 = items
                .iter()
                .map(|(_, n)| sw + 0.3 * s + text_width(n, s))
                .sum::<f64>()
                + 0.8 * s;
            let mut kx = x + w - 0.1 * s - wide;
            for (colour, name) in items {
                c.rect(kx, base - sw, sw, sw, colour);
                c.text(kx + sw + 0.3 * s, base, name, s, Anchor::Start, INK);
                kx += sw + 0.3 * s + text_width(name, s) + 0.8 * s;
            }
        }
    }
}

/// Text size for a panel `w` x `h` points: about an eighteenth of its
/// shorter side, from 5 to 8 points.
pub fn fit_font(w: f64, h: f64) -> f64 {
    (w.min(h) / 18.0).clamp(5.0, 8.0)
}

/// Kept and removed values in each of `bins` bins of `axis`; without a
/// mask every value is kept. Values that are not finite are left out.
pub fn stacks(
    axis: &Axis,
    bins: usize,
    values: &[f32],
    kept: Option<&[bool]>,
) -> Vec<(usize, usize)> {
    let k = axis.counts(values, bins, |i| kept.is_none_or(|m| m[i]));
    let r = axis.counts(values, bins, |i| kept.is_some_and(|m| !m[i]));
    k.into_iter().zip(r).collect()
}

/// `#rrggbb` as its three channels.
fn channels(c: &str) -> Option<[f64; 3]> {
    let h = c.strip_prefix('#').filter(|h| h.len() == 6)?;
    let v = |k: usize| u8::from_str_radix(&h[k..k + 2], 16).ok().map(f64::from);
    Some([v(0)?, v(2)?, v(4)?])
}

/// The colour of a square holding `kept` and `removed` values, where the
/// fullest square holds `most`: the majority's colour, mixed from white
/// deeper as the square holds more (on a log scale).
pub fn square_colour(look: &Look, kept: usize, removed: usize, most: usize) -> String {
    let full = if kept >= removed {
        &look.kept
    } else {
        &look.removed
    };
    let n = kept + removed;
    let Some(rgb) = channels(full).filter(|_| n < most) else {
        return full.clone();
    };
    let t = 0.12 + 0.88 * (n as f64).ln_1p() / (most as f64).ln_1p();
    let mix = |v: f64| (255.0 - t * (255.0 - v)).round() as u8;
    format!("#{:02x}{:02x}{:02x}", mix(rgb[0]), mix(rgb[1]), mix(rgb[2]))
}

/// A box: left, top, width, height.
pub type Area = (f64, f64, f64, f64);

/// Ticks thinned, every other one dropped, until each `pos` is at least
/// `room` from the next.
fn thin(mut ticks: Vec<f64>, pos: impl Fn(f64) -> f64, room: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    while ticks.len() > 2
        && ticks
            .windows(2)
            .any(|p| (pos(p[1]) - pos(p[0])).abs() < room(p[0], p[1]))
    {
        ticks = ticks.into_iter().step_by(2).collect();
    }
    ticks
}

/// The widest of the labels of `ticks`.
fn widest(ticks: &[f64], size: f64) -> f64 {
    ticks
        .iter()
        .map(|&t| text_width(&short(t), size))
        .fold(0.0, f64::max)
}

/// The room below a plot: ticks, their labels and the x title.
fn bottom(look: &Look, x_title: &str) -> f64 {
    let s = look.font;
    let title = if x_title.is_empty() { 0.0 } else { 1.3 * s };
    0.35 * s + 0.3 * s + s + title + 0.1 * s
}

/// The room left of a plot whose tick labels are `label` wide: the y
/// title, the labels and the ticks.
fn left(look: &Look, y_title: &str, label: f64) -> f64 {
    let s = look.font;
    let title = if y_title.is_empty() { 0.0 } else { 1.3 * s };
    0.1 * s + title + label + 0.3 * s + 0.35 * s
}

/// The room right of a plot, for half the widest x label.
fn right(look: &Look, axis: &Axis) -> f64 {
    (widest(&axis.ticks(), look.font) / 2.0 + 0.1 * look.font).max(0.5 * look.font)
}

/// The x axis under `(px, py, pw, ph)`: baseline, ticks, labels and title.
fn x_axis(c: &mut Canvas, look: &Look, axis: &Axis, (px, py, pw, ph): Area, title: &str) {
    let s = look.font;
    let base = py + ph;
    c.line(px, base, px + pw, base, INK, 0.6);
    let ticks = thin(
        axis.ticks(),
        |t| axis.frac(t) * pw,
        |a, b| (text_width(&short(a), s) + text_width(&short(b), s)) / 2.0 + 0.5 * s,
    );
    let label = base + 0.35 * s + 0.3 * s + 0.75 * s;
    for t in ticks {
        let tx = px + axis.frac(t) * pw;
        c.line(tx, base, tx, base + 0.35 * s, INK, 0.6);
        c.text(tx, label, &short(t), s, Anchor::Middle, INK);
    }
    if !title.is_empty() {
        c.text(
            px + pw / 2.0,
            label + 1.3 * s,
            title,
            s,
            Anchor::Middle,
            INK,
        );
    }
}

/// The y ticks of `axis` that fit up a plot `ph` high.
fn y_ticks(look: &Look, axis: &Axis, ph: f64) -> Vec<f64> {
    thin(axis.ticks(), |t| axis.frac(t) * ph, |_, _| 1.2 * look.font)
}

/// The y axis left of `(px, py, pw, ph)`: line, ticks, labels and the
/// title, turned, from `x0`.
fn y_axis(
    c: &mut Canvas,
    look: &Look,
    (axis, ticks): (&Axis, &[f64]),
    (px, py, _, ph): Area,
    x0: f64,
    title: &str,
) {
    let s = look.font;
    c.line(px, py, px, py + ph, INK, 0.6);
    for &t in ticks {
        let ty = py + ph - axis.frac(t) * ph;
        c.line(px - 0.35 * s, ty, px, ty, INK, 0.6);
        c.text(px - 0.65 * s, ty + 0.35 * s, &short(t), s, Anchor::End, INK);
    }
    if !title.is_empty() {
        c.vtext(x0 + 0.1 * s + 0.75 * s, py + ph / 2.0, title, s, INK);
    }
}

/// A dashed edge.
fn edge(c: &mut Canvas, (x1, y1, x2, y2): Area) {
    c.dashed(x1, y1, x2, y2, INK, 0.8);
}

/// A histogram of `values` on `axis` in `bins` bins, kept stacked under
/// removed, with dashed lines at `edges`. Values that are not finite are
/// left out: clamp infinities to the axis ends before passing them.
pub struct HistPanel<'a> {
    pub axis: Axis,
    pub bins: usize,
    pub values: &'a [f32],
    /// Which values are kept; `None` keeps them all.
    pub kept: Option<&'a [bool]>,
    pub edges: Vec<f64>,
    pub x_title: String,
    /// Empty for none.
    pub y_title: String,
    /// Empty for none; never drawn with [`Look::panel_only`].
    pub title: String,
}

impl HistPanel<'_> {
    /// The count axis and its ticks for a plot `ph` high.
    fn counts_axis(&self, look: &Look, ph: f64) -> (Axis, Vec<f64>) {
        let s = stacks(&self.axis, self.bins, self.values, self.kept);
        let top = s.iter().map(|(k, r)| k + r).max().unwrap_or(0).max(1) as f64;
        let axis = Axis::linear(0.0, top * 1.05);
        let ticks = y_ticks(look, &axis, ph);
        (axis, ticks)
    }

    /// Where the bars go inside `area`.
    pub fn plot_area(&self, look: &Look, (x, y, w, h): Area) -> Area {
        let s = look.font;
        let top = look.head(&self.title, self.kept.is_some()) + 0.5 * s;
        let ph = (h - top - bottom(look, &self.x_title)).max(1.0);
        let (_, ticks) = self.counts_axis(look, ph);
        let l = left(look, &self.y_title, widest(&ticks, s));
        let pw = (w - l - right(look, &self.axis)).max(1.0);
        (x + l, y + top, pw, ph)
    }

    /// Draw into `area` of `c`.
    pub fn draw(&self, c: &mut Canvas, look: &Look, area: Area) {
        let (x, y, w, _) = area;
        look.heading(c, (x, y, w), &self.title, self.kept.is_some());
        let (px, py, pw, ph) = self.plot_area(look, area);
        let (counts, ticks) = self.counts_axis(look, ph);
        let n = self.bins.max(1);
        let bw = pw / n as f64;
        let gap = if bw > 3.0 { 0.1 * bw } else { 0.0 };
        let base = py + ph;
        for (i, (k, r)) in stacks(&self.axis, n, self.values, self.kept)
            .into_iter()
            .enumerate()
        {
            let (hk, hr) = (counts.frac(k as f64) * ph, counts.frac(r as f64) * ph);
            let bx = px + i as f64 * bw + gap / 2.0;
            c.rect(bx, base - hk, bw - gap, hk, &look.kept);
            c.rect(bx, base - hk - hr, bw - gap, hr, &look.removed);
        }
        for &e in &self.edges {
            let ex = px + self.axis.frac(e).clamp(0.0, 1.0) * pw;
            edge(c, (ex, py, ex, base));
        }
        x_axis(c, look, &self.axis, (px, py, pw, ph), &self.x_title);
        y_axis(
            c,
            look,
            (&counts, &ticks),
            (px, py, pw, ph),
            x,
            &self.y_title,
        );
    }

    /// The panel alone as SVG, `w` x `h` points.
    pub fn svg(&self, look: &Look, w: f64, h: f64) -> String {
        let mut c = Canvas::new(w, h);
        self.draw(&mut c, look, (0.0, 0.0, w, h));
        c.finish()
    }
}

/// Which marginal histograms a [`DensityPanel`] has.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Marginals {
    /// x above the plot.
    pub top: bool,
    /// y right of the plot, bars growing rightwards.
    pub right: bool,
}

/// Where a [`DensityPanel`]'s parts go.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DensityLayout {
    /// The squares.
    pub plot: Area,
    /// The x marginal: the plot's x span, above it.
    pub top: Option<Area>,
    /// The y marginal: the plot's y span, right of it.
    pub right: Option<Area>,
}

/// Sites over two axes as squares, each coloured by [`square_colour`],
/// with dashed lines at the edges and optional marginal histograms on the
/// same axes and bins, without tick labels of their own. Values that are
/// not finite are left out: clamp infinities to the axis ends before
/// passing them.
pub struct DensityPanel<'a> {
    pub x: Axis,
    pub x_bins: usize,
    pub y: Axis,
    pub y_bins: usize,
    pub xs: &'a [f32],
    pub ys: &'a [f32],
    /// Which sites are kept; `None` keeps them all.
    pub kept: Option<&'a [bool]>,
    pub x_edges: Vec<f64>,
    pub y_edges: Vec<f64>,
    pub x_title: String,
    pub y_title: String,
    /// Empty for none; never drawn with [`Look::panel_only`].
    pub title: String,
    pub marginals: Marginals,
}

impl DensityPanel<'_> {
    /// Kept and removed sites per square, row by row from the bottom:
    /// square `(i, j)` (x bin `i`, y bin `j`) at `j * x_bins + i`. A site
    /// with either value not finite is left out.
    pub fn grid(&self) -> Vec<(usize, usize)> {
        let (nx, ny) = (self.x_bins.max(1), self.y_bins.max(1));
        let mut g = vec![(0, 0); nx * ny];
        for (i, (&x, &y)) in self.xs.iter().zip(self.ys).enumerate() {
            if !(x.is_finite() && y.is_finite()) {
                continue;
            }
            let cell = &mut g[self.y.bin(y as f64, ny) * nx + self.x.bin(x as f64, nx)];
            if self.kept.is_none_or(|m| m[i]) {
                cell.0 += 1;
            } else {
                cell.1 += 1;
            }
        }
        g
    }

    /// Where the plot and its marginals go inside `area`.
    pub fn layout(&self, look: &Look, (x, y, w, h): Area) -> DensityLayout {
        let s = look.font;
        let Marginals { top, right: side } = self.marginals;
        let head = look.head(&self.title, self.kept.is_some());
        let pad = if top { 0.15 * s } else { 0.5 * s };
        let avail_h = (h - head - pad - bottom(look, &self.x_title)).max(1.0);
        let l = left(look, &self.y_title, widest(&self.y.ticks(), s));
        let avail_w = (w - l - right(look, &self.x)).max(1.0);
        let m = 0.2 * avail_w.min(avail_h);
        let gap = 0.25 * s;
        let room = |on: bool| if on { m + gap } else { 0.0 };
        let (pw, ph) = (avail_w - room(side), avail_h - room(top));
        let (px, py) = (x + l, y + head + pad + room(top));
        DensityLayout {
            plot: (px, py, pw, ph),
            top: top.then_some((px, py - gap - m, pw, m)),
            right: side.then_some((px + pw + gap, py, m, ph)),
        }
    }

    /// Draw into `area` of `c`.
    pub fn draw(&self, c: &mut Canvas, look: &Look, area: Area) {
        let (x, y, w, _) = area;
        look.heading(c, (x, y, w), &self.title, self.kept.is_some());
        let lay = self.layout(look, area);
        let (px, py, pw, ph) = lay.plot;
        let (nx, ny) = (self.x_bins.max(1), self.y_bins.max(1));
        let (cw, ch) = (pw / nx as f64, ph / ny as f64);
        let col = |i: usize| px + i as f64 * cw;
        let row = |j: usize| py + ph - (j + 1) as f64 * ch;
        let grid = self.grid();
        let most = grid.iter().map(|(k, r)| k + r).max().unwrap_or(0);
        for (n, &(k, r)) in grid.iter().enumerate() {
            if k + r > 0 {
                let fill = square_colour(look, k, r, most);
                c.rect(col(n % nx), row(n / nx), cw, ch, &fill);
            }
        }
        if let Some((_, ty, _, th)) = lay.top {
            let st = stacks(&self.x, nx, self.xs, self.kept);
            let most = st.iter().map(|(k, r)| k + r).max().unwrap_or(0).max(1) as f64;
            let base = ty + th;
            for (i, (k, r)) in st.into_iter().enumerate() {
                let (lk, lr) = (k as f64 / most * th, r as f64 / most * th);
                c.rect(col(i), base - lk, cw, lk, &look.kept);
                c.rect(col(i), base - lk - lr, cw, lr, &look.removed);
            }
        }
        if let Some((rx, _, rw, _)) = lay.right {
            let st = stacks(&self.y, ny, self.ys, self.kept);
            let most = st.iter().map(|(k, r)| k + r).max().unwrap_or(0).max(1) as f64;
            for (j, (k, r)) in st.into_iter().enumerate() {
                let (lk, lr) = (k as f64 / most * rw, r as f64 / most * rw);
                c.rect(rx, row(j), lk, ch, &look.kept);
                c.rect(rx + lk, row(j), lr, ch, &look.removed);
            }
        }
        let y_top = lay.top.map_or(py, |t| t.1);
        for &e in &self.x_edges {
            let ex = px + self.x.frac(e).clamp(0.0, 1.0) * pw;
            edge(c, (ex, y_top, ex, py + ph));
        }
        let x_end = lay.right.map_or(px + pw, |r| r.0 + r.2);
        for &e in &self.y_edges {
            let ey = py + ph - self.y.frac(e).clamp(0.0, 1.0) * ph;
            edge(c, (px, ey, x_end, ey));
        }
        x_axis(c, look, &self.x, lay.plot, &self.x_title);
        let ticks = y_ticks(look, &self.y, ph);
        y_axis(c, look, (&self.y, &ticks), lay.plot, x, &self.y_title);
    }

    /// The panel alone as SVG, `w` x `h` points.
    pub fn svg(&self, look: &Look, w: f64, h: f64) -> String {
        let mut c = Canvas::new(w, h);
        self.draw(&mut c, look, (0.0, 0.0, w, h));
        c.finish()
    }
}

#[cfg(test)]
mod tests;
