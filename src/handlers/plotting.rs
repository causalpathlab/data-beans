//! `data-beans plot`: figures of a backend's QC and statistics, drawn as an
//! SVG with axes and rendered by legume-plot (PNG through resvg, PDF through
//! svg2pdf). Scatter points are rasterized, as legume-plot does, so a
//! million cells make an image, not a million SVG elements.

use crate::handlers::analysis::{HistStat, StatDim};
use crate::handlers::cell_qc::CellQc;
use crate::hdf5_io::resolve_backend_file;
use crate::qc::{collect_column_stat, collect_row_stat};
use crate::sparse_io::open_sparse_matrix;
use base64::Engine;
use clap::{Args, Subcommand};
use data_beans::qc_lib::QcArgs;
use legume_plot::rasterize::{rasterize_group_png, Extent, PointShape, RadiusSpec};
use legume_plot::{escape_xml as escape, FigureFormats, Rgb};
use log::info;
use std::fmt::Write as _;
use std::path::Path;

#[derive(Args, Debug)]
pub struct PlotArgs {
    #[command(subcommand)]
    pub kind: PlotKind,
}

#[derive(Subcommand, Debug)]
pub enum PlotKind {
    /// Cell QC: each metric's histogram with the edge QC held it to, and
    /// total counts against detected genes, cells coloured by verdict
    #[command(
        mut_args(|a| a.hide(false)),
        mut_arg("qc_mito_pattern", |a| a.default_value(crate::handlers::cell_qc::MITO))
    )]
    Qc(PlotQcArgs),
    /// The histogram of one statistic over the rows or the columns
    Stat(PlotStatArgs),
}

#[derive(Args, Debug)]
pub struct PlotQcArgs {
    /// data file -- `.zarr`, `.zarr.zip`, or `.h5`
    pub data_file: Box<str>,

    #[command(flatten)]
    pub qc: QcArgs,

    /// figure file: `.png`, `.svg` or `.pdf`
    #[arg(short, long)]
    pub output: Box<str>,

    /// cells per rayon job (omit for auto-scaling by feature count)
    #[arg(long)]
    pub block_size: Option<usize>,
}

#[derive(Args, Debug)]
pub struct PlotStatArgs {
    /// data file -- `.zarr`, `.zarr.zip`, or `.h5`
    pub data_file: Box<str>,

    /// over rows (features) or columns (cells)
    #[arg(short = 's', long, value_enum, default_value = "column")]
    pub stat_dim: StatDim,

    /// the statistic, on a log10(x+1) axis
    #[arg(long, value_enum, default_value = "nnz")]
    pub stat: HistStat,

    /// figure file: `.png`, `.svg` or `.pdf`
    #[arg(short, long)]
    pub output: Box<str>,

    /// cells per rayon job (omit for auto-scaling by feature count)
    #[arg(long)]
    pub block_size: Option<usize>,
}

const KEPT: Rgb = (59, 111, 182);
const OUTLIER: Rgb = (209, 73, 91);
const NEAR_EMPTY: Rgb = (154, 154, 154);
const EDGE: Rgb = (209, 73, 91);
const CAP: Rgb = (224, 138, 30);
const INK: &str = "#333";
const FONT: &str = "Helvetica, Arial, sans-serif";

const PANEL_W: f32 = 460.0;
const PANEL_H: f32 = 340.0;
const TITLE_H: f32 = 44.0;

pub fn run_plot(args: &PlotArgs) -> anyhow::Result<()> {
    let (fig, output) = match &args.kind {
        PlotKind::Qc(a) => {
            let qc = CellQc::compute(&a.data_file, &a.qc, a.block_size)?;
            (qc_figure(&qc, &a.data_file)?, &a.output)
        }
        PlotKind::Stat(a) => (stat_figure(a)?, &a.output),
    };
    write_figure(fig, output)?;
    info!("wrote {output}");
    Ok(())
}

/// Write `fig` as the format `output`'s extension names.
fn write_figure(fig: Figure, output: &str) -> anyhow::Result<()> {
    let (w, h) = (fig.w, fig.h);
    let path = Path::new(output);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default();
    let want = FigureFormats {
        svg: ext == "svg",
        png: ext == "png",
        pdf: ext == "pdf",
    };
    anyhow::ensure!(
        want.svg || want.png || want.pdf,
        "{output}: name a .png, .svg or .pdf file"
    );
    let base = path.with_extension("");
    legume_plot::write_figure(&fig.finish(), w, h, &base.to_string_lossy(), want)?;
    Ok(())
}

/// The histogram of one row or column statistic.
fn stat_figure(a: &PlotStatArgs) -> anyhow::Result<Figure> {
    let (backend, file) = resolve_backend_file(&a.data_file, None)?;
    let data = open_sparse_matrix(&file, &backend)?;
    let (what, values) = match a.stat_dim {
        StatDim::Row => (
            "row",
            a.stat
                .values(&collect_row_stat(data.as_ref(), a.block_size)?),
        ),
        StatDim::Column => (
            "column",
            a.stat
                .values(&collect_column_stat(data.as_ref(), a.block_size)?),
        ),
    };
    let label = format!("{} per {what}", a.stat.metric());
    let mut fig = Figure::new(1, 1, &format!("{}: {label}", a.data_file));
    let all = Series {
        label: format!("{} {what}s", values.len()),
        color: KEPT,
        values,
    };
    fig.histogram(0, (&label, &format!("{what}s")), &[all], true, &[]);
    Ok(fig)
}

/// The QC figure: a histogram per metric, stacked by verdict, with its
/// band edge and cap; then total counts against detected genes.
fn qc_figure(qc: &CellQc, title: &str) -> anyhow::Result<Figure> {
    let r = &qc.report;
    let n = r.train_keep.len();
    let near_empty: Vec<usize> = (0..n).filter(|&c| r.near_empty[c]).collect();
    let outlier: Vec<usize> = (0..n)
        .filter(|&c| !r.near_empty[c] && !r.train_keep[c])
        .collect();
    let kept = r.emit_idx_unmasked();
    let verdicts: [(String, Rgb, &Vec<usize>); 3] = [
        (format!("kept {}", kept.len()), KEPT, &kept),
        (format!("outlier {}", outlier.len()), OUTLIER, &outlier),
        (
            format!("near-empty {}", near_empty.len()),
            NEAR_EMPTY,
            &near_empty,
        ),
    ];
    // Each verdict's cells, as what `of` makes of a cell.
    fn by_verdict<T>(
        verdicts: &[(String, Rgb, &Vec<usize>)],
        of: impl Fn(usize) -> T,
    ) -> Vec<Series<T>> {
        verdicts
            .iter()
            .map(|(label, color, idx)| Series {
                label: label.clone(),
                color: *color,
                values: idx.iter().map(|&c| of(c)).collect(),
            })
            .collect()
    }

    let metrics = qc.metrics();
    let panels = metrics.len() + 1;
    let cols = if panels > 4 { 3 } else { 2 };
    let mut fig = Figure::new(
        cols,
        panels.div_ceil(cols),
        &format!(
            "{title}: {n} cells, {} kept, {} outliers, {} near-empty (bands {} MADs)",
            kept.len(),
            outlier.len(),
            near_empty.len(),
            qc.cfg.n_mads
        ),
    );
    for (i, m) in metrics.iter().enumerate() {
        let mut marks = vec![];
        if let Some(edge) = qc.edge(m) {
            marks.push(Mark {
                at: edge,
                color: EDGE,
                label: m.bound().0,
            });
        }
        if let Some(cap) = m.cap {
            marks.push(Mark {
                at: cap,
                color: CAP,
                label: "cap",
            });
        }
        let groups = by_verdict(&verdicts, |c| m.values[c]);
        fig.histogram(i, (m.label, "cells"), &groups, m.count, &marks);
    }
    let layers = by_verdict(&verdicts, |c| (r.total_counts[c], r.n_genes[c]));
    fig.scatter(metrics.len(), "total counts", "detected genes", &layers)?;
    Ok(fig)
}

/// Values (or points) drawn in one colour.
struct Series<T> {
    label: String,
    color: Rgb,
    values: Vec<T>,
}

/// A vertical line at a value, labelled.
struct Mark {
    at: f32,
    color: Rgb,
    label: &'static str,
}

/// `v` on a log10(x+1) axis when `log`, else as is.
fn scale(v: f32, log: bool) -> f32 {
    if log {
        (v.max(0.0) + 1.0).log10()
    } else {
        v
    }
}

/// An axis: the data range it shows, on a log10(x+1) scale or as is.
#[derive(Clone, Copy)]
struct Axis {
    lo: f32,
    hi: f32,
    log: bool,
}

impl Axis {
    fn fit(values: impl Iterator<Item = f32>, log: bool) -> Self {
        let t = |v: f32| scale(v, log);
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for v in values.filter(|v| v.is_finite()) {
            lo = lo.min(t(v));
            hi = hi.max(t(v));
        }
        if !lo.is_finite() {
            (lo, hi) = (0.0, 1.0);
        }
        if hi - lo < 1e-6 {
            hi = lo + 1.0;
        }
        let pad = (hi - lo) * 0.03;
        Self {
            lo: lo - pad,
            hi: hi + pad,
            log,
        }
    }

    fn t(self, v: f32) -> f32 {
        scale(v, self.log)
    }

    /// Where `v` falls, 0 at `lo` to 1 at `hi`.
    fn frac(self, v: f32) -> f32 {
        (self.t(v) - self.lo) / (self.hi - self.lo)
    }

    /// Tick values (on the data's own scale) and their labels: powers of
    /// ten on a log axis, steps of 1, 2 or 5 times a power of ten otherwise.
    fn ticks(self) -> Vec<(f32, String)> {
        if self.log {
            // 0, then 1, 2, 5 times each power of ten; powers alone when
            // those crowd.
            let within = |x: f32| (self.lo..=self.hi).contains(&self.t(x));
            let pick = |steps: &[f32]| -> Vec<f32> {
                let mut v = vec![0.0];
                let mut p = 1.0f32;
                while self.t(p) <= self.hi {
                    v.extend(steps.iter().map(|m| m * p));
                    p *= 10.0;
                }
                v.into_iter().filter(|&x| within(x)).collect()
            };
            let mut v = pick(&[1.0, 2.0, 5.0]);
            if v.len() > 8 {
                v = pick(&[1.0]);
            }
            return v.into_iter().map(|x| (x, short(x))).collect();
        }
        let span = self.hi - self.lo;
        let raw = span / 5.0;
        let mag = 10f32.powf(raw.log10().floor());
        let step = [1.0, 2.0, 5.0, 10.0]
            .iter()
            .map(|m| m * mag)
            .find(|s| span / s <= 6.0)
            .unwrap_or(10.0 * mag);
        let mut v = vec![];
        let mut x = (self.lo / step).ceil() * step;
        while x <= self.hi {
            // Snap the float drift around zero (-0.0, 1e-9).
            let shown = if x.abs() < step * 1e-3 { 0.0 } else { x };
            v.push((shown, short(shown)));
            x += step;
        }
        v
    }
}

/// A number for a tick label: 1k, 2.5M, 0.05.
fn short(x: f32) -> String {
    let a = x.abs();
    let trim = |s: String| {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    };
    if a >= 1e6 {
        format!("{}M", trim(format!("{:.1}", x / 1e6)))
    } else if a >= 1e3 {
        format!("{}k", trim(format!("{:.1}", x / 1e3)))
    } else if a >= 1.0 || a == 0.0 {
        trim(format!("{x:.1}"))
    } else {
        trim(format!("{x:.3}"))
    }
}

fn rgb((r, g, b): Rgb) -> String {
    format!("rgb({r},{g},{b})")
}

/// A grid of panels, written as SVG.
struct Figure {
    svg: String,
    cols: usize,
    w: u32,
    h: u32,
}

/// A panel's plot area, in figure pixels.
#[derive(Clone, Copy)]
struct Area {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

impl Figure {
    fn new(cols: usize, rows: usize, title: &str) -> Self {
        let w = (cols as f32 * PANEL_W) as u32;
        let h = (rows as f32 * PANEL_H + TITLE_H) as u32;
        let mut svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" \
             viewBox=\"0 0 {w} {h}\" font-family=\"{FONT}\">\n\
             <rect width=\"{w}\" height=\"{h}\" fill=\"white\"/>\n"
        );
        let _ = writeln!(
            svg,
            "<text x=\"16\" y=\"28\" font-size=\"16\" fill=\"{INK}\">{}</text>",
            escape(title)
        );
        Self { svg, cols, w, h }
    }

    /// The plot area of panel `i`, inside its margins.
    fn area(&self, i: usize) -> Area {
        let (col, row) = (i % self.cols, i / self.cols);
        let (left, right, top, bottom) = (64.0, 18.0, 30.0, 48.0);
        Area {
            x: col as f32 * PANEL_W + left,
            y: TITLE_H + row as f32 * PANEL_H + top,
            w: PANEL_W - left - right,
            h: PANEL_H - top - bottom,
        }
    }

    /// Axes, ticks and labels around `a`.
    fn axes(&mut self, a: Area, x: Axis, y: Axis, xlabel: &str, ylabel: &str) {
        let s = &mut self.svg;
        let (bx, by) = (a.x, a.y + a.h);
        let _ = writeln!(
            s,
            "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"none\" stroke=\"#999\" stroke-width=\"0.8\"/>",
            a.x, a.y, a.w, a.h
        );
        for (v, label) in x.ticks() {
            let px = a.x + x.frac(v) * a.w;
            let _ = writeln!(
                s,
                "<line x1=\"{px:.1}\" y1=\"{by:.1}\" x2=\"{px:.1}\" y2=\"{:.1}\" stroke=\"#999\"/>\
                 <text x=\"{px:.1}\" y=\"{:.1}\" font-size=\"11\" fill=\"{INK}\" text-anchor=\"middle\">{label}</text>",
                by + 4.0,
                by + 16.0
            );
        }
        for (v, label) in y.ticks() {
            let py = by - y.frac(v) * a.h;
            let _ = writeln!(
                s,
                "<line x1=\"{:.1}\" y1=\"{py:.1}\" x2=\"{bx:.1}\" y2=\"{py:.1}\" stroke=\"#999\"/>\
                 <text x=\"{:.1}\" y=\"{:.1}\" font-size=\"11\" fill=\"{INK}\" text-anchor=\"end\">{label}</text>",
                bx - 4.0,
                bx - 6.0,
                py + 4.0
            );
        }
        let _ = writeln!(
            s,
            "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"12\" fill=\"{INK}\" text-anchor=\"middle\">{}</text>",
            a.x + a.w / 2.0,
            by + 34.0,
            escape(xlabel)
        );
        let (lx, ly) = (a.x - 46.0, a.y + a.h / 2.0);
        let _ = writeln!(
            s,
            "<text x=\"{lx:.1}\" y=\"{ly:.1}\" font-size=\"12\" fill=\"{INK}\" text-anchor=\"middle\" transform=\"rotate(-90 {lx:.1} {ly:.1})\">{}</text>",
            escape(ylabel)
        );
    }

    /// A legend in a row above `a`.
    fn legend(&mut self, a: Area, entries: &[(&str, Rgb)]) {
        let mut x = a.x;
        let y = a.y - 20.0;
        for (label, color) in entries {
            let _ = writeln!(
                self.svg,
                "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"10\" height=\"10\" fill=\"{}\"/>\
                 <text x=\"{:.1}\" y=\"{:.1}\" font-size=\"11\" fill=\"{INK}\">{}</text>",
                rgb(*color),
                x + 14.0,
                y + 9.0,
                escape(label)
            );
            x += 24.0 + 6.5 * label.chars().count() as f32;
        }
    }

    /// Panel `i`: the histogram of `groups`' values, stacked, on a log10(x+1)
    /// axis when `log`, with `marks` drawn across it.
    fn histogram(
        &mut self,
        i: usize,
        (xlabel, ylabel): (&str, &str),
        groups: &[Series<f32>],
        log: bool,
        marks: &[Mark],
    ) {
        const BINS: usize = 50;
        let a = self.area(i);
        let all = groups.iter().flat_map(|g| g.values.iter().copied());
        let x = Axis::fit(all.chain(marks.iter().map(|m| m.at)), log);
        let bin = |v: f32| -> Option<usize> {
            let f = x.frac(v);
            (f.is_finite() && (0.0..=1.0).contains(&f))
                .then(|| ((f * BINS as f32) as usize).min(BINS - 1))
        };
        let counts: Vec<[usize; BINS]> = groups
            .iter()
            .map(|g| {
                let mut c = [0usize; BINS];
                for b in g.values.iter().filter_map(|&v| bin(v)) {
                    c[b] += 1;
                }
                c
            })
            .collect();
        let top = (0..BINS)
            .map(|b| counts.iter().map(|c| c[b]).sum::<usize>())
            .max()
            .unwrap_or(0)
            .max(1);
        let y = Axis {
            lo: 0.0,
            hi: top as f32 * 1.05,
            log: false,
        };
        let bw = a.w / BINS as f32;
        for b in 0..BINS {
            let mut base = 0usize;
            for (g, c) in groups.iter().zip(&counts) {
                if c[b] == 0 {
                    continue;
                }
                let y0 = a.y + a.h - y.frac(base as f32) * a.h;
                let y1 = a.y + a.h - y.frac((base + c[b]) as f32) * a.h;
                let _ = writeln!(
                    self.svg,
                    "<rect x=\"{:.1}\" y=\"{y1:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"{}\"/>",
                    a.x + b as f32 * bw,
                    bw.max(0.5) - 0.3,
                    y0 - y1,
                    rgb(g.color)
                );
                base += c[b];
            }
        }
        for (k, m) in marks.iter().enumerate() {
            let px = a.x + x.frac(m.at) * a.w;
            let _ = writeln!(
                self.svg,
                "<line x1=\"{px:.1}\" y1=\"{:.1}\" x2=\"{px:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"1.5\" stroke-dasharray=\"5 3\"/>\
                 <text x=\"{:.1}\" y=\"{:.1}\" font-size=\"11\" fill=\"{}\">{} {}</text>",
                a.y,
                a.y + a.h,
                rgb(m.color),
                px + 3.0,
                a.y + 12.0 + 13.0 * k as f32,
                rgb(m.color),
                m.label,
                short(m.at)
            );
        }
        self.axes(a, x, y, xlabel, ylabel);
        if groups.len() > 1 {
            let entries: Vec<(&str, Rgb)> =
                groups.iter().map(|g| (g.label.as_str(), g.color)).collect();
            self.legend(a, &entries);
        }
    }

    /// Panel `i`: each layer's points, on log10(x+1) axes, rasterized.
    fn scatter(
        &mut self,
        i: usize,
        xlabel: &str,
        ylabel: &str,
        layers: &[Series<(f32, f32)>],
    ) -> anyhow::Result<()> {
        let a = self.area(i);
        let all = || layers.iter().flat_map(|l| l.values.iter());
        let x = Axis::fit(all().map(|p| p.0), true);
        let y = Axis::fit(all().map(|p| p.1), true);
        let ext = Extent {
            w: a.w.round() as u32,
            h: a.h.round() as u32,
        };
        let n: usize = layers.iter().map(|l| l.values.len()).sum();
        let radius = if n > 100_000 { 0.8 } else { 1.8 };
        for Series {
            color,
            values: points,
            ..
        } in layers
        {
            if points.is_empty() {
                continue;
            }
            let px: Vec<(f32, f32)> = points
                .iter()
                .map(|&(u, v)| (x.frac(u) * a.w, a.h - y.frac(v) * a.h))
                .collect();
            let png = rasterize_group_png(
                &px,
                ext,
                RadiusSpec::Scalar(radius),
                *color,
                0.6,
                PointShape::Circle,
            )?;
            let _ = writeln!(
                self.svg,
                "<image x=\"{:.1}\" y=\"{:.1}\" width=\"{}\" height=\"{}\" href=\"data:image/png;base64,{}\"/>",
                a.x,
                a.y,
                ext.w,
                ext.h,
                base64::engine::general_purpose::STANDARD.encode(png)
            );
        }
        self.axes(a, x, y, xlabel, ylabel);
        let entries: Vec<(&str, Rgb)> =
            layers.iter().map(|l| (l.label.as_str(), l.color)).collect();
        self.legend(a, &entries);
        Ok(())
    }

    fn finish(mut self) -> String {
        self.svg.push_str("</svg>\n");
        self.svg
    }
}
