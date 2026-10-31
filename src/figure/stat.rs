//! The statistics figure: the histogram of each of nnz, sum, mean and sd
//! over the rows or the columns, and mean against sd with the Poisson line
//! `sd = √mean`. Marked entries are drawn in the accent on every panel;
//! the entry under the explorer's cursor gets a pointer.

use super::*;

/// The statistics in figure order.
pub const STATS: [&str; 4] = ["nnz", "sum", "mean", "sd"];

/// What the figure is drawn from.
pub struct StatData<'a> {
    /// The data's name, for the title.
    pub title: &'a str,
    /// `rows` or `columns`.
    pub side: &'a str,
    /// nnz, sum, mean and sd per entry.
    pub values: [&'a [f32]; 4],
    /// Entries marked, if any.
    pub marked: Option<&'a [bool]>,
    /// The entry under the cursor, with its name.
    pub cursor: Option<(usize, &'a str)>,
}

/// The histogram of statistic `k` (see [`STATS`]) in the box given.
pub fn draw_stat(d: &StatData, k: usize, c: &mut Canvas, (x, y, w, h): (f64, f64, f64, f64)) {
    let values = d.values[k];
    let marked = d.marked.filter(|m| m.iter().any(|&b| b));
    let mut sorted: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(f32::total_cmp);
    let median = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
    let mut subtitle = format!("median {}", short(median as f64));
    if let Some(m) = marked {
        subtitle.push_str(&format!(" · {} marked", m.iter().filter(|&&b| b).count()));
    }
    if let Some((i, name)) = d.cursor {
        let rank = sorted.len() - sorted.partition_point(|&v| v <= values[i]) + 1;
        subtitle.push_str(&format!(" · ▲ {name}, rank {rank} of {}", sorted.len()));
    }
    let side = d.side.trim_end_matches('s');
    let mut layers = vec![(None, BAR)];
    if let Some(m) = marked {
        layers.push((Some(m), ACCENT));
    }
    Hist {
        title: format!("{} per {side}", STATS[k]),
        subtitle,
        x_title: format!("{} per {side}", STATS[k]),
        axis: Axis::fit(values, true),
        values,
        layers,
        edge: None,
        cursor: d.cursor.map(|(i, _)| values[i] as f64),
    }
    .draw(c, x, y, w, h);
}

/// Mean against sd per entry, log on both, with the Poisson line.
pub fn draw_mean_sd(d: &StatData, c: &mut Canvas, (x, y, w, h): (f64, f64, f64, f64)) {
    let (mean, sd) = (d.values[2], d.values[3]);
    let marked = d.marked.filter(|m| m.iter().any(|&b| b));
    let points: Vec<Point> = (0..mean.len())
        .map(|i| {
            let on = marked.is_some_and(|m| m[i]) || d.cursor.is_some_and(|(k, _)| k == i);
            Point {
                x: mean[i],
                y: sd[i],
                fill: if on { ACCENT } else { BAR },
                ringed: false,
            }
        })
        .collect();
    let (xa, ya) = (Axis::fit(mean, true), Axis::fit(sd, true));
    let side = d.side.trim_end_matches('s');
    let mut key = vec![];
    if marked.is_some() {
        key.push((ACCENT, "marked".to_string()));
    }
    Scatter {
        title: "Mean against sd".into(),
        subtitle: format!("per {side} · dashed: Poisson, sd = √mean"),
        x_title: "mean".into(),
        y_title: "sd".into(),
        x: xa,
        y: ya,
        points: &points,
        x_edge: None,
        y_edge: None,
        key,
    }
    .draw(c, x, y, w, h);
    // The Poisson line, over the plot area Scatter used.
    let (px, py, pw, ph) = plot_area(x, y, w, h);
    let line: Vec<(f64, f64)> = (0..=60)
        .map(|k| xa.at(k as f64 / 60.0))
        .filter(|&m| m.sqrt() >= ya.lo && m.sqrt() <= ya.hi)
        .map(|m| (px + xa.frac(m) * pw, py + ph - ya.frac(m.sqrt()) * ph))
        .collect();
    if line.len() > 1 {
        c.polyline(&line, MUTED, 0.9, true);
    }
}

/// The whole figure.
pub fn figure(d: &StatData) -> String {
    let (cols, pw, ph) = (3usize, 320.0, 230.0);
    let (left, top) = (16.0, 74.0);
    let (w, h) = (left * 2.0 + cols as f64 * pw, top + 2.0 * ph + 12.0);
    let mut c = Canvas::new(w, h);
    let n = d.values[0].len();
    c.bold(
        left + 8.0,
        28.0,
        &format!("{} · {} statistics", d.title, d.side.trim_end_matches('s')),
        15.0,
        Anchor::Start,
        INK,
    );
    c.text(
        left + 8.0,
        46.0,
        &format!("{n} {}", d.side),
        10.0,
        Anchor::Start,
        MUTED,
    );
    let mut boxes = (0..6).map(|k| {
        (
            left + (k % cols) as f64 * pw,
            top + (k / cols) as f64 * ph,
            pw - 12.0,
            ph - 14.0,
        )
    });
    for k in 0..4 {
        draw_stat(d, k, &mut c, boxes.next().unwrap());
    }
    draw_mean_sd(d, &mut c, boxes.next().unwrap());
    c.finish()
}
