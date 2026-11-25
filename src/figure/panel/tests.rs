use super::*;

/// A `<text>` of an SVG: where, how big, which way, what it says.
#[derive(Debug)]
struct Txt {
    x: f64,
    y: f64,
    size: f64,
    anchor: String,
    turned: bool,
    bold: bool,
    s: String,
}

fn attr<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let key = format!(" {name}=\"");
    let start = line.find(&key)? + key.len();
    let len = line[start..].find('"')?;
    Some(&line[start..start + len])
}

fn num(line: &str, name: &str) -> f64 {
    attr(line, name).unwrap().parse().unwrap()
}

fn texts(svg: &str) -> Vec<Txt> {
    svg.lines()
        .filter(|l| l.starts_with("<text"))
        .map(|l| {
            let (x, y, turned) = match attr(l, "transform") {
                Some(t) => {
                    let inner = &t["translate(".len()..t.find(')').unwrap()];
                    let mut it = inner.split(' ').map(|v| v.parse::<f64>().unwrap());
                    (it.next().unwrap(), it.next().unwrap(), true)
                }
                None => (num(l, "x"), num(l, "y"), false),
            };
            let s = &l[l.find('>').unwrap() + 1..l.rfind("</text>").unwrap()];
            Txt {
                x,
                y,
                size: num(l, "font-size"),
                anchor: attr(l, "text-anchor").unwrap().to_string(),
                turned,
                bold: l.contains("font-weight=\"bold\""),
                s: s.to_string(),
            }
        })
        .collect()
}

/// The box a text takes: left, top, right, bottom.
fn text_box(t: &Txt) -> (f64, f64, f64, f64) {
    let w = text_width(&t.s, t.size);
    let (ascent, descent) = (0.75 * t.size, 0.25 * t.size);
    if t.turned {
        // Turned to read bottom to top, centred on its anchor.
        (t.x - ascent, t.y - w / 2.0, t.x + descent, t.y + w / 2.0)
    } else {
        let left = match t.anchor.as_str() {
            "start" => t.x,
            "middle" => t.x - w / 2.0,
            _ => t.x - w,
        };
        (left, t.y - ascent, left + w, t.y + descent)
    }
}

#[derive(Debug, Clone)]
struct R {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    fill: String,
}

fn rects(svg: &str) -> Vec<R> {
    svg.lines()
        .filter(|l| l.starts_with("<rect"))
        .map(|l| R {
            x: num(l, "x"),
            y: num(l, "y"),
            w: num(l, "width"),
            h: num(l, "height"),
            fill: attr(l, "fill").unwrap().to_string(),
        })
        .filter(|r| r.fill != "#ffffff")
        .collect()
}

/// Dashed lines: x1, y1, x2, y2.
fn dashes(svg: &str) -> Vec<(f64, f64, f64, f64)> {
    svg.lines()
        .filter(|l| l.starts_with("<line") && l.contains("stroke-dasharray"))
        .map(|l| (num(l, "x1"), num(l, "y1"), num(l, "x2"), num(l, "y2")))
        .collect()
}

fn inside(r: &R, (x, y, w, h): Area) -> bool {
    let e = 0.011;
    r.x >= x - e && r.y >= y - e && r.x + r.w <= x + w + e && r.y + r.h <= y + h + e
}

fn rgb(c: &str) -> (i32, i32, i32) {
    let v = |k: usize| i32::from_str_radix(&c[k..k + 2], 16).unwrap();
    (v(1), v(3), v(5))
}

/// Sites on a toy grid: an x like -log10 p, a y like a cell count, kept
/// where both pass, and two values that are not finite.
fn sites() -> (Vec<f32>, Vec<f32>, Vec<bool>) {
    let mut seed = 7u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f64 / (1u64 << 31) as f64
    };
    let n = 3000;
    let mut xs: Vec<f32> = (0..n).map(|_| (next() * next() * 20.0) as f32).collect();
    let ys: Vec<f32> = (0..n).map(|_| (next().powi(3) * 300.0) as f32).collect();
    xs[0] = f32::NAN;
    xs[1] = f32::INFINITY;
    let kept = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| *x > 3.0 && *y > 10.0)
        .collect();
    (xs, ys, kept)
}

fn hist<'a>(xs: &'a [f32], kept: Option<&'a [bool]>) -> HistPanel<'a> {
    HistPanel {
        axis: Axis::fit(xs, false),
        bins: 30,
        values: xs,
        kept,
        edges: vec![3.0],
        x_title: "log10 p".into(),
        y_title: "sites".into(),
        title: "Title1".into(),
    }
}

fn density<'a>(xs: &'a [f32], ys: &'a [f32], kept: Option<&'a [bool]>) -> DensityPanel<'a> {
    DensityPanel {
        x: Axis::fit(xs, false),
        x_bins: 24,
        y: Axis::fit(ys, true),
        y_bins: 20,
        xs,
        ys,
        kept,
        x_edges: vec![3.0],
        y_edges: vec![10.0],
        x_title: "log10 p".into(),
        y_title: "cells".into(),
        title: "Title2".into(),
        marginals: Marginals {
            top: true,
            right: true,
        },
    }
}

#[test]
fn svg_is_exactly_the_size_asked() {
    let (xs, ys, kept) = sites();
    let look = Look::default();
    let a = hist(&xs, Some(&kept)).svg(&look, 173.5, 101.25);
    assert!(a.starts_with(r#"<svg xmlns="http://www.w3.org/2000/svg" width="173.5" height="101.25" viewBox="0 0 173.5 101.25">"#));
    let b = density(&xs, &ys, Some(&kept)).svg(&look, 130.0, 266.0);
    assert!(b.contains(r#"width="130" height="266" viewBox="0 0 130 266""#));
}

#[test]
fn every_text_is_set_at_the_look_font() {
    let (xs, ys, kept) = sites();
    let look = Look {
        font: 6.5,
        ..Look::default()
    };
    assert!(look.title_size() > look.font);
    for svg in [
        hist(&xs, Some(&kept)).svg(&look, 240.0, 180.0),
        density(&xs, &ys, Some(&kept)).svg(&look, 240.0, 220.0),
    ] {
        let t = texts(&svg);
        assert!(t.len() > 6, "{t:?}");
        let titles: Vec<&Txt> = t.iter().filter(|t| t.bold).collect();
        assert_eq!(titles.len(), 1, "{t:?}");
        assert!(titles[0].s.starts_with("Title"));
        assert_eq!(titles[0].size, look.title_size());
        for t in t.iter().filter(|t| !t.bold) {
            assert_eq!(t.size, look.font, "{t:?}");
        }
    }
}

#[test]
fn panel_only_draws_no_title_or_key() {
    let (xs, ys, kept) = sites();
    let full = Look::default();
    let only = Look {
        panel_only: true,
        ..Look::default()
    };
    for (a, b) in [
        (
            hist(&xs, Some(&kept)).svg(&full, 240.0, 180.0),
            hist(&xs, Some(&kept)).svg(&only, 240.0, 180.0),
        ),
        (
            density(&xs, &ys, Some(&kept)).svg(&full, 240.0, 220.0),
            density(&xs, &ys, Some(&kept)).svg(&only, 240.0, 220.0),
        ),
    ] {
        let said = |svg: &str, s: &str| texts(svg).iter().any(|t| t.s == s);
        // With everything: a title and a key.
        assert!(texts(&a).iter().any(|t| t.s.starts_with("Title")));
        assert!(said(&a, "kept") && said(&a, "removed"));
        // Panel only: the axis titles and ticks, nothing else.
        let t = texts(&b);
        assert!(t.iter().all(|t| !t.bold && t.size == only.font), "{t:?}");
        assert!(!t.iter().any(|t| t.s.starts_with("Title")));
        assert!(!said(&b, "kept") && !said(&b, "removed"));
        assert!(said(&b, "log10 p"));
        let others: Vec<&Txt> = t
            .iter()
            .filter(|t| !["log10 p", "sites", "cells"].contains(&t.s.as_str()))
            .collect();
        assert!(
            others.iter().all(|t| t
                .s
                .chars()
                .all(|c| c.is_ascii_digit() || ".kM-".contains(c))),
            "{others:?}"
        );
    }
}

#[test]
fn stacks_add_up_to_the_axis_counts() {
    let (xs, _, kept) = sites();
    let axis = Axis::fit(&xs, false);
    let s = stacks(&axis, 30, &xs, Some(&kept));
    let all = axis.counts(&xs, 30, |_| true);
    let k = axis.counts(&xs, 30, |i| kept[i]);
    assert_eq!(s.len(), 30);
    for b in 0..30 {
        assert_eq!(s[b].0 + s[b].1, all[b]);
        assert_eq!(s[b].0, k[b]);
    }
    // Not finite: left out.
    assert_eq!(all.iter().sum::<usize>(), xs.len() - 2);
    // No mask: all kept.
    let none = stacks(&axis, 30, &xs, None);
    assert!(none.iter().zip(&all).all(|(s, a)| s.0 == *a && s.1 == 0));
    // Drawn: one kept bar per bin with kept values, heights by count.
    let look = Look::default();
    let p = hist(&xs, Some(&kept));
    let svg = p.svg(&look, 240.0, 180.0);
    let area = p.plot_area(&look, (0.0, 0.0, 240.0, 180.0));
    let bars: Vec<R> = rects(&svg)
        .into_iter()
        .filter(|r| inside(r, area))
        .collect();
    let kept_bars: Vec<&R> = bars.iter().filter(|r| r.fill == look.kept).collect();
    assert_eq!(kept_bars.len(), s.iter().filter(|s| s.0 > 0).count());
    let tallest = |fill: &str| {
        bars.iter()
            .filter(|r| r.fill == fill)
            .map(|r| r.h)
            .fold(0.0, f64::max)
    };
    let most_kept = s.iter().map(|s| s.0).max().unwrap() as f64;
    let most_removed = s.iter().map(|s| s.1).max().unwrap() as f64;
    let ratio = tallest(&look.kept) / tallest(&look.removed);
    assert!((ratio - most_kept / most_removed).abs() < 0.02, "{ratio}");
}

#[test]
fn density_grid_counts_each_finite_pair_once() {
    let (xs, ys, kept) = sites();
    let p = density(&xs, &ys, Some(&kept));
    let g = p.grid();
    assert_eq!(g.len(), 24 * 20);
    let total: usize = g.iter().map(|c| c.0 + c.1).sum();
    assert_eq!(total, xs.len() - 2);
    let i = 5;
    let cell = p.y.bin(ys[i] as f64, 20) * 24 + p.x.bin(xs[i] as f64, 24);
    assert!(g[cell].0 + g[cell].1 > 0);
    let k: usize = g.iter().map(|c| c.0).sum();
    assert_eq!(k, kept.iter().filter(|&&b| b).count());
}

#[test]
fn marginals_line_up_with_the_squares() {
    let (xs, ys, kept) = sites();
    let look = Look::default();
    let p = density(&xs, &ys, Some(&kept));
    let (w, h) = (240.0, 220.0);
    let svg = p.svg(&look, w, h);
    let lay = p.layout(&look, (0.0, 0.0, w, h));
    let (top, right) = (lay.top.unwrap(), lay.right.unwrap());
    let (px, py, pw, ph) = lay.plot;
    // The marginals span the plot's x and y exactly.
    assert_eq!((top.0, top.2), (px, pw));
    assert_eq!((right.1, right.3), (py, ph));
    assert!(top.1 + top.3 <= py && right.0 >= px + pw);
    let all = rects(&svg);
    let squares: Vec<&R> = all.iter().filter(|r| inside(r, lay.plot)).collect();
    let tops: Vec<&R> = all.iter().filter(|r| inside(r, top)).collect();
    let rights: Vec<&R> = all.iter().filter(|r| inside(r, right)).collect();
    let key = |a: f64, b: f64| ((a * 100.0).round() as i64, (b * 100.0).round() as i64);
    let set = |v: &[&R], f: &dyn Fn(&R) -> (i64, i64)| {
        let mut s: Vec<(i64, i64)> = v.iter().map(|r| f(r)).collect();
        s.sort();
        s.dedup();
        s
    };
    let cols = set(&squares, &|r| key(r.x, r.w));
    assert_eq!(cols, set(&tops, &|r| key(r.x, r.w)));
    assert_eq!(
        set(&squares, &|r| key(r.y, r.h)),
        set(&rights, &|r| key(r.y, r.h))
    );
    // Columns sit on the axis' bins.
    for (x, w) in &cols {
        let i = ((*x as f64 / 100.0 - px) / (pw / 24.0)).round();
        assert!((px + i * pw / 24.0 - *x as f64 / 100.0).abs() < 0.011);
        assert!((pw / 24.0 - *w as f64 / 100.0).abs() < 0.011);
    }
    // Kept and removed stacked in the look's colours, lengths by count.
    assert!(tops.iter().any(|r| r.fill == look.kept));
    assert!(tops.iter().any(|r| r.fill == look.removed));
    let xs_stack = stacks(&p.x, 24, &xs, Some(&kept));
    let most = xs_stack.iter().map(|s| s.0 + s.1).max().unwrap() as f64;
    for (i, s) in xs_stack.iter().enumerate() {
        let x0 = px + i as f64 * pw / 24.0;
        let len: f64 = tops
            .iter()
            .filter(|r| (r.x - x0).abs() < 0.011)
            .map(|r| r.h)
            .sum();
        let want = (s.0 + s.1) as f64 / most * top.3;
        assert!((len - want).abs() < 0.05, "bin {i}: {len} vs {want}");
    }
    let ys_stack = stacks(&p.y, 20, &ys, Some(&kept));
    let most = ys_stack.iter().map(|s| s.0 + s.1).max().unwrap() as f64;
    for (j, s) in ys_stack.iter().enumerate() {
        let y0 = py + ph - (j + 1) as f64 * ph / 20.0;
        let len: f64 = rights
            .iter()
            .filter(|r| (r.y - y0).abs() < 0.011)
            .map(|r| r.w)
            .sum();
        let want = (s.0 + s.1) as f64 / most * right.2;
        assert!((len - want).abs() < 0.05, "row {j}: {len} vs {want}");
    }
    // The right marginal's bars grow from its left side.
    assert!(rights.iter().all(|r| r.x >= right.0 - 0.011));
    assert!(rights.iter().any(|r| (r.x - right.0).abs() < 0.011));
    // The edges run through the marginals.
    let d = dashes(&svg);
    let ex = px + p.x.frac(3.0) * pw;
    assert!(d.iter().any(|l| (l.0 - ex).abs() < 0.011
        && l.1.min(l.3) <= top.1 + 0.011
        && l.1.max(l.3) >= py + ph - 0.011));
    let ey = py + ph - p.y.frac(10.0) * ph;
    assert!(d.iter().any(|l| (l.1 - ey).abs() < 0.011
        && l.0.min(l.2) <= px + 0.011
        && l.0.max(l.2) >= right.0 + right.2 - 0.011));
}

#[test]
fn squares_deepen_with_count_and_take_the_majority_hue() {
    let look = Look::default();
    let depth = |c: &str| {
        let (r, g, b) = rgb(c);
        765 - (r + g + b)
    };
    let few = square_colour(&look, 1, 0, 100);
    let many = square_colour(&look, 60, 0, 100);
    assert!(depth(&few) > 0 && depth(&few) < depth(&many));
    assert_eq!(square_colour(&look, 100, 0, 100), look.kept);
    assert_eq!(square_colour(&look, 30, 70, 100), look.removed);
    // A mix of white and the majority's colour: on the line between them.
    let on_line = |c: &str, full: &str| {
        let (c, f) = (rgb(c), rgb(full));
        let t = (255 - c.0) as f64 / (255 - f.0) as f64;
        [(c.1, f.1), (c.2, f.2)]
            .iter()
            .all(|(a, b)| ((255 - a) as f64 - t * (255 - b) as f64).abs() <= 1.5)
    };
    let mostly_removed = square_colour(&look, 3, 7, 100);
    assert!(on_line(&mostly_removed, &look.removed), "{mostly_removed}");
    assert!(!on_line(&mostly_removed, &look.kept));
    assert!(on_line(&square_colour(&look, 7, 3, 100), &look.kept));
    // Drawn: the fullest square at full depth.
    let (xs, ys, kept) = sites();
    let p = DensityPanel {
        kept: None,
        ..density(&xs, &ys, Some(&kept))
    };
    let svg = p.svg(&look, 240.0, 220.0);
    let lay = p.layout(&look, (0.0, 0.0, 240.0, 220.0));
    let fills: Vec<String> = rects(&svg)
        .into_iter()
        .filter(|r| inside(r, lay.plot))
        .map(|r| r.fill)
        .collect();
    let g = p.grid();
    let most = g.iter().map(|c| c.0).max().unwrap();
    assert_eq!(fills.len(), g.iter().filter(|c| c.0 > 0).count());
    assert!(fills.contains(&look.kept));
    assert!(fills.contains(&square_colour(&look, 1, 0, most)));
}

#[test]
fn gray_look_draws_in_grays() {
    let (xs, ys, kept) = sites();
    let look = Look::gray();
    let grey = |c: &str| {
        let (r, g, b) = rgb(c);
        r == g && g == b
    };
    assert!(grey(&look.kept) && grey(&look.removed));
    assert!(rgb(&look.kept).0 < rgb(&look.removed).0);
    for svg in [
        hist(&xs, Some(&kept)).svg(&look, 240.0, 180.0),
        density(&xs, &ys, Some(&kept)).svg(&look, 240.0, 220.0),
    ] {
        assert!(svg.contains(&format!("fill=\"{}\"", look.kept)));
        assert!(svg.contains(&format!("fill=\"{}\"", look.removed)));
        assert!(!svg.contains(BAR) && !svg.contains(ACCENT));
        assert!(rects(&svg).iter().all(|r| grey(&r.fill)), "{svg}");
    }
}

#[test]
fn a_tiny_panel_keeps_every_label_inside() {
    let (xs, ys, kept) = sites();
    let (w, h) = (130.0, 100.0);
    for font in [8.0, fit_font(w, h)] {
        for panel_only in [false, true] {
            let look = Look {
                font,
                panel_only,
                ..Look::default()
            };
            let mut flat = density(&xs, &ys, Some(&kept));
            flat.marginals = Marginals::default();
            for svg in [
                hist(&xs, Some(&kept)).svg(&look, w, h),
                hist(&xs, None).svg(&look, w, h),
                density(&xs, &ys, Some(&kept)).svg(&look, w, h),
                flat.svg(&look, w, h),
            ] {
                let t = texts(&svg);
                assert!(t.len() >= 4);
                for t in &t {
                    let (l, tp, r, b) = text_box(t);
                    let e = 0.011;
                    assert!(
                        l >= -e && tp >= -e && r <= w + e && b <= h + e,
                        "font {font}: {t:?} at {:?}",
                        (l, tp, r, b)
                    );
                }
                assert!(rects(&svg).iter().all(|r| inside(r, (0.0, 0.0, w, h))));
            }
        }
    }
}

#[test]
fn margins_follow_the_font() {
    let (xs, ys, kept) = sites();
    let small = Look {
        font: 5.0,
        ..Look::default()
    };
    let large = Look {
        font: 9.0,
        ..Look::default()
    };
    let area = (0.0, 0.0, 130.0, 100.0);
    let h = hist(&xs, Some(&kept));
    let (a, b) = (h.plot_area(&small, area), h.plot_area(&large, area));
    assert!(a.2 > b.2 && a.3 > b.3, "{a:?} {b:?}");
    let d = density(&xs, &ys, Some(&kept));
    let (a, b) = (d.layout(&small, area).plot, d.layout(&large, area).plot);
    assert!(a.2 > b.2 && a.3 > b.3, "{a:?} {b:?}");
    // Not the old fixed margins: a 5 pt panel leaves less than 44 on the left.
    assert!(h.plot_area(&small, area).0 < 30.0);
}

#[test]
fn fit_font_follows_the_shorter_side() {
    assert!((fit_font(130.0, 100.0) - 100.0 / 18.0).abs() < 1e-9);
    assert_eq!(fit_font(1000.0, 500.0), 8.0);
    assert_eq!(fit_font(40.0, 60.0), 5.0);
}

#[test]
fn pdf_keeps_text_as_text() {
    let mut db = usvg::fontdb::Database::new();
    db.load_system_fonts();
    let has_font = db.faces().any(|f| {
        f.families
            .iter()
            .any(|(n, _)| ["Helvetica", "Arial", "DejaVu Sans"].contains(&n.as_str()))
    });
    if !has_font {
        return;
    }
    let (xs, ys, kept) = sites();
    let svg = density(&xs, &ys, Some(&kept)).svg(&Look::default(), 180.0, 160.0);
    let dir = tempfile::tempdir().unwrap();
    let name = dir.path().join("panel.pdf").to_string_lossy().into_owned();
    save(&svg, &name).unwrap();
    let pdf = std::fs::read(&name).unwrap();
    let has = |s: &[u8]| pdf.windows(s.len()).any(|w| w == s);
    assert!(has(b"/Font"), "no font in the PDF");
    assert!(has(b"/ToUnicode"), "text not kept as text");
    // An SVG save writes the text as it is.
    let name = dir.path().join("panel.svg").to_string_lossy().into_owned();
    save(&svg, &name).unwrap();
    assert!(std::fs::read_to_string(&name)
        .unwrap()
        .contains(">log10 p</text>"));
}

/// In grays, where only the depth tells them apart, the lightest kept
/// square is darker than the darkest removed one: kept dark, removed
/// light, at any count.
#[test]
fn gray_kept_squares_stay_darker_than_removed_ones() {
    let look = Look::gray();
    let lightest_kept = rgb(&square_colour(&look, 1, 0, 1000)).0;
    let darkest_removed = rgb(&square_colour(&look, 0, 1000, 1000)).0;
    assert!(
        lightest_kept < darkest_removed,
        "kept {lightest_kept}, removed {darkest_removed}"
    );
    // Still deeper with more sites, both ways.
    assert!(rgb(&square_colour(&look, 500, 0, 1000)).0 < lightest_kept);
    let lightest_removed = rgb(&square_colour(&look, 0, 1, 1000)).0;
    assert!(darkest_removed < lightest_removed);
}

/// No bins is one bin, in the histogram as in the 2D panel, not a panic.
#[test]
fn no_bins_draw_as_one() {
    let (xs, ys, kept) = sites();
    let h = HistPanel {
        bins: 0,
        ..hist(&xs, Some(&kept))
    };
    assert!(h.svg(&Look::default(), 100.0, 100.0).contains("<rect"));
    let d = DensityPanel {
        x_bins: 0,
        y_bins: 0,
        ..density(&xs, &ys, Some(&kept))
    };
    assert!(d.svg(&Look::default(), 100.0, 100.0).contains("<rect"));
}

/// A threshold outside the axis is not drawn: at the axis end it would
/// read as a cut at the largest value.
#[test]
fn edges_outside_the_axis_are_left_out() {
    let (xs, ys, kept) = sites();
    let look = Look::default();
    let inside_only = hist(&xs, Some(&kept)).svg(&look, 240.0, 180.0);
    let with_far = HistPanel {
        edges: vec![3.0, 1e6, -1e6],
        ..hist(&xs, Some(&kept))
    }
    .svg(&look, 240.0, 180.0);
    assert_eq!(dashes(&with_far).len(), dashes(&inside_only).len());
    let far = DensityPanel {
        x_edges: vec![3.0, 1e6],
        y_edges: vec![10.0, 1e9],
        ..density(&xs, &ys, Some(&kept))
    }
    .svg(&look, 240.0, 220.0);
    let near = density(&xs, &ys, Some(&kept)).svg(&look, 240.0, 220.0);
    assert_eq!(dashes(&far).len(), dashes(&near).len());
}

/// A saved SVG states its size in points, so an editor opens it at the
/// size asked; the drawing inside is unchanged.
#[test]
fn a_saved_svg_is_sized_in_points() {
    let tmp = tempfile::tempdir().unwrap();
    let (xs, _, kept) = sites();
    let svg = hist(&xs, Some(&kept)).svg(&Look::default(), 129.6, 115.2);
    let path = tmp.path().join("panel.svg");
    save(&svg, path.to_str().unwrap()).unwrap();
    let written = std::fs::read_to_string(&path).unwrap();
    let root = written.lines().next().unwrap();
    assert!(
        root.contains(r#"width="129.6pt""#) && root.contains(r#"height="115.2pt""#),
        "{root}"
    );
    assert!(root.contains(r#"viewBox="0 0 129.6 115.2""#), "{root}");
    assert_eq!(
        written.lines().skip(1).collect::<Vec<_>>(),
        svg.lines().skip(1).collect::<Vec<_>>()
    );
}
