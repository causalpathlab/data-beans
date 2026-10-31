//! The import preview: what a `from-*` build made, before it is kept. The
//! rows' and columns' non-zeros as histograms on log bins, with where the
//! squeeze cutoffs fall and how many each would drop.

use super::Squeeze;
use crate::interactive::plan_tui::{plan_in, Control, Plan, PlanView, Planner, Setting};
use crate::interactive::ui::Tui;

/// Bars across a histogram line.
const BINS: usize = 48;

/// `values` on log bins from 0 to the largest, one bar a bin, with `┃`
/// where `cutoff` falls when it is given: what lies left of it is dropped.
fn spark(values: &[usize], cutoff: Option<usize>) -> String {
    let top = values.iter().copied().max().unwrap_or(0);
    let scale = ((top + 1) as f64).ln().max(f64::MIN_POSITIVE);
    let bin =
        |v: usize| (((v + 1) as f64).ln() / scale * BINS as f64).min((BINS - 1) as f64) as usize;
    let mut counts = [0usize; BINS];
    for &v in values {
        counts[bin(v)] += 1;
    }
    let most = counts.iter().copied().max().unwrap_or(0).max(1);
    const BARS: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    // The first bin whose values reach the cutoff.
    let edge = cutoff.map(|c| (0..BINS).find(|&b| bin(c) <= b).unwrap_or(BINS));
    let mut line = String::new();
    for (b, &n) in counts.iter().enumerate() {
        if edge == Some(b) {
            line.push('┃');
        }
        let h = if n == 0 {
            0
        } else {
            1 + (n * 7).div_ceil(most).min(7)
        };
        line.push(BARS[h.min(8)]);
    }
    format!("0 {line} {top}")
}

/// What `squeeze` keeps of the build: its table, histograms and notes.
fn plan_of(file: &str, rows: &[usize], columns: &[usize], squeeze: Squeeze) -> Plan {
    let axis = |name: &str, values: &[usize], cutoff: usize| -> Vec<String> {
        let dropped = match squeeze.on {
            true => values.iter().filter(|&&v| v < cutoff).count(),
            false => 0,
        };
        vec![
            name.into(),
            values.len().to_string(),
            (values.len() - dropped).to_string(),
            dropped.to_string(),
            match squeeze.on {
                true => format!("< {cutoff} non-zeros"),
                false => "off".into(),
            },
        ]
    };
    let cut = |c: usize| squeeze.on.then_some(c);
    let entries: usize = columns.iter().sum();
    let mut notes = vec![
        format!("{file}: {} entries", entries),
        format!("rows    {}", spark(rows, cut(squeeze.rows))),
        format!("columns {}", spark(columns, cut(squeeze.columns))),
    ];
    if squeeze.on {
        notes.push(
            "counts of each axis are before the other is cut, so a little more may go".into(),
        );
    }
    Plan {
        columns: ["", "total", "kept", "dropped", "dropped when"]
            .map(String::from)
            .to_vec(),
        rows: vec![
            axis("rows", rows, squeeze.rows),
            axis("columns", columns, squeeze.columns),
        ],
        notes,
        blocked: None,
    }
}

/// A cutoff from a setting: a whole number of non-zeros, 0 when unset.
fn cutoff(s: &Setting, what: &str) -> anyhow::Result<usize> {
    match s.number() {
        None => Ok(0),
        Some(v) if v.fract() == 0.0 && v >= 0.0 => Ok(v as usize),
        Some(v) => anyhow::bail!("{what} is a whole number of non-zeros; not {v}"),
    }
}

fn squeeze_of(s: &[Setting], base: Squeeze) -> anyhow::Result<Squeeze> {
    Ok(Squeeze {
        on: s[0].flag(),
        rows: cutoff(&s[1], "the row cutoff (r)")?,
        columns: cutoff(&s[2], "the column cutoff (c)")?,
        ..base
    })
}

/// Show what the build at `file` holds; the squeeze to keep it with, or
/// `None` to discard it.
pub(super) fn preview(
    tui: &mut Tui<'_>,
    command: &str,
    file: &str,
    rows: &[usize],
    columns: &[usize],
    squeeze: Squeeze,
) -> anyhow::Result<Option<Squeeze>> {
    let controls = vec![
        Control::flag('s', "squeeze", squeeze.on),
        Control::number('r', "row cutoff", Some(squeeze.rows as f64)),
        Control::number('c', "column cutoff", Some(squeeze.columns as f64)),
    ];
    let planner: Planner =
        Box::new(move |s: &[Setting]| Ok(plan_of(file, rows, columns, squeeze_of(s, squeeze)?)));
    let title = format!("{command} · {file}");
    let view = PlanView::new("import", &title, "keep", controls, planner).cancel_label("discard");
    match plan_in(tui, view)? {
        Some(s) => squeeze_of(&s, squeeze).map(Some),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn squeeze(on: bool, rows: usize, columns: usize) -> Squeeze {
        Squeeze {
            on,
            rows,
            columns,
            block_size: None,
        }
    }

    #[test]
    fn the_plan_counts_what_the_cutoffs_drop() {
        let rows = [0, 1, 2, 5, 50];
        let columns = [3, 3, 10, 42];
        let off = plan_of("x.zarr", &rows, &columns, squeeze(false, 3, 4));
        assert_eq!(off.rows[0][1..4], ["5", "5", "0"].map(String::from));
        assert_eq!(off.notes[0], "x.zarr: 58 entries");
        let on = plan_of("x.zarr", &rows, &columns, squeeze(true, 3, 4));
        assert_eq!(
            on.rows[0][1..],
            ["5", "2", "3", "< 3 non-zeros"].map(String::from)
        );
        assert_eq!(on.rows[1][1..4], ["4", "2", "2"].map(String::from));
        assert!(on.notes[1].contains('┃'), "{}", on.notes[1]);
        assert!(!off.notes[1].contains('┃'));
    }

    #[test]
    fn a_histogram_line_spans_zero_to_the_most() {
        let line = spark(&[1, 1, 1, 100, 1000], None);
        assert!(line.starts_with("0 ") && line.ends_with(" 1000"), "{line}");
        assert_eq!(line.chars().count(), 2 + BINS + 5);
        // The cutoff mark sits left of the bars it keeps.
        let marked = spark(&[1, 1, 1, 100, 1000], Some(50));
        let at = marked.chars().position(|c| c == '┃').unwrap();
        assert!(marked.chars().skip(at + 1).any(|c| c != ' '));
        assert!(spark(&[], Some(3)).starts_with("0 "));
    }

    #[test]
    fn settings_read_as_a_squeeze() {
        let base = squeeze(false, 1, 1);
        let n = |v: Option<f64>| Setting::Number(v);
        let s = squeeze_of(&[Setting::Flag(true), n(Some(10.0)), n(None)], base).unwrap();
        assert_eq!((s.on, s.rows, s.columns), (true, 10, 0));
        assert!(squeeze_of(&[Setting::Flag(true), n(Some(1.5)), n(None)], base).is_err());
    }
}
