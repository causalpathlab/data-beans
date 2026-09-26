use super::*;
use crate::interactive::ui::Screen;
use ratatui::backend::TestBackend;
use ratatui::crossterm::event::KeyModifiers;
use ratatui::Terminal;

fn press(e: &mut StatExplorer<'_>, code: KeyCode) {
    e.handle_key(KeyEvent::new(code, KeyModifiers::NONE));
}

fn type_text(e: &mut StatExplorer<'_>, text: &str) {
    for c in text.chars() {
        press(e, KeyCode::Char(c));
    }
}

/// Placeholder entries: GENE1..GENE9 and CT1..CT3 with distinct stats.
fn rows() -> Dataset {
    let names: Vec<Box<str>> = (1..=9)
        .map(|i| format!("GENE{i}"))
        .chain((1..=3).map(|i| format!("CT{i}")))
        .map(String::into_boxed_str)
        .collect();
    let n = names.len();
    let nnz: Vec<f32> = (0..n).map(|i| (10 * (i + 1)) as f32).collect();
    let sum: Vec<f32> = (0..n).map(|i| (1000 - 7 * i * i) as f32).collect();
    let mean: Vec<f32> = sum.iter().map(|s| s / 40.0).collect();
    let sd: Vec<f32> = (0..n).map(|i| 0.5 + i as f32 * 0.25).collect();
    Dataset {
        names,
        values: [nnz, sum, mean, sd],
    }
}

/// Placeholder columns CELL1..CELL5.
fn columns() -> Dataset {
    let names = (1..=5)
        .map(|i| format!("CELL{i}").into_boxed_str())
        .collect();
    let nnz: Vec<f32> = (1..=5).map(|i| (100 * i) as f32).collect();
    Dataset {
        values: [nnz.clone(), nnz.clone(), nnz.clone(), nnz],
        names,
    }
}

fn explorer() -> StatExplorer<'static> {
    StatExplorer::new("input", Side::Rows, rows(), None, None, Purpose::Explore)
}

/// A values reader over placeholder data: entry `i` has value `10 i + j`
/// against other-side entry `j`, of which there are five.
fn reader<'a>() -> ValuesReader<'a> {
    Box::new(|_, entries: &[usize]| {
        Ok(Values {
            names: (1..=5)
                .map(|j| format!("CELL{j}").into_boxed_str())
                .collect(),
            columns: entries
                .iter()
                .map(|&i| (0..5).map(|j| (10 * i + j) as f32).collect())
                .collect(),
        })
    })
}

fn picker<'a>() -> StatExplorer<'a> {
    StatExplorer::new(
        "input",
        Side::Rows,
        rows(),
        None,
        Some(reader()),
        Purpose::Pick { verb: "subset" },
    )
}

fn shown_names<'e>(e: &'e StatExplorer<'_>) -> Vec<&'e str> {
    e.view.iter().map(|&i| &*e.names[i]).collect()
}

fn screen_text(e: &mut StatExplorer<'_>, w: u16, h: u16) -> String {
    let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
    term.draw(|f| e.render(f)).unwrap();
    term.backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol())
        .collect()
}

#[test]
fn starts_sorted_by_nnz_descending_on_the_top_entry() {
    let e = explorer();
    assert_eq!(shown_names(&e)[..2], ["CT3", "CT2"]);
    assert_eq!(e.selected(), Some(e.view[0]));
    assert_eq!(e.hist.counts.iter().sum::<usize>(), 12);
}

#[test]
fn stat_keys_sort_and_toggle_order() {
    let mut e = explorer();
    press(&mut e, KeyCode::Char('2'));
    assert_eq!(e.stat, 1);
    assert_eq!(shown_names(&e)[0], "GENE1", "largest sum first");
    press(&mut e, KeyCode::Char('2'));
    assert_eq!(shown_names(&e)[0], "CT3", "same key flips the order");
    press(&mut e, KeyCode::Char('0'));
    assert_eq!(shown_names(&e)[0], "CT1", "name order ascending");
}

#[test]
fn filter_is_a_case_insensitive_regex_with_literal_fallback() {
    let mut e = explorer();
    press(&mut e, KeyCode::Char('/'));
    type_text(&mut e, "^ct");
    assert_eq!(e.view.len(), 3);
    let shown: usize = e.shown.as_ref().unwrap().iter().sum();
    assert_eq!(shown, 3, "the subset is binned over the same bins");

    // An invalid regex matches literally rather than hiding everything.
    type_text(&mut e, "(");
    assert!(e.view.is_empty());
    press(&mut e, KeyCode::Esc);
    assert_eq!(e.view.len(), 12);
    assert!(e.shown.is_none());
}

#[test]
fn selection_follows_its_entry_across_resorts() {
    let mut e = explorer();
    press(&mut e, KeyCode::Down);
    let before = e.selected();
    press(&mut e, KeyCode::Char('4'));
    assert_eq!(e.selected(), before);
    press(&mut e, KeyCode::Char('4'));
    assert_eq!(e.selected(), before, "a flip keeps it too");
}

#[test]
fn scales_rebin_every_statistic() {
    let mut e = explorer();
    for s in 0..4 {
        e.choose_stat(s);
        for _ in 0..3 {
            press(&mut e, KeyCode::Char('x'));
            assert_eq!(
                e.hist.counts.iter().sum::<usize>(),
                12,
                "stat {s} {:?}",
                e.x_scale
            );
        }
    }
}

#[test]
fn renders_wide_and_narrow() {
    for (w, h) in [(140, 30), (80, 40), (20, 6)] {
        let mut e = explorer();
        press(&mut e, KeyCode::End);
        let text = screen_text(&mut e, w, h);
        if w >= 80 {
            assert!(text.contains("nnz ▼"));
            assert!(text.contains("GENE1"), "last row scrolled into view");
        }
    }
}

#[test]
fn space_marks_and_a_marks_everything_shown() {
    let mut e = explorer();
    press(&mut e, KeyCode::Char(' '));
    assert_eq!(e.marked_entries(), [e.view[0]]);
    assert_eq!(e.cursor, 1, "marking moves down");

    press(&mut e, KeyCode::Char('/'));
    type_text(&mut e, "^ct");
    press(&mut e, KeyCode::Enter);
    press(&mut e, KeyCode::Char('a'));
    assert_eq!(e.marked_entries().len(), 3, "CT3 was already marked");
    press(&mut e, KeyCode::Char('a'));
    assert!(
        e.marked_entries().is_empty(),
        "all shown marked: a unmarks them"
    );

    press(&mut e, KeyCode::Char(' '));
    press(&mut e, KeyCode::Char('u'));
    assert!(e.marked_entries().is_empty());
    assert!(screen_text(&mut e, 140, 30).contains("match /^ct/"));
}

#[test]
fn enter_hands_back_the_marked_entries_when_picking() {
    let mut e = picker();
    press(&mut e, KeyCode::Enter);
    assert!(!e.done(), "nothing marked yet");
    assert!(e.status.as_deref().unwrap().contains("mark entries"));

    press(&mut e, KeyCode::Char(' '));
    press(&mut e, KeyCode::Char(' '));
    press(&mut e, KeyCode::Enter);
    assert!(e.done());
    let picked = e.picked.clone().unwrap();
    assert_eq!(picked.rows, [10, 11], "CT3 and CT2, in original order");
    assert!(picked.columns.is_empty());
}

#[test]
fn q_and_ctrl_c_hand_back_nothing() {
    for quit in [true, false] {
        let mut e = picker();
        press(&mut e, KeyCode::Char(' '));
        if quit {
            press(&mut e, KeyCode::Char('q'));
        } else {
            e.interrupt();
        }
        assert!(e.done());
        assert!(e.picked.is_none());
    }
}

#[test]
fn values_view_reads_the_marked_entries_and_sorts_by_a_column() {
    let mut e = picker();
    press(&mut e, KeyCode::Char('0')); // name order: CT1, CT2, CT3, GENE1, ...
    press(&mut e, KeyCode::Home);
    press(&mut e, KeyCode::Char(' ')); // CT1 (entry 9)
    press(&mut e, KeyCode::Char(' ')); // CT2 (entry 10)
    press(&mut e, KeyCode::Char('v'));
    assert!(e.pending_work().unwrap().contains("reading 2 rows"));
    e.do_work();
    assert!(matches!(e.mode, Mode::Values));

    let view = e.values_view.as_ref().unwrap();
    assert_eq!(&*view.labels[0], "CT1");
    assert_eq!(view.columns[0][4], 94.0);
    assert_eq!(view.order[0], 4, "sorted by the first column, descending");

    press(&mut e, KeyCode::Right);
    press(&mut e, KeyCode::Char('s'));
    press(&mut e, KeyCode::Char('s'));
    let view = e.values_view.as_ref().unwrap();
    assert_eq!(view.sort, Some((1, false)));
    assert_eq!(view.order[0], 0, "ascending after a second s");

    let text = screen_text(&mut e, 100, 20);
    assert!(text.contains("values · 2 rows × 5 columns"));
    assert!(text.contains("CT2 ▲"));

    press(&mut e, KeyCode::Esc);
    assert!(matches!(e.mode, Mode::Browse));
    assert_eq!(e.marked_entries(), [9, 10], "marks survive the values view");
}

#[test]
fn values_view_without_marks_shows_the_selected_entry() {
    let mut e = picker();
    press(&mut e, KeyCode::Char('v'));
    e.do_work();
    assert_eq!(e.values_view.as_ref().unwrap().labels.len(), 1);
}

#[test]
fn w_saves_marked_names_and_the_values_table() {
    let dir = tempfile::tempdir().unwrap();
    let names_path = dir.path().join("picked.txt");
    let values_path = dir.path().join("values.tsv");
    let mut e = picker();
    press(&mut e, KeyCode::Char('0'));
    press(&mut e, KeyCode::Home);
    press(&mut e, KeyCode::Char(' '));

    press(&mut e, KeyCode::Char('w'));
    press(&mut e, KeyCode::Backspace);
    for _ in 0..20 {
        press(&mut e, KeyCode::Backspace);
    }
    type_text(&mut e, names_path.to_str().unwrap());
    press(&mut e, KeyCode::Enter);
    assert_eq!(std::fs::read_to_string(&names_path).unwrap().trim(), "CT1");

    press(&mut e, KeyCode::Char('v'));
    e.do_work();
    press(&mut e, KeyCode::Char('w'));
    for _ in 0..20 {
        press(&mut e, KeyCode::Backspace);
    }
    type_text(&mut e, values_path.to_str().unwrap());
    press(&mut e, KeyCode::Enter);
    assert!(
        matches!(e.mode, Mode::Values),
        "saving returns to the values"
    );
    let tsv = std::fs::read_to_string(&values_path).unwrap();
    let lines: Vec<&str> = tsv.lines().collect();
    assert_eq!(lines[0], "column\tCT1");
    assert_eq!(lines.len(), 6);
    assert_eq!(lines[1], "CELL5\t94");
}

#[test]
fn picking_never_switches_sides() {
    let loader: Loader<'_> = Box::new(|_| Ok(columns()));
    let mut e = StatExplorer::new(
        "input",
        Side::Rows,
        rows(),
        Some(loader),
        None,
        Purpose::Pick { verb: "take" },
    );
    press(&mut e, KeyCode::Tab);
    assert!(e.pending_work().is_none());
    assert_eq!(e.side, Side::Rows);
}

#[test]
fn tab_computes_the_other_side_once_and_toggles_with_its_marks() {
    let calls = std::cell::Cell::new(0);
    let loader: Loader<'_> = Box::new(|side| {
        assert_eq!(side, Side::Columns);
        calls.set(calls.get() + 1);
        Ok(columns())
    });
    let mut e = StatExplorer::new(
        "input",
        Side::Rows,
        rows(),
        Some(loader),
        None,
        Purpose::Explore,
    );
    press(&mut e, KeyCode::Char(' '));
    press(&mut e, KeyCode::Char('/'));
    press(&mut e, KeyCode::Char('1'));
    press(&mut e, KeyCode::Enter);

    press(&mut e, KeyCode::Tab);
    assert!(e.pending_work().unwrap().contains("computing columns"));
    e.do_work();
    assert_eq!(e.side, Side::Columns);
    assert_eq!(calls.get(), 1);
    // The filter carries over: "1" matches CELL1 only.
    assert_eq!(shown_names(&e), ["CELL1"]);
    assert!(e.marked_entries().is_empty(), "each side has its own marks");

    press(&mut e, KeyCode::Tab);
    assert_eq!(e.side, Side::Rows);
    assert_eq!(e.marked_entries().len(), 1, "the row mark is still there");
    press(&mut e, KeyCode::Tab);
    assert_eq!(e.side, Side::Columns);
    assert!(e.pending_work().is_none());
    assert_eq!(calls.get(), 1, "computed once");
}

#[test]
fn a_failed_load_stays_put_and_says_why() {
    let loader: Loader<'_> = Box::new(|_| Err(anyhow::anyhow!("no data")));
    let mut e = StatExplorer::new(
        "input",
        Side::Rows,
        rows(),
        Some(loader),
        None,
        Purpose::Explore,
    );
    press(&mut e, KeyCode::Tab);
    e.do_work();
    assert_eq!(e.side, Side::Rows);
    assert!(e.status.as_deref().unwrap().contains("no data"));
}

#[test]
fn picking_both_sides_keeps_each_sides_marks_and_returns_both() {
    let loader: Loader<'_> = Box::new(|side| {
        assert_eq!(side, Side::Rows);
        Ok(rows())
    });
    let mut e = StatExplorer::new(
        "input",
        Side::Columns,
        columns(),
        Some(loader),
        None,
        Purpose::PickBoth { verb: "subset" },
    );
    press(&mut e, KeyCode::Enter);
    assert!(!e.done(), "nothing marked on either side");

    // Columns sort by nnz descending: CELL5 first.
    press(&mut e, KeyCode::Char(' '));
    press(&mut e, KeyCode::Tab);
    e.do_work();
    assert_eq!(e.side, Side::Rows);
    press(&mut e, KeyCode::Char(' ')); // CT3 (entry 11)
    press(&mut e, KeyCode::Char(' ')); // CT2 (entry 10)
    assert!(screen_text(&mut e, 160, 30).contains("subset 2 rows × 1 columns"));

    press(&mut e, KeyCode::Enter);
    assert!(e.done());
    assert_eq!(
        e.picked.clone().unwrap(),
        Picked {
            rows: vec![10, 11],
            columns: vec![4],
        }
    );
}

#[test]
fn picking_both_sides_allows_marks_on_one_side_only() {
    let loader: Loader<'_> = Box::new(|_| Ok(rows()));
    let mut e = StatExplorer::new(
        "input",
        Side::Columns,
        columns(),
        Some(loader),
        None,
        Purpose::PickBoth { verb: "subset" },
    );
    press(&mut e, KeyCode::Char(' '));
    press(&mut e, KeyCode::Enter);
    let picked = e.picked.clone().unwrap();
    assert_eq!(picked.columns, [4]);
    assert!(picked.rows.is_empty(), "rows never shown, so kept whole");
}
