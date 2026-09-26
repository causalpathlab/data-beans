use super::*;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

fn press(e: &mut StatExplorer<'_>, code: KeyCode) {
    e.handle_key(KeyEvent::new(code, KeyModifiers::NONE));
}

/// Placeholder entries: GENE1..GENE9 and CT1..CT3 with distinct stats.
fn explorer() -> StatExplorer<'static> {
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
    StatExplorer::new(
        "input",
        Side::Rows,
        Dataset {
            names,
            values: [nnz, sum, mean, sd],
        },
        None,
    )
}

fn shown_names<'e>(e: &'e StatExplorer<'_>) -> Vec<&'e str> {
    e.view.iter().map(|&i| &*e.names[i]).collect()
}

#[test]
fn starts_sorted_by_nnz_descending() {
    let e = explorer();
    assert_eq!(shown_names(&e)[..2], ["CT3", "CT2"]);
    assert_eq!(e.hist.all.iter().sum::<usize>(), 12);
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
    for c in "^ct".chars() {
        press(&mut e, KeyCode::Char(c));
    }
    assert_eq!(e.view.len(), 3);
    let shown: usize = e.hist.shown.as_ref().unwrap().iter().sum();
    assert_eq!(shown, 3, "the subset is binned over the same bins");

    // An invalid regex matches literally rather than hiding everything.
    for c in "(".chars() {
        press(&mut e, KeyCode::Char(c));
    }
    assert!(e.view.is_empty());
    press(&mut e, KeyCode::Esc);
    assert_eq!(e.view.len(), 12);
    assert!(e.hist.shown.is_none());
}

#[test]
fn selection_follows_its_entry_across_resorts() {
    let mut e = explorer();
    press(&mut e, KeyCode::Down);
    let before = e.selected();
    press(&mut e, KeyCode::Char('4'));
    assert_eq!(e.selected(), before);
}

#[test]
fn scales_rebin_every_statistic() {
    let mut e = explorer();
    for s in 0..4 {
        e.choose_stat(s);
        for _ in 0..3 {
            press(&mut e, KeyCode::Char('x'));
            assert_eq!(
                e.hist.all.iter().sum::<usize>(),
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
        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        term.draw(|f| e.render(f)).unwrap();
        if w >= 80 {
            let text: String = term
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|c| c.symbol())
                .collect();
            assert!(text.contains("nnz ▼"));
            assert!(text.contains("GENE1"), "last row scrolled into view");
        }
    }
}

#[test]
fn starts_on_the_top_entry() {
    let e = explorer();
    assert_eq!(e.selected(), Some(e.view[0]));
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

#[test]
fn tab_computes_the_other_side_once_and_toggles() {
    let calls = std::cell::Cell::new(0);
    let loader: Loader<'_> = Box::new(|side| {
        assert_eq!(side, Side::Columns);
        calls.set(calls.get() + 1);
        Ok(columns())
    });
    let rows = explorer();
    let mut e = StatExplorer::new(
        "input",
        Side::Rows,
        Dataset {
            names: rows.names.clone(),
            values: rows.values.clone(),
        },
        Some(loader),
    );
    press(&mut e, KeyCode::Char('/'));
    press(&mut e, KeyCode::Char('1'));
    press(&mut e, KeyCode::Enter);

    press(&mut e, KeyCode::Tab);
    assert!(e.loading, "the first Tab asks for the columns");
    e.finish_loading();
    assert_eq!(e.side, Side::Columns);
    assert_eq!(calls.get(), 1);
    // The filter carries over: "1" matches CELL1 only.
    assert_eq!(shown_names(&e), ["CELL1"]);
    assert_eq!(e.selected(), Some(e.view[0]));

    press(&mut e, KeyCode::Tab);
    assert_eq!(e.side, Side::Rows);
    press(&mut e, KeyCode::Tab);
    assert_eq!(e.side, Side::Columns);
    assert!(!e.loading);
    assert_eq!(calls.get(), 1, "computed once");
}

#[test]
fn a_failed_load_stays_put_and_says_why() {
    let loader: Loader<'_> = Box::new(|_| Err(anyhow::anyhow!("no data")));
    let rows = explorer();
    let mut e = StatExplorer::new(
        "input",
        Side::Rows,
        Dataset {
            names: rows.names.clone(),
            values: rows.values.clone(),
        },
        Some(loader),
    );
    press(&mut e, KeyCode::Tab);
    e.finish_loading();
    assert_eq!(e.side, Side::Rows);
    assert!(e.status.as_deref().unwrap().contains("no data"));
}
