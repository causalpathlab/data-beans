use super::*;
use crate::interactive::ui::Scale;
use ratatui::backend::TestBackend;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Terminal;

/// Bimodal nnz: a low ambient mode and a high cell mode.
fn bimodal() -> Vec<f32> {
    let mut v: Vec<f32> = (0..300).map(|i| (1 + i % 20) as f32).collect();
    v.extend((0..200).map(|i| (800 + 7 * i) as f32));
    v
}

fn press(p: &mut CutoffPicker, code: KeyCode) {
    p.handle_key(KeyEvent::new(code, KeyModifiers::NONE));
}

fn picker(in_place: Option<&str>) -> CutoffPicker {
    let nnz = bimodal();
    CutoffPicker::new(
        "input",
        AxisView::new("Rows", &nnz, 0, None),
        AxisView::new("Columns", &nnz, 100, Some(200)),
        in_place,
    )
}

#[test]
fn removed_matches_a_linear_count() {
    let nnz = bimodal();
    let a = AxisView::new("x", &nnz, 0, None);
    for c in [0, 1, 5, 21, 500, 800, 2193, 5000] {
        let want = nnz.iter().filter(|&&x| (x as usize) < c).count();
        assert_eq!(a.removed_at(c), want, "cutoff {c}");
    }
}

#[test]
fn bin_steps_move_one_bar_and_round_trip() {
    let mut a = AxisView::new("x", &bimodal(), 0, None);
    let mut seen = vec![0];
    for _ in 0..40 {
        let before = a.cutoff;
        a.step_bin(1);
        assert!(a.cutoff > before || a.cutoff == a.max_value() + 1);
        seen.push(a.cutoff);
    }
    assert_eq!(
        a.cutoff,
        a.max_value() + 1,
        "stepping right stops past the max"
    );
    for &want in seen.iter().rev().skip(1) {
        a.step_bin(-1);
        if a.cutoff != want {
            // Only the capped end may be off the bin grid.
            assert!(want > a.max_value() || a.cutoff <= want);
        }
    }
    for _ in 0..60 {
        a.step_bin(-1);
    }
    assert_eq!(a.cutoff, 0);
}

#[test]
fn keys_edit_the_focused_axis_and_proceed() {
    let mut p = picker(None);
    press(&mut p, KeyCode::Right);
    assert!(p.axes[0].cutoff > 0);
    assert_eq!(p.axes[1].cutoff, 100);

    press(&mut p, KeyCode::Tab);
    press(&mut p, KeyCode::Char('s'));
    assert_eq!(p.axes[1].cutoff, 200);
    press(&mut p, KeyCode::Char('+'));
    assert_eq!(p.axes[1].cutoff, 201);
    press(&mut p, KeyCode::Char('r'));
    assert_eq!(p.axes[1].cutoff, 100);

    for c in ['4', '2'] {
        press(&mut p, KeyCode::Char(c));
    }
    press(&mut p, KeyCode::Enter);
    assert_eq!(p.axes[1].cutoff, 42);
    assert!(
        p.decision.as_ref().is_none(),
        "Enter in edit mode only sets the value"
    );

    press(&mut p, KeyCode::Enter);
    let row = p.axes[0].cutoff;
    assert_eq!(
        p.decision.as_ref(),
        Some(&Decision::Proceed { row, column: 42 })
    );
}

#[test]
fn in_place_asks_first_and_n_goes_back() {
    let mut p = picker(Some("target"));
    press(&mut p, KeyCode::Enter);
    assert!(p.decision.as_ref().is_none());
    press(&mut p, KeyCode::Char('n'));
    assert!(p.decision.as_ref().is_none());
    press(&mut p, KeyCode::Enter);
    press(&mut p, KeyCode::Char('y'));
    assert_eq!(
        p.decision.as_ref(),
        Some(&Decision::Proceed {
            row: 0,
            column: 100
        })
    );
}

#[test]
fn q_cancels() {
    let mut p = picker(None);
    press(&mut p, KeyCode::Char('q'));
    assert_eq!(p.decision.as_ref(), Some(&Decision::Cancel));
}

#[test]
fn renders_both_axes_with_stats_and_markers() {
    let p = picker(Some("target"));
    let mut term = Terminal::new(TestBackend::new(100, 30)).unwrap();
    term.draw(|f| p.render(f)).unwrap();
    let text: String = term
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol())
        .collect();
    assert!(text.contains("Rows nnz"));
    assert!(text.contains("Columns nnz"));
    assert!(
        text.contains("drops 300 / 500"),
        "column cutoff 100 drops the low mode"
    );
    assert!(text.contains("suggested 200"));
    assert!(text.contains('▲') && text.contains('◆'));
}

#[test]
fn renders_on_a_tiny_terminal_without_panicking() {
    let mut p = picker(Some("a-long-target-name"));
    press(&mut p, KeyCode::Enter);
    let mut term = Terminal::new(TestBackend::new(20, 6)).unwrap();
    term.draw(|f| p.render(f)).unwrap();
}

#[test]
fn first_bin_step_drops_the_first_bar() {
    // Placeholder counts that start well above zero.
    let nnz: Vec<f32> = (0..100).map(|i| (30 + 3 * i) as f32).collect();
    let mut a = AxisView::new("x", &nnz, 0, None);
    a.step_bin(1);
    assert!(a.removed() > 0, "cutoff {} drops nothing", a.cutoff);
    assert_eq!(a.removed(), a.counts[0]);
    a.step_bin(-1);
    assert_eq!(a.cutoff, 0);
}

#[test]
fn every_scale_bins_all_entries_and_steps_by_bar() {
    let nnz = bimodal();
    let mut a = AxisView::new("x", &nnz, 0, None);
    for scale in [Scale::Log, Scale::Sqrt, Scale::Linear] {
        a.set_scale(scale);
        assert_eq!(a.counts.iter().sum::<usize>(), nnz.len(), "{scale:?}");
        assert!(a.stops.windows(2).all(|w| w[0] < w[1]), "{scale:?}");
        // Each stop drops exactly the bars left of the bin it opens.
        for &stop in &a.stops[1..a.stops.len() - 1] {
            let k = a.bins.key(stop as f64);
            let left: usize = a.counts[..(k - a.kmin) as usize].iter().sum();
            assert_eq!(a.removed_at(stop), left, "{scale:?} stop {stop}");
        }
    }
}

#[test]
fn x_and_y_cycle_scales_for_both_axes() {
    let mut p = picker(None);
    press(&mut p, KeyCode::Char('x'));
    assert_eq!(p.x_scale, Scale::Sqrt);
    assert!(p.axes.iter().all(|a| a.bins.scale == Scale::Sqrt));
    press(&mut p, KeyCode::Char('y'));
    press(&mut p, KeyCode::Char('y'));
    assert_eq!(p.y_scale, Scale::Linear);
    let mut term = Terminal::new(TestBackend::new(100, 30)).unwrap();
    term.draw(|f| p.render(f)).unwrap();
    let text: String = term
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol())
        .collect();
    assert!(text.contains("x sqrt · y linear"));
}
