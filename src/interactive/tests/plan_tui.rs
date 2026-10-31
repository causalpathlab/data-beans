use super::*;
use ratatui::backend::TestBackend;
use ratatui::crossterm::event::KeyModifiers;
use ratatui::Terminal;

fn press(v: &mut PlanView<'_>, code: KeyCode) {
    v.handle_key(KeyEvent::new(code, KeyModifiers::NONE));
}

/// Two files, one row only in the second: written only with common rows.
fn view() -> PlanView<'static> {
    let planner: Planner = Box::new(|s: &[Setting]| {
        let common = s[0].flag();
        let cells = s[1].number().unwrap_or(10.0);
        anyhow::ensure!(cells >= 1.0, "cells must be at least 1");
        Ok(Plan {
            columns: vec!["file".into(), "rows".into()],
            rows: vec![vec!["a".into(), "2".into()], vec!["b".into(), "3".into()]],
            notes: vec![format!("{cells} cells")],
            blocked: (!common).then(|| "rows differ: press c".into()),
        })
    });
    PlanView::new(
        "merge",
        "a, b",
        "merge",
        vec![
            Control::flag('c', "common rows", false),
            Control::number('n', "cells", Some(10.0)),
        ],
        planner,
    )
}

fn render(v: &mut PlanView<'_>) -> String {
    let mut t = Terminal::new(TestBackend::new(80, 14)).unwrap();
    t.draw(|f| v.render(f)).unwrap();
    let buf = t.backend().buffer().clone();
    (0..buf.area.height)
        .map(|y| {
            (0..buf.area.width)
                .map(|x| buf[(x, y)].symbol().to_string())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn enter_waits_for_a_plan_that_can_be_written() {
    let mut v = view();
    press(&mut v, KeyCode::Enter);
    assert!(!v.done());
    assert!(v.status.as_deref().unwrap().contains("rows differ"));
    press(&mut v, KeyCode::Char('c'));
    assert!(v.plan.blocked.is_none());
    press(&mut v, KeyCode::Enter);
    assert!(v.done());
    assert_eq!(
        v.decided(),
        Some(vec![Setting::Flag(true), Setting::Number(Some(10.0))])
    );
}

#[test]
fn a_number_is_typed_and_replans() {
    let mut v = view();
    press(&mut v, KeyCode::Char('n'));
    press(&mut v, KeyCode::Backspace);
    press(&mut v, KeyCode::Backspace);
    for c in "250".chars() {
        press(&mut v, KeyCode::Char(c));
    }
    press(&mut v, KeyCode::Enter);
    assert_eq!(v.plan.notes, ["250 cells"]);
    // An error from the planner blocks, and says so.
    press(&mut v, KeyCode::Char('n'));
    press(&mut v, KeyCode::Backspace);
    press(&mut v, KeyCode::Backspace);
    press(&mut v, KeyCode::Backspace);
    press(&mut v, KeyCode::Char('0'));
    press(&mut v, KeyCode::Enter);
    assert_eq!(v.plan.blocked.as_deref(), Some("cells must be at least 1"));
}

#[test]
fn q_cancels_and_hands_nothing_back() {
    let mut v = view();
    press(&mut v, KeyCode::Char('q'));
    assert!(v.done());
    assert_eq!(v.decided(), None);
}

#[test]
fn the_plan_is_drawn_with_its_settings() {
    let mut v = view();
    let screen = render(&mut v);
    assert!(screen.contains("file  rows"), "{screen}");
    assert!(screen.contains("✗ rows differ: press c"), "{screen}");
    assert!(screen.contains("c common rows off"), "{screen}");
    assert!(screen.contains("n cells 10"), "{screen}");
    assert!(screen.contains("cannot be written yet"), "{screen}");
}

/// A setting on a letter a view might use to move (`k`) is the setting's.
#[test]
fn a_settings_key_is_never_taken_for_moving() {
    let planner: Planner = Box::new(|s: &[Setting]| {
        Ok(Plan {
            notes: vec![format!("{:?}", s[0].number())],
            ..Plan::default()
        })
    });
    let mut v = PlanView::new(
        "x",
        "x",
        "go",
        vec![Control::number('k', "folds", None)],
        planner,
    );
    press(&mut v, KeyCode::Char('k'));
    press(&mut v, KeyCode::Char('5'));
    press(&mut v, KeyCode::Enter);
    assert!(!v.done());
    assert_eq!(v.plan.notes, ["Some(5.0)"]);
}
