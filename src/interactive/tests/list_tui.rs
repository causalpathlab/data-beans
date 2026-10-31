use super::*;
use ratatui::backend::TestBackend;
use ratatui::crossterm::event::KeyModifiers;
use ratatui::Terminal;

fn press(v: &mut ListView, code: KeyCode) {
    v.handle_key(KeyEvent::new(code, KeyModifiers::NONE));
}

fn type_text(v: &mut ListView, text: &str) {
    for c in text.chars() {
        press(v, KeyCode::Char(c));
    }
}

/// `matrix/` holding `data`, `indices` and `features/` (with `name`), then
/// a top-level `shape`.
fn tree() -> ListView {
    let items = vec![
        Item::new("matrix/", 0, ""),
        Item::new("data", 1, "[100] f32"),
        Item::new("indices", 1, "[100] u64"),
        Item::new("features/", 1, ""),
        Item::new("name", 2, "[10] string"),
        Item::new("shape", 0, "[2] i64"),
    ];
    ListView::new("h5", "x.h5", items, "x.txt")
}

fn shown(v: &ListView) -> Vec<&str> {
    v.view.iter().map(|&i| v.items[i].name.as_str()).collect()
}

fn render(v: &mut ListView) -> String {
    let mut t = Terminal::new(TestBackend::new(60, 12)).unwrap();
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
fn a_tree_entry_is_written_as_its_path() {
    let v = tree();
    assert_eq!(
        v.paths,
        [
            "matrix",
            "matrix/data",
            "matrix/indices",
            "matrix/features",
            "matrix/features/name",
            "shape"
        ]
    );
    assert!(v.is_tree());
    assert!(!ListView::names("names", "x", &["a".into(), "b".into()], "n.txt").is_tree());
}

#[test]
fn a_filter_keeps_the_matches_and_their_groups() {
    let mut v = tree();
    press(&mut v, KeyCode::Char('/'));
    type_text(&mut v, "NAME");
    assert_eq!(shown(&v), ["matrix/", "features/", "name"]);
    // Esc clears it.
    press(&mut v, KeyCode::Esc);
    assert_eq!(shown(&v).len(), 6);
    // Not a regular expression: taken as text.
    press(&mut v, KeyCode::Char('/'));
    type_text(&mut v, "[1");
    assert!(shown(&v).is_empty());
}

#[test]
fn closing_a_group_hides_what_is_under_it() {
    let mut v = tree();
    press(&mut v, KeyCode::Enter);
    assert_eq!(shown(&v), ["matrix/", "shape"]);
    press(&mut v, KeyCode::Right);
    assert_eq!(shown(&v).len(), 6);
    // ← on a leaf goes to its group; again, closes it.
    press(&mut v, KeyCode::Down);
    press(&mut v, KeyCode::Left);
    assert_eq!(v.selected(), Some(0));
    press(&mut v, KeyCode::Left);
    assert_eq!(shown(&v), ["matrix/", "shape"]);
}

#[test]
fn w_writes_the_marked_else_the_shown() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.txt").to_string_lossy().into_owned();
    let names: Vec<Box<str>> = ["AAA", "ABB", "CCC"].iter().map(|&s| s.into()).collect();
    let mut v = ListView::names("names", "x", &names, &path);
    press(&mut v, KeyCode::Char('/'));
    type_text(&mut v, "^a");
    press(&mut v, KeyCode::Enter);
    press(&mut v, KeyCode::Char('w'));
    press(&mut v, KeyCode::Enter);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "AAA\nABB\n");
    // Marked ones win over the filter.
    press(&mut v, KeyCode::Char(' '));
    press(&mut v, KeyCode::Char('w'));
    press(&mut v, KeyCode::Enter);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "AAA\n");
    assert!(v.status.as_deref().unwrap().starts_with("wrote 1"));
    press(&mut v, KeyCode::Char('u'));
    assert_eq!(v.marked_count(), 0);
    press(&mut v, KeyCode::Char('a'));
    assert_eq!(v.marked_count(), 2);
}

#[test]
fn the_view_draws_details_notes_and_counts() {
    let mut v = tree().with_notes(vec!["feature types: 10 Gene Expression".into()]);
    press(&mut v, KeyCode::Char(' '));
    let screen = render(&mut v);
    assert!(screen.contains("6 entries · 1 marked"), "{screen}");
    assert!(screen.contains("▾ matrix/"), "{screen}");
    assert!(screen.contains("data  [100] f32"), "{screen}");
    assert!(
        screen.contains("feature types: 10 Gene Expression"),
        "{screen}"
    );
    press(&mut v, KeyCode::Char('q'));
    assert!(v.done());
}
