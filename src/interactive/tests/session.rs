use super::ui::{modes_of, Screen};
use super::*;
use legume_tui::Mouse;
use ratatui::crossterm::event::{KeyEvent, KeyboardEnhancementFlags};
use ratatui::Frame;

/// A view that asks for the mouse and for chords as told.
struct Asks {
    mouse: bool,
    chords: bool,
}

impl Screen for Asks {
    fn render(&mut self, _: &mut Frame) {}
    fn handle_key(&mut self, _: KeyEvent) {}
    fn interrupt(&mut self) {}
    fn done(&self) -> bool {
        true
    }
    fn takes_mouse(&self) -> bool {
        self.mouse
    }
    fn reports_chords(&self) -> bool {
        self.chords
    }
}

#[test]
fn a_views_modes_are_the_sessions() {
    let none = modes_of(&Asks {
        mouse: false,
        chords: false,
    });
    assert_eq!(none, legume_tui::Modes::default());

    let both = modes_of(&Asks {
        mouse: true,
        chords: true,
    });
    assert_eq!(both.mouse, Mouse::Buttons, "buttons and the wheel only");
    assert_eq!(
        both.keyboard_enhancement,
        Some(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
    );
    assert!(!both.bracketed_paste && !both.focus);
}

#[test]
fn a_plain_session_offers_no_keys_and_never_stops() {
    let session = plain_session().unwrap();
    assert!(!session.is_hosted() && !session.offers_keys());
    assert!(session.attached() && !session.stop_requested());
    assert_eq!(session.name(), None);
    assert!(session_keys().is_empty(), "no detach key to list");
}

#[test]
fn the_session_paragraph_names_the_keys_and_the_subcommands() {
    let more = "The title bar shows the detach and switch keys.";
    let text = session_help(None, more);
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    for said in [
        "detachable session",
        "the command goes on",
        "ctrl+alt+d",
        "ctrl+alt+s",
        "`data-beans attach` with no name offers the live sessions",
        "--no-detach",
        more,
    ] {
        assert!(flat.contains(said), "{said:?} not in {text:?}");
    }
    assert!(text.lines().all(|l| l.chars().count() <= 70), "{text}");
}

#[test]
fn an_optional_views_paragraph_says_so_first() {
    let lead = "With --interactive, its view runs as a session.";
    let text = session_help(Some(lead), "More.");
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(flat.starts_with(lead), "{text}");
    assert!(flat.ends_with("More."), "{text}");
    assert!(text.lines().all(|l| l.chars().count() <= 70), "{text}");
}

#[test]
fn a_view_inside_a_session_follows_it_rather_than_opting_out() {
    // Outside a session: opted out, so it is never hosted by itself.
    let alone = plain_options(false);
    assert!(alone.no_detach && alone.target);
    // Under a session's marker the marker keeps it plain; opting out too
    // would make it plain by the flag, with no repaint after an attach.
    let nested = plain_options(true);
    assert!(!nested.no_detach && nested.target && nested.name.is_none());
}
