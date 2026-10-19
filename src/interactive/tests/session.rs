use super::ui::{modes_of, Screen};
use super::*;
use legume_tui::{Keyboard, Modes, Mouse};
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
    assert_eq!(none, Modes::new());

    let both = modes_of(&Asks {
        mouse: true,
        chords: true,
    });
    assert_eq!(both.mouse, Mouse::Buttons, "buttons and the wheel only");
    assert_eq!(
        both.keyboard_enhancement,
        Keyboard::On(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
    );
    assert!(!both.bracketed_paste && !both.focus);
}

#[test]
fn outside_a_session_the_title_bar_lists_no_keys() {
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
        "`data-beans attach` offers them to return to",
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
