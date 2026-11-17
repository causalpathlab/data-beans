//! Figures drawn as images inside the full-screen views, in terminals that
//! speak an image protocol (kitty, iTerm2, WezTerm, Ghostty, sixel):
//! legume-tui's (`legume_tui::graphics::figure`), asked for as
//! `DATA_BEANS_TUI_IMAGES` says. Elsewhere the views keep their text plots.

pub use legume_tui::graphics::figure::PlotImage;
use ratatui_image::picker::Picker;

/// The image support of the terminal `session` is attached to now. `None`
/// without an image protocol, in a terminal not known to have one
/// (`DATA_BEANS_TUI_IMAGES=1` asks anyway), or with
/// `DATA_BEANS_TUI_IMAGES=0`. Call it with the screen up, and again after
/// every attach.
pub fn picker(session: &legume_tui::Session) -> Option<Picker> {
    legume_tui::graphics::figure::picker(session, "DATA_BEANS_TUI_IMAGES")
}
