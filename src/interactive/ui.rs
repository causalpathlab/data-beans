//! Shared pieces of the full-screen views, legume-tui's: the screen loop
//! and the chrome (`legume_tui::view`), histogram scales, binning and plots
//! (`legume_tui::hist`), all drawn in data-beans' [`LOOK`].

use ratatui::crossterm::event::KeyEvent;
use ratatui::style::{Color, Modifier, Style};

pub use legume_tui::hist::{
    compact, median, BarValue, Binned, Binning, HistPlot, MirrorPlot, MirrorSide, Scale,
};
pub use legume_tui::view::{
    header, help_line, input_line, modes_of, notify, panel, run_screen_in, run_screen_in_with,
    title_bar, Screen, Tui, TICK,
};
pub use legume_tui::Look;

/// The one accent, for what needs the eye: what a cutoff drops, the
/// selection, and key hints.
const ACCENT: Color = Color::Rgb(217, 119, 87);

/// The views' styles: the terminal's own foreground (so light and dark
/// backgrounds both work) for nearly everything, dim for what is
/// secondary, and [`ACCENT`].
pub const LOOK: Look = Look::new(
    Style::new(),
    Style::new().add_modifier(Modifier::BOLD),
    Style::new().add_modifier(Modifier::DIM),
    Style::new().add_modifier(Modifier::REVERSED),
    Style::new().add_modifier(Modifier::DIM),
    Style::new().fg(ACCENT),
);

/// Ask for `text` on the clipboard (`pbcopy`, else the terminal's OSC 52).
/// Whether a terminal took it cannot be known.
pub fn copy(text: &str) {
    // Nothing to do about a clipboard that is not there.
    let _ = legume_tui::keys::copy(text, true);
}

/// Run `screen` full screen until it is done, in a plain session of
/// data-beans' own. The terminal is restored on return and on panic.
pub fn run_screen<S: Screen>(screen: &mut S) -> anyhow::Result<()> {
    Ok(legume_tui::view::run_screen(&super::TOOL, screen)?)
}

/// [`run_screen`], with each key (but Ctrl-C) handed to `on_key` rather
/// than [`Screen::handle_key`].
pub fn run_screen_with<S: Screen>(
    screen: &mut S,
    on_key: impl FnMut(&mut S, KeyEvent),
) -> anyhow::Result<()> {
    Ok(legume_tui::view::run_screen_with(
        &super::TOOL,
        screen,
        on_key,
    )?)
}
