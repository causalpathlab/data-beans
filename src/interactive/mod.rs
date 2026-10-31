//! Full-screen terminal views (feature `tui`), and line prompts for when
//! there is no terminal.
//!
//! - [`ui`]: the shared pieces for building a view. The palette (terminal
//!   foreground plus one accent), [`ui::panel`], [`ui::header`],
//!   [`ui::help_line`], [`ui::input_line`], the [`ui::Screen`] trait driven by
//!   [`ui::run_screen`], and histograms: [`ui::Scale`], [`ui::Binning`],
//!   [`ui::Binned`], [`ui::HistPlot`], and [`ui::MirrorPlot`] for two series
//!   around one zero line.
//! - [`qc_tui`]: `qc`'s view: cell QC rules moved on their histograms.
//! - [`list_tui`]: a list to browse, flat (names) or a tree (a file's
//!   hierarchy): filter, mark, write, copy.
//! - [`plan_tui`]: what a command is about to write, with its settings on
//!   keys: Enter writes, `q` does not.
//! - [`stat_tui`]: table-and-histogram explorer over plain name and value
//!   columns, optionally marking entries and handing them back.
//!
//! Open a view only when [`tui_available`]; otherwise fall back to text.
//!
//! The views run in a legume-tui session ([`TOOL`]): a command opens it
//! first thing and hands it to [`ui::Tui`] or [`ui::run_screen_in`]; where
//! the session is hosted, the command keeps running when the connection
//! drops. [`ui::run_screen`] runs a view in a plain session of its own
//! (`legume_tui::Session::plain`).

pub mod list_tui;
pub mod plan_tui;
pub mod qc_tui;
pub mod stat_tui;
pub mod ui;

use std::io::{self, IsTerminal};

/// Set, to anything but empty or `0`, to run every command straight through
/// as `--batch-process` does: for a script or pipeline started from a
/// terminal, which its commands would otherwise take for the user's.
pub const BATCH_ENV: &str = "DATA_BEANS_BATCH";

/// Whether a full-screen view can run: stdin and stdout are a terminal, and
/// one that can draw it (`TERM` is not `dumb`).
pub fn tui_available() -> bool {
    can_draw(
        io::stdin().is_terminal() && io::stdout().is_terminal(),
        std::env::var("TERM").ok().as_deref(),
    )
}

fn can_draw(tty: bool, term: Option<&str>) -> bool {
    tty && term != Some("dumb")
}

/// Why a command runs without its view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NoView {
    /// `--batch-process`, or [`BATCH_ENV`].
    Batch,
    /// No terminal to draw it on.
    NoTerminal,
}

/// The one rule every command's view follows: batch when asked (`flag`, or
/// `env`, the value of [`BATCH_ENV`]), else a view where there is a
/// terminal (`terminal`).
fn decide(flag: bool, env: Option<&str>, terminal: bool) -> Result<(), NoView> {
    let asked = flag || env.is_some_and(|v| !v.is_empty() && v != "0");
    match (asked, terminal) {
        (true, _) => Err(NoView::Batch),
        (false, false) => Err(NoView::NoTerminal),
        (false, true) => Ok(()),
    }
}

fn decide_here(flag: bool) -> Result<(), NoView> {
    decide(
        flag,
        std::env::var(BATCH_ENV).ok().as_deref(),
        tui_available(),
    )
}

/// The line said when `command`'s view is skipped for want of a terminal.
fn no_terminal_line(command: &str) -> String {
    format!(
        "{command}: no terminal to draw a view on (stdin or stdout is not one, or \
         TERM=dumb): running straight through; --batch-process or {BATCH_ENV}=1 \
         does this without this line"
    )
}

/// Whether `command` opens its view, for a command that runs without one
/// too (`qc`, `stat`): not when batch is asked (`batch`, its
/// `--batch-process`, or [`BATCH_ENV`]), and only on a terminal. A view
/// skipped for want of a terminal (output redirected, say) is said in one
/// line on stderr, so the switch is never silent; asked for, it is quiet.
pub fn opens_view(command: &str, batch: bool) -> bool {
    match decide_here(batch) {
        Ok(()) => true,
        Err(NoView::Batch) => false,
        Err(NoView::NoTerminal) => {
            eprintln!("{}", no_terminal_line(command));
            false
        }
    }
}

/// Whether `command` picks its rows or columns in the explorer, for a
/// command that takes them on the command line too (`columns`, `subset-rows`,
/// ...): only when the command line `selected` none. Then batch asked
/// (`batch`, or [`BATCH_ENV`]) or no terminal is an error naming
/// `options`, what selects them instead.
pub fn picks(command: &str, batch: bool, selected: bool, options: &str) -> anyhow::Result<bool> {
    if selected {
        return Ok(false);
    }
    match decide_here(batch) {
        Ok(()) => Ok(true),
        Err(why) => anyhow::bail!("{}", pick_refused(command, options, why)),
    }
}

fn pick_refused(command: &str, options: &str, why: NoView) -> String {
    match why {
        NoView::Batch => {
            format!("{command}: give {options}; run in batch, it has no view to pick them in")
        }
        NoView::NoTerminal => {
            format!("{command}: give {options}, or run it on a terminal to pick them in a view")
        }
    }
}

/// The tool data-beans' sessions belong to: they are named
/// `data-beans-qc-1`, `data-beans-stat-2`, and `data-beans attach`
/// finds them.
pub const TOOL: legume_tui::Tool = legume_tui::Tool::new("data-beans");

/// The detach and switch keys for a title bar ([`ui::title_bar`]), each
/// with what it does in a word (`detach`, `switch`); none when the views
/// run plain.
pub fn session_keys() -> Vec<(String, String)> {
    legume_tui::help_keys()
        .into_iter()
        .map(|k| (k.key.to_lowercase(), k.short))
        .collect()
}

/// The paragraph a command's long help ends with about its session:
/// `lead` first, for a command whose view is optional (when it runs as a
/// session), then how it runs, the detach and switch keys, how to return,
/// and `more` at the end. Its views are always opened on something
/// named, so a new start never offers the live sessions; `data-beans
/// attach` does.
pub fn session_help(lead: Option<&str>, more: &str) -> String {
    let help = legume_tui::cli::SessionHelp::new()
        .what_goes_on("the command goes on")
        .resume_hint(Some(
            "Starting it again starts a new one (naming the detached ones); \
             `data-beans attach` with no name",
        ))
        .more(more);
    let help = match lead {
        Some(lead) => help.lead(lead),
        None => help,
    };
    legume_tui::cli::session_help(&TOOL, help)
}

#[cfg(test)]
#[path = "tests/session.rs"]
mod tests;
