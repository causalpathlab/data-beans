//! Full-screen terminal views (feature `tui`), and line prompts for when
//! there is no terminal.
//!
//! - [`ui`]: the shared pieces for building a view. The palette (terminal
//!   foreground plus one accent), [`ui::panel`], [`ui::header`],
//!   [`ui::help_line`], [`ui::input_line`], the [`ui::Screen`] trait driven by
//!   [`ui::run_screen`], and histograms: [`ui::Scale`], [`ui::Binning`],
//!   [`ui::Binned`], [`ui::HistPlot`], and [`ui::MirrorPlot`] for two series
//!   around one zero line.
//! - [`cutoff_tui`]: row and column nnz cutoff picker over plain nnz vectors.
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

pub mod cutoff_tui;
pub mod stat_tui;
pub mod ui;

use std::io::{self, IsTerminal, Write};

/// Whether a full-screen session can run: both stdin and stdout are a terminal.
/// Otherwise callers fall back to the line prompts below.
pub fn tui_available() -> bool {
    io::stdin().is_terminal() && io::stdout().is_terminal()
}

/// The tool data-beans' sessions belong to: they are named
/// `data-beans-squeeze-1`, `data-beans-stat-2`, and `data-beans attach`
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
        .resume_hint(None)
        .more(more);
    let help = match lead {
        Some(lead) => help.lead(lead),
        None => help,
    };
    legume_tui::cli::session_help(&TOOL, help)
}

/// User action after viewing histogram or other interactive prompts
#[derive(Debug, Clone)]
pub enum UserAction {
    Proceed,
    AdjustCutoffs(usize, usize),
    Cancel,
}

/// Prompt user for action in interactive mode after showing histogram
pub fn prompt_user_action(
    current_row_cutoff: usize,
    current_col_cutoff: usize,
) -> anyhow::Result<UserAction> {
    println!("\nOptions:");
    println!("  [p] Proceed with current cutoffs");
    println!("  [a] Adjust cutoffs");
    println!("  [c] Cancel");
    print!("\nChoose an option (p/a/c): ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let choice = input.trim().to_lowercase();

    match choice.as_str() {
        "p" | "proceed" | "y" | "yes" => Ok(UserAction::Proceed),
        "c" | "cancel" | "n" | "no" => Ok(UserAction::Cancel),
        "a" | "adjust" => {
            let new_row_cutoff = prompt_cutoff_value("row", current_row_cutoff)?;
            let new_col_cutoff = prompt_cutoff_value("column", current_col_cutoff)?;
            Ok(UserAction::AdjustCutoffs(new_row_cutoff, new_col_cutoff))
        }
        _ => {
            println!("Invalid choice. Cancelling operation.");
            Ok(UserAction::Cancel)
        }
    }
}

/// Prompt user for a single cutoff value
fn prompt_cutoff_value(label: &str, current: usize) -> anyhow::Result<usize> {
    print!("\nEnter new {} nnz cutoff (current: {}): ", label, current);
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    let value = input.trim().parse::<usize>().unwrap_or(current);
    Ok(value)
}

/// Simple yes/no confirmation prompt
pub fn confirm(message: &str) -> anyhow::Result<bool> {
    print!("{} (y/n): ", message);
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let choice = input.trim().to_lowercase();

    Ok(matches!(choice.as_str(), "y" | "yes"))
}

#[cfg(test)]
#[path = "tests/session.rs"]
mod tests;
