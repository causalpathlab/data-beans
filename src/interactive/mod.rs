//! Full-screen terminal views (feature `tui`), and line prompts for when
//! there is no terminal.
//!
//! - [`ui`]: the shared pieces for building a view. The palette (terminal
//!   foreground plus one accent), [`ui::panel`], [`ui::header`],
//!   [`ui::help_line`], [`ui::input_line`], the [`ui::Screen`] trait driven by
//!   [`ui::run_screen`], and histograms: [`ui::Scale`], [`ui::Binning`],
//!   [`ui::Binned`], [`ui::HistPlot`].
//! - [`cutoff_tui`]: row and column nnz cutoff picker over plain nnz vectors.
//! - [`stat_tui`]: table-and-histogram explorer over plain name and value
//!   columns, optionally marking entries and handing them back.
//!
//! Open a view only when [`tui_available`]; otherwise fall back to text.

pub mod cutoff_tui;
pub mod stat_tui;
pub mod ui;

use std::io::{self, IsTerminal, Write};

/// Whether a full-screen session can run: both stdin and stdout are a terminal.
/// Otherwise callers fall back to the line prompts below.
pub fn tui_available() -> bool {
    io::stdin().is_terminal() && io::stdout().is_terminal()
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
