//! Shared logger setup for the workspace.
//!
//! Reachable as `data_beans::logging` and, for callers that take it with the
//! other helpers, as `data_beans::aux::logging`. The progress-bar style and
//! the single [`MULTI_PROGRESS`] are defined in
//! [`legume_numeric::matrix::progress`], the lowest common dependency of the
//! workspace; this module re-exports [`new_progress_bar`], [`new_spinner`],
//! and [`MULTI_PROGRESS`] from there, so every crate that draws a bar shares
//! one set of them, and adds [`init_logger`],
//! which wraps `env_logger` in `indicatif_log_bridge` so `log` output renders
//! above the bars instead of corrupting them. With the `tui` feature the
//! logger sits behind legume-tui's log gate, so a view's
//! `legume_tui::logs::hold` keeps records back while it owns the terminal.

pub use legume_numeric::matrix::progress::{new_progress_bar, new_spinner, MULTI_PROGRESS};

/// Install `env_logger` wrapped in `indicatif_log_bridge::LogWrapper` so log
/// messages render above any active progress bar. `verbose` selects between
/// `legume_numeric::matrix::common_io::{VERBOSE,QUIET}_LOG_FILTER`; an external `RUST_LOG`
/// still overrides the default filter.
pub fn init_logger(verbose: bool) {
    init_logger_with(if verbose {
        legume_numeric::matrix::common_io::VERBOSE_LOG_FILTER
    } else {
        legume_numeric::matrix::common_io::QUIET_LOG_FILTER
    });
}

/// [`init_logger`] with `default_filter` (`error`, `info`) where no
/// `RUST_LOG` is set. For the `data-beans` binary's own default; not part
/// of the library's surface.
#[doc(hidden)]
pub fn init_logger_with(default_filter: &str) {
    let logger =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default_filter))
            .build();
    let max_level = logger.filter();
    let wrapped = indicatif_log_bridge::LogWrapper::new(MULTI_PROGRESS.clone(), logger);
    #[cfg(feature = "tui")]
    let installed = log::set_logger(legume_tui::logs::gate(wrapped));
    #[cfg(not(feature = "tui"))]
    let installed = log::set_boxed_logger(Box::new(wrapped));
    let _ = installed.map(|()| log::set_max_level(max_level));
}
