//! Shared logger setup for the workspace.
//!
//! The progress-bar style and the single [`MULTI_PROGRESS`] now live in
//! [`legume_numeric::matrix::progress`] — the lowest common dependency, so `data-beans`
//! and friends can share them without a dependency cycle (this crate depends
//! *on* `data-beans`, so the primitive can't live here). This module
//! re-exports [`new_progress_bar`], [`new_spinner`], and [`MULTI_PROGRESS`] so
//! every existing caller (`data_beans::aux::logging::new_progress_bar`, senna,
//! graph-embedding-util) keeps working unchanged, and adds [`init_logger`],
//! which wraps `env_logger` in `indicatif_log_bridge` so `log` output renders
//! above the bars instead of corrupting them, and [`hold_logs`], which keeps
//! records back while a full-screen view owns the terminal.

use std::sync::Mutex;

pub use legume_numeric::matrix::progress::{new_progress_bar, new_spinner, MULTI_PROGRESS};

/// Install `env_logger` wrapped in `indicatif_log_bridge::LogWrapper` so log
/// messages render above any active progress bar. `verbose` selects between
/// `legume_numeric::matrix::common_io::{VERBOSE,QUIET}_LOG_FILTER`; an external `RUST_LOG`
/// still overrides the default filter.
pub fn init_logger(verbose: bool) {
    let default_filter = if verbose {
        legume_numeric::matrix::common_io::VERBOSE_LOG_FILTER
    } else {
        legume_numeric::matrix::common_io::QUIET_LOG_FILTER
    };
    let logger =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default_filter))
            .build();
    let max_level = logger.filter();
    let wrapped = indicatif_log_bridge::LogWrapper::new(MULTI_PROGRESS.clone(), logger);
    let _ =
        log::set_boxed_logger(Box::new(Holding(wrapped))).map(|()| log::set_max_level(max_level));
}

/// Records held back while a full-screen view owns the terminal: level,
/// target and message. `None` when not holding.
static HELD: Mutex<Option<Vec<(log::Level, String, String)>>> = Mutex::new(None);

/// The logger [`init_logger`] installs: forwards records, or keeps them in
/// [`HELD`] while [`hold_logs`] is on.
struct Holding<L>(L);

impl<L: log::Log> log::Log for Holding<L> {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        self.0.enabled(metadata)
    }

    fn log(&self, record: &log::Record) {
        if let Some(held) = HELD.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            if self.0.enabled(record.metadata()) {
                let entry = (
                    record.level(),
                    record.target().into(),
                    record.args().to_string(),
                );
                held.push(entry);
            }
            return;
        }
        self.0.log(record);
    }

    fn flush(&self) {
        self.0.flush();
    }
}

/// Hold log records back (`true`) while a full-screen view owns the
/// terminal, where a printed line would scroll it and tear the picture; on
/// `false`, write what was held, in order. Only affects the logger that
/// [`init_logger`] installed.
pub fn hold_logs(on: bool) {
    let held = {
        let mut slot = HELD.lock().unwrap_or_else(|e| e.into_inner());
        if on {
            slot.get_or_insert_with(Vec::new);
            return;
        }
        slot.take()
    };
    for (level, target, message) in held.into_iter().flatten() {
        log::logger().log(
            &log::Record::builder()
                .level(level)
                .target(&target)
                .args(format_args!("{message}"))
                .build(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use log::Log;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Counting(AtomicUsize);

    impl Log for Counting {
        fn enabled(&self, _: &log::Metadata) -> bool {
            true
        }
        fn log(&self, _: &log::Record) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
        fn flush(&self) {}
    }

    #[test]
    fn held_records_wait_for_release() {
        let logger = Holding(Counting(AtomicUsize::new(0)));
        let record = |msg| {
            logger.log(
                &log::Record::builder()
                    .level(log::Level::Warn)
                    .args(format_args!("{msg}"))
                    .build(),
            )
        };
        hold_logs(true);
        record("held");
        assert_eq!(logger.0 .0.load(Ordering::Relaxed), 0, "nothing written");
        let held = HELD.lock().unwrap().as_ref().map(|h| h.len());
        assert_eq!(held, Some(1));
        hold_logs(false);
        assert!(HELD.lock().unwrap().is_none(), "released");
        record("through");
        assert_eq!(logger.0 .0.load(Ordering::Relaxed), 1);
    }
}
