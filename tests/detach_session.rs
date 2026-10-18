//! The interactive views as detachable sessions, driven through
//! pseudo-terminals standing in for the user's: `stat --interactive`
//! starts hosted, a detach leaves it running and asks for a name, an
//! attach from another terminal repaints it, a lost client leaves it
//! running, and `data-beans kill` stops it through its own quit path,
//! leaving no process or socket behind. A kill quits `subset` as `q` does
//! (nothing written), lets a write under way finish (also one before the
//! first view), and stops `squeeze` before the files after it.
//! `--no-detach`, and a view started inside a session, run plain.

#![cfg(all(unix, feature = "tui"))]

use legume_tui::testing::{alive, Private, Term, PROMPT};
use legume_tui::Size;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

const EXE: &str = env!("CARGO_BIN_EXE_data-beans");

/// Wide enough for every footer key, the session keys at the end.
const SIZE: Size = Size {
    rows: 40,
    cols: 260,
    width: 2600,
    height: 800,
};

/// A private directory for data-beans' sessions; a session a failed test
/// left running is killed with it.
fn private() -> Private {
    Private::new(EXE).patience(Duration::from_secs(30))
}

fn term(dir: &Private, args: &[&str], env: &[(&str, &str)]) -> Term {
    dir.term_sized(args, env, SIZE)
}

/// A 6 x 8 backend named `name` in `dir` (a `.zarr` directory), built with
/// the tool itself from a small MatrixMarket file.
fn fixture(dir: &Path, name: &str) -> String {
    let mtx = dir.join(format!("{name}.mtx"));
    let rows = dir.join(format!("{name}.rows.tsv"));
    let cols = dir.join(format!("{name}.cols.tsv"));
    let entries = [
        (1, 1, 3),
        (2, 1, 1),
        (1, 2, 2),
        (3, 3, 5),
        (4, 4, 1),
        (5, 5, 2),
        (6, 6, 7),
        (1, 7, 1),
        (2, 7, 4),
        (3, 8, 2),
        (4, 8, 1),
        (6, 8, 3),
    ];
    let mut text = format!(
        "%%MatrixMarket matrix coordinate integer general\n6 8 {}\n",
        entries.len()
    );
    for (i, j, x) in entries {
        text += &format!("{i} {j} {x}\n");
    }
    std::fs::write(&mtx, text).unwrap();
    let names = |prefix: &str, n: usize| {
        (1..=n)
            .map(|i| format!("{prefix}{i}\n"))
            .collect::<String>()
    };
    std::fs::write(&rows, names("R", 6)).unwrap();
    std::fs::write(&cols, names("C", 8)).unwrap();
    let out = dir.join(name);
    let built = Command::new(EXE)
        .arg("from-mtx")
        .arg(&mtx)
        .arg("-r")
        .arg(&rows)
        .arg("-c")
        .arg(&cols)
        .arg("-o")
        .arg(&out)
        .arg("--no-zip")
        .output()
        .unwrap();
    assert!(built.status.success(), "{built:?}");
    format!("{}.zarr", out.display())
}

/// What `data-beans info` says of `file`.
fn info(file: &str) -> String {
    let out = Command::new(EXE).args(["info", file]).output().unwrap();
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn stat_detaches_attaches_and_is_killed_through_its_quit_path() {
    let dir = private();
    let work = tempfile::tempdir().unwrap();
    let data = fixture(work.path(), "data");

    // Started in a terminal, it is hosted: this terminal is its first
    // client, and the footer lists the session keys.
    let mut first = term(&dir, &["stat", &data, "-i"], &[]);
    first.wait_for("ctrl+alt+d");
    first.wait_for("detach");
    first.wait_for("ctrl+alt+s");
    assert!(!first.cooked(), "the client's terminal is raw");
    let listed = dir.wait_listed("exploring columns of");
    assert!(listed.contains("data-beans-stat-1\tattached"), "{listed:?}");
    let pid = dir.program_pid();
    assert_ne!(
        pid,
        first.child.id(),
        "the program runs apart from its client"
    );

    // Detached (ctrl+alt+d): the client ends, its terminal restored, the
    // program runs on; Enter leaves it unnamed.
    first.type_(b"\x1b\x04");
    first.answer(1, b"\r");
    assert_eq!(first.exit_code(), 0);
    assert!(first.cooked(), "terminal restored after detach");
    first.wait_for("Detached data-beans-stat-1.");
    dir.wait_listed("data-beans-stat-1\tdetached");
    assert!(alive(pid));

    // Started again on the same file: a new session, the detached one
    // named; dropped at once.
    let mut again = term(&dir, &["stat", &data, "-i"], &[]);
    again.wait_for("[detached: data-beans-stat-1");
    again.wait_for("ctrl+alt+d");
    let said = dir.run(&["kill", "stat-2"]);
    assert!(said.contains("data-beans-stat-2 stopped"), "{said:?}");
    assert_eq!(again.exit_code(), 0);

    // Attached from another terminal: back in the alternate screen and
    // painted whole there, keys reaching the program.
    let mut second = term(&dir, &["attach", "stat-1"], &[]);
    second.wait_for("\x1b[?1049h");
    let reset = second.text().rfind("\x1b[?1049h").unwrap();
    second.wait_for_after(reset, "rows (computed on first use)");
    second.wait_for_after(reset, "ctrl+alt+d");
    let before = second.seen_len();
    second.type_(b"/");
    second.wait_for_after(before, "filter /");
    second.type_(b"\x1b");
    // Apart from the next key, or the two read as Alt+Tab.
    std::thread::sleep(Duration::from_millis(300));
    // Work a key asks for runs on the normal screen in the session, and
    // the view comes back after it.
    let before = second.seen_len();
    second.type_(b"\t");
    second.wait_for_after(before, "computing rows statistics");
    second.wait_for_after(before, "6 rows");
    second.wait_for_after(before, "ctrl+alt+d");

    // The client lost without a word (a dropped connection): it runs on.
    second.child.kill().unwrap();
    second.child.wait().unwrap();
    drop(second);
    dir.wait_listed("data-beans-stat-1\tdetached");
    assert!(alive(pid));

    // Killed while detached: it quits as ctrl+c does, leaving nothing.
    let said = dir.run(&["kill", "stat-1"]);
    assert!(said.contains("data-beans-stat-1 stopped"), "{said:?}");
    dir.gone(pid);
    assert!(dir.sockets().is_empty());
    assert!(dir
        .run(&["sessions"])
        .contains("no live data-beans sessions"));
}

#[test]
fn a_kill_quits_subset_without_writing() {
    let dir = private();
    let work = tempfile::tempdir().unwrap();
    let data = fixture(work.path(), "data");
    let out = work.path().join("picked");
    let out = out.to_string_lossy().into_owned();

    let mut t = term(&dir, &["subset", &data, "-o", &out], &[]);
    t.wait_for("ctrl+alt+d");
    dir.wait_listed("data-beans-subset-1\tattached");
    let pid = dir.program_pid();
    // Marked, not written: the kill discards it as q would.
    t.type_(b" ");
    t.wait_for("1 marked");
    let said = dir.run(&["kill"]);
    assert!(said.contains("data-beans-subset-1 stopped"), "{said:?}");
    assert_eq!(t.exit_code(), 0);
    assert!(t.cooked(), "terminal restored after the program ended");
    dir.gone(pid);
    assert!(dir.sockets().is_empty());
    let left: Vec<_> = std::fs::read_dir(work.path())
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("picked"))
        .collect();
    assert!(left.is_empty(), "written: {left:?}");
}

#[test]
fn a_kill_stops_squeeze_before_the_next_file() {
    let dir = private();
    let work = tempfile::tempdir().unwrap();
    let one = fixture(work.path(), "one");
    let two = fixture(work.path(), "two");
    let (one_before, two_before) = (info(&one), info(&two));

    // In place, two files, one session; logged so the stop is seen.
    let files = format!("{one},{two}");
    let mut t = term(&dir, &["-v", "squeeze", &files, "-i", "-r", "2"], &[]);
    t.wait_for("rows/cols");
    dir.wait_listed("picking nnz cutoffs for");
    let pid = dir.program_pid();
    let said = dir.run(&["kill"]);
    assert!(said.contains("data-beans-squeeze-1 stopped"), "{said:?}");
    assert_eq!(t.exit_code(), 0);
    // Held while the screen was up and written once the normal screen is
    // back, where it stays, rather than lost with the alternate screen.
    let text = t.text();
    let back = text.rfind("\x1b[?1049l").expect("the screen was up");
    assert!(
        text[back..].contains("Stopped before"),
        "{:?}",
        &text[back..]
    );
    dir.gone(pid);
    assert!(dir.sockets().is_empty());
    assert_eq!(info(&one), one_before, "the cancelled file is untouched");
    assert_eq!(info(&two), two_before, "the next file is not reached");
}

#[test]
fn a_kill_lets_a_write_under_way_finish() {
    let dir = private();
    let work = tempfile::tempdir().unwrap();
    let data = fixture(work.path(), "data");
    // A pipe the test reads only once the kill is in: the write opening
    // it waits for the reader, so the kill lands mid-write.
    let out = work.path().join("taken.tsv");
    let made = Command::new("mkfifo").arg(&out).status().unwrap();
    assert!(made.success());
    let out_arg = out.to_string_lossy().into_owned();

    let mut t = term(&dir, &["-v", "columns", &data, "-I", "-o", &out_arg], &[]);
    t.wait_for("ctrl+alt+d");
    dir.wait_listed("data-beans-columns-1\tattached");
    let pid = dir.program_pid();
    t.type_(b" ");
    t.wait_for("1 marked");
    t.type_(b"\r");
    // Logged as the view ends; written as the write starts, off the screen.
    t.wait_for("picked 1 entries");

    let killer = std::thread::scope(|s| {
        let killer = s.spawn(|| dir.run(&["kill"]));
        // The kill in before the write can go on.
        std::thread::sleep(Duration::from_secs(1));
        assert!(alive(pid), "waiting on the write");
        let written = std::fs::read_to_string(&out).unwrap();
        let lines: Vec<&str> = written.lines().collect();
        assert_eq!(lines.len(), 7, "a header and every row: {written:?}");
        assert!(lines[0].starts_with("row_name\t"), "{written:?}");
        killer.join().unwrap()
    });
    assert!(
        killer.contains("data-beans-columns-1 stopped"),
        "{killer:?}"
    );
    assert_eq!(t.exit_code(), 0);
    dir.gone(pid);
    assert!(dir.sockets().is_empty());
}

#[test]
fn a_kill_before_the_explorer_lets_its_output_file_finish() {
    let dir = private();
    let work = tempfile::tempdir().unwrap();
    let data = fixture(work.path(), "data");
    // The statistics file is written before the explorer comes up; a
    // pipe holds that write open until the kill is in.
    let out = work.path().join("stats.parquet");
    let made = Command::new("mkfifo").arg(&out).status().unwrap();
    assert!(made.success());
    let out_arg = out.to_string_lossy().into_owned();

    let mut t = term(&dir, &["stat", &data, "-i", "-o", &out_arg], &[]);
    dir.wait_listed("data-beans-stat-1\tattached");
    let pid = dir.program_pid();

    let killer = std::thread::scope(|s| {
        let killer = s.spawn(|| dir.run(&["kill"]));
        std::thread::sleep(Duration::from_secs(1));
        assert!(alive(pid), "waiting on the write, not ended by the kill");
        let written = std::fs::read(&out).unwrap();
        assert!(
            written.starts_with(b"PAR1") && written.ends_with(b"PAR1"),
            "a whole parquet file: {} bytes",
            written.len()
        );
        killer.join().unwrap()
    });
    assert!(killer.contains("data-beans-stat-1 stopped"), "{killer:?}");
    assert_eq!(t.exit_code(), 0);
    dir.gone(pid);
    assert!(dir.sockets().is_empty());
}

#[test]
fn no_detach_and_a_view_inside_a_session_run_plain() {
    let dir = private();
    let work = tempfile::tempdir().unwrap();
    let data = fixture(work.path(), "data");

    // Opted out by the flag, or started under a session's marker (as a
    // hosted program starting another view): in this process, no client
    // between, no socket, no session keys; q ends it, terminal restored.
    for (args, env) in [
        (vec!["stat", &data, "-i", "--no-detach"], vec![]),
        (
            vec!["stat", &data, "-i"],
            vec![(legume_tui::SESSION_VAR, "data-beans-stat-9")],
        ),
    ] {
        let mut plain = term(&dir, &args, &env);
        plain.wait_for("rows (computed on first use)");
        assert!(!plain.cooked());
        assert!(dir.sockets().is_empty(), "{args:?} {env:?}");
        assert!(dir
            .run(&["sessions"])
            .contains("no live data-beans sessions"));
        assert!(!plain.text().contains("ctrl+alt+d"), "no session keys");
        assert!(!plain.text().contains(PROMPT));
        plain.type_(b"q");
        assert_eq!(plain.exit_code(), 0);
        assert!(plain.cooked(), "terminal restored");
    }
}
