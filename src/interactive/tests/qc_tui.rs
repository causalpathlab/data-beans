use super::*;
use crate::qc_lib::{QcConfig, QcMetrics};
use ratatui::backend::TestBackend;
use ratatui::Terminal;

fn press(v: &mut QcView, code: KeyCode) {
    v.handle_key(KeyEvent::new(code, KeyModifiers::NONE));
}

/// 600 cells, one in ten near-ambient, one in fifty with a high mito
/// fraction; 200 genes detected in 0 to 199 cells.
fn rules() -> CellRules {
    let n = 600;
    let n_genes: Vec<f32> = (0..n)
        .map(|c| {
            if c % 10 == 0 {
                20.0 + (c % 7) as f32
            } else {
                900.0 + (c % 37) as f32 * 10.0
            }
        })
        .collect();
    let metrics = QcMetrics {
        total_counts: n_genes.iter().map(|g| g * 2.2).collect(),
        n_genes,
        mito_frac: Some(
            (0..n)
                .map(|c| {
                    if c % 50 == 3 {
                        0.4
                    } else {
                        0.02 + (c % 13) as f32 * 0.002
                    }
                })
                .collect(),
        ),
        ribo_frac: None,
        feature_n_cells: Some((0..200).map(|g| g as f32).collect()),
        n_rows: 200,
    };
    let cfg = QcConfig {
        mito_pattern: Some("^MT-".into()),
        mad_on_mito: true,
        ..QcConfig::default()
    };
    CellRules::new(metrics, cfg)
}

fn screen(v: &mut QcView, w: u16, h: u16) -> String {
    let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
    terminal.draw(|f| v.render(f)).unwrap();
    let buf = terminal.backend().buffer().clone();
    (0..h)
        .map(|y| (0..w).map(|x| buf[(x, y)].symbol()).collect::<String>())
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn draws_every_panel_as_text() {
    let mut v = QcView::new(rules(), "toy.zarr.zip", Some("out".into()), "toy.qc");
    let s = screen(&mut v, 150, 44);
    for want in [
        "cell QC rules",
        "detected genes",
        "total counts",
        "mito fraction",
        "cells per gene",
        "removed cells, by the rules they fail",
        "total counts × detected genes",
        "kept 528 / 600 cells",
        "Enter write out",
    ] {
        assert!(s.contains(want), "missing {want:?} in\n{s}");
    }
    // The ambient cells fail genes and counts both.
    assert!(s.contains("genes + counts"), "{s}");
    // Narrow terminals draw without panicking.
    screen(&mut v, 60, 16);
    screen(&mut v, 20, 6);
}

#[test]
fn keys_move_the_selected_rule() {
    let mut v = QcView::new(rules(), "toy", Some("kept".into()), "toy.qc");
    let before = v.rules().edge(Rule::Genes).unwrap();
    // Right on a lower MAD edge narrows the band: the edge goes up.
    press(&mut v, KeyCode::Right);
    assert!(v.rules().edge(Rule::Genes).unwrap() > before);
    press(&mut v, KeyCode::Char('m'));
    assert_eq!(v.rules().mode(Rule::Genes), Mode::Value);
    let held = v.rules().edge(Rule::Genes).unwrap();
    press(&mut v, KeyCode::Right);
    assert!(v.rules().edge(Rule::Genes).unwrap() > held);
    // Typed edge.
    press(&mut v, KeyCode::Char('e'));
    for _ in 0..12 {
        press(&mut v, KeyCode::Backspace);
    }
    for c in "50".chars() {
        press(&mut v, KeyCode::Char(c));
    }
    press(&mut v, KeyCode::Enter);
    assert_eq!(v.rules().edge(Rule::Genes), Some(50.0));
    // The gene rule, off at the start, comes on where it is moved.
    press(&mut v, KeyCode::Up);
    assert!(!v.rules().is_on(Rule::GeneCells));
    press(&mut v, KeyCode::Right);
    assert!(v.rules().is_on(Rule::GeneCells));
    press(&mut v, KeyCode::Char('r'));
    assert!(!v.rules().is_on(Rule::GeneCells));
    assert_eq!(v.rules().edge(Rule::Genes).unwrap(), before);
    assert_eq!(v.decision(), None);
}

#[test]
fn p_shows_the_flags_and_enter_applies() {
    let mut v = QcView::new(rules(), "toy", Some("kept".into()), "toy.qc");
    press(&mut v, KeyCode::Down);
    press(&mut v, KeyCode::Char('m'));
    press(&mut v, KeyCode::Char('p'));
    let s = screen(&mut v, 150, 44);
    assert!(
        s.contains("flags: --qc-min-counts 1121.4731 --qc-mito-pattern '^MT-'"),
        "{s}"
    );
    assert!(s.contains("--qc-mad-on-counts=false"), "{s}");
    press(&mut v, KeyCode::Enter);
    assert_eq!(v.decision(), Some(Decision::Apply));
    let mut q = QcView::new(rules(), "toy", Some("kept".into()), "toy.qc");
    press(&mut q, KeyCode::Char('q'));
    assert_eq!(q.decision(), Some(Decision::Cancel));
}

#[test]
fn s_saves_the_figure() {
    let dir = tempfile::tempdir().unwrap();
    let mut v = QcView::new(rules(), "toy", Some("kept".into()), "toy.qc");
    press(&mut v, KeyCode::Char('s'));
    for _ in 0..40 {
        press(&mut v, KeyCode::Backspace);
    }
    let name = dir.path().join("fig").to_string_lossy().into_owned();
    for c in name.chars() {
        press(&mut v, KeyCode::Char(c));
    }
    press(&mut v, KeyCode::Enter);
    assert!(dir.path().join("fig.pdf").exists() && dir.path().join("fig.png").exists());
    assert!(screen(&mut v, 150, 44).contains("saved "));
}

#[test]
fn stems() {
    assert_eq!(stem("data/pbmc.zarr.zip"), "pbmc");
    assert_eq!(stem("x.h5"), "x");
    assert_eq!(stem("plain"), "plain");
}

#[test]
fn enter_without_an_output_asks_for_one_and_never_replaces() {
    let dir = tempfile::tempdir().unwrap();
    let taken = dir.path().join("toy.qc");
    std::fs::write(taken.with_extension("qc.zarr.zip"), b"").unwrap();
    let suggest = taken.to_string_lossy().into_owned();
    let mut v = QcView::new(rules(), "toy", None, &suggest);
    assert!(screen(&mut v, 150, 44).contains("Enter write…"));
    press(&mut v, KeyCode::Enter);
    assert_eq!(v.decision(), None);
    // The suggestion skips the name already there.
    let s = screen(&mut v, 150, 44);
    assert!(
        s.contains(&format!("write the kept cells to: {suggest}-2")),
        "{s}"
    );
    // Typing the taken name back is refused, offering the free one.
    for _ in 0..2 {
        press(&mut v, KeyCode::Backspace);
    }
    press(&mut v, KeyCode::Enter);
    assert_eq!(v.decision(), None);
    assert!(screen(&mut v, 150, 44).contains("is already there; not replacing it"));
    press(&mut v, KeyCode::Enter);
    assert_eq!(v.decision(), Some(Decision::Apply));
    assert_eq!(v.output(), Some(format!("{suggest}-2").as_str()));
}

#[test]
fn c_copies_the_flags_and_says_so() {
    let mut v = QcView::new(rules(), "toy", Some("kept".into()), "toy.qc");
    press(&mut v, KeyCode::Char('c'));
    // The mito pattern is past the defaults already.
    let s = screen(&mut v, 150, 44);
    assert!(
        s.contains("copied to the clipboard: --qc-mito-pattern '^MT-'"),
        "{s}"
    );
    press(&mut v, KeyCode::Down);
    press(&mut v, KeyCode::Char('m'));
    press(&mut v, KeyCode::Char('c'));
    let s = screen(&mut v, 150, 44);
    assert!(
        s.contains("copied to the clipboard: --qc-min-counts"),
        "{s}"
    );
}

#[test]
fn s_never_replaces_a_figure() {
    let dir = tempfile::tempdir().unwrap();
    let name = dir.path().join("fig").to_string_lossy().into_owned();
    std::fs::write(format!("{name}.pdf"), b"mine").unwrap();
    let mut v = QcView::new(rules(), "toy", Some("kept".into()), "toy.qc");
    press(&mut v, KeyCode::Char('s'));
    for _ in 0..80 {
        press(&mut v, KeyCode::Backspace);
    }
    for c in name.chars() {
        press(&mut v, KeyCode::Char(c));
    }
    press(&mut v, KeyCode::Enter);
    assert_eq!(std::fs::read(format!("{name}.pdf")).unwrap(), b"mine");
    assert!(screen(&mut v, 150, 44).contains("is already there; not replacing it"));
    // Enter again takes the free name offered.
    press(&mut v, KeyCode::Enter);
    assert!(dir.path().join("fig_2.pdf").exists());
}

#[test]
fn the_guardrail_is_said() {
    let mut v = QcView::new(rules(), "toy", Some("kept".into()), "toy.qc");
    // The counts floor above every cell: QC would drop all, and keeps all.
    press(&mut v, KeyCode::Down);
    press(&mut v, KeyCode::Char('e'));
    for _ in 0..20 {
        press(&mut v, KeyCode::Backspace);
    }
    for c in "1000000".chars() {
        press(&mut v, KeyCode::Char(c));
    }
    press(&mut v, KeyCode::Enter);
    assert!(v.rules().verdicts().guarded);
    assert!(screen(&mut v, 150, 44).contains("QC keeps them all instead"));
}
