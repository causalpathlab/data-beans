use super::*;
use crate::qc_lib::{QcConfig, QcMetrics};
use crate::qc_rules::CellRules;

#[test]
fn log_axis_ticks_run_one_two_five_from_zero() {
    let axis = Axis::log1p(0.0, 300.0);
    assert_eq!(axis.ticks()[0], 0.0);
    assert!(axis.ticks().len() <= 8);
    assert!(axis.ticks().contains(&100.0));
    let lin = Axis::linear(0.0, 0.18);
    assert_eq!(lin.ticks(), vec![0.0, 0.05, 0.1, 0.15]);
}

#[test]
fn bins_cover_the_range_and_clamp() {
    let axis = Axis::linear(0.0, 10.0);
    assert_eq!(axis.bin(0.0, 10), 0);
    assert_eq!(axis.bin(10.0, 10), 9);
    assert_eq!(axis.bin(-5.0, 10), 0);
    assert_eq!(
        axis.counts(&[1.0, 1.5, 9.9, f32::NAN], 10, |_| true),
        vec![0, 2, 0, 0, 0, 0, 0, 0, 0, 1]
    );
    let log = Axis::fit(&[0.0, 9.0, 99.0], true);
    assert!((log.frac(9.0) - 0.5).abs() < 1e-12);
    // Small values get a smaller shift, so they spread out.
    let small = Axis::fit(&[0.0, 0.01, 0.1, 1.0, 8.0], true);
    assert!((small.shift - 0.005).abs() < 1e-9);
    assert!(small.frac(0.1) > 0.3);
    assert!((small.at(small.frac(0.5)) - 0.5).abs() < 1e-9);
}

#[test]
fn short_labels() {
    assert_eq!(short(0.0), "0");
    assert_eq!(short(0.05), "0.05");
    assert_eq!(short(2.5), "2.5");
    assert_eq!(short(144.2), "144");
    assert_eq!(short(1500.0), "1.5k");
    assert_eq!(short(25_000.0), "25k");
    assert_eq!(short(2_000_000.0), "2M");
}

#[test]
fn saved_paths_follow_the_extension() {
    assert_eq!(saved_paths("qc"), vec!["qc.pdf", "qc.png"]);
    assert_eq!(saved_paths("out/qc.png"), vec!["out/qc.png"]);
    assert_eq!(saved_paths("qc.svg"), vec!["qc.svg"]);
}

fn rules() -> CellRules {
    let n = 400;
    let n_genes: Vec<f32> = (0..n)
        .map(|c| {
            if c % 10 == 0 {
                20.0
            } else {
                900.0 + (c % 37) as f32 * 10.0
            }
        })
        .collect();
    let total: Vec<f32> = n_genes.iter().map(|g| g * 2.2).collect();
    let mito: Vec<f32> = (0..n)
        .map(|c| {
            if c % 50 == 3 {
                0.4
            } else {
                0.02 + (c % 13) as f32 * 0.002
            }
        })
        .collect();
    let metrics = QcMetrics {
        n_genes,
        total_counts: total,
        mito_frac: Some(mito),
        ribo_frac: None,
        feature_n_cells: Some((0..100).map(|g| g as f32).collect()),
        n_rows: 100,
    };
    let cfg = QcConfig {
        mito_pattern: Some("^MT-".into()),
        mad_on_mito: true,
        ..QcConfig::default()
    };
    CellRules::new(metrics, cfg)
}

#[test]
fn qc_figure_saves_as_pdf_and_png() {
    let rules = rules();
    let svg = qc::figure(&rules, "toy");
    assert!(svg.contains("Detected genes") && svg.contains("Why cells were removed"));
    let dir = tempfile::tempdir().unwrap();
    let name = dir.path().join("qc").to_string_lossy().into_owned();
    let paths = save(&svg, &name).unwrap();
    assert_eq!(paths.len(), 2);
    let png = std::fs::read(&paths[1]).unwrap();
    assert_eq!(&png[1..4], b"PNG");
    assert!(std::fs::read(&paths[0]).unwrap().starts_with(b"%PDF"));
    assert_eq!(free_name(&name), format!("{name}_2"));
    // A second save under the name is refused, the files kept.
    let err = save(&svg, &name).unwrap_err().to_string();
    assert!(err.contains("already there; not replacing it"), "{err}");
}

#[test]
fn stat_figure_draws_every_panel() {
    let nnz: Vec<f32> = (0..50).map(|i| i as f32).collect();
    let sum: Vec<f32> = nnz.iter().map(|v| v * 3.0).collect();
    let mean: Vec<f32> = nnz.iter().map(|v| v / 10.0).collect();
    let sd: Vec<f32> = mean.iter().map(|v| v.sqrt()).collect();
    let marked: Vec<bool> = (0..50).map(|i| i % 7 == 0).collect();
    let svg = stat::figure(&stat::StatData {
        title: "toy",
        side: "rows",
        values: [&nnz, &sum, &mean, &sd],
        marked: Some(&marked),
        cursor: Some((10, "GENE10")),
    });
    assert!(svg.contains("Mean against sd") && svg.contains("GENE10"));
    assert!(png(&svg, 1.0).unwrap().len() > 1000);
}

/// Writes the QC figure of the backend `DATA_BEANS_FIGURE_DATA` names to
/// `DATA_BEANS_FIGURE_OUT`, for looking at; skipped without them.
#[test]
fn qc_figure_of_a_backend() {
    let (Ok(data), Ok(out)) = (
        std::env::var("DATA_BEANS_FIGURE_DATA"),
        std::env::var("DATA_BEANS_FIGURE_OUT"),
    ) else {
        return;
    };
    let (backend, file) = crate::hdf5_io::resolve_backend_file(&data, None).unwrap();
    let m = crate::sparse_io::open_sparse_matrix(&file, &backend).unwrap();
    let mut v = crate::sparse_io_vector::SparseIoVec::new();
    v.push(std::sync::Arc::from(m), None).unwrap();
    let cfg = QcConfig {
        mito_pattern: Some("(?i)^MT-".into()),
        mad_on_mito: true,
        feature_min_cells: 10,
        ..QcConfig::default()
    };
    let metrics = crate::qc_lib::compute_qc_metrics(&v, &cfg, None, true).unwrap();
    let rules = CellRules::new(metrics, cfg);
    save(&qc::figure(&rules, "demo.zarr.zip"), &out).unwrap();
}
