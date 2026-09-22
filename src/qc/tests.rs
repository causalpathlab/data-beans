use super::*;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};

////////////////////////////////////////////////////////////////////
// histogram-trough cutoff suggestion                              //
////////////////////////////////////////////////////////////////////

/// `n` draws whose log10(nnz) is `N(log10(centre), sd_decades²)`.
fn log_normal(rng: &mut StdRng, n: usize, centre: f64, sd_decades: f64) -> Vec<f32> {
    let d = Normal::new(centre.log10(), sd_decades).unwrap();
    (0..n)
        .map(|_| 10f64.powf(d.sample(rng)).round().max(1.0) as f32)
        .collect()
}

#[test]
fn trough_accepts_clear_bimodal() {
    // A large ambient peak (~5) + a real-cell mode (~300), well separated.
    let mut rng = StdRng::seed_from_u64(7);
    let amb = Normal::new(5.0_f64, 1.5).unwrap();
    let real = Normal::new(300.0_f64, 40.0).unwrap();
    let mut nnz: Vec<f32> = (0..2000)
        .map(|_| amb.sample(&mut rng).max(1.0) as f32)
        .collect();
    nnz.extend((0..400).map(|_| real.sample(&mut rng).max(1.0) as f32));
    let c = suggest_nnz_cutoff(&nnz).expect("clear bimodal → Some(cutoff)");
    assert!(
        c > 15 && c < 200,
        "cutoff {c} should sit in the ambient↔real valley"
    );
}

#[test]
fn trough_rejects_unimodal() {
    let mut rng = StdRng::seed_from_u64(11);
    let one = Normal::new(60.0_f64, 12.0).unwrap();
    let nnz: Vec<f32> = (0..3000)
        .map(|_| one.sample(&mut rng).max(1.0) as f32)
        .collect();
    assert!(
        suggest_nnz_cutoff(&nnz).is_none(),
        "single-mode nnz → no cutoff"
    );
}

#[test]
fn trough_rejects_discrete_lowcount_unimodal() {
    // Integer counts leave empty log-bins between small values; those gaps
    // must not read as troughs.
    let mut rng = StdRng::seed_from_u64(3);
    let p = rand_distr::Poisson::new(6.0_f64).unwrap();
    let nnz: Vec<f32> = (0..5000)
        .map(|_| p.sample(&mut rng).max(1.0) as f32)
        .collect();
    assert!(
        suggest_nnz_cutoff(&nnz).is_none(),
        "unimodal discrete low-count nnz → no cutoff"
    );
}

#[test]
fn trough_accepts_lowcount_bimodal() {
    // Genuine bimodality is still found when the ambient mode is a tight
    // low-count spike.
    let mut rng = StdRng::seed_from_u64(5);
    let amb = rand_distr::Poisson::new(2.0_f64).unwrap();
    let real = Normal::new(500.0_f64, 60.0).unwrap();
    let mut nnz: Vec<f32> = (0..6000)
        .map(|_| amb.sample(&mut rng).max(1.0) as f32)
        .collect();
    nnz.extend((0..800).map(|_| real.sample(&mut rng).max(1.0) as f32));
    let c = suggest_nnz_cutoff(&nnz).expect("low-count ambient + real mode → Some(cutoff)");
    assert!(c > 5 && c < 400, "cutoff {c} should land in the valley");
}

#[test]
fn trough_rejects_called_cells_with_a_long_left_tail() {
    // Already-called cells: one mode with a long, thin left tail of
    // low-complexity cells. No trough separates them — nothing is empty here.
    let mut rng = StdRng::seed_from_u64(13);
    let mut nnz = log_normal(&mut rng, 10_000, 11_000.0, 0.08);
    let tail = rand_distr::Uniform::new(800f64.log10(), 9_000f64.log10()).unwrap();
    nnz.extend((0..1_500).map(|_| 10f64.powf(tail.sample(&mut rng)).round() as f32));
    assert!(
        suggest_nnz_cutoff(&nnz).is_none(),
        "called cells with a left tail → no cutoff"
    );
}

#[test]
fn trough_finds_rare_cells_under_an_ambient_majority() {
    // Unfiltered barcodes: ~98% ambient (a spike at 1 plus a broad mode
    // around tens) and a small cell mode three decades up. A two-Gaussian
    // model on this shape prefers one cluster and cuts nothing.
    let mut rng = StdRng::seed_from_u64(17);
    let mut nnz = vec![1.0_f32; 6_000];
    nnz.extend(log_normal(&mut rng, 54_000, 30.0, 0.45));
    let n_cells = 1_200;
    nnz.extend(log_normal(&mut rng, n_cells, 12_000.0, 0.12));
    let c = suggest_nnz_cutoff(&nnz).expect("ambient majority + cell mode → Some(cutoff)");
    assert!(c > 500 && c < 6_000, "cutoff {c} should sit in the trough");
    let kept = nnz.iter().filter(|&&x| x as usize >= c).count();
    assert!(
        kept >= n_cells && kept < n_cells + n_cells / 5,
        "kept {kept} barcodes for {n_cells} cells"
    );
}

#[test]
fn trough_ignores_a_nearby_doublet_bump() {
    // A doublet bump at twice the main mode is a trough, but not an
    // ambient↔cell one: cutting there would drop every singlet.
    let mut rng = StdRng::seed_from_u64(19);
    let mut nnz = log_normal(&mut rng, 5_000, 3_000.0, 0.03);
    nnz.extend(log_normal(&mut rng, 300, 6_500.0, 0.02));
    assert!(
        suggest_nnz_cutoff(&nnz).is_none(),
        "modes within a decade → no cutoff"
    );
}

#[test]
fn trough_handles_zero_counts() {
    // All-zero columns/rows are routine (a feature seen in no cell); count 0
    // owns `[-1/2, 1/2)`, i.e. `ln(1/2)..ln(3/2)`, never `ln 0`.
    let mut rng = StdRng::seed_from_u64(23);
    let mut nnz = vec![0.0_f32; 3_000];
    nnz.extend(log_normal(&mut rng, 800, 2_000.0, 0.1));
    let c = suggest_nnz_cutoff(&nnz).expect("zeros + a real mode → Some(cutoff)");
    assert!((1..1_000).contains(&c), "cutoff {c}");
}

#[test]
fn cutoff_is_deterministic() {
    let mut rng = StdRng::seed_from_u64(9);
    let amb = Normal::new(4.0_f64, 1.0).unwrap();
    let real = Normal::new(200.0_f64, 30.0).unwrap();
    let mut nnz: Vec<f32> = (0..3000)
        .map(|_| amb.sample(&mut rng).max(1.0) as f32)
        .collect();
    nnz.extend((0..500).map(|_| real.sample(&mut rng).max(1.0) as f32));
    let a = suggest_nnz_cutoff(&nnz);
    let b = suggest_nnz_cutoff(&nnz);
    assert_eq!(a, b, "the trough search has no RNG");
    assert!(a.is_some());
}

////////////////////////////////////////////////////////////////////
// generalized histogram rendering (used by `histogram` command)   //
////////////////////////////////////////////////////////////////////

#[test]
fn fmt_stat_is_integer_for_whole_values() {
    // nnz / sum are whole -> no decimal point; mean / sd keep 2 decimals.
    assert_eq!(fmt_stat(0.0), "0");
    assert_eq!(fmt_stat(5.0), "5");
    assert_eq!(fmt_stat(900.0), "900");
    assert_eq!(fmt_stat(0.3), "0.30");
    assert_eq!(fmt_stat(12.53), "12.53");
}

#[test]
fn log_histogram_tracks_exact_ranges_and_no_cutoff_when_zero() {
    // Counts are preserved, per-bin ranges are exact f32, and cutoff == 0
    // (the `histogram` command) marks no bin.
    let vals: Vec<f32> = vec![1.0, 1.0, 2.0, 100.0, 100.0, 100.0];
    let hist = create_log_histogram(&vals, 0);
    let total: usize = hist.iter().map(|b| b.count).sum();
    assert_eq!(total, vals.len());
    assert!(
        hist.iter().all(|b| !b.is_cutoff),
        "cutoff == 0 must not mark any bin"
    );
    // The 100-valued bin collapses to a single exact value.
    let big = hist
        .iter()
        .find(|b| b.count == 3)
        .expect("three 100s land in one bin");
    assert_eq!(big.val_min, 100.0);
    assert_eq!(big.val_max, 100.0);
}

#[test]
fn log_histogram_marks_cutoff_bin_when_positive() {
    let vals: Vec<f32> = vec![1.0, 2.0, 3.0, 200.0, 300.0];
    let hist = create_log_histogram(&vals, 50);
    assert!(
        hist.iter().any(|b| b.is_cutoff),
        "a positive cutoff should mark exactly the first bin at/above it"
    );
}
