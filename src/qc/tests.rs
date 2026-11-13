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
    assert_eq!(fmt_stat(0.0, 2), "0");
    assert_eq!(fmt_stat(5.0, 2), "5");
    assert_eq!(fmt_stat(900.0, 2), "900");
    assert_eq!(fmt_stat(0.3, 2), "0.30");
    assert_eq!(fmt_stat(12.53, 2), "12.53");
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

#[test]
fn printed_ranges_keep_three_significant_digits() {
    assert_eq!(fmt_sig(0.000412), "0.000412");
    assert_eq!(fmt_sig(0.0731), "0.0731");
    assert_eq!(fmt_sig(2.4567), "2.46");
    assert_eq!(fmt_sig(145.0), "145");
    assert_eq!(fmt_sig(1234.5), "1234");
}

#[test]
fn printed_mean_holds_past_f32_sum_precision() {
    // 2^24 + 1 is not an f32: a sum kept in f32 would stay at 2^24.
    let vals: Vec<f32> = vec![16_777_216.0, 1.0, 1.0, 1.0, 1.0];
    let mut out = Vec::new();
    write_nnz_summary(&mut out, "Cell", "nnz", &vals, 0, None).unwrap();
    let text = String::from_utf8(out).unwrap();
    assert!(text.contains("Mean: 3355444.00"), "{text}");
}

#[test]
fn log_histogram_with_no_cutoff_spans_the_values_alone() {
    // Values within a factor of 1.5: with no cutoff the axis must not reach
    // down to 0, which would crowd them into its last two bins.
    let vals: Vec<f32> = (0..=1000).map(|i| 20_000.0 + 10.0 * i as f32).collect();
    let hist = create_log_histogram(&vals, 0);
    assert_eq!(hist.len(), PRINTED_BINS);
    // A cutoff that is set still has its place on the axis.
    let hist = create_log_histogram(&vals, 100);
    assert!(hist.len() < 5 && hist[0].is_cutoff);
}

#[test]
fn linear_ticks_of_a_range_with_no_span_end() {
    assert_eq!(Axis::linear(0.0, 0.0).ticks(), vec![0.0]);
    assert_eq!(Axis::linear(3.0, 3.0).ticks(), vec![3.0]);
    assert!(Axis::linear(1.0, 0.0).ticks().is_empty());
    assert!(Axis::linear(0.0, f64::INFINITY).ticks().is_empty());
    assert!(Axis::linear(0.0, f64::NAN).ticks().is_empty());
    assert!(Axis::linear(0.0, 5e-324).ticks().is_empty());
    assert_eq!(
        Axis::linear(0.0, 1.0).ticks(),
        vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    );
}

/// 40 rows by 150 columns, about a fifth filled, with empty rows and columns.
fn sparse_fixture() -> Box<dyn SparseIo<IndexIter = Vec<usize>>> {
    let (nrows, ncols) = (40u64, 150u64);
    let mut triplets = vec![];
    for c in 0..ncols {
        for r in 0..nrows {
            let h = (r * 7919 + c * 104_729 + r * c) % 97;
            if h < 20 && r != 11 && c % 31 != 5 {
                triplets.push((r, c, 0.5 + h as f32));
            }
        }
    }
    let shape = (nrows as usize, ncols as usize, triplets.len());
    let mut data = create_sparse_from_triplets(&triplets, shape, None, None).unwrap();
    let names =
        |n: u64, p: &str| -> Vec<Box<str>> { (0..n).map(|i| format!("{p}{i}").into()).collect() };
    data.register_row_names_vec(&names(nrows, "g"));
    data.register_column_names_vec(&names(ncols, "c"));
    data
}

#[test]
fn one_pass_statistics_match_the_separate_passes() {
    let data = sparse_fixture();
    let block = Some(16);
    let (col, row) = collect_column_row_stat(data.as_ref(), block).unwrap();
    let (col_alone, row_alone) = (
        collect_column_stat(data.as_ref(), block).unwrap(),
        collect_row_stat(data.as_ref(), block).unwrap(),
    );
    assert!(col.sum().iter().any(|&s| s > 0.0));
    assert_eq!(col.count_positives(), col_alone.count_positives());
    assert_eq!(col.sum(), col_alone.sum());
    assert_eq!(col.mean(), col_alone.mean());
    assert_eq!(row.count_positives(), row_alone.count_positives());
}

/// A backend nothing was written to has no shape, so no counts to squeeze
/// by: it is refused.
#[test]
fn squeeze_refuses_a_backend_without_a_shape() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.zarr");
    let data = create_sparse_streaming_empty(path.to_str(), Some(&SparseIoBackend::Zarr)).unwrap();
    assert!(data.num_rows().is_none() || data.num_columns().is_none());
    let cutoffs = SqueezeCutoffs { row: 1, column: 1 };
    let err = squeeze_by_nnz(data.as_ref(), cutoffs, None, false).unwrap_err();
    assert!(
        err.to_string().contains("missing shape information"),
        "{err}"
    );
}

/// A row's sum runs across the column blocks, and a sum of non-integer
/// values depends on how it is grouped: the grouping follows the blocks, not
/// the threads that read them.
#[test]
fn row_statistics_do_not_depend_on_the_thread_count() -> anyhow::Result<()> {
    let (nrows, ncols) = (8u64, 600u64);
    let mut triplets = vec![];
    for c in 0..ncols {
        for r in 0..nrows {
            if (r * 5 + c * 3) % 4 != 0 {
                triplets.push((r, c, 0.1 + ((r * 131 + c * 17) % 97) as f32 * 0.37));
            }
        }
    }
    let shape = (nrows as usize, ncols as usize, triplets.len());
    let data = create_sparse_from_triplets(&triplets, shape, None, None)?;

    let bits = |x: Vec<f32>| -> Vec<u32> { x.iter().map(|v| v.to_bits()).collect() };
    // Blocks of two columns: several leaves of the fold, merged.
    let run = |threads: usize| -> anyhow::Result<_> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()?
            .install(|| {
                let alone = collect_row_stat(data.as_ref(), Some(2))?;
                let (_, with_columns) = collect_column_row_stat(data.as_ref(), Some(2))?;
                Ok((
                    bits(alone.sum()),
                    bits(alone.variance()),
                    bits(with_columns.sum()),
                    bits(with_columns.variance()),
                ))
            })
    };
    let one = run(1)?;
    assert_eq!(one.0, one.2, "the two passes group a row's sum alike");
    for threads in [3, 6] {
        assert_eq!(run(threads)?, one, "{threads} threads");
    }
    Ok(())
}
