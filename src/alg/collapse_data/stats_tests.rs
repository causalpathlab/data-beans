use super::*;
use legume_numeric::param::traits::Inference;
use nalgebra::{DMatrix, DVector};

/// Build a small but non-trivial multi-batch `CollapsedStat`.
fn toy_stat(num_genes: usize, num_samples: usize, num_batches: usize) -> CollapsedStat {
    let mut s = CollapsedStat::new(num_genes, num_samples, num_batches);
    // Deterministic pseudo-random fills; keep everything strictly
    // positive so the Gamma updates are well-defined.
    let f = |a: usize, b: usize| -> f32 { 1.0 + ((a * 7 + b * 13) % 11) as f32 };
    s.observed_sum_ds = DMatrix::from_fn(num_genes, num_samples, &f);
    s.imputed_sum_ds = DMatrix::from_fn(num_genes, num_samples, |g, c| 0.5 * f(g + 1, c + 2));
    s.size_s = DVector::from_fn(num_samples, |c, _| 2.0 + (c % 3) as f32);
    s.observed_sum_db = DMatrix::from_fn(num_genes, num_batches, |g, b| f(g, b) + 0.7);
    s.n_bs = DMatrix::from_fn(num_batches, num_samples, |b, c| 1.0 + ((b + c) % 4) as f32);
    s.matched_bs = DMatrix::from_fn(num_batches, num_samples, |b, c| {
        0.5 + ((b + 2 * c) % 3) as f32
    });
    s
}

/// Two batch-pure pseudobulks of ONE cell type: pb 0 in batch 0 (frame `a`),
/// pb 1 in batch 1 (frame `b`), each matched to the other. `n` cells each, so the
/// unit priors are negligible.
fn two_pure_pbs(a: &[f32], b: &[f32], n: f32) -> CollapsedStat {
    let g = a.len();
    let mut s = CollapsedStat::new(g, 2, 2);
    for i in 0..g {
        s.observed_sum_ds[(i, 0)] = a[i] * n;
        s.imputed_sum_ds[(i, 0)] = b[i] * n;
        s.observed_sum_ds[(i, 1)] = b[i] * n;
        s.imputed_sum_ds[(i, 1)] = a[i] * n;
        s.observed_sum_db[(i, 0)] = a[i] * n;
        s.observed_sum_db[(i, 1)] = b[i] * n;
    }
    s.size_s = DVector::from_element(2, n);
    s.n_bs = DMatrix::from_row_slice(2, 2, &[n, 0.0, 0.0, n]);
    // pb 0's counterfactual comes from batch 1 and vice versa
    s.matched_bs = DMatrix::from_row_slice(2, 2, &[0.0, n, n, 0.0]);
    s
}

fn assert_rel(got: f32, want: f32, tol: f32, tag: &str) {
    assert!(
        (got / want - 1.0).abs() < tol,
        "{tag}: got {got}, want {want} (rel tol {tol})"
    );
}

/// Pooled adjustment: both pseudobulks land in ONE frame — the per-gene
/// geometric mean of the two batch frames — with `δ` carrying the whole
/// batch ratio, normalised to geometric mean 1 across batches. The per-pb
/// outputs keep their documented meaning: `mu_residual` is the own-batch fold
/// `E[y]/E[μ]`, `gamma` the counterfactual's `E[ŷ]/E[μ]`.
#[test]
#[allow(clippy::needless_range_loop)] // `g` indexes four parallel tables
fn pooled_two_pure_batches_share_one_frame() {
    let a = [8.0f32, 2.0, 12.0, 1.0];
    let b = [2.0f32, 8.0, 3.0, 4.0];
    let stat = two_pure_pbs(&a, &b, 2000.0);
    let out = optimize_block(&stat, (1.0, 1.0), 60, CalibrateTarget::All, None).unwrap();
    let mu = out.mu_adjusted.as_ref().unwrap().posterior_mean();
    let delta = out.delta.as_ref().unwrap().posterior_mean();
    let resid = out.mu_residual.as_ref().unwrap().posterior_mean();
    let gamma = out.gamma.as_ref().unwrap().posterior_mean();
    for g in 0..a.len() {
        let common = (a[g] * b[g]).sqrt();
        assert_rel(mu[(g, 0)], common, 0.02, &format!("gene {g} mu pb0"));
        assert_rel(mu[(g, 1)], common, 0.02, &format!("gene {g} mu pb1"));
        assert_rel(
            delta[(g, 0)] / delta[(g, 1)],
            a[g] / b[g],
            0.02,
            &format!("gene {g} delta ratio"),
        );
        assert_rel(
            delta[(g, 0)] * delta[(g, 1)],
            1.0,
            0.02,
            &format!("gene {g} delta geometric mean"),
        );
        assert_rel(
            resid[(g, 0)],
            delta[(g, 0)],
            0.02,
            &format!("gene {g} mu_residual pb0 = own delta"),
        );
        assert_rel(
            resid[(g, 1)],
            delta[(g, 1)],
            0.02,
            &format!("gene {g} mu_residual pb1 = own delta"),
        );
        assert_rel(
            gamma[(g, 0)],
            delta[(g, 1)],
            0.02,
            &format!("gene {g} gamma pb0 = source delta"),
        );
    }
}

/// Anchored: the anchor batch's frame is the frame. Its pseudobulk self-matches
/// (counterfactual = its own profile), the other batch is matched to it. Then
/// `δ_anchor = 1`, both `μ` equal the anchor profile, and the new batch's `δ` is
/// its fold relative to the anchor.
#[test]
fn anchored_frame_is_the_anchor_batch() {
    let a = [8.0f32, 2.0, 12.0, 1.0];
    let b = [2.0f32, 8.0, 3.0, 4.0];
    let n = 2000.0;
    let mut stat = two_pure_pbs(&a, &b, n);
    // pb 1 (batch 1) is the anchor: it self-matches.
    for (g, &bg) in b.iter().enumerate() {
        stat.imputed_sum_ds[(g, 1)] = bg * n;
    }
    stat.matched_bs = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, n, n]);
    stat.anchor_batches = vec![1];
    let out = optimize_block(&stat, (1.0, 1.0), 60, CalibrateTarget::All, None).unwrap();
    let mu = out.mu_adjusted.as_ref().unwrap().posterior_mean();
    let delta = out.delta.as_ref().unwrap().posterior_mean();
    for g in 0..a.len() {
        assert_rel(delta[(g, 1)], 1.0, 0.02, &format!("gene {g} anchor delta"));
        assert_rel(
            delta[(g, 0)],
            a[g] / b[g],
            0.03,
            &format!("gene {g} new-batch delta"),
        );
        assert_rel(
            mu[(g, 0)],
            b[g],
            0.03,
            &format!("gene {g} mu pb0 in the anchor frame"),
        );
        assert_rel(mu[(g, 1)], b[g], 0.02, &format!("gene {g} mu pb1"));
    }
}

/// Coarsening adds the counterfactual source mass like every other per-sample
/// statistic, and carries the anchor set.
#[test]
fn merge_stat_sums_matched_mass_and_keeps_anchors() {
    let mut fine = toy_stat(3, 4, 2);
    fine.anchor_batches = vec![0];
    let coarse = merge_stat(&fine, &[0, 1, 0, 1], 2);
    for b in 0..2 {
        assert_eq!(
            coarse.matched_bs[(b, 0)],
            fine.matched_bs[(b, 0)] + fine.matched_bs[(b, 2)]
        );
        assert_eq!(
            coarse.matched_bs[(b, 1)],
            fine.matched_bs[(b, 1)] + fine.matched_bs[(b, 3)]
        );
    }
    assert_eq!(coarse.anchor_batches, vec![0]);
    let sub = fine.select_rows(1, 2);
    assert_eq!(sub.matched_bs, fine.matched_bs);
    assert_eq!(sub.anchor_batches, vec![0]);
    let cols = fine.select_columns(&[3, 1]);
    assert_eq!(cols.matched_bs.column(0), fine.matched_bs.column(3));
    assert_eq!(cols.anchor_batches, vec![0]);
}

fn assert_mat_close(a: &DMatrix<f32>, b: &DMatrix<f32>, tag: &str) {
    assert_eq!(a.shape(), b.shape(), "{tag}: shape mismatch");
    for (x, y) in a.iter().zip(b.iter()) {
        assert!(
            (x - y).abs() <= 1e-5 * (1.0 + x.abs().max(y.abs())),
            "{tag}: {x} vs {y}"
        );
    }
}

/// The whole point of gene-blocking: a per-row-block fit reassembled by
/// `vconcat` must equal the single-shot fit, because every update is
/// separable across feature rows.
#[test]
fn blocked_optimize_matches_single_block() {
    let (g, k, b) = (10usize, 4usize, 2usize);
    let stat = toy_stat(g, k, b);
    let hyper = (1.0, 1.0);
    let iters = 25;

    let full = optimize_block(&stat, hyper, iters, CalibrateTarget::All, None).unwrap();

    // Split rows into uneven blocks and reassemble.
    let ranges = [(0usize, 3usize), (3, 4), (7, 3)];
    let mut mu_obs = Vec::new();
    let mut mu_adj = Vec::new();
    let mut mu_res = Vec::new();
    let mut gam = Vec::new();
    let mut del = Vec::new();
    for (r0, nr) in ranges {
        let sub = stat.select_rows(r0, nr);
        let out = optimize_block(&sub, hyper, iters, CalibrateTarget::All, None).unwrap();
        mu_obs.push(out.mu_observed);
        mu_adj.push(out.mu_adjusted.unwrap());
        mu_res.push(out.mu_residual.unwrap());
        gam.push(out.gamma.unwrap());
        del.push(out.delta.unwrap());
    }
    let blk_obs = GammaMatrix::vconcat(mu_obs, true);
    let blk_adj = GammaMatrix::vconcat(mu_adj, true);
    let blk_res = GammaMatrix::vconcat(mu_res, true);
    let blk_gam = GammaMatrix::vconcat(gam, true);
    let blk_del = GammaMatrix::vconcat(del, true);

    assert_mat_close(
        full.mu_observed.posterior_mean(),
        blk_obs.posterior_mean(),
        "mu_obs mean",
    );
    assert_mat_close(
        full.mu_adjusted.as_ref().unwrap().posterior_mean(),
        blk_adj.posterior_mean(),
        "mu_adj mean",
    );
    assert_mat_close(
        full.mu_residual.as_ref().unwrap().posterior_mean(),
        blk_res.posterior_mean(),
        "mu_resid mean",
    );
    assert_mat_close(
        full.gamma.as_ref().unwrap().posterior_mean(),
        blk_gam.posterior_mean(),
        "gamma mean",
    );
    assert_mat_close(
        full.delta.as_ref().unwrap().posterior_mean(),
        blk_del.posterior_mean(),
        "delta mean",
    );
    // sd / log planes too (All target).
    assert_mat_close(
        full.mu_adjusted.as_ref().unwrap().posterior_log_mean(),
        blk_adj.posterior_log_mean(),
        "mu_adj log_mean",
    );
}

/// MeanOnly drops each mean's per-column prior baseline (unobserved cells
/// → exactly 0), so triplet-ization is sparse; `All` keeps the baseline.
#[test]
fn mean_only_sparsifies_unobserved_cells() {
    let (g, k, b) = (4usize, 3usize, 2usize);
    let mut stat = CollapsedStat::new(g, k, b);
    stat.observed_sum_ds[(0, 0)] = 5.0; // observed support
    stat.observed_sum_ds[(1, 1)] = 3.0;
    stat.imputed_sum_ds[(2, 2)] = 2.0; // imputed-only support
    stat.size_s = DVector::from_element(k, 10.0);
    stat.n_bs = DMatrix::from_element(b, k, 5.0);
    stat.observed_sum_db.fill(1.0); // so δ is well-defined

    let out = optimize_block(&stat, (1.0, 1.0), 10, CalibrateTarget::MeanOnly, None).unwrap();
    let adj = out.mu_adjusted.unwrap();
    let m = adj.posterior_mean();
    // support of mu_adjusted = (observed ∪ imputed) > 0
    assert!(m[(0, 0)] > 0.0);
    assert!(m[(1, 1)] > 0.0);
    assert!(m[(2, 2)] > 0.0);
    // unobserved & unimputed cells → exactly 0 (baseline dropped)
    assert_eq!(m[(3, 0)], 0.0);
    assert_eq!(m[(0, 1)], 0.0);

    // All keeps the prior baseline (nonzero everywhere).
    let out_all = optimize_block(&stat, (1.0, 1.0), 10, CalibrateTarget::All, None).unwrap();
    let ma = out_all.mu_adjusted.unwrap();
    assert!(
        ma.posterior_mean()[(3, 0)] > 0.0,
        "All path must keep the prior baseline"
    );
}

/// MeanOnly: `vconcat(.., false)` keeps the assembled means but drops the
/// sufficient-stat planes, and the means still match the calibrated fit.
#[test]
fn mean_only_vconcat_drops_stats_keeps_means() {
    let (g, k, b) = (9usize, 3usize, 2usize);
    let stat = toy_stat(g, k, b);
    let hyper = (1.0, 1.0);
    let iters = 20;

    let reference = optimize_block(&stat, hyper, iters, CalibrateTarget::All, None).unwrap();

    let mut blocks = Vec::new();
    for (r0, nr) in [(0usize, 5usize), (5, 4)] {
        let sub = stat.select_rows(r0, nr);
        let mut out = optimize_block(&sub, hyper, iters, CalibrateTarget::MeanOnly, None).unwrap();
        out.release_stats();
        blocks.push(out.mu_observed);
    }
    let assembled = GammaMatrix::vconcat(blocks, false);

    // means equal the All-target reference …
    assert_mat_close(
        reference.mu_observed.posterior_mean(),
        assembled.posterior_mean(),
        "mu_obs mean (mean-only)",
    );
    // … but the stat planes were dropped (empty), proving the memory win.
    assert_eq!(
        assembled.posterior_sd().nrows(),
        0,
        "sd should be empty under MeanOnly"
    );
}

#[test]
fn observed_counts_requires_kept_stats() {
    let stat = toy_stat(4, 2, 1);
    let mut out = optimize(
        &stat,
        (1.0, 1.0),
        5,
        "test",
        CalibrateTarget::MeanOnly,
        false,
    )
    .unwrap();
    assert!(!out.stats_kept);
    let err = out
        .observed_counts(&[0, 1])
        .expect_err("released stats must error");
    assert!(
        err.to_string().contains("keep_finest_stats"),
        "unexpected: {err}"
    );

    // Re-fit with stats kept: membership OOB is a soft error, not a panic.
    out = optimize(
        &stat,
        (1.0, 1.0),
        5,
        "test",
        CalibrateTarget::MeanOnly,
        true,
    )
    .unwrap();
    assert!(out.stats_kept);
    let err = out
        .observed_counts(&[0, 99])
        .expect_err("OOB membership must error");
    assert!(
        err.to_string().contains("out of range"),
        "unexpected: {err}"
    );
    let (counts, sizes) = out.observed_counts(&[0, 0, 1]).unwrap();
    assert_eq!(counts.nrows(), 4);
    assert_eq!(sizes, vec![2.0, 1.0]);
}

/// An unmatched sample spanning two batches is split by cell mass, which can
/// charge a batch more of a gene than the batch holds. The δ numerator must
/// stay a valid Gamma shape: positive, finite fold and mean.
#[test]
fn unmatched_split_keeps_the_delta_numerator_nonnegative() {
    let mut s = CollapsedStat::new(2, 2, 2);
    s.exclude_unmatched_from_delta = true;
    // Sample 0: matched, five cells of each batch.
    s.size_s[0] = 10.0;
    s.n_bs[(0, 0)] = 5.0;
    s.n_bs[(1, 0)] = 5.0;
    s.matched_bs[(0, 0)] = 5.0;
    s.matched_bs[(1, 0)] = 5.0;
    // Sample 1: unmatched, one cell per batch; gene 0 comes from batch 1 only.
    s.size_s[1] = 2.0;
    s.n_bs[(0, 1)] = 1.0;
    s.n_bs[(1, 1)] = 1.0;
    for g in 0..2 {
        s.observed_sum_ds[(g, 0)] = 20.0;
        s.imputed_sum_ds[(g, 0)] = 20.0;
        s.observed_sum_db[(g, 0)] = 10.0;
        s.observed_sum_db[(g, 1)] = 10.0;
    }
    s.observed_sum_ds[(0, 1)] = 100.0;
    s.observed_sum_db[(0, 1)] += 100.0;

    let out = optimize(&s, (1.0, 1.0), 20, "test", CalibrateTarget::All, false).unwrap();
    let delta = out.delta.as_ref().expect("delta").posterior_mean();
    let mu = out.mu_adjusted.as_ref().expect("mu").posterior_mean();
    for &v in delta.iter().chain(mu.iter()) {
        assert!(v.is_finite() && v > 0.0, "delta {delta} mu_adjusted {mu}");
    }
}

/// The matched statistics against a direct evaluation of their definition:
/// each pb-sample's counterfactual is the softmax(−distance)-weighted mean of
/// its matches' per-cell profiles, scaled by its own cell count.
#[test]
fn matched_stat_coarse_matches_its_definition() {
    let (d, num_groups, num_batches) = (5usize, 3usize, 2usize);
    let cell_counts = vec![3.0f32, 2.0, 4.0, 0.5, 1.0, 6.0];
    let pb_sample_to_batch = vec![0usize, 0, 1, 1, 1, 0];
    let num_pb = cell_counts.len();
    let layout = PbSampleLayout {
        centroids: DMatrix::zeros(1, num_pb),
        cell_counts: cell_counts.clone(),
        pb_sample_to_batch: pb_sample_to_batch.clone(),
        pb_sample_to_group: vec![0; num_pb],
        bg_to_pbsamp: HashMap::default(),
        cell_to_pbsamp: Vec::new(),
        singleton_col: vec![None; num_pb],
        bulk_batches: Vec::new(),
        pb_sample_to_stratum: None,
    };
    let gene_sums: Vec<Vec<(usize, f32)>> = vec![
        vec![(0, 6.0), (2, 3.0)],
        vec![(1, 4.0), (2, 0.0), (4, 2.0)],
        vec![(0, 8.0), (3, 12.0)],
        vec![(1, 5.0)],
        vec![(2, 7.0), (4, 1.0)],
        vec![(0, 3.0), (1, 9.0), (3, 6.0)],
    ];
    // pb 3 stands for less than a cell: no counterfactual of its own, and no
    // weight as a match. pb 5 has no match at all.
    let matches: Vec<Vec<(usize, f32)>> = vec![
        vec![(2, 0.5), (4, 1.5), (3, 0.1)],
        vec![(4, 0.2), (2, 0.2)],
        vec![(0, 1.0), (1, 0.3), (5, 2.0)],
        vec![(0, 0.4)],
        vec![(5, 0.7), (1, 0.9)],
        vec![],
    ];
    let pbsamp_to_group = vec![0usize, 1, 0, 2, 1, 2];

    let mut want_imputed = DMatrix::<f32>::zeros(d, num_groups);
    let mut want_matched = DMatrix::<f32>::zeros(num_batches, num_groups);
    for p in 0..num_pb {
        if cell_counts[p] < 1.0 || matches[p].is_empty() {
            continue;
        }
        let min_d = matches[p].iter().map(|m| m.1).fold(f32::INFINITY, f32::min);
        let raw: Vec<f32> = matches[p].iter().map(|m| (min_d - m.1).exp()).collect();
        let total: f32 = raw.iter().sum();
        for (&(k, _), r) in matches[p].iter().zip(&raw) {
            if cell_counts[k] < 1.0 {
                continue;
            }
            let w = r / total;
            for &(g, v) in &gene_sums[k] {
                want_imputed[(g, pbsamp_to_group[p])] += cell_counts[p] * w * v / cell_counts[k];
            }
            want_matched[(pb_sample_to_batch[k], pbsamp_to_group[p])] += cell_counts[p] * w;
        }
    }

    let mut stat = CollapsedStat::new(d, num_groups, num_batches);
    collect_matched_stat_coarse(
        &layout,
        &gene_sums,
        &pbsamp_to_group,
        &matches,
        Some(&[1]),
        &mut stat,
    )
    .unwrap();
    assert_eq!(stat.anchor_batches, vec![1]);
    for (got, want) in stat.imputed_sum_ds.iter().zip(want_imputed.iter()) {
        assert!(
            (got - want).abs() <= 1e-4 * (1.0 + want.abs()),
            "{got} vs {want}"
        );
    }
    for (got, want) in stat.matched_bs.iter().zip(want_matched.iter()) {
        assert!(
            (got - want).abs() <= 1e-4 * (1.0 + want.abs()),
            "{got} vs {want}"
        );
    }
    // The untouched planes stay empty.
    assert!(stat.observed_sum_ds.iter().all(|&v| v == 0.0));
}

const TOY_GENES: usize = 6;
const TOY_PER_BATCH: usize = 14;

/// Two batches of sparse integer counts in one Zarr backend: columns
/// `0..TOY_PER_BATCH` are batch "a", the rest batch "b".
fn toy_data() -> (SparseIoVec, Vec<&'static str>, tempfile::TempDir) {
    use crate::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("toy.zarr");
    let n = 2 * TOY_PER_BATCH;
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for j in 0..n {
        for g in 0..TOY_GENES {
            let v = (g * 3 + j * 5 + (j / TOY_PER_BATCH) * g) % 7;
            if v > 0 {
                triplets.push((g as u64, j as u64, v as f32));
            }
        }
    }
    let mut backend = create_sparse_from_triplets(
        &triplets,
        (TOY_GENES, n, triplets.len()),
        Some(path.to_str().expect("utf8")),
        Some(&SparseIoBackend::Zarr),
    )
    .expect("backend");
    let genes: Vec<Box<str>> = (0..TOY_GENES).map(|g| format!("g{g}").into()).collect();
    let cols: Vec<Box<str>> = (0..n).map(|c| format!("c{c}").into()).collect();
    backend.register_row_names_vec(&genes);
    backend.register_column_names_vec(&cols);
    let mut data = SparseIoVec::new();
    data.push(Arc::from(backend), None).expect("push");
    let mut batches = vec!["a"; TOY_PER_BATCH];
    batches.extend(vec!["b"; TOY_PER_BATCH]);
    (data, batches, dir)
}

/// A projection in which no two columns coincide.
fn toy_projection(n: usize) -> DMatrix<f32> {
    DMatrix::from_fn(3, n, |d, c| {
        ((c * (2 * d + 3) + d) % 11) as f32 - 5.0 + 0.013 * c as f32
    })
}

/// The finest observed statistics derived from the pb-sample aggregates are
/// the ones the two passes over the data collect, for ordinary and singleton
/// (summary-batch) pb-samples and under column multiplicities.
#[test]
fn observed_stat_from_pb_samples_matches_the_data_passes() {
    for summary_batches in [&[][..], &[1][..]] {
        let (mut data, batches, _dir) = toy_data();
        let n = data.num_columns();
        let proj = toy_projection(n);
        data.register_batch_membership(&batches);
        let multiplicity: Vec<f32> = (0..n).map(|c| 1.0 + (c % 3) as f32).collect();
        data.register_column_multiplicity(&multiplicity).unwrap();

        // A hash-like finest partition, then pb-samples on it.
        let hash_groups: Vec<usize> = (0..n).map(|c| (c * 7) % 6).collect();
        data.assign_groups(&pad_numeric_labels(&hash_groups, 6), None);
        let pbs = build_pb_samples(&data, &proj, TOY_GENES, summary_batches, &[], None).unwrap();
        let num_pb = pbs.layout.cell_counts.len();
        assert!(num_pb > 6);

        // A refined finest partition: unions of pb-samples across hash groups.
        let k = 4;
        let pbsamp_to_group: Vec<usize> = (0..num_pb).map(|p| (p * 3) % k).collect();
        let cell_to_group: Vec<usize> = pbs
            .layout
            .cell_to_pbsamp
            .iter()
            .map(|&p| pbsamp_to_group[p])
            .collect();
        data.assign_groups(&pad_numeric_labels(&cell_to_group, k), None);
        assert_eq!(data.num_groups(), k);

        let mut want = CollapsedStat::new(TOY_GENES, k, 2);
        data.collect_basic_stat(&mut want).unwrap();
        data.collect_batch_stat(&mut want).unwrap();

        let mut got = CollapsedStat::new(TOY_GENES, k, 2);
        collect_observed_stat_from_pb_samples(
            &data,
            &pbs.layout,
            &pbs.gene_sums,
            &pbsamp_to_group,
            &mut got,
        )
        .unwrap();

        assert!(want.observed_sum_ds.iter().any(|&v| v > 0.0));
        assert_eq!(got.observed_sum_ds, want.observed_sum_ds);
        assert_eq!(got.size_s, want.size_s);
        assert_eq!(got.observed_sum_db, want.observed_sum_db);
        assert_eq!(got.n_bs, want.n_bs);
    }
}

/// The legacy per-cell matched pass credits the counterfactual's source mass
/// to the batch the matched cells came from: a batch-pure sample draws all of
/// it from the other batch and none from its own.
#[test]
fn legacy_matched_mass_is_credited_to_the_matched_batch() {
    let (mut data, batches, _dir) = toy_data();
    let n = data.num_columns();
    data.build_hnsw_per_batch(&toy_projection(n), &batches)
        .unwrap();
    // One sample per batch.
    data.assign_groups(&batches, None);
    let batch_of = data.get_batch_membership(0..n);
    let groups = data.take_grouped_columns().expect("groups").clone();
    assert_eq!(groups.len(), 2);

    let mut stat = CollapsedStat::new(TOY_GENES, 2, 2);
    data.collect_matched_stat(2, 3, None, &mut stat).unwrap();

    for (s, cells) in groups.iter().enumerate() {
        let own = batch_of[cells[0]];
        assert!(cells.iter().all(|&c| batch_of[c] == own));
        assert_eq!(stat.matched_bs[(own, s)], 0.0, "sample {s} own batch");
        let from_other = stat.matched_bs[(1 - own, s)];
        assert!(
            (from_other - cells.len() as f32).abs() < 1e-3,
            "sample {s}: {from_other} matched cells of the other batch, want {}",
            cells.len()
        );
    }
}
