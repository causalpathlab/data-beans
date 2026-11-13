use super::*;
use rand::SeedableRng;

fn toy_gene_sums(
    num_entities: usize,
    num_features: usize,
    labels: &[usize],
    rng_seed: u64,
) -> Vec<Vec<(usize, f32)>> {
    let mut rng = SmallRng::seed_from_u64(rng_seed);
    let num_blocks = *labels.iter().max().unwrap() + 1;
    let per_block = num_features / num_blocks;
    (0..num_entities)
        .map(|e| {
            let c = labels[e];
            let start = c * per_block;
            let end = ((c + 1) * per_block).min(num_features);
            let mut row: Vec<(usize, f32)> = (start..end)
                .map(|g| (g, 5.0 + rng.random_range(0.0..3.0_f32)))
                .collect();
            for _ in 0..3 {
                let g: usize = rng.random_range(0..num_features);
                row.push((g, rng.random_range(0.0..1.0_f32)));
            }
            row.sort_unstable_by_key(|&(g, _)| g);
            row.dedup_by_key(|&mut (g, _)| g);
            row
        })
        .collect()
}

fn make_profiles(gene_sums: &[Vec<(usize, f32)>], num_features: usize) -> Profiles {
    Profiles::from_gene_sums(gene_sums, num_features)
}

#[test]
fn test_log_probs_match_after_delta_moves() {
    let n = 24;
    let m = 16;
    let labels: Vec<usize> = (0..n).map(|i| i % 4).collect();
    let gs = toy_gene_sums(n, m, &labels, 1);
    let profiles = make_profiles(&gs, m);

    let mut stats = DcPoissonStats::from_profiles(&profiles, 4, &labels);
    let mut rng = SmallRng::seed_from_u64(7);
    for _ in 0..50 {
        let e: usize = rng.random_range(0..n);
        let to: usize = rng.random_range(0..4);
        let from = stats.membership[e];
        stats.delta_move(e, from, to, &profiles);
    }

    let kept_mem = stats.membership.clone();
    let mut fresh = DcPoissonStats::from_profiles(&profiles, 4, &kept_mem);
    fresh.recompute(&profiles);

    for i in 0..stats.gene_sum.len() {
        assert!((stats.gene_sum[i] - fresh.gene_sum[i]).abs() < 1e-6);
        assert!((stats.log_gene[i] - fresh.log_gene[i]).abs() < 1e-6);
    }
    for i in 0..stats.size_sum.len() {
        assert!((stats.size_sum[i] - fresh.size_sum[i]).abs() < 1e-6);
        assert!((stats.log_size_offset[i] - fresh.log_size_offset[i]).abs() < 1e-6);
    }
}

#[test]
fn test_restricted_matches_unrestricted_on_allowed() {
    let n = 20;
    let m = 12;
    let labels: Vec<usize> = (0..n).map(|i| i % 4).collect();
    let gs = toy_gene_sums(n, m, &labels, 2);
    let profiles = make_profiles(&gs, m);
    let stats = DcPoissonStats::from_profiles(&profiles, 4, &labels);

    let mut full = vec![f64::NEG_INFINITY; 4];
    let mut restricted = vec![f64::NEG_INFINITY; 4];
    compute_log_probs(3, &stats, &profiles, &mut full);

    let allowed = vec![1usize, 3usize];
    compute_log_probs_restricted(3, &stats, &profiles, &allowed, &mut restricted);
    for &k in &allowed {
        assert!((full[k] - restricted[k]).abs() < 1e-9);
    }
    for (k, &lp) in restricted.iter().enumerate() {
        if !allowed.contains(&k) {
            assert!(lp.is_infinite() && lp < 0.0);
        }
    }
}

#[test]
fn test_empty_block_finite() {
    let gs = vec![vec![(0usize, 3.0f32)], vec![(1usize, 4.0f32)]];
    let profiles = Profiles::from_gene_sums(&gs, 2);
    let stats = DcPoissonStats::from_profiles(&profiles, 3, &[1, 2]);
    assert!(stats.log_size_offset[0].is_finite());
    assert_eq!(stats.size_sum[0], 0.0);
}

#[test]
fn test_delta_move_drift_keeps_logs_finite() {
    // Large-magnitude masses make the incremental f64 accumulator's residues
    // exceed LOG_EPS when a block's mass is drained back to zero; without
    // clamping, `ln(negative)` poisons the log caches with NaN.
    let n = 64;
    let m = 8;
    let k = 4;
    let mut rng = SmallRng::seed_from_u64(11);
    // Wide magnitude spread maximizes rounding in the running sums: adding a
    // tiny value to a huge accumulator loses low bits, so draining a block
    // back to (true) zero leaves a signed residue.
    let gs: Vec<Vec<(usize, f32)>> = (0..n)
        .map(|e| {
            let scale = if e % 3 == 0 { 1.0e-3 } else { 1.0e7 };
            (0..m)
                .map(|g| (g, scale * (1.0 + rng.random_range(0.0..1.0_f32))))
                .collect()
        })
        .collect();
    let profiles = make_profiles(&gs, m);
    let labels: Vec<usize> = (0..n).map(|i| i % k).collect();
    let mut stats = DcPoissonStats::from_profiles(&profiles, k, &labels);
    let mut order: Vec<usize> = (0..n).collect();
    // Churn: repeatedly collapse every entity into one block and scatter
    // back, visiting entities in a fresh random order each round.
    for round in 0..100 {
        order.shuffle(&mut rng);
        for &e in &order {
            let from = stats.membership[e];
            let to = if round % 2 == 0 { round % k } else { e % k };
            stats.delta_move(e, from, to, &profiles);
        }
        for &s in &stats.gene_sum {
            assert!(s >= 0.0, "gene_sum went negative: {s}");
        }
        for &s in &stats.size_sum {
            assert!(s >= 0.0, "size_sum went negative: {s}");
        }
        for &l in &stats.log_gene {
            assert!(l.is_finite(), "log_gene not finite: {l}");
        }
        for &l in &stats.log_size_offset {
            assert!(l.is_finite(), "log_size_offset not finite: {l}");
        }
    }
}

#[test]
fn test_score_is_leave_one_out_for_current_block() {
    // The score of e's current block must equal the score of the same block
    // computed against stats built with e removed — otherwise staying put
    // earns a self-inclusion bonus over every candidate destination.
    let n = 12;
    let m = 10;
    let labels: Vec<usize> = (0..n).map(|i| i % 3).collect();
    let gs = toy_gene_sums(n, m, &labels, 5);
    let profiles = make_profiles(&gs, m);
    // No noise charge: it applies to destinations only, and the block is
    // the current one on one side and a destination on the other.
    let flat = RefineParams {
        min_gain_z: 0.0,
        ..Default::default()
    };
    let stats = DcPoissonStats::from_profiles_with(&profiles, 3, &labels, &flat);

    let e = 4;
    let k_cur = labels[e];
    let mut lp = vec![f64::NEG_INFINITY; 3];
    compute_log_probs_restricted(e, &stats, &profiles, &[0, 1, 2], &mut lp);

    // Rebuild stats with e parked in a scratch block so k_cur excludes it;
    // scoring e -> k_cur then goes through the ordinary cached path.
    let mut labels_wo = labels.clone();
    labels_wo[e] = 3;
    let stats_wo = DcPoissonStats::from_profiles_with(&profiles, 4, &labels_wo, &flat);
    let mut lp_wo = vec![f64::NEG_INFINITY; 4];
    compute_log_probs_restricted(e, &stats_wo, &profiles, &[k_cur], &mut lp_wo);

    assert!(
        (lp[k_cur] - lp_wo[k_cur]).abs() < 1e-3,
        "current-block score {} differs from leave-one-out score {}",
        lp[k_cur],
        lp_wo[k_cur]
    );
}

fn planted_partition_recovery(parallel: bool) {
    let n = 80;
    let m = 40;
    let k = 4;
    let planted: Vec<usize> = (0..n).map(|i| i % k).collect();
    let gs = toy_gene_sums(n, m, &planted, 9);
    let profiles = make_profiles(&gs, m);

    let mut labels = planted.clone();
    let mut rng = SmallRng::seed_from_u64(3);
    for (e, l) in labels.iter_mut().enumerate() {
        if e % 4 == 0 {
            *l = rng.random_range(0..k);
        }
    }
    let candidates: Vec<Vec<usize>> = vec![(0..k).collect(); n];
    let params = RefineParams {
        parallel,
        ..Default::default()
    };
    let ctx = RefineContext {
        profiles: &profiles,
        k,
        params: &params,
        level_label: "test",
    };
    let mut sweep_rng = SmallRng::seed_from_u64(4);
    refine_with_candidates(&mut labels, &candidates, &mut sweep_rng, &ctx);
    let acc = labels
        .iter()
        .zip(planted.iter())
        .filter(|(a, b)| a == b)
        .count();
    assert!(
        acc as f64 >= 0.95 * n as f64,
        "recovered only {acc}/{n} planted labels (parallel={parallel})"
    );
}

#[test]
fn test_refine_recovers_planted_partition_sequential() {
    planted_partition_recovery(false);
}

#[test]
fn test_refine_recovers_planted_partition_jacobi() {
    planted_partition_recovery(true);
}

#[test]
fn test_compact_labels() {
    let (c, k) = compact_labels(&[5, 5, 2, 7, 2, 7, 5]);
    assert_eq!(k, 3);
    assert_eq!(c, vec![0, 0, 1, 2, 1, 2, 0]);
}

#[test]
fn test_compute_sibling_sets_at_coarsest_gives_all() {
    let refined = vec![
        vec![0usize, 1, 2, 3], // finest
        vec![0usize, 0, 1, 1], // coarsest
    ];
    let top = refined.len() - 1;
    let sibs = compute_sibling_sets(&refined, top, 2);
    assert!(sibs.iter().all(|s| s == &vec![0usize, 1]));
}

#[test]
fn test_compute_sibling_sets_respects_parent() {
    let refined = vec![
        vec![0usize, 1, 2, 3], // finest: 4 groups
        vec![0usize, 0, 1, 1], // parent: {0,1}→0, {2,3}→1
    ];
    let sibs = compute_sibling_sets(&refined, 0, 4);
    assert_eq!(sibs[0], vec![0, 1]);
    assert_eq!(sibs[1], vec![0, 1]);
    assert_eq!(sibs[2], vec![2, 3]);
    assert_eq!(sibs[3], vec![2, 3]);
}

#[test]
fn test_one_pass_scores_match_per_block_scores_exactly() {
    // The one-pass kernel must reproduce the per-block reference bit for bit,
    // for the full block set (dense path), for unsorted subsets with and
    // without the entity's current block (gather path), after moves.
    let n = 40;
    let m = 30;
    let k = 6;
    let labels: Vec<usize> = (0..n).map(|i| i % k).collect();
    let gs = toy_gene_sums(n, m, &labels, 21);
    let mut profiles = make_profiles(&gs, m);
    profiles.apply_feature_weighting(FeatureWeighting::FisherInfoNb);
    let mut stats = DcPoissonStats::from_profiles(&profiles, k, &labels);
    let mut rng = SmallRng::seed_from_u64(5);
    for _ in 0..30 {
        let e: usize = rng.random_range(0..n);
        let to: usize = rng.random_range(0..k);
        let from = stats.membership[e];
        stats.delta_move(e, from, to, &profiles);
    }
    let subsets: Vec<Vec<usize>> = vec![
        (0..k).collect(),
        (0..k).rev().collect(),
        vec![4, 1],
        vec![5, 0, 2],
        vec![3],
    ];
    for e in 0..n {
        let mut reference = vec![f64::NAN; k];
        compute_log_probs(e, &stats, &profiles, &mut reference);
        for allowed in &subsets {
            let mut fast = vec![f64::NEG_INFINITY; k];
            compute_log_probs_restricted(e, &stats, &profiles, allowed, &mut fast);
            for &b in allowed {
                assert_eq!(
                    fast[b].to_bits(),
                    reference[b].to_bits(),
                    "entity {e}, block {b}, allowed {allowed:?}"
                );
            }
        }
    }
}

#[test]
fn test_stats_layout_is_feature_major() {
    let gs = vec![vec![(0usize, 1.0f32), (2, 3.0)], vec![(1usize, 4.0f32)]];
    let profiles = Profiles::from_gene_sums(&gs, 3);
    let stats = DcPoissonStats::from_profiles(&profiles, 2, &[0, 1]);
    assert_eq!(stats.idx(1, 2), 2 * 2 + 1);
    assert_eq!(stats.gene_sum, vec![1.0, 0.0, 0.0, 4.0, 3.0, 0.0]);
}

#[test]
fn test_repeated_candidates_score_like_single_ones() {
    // A proposer may list a block twice; each listing must get the block's
    // own score (the per-block reference), never a double-counted one.
    let n = 30;
    let m = 20;
    let k = 5;
    let labels: Vec<usize> = (0..n).map(|i| i % k).collect();
    let gs = toy_gene_sums(n, m, &labels, 33);
    let profiles = make_profiles(&gs, m);
    let stats = DcPoissonStats::from_profiles(&profiles, k, &labels);
    let subsets: Vec<Vec<usize>> = vec![
        vec![2, 2],
        vec![1, 3, 1],
        vec![0, 4, 0, 4, 2],
        // as long as K, with a repeat (dense path, block 4 missing)
        vec![0, 1, 1, 2, 3],
        // longer than K
        vec![0, 1, 2, 3, 4, 3, 0],
    ];
    for e in 0..n {
        let mut reference = vec![f64::NAN; k];
        compute_log_probs(e, &stats, &profiles, &mut reference);
        let cur = stats.membership[e];
        for base in &subsets {
            for allowed in [base.clone(), {
                let mut a = base.clone();
                a.push(cur);
                a.push(cur);
                a
            }] {
                let mut fast = vec![f64::NEG_INFINITY; k];
                compute_log_probs_restricted(e, &stats, &profiles, &allowed, &mut fast);
                for &b in &allowed {
                    assert_eq!(
                        fast[b].to_bits(),
                        reference[b].to_bits(),
                        "entity {e}, block {b}, allowed {allowed:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn fast_ln_is_ln_to_f32_rounding() {
    // f32 rounding of the input and of the result: below 4e-6 for |ln x| < 30.
    let mut x = LOG_EPS;
    while x < 1e12 {
        let err = (fast_ln(x) - x.ln()).abs();
        assert!(err < 4e-6, "x = {x}: {} vs {}", fast_ln(x), x.ln());
        x *= 1.0137;
    }
    // Monotone and exact near 1, where -ln(u) is tiny.
    let mut prev = f64::NEG_INFINITY;
    for i in (0..1000).rev() {
        let u = 1.0 - i as f64 * 1e-7;
        let v = fast_ln(u);
        assert!(v >= prev && (v - u.ln()).abs() < 1e-7, "u = {u}");
        prev = v;
    }
    assert!(fast_ln(f64::NAN).is_nan());
    assert!(fast_ln(-1.0).is_nan());
    assert_eq!(fast_ln(0.0), f64::NEG_INFINITY);
    assert_eq!(fast_ln(f64::INFINITY), f64::INFINITY);
}

/// An entity with an empty profile scores every block alike; a greedy sweep
/// leaves it where it is instead of moving it to its first candidate.
fn greedy_keeps_an_entity_without_evidence(parallel: bool) {
    let (n, m, k) = (24, 16, 4);
    let planted: Vec<usize> = (0..n).map(|i| i % k).collect();
    let mut gs = toy_gene_sums(n, m, &planted, 5);
    let blank = 7;
    gs[blank].clear();
    let profiles = make_profiles(&gs, m);
    assert_eq!(profiles.size_factor[blank], 0.0);

    let mut labels = planted.clone();
    assert_eq!(labels[blank], 3);
    let candidates: Vec<Vec<usize>> = vec![vec![1, 3]; n];
    let params = RefineParams {
        num_gibbs: 0,
        num_greedy: 3,
        parallel,
        ..Default::default()
    };
    let ctx = RefineContext {
        profiles: &profiles,
        k,
        params: &params,
        level_label: "test",
    };
    let mut rng = SmallRng::seed_from_u64(1);
    refine_with_candidates(&mut labels, &candidates, &mut rng, &ctx);
    assert_eq!(labels[blank], 3, "parallel={parallel}");
}

#[test]
fn greedy_keeps_an_entity_without_evidence_sequential() {
    greedy_keeps_an_entity_without_evidence(false);
}

#[test]
fn greedy_keeps_an_entity_without_evidence_jacobi() {
    greedy_keeps_an_entity_without_evidence(true);
}

/// The Fisher weights of (non-integer) profiles come out the same bits on
/// any number of threads.
#[test]
fn nb_fisher_weights_do_not_depend_on_the_thread_count() {
    let (n, m, k) = (3000, 24, 4);
    let labels: Vec<usize> = (0..n).map(|i| i % k).collect();
    let profiles = make_profiles(&toy_gene_sums(n, m, &labels, 11), m);
    let on_threads = |threads: usize| -> Vec<u32> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool")
            .install(|| profiles.nb_fisher_weights())
            .iter()
            .map(|w| w.to_bits())
            .collect()
    };
    let one = on_threads(1);
    assert_eq!(one.len(), m);
    for threads in [3, 6] {
        assert_eq!(on_threads(threads), one, "{threads} threads");
    }
}

/////////////////////////////////////////////
// Block sizes: ties, floors, small blocks //
/////////////////////////////////////////////

use rand_distr::{Distribution, LogNormal, Poisson};

/// Feature count of the simulated profiles below.
const SIM_M: usize = 1000;

/// Lognormal(0, 1) rates on features `lo..hi`, normalised to one.
fn sim_profile(rng: &mut SmallRng, lo: usize, hi: usize) -> Vec<f64> {
    let ln = LogNormal::new(0.0, 1.0).unwrap();
    let mut p = vec![0f64; SIM_M];
    for x in p[lo..hi].iter_mut() {
        *x = ln.sample(rng);
    }
    let s: f64 = p.iter().sum();
    p.iter_mut().for_each(|x| *x /= s);
    p
}

/// `p` with a tenth of its features doubled or halved, renormalised.
fn sim_perturbed(rng: &mut SmallRng, p: &[f64]) -> Vec<f64> {
    let mut q = p.to_vec();
    for x in q.iter_mut() {
        if rng.random::<f64>() < 0.1 {
            *x *= if rng.random::<f64>() < 0.5 { 2.0 } else { 0.5 };
        }
    }
    let s: f64 = q.iter().sum();
    q.iter_mut().for_each(|x| *x /= s);
    q
}

/// One entity of `depth` expected counts drawn from profile `p`.
fn sim_entity(rng: &mut SmallRng, p: &[f64], depth: f64) -> Vec<(usize, f32)> {
    let mut row = Vec::new();
    for (g, &pg) in p.iter().enumerate() {
        if pg > 0.0 {
            let x: f64 = Poisson::new(depth * pg).unwrap().sample(rng);
            if x > 0.0 {
                row.push((g, x as f32));
            }
        }
    }
    row
}

/// Refine `labels` with every block a candidate of every entity; returns
/// the number of moves.
fn sim_refine(
    rows: &[Vec<(usize, f32)>],
    labels: &mut [usize],
    k: usize,
    params: &RefineParams,
) -> usize {
    let mut profiles = Profiles::from_gene_sums(rows, SIM_M);
    profiles.apply_feature_weighting(params.feature_weighting);
    let candidates: Vec<Vec<usize>> = vec![(0..k).collect(); labels.len()];
    let ctx = RefineContext {
        profiles: &profiles,
        k,
        params,
        level_label: "test",
    };
    let mut rng = SmallRng::seed_from_u64(params.seed);
    refine_with_candidates(labels, &candidates, &mut rng, &ctx)
}

fn block_sizes(labels: &[usize], k: usize) -> Vec<usize> {
    let mut n = vec![0usize; k];
    labels.iter().for_each(|&l| n[l] += 1);
    n
}

/// Sweep settings of the two callers, Jacobi and sequential.
fn sim_param_sets() -> Vec<RefineParams> {
    let mut out = Vec::new();
    for parallel in [true, false] {
        for (num_gibbs, num_greedy) in [(3, 10), (20, 10), (0, 10)] {
            out.push(RefineParams {
                num_gibbs,
                num_greedy,
                parallel,
                ..Default::default()
            });
        }
    }
    out
}

#[test]
fn equal_scores_keep_the_current_block() {
    let lp = [1.0, 1.0, 1.0, 0.5];
    for current in 0..3 {
        assert_eq!(argmax_log_staying(&lp, &[0, 1, 2, 3], current), current);
        assert_eq!(argmax_log_staying(&lp, &[3, 2, 1, 0], current), current);
    }
    assert_eq!(argmax_log_staying(&lp, &[0, 1, 2, 3], 3), 0);
    // a score that does not compare is no evidence either
    assert_eq!(argmax_log_staying(&[f64::NAN, 2.0], &[0, 1], 0), 0);
    assert_eq!(argmax_log_staying(&[1.0, f64::NAN], &[0, 1], 0), 0);

    // Draws never pick a block that does not beat the current one.
    let mut rng = SmallRng::seed_from_u64(1);
    for _ in 0..200 {
        assert_eq!(sample_improving_log(&lp, &[0, 1, 2, 3], 1, &mut rng), 1);
    }
    let stays = (0..2000)
        .filter(|_| sample_improving_log(&[0.0, 0.1], &[0, 1], 0, &mut rng) == 0)
        .count();
    assert!(
        (800..1100).contains(&stays),
        "a slightly better block is drawn in proportion: {stays}"
    );
}

#[test]
fn entities_without_evidence_do_not_move() {
    // Four entities, each alone in its block and on features nothing else
    // shows, next to a large block and four empty blocks of lower index.
    // With the floor lifted nothing but the scores holds them: leaving one
    // out, its block is empty too, an exact tie with every empty block —
    // which must not pull the four into one of them. A fifth entity has no
    // counts at all.
    let mut rng = SmallRng::seed_from_u64(2);
    let p = sim_profile(&mut rng, 0, 800);
    let mut rows = Vec::new();
    let mut start = Vec::new();
    for i in 0..4 {
        let q = sim_profile(&mut rng, 800 + 50 * i, 850 + 50 * i);
        rows.push(sim_entity(&mut rng, &q, 300.0));
        start.push(5 + i);
    }
    rows.push(Vec::new());
    start.push(4);
    for _ in 0..40 {
        rows.push(sim_entity(&mut rng, &p, 300.0));
        start.push(4);
    }
    for mut params in sim_param_sets() {
        params.size_floor = 0.0;
        params.size_cap = 0.0;
        let mut labels = start.clone();
        sim_refine(&rows, &mut labels, 9, &params);
        assert_eq!(
            &labels[..5],
            &start[..5],
            "parallel={} gibbs={}",
            params.parallel,
            params.num_gibbs
        );
    }
}

#[test]
fn singleton_blocks_keep_their_member() {
    // Three large blocks and nine one-member blocks: three drawn from the
    // large blocks' own profiles, three from fresh profiles, three on
    // features nothing else shows.
    let mut rng = SmallRng::seed_from_u64(3);
    let big: Vec<Vec<f64>> = (0..3).map(|_| sim_profile(&mut rng, 0, 850)).collect();
    let mut rows = Vec::new();
    let mut start = Vec::new();
    for (b, p) in big.iter().enumerate() {
        for _ in 0..60 {
            rows.push(sim_entity(&mut rng, p, 500.0));
            start.push(b);
        }
    }
    let mut singles: Vec<Vec<f64>> = big.clone();
    for _ in 0..3 {
        singles.push(sim_profile(&mut rng, 0, 850));
    }
    for i in 0..3 {
        singles.push(sim_profile(&mut rng, 850 + 50 * i, 900 + 50 * i));
    }
    let k = 3 + singles.len();
    for (i, p) in singles.iter().enumerate() {
        rows.push(sim_entity(&mut rng, p, 500.0));
        start.push(3 + i);
    }
    for params in sim_param_sets() {
        let mut labels = start.clone();
        sim_refine(&rows, &mut labels, k, &params);
        assert_eq!(
            &labels[180..],
            &start[180..],
            "singletons moved (parallel={})",
            params.parallel
        );
        assert!(block_sizes(&labels, k).iter().all(|&n| n > 0));
        // A fixed point: one more greedy sweep moves nothing.
        let again = RefineParams {
            num_gibbs: 0,
            num_greedy: 1,
            ..params.clone()
        };
        assert_eq!(sim_refine(&rows, &mut labels, k, &again), 0);
    }
}

/// 30 entities of profile A in block 0 against 300 in block 1, of profile A
/// again (`same`) or of A with a tenth of the features doubled or halved.
fn small_against_large(same: bool, seed: u64) -> (Vec<Vec<(usize, f32)>>, Vec<usize>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let pa = sim_profile(&mut rng, 0, SIM_M);
    let pb = if same {
        pa.clone()
    } else {
        sim_perturbed(&mut rng, &pa)
    };
    let mut rows = Vec::new();
    let mut labels = Vec::new();
    for e in 0..330 {
        rows.push(sim_entity(&mut rng, if e < 30 { &pa } else { &pb }, 500.0));
        labels.push(usize::from(e >= 30));
    }
    (rows, labels)
}

#[test]
fn small_distinct_block_keeps_its_members() {
    // The score, not the size limits, has to hold the small block: with
    // the limits lifted its members still stay, and the large block's
    // members do not flood it.
    for limits in [true, false] {
        let (mut kept, mut joined) = (0usize, 0usize);
        for seed in 0..4 {
            let (rows, start) = small_against_large(false, 10 + seed);
            let mut params = RefineParams {
                seed,
                ..Default::default()
            };
            if !limits {
                params.size_floor = 0.0;
                params.size_cap = 0.0;
            }
            let mut labels = start.clone();
            sim_refine(&rows, &mut labels, 2, &params);
            kept += labels[..30].iter().filter(|&&l| l == 0).count();
            joined += labels[30..].iter().filter(|&&l| l == 0).count();
        }
        assert!(kept >= 4 * 27, "limits={limits}: kept {kept} of 120");
        assert!(joined <= 4 * 3, "limits={limits}: {joined} joined");
    }
}

#[test]
fn small_block_of_the_same_profile_is_not_drained() {
    // Same profile, a tenth of the size. Nothing tells the blocks apart, so
    // members may go either way, but the small block must not empty — with
    // the size limits lifted too.
    for limits in [true, false] {
        for seed in 0..4 {
            let (rows, start) = small_against_large(true, 20 + seed);
            let mut params = RefineParams {
                seed,
                ..Default::default()
            };
            if !limits {
                params.size_floor = 0.0;
                params.size_cap = 0.0;
            }
            let mut labels = start.clone();
            sim_refine(&rows, &mut labels, 2, &params);
            let sizes = block_sizes(&labels, 2);
            let floor = if limits { 27 } else { 15 };
            assert!(
                sizes[0] >= floor,
                "limits={limits} seed={seed}: small block left with {}",
                sizes[0]
            );
        }
    }
}

#[test]
fn mislabelled_entities_are_corrected() {
    // Eight blocks of sixty with distinct profiles; a fifth of the entities
    // start in a wrong block.
    let k = 8;
    let mut rng = SmallRng::seed_from_u64(30);
    let mut rows = Vec::new();
    let mut truth = Vec::new();
    let mut start = Vec::new();
    for b in 0..k {
        let p = sim_profile(&mut rng, 0, SIM_M);
        for _ in 0..60 {
            rows.push(sim_entity(&mut rng, &p, 500.0));
            truth.push(b);
            start.push(if rng.random::<f64>() < 0.2 {
                (b + 1 + rng.random_range(0..k - 1)) % k
            } else {
                b
            });
        }
    }
    let wrong = start.iter().zip(&truth).filter(|(a, b)| a != b).count();
    assert!(wrong >= 60, "fixture should start mislabelled ({wrong})");
    for params in sim_param_sets() {
        let mut labels = start.clone();
        sim_refine(&rows, &mut labels, k, &params);
        let right = labels.iter().zip(&truth).filter(|(a, b)| a == b).count();
        assert!(
            right >= truth.len() - 2,
            "{right} of {} in their block (parallel={}, gibbs={})",
            truth.len(),
            params.parallel,
            params.num_gibbs
        );
    }
}

/// Twelve blocks over three profiles, sizes 6 to 48: every block starts
/// with two-thirds of its members of one profile and the rest of the two
/// others.
struct MixedBlocks {
    rows: Vec<Vec<(usize, f32)>>,
    /// Profile of each entity.
    truth: Vec<usize>,
    /// Starting block of each entity.
    start: Vec<usize>,
    /// Number of blocks.
    k: usize,
    /// Number of profiles.
    t: usize,
}

fn mixed_blocks_fixture() -> MixedBlocks {
    let (k, t) = (12, 3);
    let mut rng = SmallRng::seed_from_u64(40);
    let profs: Vec<Vec<f64>> = (0..t).map(|_| sim_profile(&mut rng, 0, SIM_M)).collect();
    let mut rows = Vec::new();
    let mut truth = Vec::new();
    let mut start = Vec::new();
    for b in 0..k {
        let n = [6, 12, 24, 48][b % 4];
        for i in 0..n {
            let ty = if i % 3 == 2 {
                (b + 1 + i % 2) % t
            } else {
                b % t
            };
            rows.push(sim_entity(&mut rng, &profs[ty], 1000.0));
            truth.push(ty);
            start.push(b);
        }
    }
    MixedBlocks {
        rows,
        truth,
        start,
        k,
        t,
    }
}

#[test]
fn mixed_blocks_are_purified_within_the_size_limits() {
    // Twelve blocks of unequal size over three profiles, each two-thirds
    // one profile. The members that do not belong move out; no block falls
    // under its floor, exceeds its cap, or empties, and the members a
    // block is named after stay.
    let MixedBlocks {
        rows,
        truth,
        start,
        k,
        t,
    } = mixed_blocks_fixture();
    let purity = |labels: &[usize]| {
        let mut c = vec![0usize; k * t];
        for (&l, &ty) in labels.iter().zip(&truth) {
            c[l * t + ty] += 1;
        }
        let dom: usize = c.chunks(t).map(|b| *b.iter().max().unwrap()).sum();
        dom as f64 / labels.len() as f64
    };
    let n0 = block_sizes(&start, k);
    // 270 members in 12 blocks: a mean of 22.5. Nine tenths of
    // min(n0, mean) and five quarters of max(n0, mean), rounded up.
    assert_eq!(start.len(), 270);
    let limit_of = |n: usize, limits: [usize; 4]| {
        limits[[6, 12, 24, 48].iter().position(|&x| x == n).unwrap()]
    };
    let floor = |n: usize| limit_of(n, [6, 11, 21, 21]);
    let cap = |n: usize| limit_of(n, [29, 29, 30, 60]);
    for params in sim_param_sets() {
        let mut labels = start.clone();
        sim_refine(&rows, &mut labels, k, &params);
        let n1 = block_sizes(&labels, k);
        for b in 0..k {
            assert!(
                (floor(n0[b])..=cap(n0[b])).contains(&n1[b]),
                "block {b}: {} -> {} outside {}..={}",
                n0[b],
                n1[b],
                floor(n0[b]),
                cap(n0[b])
            );
        }
        assert!(
            purity(&labels) >= purity(&start) + 0.1,
            "purity {} -> {} (parallel={})",
            purity(&start),
            purity(&labels),
            params.parallel
        );
        let named = (0..start.len()).filter(|&e| truth[e] == start[e] % t);
        let (mut all, mut stayed) = (0, 0);
        for e in named {
            all += 1;
            stayed += usize::from(labels[e] == start[e]);
        }
        // A Jacobi sweep applies its moves by gain, so the members that fit
        // worst leave first; a sequential sweep takes them as it visits,
        // and more of a block's own members leave for a purer block of
        // their profile.
        let share = if params.parallel { 0.9 } else { 0.6 };
        assert!(
            stayed as f64 >= share * all as f64,
            "{stayed} of {all} stayed (parallel={})",
            params.parallel
        );
    }
}

#[test]
fn lifted_floor_lets_a_mixed_block_shed_one_profile() {
    // Without a floor the likelihood alone decides: a block that is half
    // one profile and half another gives the members of one of them to
    // that profile's pure block. What is left is a small block of the
    // other profile, which nothing tells apart from the large pure block
    // of that profile, so it stays.
    let mut rng = SmallRng::seed_from_u64(50);
    let pa = sim_profile(&mut rng, 0, SIM_M);
    let pb = sim_profile(&mut rng, 0, SIM_M);
    let mut rows = Vec::new();
    let mut start = Vec::new();
    for b in 0..3 {
        for i in 0..40 {
            let p = match b {
                0 => &pa,
                1 => &pb,
                _ if i % 2 == 0 => &pa,
                _ => &pb,
            };
            rows.push(sim_entity(&mut rng, p, 500.0));
            start.push(b);
        }
    }
    let free = RefineParams {
        size_floor: 0.0,
        size_cap: 0.0,
        ..Default::default()
    };
    let mut labels = start.clone();
    sim_refine(&rows, &mut labels, 3, &free);
    let left: Vec<usize> = (80..120).filter(|&e| labels[e] == 2).collect();
    assert!(left.len() <= 20, "{} of 40 left", left.len());
    assert!(
        left.iter().all(|e| e % 2 == 0) || left.iter().all(|e| e % 2 == 1),
        "what is left is of one profile"
    );
    assert_eq!(
        labels[..80],
        start[..80],
        "the pure blocks keep their members"
    );

    let mut labels = start.clone();
    sim_refine(&rows, &mut labels, 3, &RefineParams::default());
    assert!(block_sizes(&labels, 3)[2] >= 36, "the default floor holds");
}

#[test]
fn log_caches_follow_delta_moves_with_counts() {
    let n = 30;
    let m = 12;
    let labels: Vec<usize> = (0..n).map(|i| i % 3).collect();
    let gs = toy_gene_sums(n, m, &labels, 8);
    let profiles = make_profiles(&gs, m);
    let mut stats = DcPoissonStats::from_profiles(&profiles, 3, &labels);
    let mut rng = SmallRng::seed_from_u64(9);
    for _ in 0..40 {
        let e: usize = rng.random_range(0..n);
        let from = stats.membership[e];
        stats.delta_move(e, from, rng.random_range(0..3), &profiles);
    }
    let fresh = DcPoissonStats::from_profiles(&profiles, 3, &stats.membership);
    assert_eq!(stats.count, fresh.count);
    assert_eq!(stats.count.iter().sum::<usize>(), n);
    for (moved, built) in [
        (&stats.class_null, &fresh.class_null),
        (&stats.class_var, &fresh.class_var),
    ] {
        assert_eq!(moved.len(), built.len());
        for (a, b) in moved.iter().zip(built) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
    }
}

#[test]
fn empty_block_scores_as_the_pooled_profile() {
    // An empty block and the leave-one-out view of a one-member block are
    // the same thing, the pooled profile, and score alike.
    let mut rng = SmallRng::seed_from_u64(60);
    let p = sim_profile(&mut rng, 0, SIM_M);
    let rows: Vec<Vec<(usize, f32)>> = (0..20).map(|_| sim_entity(&mut rng, &p, 400.0)).collect();
    let profiles = Profiles::from_gene_sums(&rows, SIM_M);
    // entity 0 alone in block 1; block 2 empty
    let labels: Vec<usize> = (0..20).map(|e| usize::from(e == 0)).collect();
    let flat = RefineParams {
        min_gain_z: 0.0,
        ..Default::default()
    };
    let stats = DcPoissonStats::from_profiles_with(&profiles, 3, &labels, &flat);
    let mut lp = vec![0f64; 3];
    compute_log_probs_restricted(0, &stats, &profiles, &[0, 1, 2], &mut lp);
    assert!(lp.iter().all(|x| x.is_finite()));
    assert!(
        (lp[1] - lp[2]).abs() < 1e-6 * lp[1].abs(),
        "own block {} vs empty block {}",
        lp[1],
        lp[2]
    );
    // -ln M per count would be the uniform profile; the pooled one is better.
    let per_count = lp[2] / profiles.size_factor[0] as f64;
    assert!(per_count > -(SIM_M as f64).ln());
}

#[test]
fn repeated_calls_do_not_empty_a_block() {
    // The limits are those of each call's input, so a block may shrink a
    // step per call: a tenth of its members while it has ten or more, and
    // not at all below ten. Called again and again on its own output, no
    // block of the mixed partition falls under min(its start, nine).
    let MixedBlocks { rows, start, k, .. } = mixed_blocks_fixture();
    let n0 = block_sizes(&start, k);
    for parallel in [true, false] {
        let params = RefineParams {
            parallel,
            ..Default::default()
        };
        let mut labels = start.clone();
        for call in 0..12 {
            sim_refine(&rows, &mut labels, k, &params);
            let n = block_sizes(&labels, k);
            for b in 0..k {
                assert!(
                    n[b] >= n0[b].min(9),
                    "call {call}, block {b}: {} -> {} (parallel={parallel})",
                    n0[b],
                    n[b]
                );
            }
        }
    }
}

#[test]
fn size_limits_of_small_and_empty_blocks() {
    let limits = |count: &[usize], size_floor: f32, size_cap: f32| {
        let l = SizeLimits::new(
            count,
            &RefineParams {
                size_floor,
                size_cap,
                ..Default::default()
            },
        );
        (l.floor, l.cap)
    };
    // Four occupied blocks of 120 members: a mean of 30.
    let count = [0, 1, 9, 10, 100];
    let (floor, cap) = limits(&count, 0.9, 1.5);
    // An empty block and a block of one have a floor of one; nine members
    // stay nine and ten may fall to nine; the large block may fall to nine
    // tenths of the mean.
    assert_eq!(floor, vec![1, 1, 9, 9, 27]);
    assert_eq!(cap, vec![45, 45, 45, 45, 150]);
    let l = SizeLimits::new(
        &count,
        &RefineParams {
            size_floor: 0.9,
            size_cap: 1.5,
            ..Default::default()
        },
    );
    assert!(!l.can_leave(&count, 0) && !l.can_leave(&count, 1));
    assert!(!l.can_leave(&count, 2) && l.can_leave(&count, 3));
    assert!(l.admits(&count, 4, 0), "an empty block takes members");
    assert!(l.admits(&count, 3, 2), "a block at its floor takes members");

    // Zero switches a limit off.
    let (floor, cap) = limits(&count, 0.0, 0.0);
    assert_eq!(floor, vec![0; 5]);
    assert_eq!(cap, vec![usize::MAX; 5]);

    // A decimal fraction gives the limit it gives on paper: eight tenths of
    // ten is eight, both ways.
    let (floor, cap) = limits(&[10, 10], 0.8, 0.8);
    assert_eq!((floor, cap), (vec![8, 8], vec![8, 8]));
    let (floor, cap) = limits(&[10, 10, 10], 0.7, 1.1);
    assert_eq!((floor, cap), (vec![7; 3], vec![11; 3]));
    // A mean that is no whole number: 7 members in 2 blocks.
    let (floor, cap) = limits(&[3, 4], 1.0, 1.0);
    assert_eq!((floor, cap), (vec![3, 4], vec![4, 4]));
    // No block at all.
    let (floor, cap) = limits(&[], 0.9, 1.5);
    assert!(floor.is_empty() && cap.is_empty());
}

/// Blocks 0 (ten of profile A), 1 (ten of B) and 2 (ten of C and twenty of
/// A, which belong in block 0). The mean size is 50 / 3.
fn crowded_block_fixture() -> (Vec<Vec<(usize, f32)>>, Vec<usize>) {
    let mut rng = SmallRng::seed_from_u64(80);
    let profs: Vec<Vec<f64>> = (0..3).map(|_| sim_profile(&mut rng, 0, SIM_M)).collect();
    let mut rows = Vec::new();
    let mut start = Vec::new();
    for (block, ty, n) in [(0, 0, 10), (1, 1, 10), (2, 2, 10), (2, 0, 20)] {
        for _ in 0..n {
            rows.push(sim_entity(&mut rng, &profs[ty], 800.0));
            start.push(block);
        }
    }
    (rows, start)
}

#[test]
fn the_cap_and_the_floor_bind() {
    let (rows, start) = crowded_block_fixture();
    for parallel in [true, false] {
        let run = |size_floor: f32, size_cap: f32| {
            let params = RefineParams {
                size_floor,
                size_cap,
                parallel,
                ..Default::default()
            };
            let mut labels = start.clone();
            sim_refine(&rows, &mut labels, 3, &params);
            block_sizes(&labels, 3)
        };
        // Cap of block 0: the mean, rounded up.
        assert_eq!(run(0.9, 1.0), vec![17, 10, 23], "parallel={parallel}");
        // No cap: block 2 gives members down to nine tenths of the mean.
        assert_eq!(run(0.9, 0.0), vec![25, 10, 15], "parallel={parallel}");
        // No limits: every member of profile A joins block 0.
        assert_eq!(run(0.0, 0.0), vec![30, 10, 10], "parallel={parallel}");
    }
}

/// Vetoes every move of an entity with an even index.
struct EvenStay;

impl MoveGuard for EvenStay {
    fn accept_move(&self, entity: usize, _from: usize, _labels: &[usize]) -> bool {
        entity % 2 == 1
    }
}

#[test]
fn a_vetoed_move_changes_no_count() {
    // The limits are read off the member counts, so a move the guard
    // refuses must leave them as they were: afterwards the counts are those
    // of the labels, the vetoed entities are where they started, and the
    // cap still holds.
    let (rows, start) = crowded_block_fixture();
    let mut profiles = Profiles::from_gene_sums(&rows, SIM_M);
    profiles.apply_feature_weighting(FeatureWeighting::FisherInfoNb);
    let candidates: Vec<Vec<usize>> = vec![(0..3).collect(); start.len()];
    for parallel in [true, false] {
        let params = RefineParams {
            size_cap: 1.0,
            parallel,
            ..Default::default()
        };
        let ctx = RefineContext {
            profiles: &profiles,
            k: 3,
            params: &params,
            level_label: "test",
        };
        let mut labels = start.clone();
        let mut rng = SmallRng::seed_from_u64(1);
        let moves =
            refine_with_candidates_guarded(&mut labels, &candidates, &EvenStay, &mut rng, &ctx);
        for e in (0..labels.len()).step_by(2) {
            assert_eq!(labels[e], start[e], "entity {e} moved past the guard");
        }
        let moved = labels.iter().zip(&start).filter(|(a, b)| a != b).count();
        assert!(moved > 0 && moves >= moved, "parallel={parallel}");
        // Seven places under the cap of block 0, ten odd entities that want
        // them: the vetoed ones did not use any up.
        assert_eq!(
            block_sizes(&labels, 3),
            vec![17, 10, 23],
            "parallel={parallel}"
        );
    }
}

/// Vetoes every move of one entity.
struct Pinned(usize);

impl MoveGuard for Pinned {
    fn accept_move(&self, entity: usize, _from: usize, _labels: &[usize]) -> bool {
        entity != self.0
    }
}

#[test]
fn small_blocks_exchange_their_misfits() {
    // Eight entities in two blocks of four, `n_a` of profile A and the
    // rest of profile B. Nine tenths of four, rounded up, is four: neither
    // block may give a member.
    let refine = |n_a: usize, start: &[usize], params: &RefineParams, pinned: Option<usize>| {
        let mut rng = SmallRng::seed_from_u64(85);
        let pa = sim_profile(&mut rng, 0, SIM_M);
        let pb = sim_profile(&mut rng, 0, SIM_M);
        let rows: Vec<Vec<(usize, f32)>> = (0..8)
            .map(|e| sim_entity(&mut rng, if e < n_a { &pa } else { &pb }, 500.0))
            .collect();
        let profiles = Profiles::from_gene_sums(&rows, SIM_M);
        let candidates: Vec<Vec<usize>> = vec![vec![0, 1]; 8];
        let ctx = RefineContext {
            profiles: &profiles,
            k: 2,
            params,
            level_label: "test",
        };
        let mut labels = start.to_vec();
        let mut rng = SmallRng::seed_from_u64(1);
        let moves = match pinned {
            Some(e) => {
                refine_with_candidates_guarded(&mut labels, &candidates, &Pinned(e), &mut rng, &ctx)
            }
            None => refine_with_candidates(&mut labels, &candidates, &mut rng, &ctx),
        };
        (labels, moves)
    };
    for parallel in [true, false] {
        let params = RefineParams {
            feature_weighting: FeatureWeighting::None,
            parallel,
            ..Default::default()
        };
        // One member of each block belongs in the other: the two change
        // places, and the sizes stay.
        let swapped = [0, 0, 0, 1, 0, 1, 1, 1];
        assert_eq!(
            refine(4, &swapped, &params, None),
            (vec![0, 0, 0, 0, 1, 1, 1, 1], 2),
            "parallel={parallel}"
        );
        // With either of the two held by the guard, neither moves.
        for held in [3, 4] {
            assert_eq!(
                refine(4, &swapped, &params, Some(held)).0,
                swapped,
                "parallel={parallel}, entity {held} held"
            );
        }
        // Three of profile A and one of B against four of B: the misfit
        // has no counterpart and stays, since its block would shrink.
        let lone = [0, 0, 0, 0, 1, 1, 1, 1];
        assert_eq!(
            refine(3, &lone, &params, None),
            (lone.to_vec(), 0),
            "parallel={parallel}"
        );
        // Without the floor it joins its own.
        let free = RefineParams {
            size_floor: 0.0,
            ..params.clone()
        };
        assert_eq!(
            refine(3, &lone, &free, None).0,
            vec![0, 0, 0, 1, 1, 1, 1, 1],
            "parallel={parallel}"
        );
    }
}

#[test]
fn a_jacobi_sweep_rescores_moves_out_of_a_changed_block() {
    // Block 2 is half profile A, half profile B, next to a pure block of
    // each. Against the snapshot every member of block 2 gains by leaving.
    // Applied in turn, the moves of one half leave a block of the other
    // half only, which the rest then has no reason to leave: one sweep
    // moves all of one half and keeps the block.
    let mut rng = SmallRng::seed_from_u64(90);
    let pa = sim_profile(&mut rng, 0, SIM_M);
    let pb = sim_profile(&mut rng, 0, SIM_M);
    let mut rows = Vec::new();
    let mut start = Vec::new();
    for (block, p, n) in [(0, &pa, 30), (1, &pb, 30), (2, &pa, 6), (2, &pb, 6)] {
        for _ in 0..n {
            rows.push(sim_entity(&mut rng, p, 500.0));
            start.push(block);
        }
    }
    let params = RefineParams {
        num_gibbs: 0,
        num_greedy: 1,
        size_floor: 0.0,
        size_cap: 0.0,
        parallel: true,
        ..Default::default()
    };
    // Every member of block 2 proposes to leave.
    let profiles = Profiles::from_gene_sums(&rows, SIM_M);
    let stats = DcPoissonStats::from_profiles_with(&profiles, 3, &start, &params);
    let mut lp = vec![0f64; 3];
    for e in 60..72 {
        compute_log_probs_restricted(e, &stats, &profiles, &[0, 1, 2], &mut lp);
        assert_ne!(argmax_log_staying(&lp, &[0, 1, 2], 2), 2, "entity {e}");
    }
    let mut labels = start.clone();
    let moves = sim_refine(&rows, &mut labels, 3, &params);
    let left: Vec<usize> = (60..72).filter(|&e| labels[e] == 2).collect();
    assert_eq!(moves, 12 - left.len());
    assert!(left.len() >= 2, "part of one half stays: {left:?}");
    assert!(
        left.iter().all(|&e| e < 66) || left.iter().all(|&e| e >= 66),
        "what stays is of one profile: {left:?}"
    );
}

#[test]
fn zero_prior_count_scores_stay_finite() {
    // `prior_count = 0` is the floor under the log and nothing more: scores
    // are finite, empty blocks included, and distinct profiles are still
    // told apart.
    let k = 4;
    let mut rng = SmallRng::seed_from_u64(100);
    let mut rows = Vec::new();
    let mut truth = Vec::new();
    for b in 0..3 {
        let p = sim_profile(&mut rng, 0, SIM_M);
        for _ in 0..40 {
            rows.push(sim_entity(&mut rng, &p, 500.0));
            truth.push(b);
        }
    }
    let mut start = truth.clone();
    for e in (0..start.len()).step_by(5) {
        start[e] = (truth[e] + 1) % 3;
    }
    let profiles = Profiles::from_gene_sums(&rows, SIM_M);
    for parallel in [true, false] {
        let params = RefineParams {
            prior_count: 0.0,
            parallel,
            ..Default::default()
        };
        let stats = DcPoissonStats::from_profiles_with(&profiles, k, &start, &params);
        let all: Vec<usize> = (0..k).collect();
        let mut lp = vec![0f64; k];
        for e in 0..rows.len() {
            compute_log_probs_restricted(e, &stats, &profiles, &all, &mut lp);
            assert!(lp.iter().all(|x| x.is_finite()), "entity {e}: {lp:?}");
        }
        let mut labels = start.clone();
        sim_refine(&rows, &mut labels, k, &params);
        assert!(labels.iter().all(|&l| l < k));
        let right = labels.iter().zip(&truth).filter(|(a, b)| a == b).count();
        assert!(right >= truth.len() - 2, "{right} of {} right", truth.len());
    }
}

#[test]
fn labels_do_not_depend_on_the_thread_count() {
    // Eight blocks of forty, a fifth mislabelled. Every third entity is
    // given a candidate list without its current block, which leaves that
    // block's slot of the score buffer to whatever a thread scored last.
    let k = 8;
    let mut rng = SmallRng::seed_from_u64(110);
    let mut rows = Vec::new();
    let mut start = Vec::new();
    let base = sim_profile(&mut rng, 0, SIM_M);
    for b in 0..k {
        let p = sim_perturbed(&mut rng, &base);
        for _ in 0..40 {
            rows.push(sim_entity(&mut rng, &p, 400.0));
            start.push(if rng.random::<f64>() < 0.2 {
                (b + 1 + rng.random_range(0..k - 1)) % k
            } else {
                b
            });
        }
    }
    let mut profiles = Profiles::from_gene_sums(&rows, SIM_M);
    profiles.apply_feature_weighting(FeatureWeighting::FisherInfoNb);
    let candidates: Vec<Vec<usize>> = (0..rows.len())
        .map(|e| {
            (0..k)
                .filter(|&b| e % 3 != 0 || b != start[e])
                .collect::<Vec<usize>>()
        })
        .collect();
    for parallel in [true, false] {
        let params = RefineParams {
            parallel,
            ..Default::default()
        };
        let on_threads = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool")
                .install(|| {
                    let ctx = RefineContext {
                        profiles: &profiles,
                        k,
                        params: &params,
                        level_label: "test",
                    };
                    let mut labels = start.clone();
                    let mut rng = SmallRng::seed_from_u64(params.seed);
                    let moves = refine_with_candidates(&mut labels, &candidates, &mut rng, &ctx);
                    (labels, moves)
                })
        };
        let one = on_threads(1);
        assert!(one.1 > 0, "the fixture has moves to make");
        for threads in [3, 6] {
            assert_eq!(
                on_threads(threads),
                one,
                "{threads} threads, parallel={parallel}"
            );
        }
    }
}

/////////////////////////////////////////////
// Many features, most of them rarely seen //
/////////////////////////////////////////////

/// Lognormal(0, `sd`) rates on `m` features, normalised to one.
fn heavy_tailed_profile(rng: &mut SmallRng, m: usize, sd: f64) -> Vec<f64> {
    let ln = LogNormal::new(0.0, sd).unwrap();
    let mut p: Vec<f64> = (0..m).map(|_| ln.sample(rng)).collect();
    let s: f64 = p.iter().sum();
    p.iter_mut().for_each(|x| *x /= s);
    p
}

/// Blocks 0 (`n_small` entities) and 1 (`n_big`) of one heavy-tailed
/// profile over 20,000 features, `N_OUT` further entities of the same
/// profile in block 2, and `n_rest` more in block 3. Returns, for each
/// entity of block 2, its score for the small block less its score for
/// the large one, and the charge on a move between the two.
fn size_gap(n_small: usize, n_big: usize, n_rest: usize, depth: f64, seed: u64) -> Vec<(f64, f64)> {
    const N_OUT: usize = 24;
    let m = 20_000;
    let mut rng = SmallRng::seed_from_u64(seed);
    let p = heavy_tailed_profile(&mut rng, m, 2.0);
    let mut labels = vec![0; n_small];
    labels.extend(vec![1; n_big]);
    labels.extend(vec![2; N_OUT]);
    labels.extend(vec![3; n_rest]);
    let rows: Vec<Vec<(usize, f32)>> = labels
        .iter()
        .map(|_| sim_entity(&mut rng, &p, depth))
        .collect();
    let profiles = Profiles::from_gene_sums(&rows, m);
    let flat = RefineParams {
        min_gain_z: 0.0,
        ..Default::default()
    };
    let mut raw = DcPoissonStats::from_profiles_with(&profiles, 4, &labels, &flat);
    let mut charged = DcPoissonStats::from_profiles(&profiles, 4, &labels);
    let mut lp = vec![0f64; 4];
    (n_small + n_big..n_small + n_big + N_OUT)
        .map(|e| {
            // As a member of the small block, the entity's leave-one-out
            // score there is its score against the block as an outsider.
            raw.delta_move(e, 2, 0, &profiles);
            charged.delta_move(e, 2, 0, &profiles);
            compute_log_probs_restricted(e, &raw, &profiles, &[0, 1], &mut lp);
            let gap = lp[0] - lp[1];
            compute_log_probs_restricted(e, &charged, &profiles, &[0, 1], &mut lp);
            let charge = (lp[0] - lp[1]) - gap;
            raw.delta_move(e, 0, 2, &profiles);
            charged.delta_move(e, 0, 2, &profiles);
            (gap, charge)
        })
        .collect()
}

#[test]
fn blocks_of_one_profile_neither_attract_nor_repel_by_size() {
    // Most of 20,000 heavy-tailed features have a total of zero or one in
    // the small block. The score difference between the two blocks stays
    // within the charge a move has to clear, whichever way. (The two
    // blocks are a quarter of the entities: a block that makes up most of
    // the pooled profile is its own null, and is favoured for it.)
    for (n_small, n_big, depth) in [(6, 60, 3000.0), (3, 50, 1000.0)] {
        let gaps = size_gap(n_small, n_big, 200, depth, 120);
        let n = gaps.len() as f64;
        let mean_gap = gaps.iter().map(|g| g.0).sum::<f64>() / n;
        let mean_charge = gaps.iter().map(|g| g.1).sum::<f64>() / n;
        let beyond = gaps.iter().filter(|g| g.0.abs() > g.1).count();
        assert!(mean_charge > 0.0);
        assert!(
            mean_gap.abs() < 0.5 * mean_charge,
            "{n_small} vs {n_big}: mean gap {mean_gap:.1}, charge {mean_charge:.1}"
        );
        assert!(
            beyond as f64 <= 0.25 * n,
            "{n_small} vs {n_big}: {beyond} of {n} entities beyond the charge"
        );
    }
}

#[test]
fn rate_classes_reproduce_the_score_feature_by_feature() {
    // The null means are kept on a grid of pooled rates. Against the same
    // score with every feature's null mean taken at its own rate, the grid
    // costs a small fraction of a nat per thousand counts.
    let m = 4000;
    let mut rng = SmallRng::seed_from_u64(130);
    let p = heavy_tailed_profile(&mut rng, m, 2.0);
    let labels: Vec<usize> = (0..60).map(|e| [0, 1, 1, 2, 2, 2][e % 6]).collect();
    let rows: Vec<Vec<(usize, f32)>> = labels
        .iter()
        .map(|_| sim_entity(&mut rng, &p, 1500.0))
        .collect();
    let profiles = Profiles::from_gene_sums(&rows, m);
    let flat = RefineParams {
        min_gain_z: 0.0,
        ..Default::default()
    };
    let stats = DcPoissonStats::from_profiles_with(&profiles, 3, &labels, &flat);
    let mut pooled = vec![0f64; m];
    for row in &profiles.rows {
        for &(g, v) in row {
            pooled[g as usize] += v as f64;
        }
    }
    let total: f64 = pooled.iter().sum();
    let mut lp = vec![0f64; 3];
    let mut worst = 0f64;
    for (e, row) in profiles.rows.iter().enumerate() {
        compute_log_probs_restricted(e, &stats, &profiles, &[0, 1, 2], &mut lp);
        let sf = profiles.size_factor[e] as f64;
        for (k, &score) in lp.iter().enumerate() {
            let own = usize::from(k == labels[e]) as f64;
            let size = stats.size_sum[k] - own * sf;
            let exact: f64 = row
                .iter()
                .map(|&(g, v)| {
                    let v = v as f64;
                    let total_k = stats.gene_sum[stats.idx(k, g as usize)] - own * v;
                    let rate = (pooled[g as usize] - v) / (total - sf);
                    let mu = size * rate;
                    v * ((total_k + stats.prior).ln() - stats.null.at(mu.ln(), mu).0)
                })
                .sum();
            worst = worst.max((score - exact).abs() / sf);
        }
    }
    assert!(worst < 2e-4, "{worst} nats per count");
}

////////////////////
// Batch factors  //
////////////////////

/// Mean profile (normalised to one) of the entities `pick` selects.
fn mean_profile(profiles: &Profiles, pick: impl Fn(usize) -> bool) -> Vec<f64> {
    let mut sum = vec![0f64; profiles.num_features];
    for (e, row) in profiles.rows.iter().enumerate() {
        if pick(e) {
            for &(g, v) in row {
                sum[g as usize] += v as f64;
            }
        }
    }
    let s: f64 = sum.iter().sum();
    sum.iter_mut().for_each(|x| *x /= s);
    sum
}

fn l1(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

#[test]
fn rescaling_removes_a_batch_factor() {
    // Batch 1 doubles the even features. After rescaling, entities of the
    // two batches drawn from one profile have the same expected profile.
    let mut rng = SmallRng::seed_from_u64(70);
    let p = sim_profile(&mut rng, 0, SIM_M);
    let mut q = p.clone();
    q.iter_mut().step_by(2).for_each(|x| *x *= 2.0);
    let s: f64 = q.iter().sum();
    q.iter_mut().for_each(|x| *x /= s);
    let mut rows = Vec::new();
    let mut batch = Vec::new();
    for e in 0..200 {
        let b = e % 2;
        rows.push(sim_entity(&mut rng, if b == 0 { &p } else { &q }, 2000.0));
        batch.push(b);
    }
    let plain = Profiles::from_gene_sums(&rows, SIM_M);
    let apart = l1(
        &mean_profile(&plain, |e| batch[e] == 0),
        &mean_profile(&plain, |e| batch[e] == 1),
    );
    let mut profiles = Profiles::from_gene_sums(&rows, SIM_M);
    profiles.rescale_batches(&batch, None);
    let together = l1(
        &mean_profile(&profiles, |e| batch[e] == 0),
        &mean_profile(&profiles, |e| batch[e] == 1),
    );
    assert!(apart > 0.3, "the batches start apart: {apart}");
    assert!(together < 0.1 * apart, "{apart} -> {together}");
    for (row, &sf) in profiles.rows.iter().zip(&profiles.size_factor) {
        let sum: f32 = row.iter().map(|&(_, v)| v).sum();
        assert!((sum - sf).abs() <= 1e-3 * sf);
    }
    // One batch: nothing to remove.
    let mut single = Profiles::from_gene_sums(&rows, SIM_M);
    single.rescale_batches(&vec![0; rows.len()], None);
    assert_eq!(single.rows, plain.rows);
    assert_eq!(single.size_factor, plain.size_factor);
}

#[test]
fn a_stray_count_and_a_lone_entity_are_left_nearly_as_they_are() {
    // Feature 0: a thousand counts in batch 0, one stray count in batch 1.
    // Feature 1: five counts in batch 0, one in batch 1. Feature 2 carries
    // the bulk of every entity. Batch 2 is a single entity with a profile
    // of its own.
    let mut rows: Vec<Vec<(usize, f32)>> = Vec::new();
    let mut batch = Vec::new();
    for e in 0..40 {
        let b = e % 2;
        let mut row = vec![(2usize, 500.0f32), (3, 100.0 + (e % 7) as f32)];
        if b == 0 {
            row.push((0, 50.0));
        }
        if e < 10 && b == 0 {
            row.push((1, 1.0));
        }
        rows.push(row);
        batch.push(b);
    }
    rows[1].push((0, 1.0));
    rows[3].push((1, 1.0));
    rows.push(vec![(2, 10.0), (3, 700.0), (4, 90.0)]);
    batch.push(2);
    let plain = Profiles::from_gene_sums(&rows, 5);
    let mut profiles = Profiles::from_gene_sums(&rows, 5);
    profiles.rescale_batches(&batch, None);
    let value = |p: &Profiles, e: usize, g: u32| {
        p.rows[e]
            .iter()
            .find(|&&(f, _)| f == g)
            .map_or(0.0, |&(_, v)| v)
    };
    // The stray count: the raw ratio of pooled to batch rate is in the
    // hundreds; the factor is bounded.
    let stray = value(&profiles, 1, 0);
    assert!(
        (1.0..=BATCH_FACTOR_BOUND).contains(&stray),
        "stray count became {stray}"
    );
    // Thin evidence either way: nearly untouched.
    let thin = value(&profiles, 3, 1);
    assert!((0.8..=1.25).contains(&thin), "thin count became {thin}");
    // The lone entity of batch 2 keeps its profile exactly.
    let lone = rows.len() - 1;
    assert_eq!(profiles.rows[lone], plain.rows[lone]);
    assert_eq!(profiles.size_factor[lone], plain.size_factor[lone]);
}

#[test]
fn strata_keep_a_type_alike_across_batches_of_unequal_composition() {
    // Two types; batch 0 is nine tenths type 0, batch 1 one tenth, and
    // batch 1 doubles the even features. Compared with the pooled profile,
    // batch 0 looks like "too much of type 0" and the fit scales the
    // type's features down there and up in batch 1: the type no longer
    // matches itself across the batches. Compared within the types, only
    // the doubling is taken out.
    let mut rng = SmallRng::seed_from_u64(140);
    let types = [
        sim_profile(&mut rng, 0, SIM_M),
        sim_profile(&mut rng, 0, SIM_M),
    ];
    let doubled: Vec<Vec<f64>> = types
        .iter()
        .map(|p| {
            let mut q = p.clone();
            q.iter_mut().step_by(2).for_each(|x| *x *= 2.0);
            let s: f64 = q.iter().sum();
            q.iter_mut().for_each(|x| *x /= s);
            q
        })
        .collect();
    let mut rows = Vec::new();
    let mut batch = Vec::new();
    let mut ty = Vec::new();
    for e in 0..400 {
        let b = e % 2;
        let t = usize::from((e / 2) % 10 == 0) ^ b;
        rows.push(sim_entity(
            &mut rng,
            if b == 0 { &types[t] } else { &doubled[t] },
            2000.0,
        ));
        batch.push(b);
        ty.push(t);
    }
    assert_eq!(
        (0..400).filter(|&e| batch[e] == 0 && ty[e] == 0).count(),
        180
    );
    assert_eq!(
        (0..400).filter(|&e| batch[e] == 1 && ty[e] == 0).count(),
        20
    );
    // Distance between the two batches' entities of one type, summed over
    // the types.
    let apart = |p: &Profiles| -> f64 {
        (0..2)
            .map(|t| {
                l1(
                    &mean_profile(p, |e| ty[e] == t && batch[e] == 0),
                    &mean_profile(p, |e| ty[e] == t && batch[e] == 1),
                )
            })
            .sum()
    };
    let raw = apart(&Profiles::from_gene_sums(&rows, SIM_M));
    let mut pooled = Profiles::from_gene_sums(&rows, SIM_M);
    pooled.rescale_batches(&batch, None);
    let mut within = Profiles::from_gene_sums(&rows, SIM_M);
    within.rescale_batches(&batch, Some(&ty));
    let (pooled, within) = (apart(&pooled), apart(&within));
    assert!(raw > 0.6, "the doubling sets the batches apart: {raw}");
    assert!(
        within < 0.3 * raw,
        "within the types the factor is removed: {raw} -> {within}"
    );
    assert!(
        pooled > raw,
        "against the pool the composition is taken for a factor: {raw} -> {pooled}"
    );
}
