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
    // Same profile, a tenth of the size: by the old score every member of
    // the small block left it. Nothing tells the blocks apart, so members
    // may go either way, but the small block must not empty — with the
    // size limits lifted too.
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

#[test]
fn mixed_blocks_are_purified_within_the_size_limits() {
    // Twelve blocks of unequal size over three profiles, each two-thirds
    // one profile. The members that do not belong move out; no block falls
    // under its floor, exceeds its cap, or empties, and the members a
    // block is named after stay.
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
    let purity = |labels: &[usize]| {
        let mut c = vec![0usize; k * t];
        for (&l, &ty) in labels.iter().zip(&truth) {
            c[l * t + ty] += 1;
        }
        let dom: usize = c.chunks(t).map(|b| *b.iter().max().unwrap()).sum();
        dom as f64 / labels.len() as f64
    };
    let n0 = block_sizes(&start, k);
    let n_bar = start.len() as f64 / k as f64;
    for params in sim_param_sets() {
        let mut labels = start.clone();
        sim_refine(&rows, &mut labels, k, &params);
        let n1 = block_sizes(&labels, k);
        for b in 0..k {
            let floor = (0.9 * (n0[b] as f64).min(n_bar)).ceil() as usize;
            let cap = (1.5 * (n0[b] as f64).max(n_bar)).ceil() as usize;
            assert!(
                (floor..=cap).contains(&n1[b]),
                "block {b}: {} -> {} outside {floor}..={cap}",
                n0[b],
                n1[b]
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
        // worst leave first; a sequential sweep takes them as it visits.
        let share = if params.parallel { 0.9 } else { 0.8 };
        assert!(
            stayed as f64 >= share * all as f64,
            "{stayed} of {all} stayed (parallel={})",
            params.parallel
        );
    }
}

#[test]
fn lifted_floor_lets_a_mixed_block_dissolve() {
    // Without a floor the likelihood alone decides: a block that is half
    // one profile and half another gives all its members to the two pure
    // blocks.
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
    assert_eq!(block_sizes(&labels, 3), vec![60, 60, 0]);

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
    for b in 0..3 {
        assert!((stats.block_var[b] - fresh.block_var[b]).abs() < 1e-9);
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
    let mut profiles = Profiles::from_gene_sums(&rows, SIM_M);
    profiles.rescale_batches_to_pooled(&batch);
    let mut share = vec![[0f64; 2]; SIM_M];
    let mut tot = [0f64; 2];
    for (row, &b) in profiles.rows.iter().zip(&batch) {
        for &(g, v) in row {
            share[g as usize][b] += v as f64;
            tot[b] += v as f64;
        }
    }
    // (features both batches show; one a batch never shows keeps its values)
    let seen: f64 = share.iter().filter(|s| s[0] > 0.0 && s[1] > 0.0).count() as f64;
    assert!(seen > 0.9 * SIM_M as f64);
    for (g, s) in share.iter().enumerate() {
        if s[0] > 0.0 && s[1] > 0.0 {
            let (a, b) = (s[0] / tot[0], s[1] / tot[1]);
            assert!((a - b).abs() <= 1e-2 * (a + b), "feature {g}: {a} vs {b}");
        }
    }
    for (row, &sf) in profiles.rows.iter().zip(&profiles.size_factor) {
        let sum: f32 = row.iter().map(|&(_, v)| v).sum();
        assert!((sum - sf).abs() <= 1e-3 * sf);
    }
    // One batch: nothing to remove.
    let mut single = Profiles::from_gene_sums(&rows, SIM_M);
    single.rescale_batches_to_pooled(&vec![0; rows.len()]);
    let plain = Profiles::from_gene_sums(&rows, SIM_M);
    assert_eq!(single.size_factor, plain.size_factor);
}
