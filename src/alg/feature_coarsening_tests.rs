use super::*;
use approx::assert_relative_eq;

#[test]
fn test_aggregate_rows_sums_match() {
    // 6 features, 3 samples
    let data = DMatrix::from_row_slice(
        6,
        3,
        &[
            1.0, 2.0, 3.0, // feature 0
            4.0, 5.0, 6.0, // feature 1
            7.0, 8.0, 9.0, // feature 2
            10.0, 11.0, 12.0, // feature 3
            13.0, 14.0, 15.0, // feature 4
            16.0, 17.0, 18.0, // feature 5
        ],
    );

    let fc = FeatureCoarsening {
        fine_to_coarse: vec![0, 0, 1, 1, 2, 2],
        coarse_to_fine: vec![vec![0, 1], vec![2, 3], vec![4, 5]],
        num_coarse: 3,
    };

    let agg = fc.aggregate_rows_ds(&data);
    assert_eq!(agg.nrows(), 3);
    assert_eq!(agg.ncols(), 3);

    // Group 0: features 0+1
    assert_relative_eq!(agg[(0, 0)], 5.0);
    assert_relative_eq!(agg[(0, 1)], 7.0);
    assert_relative_eq!(agg[(0, 2)], 9.0);

    // Group 1: features 2+3
    assert_relative_eq!(agg[(1, 0)], 17.0);

    // Group 2: features 4+5
    assert_relative_eq!(agg[(2, 0)], 29.0);

    // Column sums should be preserved
    let orig_col_sum: f32 = data.column(0).iter().sum();
    let agg_col_sum: f32 = agg.column(0).iter().sum();
    assert_relative_eq!(orig_col_sum, agg_col_sum);
}

#[test]
fn test_aggregate_columns_nd() {
    // 2 samples, 4 features → 2 groups
    let data = DMatrix::from_row_slice(
        2,
        4,
        &[
            1.0, 2.0, 3.0, 4.0, // sample 0
            5.0, 6.0, 7.0, 8.0, // sample 1
        ],
    );

    let fc = FeatureCoarsening {
        fine_to_coarse: vec![0, 0, 1, 1],
        coarse_to_fine: vec![vec![0, 1], vec![2, 3]],
        num_coarse: 2,
    };

    let agg = fc.aggregate_columns_nd(&data);
    assert_eq!(agg.nrows(), 2);
    assert_eq!(agg.ncols(), 2);
    assert_relative_eq!(agg[(0, 0)], 3.0); // 1+2
    assert_relative_eq!(agg[(0, 1)], 7.0); // 3+4
    assert_relative_eq!(agg[(1, 0)], 11.0); // 5+6
    assert_relative_eq!(agg[(1, 1)], 15.0); // 7+8
}

#[test]
fn test_expand_logits_preserves_probabilities() {
    // 2 topics, 3 coarse features → expand to 6 fine features
    // Groups: {0,1}, {2,3}, {4,5}
    let logits = DMatrix::from_row_slice(
        3,
        2,
        &[
            -1.2, -0.8, // coarse 0
            -0.5, -1.5, // coarse 1
            -1.0, -1.0, // coarse 2
        ],
    );

    let fc = FeatureCoarsening {
        fine_to_coarse: vec![0, 0, 1, 1, 2, 2],
        coarse_to_fine: vec![vec![0, 1], vec![2, 3], vec![4, 5]],
        num_coarse: 3,
    };

    let expanded = fc.expand_log_dict_dk(&logits, 6);
    assert_eq!(expanded.nrows(), 6);
    assert_eq!(expanded.ncols(), 2);

    let ln2 = 2.0f32.ln();

    // For each topic, sum of exp(expanded) within each group
    // should equal exp(coarse logit)
    for k in 0..2 {
        for (c, group) in fc.coarse_to_fine.iter().enumerate() {
            let coarse_prob: f32 = logits[(c, k)].exp();
            let fine_sum: f32 = group.iter().map(|&f| expanded[(f, k)].exp()).sum();
            assert_relative_eq!(fine_sum, coarse_prob, epsilon = 1e-6);
        }
    }

    // Each fine feature in a group of size 2 gets logit - ln(2)
    assert_relative_eq!(expanded[(0, 0)], -1.2 - ln2, epsilon = 1e-6);
    assert_relative_eq!(expanded[(1, 0)], -1.2 - ln2, epsilon = 1e-6);
}

////////////////////////////////////////
// Building a coarsening from counts //
////////////////////////////////////////

/// `n_pb` pseudobulks of 6 cells. `n_programs` programs of `per` features,
/// each raised 8× on its own block of pseudobulks, then `n_flat` features at
/// one flat rate and `n_empty` never counted — the empty features a real axis
/// is mostly made of.
fn planted(
    n_programs: usize,
    per: usize,
    n_flat: usize,
    n_empty: usize,
    n_pb: usize,
    seed: u64,
) -> (DMatrix<f32>, Vec<f32>) {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Poisson};
    let mut rng = StdRng::seed_from_u64(seed);
    let sizes = vec![6.0f32; n_pb];
    let d = n_programs * per + n_flat + n_empty;
    let mut counts = DMatrix::<f32>::zeros(d, n_pb);
    let block = (n_pb / n_programs.max(1)).max(1);
    for p in 0..n_programs {
        for i in 0..per {
            let base = 0.5 + (i % 3) as f64 * 0.5;
            for s in 0..n_pb {
                let lift = if s / block == p { 8.0 } else { 1.0 };
                let rate = base * lift * f64::from(sizes[s]);
                counts[(p * per + i, s)] = Poisson::new(rate).unwrap().sample(&mut rng) as f32;
            }
        }
    }
    for f in 0..n_flat {
        for s in 0..n_pb {
            let rate = 0.3 * f64::from(sizes[s]);
            counts[(n_programs * per + f, s)] = Poisson::new(rate).unwrap().sample(&mut rng) as f32;
        }
    }
    (counts, sizes)
}

fn groups_of(labels: &[usize], rows: std::ops::Range<usize>) -> std::collections::BTreeSet<usize> {
    rows.map(|i| labels[i]).collect()
}

#[test]
fn a_flat_or_empty_feature_carries_no_grouping_evidence() {
    let (counts, sizes) = planted(2, 5, 5, 5, 20, 11);
    let informative = informative_features(&counts, &sizes);
    assert!(
        informative[..10].iter().all(|&b| b),
        "programs: {informative:?}"
    );
    assert!(
        informative[10..].iter().all(|&b| !b),
        "flat/empty: {informative:?}"
    );
}

#[test]
fn programs_are_recovered_and_empty_features_share_the_background() {
    let (counts, sizes) = planted(3, 10, 20, 20, 24, 7);
    let fc = coarsen_features(&counts, &sizes, &[4], 3)
        .unwrap()
        .remove(0);
    let l = &fc.fine_to_coarse;
    for p in 0..3 {
        assert_eq!(
            groups_of(l, p * 10..(p + 1) * 10).len(),
            1,
            "program {p} split: {l:?}"
        );
    }
    let programs: std::collections::BTreeSet<usize> = (0..3).map(|p| l[p * 10]).collect();
    assert_eq!(programs.len(), 3, "programs merged: {l:?}");
    let background = groups_of(l, 30..70);
    assert_eq!(background.len(), 1, "flat and empty features split: {l:?}");
    assert!(
        programs.is_disjoint(&background),
        "a program joined the background"
    );
    assert_eq!(fc.num_coarse, 4);
}

#[test]
fn groups_stay_bounded_when_most_features_are_empty() {
    // 30 programs of 8 among 2,000 empty and flat features: no group may
    // swallow the informative features, however many empty ones there are.
    let (counts, sizes) = planted(30, 8, 1000, 1000, 60, 5);
    let fc = coarsen_features(&counts, &sizes, &[31], 9)
        .unwrap()
        .remove(0);
    let mut size = std::collections::BTreeMap::<usize, usize>::new();
    for &g in &fc.fine_to_coarse[..240] {
        *size.entry(g).or_default() += 1;
    }
    let largest = size.values().copied().max().unwrap();
    assert!(
        largest <= 24,
        "an informative group holds {largest} of 240: {size:?}"
    );
    assert!(
        size.len() >= 25,
        "only {} groups for 30 programs",
        size.len()
    );
}

#[test]
fn coarser_levels_nest_and_keep_the_background_whole() {
    let (counts, sizes) = planted(6, 6, 10, 10, 36, 13);
    let levels = coarsen_features(&counts, &sizes, &[3, 5, 7], 21).unwrap();
    assert_eq!(levels.len(), 3);
    for w in levels.windows(2) {
        let mut parent = std::collections::HashMap::<usize, usize>::new();
        for (g, (&c, &f)) in w[0]
            .fine_to_coarse
            .iter()
            .zip(&w[1].fine_to_coarse)
            .enumerate()
        {
            let p = *parent.entry(f).or_insert(c);
            assert_eq!(p, c, "feature {g}: fine group {f} spans coarse {p} and {c}");
        }
    }
    for level in &levels {
        assert_eq!(
            groups_of(&level.fine_to_coarse, 36..56).len(),
            1,
            "background split"
        );
        assert!(level.num_coarse <= 7);
    }
}

#[test]
fn coarsening_is_reproducible() {
    let (counts, sizes) = planted(3, 10, 20, 20, 24, 7);
    let a = coarsen_features(&counts, &sizes, &[4], 3).unwrap();
    let b = coarsen_features(&counts, &sizes, &[4], 3).unwrap();
    assert_eq!(a[0].fine_to_coarse, b[0].fine_to_coarse);
}

#[test]
fn target_one_stays_one_group_when_background_is_reserved() {
    // Informative + empty features: a budget of 1 must not force a second
    // (informative) cluster on top of the background.
    let (counts, sizes) = planted(2, 8, 20, 20, 24, 11);
    let fc = coarsen_features(&counts, &sizes, &[1], 5)
        .unwrap()
        .remove(0);
    assert_eq!(fc.num_coarse, 1, "fine_to_coarse={:?}", fc.fine_to_coarse);
    assert!(fc.fine_to_coarse.iter().all(|&g| g == 0));
}

#[test]
fn a_coarsening_expands_back_exactly() {
    // Within each group, the fine logits exponentiate back to the coarse one.
    use legume_numeric::matrix::traits::SampleOps;
    let (counts, sizes) = planted(4, 8, 20, 20, 32, 17);
    let d = counts.nrows();
    let k = 5;
    let fc = coarsen_features(&counts, &sizes, &[6], 1)
        .unwrap()
        .remove(0);
    let coarse_logits = DMatrix::<f32>::rnorm(fc.num_coarse, k);
    let expanded = fc.expand_log_dict_dk(&coarse_logits, d);
    assert_eq!((expanded.nrows(), expanded.ncols()), (d, k));
    for kk in 0..k {
        for (c, group) in fc.coarse_to_fine.iter().enumerate() {
            let fine_sum: f32 = group.iter().map(|&f| expanded[(f, kk)].exp()).sum();
            assert_relative_eq!(fine_sum, coarse_logits[(c, kk)].exp(), epsilon = 1e-4);
        }
    }
}

/// Source axis [g0 g1 g2 g3], groups {g0,g1} and {g2,g3}. The new axis is
/// [g1 gX g3 gY g0]: two source genes reordered, one dropped, two new.
fn grown_fixture() -> (FeatureCoarsening, Vec<Option<usize>>, Vec<Vec<f32>>) {
    let source = FeatureCoarsening::from_fine_to_coarse(vec![0, 0, 1, 1], 2).unwrap();
    let remap = vec![Some(1), None, Some(3), None, Some(0)];
    // Two pseudobulks: group 0's genes lean to the first, group 1's to the
    // second; gX leans to the first, gY to the second.
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let unit = vec![
        vec![s, -s], // g1
        vec![s, -s], // gX
        vec![-s, s], // g3
        vec![-s, s], // gY
        vec![s, -s], // g0
    ];
    (source, remap, unit)
}

#[test]
fn grown_known_features_keep_their_group_and_new_ones_join_the_nearest() {
    let (source, remap, unit) = grown_fixture();
    let grown = source.grow_by_profile(&remap, &unit).unwrap();
    assert_eq!(grown.fine_to_coarse, vec![0, 0, 1, 1, 0]);
    assert_eq!(
        grown.num_coarse, 2,
        "the group count is what consumers are keyed to"
    );
    assert_eq!(grown.coarse_to_fine[0], vec![0, 1, 4]);
    assert_eq!(grown.coarse_to_fine[1], vec![2, 3]);
}

#[test]
fn grown_group_with_no_surviving_member_attracts_nothing_but_keeps_its_index() {
    // Three source groups; group 2's only gene is absent from the new axis.
    let source = FeatureCoarsening::from_fine_to_coarse(vec![0, 1, 2], 3).unwrap();
    let remap = vec![Some(0), Some(1), None];
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let unit = vec![vec![s, -s], vec![-s, s], vec![-s, s]];
    let grown = source.grow_by_profile(&remap, &unit).unwrap();
    assert_eq!(grown.num_coarse, 3);
    assert_eq!(grown.fine_to_coarse, vec![0, 1, 1]);
    assert!(grown.coarse_to_fine[2].is_empty());
}

#[test]
fn grown_feature_without_a_profile_goes_to_the_largest_group() {
    let source = FeatureCoarsening::from_fine_to_coarse(vec![0, 0, 1], 2).unwrap();
    let remap = vec![Some(0), Some(1), Some(2), None];
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let unit = vec![vec![s, -s], vec![s, -s], vec![-s, s], vec![0.0, 0.0]];
    let grown = source.grow_by_profile(&remap, &unit).unwrap();
    assert_eq!(grown.fine_to_coarse[3], 0);
}

#[test]
fn grown_axis_with_nothing_in_common_is_refused() {
    let source = FeatureCoarsening::from_fine_to_coarse(vec![0, 1], 2).unwrap();
    let err = match source.grow_by_profile(&[None, None], &[vec![1.0], vec![1.0]]) {
        Ok(_) => panic!("an axis with nothing in common must be refused"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("no feature"), "{err}");
}

#[test]
fn from_fine_to_coarse_rejects_a_group_index_out_of_range() {
    assert!(FeatureCoarsening::from_fine_to_coarse(vec![0, 2], 2).is_err());
}

/// Turning the grouping off should be sayable, not encoded as a number whose
/// literal reading ("at most zero features") is the opposite of its meaning.
/// The numeric spelling stays, because recorded runs and existing scripts use
/// it, so the two have to agree.
mod switching_it_off {
    use super::FeatureCoarseningArgs;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        args: FeatureCoarseningArgs,
    }

    fn parse(extra: &[&str]) -> Result<FeatureCoarseningArgs, clap::Error> {
        Cli::try_parse_from(["x"].iter().copied().chain(extra.iter().copied())).map(|c| c.args)
    }

    #[test]
    fn the_named_switch_and_the_zero_agree() {
        assert_eq!(
            parse(&[]).unwrap().cap().map(std::num::NonZeroUsize::get),
            Some(1000)
        );
        assert!(parse(&["--no-feature-coarsening"]).unwrap().cap().is_none());
        assert!(parse(&["--max-coarse-features", "0"])
            .unwrap()
            .cap()
            .is_none());
        assert_eq!(
            parse(&["--max-coarse-features", "250"])
                .unwrap()
                .cap()
                .map(std::num::NonZeroUsize::get),
            Some(250)
        );
    }

    /// One says group at most N, the other says do not group. Asking for both
    /// is a contradiction rather than a precedence puzzle.
    #[test]
    fn asking_for_both_is_refused() {
        assert!(parse(&["--no-feature-coarsening", "--max-coarse-features", "250"]).is_err());
    }
}

//////////////////////////////////////
// Single-level partition, options //
//////////////////////////////////////

/// `planted` plus `n_scattered` isolated rows: each counted heavily in ONE
/// pseudobulk and nowhere else, so it is not flat yet shares a profile with
/// nothing — a feature that arose at random.
fn planted_with_scattered(n_scattered: usize) -> (DMatrix<f32>, Vec<f32>, usize) {
    let (base, sizes) = planted(3, 10, 5, 10, 24, 11);
    let d0 = base.nrows();
    let mut counts = DMatrix::<f32>::zeros(d0 + n_scattered, base.ncols());
    counts.rows_mut(0, d0).copy_from(&base);
    for j in 0..n_scattered {
        counts[(d0 + j, (7 * j + 3) % base.ncols())] = 40.0;
    }
    (counts, sizes, d0)
}

/// With no options the partition is the single-level coarsening, its
/// background is the last group and its flags are the homogeneity test.
#[test]
fn a_partition_without_options_is_the_single_level_coarsening() {
    let (counts, sizes) = planted(3, 10, 20, 20, 24, 7);
    let fc = coarsen_features(&counts, &sizes, &[4], 3)
        .unwrap()
        .remove(0);
    let part = partition_features(&counts, &sizes, 4, 3, &PartitionOptions::default()).unwrap();
    assert_eq!(part.labels, fc.fine_to_coarse);
    assert_eq!(part.num_groups, fc.num_coarse);
    assert_eq!(part.background, Some(fc.num_coarse - 1));
    assert_eq!(part.informative, informative_features(&counts, &sizes));
}

/// With a minimum group size, scattered and near-empty rows share the
/// background, every other group meets the minimum, the budget holds, and no
/// group mixes two programs.
#[test]
fn scattered_features_join_the_background_and_groups_meet_the_minimum() {
    let (counts, sizes, d0) = planted_with_scattered(4);
    let opts = PartitionOptions {
        min_group_size: 4,
        block: None,
    };
    let part = partition_features(&counts, &sizes, 8, 3, &opts).unwrap();
    let bg = part.background.expect("a background group");
    assert!(part.num_groups <= 8);
    assert!(part.labels.iter().all(|&m| m < part.num_groups));
    for (i, &m) in part.labels.iter().enumerate().skip(d0) {
        assert_eq!(m, bg, "scattered row {i} left the background");
    }
    for (i, &m) in part.labels.iter().enumerate().take(45).skip(35) {
        assert_eq!(m, bg, "empty row {i} left the background");
    }
    let mut size = std::collections::HashMap::<usize, usize>::new();
    for &m in &part.labels {
        *size.entry(m).or_default() += 1;
    }
    for (&m, &n) in &size {
        assert!(m == bg || n >= 4, "group {m} holds {n} < 4");
    }
    let programs: Vec<_> = (0..3)
        .map(|p| groups_of(&part.labels, p * 10..(p + 1) * 10))
        .collect();
    for a in 0..3 {
        for b in a + 1..3 {
            assert!(
                programs[a].is_disjoint(&programs[b]),
                "programs {a} and {b} share a group"
            );
        }
    }
}

/// Two blocks carrying the same programs: without blocks a program's rows in
/// both blocks share a group; with blocks no group spans two blocks, the
/// near-empty rows of both share one background, and the budget holds.
#[test]
fn a_blocked_partition_never_puts_two_blocks_in_one_group() {
    let (half, sizes) = planted(3, 10, 5, 10, 24, 13);
    let h = half.nrows();
    let mut counts = DMatrix::<f32>::zeros(2 * h, half.ncols());
    counts.rows_mut(0, h).copy_from(&half);
    counts.rows_mut(h, h).copy_from(&half);
    let block: Vec<u32> = (0..2 * h).map(|i| u32::from(i >= h)).collect();

    let plain = partition_features(&counts, &sizes, 8, 3, &PartitionOptions::default()).unwrap();
    assert!(
        (0..30).any(|i| plain.labels[i] == plain.labels[i + h]),
        "the fixture should merge across blocks without them"
    );

    let opts = PartitionOptions {
        min_group_size: 0,
        block: Some(block.clone()),
    };
    let part = partition_features(&counts, &sizes, 8, 3, &opts).unwrap();
    let bg = part.background.expect("a background");
    assert!(part.num_groups <= 8);
    for m in (0..part.num_groups).filter(|&m| m != bg) {
        let blocks: std::collections::BTreeSet<u32> = (0..2 * h)
            .filter(|&i| part.labels[i] == m)
            .map(|i| block[i])
            .collect();
        assert!(blocks.len() <= 1, "group {m} spans blocks {blocks:?}");
    }
    for i in (35..45).chain(h + 35..h + 45) {
        assert_eq!(part.labels[i], bg, "empty row {i} left the one background");
    }
}

/// More blocks with informative features than group slots is refused.
#[test]
fn more_blocks_than_slots_is_refused() {
    let (counts, sizes) = planted(3, 10, 0, 0, 24, 5);
    let opts = PartitionOptions {
        min_group_size: 0,
        block: Some((0..30).map(|i| i as u32).collect()),
    };
    assert!(partition_features(&counts, &sizes, 8, 3, &opts).is_err());
}
