use super::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// The merge loop as it was before the pairwise cache: every active pair
/// scored afresh at every step. The reference the cached loop must match.
fn bhc_merge_reference(input: BhcInput<'_>, gamma: f64) -> Vec<BhcMerge> {
    const BG_EPS: f64 = 1e-9;

    let k = input.k;
    let m = input.m;
    let gamma = gamma.max(1e-6);
    assert_eq!(input.gene_sum.len(), k * m, "gene_sum must be k × m");
    assert_eq!(input.size_sum.len(), k);
    assert_eq!(input.effective_size.len(), k);

    // Pooled empirical marginal bg[g] = (Σ_k T_k,g) / (Σ_k S_k). Floor at BG_EPS so
    // lgamma(γ·bg) stays finite for genes with no mass anywhere (those genes then
    // cancel out of any Bayes factor — see module docstring).
    let mut bg = vec![0.0f64; m];
    let mut total_mass = 0.0f64;
    for c in 0..k {
        if input.effective_size[c] == 0 {
            continue;
        }
        let row = &input.gene_sum[c * m..(c + 1) * m];
        for (dst, &src) in bg.iter_mut().zip(row.iter()) {
            *dst += src;
        }
        total_mass += input.size_sum[c];
    }
    let total_mass = total_mass.max(BG_EPS);
    for p in bg.iter_mut() {
        *p = (*p / total_mass).max(BG_EPS);
    }

    let gamma_bg: Vec<f64> = bg.iter().map(|&p| gamma * p).collect();
    let sum_lgamma_gamma_bg: f64 = gamma_bg
        .iter()
        .map(|&gb| SpecialGamma::ln_gamma(gb).0)
        .sum();
    let prior = Prior {
        gamma,
        gamma_bg,
        sum_lgamma_gamma_bg,
        lgamma_gamma: SpecialGamma::ln_gamma(gamma).0,
    };

    // Rescale (T, S) so effective sample size S_eff = effective_size. Preserves
    // proportions T_g / S; puts log BF magnitudes in an interpretable range
    // (see module docstring).
    let mut arena: Vec<NodeStats> = Vec::with_capacity(k);
    let mut active: Vec<usize> = Vec::with_capacity(k);
    for c in 0..k {
        if input.effective_size[c] == 0 {
            continue;
        }
        let s_raw = input.size_sum[c];
        let n_eff = input.effective_size[c] as f64;
        let scale = if s_raw > 0.0 { n_eff / s_raw } else { 0.0 };
        let t: Vec<f64> = input.gene_sum[c * m..(c + 1) * m]
            .iter()
            .map(|&x| x * scale)
            .collect();
        let f = f_node(&t, n_eff, &prior);
        arena.push(NodeStats {
            t_gene: t,
            s_size: n_eff,
            n_samples: input.effective_size[c] as i32,
            f_cache: f,
            id: c as i32,
        });
        active.push(arena.len() - 1);
    }

    let k_eff = active.len();
    if k_eff < 2 {
        return Vec::new();
    }
    let mut merges: Vec<BhcMerge> = Vec::with_capacity(k_eff - 1);
    let mut next_id: i32 = k as i32;

    while active.len() >= 2 {
        // Maximize log BF; tiebreak on ascending (min_id, max_id).
        let mut best: Option<(usize, usize, f64, (i32, i32))> = None;
        for i in 0..active.len() {
            for j in (i + 1)..active.len() {
                let a = &arena[active[i]];
                let b = &arena[active[j]];
                let bf = pairwise_log_bf(a, b, &prior);
                let key = order_pair(a.id, b.id);
                let pick = match best {
                    None => true,
                    Some((_, _, best_bf, best_key)) => {
                        bf > best_bf || (bf == best_bf && key < best_key)
                    }
                };
                if pick {
                    best = Some((i, j, bf, key));
                }
            }
        }

        let (i, j, log_bf, _) = best.expect("at least one pair exists");
        let ai = active[i];
        let aj = active[j];
        let (left_id, right_id) = order_pair(arena[ai].id, arena[aj].id);

        let new_t: Vec<f64> = arena[ai]
            .t_gene
            .iter()
            .zip(arena[aj].t_gene.iter())
            .map(|(x, y)| x + y)
            .collect();
        let new_s = arena[ai].s_size + arena[aj].s_size;
        let new_n = arena[ai].n_samples + arena[aj].n_samples;
        let new_f = f_node(&new_t, new_s, &prior);

        arena.push(NodeStats {
            t_gene: new_t,
            s_size: new_s,
            n_samples: new_n,
            f_cache: new_f,
            id: next_id,
        });
        let new_arena_idx = arena.len() - 1;

        merges.push(BhcMerge {
            id: next_id,
            left: left_id,
            right: right_id,
            log_bf,
            n_samples: new_n,
        });
        next_id += 1;

        // Remove j first (j > i) so the index for i stays valid.
        active.remove(j);
        active.remove(i);
        active.push(new_arena_idx);
    }

    merges
}

#[test]
fn cached_merges_match_scoring_every_pair_afresh() {
    for seed in 0..6u64 {
        let mut rng = SmallRng::seed_from_u64(seed);
        let (k, m) = (12 + seed as usize * 5, 40);
        // A few shared shapes plus noise, some clusters empty, some exact
        // duplicates so ties in the log BF come up.
        let shapes: Vec<Vec<f64>> = (0..4)
            .map(|_| (0..m).map(|_| rng.random_range(0.0..5.0)).collect())
            .collect();
        let mut gene_sum = Vec::with_capacity(k * m);
        let mut effective_size = Vec::with_capacity(k);
        for c in 0..k {
            let row: Vec<f64> = if c % 7 == 6 {
                vec![0.0; m]
            } else if c % 5 == 4 {
                gene_sum[(c - 1) * m..c * m].to_vec()
            } else {
                let s = &shapes[rng.random_range(0..shapes.len())];
                s.iter().map(|&x| x * rng.random_range(0.5..2.0)).collect()
            };
            effective_size.push(if row.iter().sum::<f64>() > 0.0 {
                1 + c % 9
            } else {
                0
            });
            gene_sum.extend(row);
        }
        let size_sum: Vec<f64> = gene_sum.chunks(m).map(|r| r.iter().sum()).collect();
        let input = || BhcInput {
            k,
            m,
            gene_sum: &gene_sum,
            size_sum: &size_sum,
            effective_size: &effective_size,
        };
        let got = bhc_merge(input(), 0.5 * m as f64);
        let want = bhc_merge_reference(input(), 0.5 * m as f64);
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(&want) {
            assert_eq!(
                (g.id, g.left, g.right, g.n_samples),
                (w.id, w.left, w.right, w.n_samples)
            );
            assert_eq!(
                g.log_bf.to_bits(),
                w.log_bf.to_bits(),
                "seed {seed}, merge {}",
                g.id
            );
        }
    }
}
