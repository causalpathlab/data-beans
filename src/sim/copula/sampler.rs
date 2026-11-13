//! Count sampler shared by the reference-mode simulators (`topic
//! --reference` and the multiome reference mode).
//!
//! Per synthetic cell `j` and active gene `g`:
//!
//! ```text
//! log m_{g,j} = depth offset                         (the `--depth` rescale)
//!             + a_g · t_{g,j} − ln M1_g              (topic, mean-centred)
//!             + b_g · ε_{g,j} − b_g² / 2             (noise, mean-centred)
//!             + δ_{g, b(j)} − ln E_b[exp δ_{g,b}]    (batch, mean-centred)
//! z*_g        = ρ_gs · z_s + √(1 − ρ_gs²) · e_g
//! y_{g,j}     = F⁻¹_g(Φ(z*_g); rates × m_{g,j})
//! ```
//!
//! `F_g` is the gene's Poisson mixture ([`crate::sim::copula::mixture`]),
//! its rates scaled by the cell's mean multiplier `m_{g,j}`.
//!
//! - Depth enters through the latent: one `z_s ~ N(0, 1)` per cell, and
//!   each gene moves with it as much as it does in the reference (`ρ_gs`).
//! - `t_{g,j}` is the z-scored log topic loading
//!   ([`crate::sim::core::standardized_log_topic`]), or any other per-cell
//!   term ([`CellTerm`], e.g. a trajectory's curves); `M1_g` is the mean of
//!   `exp(a_g · t_{g,j})` over the simulated cells, so no term raises a
//!   gene's mean.
//! - The mixture already holds all of the gene's extra-Poisson variation,
//!   cell types included: the log-variance budget `L_g = ln(1 + cv_g²)`,
//!   `cv_g²` its rates' squared coefficient of variation. The topic term
//!   keeps the dictionary's effects, `a_g = √pve_topic`, unless its
//!   log-variance `a_g² · Var_j t_{g,j}` would exceed `L_g`; then `a_g`
//!   shrinks to fit it exactly. The noise term takes `b_g = √pve_noise`,
//!   capped by what the topic term left, `L_g − ln spread_g` (below). The
//!   mixture keeps the rest: its
//!   rates are tempered (`λ^γ`, rescaled to the same mean) until
//!   `(1 + cv²(γ)) · spread_g = 1 + cv_g²`, with `spread_g = E[T²]/E[T]²` of
//!   the realised multipliers. Every gene keeps the reference's variance; a
//!   gene with no extra-Poisson variation gets no topic term. When even
//!   flat rates leave too little room (a skewed topic term, or many excess
//!   zeros), the variance overshoots.
//! - The batch shift is the effect asked for on top of the reference and
//!   stays outside that budget; it is centred so the mean over batches is
//!   the reference mean.

use crate::sim::copula::marginals::phi;
use crate::sim::copula::mixture::{PoissonMixture, Prepared};
use crate::sim::copula::GlobalCopulaFit;
use crate::sim::core::standardized_log_topic;
use indicatif::ParallelProgressIterator;
use log::info;
use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

/// Cells per block of the moment passes. The block sums are added in block
/// order, so the moments do not depend on the thread count.
const MOMENT_BLOCK: usize = 256;

/// Bisection steps of the spread cap ([`CellPrograms::cap_spread`]): the
/// amplitude to within `2⁻²⁰` of its log-variance cap.
const SPREAD_STEPS: usize = 20;

/// `Var_j t_{g,j}` below which a gene has no topic term.
const MIN_TOPIC_VARIANCE: f64 = 1e-8;

/// The per-cell log-mean term `t_{·,j}` that [`CellPrograms`] scales by
/// `√pve_topic`, capped per gene by its reference dispersion.
pub enum CellTerm<'a> {
    /// z-scored `log(β·θ_j)` ([`standardized_log_topic`]) from gene × topic
    /// loadings (`β`, or `γ ⊙ β` in the multiome) and topic × cell
    /// proportions `θ`.
    Topics {
        dict_dk: &'a DMatrix<f32>,
        theta_kn: &'a DMatrix<f32>,
    },
    /// Any per-cell gene vector, e.g. a trajectory's curves at each cell's
    /// branch and pseudotime ([`crate::sim::trajectory`]).
    PerCell(&'a (dyn Fn(usize) -> DVector<f32> + Sync)),
}

impl CellTerm<'_> {
    fn at(&self, j: usize) -> DVector<f32> {
        match self {
            Self::Topics { dict_dk, theta_kn } => {
                standardized_log_topic(&(*dict_dk * theta_kn.column(j)))
            }
            Self::PerCell(term) => term(j),
        }
    }
}

/// Per-cell programs on top of the reference baseline.
pub struct CellPrograms<'a> {
    pub term: CellTerm<'a>,
    /// Log-variance of the per-cell term (`√pve_topic` scales it), capped
    /// per gene by its reference dispersion.
    pub pve_topic: f32,
    /// Also cap the per-cell term so its multiplier `T = exp(a_g · t)`
    /// spreads no more than the gene's reference rates, `E[T²]/E[T]² ≤
    /// 1 + cv_g²`. The log-variance cap alone bounds that only for a
    /// log-normal `T`; a skewed term then overshoots the reference variance.
    pub cap_spread: bool,
    /// Log-variance of the iid noise term, capped by what the topic term left.
    pub pve_noise: f32,
    /// Per-batch log shift of every gene, already scaled.
    pub batch_delta: &'a [DVector<f32>],
    pub batch_membership: &'a [usize],
}

/// How the sampler scaled the per-cell term: gene `g`'s log-mean term is
/// `amp[g] · t_{g,j} − centre[g]`.
pub struct TermScaling {
    pub amp: Vec<f32>,
    pub centre: Vec<f32>,
}

/// Triplets `(gene, cell, count)` of the non-zero simulated counts.
pub type Triplets = Vec<(u64, u64, f32)>;

/// `ln(depth / Σ_g μ̂_g)`: the log shift of every gene's mean that makes the
/// mean library size `depth`; 0 (the reference's own) when unset.
pub fn depth_log_offset(fit: &GlobalCopulaFit, depth: Option<usize>) -> f32 {
    let lib_ref: f32 = fit.mu_hat.iter().sum::<f32>().max(1e-30);
    match depth {
        Some(d) if d > 0 => ((d as f32) / lib_ref).ln(),
        _ => 0.0,
    }
}

/// Sample one count matrix from a reference fit, with how the per-cell term
/// was scaled. `depth_offset` shifts every gene's log mean
/// ([`depth_log_offset`]); `couple_depth` ties the genes to one depth
/// coordinate per cell as in the reference, instead of giving every cell the
/// same expected depth. Cell `j` uses the seed `rseed + seed_offset + j`.
pub fn sample_reference_counts(
    fit: &GlobalCopulaFit,
    depth_offset: f32,
    programs: &CellPrograms,
    couple_depth: bool,
    rseed: u64,
    seed_offset: u64,
) -> anyhow::Result<(Triplets, TermScaling)> {
    let dd = fit.n_genes;
    let nn = programs.batch_membership.len();
    if let CellTerm::Topics { theta_kn, .. } = &programs.term {
        anyhow::ensure!(
            theta_kn.ncols() == nn,
            "reference sampler: {} cells of topic proportions for {} cells",
            theta_kn.ncols(),
            nn
        );
    }
    anyhow::ensure!(
        programs
            .batch_membership
            .iter()
            .all(|&b| b < programs.batch_delta.len()),
        "reference sampler: a cell's batch has no shift"
    );

    if nn > 0 {
        let len = programs.term.at(0).len();
        anyhow::ensure!(
            len == dd,
            "reference sampler: a per-cell term of {len} genes for {dd} genes"
        );
    }
    let topic = |j: usize| programs.term.at(j);
    let budget = DispersionBudget::new(fit, programs, nn, &topic);
    info!(
        "reference sampler: topic term capped by the reference dispersion for {} genes; \
         {} of {} active genes over their variance; depth {}",
        budget.n_capped,
        budget.n_over,
        fit.active_genes.len(),
        if couple_depth {
            "coupled as in the reference"
        } else {
            "fixed"
        }
    );
    let batch_centre = batch_centre(programs.batch_delta, programs.batch_membership, dd);
    let with_topic = budget.topic_amp.iter().any(|&a| a > 0.0);
    let with_noise = budget.noise_amp.iter().any(|&b| b > 0.0);
    // A gene with no per-cell term has one multiplier per batch: one CDF
    // table per batch, searched per draw. The others walk their mixture.
    let fixed = |g: usize| budget.topic_amp[g] == 0.0 && budget.noise_amp[g] == 0.0;
    let tables: Vec<Vec<Vec<f64>>> = programs
        .batch_delta
        .iter()
        .map(|shift| {
            let mut per_gene = vec![Vec::new(); dd];
            let built: Vec<(usize, Vec<f64>)> = fit
                .active_genes
                .par_iter()
                .filter(|&&g| fixed(g))
                .map(|&g| {
                    let m = (depth_offset + shift[g] - batch_centre[g]).exp() as f64;
                    (g, budget.marginal[g].cdf_table(m))
                })
                .collect();
            for (g, t) in built {
                per_gene[g] = t;
            }
            per_gene
        })
        .collect();

    let triplets: Triplets = (0..nn)
        .into_par_iter()
        .progress_count(nn as u64)
        .map(|j| -> Triplets {
            let mut rng = rand::rngs::StdRng::seed_from_u64(
                rseed.wrapping_add(seed_offset).wrapping_add(j as u64),
            );
            let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
            let z_depth: f32 = if couple_depth {
                normal.sample(&mut rng)
            } else {
                0.0
            };
            let t = with_topic.then(|| topic(j));
            let b = programs.batch_membership[j];
            let shift = &programs.batch_delta[b];
            let z_hvg = fit.copula.sample(&mut rng);

            let mut counts = Triplets::with_capacity(fit.active_genes.len() / 8);
            for &g in &fit.active_genes {
                let mut ln_m = depth_offset + shift[g] - batch_centre[g];
                if let Some(t) = t.as_ref() {
                    ln_m += budget.topic_amp[g] * t[g] - budget.ln_m1[g];
                }
                if with_noise {
                    let a = budget.noise_amp[g];
                    ln_m += a * normal.sample(&mut rng) - 0.5 * a * a;
                }
                let e = match fit.hvg_pos[g] {
                    Some(h) => z_hvg[h as usize],
                    None => normal.sample(&mut rng),
                };
                let rho = if couple_depth { fit.depth_rho[g] } else { 0.0 };
                let z = rho * z_depth + (1.0 - rho * rho).max(0.0).sqrt() * e;
                let u = phi(z as f64).clamp(1e-7, 1.0 - 1e-7);
                let x = if fixed(g) {
                    tables[b][g].partition_point(|&f| f < u) as u32
                } else {
                    let m = (ln_m as f64).exp();
                    if !m.is_finite() || m <= 0.0 {
                        continue;
                    }
                    budget.marginal[g].inverse_cdf(u, m)
                };
                if x > 0 {
                    counts.push((g as u64, j as u64, x as f32));
                }
            }
            counts
        })
        .flatten()
        .collect();
    let scaling = TermScaling {
        amp: budget.topic_amp,
        centre: budget.ln_m1,
    };
    Ok((triplets, scaling))
}

/// `ln E_b[exp δ_{g,b}]` over the cells' batches: subtracting it keeps a
/// gene's mean over all cells at the reference mean.
pub(crate) fn batch_centre(delta: &[DVector<f32>], membership: &[usize], dd: usize) -> Vec<f32> {
    if membership.is_empty() || delta.is_empty() {
        return vec![0.0; dd];
    }
    let mut counts = vec![0usize; delta.len()];
    for &b in membership {
        counts[b] += 1;
    }
    let n = membership.len() as f64;
    (0..dd)
        .map(|g| {
            let mean: f64 = delta
                .iter()
                .zip(&counts)
                .map(|(d, &c)| c as f64 / n * (d[g] as f64).exp())
                .sum();
            mean.ln() as f32
        })
        .collect()
}

/// Per-gene topic and noise amplitudes, the topic centring `ln M1_g`, and
/// the tempered mixture left for the draw.
struct DispersionBudget {
    n_capped: usize,
    /// Active genes whose mixture could not make room for the other terms.
    n_over: usize,
    topic_amp: Vec<f32>,
    noise_amp: Vec<f32>,
    ln_m1: Vec<f32>,
    marginal: Vec<Prepared>,
}

impl DispersionBudget {
    fn new(
        fit: &GlobalCopulaFit,
        programs: &CellPrograms,
        nn: usize,
        topic: &(impl Fn(usize) -> DVector<f32> + Sync),
    ) -> Self {
        let dd = fit.n_genes;
        let genes = &fit.active_genes;
        let alpha_topic = (programs.pve_topic.max(0.0) as f64).sqrt();
        let alpha_noise = (programs.pve_noise.max(0.0) as f64).sqrt();
        let budget: Vec<f64> = fit.marginals.iter().map(|m| m.rate_cv2().ln_1p()).collect();

        let mut topic_amp = vec![0.0_f32; dd];
        let mut ln_m1 = vec![0.0_f32; dd];
        let mut spread = vec![1.0_f64; dd];
        let mut n_capped = 0;
        if alpha_topic > 0.0 && nn > 0 {
            // Moments over the active genes, in their order.
            let (mean_t, mean_t2) = first_two_moments(genes.len(), nn, |j, out| {
                let t = topic(j);
                for (o, &g) in out.iter_mut().zip(genes) {
                    *o = t[g] as f64;
                }
            });
            for (i, &g) in genes.iter().enumerate() {
                let v = mean_t2[i] - mean_t[i] * mean_t[i];
                if v <= MIN_TOPIC_VARIANCE {
                    continue;
                }
                topic_amp[g] = alpha_topic.min((budget[g] / v).sqrt()) as f32;
            }
            // `E[T]` and `E[T²]` of `T = exp(amp_g · t_g)` over `idx`.
            let exp_moments = |amp: &[f32], idx: &[usize]| {
                first_two_moments(idx.len(), nn, |j, out| {
                    let t = topic(j);
                    for (o, &g) in out.iter_mut().zip(idx) {
                        *o = ((amp[g] * t[g]) as f64).exp();
                    }
                })
            };
            let (mut m1, mut m2) = exp_moments(&topic_amp, genes);
            if programs.cap_spread {
                // The spread grows with the amplitude from 1 at 0: bisect
                // each gene over it, all genes in one pass per step.
                let fits = |g: usize, a: f64, b: f64| {
                    b / (a * a).max(f64::MIN_POSITIVE) <= budget[g].exp()
                };
                let over: Vec<usize> = genes
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &g)| (!fits(g, m1[i], m2[i])).then_some(g))
                    .collect();
                if !over.is_empty() {
                    let mut lo = vec![0.0_f32; dd];
                    let mut hi = topic_amp.clone();
                    let mut trial = topic_amp.clone();
                    for _ in 0..SPREAD_STEPS {
                        for &g in &over {
                            trial[g] = 0.5 * (lo[g] + hi[g]);
                        }
                        let (a, b) = exp_moments(&trial, &over);
                        for (i, &g) in over.iter().enumerate() {
                            if fits(g, a[i], b[i]) {
                                lo[g] = trial[g];
                            } else {
                                hi[g] = trial[g];
                            }
                        }
                    }
                    for &g in &over {
                        topic_amp[g] = lo[g];
                    }
                    (m1, m2) = exp_moments(&topic_amp, genes);
                }
            }
            n_capped = genes
                .iter()
                .filter(|&&g| {
                    let a = topic_amp[g] as f64;
                    a > 0.0 && a < alpha_topic * (1.0 - 1e-6)
                })
                .count();
            for (i, &g) in genes.iter().enumerate() {
                ln_m1[g] = m1[i].max(f64::MIN_POSITIVE).ln() as f32;
                spread[g] = m2[i] / (m1[i] * m1[i]).max(f64::MIN_POSITIVE);
            }
        }
        let mut noise_amp = vec![0.0_f32; dd];
        for &g in genes {
            // What the topic term left: its multiplier's log spread, which
            // exceeds its log-variance when the term is skewed.
            let left = budget[g] - spread[g].ln();
            noise_amp[g] = alpha_noise.min(left.max(0.0).sqrt()) as f32;
        }
        let left: Vec<(PoissonMixture, bool)> = (0..dd)
            .into_par_iter()
            .map(|g| {
                if noise_amp[g] == 0.0 && spread[g] <= 1.0 {
                    return (fit.marginals[g].clone(), true);
                }
                let b = noise_amp[g] as f64;
                fit.marginals[g].leaving_room_for(spread[g] * (b * b).exp())
            })
            .collect();
        let n_over = genes.iter().filter(|&&g| !left[g].1).count();
        Self {
            n_capped,
            n_over,
            topic_amp,
            noise_amp,
            ln_m1,
            marginal: left.into_iter().map(|(m, _)| m.prepare()).collect(),
        }
    }
}

/// Per entry, the mean over cells of `x` and `x²`, with `fill(j, out)`
/// writing cell `j`'s `n` values; summed in fixed blocks so the result does
/// not depend on the thread schedule.
pub(crate) fn first_two_moments(
    n: usize,
    nn: usize,
    fill: impl Fn(usize, &mut [f64]) + Sync,
) -> (Vec<f64>, Vec<f64>) {
    let blocks: Vec<(Vec<f64>, Vec<f64>)> = (0..nn.div_ceil(MOMENT_BLOCK))
        .into_par_iter()
        .map(|b| {
            let (mut s1, mut s2, mut buf) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            for j in (b * MOMENT_BLOCK)..((b + 1) * MOMENT_BLOCK).min(nn) {
                fill(j, &mut buf);
                for ((a, c), &x) in s1.iter_mut().zip(s2.iter_mut()).zip(&buf) {
                    *a += x;
                    *c += x * x;
                }
            }
            (s1, s2)
        })
        .collect();
    let (mut m1, mut m2) = (vec![0.0_f64; n], vec![0.0_f64; n]);
    for (s1, s2) in blocks {
        for i in 0..n {
            m1[i] += s1[i];
            m2[i] += s2[i];
        }
    }
    let k = nn.max(1) as f64;
    m1.iter_mut().for_each(|x| *x /= k);
    m2.iter_mut().for_each(|x| *x /= k);
    (m1, m2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_centre_is_the_log_mean_shift() {
        let delta = vec![
            DVector::from_vec(vec![0.5_f32, 0.0]),
            DVector::from_vec(vec![-0.5, 0.0]),
        ];
        let c = batch_centre(&delta, &[0, 0, 0, 1], 2);
        let want = (0.75 * 0.5_f64.exp() + 0.25 * (-0.5_f64).exp()).ln();
        assert!((c[0] as f64 - want).abs() < 1e-6);
        assert_eq!(c[1], 0.0);
    }

    #[test]
    fn moments_are_thread_independent_block_sums() {
        let (m1, m2) = first_two_moments(2, 1000, |j, out| {
            out[0] = j as f64;
            out[1] = 1.0;
        });
        assert!((m1[0] - 499.5).abs() < 1e-9);
        assert!((m1[1] - 1.0).abs() < 1e-12 && (m2[1] - 1.0).abs() < 1e-12);
    }
}
