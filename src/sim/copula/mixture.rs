//! A gene's count distribution as a Poisson mixture
//! `P(x) = Σ_k w_k · Poisson(x; λ_k)`, with the mixing weights `w` the
//! nonparametric maximum-likelihood estimate on a fixed grid of rates
//! (Kiefer & Wolfowitz 1956; fitted by EM as in Laird 1978).
//!
//! A pooled negative binomial matches a gene's mean and variance but not
//! its shape: for a gene high in one population and low elsewhere (TMSB4X,
//! ACTB, platelet and erythroid genes), the variance forces a tiny NB size,
//! and a tiny size piles mass at zero — 58% zeros where the reference has
//! 1.4%. The mixture fits the whole count distribution (a rate-0 atom takes
//! any excess zeros), and a change of mean — topics, batches, `--depth` —
//! scales its rates.

/// Log-spaced rates on the grid, besides the rate-0 atom.
const GRID_RATES: usize = 50;

/// EM steps at most.
const MAX_EM_STEPS: usize = 1000;

/// Relative log-likelihood gain below which EM stops.
const EM_TOLERANCE: f64 = 1e-9;

/// Atoms lighter than this are dropped after the fit.
const MIN_WEIGHT: f64 = 1e-8;

/// Atoms of a fitted mixture at most: the grid and the rate-0 atom.
const MAX_ATOMS: usize = GRID_RATES + 1;

/// Tail mass at which a CDF walk or table stops. The sampler never asks
/// for a quantile above `1 − 1e-7`, so a walk always ends on its count.
const TAIL_MASS: f64 = 1e-9;

/// Counts walked at most, whatever the tail: a guard, never reached by a
/// mixture fitted to counts.
const MAX_WALK: usize = 1 << 24;

/// A Poisson mixture: atoms `rates[k]` with weights `weights[k]` summing to 1.
#[derive(Debug, Clone, PartialEq)]
pub struct PoissonMixture {
    pub rates: Vec<f64>,
    pub weights: Vec<f64>,
}

impl PoissonMixture {
    /// All mass at zero.
    pub fn zero() -> Self {
        Self {
            rates: vec![0.0],
            weights: vec![1.0],
        }
    }

    /// The maximum-likelihood mixture of the counts with histogram `hist`
    /// (`hist[x]` cells with count `x`).
    pub fn fit(hist: &[f64]) -> Self {
        let n: f64 = hist.iter().sum();
        let observed: Vec<(usize, f64)> = hist
            .iter()
            .enumerate()
            .filter(|(_, &h)| h > 0.0)
            .map(|(x, &h)| (x, h))
            .collect();
        let x_max = observed.last().map_or(0, |&(x, _)| x);
        if n <= 0.0 || x_max == 0 {
            return Self::zero();
        }
        let mean = observed.iter().map(|&(x, h)| x as f64 * h).sum::<f64>() / n;
        let lo = (mean / 100.0).min(0.01);
        let hi = x_max as f64 * 1.2;
        let step = (hi / lo).ln() / (GRID_RATES - 1) as f64;
        let rates: Vec<f64> = std::iter::once(0.0)
            .chain((0..GRID_RATES).map(|k| lo * (step * k as f64).exp()))
            .collect();

        // Likelihood of each observed count under each atom, count-major
        // so the mixture and the responsibilities sweep it in one order.
        let k_atoms = rates.len();
        let ln_fact = ln_factorials(x_max);
        let like: Vec<f64> = observed
            .iter()
            .flat_map(|&(x, _)| rates.iter().map(move |&lam| (x, lam)))
            .map(|(x, lam)| poisson_pmf(x, lam, &ln_fact))
            .collect();
        let mut weights = vec![1.0 / k_atoms as f64; k_atoms];
        let mut resp = vec![0.0_f64; k_atoms];
        let mut last = f64::NEG_INFINITY;
        for _ in 0..MAX_EM_STEPS {
            resp.iter_mut().for_each(|r| *r = 0.0);
            let mut ll = 0.0_f64;
            for (row, &(_, h)) in like.chunks_exact(k_atoms).zip(&observed) {
                let f: f64 = row
                    .iter()
                    .zip(&weights)
                    .map(|(p, w)| p * w)
                    .sum::<f64>()
                    .max(f64::MIN_POSITIVE);
                ll += h * f.ln();
                let ratio = h / f;
                for (r, p) in resp.iter_mut().zip(row) {
                    *r += ratio * p;
                }
            }
            for (w, r) in weights.iter_mut().zip(&resp) {
                *w *= r / n;
            }
            if ll - last < EM_TOLERANCE * ll.abs() {
                break;
            }
            last = ll;
        }
        let (rates, weights): (Vec<f64>, Vec<f64>) = rates
            .into_iter()
            .zip(weights)
            .filter(|&(_, w)| w >= MIN_WEIGHT)
            .unzip();
        let total: f64 = weights.iter().sum();
        Self {
            rates,
            weights: weights.into_iter().map(|w| w / total).collect(),
        }
    }

    pub fn mean(&self) -> f64 {
        self.rates
            .iter()
            .zip(&self.weights)
            .map(|(r, w)| r * w)
            .sum()
    }

    /// Squared coefficient of variation of the rates: the mixture's
    /// extra-Poisson variance over its squared mean (`1/r` of an NB).
    pub fn rate_cv2(&self) -> f64 {
        let m = self.mean();
        if m <= 0.0 {
            return 0.0;
        }
        let m2: f64 = self
            .rates
            .iter()
            .zip(&self.weights)
            .map(|(r, w)| r * r * w)
            .sum();
        (m2 / (m * m) - 1.0).max(0.0)
    }

    pub fn variance(&self) -> f64 {
        let m = self.mean();
        m + m * m * self.rate_cv2()
    }

    pub fn zero_fraction(&self) -> f64 {
        self.rates
            .iter()
            .zip(&self.weights)
            .map(|(r, w)| w * (-r).exp())
            .sum()
    }

    /// The rates raised to `gamma` and rescaled to the same mean: `gamma = 1`
    /// is this mixture, `gamma = 0` puts every non-zero atom at one rate.
    /// The rate-0 atom stays at zero.
    pub fn tempered(&self, gamma: f64) -> Self {
        let m = self.mean();
        let raw: Vec<f64> = self
            .rates
            .iter()
            .map(|&r| if r > 0.0 { r.powf(gamma) } else { 0.0 })
            .collect();
        let raw_mean: f64 = raw.iter().zip(&self.weights).map(|(r, w)| r * w).sum();
        let scale = if raw_mean > 0.0 { m / raw_mean } else { 0.0 };
        Self {
            rates: raw.into_iter().map(|r| r * scale).collect(),
            weights: self.weights.clone(),
        }
    }

    /// `tempered` to the largest `gamma` whose rate spread, multiplied by
    /// `spread ≥ 1` from the other terms, is no more than this mixture's:
    /// `(1 + cv²(γ))·spread ≤ 1 + cv²(1)`. When even `gamma = 0` exceeds it,
    /// that one, and `false`.
    pub fn leaving_room_for(&self, spread: f64) -> (Self, bool) {
        if spread <= 1.0 + 1e-12 {
            return (self.clone(), true);
        }
        let target = (1.0 + self.rate_cv2()) / spread;
        let fits = |g: f64| 1.0 + self.tempered(g).rate_cv2() <= target;
        if !fits(0.0) {
            return (self.tempered(0.0), false);
        }
        let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if fits(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (self.tempered(lo), true)
    }

    /// The atoms in the form a draw needs, computed once.
    pub fn prepare(&self) -> Prepared {
        Prepared {
            ln_weights: self.weights.iter().map(|w| w.ln()).collect(),
            ln_rates: self
                .rates
                .iter()
                .map(|&r| if r > 0.0 { r.ln() } else { f64::NEG_INFINITY })
                .collect(),
            rates: self.rates.clone(),
        }
    }
}

/// A mixture's atoms as logarithms, so a draw at any scale needs no
/// allocation and no logarithm of its own besides `ln scale`.
#[derive(Debug, Clone)]
pub struct Prepared {
    ln_weights: Vec<f64>,
    ln_rates: Vec<f64>,
    rates: Vec<f64>,
}

impl Prepared {
    /// `F(0), F(1), …` with the rates scaled by `scale`, until the tail
    /// mass is below [`TAIL_MASS`].
    pub fn cdf_table(&self, scale: f64) -> Vec<f64> {
        let mut table = Vec::new();
        self.walk(scale, |cdf| {
            table.push(cdf);
            cdf >= 1.0 - TAIL_MASS
        });
        table
    }

    /// `min { x : F(x) ≥ u }` with the rates scaled by `scale`.
    pub fn inverse_cdf(&self, u: f64, scale: f64) -> u32 {
        if u <= 0.0 {
            return 0;
        }
        self.walk(scale, |cdf| cdf >= u.min(1.0 - TAIL_MASS)) as u32
    }

    /// Walk `F(0), F(1), …` until `stop(F(x))`; that `x`. Each atom's term
    /// is kept as a logarithm, `ln w_k + x·ln λ_k − λ_k − ln x!`, so an atom
    /// of rate in the thousands does not underflow at `x = 0`.
    fn walk(&self, scale: f64, mut stop: impl FnMut(f64) -> bool) -> usize {
        let k = self.rates.len().min(MAX_ATOMS);
        let ln_scale = scale.ln();
        let mut ln_terms = [f64::NEG_INFINITY; MAX_ATOMS];
        let mut ln_rates = [f64::NEG_INFINITY; MAX_ATOMS];
        for i in 0..k {
            ln_terms[i] = self.ln_weights[i] - self.rates[i] * scale;
            ln_rates[i] = self.ln_rates[i] + ln_scale;
        }
        let mut cdf = 0.0_f64;
        for x in 0..MAX_WALK {
            cdf += ln_terms[..k].iter().map(|t| t.exp()).sum::<f64>();
            if stop(cdf.min(1.0)) {
                return x;
            }
            let ln_next = ((x + 1) as f64).ln();
            for (t, &lr) in ln_terms[..k].iter_mut().zip(&ln_rates[..k]) {
                *t += lr - ln_next;
            }
        }
        MAX_WALK
    }
}

/// `ln x!` for `x = 0..=x_max`.
fn ln_factorials(x_max: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(x_max + 1);
    let mut acc = 0.0_f64;
    out.push(0.0);
    for i in 1..=x_max {
        acc += (i as f64).ln();
        out.push(acc);
    }
    out
}

fn poisson_pmf(x: usize, lam: f64, ln_fact: &[f64]) -> f64 {
    if lam <= 0.0 {
        return if x == 0 { 1.0 } else { 0.0 };
    }
    (x as f64 * lam.ln() - lam - ln_fact[x]).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A two-population gene with 1% zeros, like TMSB4X: most cells low,
    /// a fifth very high. The fit keeps the mean, the zero fraction and
    /// (nearly) the variance; an NB with the same mean and variance puts
    /// a third of the cells at zero.
    #[test]
    fn a_bimodal_gene_keeps_its_zeros_and_moments() {
        let mut hist = vec![0.0; 300];
        hist[0] = 10.0;
        for h in hist.iter_mut().take(9).skip(3) {
            *h = 790.0 / 6.0;
        }
        for h in hist.iter_mut().take(261).skip(200) {
            *h = 200.0 / 61.0;
        }
        let n: f64 = hist.iter().sum();
        let mean = hist
            .iter()
            .enumerate()
            .map(|(x, h)| x as f64 * h)
            .sum::<f64>()
            / n;
        let ex2 = hist
            .iter()
            .enumerate()
            .map(|(x, h)| (x * x) as f64 * h)
            .sum::<f64>()
            / n;
        let var = ex2 - mean * mean;
        let zero = hist[0] / n;

        let fit = PoissonMixture::fit(&hist);
        assert!(
            (fit.mean() / mean - 1.0).abs() < 0.01,
            "mean {}",
            fit.mean()
        );
        assert!(
            (fit.zero_fraction() - zero).abs() < 0.01,
            "zero {}",
            fit.zero_fraction()
        );
        assert!(
            (fit.variance() / var - 1.0).abs() < 0.05,
            "var {} vs {var}",
            fit.variance()
        );
        let r = mean * mean / (var - mean);
        let nb_zero = (r / (r + mean)).powf(r);
        assert!(nb_zero - zero > 0.1, "the NB's zeros {nb_zero} vs {zero}");
    }

    /// A light atom far beyond the mixture's SD is still reached: walks
    /// stop on tail mass, not on a moment bound.
    #[test]
    fn walks_reach_a_light_atom_far_out() {
        let fit = PoissonMixture {
            rates: vec![0.0, 0.5, 8000.0],
            weights: vec![0.9, 0.0999, 1e-4],
        };
        let p = fit.prepare();
        assert!(p.inverse_cdf(1.0 - 1e-5, 1.0) > 7000);
        assert!(*p.cdf_table(1.0).last().unwrap() >= 1.0 - TAIL_MASS);
    }

    #[test]
    fn inverse_cdf_matches_the_table() {
        let fit = PoissonMixture {
            rates: vec![0.0, 0.5, 12.0],
            weights: vec![0.3, 0.5, 0.2],
        };
        let fit = fit.prepare();
        for scale in [0.5, 1.0, 3.0] {
            let table = fit.cdf_table(scale);
            assert!((table.last().unwrap() - 1.0).abs() < 1e-9);
            for u in [1e-7, 0.2, 0.31, 0.6, 0.9, 0.999] {
                let from_table = table.iter().position(|&f| f >= u).unwrap() as u32;
                assert_eq!(fit.inverse_cdf(u, scale), from_table, "u {u} scale {scale}");
            }
        }
        assert_eq!(fit.inverse_cdf(0.29, 1.0), 0);
    }

    #[test]
    fn tempering_keeps_the_mean_and_makes_room() {
        let fit = PoissonMixture {
            rates: vec![0.0, 1.0, 10.0],
            weights: vec![0.2, 0.6, 0.2],
        };
        let flat = fit.tempered(0.0);
        assert!((flat.mean() - fit.mean()).abs() < 1e-12);
        assert!(flat.rate_cv2() < fit.rate_cv2());
        let (room, ok) = fit.leaving_room_for(1.5);
        assert!(ok);
        let total = (1.0 + room.rate_cv2()) * 1.5;
        assert!((total - (1.0 + fit.rate_cv2())).abs() < 1e-6);
        // More spread than the zero atom allows: flat, and flagged.
        let (_, ok) = fit.leaving_room_for(100.0);
        assert!(!ok);
        assert_eq!(PoissonMixture::fit(&[5.0]), PoissonMixture::zero());
    }
}
