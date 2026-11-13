//! Latent Gaussian correlations from sparse counts.
//!
//! The copula assumes each gene's count is a step function of a standard
//! normal latent `z`: count `x` is observed when `z` falls in
//! `(a_x, b_x] = (Φ⁻¹F(x−1), Φ⁻¹F(x)]` of the gene's marginal `F`. The plain
//! Pearson correlation of any one score per count (the PIT mid-point, say)
//! underestimates the latent correlation, badly for sparse genes, where one
//! score stands for the whole zero mass.
//!
//! The thresholds come from each gene's empirical count distribution in the
//! reference, not from its fitted NB: a pooled NB that matches a marker
//! gene's mean and variance can still miss its detection rate by half, and
//! thresholds from the wrong marginal turn ordinary co-detection into a
//! near-perfect latent correlation. With the empirical marginal the estimate
//! is the rank-based one of a Gaussian copula with discrete margins.
//!
//! This module scores each count by its conditional mean
//! `m(x) = E[z | a_x < z ≤ b_x] = (φ(a_x) − φ(b_x)) / P(x)` and undoes the
//! attenuation with the Hermite expansion of a bivariate normal: for two
//! genes with latent correlation `ρ`,
//!
//! ```text
//! E[m_h m_k] = Σ_{q≥1} ρ^q · c_q(h) · c_q(k) / q!,
//! c_q(g)     = E[m_g(z) He_q(z)] = Σ_x m_x · (He_{q−1}(a_x) φ(a_x) − He_{q−1}(b_x) φ(b_x)),
//! ```
//!
//! with `He_q` the probabilists' Hermite polynomials. Truncated at
//! [`HERMITE_ORDER`] terms, this is solved for `ρ` pair by pair. Against a
//! library-size coordinate `z_s` that is itself standard normal only the
//! first term survives, so `ρ_gs = E[m_g z_s] / c_1(g)` exactly.
//!
//! Every estimate is shrunk toward zero by its sampling variance, about
//! `1 / (N · c_1(h) · c_1(k))` at `ρ = 0`, so that pairs of sparse genes,
//! whose correlation the reference barely identifies, do not hand the
//! factor model noise.

use crate::sim::copula::marginals::{inv_phi, std_normal_pdf};
use crate::sim::copula::reference::library_coordinate;
use nalgebra::DMatrix;
use nalgebra_sparse::CscMatrix;
use rayon::prelude::*;

/// Hermite terms kept in the pairwise moment equation.
pub const HERMITE_ORDER: usize = 4;

/// Largest absolute latent correlation an estimate may take.
pub const MAX_ABS_CORRELATION: f64 = 0.99;

/// Points scanned to find the monotone segment of the moment equation.
const SEGMENT_GRID: usize = 99;

/// Bisection steps of the moment equation.
const BISECTION_STEPS: usize = 50;

/// Conditional-mean scores of one marginal and its Hermite coefficients.
#[derive(Debug, Clone)]
pub struct LatentScores {
    /// `m(x)` for `x = 0, 1, …`; counts past the end score as the last entry.
    pub m: Vec<f64>,
    /// `c_q` for `q = 1..=HERMITE_ORDER`; `c_1 = E[m²]` is the share of the
    /// latent variance one count carries.
    pub c: [f64; HERMITE_ORDER],
}

impl LatentScores {
    /// Score every count of the distribution `pmf` (count `x` at index `x`,
    /// any positive scale). The coefficients use the whole support.
    pub fn from_pmf(pmf: &[f64]) -> Self {
        // Survival function from the far end, so upper thresholds keep their
        // precision where F(x) is within rounding of 1.
        let mut surv = vec![0.0_f64; pmf.len()];
        let mut acc = 0.0_f64;
        for x in (0..pmf.len()).rev() {
            surv[x] = acc;
            acc += pmf[x];
        }
        let total = acc.max(f64::MIN_POSITIVE);
        let mut m = Vec::with_capacity(pmf.len());
        let mut c = [0.0_f64; HERMITE_ORDER];
        let mut cdf = 0.0_f64;
        let mut lower = f64::NEG_INFINITY;
        for (x, &p) in pmf.iter().enumerate() {
            let p = p / total;
            cdf += p;
            let upper = if x + 1 == pmf.len() {
                f64::INFINITY
            } else {
                threshold(cdf, surv[x] / total)
            };
            if p <= 0.0 {
                // A count never seen: score it at the threshold it sits on;
                // it carries no mass, so no coefficient.
                m.push(if upper.is_finite() { upper } else { lower });
                continue;
            }
            let score = conditional_mean(lower, upper, p);
            m.push(score);
            let (hl, hu) = (hermite_densities(lower), hermite_densities(upper));
            for q in 0..HERMITE_ORDER {
                c[q] += score * (hl[q] - hu[q]);
            }
            lower = upper;
        }
        Self { m, c }
    }

    /// Score of count `x`.
    pub fn score(&self, x: usize) -> f64 {
        self.m[x.min(self.m.len().saturating_sub(1))]
    }
}

/// `Φ⁻¹` of a CDF value, from the survival side where that is the more
/// precise of the two.
fn threshold(cdf: f64, surv: f64) -> f64 {
    const EPS: f64 = 1e-300;
    if cdf <= 0.5 {
        inv_phi(cdf.max(EPS))
    } else {
        -inv_phi(surv.max(EPS))
    }
}

/// `E[z | a < z ≤ b]` for `P(a < z ≤ b) = p`. Written as
/// `φ(a)·(1 − e^{−(b−a)(b+a)/2}) / p` (or the mirror for `b < 0`), so a
/// narrow interval far in the tail keeps its precision; an interval too
/// narrow to resolve scores at its midpoint.
fn conditional_mean(a: f64, b: f64, p: f64) -> f64 {
    let diff = if a.is_finite() && b.is_finite() {
        if a >= 0.0 {
            std_normal_pdf(a) * -(-(b - a) * (b + a) / 2.0).exp_m1()
        } else if b <= 0.0 {
            -std_normal_pdf(b) * -(-(a - b) * (a + b) / 2.0).exp_m1()
        } else {
            std_normal_pdf(a) - std_normal_pdf(b)
        }
    } else {
        std_normal_pdf(a) - std_normal_pdf(b)
    };
    let m = diff / p;
    if m.is_finite() {
        m
    } else if a.is_finite() && b.is_finite() {
        0.5 * (a + b)
    } else if a.is_finite() {
        a
    } else {
        b
    }
}

/// `He_{q−1}(t)·φ(t)` for `q = 1..=HERMITE_ORDER`; zero at `±∞`.
fn hermite_densities(t: f64) -> [f64; HERMITE_ORDER] {
    let mut out = [0.0_f64; HERMITE_ORDER];
    if !t.is_finite() {
        return out;
    }
    let pdf = std_normal_pdf(t);
    let (mut h_prev, mut h) = (0.0_f64, 1.0_f64);
    for (q, o) in out.iter_mut().enumerate() {
        *o = h * pdf;
        // He_{q+1}(t) = t·He_q(t) − q·He_{q−1}(t).
        let next = t * h - q as f64 * h_prev;
        h_prev = h;
        h = next;
    }
    out
}

/// `Σ_q ρ^q a_q / q!` and its derivative in `ρ`, with `a_q = c_q(h) c_q(k)`.
fn moment(rho: f64, a: &[f64; HERMITE_ORDER]) -> (f64, f64) {
    let (mut f, mut df) = (0.0_f64, 0.0_f64);
    let (mut pow, mut fact) = (1.0_f64, 1.0_f64);
    for (q, &aq) in a.iter().enumerate() {
        let qf = (q + 1) as f64;
        df += aq * pow / fact; // ρ^q / q! · q, with pow = ρ^q, fact = q!
        fact *= qf;
        pow *= rho;
        f += aq * pow / fact;
    }
    (f, df)
}

/// Solve `Σ_q ρ^q a_q / q! = target` for `ρ` on the segment around 0 where
/// the left side increases, by bisection; a target outside the segment's
/// range takes the nearer end.
pub fn solve_moment(target: f64, a: &[f64; HERMITE_ORDER]) -> f64 {
    if a[0] <= 0.0 {
        return 0.0;
    }
    let step = MAX_ABS_CORRELATION / SEGMENT_GRID as f64;
    let edge = |dir: f64| {
        let mut rho = 0.0_f64;
        for _ in 0..SEGMENT_GRID {
            let next = rho + dir * step;
            if moment(next, a).1 <= 0.0 {
                break;
            }
            rho = next;
        }
        rho
    };
    let (mut lo, mut hi) = (edge(-1.0), edge(1.0));
    if target <= moment(lo, a).0 {
        return lo;
    }
    if target >= moment(hi, a).0 {
        return hi;
    }
    for _ in 0..BISECTION_STEPS {
        let mid = 0.5 * (lo + hi);
        if moment(mid, a).0 < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Shrink an estimate toward zero by its sampling variance `se2`:
/// `ρ · ρ² / (ρ² + se2)`.
pub fn shrink(rho: f64, se2: f64) -> f64 {
    let r2 = rho * rho;
    if r2 + se2 <= 0.0 {
        0.0
    } else {
        rho * r2 / (r2 + se2)
    }
}

/// Rows of the Gram matrix computed together.
const GRAM_BLOCK: usize = 64;

/// Latent correlations among `scores.len()` genes from their centred
/// conditional-mean scores `m` (cells × genes, so a block of genes is a
/// block of columns and its Gram rows need no transposed copy), as a dense
/// matrix with unit diagonal, each entry shrunk by its sampling variance.
pub fn pairwise_correlation(m: &DMatrix<f32>, scores: &[LatentScores]) -> DMatrix<f64> {
    let h = m.ncols();
    let n = m.nrows().max(1) as f64;
    // Upper triangle, one block of rows at a time.
    let upper: Vec<(usize, Vec<f64>)> = (0..h.div_ceil(GRAM_BLOCK))
        .into_par_iter()
        .flat_map_iter(|blk| {
            let i0 = blk * GRAM_BLOCK;
            let len = GRAM_BLOCK.min(h - i0);
            // Rows i0.. against columns i0..: only k > i is read.
            let gram = m.columns(i0, len).tr_mul(&m.columns(i0, h - i0));
            (0..len)
                .map(|di| {
                    let i = i0 + di;
                    let row = ((i + 1)..h)
                        .map(|k| {
                            let a: [f64; HERMITE_ORDER] =
                                std::array::from_fn(|q| scores[i].c[q] * scores[k].c[q]);
                            let rho = solve_moment(gram[(di, k - i0)] as f64 / n, &a);
                            let c11 = (scores[i].c[0] * scores[k].c[0]).max(f64::MIN_POSITIVE);
                            shrink(rho, 1.0 / (n * c11))
                        })
                        .collect();
                    (i, row)
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let mut out = DMatrix::<f64>::identity(h, h);
    for (i, row) in upper {
        for (dk, rho) in row.into_iter().enumerate() {
            let k = i + 1 + dk;
            out[(i, k)] = rho;
            out[(k, i)] = rho;
        }
    }
    out
}

/// Cells per block of the depth-coupling pass; block sums are added in
/// block order, so the result does not depend on the thread count.
const CELL_BLOCK: usize = 1024;

/// Latent structure of a reference: every active gene's coupling to the
/// library-size coordinate, and the HVGs' correlation beyond it.
pub struct ReferenceLatent {
    /// `ρ_gs` per gene; 0 for genes outside `active`.
    pub depth_rho: Vec<f32>,
    /// Latent correlation of the HVGs given the library coordinate:
    /// `(ρ_hk − ρ_hs ρ_ks) / √((1 − ρ_hs²)(1 − ρ_ks²))`, unit diagonal.
    pub hvg_residual: DMatrix<f64>,
}

impl ReferenceLatent {
    /// Fit from the reference counts `mat` (genes × cells; values are taken
    /// as whole counts) and their per-gene histograms `hist`
    /// ([`gene_histograms`]). `libs` are the cells' library sizes, `active`
    /// the genes to couple to depth, `hvg` the genes of the copula; both must
    /// have a histogram.
    pub fn fit(
        mat: &CscMatrix<f32>,
        libs: &[f64],
        hist: &[Option<Vec<f64>>],
        hvg: &[usize],
    ) -> Self {
        let n_genes = mat.nrows();
        let n = libs.len().max(1) as f64;
        let scores: Vec<Option<LatentScores>> = hist
            .par_iter()
            .map(|h| h.as_deref().map(LatentScores::from_pmf))
            .collect();

        let mut z = library_coordinate(libs);
        let z_mean = z.iter().sum::<f64>() / n;
        z.iter_mut().for_each(|v| *v -= z_mean);

        // Σ_c (m_g(x_gc) − m̄_g) z_c over the non-zeros only: Σ_c z_c = 0, so
        // the zero cells and the centring drop out.
        let cols = mat.ncols().min(libs.len());
        let blocks: Vec<Vec<f64>> = (0..cols.div_ceil(CELL_BLOCK))
            .into_par_iter()
            .map(|b| {
                let mut acc = vec![0.0_f64; n_genes];
                let start = b * CELL_BLOCK;
                let end = (start + CELL_BLOCK).min(cols);
                for (c, &zc) in z.iter().enumerate().take(end).skip(start) {
                    let col = mat.col(c);
                    for (&g, &v) in col.row_indices().iter().zip(col.values()) {
                        if let Some(sc) = &scores[g] {
                            acc[g] += (sc.score(v as usize) - sc.score(0)) * zc;
                        }
                    }
                }
                acc
            })
            .collect();
        let mut cross = vec![0.0_f64; n_genes];
        for acc in blocks {
            for (t, a) in cross.iter_mut().zip(acc) {
                *t += a;
            }
        }
        let depth_rho: Vec<f64> = (0..n_genes)
            .map(|g| match &scores[g] {
                Some(sc) if sc.c[0] > 0.0 => {
                    let rho =
                        (cross[g] / (n * sc.c[0])).clamp(-MAX_ABS_CORRELATION, MAX_ABS_CORRELATION);
                    shrink(rho, 1.0 / (n * sc.c[0]))
                }
                _ => 0.0,
            })
            .collect();

        let hvg_scores: Vec<LatentScores> = hvg
            .iter()
            .map(|&g| scores[g].clone().expect("HVG scores were built"))
            .collect();
        let m = centred_scores(mat, libs.len(), hvg, &hvg_scores, hist);
        let joint = pairwise_correlation(&m, &hvg_scores);
        let hvg_residual = DMatrix::from_fn(hvg.len(), hvg.len(), |i, k| {
            if i == k {
                return 1.0;
            }
            let (ri, rk) = (depth_rho[hvg[i]], depth_rho[hvg[k]]);
            let scale = ((1.0 - ri * ri) * (1.0 - rk * rk)).sqrt().max(1e-6);
            ((joint[(i, k)] - ri * rk) / scale).clamp(-MAX_ABS_CORRELATION, MAX_ABS_CORRELATION)
        });
        Self {
            depth_rho: depth_rho.iter().map(|&r| r as f32).collect(),
            hvg_residual,
        }
    }
}

/// The count histogram (`hist[x]` cells with count `x`) of each gene in
/// `genes` over `n_cells` cells, the cells without a stored entry counting
/// as zeros; `None` for the other genes.
pub fn gene_histograms(
    mat: &CscMatrix<f32>,
    n_cells: usize,
    genes: &[usize],
) -> Vec<Option<Vec<f64>>> {
    let mut hist: Vec<Option<Vec<f64>>> = vec![None; mat.nrows()];
    for &g in genes {
        hist[g] = Some(vec![n_cells as f64]);
    }
    for col in mat.col_iter() {
        for (&g, &v) in col.row_indices().iter().zip(col.values()) {
            let Some(h) = hist[g].as_mut() else {
                continue;
            };
            let x = v.max(0.0) as usize;
            if h.len() <= x {
                h.resize(x + 1, 0.0);
            }
            h[x] += 1.0;
            h[0] -= 1.0;
        }
    }
    hist
}

/// `m_h(x_hc) − m̄_h` over `n_cells` cells (rows) for the genes `hvg` (columns).
fn centred_scores(
    mat: &CscMatrix<f32>,
    n_cells: usize,
    hvg: &[usize],
    scores: &[LatentScores],
    hist: &[Option<Vec<f64>>],
) -> DMatrix<f32> {
    let mut pos = vec![usize::MAX; mat.nrows()];
    for (h, &g) in hvg.iter().enumerate() {
        pos[g] = h;
    }
    let n = n_cells.max(1) as f64;
    // m̄_h = Σ_x hist_h[x] · m_h(x) / N.
    let mean: Vec<f64> = hvg
        .iter()
        .zip(scores)
        .map(|(&g, s)| {
            hist[g].as_deref().map_or(0.0, |h| {
                h.iter()
                    .enumerate()
                    .map(|(x, &c)| c * s.score(x))
                    .sum::<f64>()
                    / n
            })
        })
        .collect();
    let mut m = DMatrix::<f32>::from_fn(n_cells, hvg.len(), |_, h| {
        (scores[h].score(0) - mean[h]) as f32
    });
    for (c, col) in mat.col_iter().enumerate().take(n_cells) {
        for (&g, &v) in col.row_indices().iter().zip(col.values()) {
            let h = pos[g];
            if h != usize::MAX {
                m[(c, h)] = (scores[h].score(v as usize) - mean[h]) as f32;
            }
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::copula::marginals::phi;

    /// A negative binomial of mean `mu` and size `r` (`∞`: Poisson), the
    /// test marginal.
    #[derive(Debug, Clone, Copy)]
    struct NbFit {
        mu: f32,
        r: f32,
    }

    /// `min { x : F(x) ≥ u }` of `fit`.
    fn nb_inverse_cdf(u: f64, fit: NbFit) -> usize {
        let mut cdf = 0.0;
        for (x, p) in support_pmf(fit).into_iter().enumerate() {
            cdf += p;
            if cdf >= u {
                return x;
            }
        }
        usize::MAX
    }

    /// PMF of `fit` over `0..=end`, `end` the first count past which the
    /// tail mass is below 1e-12.
    fn support_pmf(fit: NbFit) -> Vec<f64> {
        let (mu, r) = (fit.mu as f64, fit.r as f64);
        let poisson = !r.is_finite();
        let (mut ln_pk, ln_step) = if poisson {
            (-mu, mu.ln())
        } else {
            let p = r / (r + mu);
            (r * p.ln(), (1.0 - p).ln())
        };
        let (mut pmf, mut cdf) = (Vec::new(), 0.0_f64);
        for k in 0..(1usize << 20) {
            pmf.push(ln_pk.exp());
            cdf += ln_pk.exp();
            if 1.0 - cdf < 1e-12 {
                break;
            }
            let kf = k as f64;
            ln_pk += if poisson {
                ln_step - (kf + 1.0).ln()
            } else {
                ((kf + r) / (kf + 1.0)).ln() + ln_step
            };
        }
        pmf
    }
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    /// `c_1 = E[m²]` and `Σ_q c_q² / q!` never exceeds it: the Hermite
    /// coefficients of the score are a part of its variance.
    #[test]
    fn coefficients_are_a_share_of_the_score_variance() {
        for fit in [
            NbFit { mu: 0.05, r: 0.3 },
            NbFit { mu: 2.0, r: 1.5 },
            NbFit {
                mu: 40.0,
                r: f32::INFINITY,
            },
        ] {
            let pmf = support_pmf(fit);
            let s = LatentScores::from_pmf(&pmf);
            let total: f64 = pmf.iter().sum();
            let e_m2: f64 = pmf.iter().zip(&s.m).map(|(p, m)| p / total * m * m).sum();
            assert!(
                (s.c[0] - e_m2).abs() < 1e-9,
                "{fit:?}: c1 {} E[m²] {e_m2}",
                s.c[0]
            );
            let mut partial = 0.0;
            let mut fact = 1.0;
            for q in 0..HERMITE_ORDER {
                fact *= (q + 1) as f64;
                partial += s.c[q] * s.c[q] / fact;
            }
            assert!(partial <= s.c[0] + 1e-9, "{fit:?}");
            assert!(s.c[0] > 0.0 && s.c[0] <= 1.0 + 1e-9, "{fit:?}");
        }
    }

    /// The moment equation recovers a planted correlation, including a
    /// negative one, and stays on its monotone segment.
    #[test]
    fn moment_equation_round_trips() {
        let a = [0.2, -0.3, 0.25, 0.1];
        for rho in [-0.6, -0.2, 0.0, 0.3, 0.8] {
            let target = moment(rho, &a).0;
            let back = solve_moment(target, &a);
            assert!((back - rho).abs() < 1e-6, "ρ {rho} back {back}");
        }
        assert_eq!(solve_moment(1.0, &[0.0; HERMITE_ORDER]), 0.0);
    }

    /// Planted latent correlations among sparse NB genes come back near
    /// their value, where the Pearson correlation of the mid-point PIT
    /// scores falls well short.
    #[test]
    fn sparse_counts_recover_the_latent_correlation() {
        let n = 20_000;
        let rho = 0.6_f64;
        let fits = [NbFit { mu: 0.08, r: 0.5 }, NbFit { mu: 0.15, r: 1.0 }];
        let mut rng = rand::rngs::StdRng::seed_from_u64(17);
        let normal = Normal::new(0.0_f64, 1.0).unwrap();
        let mut counts: [Vec<usize>; 2] = std::array::from_fn(|_| Vec::with_capacity(n));
        for _ in 0..n {
            let shared = normal.sample(&mut rng);
            for (g, fit) in fits.iter().enumerate() {
                let z = rho.sqrt() * shared + (1.0 - rho).sqrt() * normal.sample(&mut rng);
                counts[g].push(nb_inverse_cdf(phi(z), *fit));
            }
        }
        // The model marginal here; the empirical one in `ReferenceLatent`.
        let scores: Vec<LatentScores> = (0..2)
            .map(|g| LatentScores::from_pmf(&support_pmf(fits[g])))
            .collect();
        let mut m = DMatrix::<f32>::zeros(n, 2);
        for g in 0..2 {
            let mean = counts[g].iter().map(|&x| scores[g].score(x)).sum::<f64>() / n as f64;
            for c in 0..n {
                m[(c, g)] = (scores[g].score(counts[g][c]) - mean) as f32;
            }
        }
        let est = pairwise_correlation(&m, &scores)[(0, 1)];
        assert!((est - rho).abs() < 0.08, "estimated {est}, planted {rho}");
        let row = |g: usize| -> Vec<f64> { (0..n).map(|c| m[(c, g)] as f64).collect() };
        let (x, y) = (row(0), row(1));
        let dot: f64 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
        let pearson = dot
            / (x.iter().map(|a| a * a).sum::<f64>() * y.iter().map(|b| b * b).sum::<f64>()).sqrt();
        assert!(pearson < 0.7 * rho, "Pearson of the scores {pearson}");
    }

    #[test]
    fn shrinkage_keeps_strong_estimates_and_quiets_noise() {
        assert!((shrink(0.5, 1e-4) - 0.5).abs() < 1e-3);
        assert!(shrink(0.05, 0.01).abs() < 0.02);
        assert_eq!(shrink(0.0, 0.0), 0.0);
    }
}
