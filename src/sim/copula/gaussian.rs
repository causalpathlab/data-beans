//! Low-rank Gaussian copula: a rank-`r` factor `F` plus a per-row residual
//! `ridge_sd`, so that `z*[h] = F[h,:]·η + ridge_sd[h]·ε` is standard normal
//! for every row and `corr(z*[h], z*[k]) = F[h,:]·F[k,:]`.
//!
//! [`CopulaCovariance::from_correlation`] fits `F` to a latent correlation
//! matrix (see [`crate::sim::copula::latent`]) by iterated principal-axis
//! factoring: the diagonal of the correlation matrix is replaced by the
//! rows' communalities `‖F[h,:]‖²`, the leading positive eigenpairs give the
//! next `F`, and each row's residual `1 − ‖F[h,:]‖²` goes to `ridge_sd`.
//! The off-diagonal correlations are what the factor fits; each gene keeps
//! the variance the shared factors do not explain as its own, instead of
//! being stretched into pure shared variance. The dense `G × G` matrix is
//! only formed for the copula's genes (the HVGs); sampling uses `F` alone.

use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Principal-axis refits of [`CopulaCovariance::from_correlation`] at most.
const PAF_ITERATIONS: usize = 3;

/// Largest change of a communality at which the refits stop early.
const PAF_TOLERANCE: f64 = 1e-3;

#[derive(Debug, Clone)]
pub struct CopulaCovariance {
    /// Factor loadings, shape `(g, rank)`, columns in decreasing order of
    /// the variance they carry. `‖F[h,:]‖² + ridge_sd[h]² = 1`, so sampled
    /// `z*[h]` is unit-variance — the latent of a *correlation* copula.
    pub factor: DMatrix<f32>,
    /// Per-row residual SD `√(1 − ‖F[h,:]‖²)`. Length `g`.
    pub ridge_sd: DVector<f32>,
}

impl CopulaCovariance {
    pub fn dim(&self) -> usize {
        self.factor.nrows()
    }

    pub fn rank(&self) -> usize {
        self.factor.ncols()
    }

    /// Fit a rank-`rank` factor to the correlation matrix `corr` by iterated
    /// principal-axis factoring. Every row keeps at least `regularization`
    /// of its variance as its own residual, so no gene is a pure copy of the
    /// shared factors. Only positive eigenvalues become factors; with none
    /// (or `rank = 0`) every row is an independent standard normal.
    pub fn from_correlation(
        corr: &DMatrix<f64>,
        rank: usize,
        regularization: f32,
    ) -> anyhow::Result<Self> {
        let g = corr.nrows();
        anyhow::ensure!(
            corr.ncols() == g,
            "copula correlation must be square, got {} × {}",
            g,
            corr.ncols()
        );
        let max_communality = 1.0 - (regularization as f64).clamp(0.0, 1.0);
        let r_eff = rank.min(g);
        // Start from each row's largest off-diagonal |ρ|: the matrix need
        // not be positive semi-definite, so 1 would overstate the shared part.
        let mut communality: Vec<f64> = (0..g)
            .map(|h| {
                (0..g)
                    .filter(|&k| k != h)
                    .map(|k| corr[(h, k)].abs())
                    .fold(0.0_f64, f64::max)
                    .min(max_communality)
            })
            .collect();
        let mut factor = DMatrix::<f64>::zeros(g, 0);
        for _ in 0..PAF_ITERATIONS {
            if r_eff == 0 {
                break;
            }
            let mut reduced = corr.clone();
            for (h, &c) in communality.iter().enumerate() {
                reduced[(h, h)] = c;
            }
            let eig = reduced.symmetric_eigen();
            let mut order: Vec<usize> = (0..g).filter(|&i| eig.eigenvalues[i] > 0.0).collect();
            order.sort_by(|&i, &j| eig.eigenvalues[j].total_cmp(&eig.eigenvalues[i]));
            order.truncate(r_eff);
            factor = DMatrix::from_fn(g, order.len(), |h, k| {
                eig.eigenvectors[(h, order[k])] * eig.eigenvalues[order[k]].sqrt()
            });
            let mut change = 0.0_f64;
            for (h, c) in communality.iter_mut().enumerate() {
                let norm_sq = factor.row(h).norm_squared();
                if norm_sq > max_communality {
                    factor
                        .row_mut(h)
                        .scale_mut((max_communality / norm_sq).sqrt());
                }
                let next = norm_sq.min(max_communality);
                change = change.max((next - *c).abs());
                *c = next;
            }
            if change < PAF_TOLERANCE {
                break;
            }
        }
        let ridge_sd = DVector::from_fn(g, |h, _| {
            ((1.0 - factor.row(h).norm_squared()).max(0.0).sqrt()) as f32
        });
        Ok(Self {
            factor: factor.map(|x| x as f32),
            ridge_sd,
        })
    }

    /// Per-batch log shifts of the reference means: one draw `z_b` per
    /// batch, centred across batches per gene, so the shifts of a gene sum to
    /// zero and only the contrast between batches remains. Each shift is
    /// rescaled by `√(B/(B−1))` to keep the variance `scale²` that centring
    /// would otherwise shrink. With fewer than two batches there is nothing
    /// to contrast and every shift is zero.
    pub fn sample_batch_shifts<R: Rng + ?Sized>(
        &self,
        n_batches: usize,
        scale: f32,
        rng: &mut R,
    ) -> Vec<DVector<f32>> {
        if n_batches < 2 {
            return vec![DVector::zeros(self.dim()); n_batches];
        }
        let draws: Vec<DVector<f32>> = (0..n_batches).map(|_| self.sample(rng)).collect();
        let b = n_batches as f32;
        let mean = draws
            .iter()
            .fold(DVector::zeros(self.dim()), |acc, z| acc + z)
            / b;
        let contrast_scale = scale * (b / (b - 1.0)).sqrt();
        draws
            .into_iter()
            .map(|z| (z - &mean).scale(contrast_scale))
            .collect()
    }

    /// Sample `z*[h] = F[h,:]·η + ridge_sd[h]·ε_h`, `η ~ N(0, I_rank)`,
    /// `ε ~ N(0, I_g)`. After the per-row rescale in `fit`, `Var(z*[h]) = 1`.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> DVector<f32> {
        let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
        let eta = DVector::from_fn(self.rank(), |_, _| normal.sample(rng));
        let mut z = &self.factor * eta;
        for i in 0..self.dim() {
            z[i] += self.ridge_sd[i] * normal.sample(rng);
        }
        z
    }

    /// Take the top `new_rank` columns of `self.factor` and re-fill the
    /// per-row residual to keep `‖F[h,:]‖² + ridge_sd[h]² = 1`. Used by
    /// the two-stage simulator's `--batch-program empirical` mode: batch
    /// shifts ride the dominant gene-gene correlation axes from the
    /// reference-fitted copula.
    pub fn truncate_rank(&self, new_rank: usize) -> Self {
        let g = self.dim();
        let r_keep = new_rank.min(self.rank());
        let factor = if r_keep == 0 {
            DMatrix::<f32>::zeros(g, 0)
        } else {
            self.factor.columns(0, r_keep).into_owned()
        };
        let mut ridge_sd = DVector::<f32>::zeros(g);
        for h in 0..g {
            let row_norm_sq: f32 = (0..r_keep).map(|k| factor[(h, k)].powi(2)).sum();
            ridge_sd[h] = (1.0 - row_norm_sq).max(0.0).sqrt();
        }
        Self { factor, ridge_sd }
    }

    /// Lift a covariance over a feature subset to all `dim` features: row `h`
    /// of `self` becomes row `indices[h]`, and every feature outside
    /// `indices` gets a zero factor row with `ridge_sd = 1` (an independent
    /// unit-variance draw). The copula is fit on HVGs only, so this is what
    /// turns its truncated factor into a per-gene batch program.
    pub fn embed(&self, indices: &[usize], dim: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(
            indices.len() == self.dim(),
            "embed: {} indices for a {}-row covariance",
            indices.len(),
            self.dim()
        );
        let mut factor = DMatrix::<f32>::zeros(dim, self.rank());
        let mut ridge_sd = DVector::<f32>::from_element(dim, 1.0);
        for (h, &g) in indices.iter().enumerate() {
            anyhow::ensure!(g < dim, "embed: index {} out of range for dim {}", g, dim);
            factor.set_row(g, &self.factor.row(h));
            ridge_sd[g] = self.ridge_sd[h];
        }
        Ok(Self { factor, ridge_sd })
    }

    /// Construct a fresh low-rank factor from iid N(0, 1) entries, then
    /// per-row-rescale so each gene has unit marginal variance under
    /// `Var(z*[h]) = ‖F[h,:]‖² + ridge_sd[h]² = 1`. Used by the two-stage
    /// simulator's `--batch-program random` mode: batch shifts live on a
    /// random `rank`-dim subspace independent of the reference's structure.
    ///
    /// `rank = 0` produces an isotropic sampler (`F` empty, all weight on
    /// the per-gene ridge), equivalent to Splatter-style iid log-normal
    /// batch effects.
    pub fn random_low_rank<R: Rng + ?Sized>(g: usize, rank: usize, rng: &mut R) -> Self {
        let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
        let mut factor = DMatrix::<f32>::from_fn(g, rank, |_, _| normal.sample(rng));
        // Cap each row's norm² at 0.99 so the residual ε always has nonzero
        // variance (avoids degenerate `ridge_sd = 0` rows). The 0.99 ceiling
        // is well above what random-Gaussian rows produce for typical g, r.
        const ROW_NORM_SQ_CAP: f32 = 0.99;
        let mut ridge_sd = DVector::<f32>::zeros(g);
        for h in 0..g {
            let row_norm_sq: f32 = (0..rank).map(|k| factor[(h, k)].powi(2)).sum();
            // Rescale the row to unit norm, then attenuate by √ROW_NORM_SQ_CAP
            // so ‖F[h,:]‖² = ROW_NORM_SQ_CAP and the residual gets the rest.
            let target_norm = ROW_NORM_SQ_CAP.sqrt();
            let scale = if row_norm_sq > 1e-12 {
                target_norm / row_norm_sq.sqrt()
            } else {
                0.0
            };
            for k in 0..rank {
                factor[(h, k)] *= scale;
            }
            let new_norm_sq = if rank > 0 { ROW_NORM_SQ_CAP } else { 0.0 };
            ridge_sd[h] = (1.0 - new_norm_sq).max(0.0).sqrt();
        }
        Self { factor, ridge_sd }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// Pearson correlation of the rows of `z`.
    fn pearson(z: &DMatrix<f32>) -> DMatrix<f64> {
        let z = z.map(|x| x as f64);
        let n = z.ncols() as f64;
        let centred = DMatrix::from_fn(z.nrows(), z.ncols(), |h, c| z[(h, c)] - z.row(h).mean());
        let cov = &centred * centred.transpose() / n;
        DMatrix::from_fn(cov.nrows(), cov.ncols(), |i, j| {
            cov[(i, j)] / (cov[(i, i)] * cov[(j, j)]).sqrt()
        })
    }

    /// Embedding keeps the subset's rows where the index map puts them and
    /// gives every other feature an independent unit-variance draw.
    #[test]
    fn embed_scatters_rows_and_fills_unit_ridge() {
        let (h, n, dim) = (6usize, 400usize, 15usize);
        let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let z = DMatrix::from_fn(h, n, |_, _| normal.sample(&mut rng));
        let sub = CopulaCovariance::from_correlation(&pearson(&z), 4, 1e-3)
            .unwrap()
            .truncate_rank(2);
        let indices = [12usize, 0, 7, 3, 14, 9];

        let full = sub.embed(&indices, dim).unwrap();
        assert_eq!(full.dim(), dim);
        assert_eq!(full.rank(), 2);
        assert_eq!(full.sample(&mut rng).len(), dim);
        for (row, &g) in indices.iter().enumerate() {
            assert_eq!(full.factor.row(g), sub.factor.row(row));
            assert_eq!(full.ridge_sd[g], sub.ridge_sd[row]);
        }
        for g in (0..dim).filter(|g| !indices.contains(g)) {
            assert!(full.factor.row(g).iter().all(|&v| v == 0.0));
            assert_eq!(full.ridge_sd[g], 1.0);
        }
        assert!(sub.embed(&indices[..3], dim).is_err());
        assert!(sub.embed(&indices, 10).is_err());
    }

    /// One batch has no other batch to differ from, so its shift is zero;
    /// with several, each batch gets its own shift at the requested scale,
    /// and a gene's shifts sum to zero.
    #[test]
    fn batch_shifts_are_contrasts_between_batches() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        // Rank 0: every gene's draw is its own N(0, 1), so the variance
        // over genes of one batch's shift is the squared scale.
        let cov = CopulaCovariance::random_low_rank(2000, 0, &mut rng);

        let one = cov.sample_batch_shifts(1, 1.0, &mut rng);
        assert_eq!(one.len(), 1);
        assert!(one[0].iter().all(|&v| v == 0.0));

        let three = cov.sample_batch_shifts(3, 0.5, &mut rng);
        assert_eq!(three.len(), 3);
        assert_ne!(three[0], three[1]);
        for shift in &three {
            let var = shift.iter().map(|v| v * v).sum::<f32>() / shift.len() as f32;
            assert!((var - 0.25).abs() < 0.05, "shift variance {var}");
        }
        // The shifts of each gene cancel across batches.
        let total = &three[0] + &three[1] + &three[2];
        assert!(total.iter().all(|v| v.abs() < 1e-5));
        assert!(cov.sample_batch_shifts(0, 1.0, &mut rng).is_empty());
    }

    /// Independent rows stay (nearly) independent: the factors fit only
    /// the sampling noise of the off-diagonals, and every row keeps unit
    /// variance.
    #[test]
    fn independent_rows_stay_independent() {
        let g = 5;
        let n = 5000;
        let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let z = DMatrix::from_fn(g, n, |_, _| normal.sample(&mut rng));
        let cov = CopulaCovariance::from_correlation(&pearson(&z), g, 0.0).unwrap();
        let recovered = &cov.factor * cov.factor.transpose();
        for i in 0..g {
            let var = recovered[(i, i)] + cov.ridge_sd[i] * cov.ridge_sd[i];
            assert!((var - 1.0).abs() < 1e-4, "Var(z*[{i}]) = {var}");
            for j in (0..g).filter(|&j| j != i) {
                assert!(
                    recovered[(i, j)].abs() < 0.1,
                    "Σ[{i},{j}]={}",
                    recovered[(i, j)]
                );
            }
        }
    }

    #[test]
    fn correlation_round_trip() {
        // Build a known Σ with a strong off-diagonal block.
        let g = 3;
        let mut sigma = DMatrix::<f32>::identity(g, g);
        sigma[(0, 1)] = 0.7;
        sigma[(1, 0)] = 0.7;
        let l = sigma.cholesky().unwrap().l();

        let n = 10_000;
        let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(13);
        let mut z = DMatrix::<f32>::zeros(g, n);
        for j in 0..n {
            let eta = DVector::from_fn(g, |_, _| normal.sample(&mut rng));
            z.set_column(j, &(&l * eta));
        }

        let cov = CopulaCovariance::from_correlation(&pearson(&z), g, 0.0).unwrap();
        let recovered = &cov.factor * cov.factor.transpose();
        assert!(
            (recovered[(0, 1)] - 0.7).abs() < 0.05,
            "off-diag={}",
            recovered[(0, 1)]
        );
        // The shared part of row 0 is about the correlation it shares; the
        // rest is its own residual.
        let total = recovered[(0, 0)] + cov.ridge_sd[0] * cov.ridge_sd[0];
        assert!((total - 1.0).abs() < 1e-4, "Var(z*[0]) = {total}");
        assert!(
            recovered[(2, 2)] < 0.1,
            "row 2 shares {}",
            recovered[(2, 2)]
        );
    }

    #[test]
    fn sample_shape_and_rank_cap() {
        let g = 50;
        let n = 200;
        let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let z = DMatrix::from_fn(g, n, |_, _| normal.sample(&mut rng));
        // Request rank=200 but expect it capped at g = 50.
        let cov = CopulaCovariance::from_correlation(&pearson(&z), 200, 1e-3).unwrap();
        assert!(cov.rank() <= g);
        assert_eq!(cov.dim(), g);
        let s = cov.sample(&mut rng);
        assert_eq!(s.len(), g);
        assert_eq!(
            CopulaCovariance::from_correlation(&pearson(&z), 0, 1e-3)
                .unwrap()
                .rank(),
            0
        );
    }

    /// One shared factor between rows 0 and 1 comes back as one column,
    /// rows 2 and 3 load on nothing, and the regularization caps how much
    /// of a row the factor may take.
    #[test]
    fn rank_one_recovery() {
        let mut corr = DMatrix::<f64>::identity(4, 4);
        corr[(0, 1)] = 0.8;
        corr[(1, 0)] = 0.8;
        let cov = CopulaCovariance::from_correlation(&corr, 1, 0.0).unwrap();
        let recovered = &cov.factor * cov.factor.transpose();
        assert!(
            (recovered[(0, 1)] - 0.8).abs() < 0.02,
            "Σ̂[0,1]={}",
            recovered[(0, 1)]
        );
        assert!(
            (recovered[(0, 0)] - 0.8).abs() < 0.02,
            "Σ̂[0,0]={}",
            recovered[(0, 0)]
        );
        assert!(
            recovered[(2, 2)].abs() < 1e-6,
            "Σ̂[2,2]={}",
            recovered[(2, 2)]
        );
        assert!((cov.ridge_sd[2] - 1.0).abs() < 1e-6);

        corr[(0, 1)] = 0.99;
        corr[(1, 0)] = 0.99;
        let capped = CopulaCovariance::from_correlation(&corr, 1, 0.1).unwrap();
        for h in 0..2 {
            let shared = capped.factor.row(h).norm_squared();
            assert!(shared <= 0.9 + 1e-5, "row {h} shares {shared}");
        }
    }

    /// Sub-unit-variance input (like PIT z-scores of sparse counts): the
    /// fit works on correlations, so every sampled marginal has Var≈1
    /// whatever the scale of the rows it came from. Var ≪ 1 would give
    /// `Φ(z*) ≈ 0.5` and an inverse CDF that always returns 0.
    #[test]
    fn unit_marginal_variance_on_subunit_input() {
        let g = 30;
        let n = 5000;
        let mut rng = rand::rngs::StdRng::seed_from_u64(2026);
        // Each row drawn N(0, σ_h²) with σ_h ranging over [0.05, 1.0].
        // Pre-fix code would carry these σ_h into z*; post-fix code must
        // rescale every row back to unit variance.
        let row_sds: Vec<f32> = (0..g)
            .map(|h| 0.05 + 0.95 * (h as f32) / (g as f32 - 1.0))
            .collect();
        let mut z = DMatrix::<f32>::zeros(g, n);
        for h in 0..g {
            let normal = Normal::new(0.0_f32, row_sds[h]).unwrap();
            for j in 0..n {
                z[(h, j)] = normal.sample(&mut rng);
            }
        }
        let cov = CopulaCovariance::from_correlation(&pearson(&z), 5, 1e-3).unwrap();

        // Empirically estimate Var(z*[h]) over many draws.
        let n_samples = 20_000;
        let mut s2 = vec![0.0_f64; g];
        for _ in 0..n_samples {
            let s = cov.sample(&mut rng);
            for h in 0..g {
                let v = s[h] as f64;
                s2[h] += v * v;
            }
        }
        for h in 0..g {
            let var_h = s2[h] / n_samples as f64;
            assert!(
                (var_h - 1.0).abs() < 0.08,
                "row {} (input σ={:.2}): Var(z*[{}]) = {:.3} (expected ≈1)",
                h,
                row_sds[h],
                h,
                var_h
            );
        }
    }

    /// Off-diagonal correlation must be preserved under the per-row rescale.
    /// We build Σ_in with strong off-diagonal block (rows 0,1 strongly
    /// correlated; rows 2,3 independent) then check Σ_out has corr(0,1)≈high,
    /// corr(0,2)≈0 — i.e. structure is conserved when only diag is normalized.
    #[test]
    fn correlation_preserved_under_rescale() {
        let g = 4;
        let n = 10_000;
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
        // Latent: rows 0,1 share η₁ (scaled differently); rows 2,3 are independent.
        let mut z = DMatrix::<f32>::zeros(g, n);
        for j in 0..n {
            let eta1: f32 = normal.sample(&mut rng);
            let eps0: f32 = normal.sample(&mut rng);
            let eps1: f32 = normal.sample(&mut rng);
            z[(0, j)] = 0.2 * (0.9 * eta1 + 0.44 * eps0); // small σ but corr w/ 1
            z[(1, j)] = 1.0 * (0.9 * eta1 + 0.44 * eps1); // big σ but corr w/ 0
            z[(2, j)] = 0.5 * normal.sample(&mut rng);
            z[(3, j)] = 1.5 * normal.sample(&mut rng);
        }
        let cov = CopulaCovariance::from_correlation(&pearson(&z), 4, 1e-4).unwrap();

        // Empirical corr from samples.
        let n_samples = 30_000;
        let mut sum = vec![0.0_f64; g];
        let mut sum2 = vec![0.0_f64; g];
        let mut sxy = vec![vec![0.0_f64; g]; g];
        for _ in 0..n_samples {
            let s = cov.sample(&mut rng);
            for h in 0..g {
                let vh = s[h] as f64;
                sum[h] += vh;
                sum2[h] += vh * vh;
                for k in 0..g {
                    sxy[h][k] += vh * (s[k] as f64);
                }
            }
        }
        let nf = n_samples as f64;
        let mean: Vec<f64> = sum.iter().map(|s| s / nf).collect();
        let var: Vec<f64> = (0..g).map(|h| sum2[h] / nf - mean[h] * mean[h]).collect();
        let corr =
            |h: usize, k: usize| (sxy[h][k] / nf - mean[h] * mean[k]) / (var[h] * var[k]).sqrt();

        // Rows 0 and 1 share latent η₁ with weight 0.9 → cosine ≈ 0.81/(0.81+0.44²) ≈ 0.808.
        let c01 = corr(0, 1);
        assert!(c01 > 0.6, "corr(z*[0], z*[1])={:.3} expected >0.6", c01);
        // Rows 2 and 3 are independent.
        let c23 = corr(2, 3);
        assert!(c23.abs() < 0.1, "corr(z*[2], z*[3])={:.3} expected ≈0", c23);
        // Cross: row 0 vs row 2 — independent.
        let c02 = corr(0, 2);
        assert!(c02.abs() < 0.1, "corr(z*[0], z*[2])={:.3} expected ≈0", c02);
        // All marginals unit variance.
        for (h, &v) in var.iter().enumerate().take(g) {
            assert!(
                (v - 1.0).abs() < 0.05,
                "Var(z*[{}])={:.3}, expected ≈1",
                h,
                v
            );
        }
    }

    /// `truncate_rank` keeps the top-r columns of an existing rank-R fit
    /// and refills the per-row residual to maintain unit marginal variance.
    /// Used by the simulator's `--batch-program empirical` mode.
    #[test]
    fn truncate_rank_preserves_unit_variance() {
        let g = 25;
        let n = 4_000;
        let mut rng = rand::rngs::StdRng::seed_from_u64(2027);
        let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
        let z = DMatrix::from_fn(g, n, |_, _| normal.sample(&mut rng));
        let full = CopulaCovariance::from_correlation(&pearson(&z), 10, 1e-3).unwrap();
        for new_rank in [0usize, 1, 3, 7] {
            let truncated = full.truncate_rank(new_rank);
            assert_eq!(truncated.dim(), g);
            assert_eq!(truncated.rank(), new_rank.min(full.rank()));

            let n_samples = 15_000;
            let mut s2 = vec![0.0_f64; g];
            for _ in 0..n_samples {
                let s = truncated.sample(&mut rng);
                for h in 0..g {
                    let v = s[h] as f64;
                    s2[h] += v * v;
                }
            }
            for (h, ss) in s2.iter().enumerate().take(g) {
                let var_h = ss / n_samples as f64;
                assert!(
                    (var_h - 1.0).abs() < 0.08,
                    "rank={}, row {}: Var(z*) = {:.3} (expected ≈1)",
                    new_rank,
                    h,
                    var_h
                );
            }
        }
    }

    /// `random_low_rank` builds a fresh `(g, rank)` factor with iid Gaussian
    /// entries, rescaled to a fixed row-norm cap so the residual carries
    /// the rest of the unit-variance budget. Used by `--batch-program random`.
    #[test]
    fn random_low_rank_unit_variance() {
        let g = 30;
        let mut rng = rand::rngs::StdRng::seed_from_u64(2028);
        for rank in [0usize, 1, 3, 5] {
            let cov = CopulaCovariance::random_low_rank(g, rank, &mut rng);
            assert_eq!(cov.dim(), g);
            assert_eq!(cov.rank(), rank);

            let n_samples = 15_000;
            let mut s2 = vec![0.0_f64; g];
            for _ in 0..n_samples {
                let s = cov.sample(&mut rng);
                for h in 0..g {
                    let v = s[h] as f64;
                    s2[h] += v * v;
                }
            }
            for (h, ss) in s2.iter().enumerate().take(g) {
                let var_h = ss / n_samples as f64;
                assert!(
                    (var_h - 1.0).abs() < 0.08,
                    "rank={}, row {}: Var(z*) = {:.3} (expected ≈1)",
                    rank,
                    h,
                    var_h
                );
            }
        }
    }
}
