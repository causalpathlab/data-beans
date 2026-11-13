//! Degree-corrected Poisson refinement core.
//!
//! Shared Poisson-scoring machinery for top-down multi-level refinement.
//! Given per-entity sparse profiles (gene sums or a dense projection) and
//! current group labels, refinement reassigns entities between groups by
//! scoring moves under a Poisson likelihood with size-factor offsets.
//! Candidate-set construction is delegated to [`CandidateProposer`] so
//! different front-ends (cross-batch BBKNN, spatial KNN graph, ...) can
//! plug in their own neighborhood definition.
//!
//! The partition handed in is kept as a partition: refinement makes its
//! blocks purer without choosing how many there are or how large. Scores
//! are built so that a block is not preferred for being large (see
//! [`DcPoissonStats`]), an entity moves only on evidence beyond the noise
//! of the block estimates, and member-count limits ([`RefineParams`]) keep
//! every block near its starting size.
//!
//! The name "DC-Poisson" is deliberate: we are *not* inferring a stochastic
//! block model — the blocks are given externally (by a prior coarsening
//! step) and we only score membership moves. Compare to the full DC-SBM
//! where blocks themselves are latent variables.
//!
//! Entry points:
//! - [`refine_with_candidates`] — lowest-level sweep driver given pre-built candidates.
//! - [`refine_with_proposer`] — generic driver that first asks a [`CandidateProposer`].
//!
//! See the `BbknnProposer` in `refine_multilevel` for the data_beans::alg
//! front-end, and pinto's `GraphProposer` for the spatial-graph front-end.

use log::info;
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;

/// Additive floor to keep `ln(.)` finite when a block or feature has zero
/// mass.
pub const LOG_EPS: f64 = 1e-9;

/// `ln(x)` in `f32`: libm's single-precision log, near half the cost of
/// the `f64` one. Its error is `f32` rounding (about 1e-7 relative), the
/// precision the cached block logs are stored at anyway. NaN, infinite and
/// non-positive inputs keep their usual `ln` values.
#[inline]
fn fast_ln(x: f64) -> f64 {
    f64::from((x as f32).ln())
}

//////////////////////////
// Public configuration //
//////////////////////////

/// Feature representation used to score entity → block moves.
#[derive(Clone, Debug)]
pub enum ProfileSource {
    /// Sparse entity × feature counts (the existing `gene_sums` format —
    /// the field name is historical; the axis is generic).
    /// [`FeatureWeighting`] is applied per-feature on the sparse profile.
    Raw,
    /// Dense projection: `basis * indicator(entity)` per entity. `basis` is
    /// `proj_dim × num_cells`-shaped; the per-entity profile is the sum over
    /// its member cells' projection columns. [`FeatureWeighting`] is skipped
    /// because the feature axis is no longer count-distributed.
    Projected { basis: DMatrix<f32> },
}

/// Per-feature weighting applied to the sparse profile before DC-Poisson
/// scoring. All variants multiply each nonzero entry `y_{e,f}` by a scalar
/// `w_f` and recompute per-row size factors; only the formula for `w_f`
/// differs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureWeighting {
    /// No reweighting: `w_f = 1`. Recovers proper DC-Poisson with entity-level
    /// degree correction (Karrer–Newman MAP under `Gamma(α, 0)` on `θ`).
    None,
    /// Fisher-information weight under the fitted NB trend (genes-as-features
    /// case): `w_f = 1 / (1 + π_f · s̄ · φ(μ_f))`. Bounded in `(0, 1]`,
    /// attenuates high-mean / high-dispersion features, recovers
    /// `w_f = 1` in the Poisson limit (`φ → 0`).
    FisherInfoNb,
}

/// Parameters controlling the refinement pass.
#[derive(Clone, Debug)]
pub struct RefineParams {
    /// Gibbs sweeps per level (0 disables Gibbs; greedy still runs).
    pub num_gibbs: usize,
    /// Greedy sweeps per level (early-exits on zero moves).
    pub num_greedy: usize,
    /// Per-feature weighting scheme (only meaningful for `Raw` profile source).
    pub feature_weighting: FeatureWeighting,
    /// Seed for Gibbs RNG.
    pub seed: u64,
    /// Gibbs stagnation threshold (fraction of entities moving per sweep).
    /// Breaks early when three consecutive sweeps are below this bound.
    /// `0.0` disables early exit.
    pub gibbs_stagnation: f64,
    /// Feature source. Defaults to `Raw`.
    pub profile_source: ProfileSource,
    /// Jacobi-style parallel sweeps (default `true`).
    ///
    /// When `true`, every entity proposes its move against a frozen pre-sweep
    /// snapshot of the sufficient stats (rayon `par_iter`); proposals are
    /// then applied sequentially with the move guard. Per-entity RNG seeds
    /// are derived from `(sweep_seed, entity_idx)`, so the result is
    /// deterministic given `seed` and independent of thread count (per
    /// `rand` release — `SmallRng`'s stream is not guaranteed stable across
    /// `rand` versions).
    ///
    /// When `false`, runs the classic Gauss–Seidel loop: each entity scores
    /// against the live stats (updated by every accepted move so far in the
    /// sweep), single-threaded.
    ///
    /// Jacobi is biased — within a sweep, entities can't see each other's
    /// moves — but converges fine for the multilevel DC-Poisson use case
    /// and is roughly P× faster on large pb-sample counts.
    pub parallel: bool,
    /// Pseudo-counts per expressed feature behind every block profile,
    /// spread over the features by the pooled profile (default `1`).
    /// Replaces a hard floor under `ln(0)`: a feature a block has not seen
    /// scores as the pooled rate shrunk by the block's mass, and an empty
    /// block scores as the pooled profile. See [`DcPoissonStats`].
    pub prior_count: f32,
    /// Standard deviations of block-estimate noise by which a destination
    /// has to beat the current block (default `1.5`; `0` asks for any
    /// positive gain). The noise is larger for smaller blocks, so this is
    /// what keeps a small block from losing, or taking in, members on the
    /// strength of its own sampling error. See [`DcPoissonStats`].
    pub min_gain_z: f32,
    /// Fraction of `min(starting size, mean block size)` below which no
    /// block is taken (default `0.9`); any positive value also keeps the
    /// last member of every block. `0` lets blocks shrink and empty.
    ///
    /// The refinement is asked to make a given partition purer, not to
    /// choose the number of blocks: by the likelihood alone, the members of
    /// a mixed block all do better elsewhere, and the block dissolves.
    pub size_floor: f32,
    /// Multiple of `max(starting size, mean block size)` above which no
    /// block is taken (default `1.5`). `0` lifts the limit.
    pub size_cap: f32,
}

impl Default for RefineParams {
    fn default() -> Self {
        Self {
            num_gibbs: 20,
            num_greedy: 10,
            feature_weighting: FeatureWeighting::FisherInfoNb,
            seed: 42,
            gibbs_stagnation: 0.005,
            profile_source: ProfileSource::Raw,
            parallel: true,
            prior_count: 1.0,
            min_gain_z: 1.5,
            size_floor: 0.9,
            size_cap: 1.5,
        }
    }
}

////////////////////////////////////
// Sparse row-per-entity profiles //
////////////////////////////////////

/// Sparse row-per-entity profile, owned by the refinement pass.
///
/// Materialized from either a sparse gene-sum format (`from_gene_sums`) or a
/// projected centroid matrix (`from_projection`). Values are stored sorted
/// by feature index within each row so downstream accumulators can rely on
/// the invariant.
pub struct Profiles {
    pub rows: Vec<Vec<(u32, f32)>>,
    pub size_factor: Vec<f32>,
    pub num_entities: usize,
    pub num_features: usize,
}

impl Profiles {
    pub fn from_gene_sums(gene_sums: &[Vec<(usize, f32)>], num_features: usize) -> Self {
        let num_entities = gene_sums.len();
        let (rows, size_factor): (Vec<Vec<(u32, f32)>>, Vec<f32>) = gene_sums
            .par_iter()
            .map(|row| {
                let mut out: Vec<(u32, f32)> = row
                    .iter()
                    .filter(|(_, v)| *v > 0.0)
                    .map(|(g, v)| (*g as u32, *v))
                    .collect();
                out.sort_unstable_by_key(|&(g, _)| g);
                let sf: f32 = out.iter().map(|(_, v)| *v).sum();
                (out, sf)
            })
            .unzip();
        Self {
            rows,
            size_factor,
            num_entities,
            num_features,
        }
    }

    /// Build profiles from a dense projection.
    ///
    /// `basis` is `proj_dim × num_cells`; each entity profile is the sum of
    /// basis columns over the cells constituting it. Gene weighting is not
    /// applied — projection dims aren't gene-aligned.
    ///
    /// The Poisson score needs nonnegative profiles, so the accumulated
    /// per-dimension sums must be `>= 0` — a signed basis (e.g. a raw random
    /// projection) is rejected loudly rather than silently truncated.
    pub fn from_projection(basis: &DMatrix<f32>, entity_to_cells: &[Vec<usize>]) -> Self {
        let num_features = basis.nrows();
        let num_entities = entity_to_cells.len();
        let (rows, size_factor): (Vec<Vec<(u32, f32)>>, Vec<f32>) = entity_to_cells
            .par_iter()
            .map(|cells| {
                let mut acc = vec![0f32; num_features];
                for &c in cells {
                    for d in 0..num_features {
                        acc[d] += basis[(d, c)];
                    }
                }
                let mut out = Vec::with_capacity(num_features);
                let mut sf = 0f32;
                for (d, &v) in acc.iter().enumerate() {
                    assert!(
                        v >= 0.0,
                        "projection profile is negative at dim {d} ({v}); \
                         DC-Poisson requires a nonnegative basis"
                    );
                    if v > 0.0 {
                        out.push((d as u32, v));
                        sf += v;
                    }
                }
                (out, sf)
            })
            .unzip();
        Self {
            rows,
            size_factor,
            num_entities,
            num_features,
        }
    }

    /// In-place reweighting by a caller-supplied per-feature weight vector.
    /// Used by [`Profiles::apply_feature_weighting`] for the NB Fisher-info path.
    pub fn weight_by_vec(&mut self, w: &[f32]) {
        assert_eq!(
            w.len(),
            self.num_features,
            "weight vector length must match num_features"
        );
        self.rows
            .par_iter_mut()
            .zip(self.size_factor.par_iter_mut())
            .for_each(|(row, sf)| {
                let mut new_sf = 0f32;
                for (g, v) in row.iter_mut() {
                    *v *= w[*g as usize];
                    new_sf += *v;
                }
                *sf = new_sf;
            });
    }

    /// Rescale each entity's features by `pooled rate / rate in its batch`,
    /// then recompute the size factors.
    ///
    /// `batch[e]` is the batch of entity `e`. A batch effect multiplies a
    /// feature's rate in every entity of the batch, so entities of one
    /// batch fit a block by how much of that batch it holds, whatever its
    /// content, and a refinement on the raw profiles sorts them by batch.
    /// Dividing by the batch's marginal rate removes the shared factor and
    /// leaves the differences between entities of the same batch. A feature
    /// a batch never shows is left as it is.
    pub fn rescale_batches_to_pooled(&mut self, batch: &[usize]) {
        assert_eq!(batch.len(), self.rows.len(), "one batch per entity");
        let nb = batch.iter().copied().max().map_or(0, |b| b + 1);
        if nb < 2 {
            return;
        }
        let m = self.num_features;
        let mut sums = vec![0f64; nb * m];
        let mut batch_total = vec![0f64; nb];
        for (row, &b) in self.rows.iter().zip(batch) {
            for &(g, v) in row {
                sums[b * m + g as usize] += v as f64;
                batch_total[b] += v as f64;
            }
        }
        let total: f64 = batch_total.iter().sum();
        if total <= 0.0 {
            return;
        }
        let mut ratio = vec![1f32; nb * m];
        for g in 0..m {
            let pooled = (0..nb).map(|b| sums[b * m + g]).sum::<f64>() / total;
            for b in 0..nb {
                let s = sums[b * m + g];
                if s > 0.0 {
                    ratio[b * m + g] = (pooled * batch_total[b] / s) as f32;
                }
            }
        }
        self.rows
            .par_iter_mut()
            .zip(self.size_factor.par_iter_mut())
            .zip(batch.par_iter())
            .for_each(|((row, sf), &b)| {
                let mut new_sf = 0f32;
                for (g, v) in row.iter_mut() {
                    *v *= ratio[b * m + *g as usize];
                    new_sf += *v;
                }
                *sf = new_sf;
            });
    }

    /// Apply the chosen feature weighting in place.
    ///
    /// [`FeatureWeighting::None`] is a no-op; [`FeatureWeighting::FisherInfoNb`]
    /// fits an NB dispersion trend from the current profiles and reweights each
    /// feature by `1 / (1 + π_f · s̄ · φ(μ_f))`.
    pub fn apply_feature_weighting(&mut self, method: FeatureWeighting) {
        match method {
            FeatureWeighting::None => {}
            FeatureWeighting::FisherInfoNb => {
                let w = self.nb_fisher_weights();
                self.weight_by_vec(&w);
            }
        }
    }

    /// Compute per-feature Fisher-info weights under an NB trend fit from
    /// the current profile contents. Returned vector has length `num_features`
    /// and is suitable for [`Profiles::weight_by_vec`].
    pub fn nb_fisher_weights(&self) -> Vec<f32> {
        use crate::alg::nb_dispersion::DispersionTrend;
        use legume_numeric::matrix::sparse_stat::SparseRunningStatistics;
        use legume_numeric::matrix::traits::RunningStatOps;

        // Rows are folded in parallel and merged in a fixed order, so the
        // (non-integer) sums come out the same on any number of threads.
        // Avoids the single-threaded `for row in &self.rows` over ~10⁴–10⁵
        // pb-samples.
        const ROWS_PER_LEAF: usize = 256;
        let num_features = self.num_features;
        let (stats, ..) = crate::qc::fold_in_fixed_order(
            &self.rows,
            ROWS_PER_LEAF,
            &|| {
                (
                    SparseRunningStatistics::<f32>::new(num_features),
                    Vec::<usize>::new(),
                    Vec::<f32>::new(),
                )
            },
            &|(acc, col_rows, col_vals), row| {
                col_rows.clear();
                col_vals.clear();
                for &(g, v) in row {
                    col_rows.push(g as usize);
                    col_vals.push(v);
                }
                acc.add_sparse_column(col_rows, col_vals);
                Ok(())
            },
            &|(a, ..), (b, ..)| a.merge(&b),
        )
        .expect("the row fold cannot fail");

        // `π_g = sum[g] / Σ sum` is derived directly from the stats without
        // a second sparse traversal. `mean_g` uses the entity-count denominator
        // that `SparseRunningStatistics` already provides.
        let trend = DispersionTrend::from_sparse_stats(&stats);
        let means = stats.mean();
        let sums = stats.sum();
        let total_mass: f64 = sums.iter().map(|&s| s as f64).sum();
        let avg_s = if self.num_entities > 0 {
            (total_mass / self.num_entities as f64) as f32
        } else {
            1.0
        };
        let inv_total = if total_mass > 0.0 {
            1.0 / total_mass as f32
        } else {
            0.0
        };
        (0..self.num_features)
            .map(|g| trend.fisher_weight(sums[g] * inv_total, avg_s, means[g]))
            .collect()
    }
}

///////////////////////////////////
// Poisson sufficient statistics //
///////////////////////////////////

/// `ln(x + a)` plus half the relative variance of a count `x` around
/// `x + a`: the log of a smoothed block total, corrected for the downward
/// bias of the plain log (see [`DcPoissonStats`]).
///
/// The variance of a Poisson count is the count. A total below one is not
/// a count but a weighted fraction of one (`w · n` has variance `w² · n`,
/// at most its square), so the variance is taken as `min(x, x²)`. That
/// keeps the correction within `[0, 1/2]` whatever `a`, where `x / (x + a)²`
/// alone would grow without bound for a small total on a rare feature.
#[inline(always)]
fn log_total(x: f64, a: f64) -> f64 {
    let t = x + a;
    fast_ln(t) + x.min(x * x) / (2.0 * t * t)
}

/// Sufficient statistics with cached log quantities for O(K · nnz(row)) scoring.
///
/// `gene_sum[g·K + k]` = Σ_{e : z_e = k} y_eg; `size_sum[k]` = Σ_{e : z_e = k} s_e.
/// The per-feature tables are laid out **feature-major** (`g·K + k`): scoring
/// one entity walks its nonzero features once and, for each, reads the `K`
/// group values side by side, instead of striding across a `K × M` table
/// once per candidate group. Use [`DcPoissonStats::idx`] to address them.
/// Log caches score every non-current block with no log; the current block's
/// leave-one-out score takes one `fast_ln` per nonzero feature, and
/// `delta_move` refreshes the logs of the columns a move touches.
///
/// # The score
///
/// Block `k`'s rate for feature `g` is the smoothed share
/// `(G_kg + a_g) / (S_k + α)`, with `a_g = α · π_g`, `π` the pooled feature
/// profile and `α = prior_count · M₊` (`M₊` = features with positive pooled
/// mass): on average `prior_count` pseudo-counts per expressed feature. A
/// feature a block has not seen costs `ln(a_g / (S_k + α))`, the pooled rate
/// shrunk by the block's own mass, and an empty block scores as the pooled
/// profile. The mass is set in counts per feature, not in entities: the
/// expansions below need `a_g` of the order of one count, which a mass tied
/// to the entity size loses at low depth and overshoots at high depth,
/// where it would pull small distinct blocks toward the pooled profile.
///
/// A block's total `G_kg` is a noisy count. With `G ~ Poisson(μ)`,
/// `E ln(G + a) ≈ ln(μ + a) − μ / (2 (μ + a)²)`: the plain log is biased
/// low, by more the smaller the block. Summed over an entity's counts the
/// deficit is about `s_e · M₊ / (2 S_k)`, that is `(s_e / s̄) · M₊ / (2 n_k)`
/// nats for a block of `n_k` typical entities: it depends on the block's
/// size, not on its profile, and not on the depth. Left in, it makes every
/// entity prefer the larger of two blocks with one profile, and small
/// blocks drain. The cached value is therefore
/// `ln(G + a) + G / (2 (G + a)²)`, which cancels that term to first order,
/// so that blocks of one profile score alike whatever their size (for a
/// total below one count, as weighted profiles have, the variance is taken
/// as `G²`, which bounds the correction by one half).
///
/// What remains is zero-mean noise. Its variance is
/// `Σ_g y_eg² · μ_g / (μ_g + a_g)²`, which for a block that shares the
/// pooled profile (`μ_g = S_k π_g`) is `r_e · v_k` with
/// `r_e = Σ_g y_eg² / π_g` and `v_k = S_k / (S_k + α)²`. A destination is
/// charged `min_gain_z` standard deviations of the difference between its
/// score and the current block's, `z · sqrt(r_e · (v_cur + v_k))`: a move
/// has to beat what the block estimates alone would produce between two
/// blocks of one profile. Everything is in nats derived from the same
/// Poisson model, so `z` means the same at every depth.
///
/// Log caches are `f32` — values live in roughly `[-25, +25]` and scoring
/// is noise-dominated, so f32's ~7-digit precision is ample. Halves the
/// largest allocation versus f64.
#[derive(Clone)]
pub struct DcPoissonStats {
    pub k: usize,
    pub num_features: usize,
    pub membership: Vec<usize>,
    /// Feature-major `M × K`: entry `g·K + k`.
    pub gene_sum: Vec<f64>,
    pub size_sum: Vec<f64>,
    /// Feature-major `M × K`: entry `g·K + k`. Holds the bias-corrected
    /// log `ln(G + a_g) + min(G, G²) / (2 (G + a_g)²)`.
    pub log_gene: Vec<f32>,
    /// `−ln(S_k + α)`.
    pub log_size_offset: Vec<f32>,
    /// Members per block.
    pub count: Vec<usize>,
    /// Pseudo-count `a_g` of each feature (never below [`LOG_EPS`]).
    pub prior: Vec<f64>,
    /// Total pseudo-count mass `Σ_g a_g`.
    pub prior_mass: f64,
    /// Noise weight `r_e = Σ_g y_eg² / π_g` of each entity.
    pub noise: Vec<f32>,
    /// Estimate variance `v_k = S_k / (S_k + α)²` of each block.
    pub block_var: Vec<f64>,
    /// Standard deviations of estimate noise a move has to clear.
    pub min_gain_z: f64,
}

impl DcPoissonStats {
    /// Index of (group `k`, feature `g`) in `gene_sum` / `log_gene`.
    #[inline(always)]
    pub fn idx(&self, k: usize, g: usize) -> usize {
        g * self.k + k
    }

    /// Statistics under the default [`RefineParams`] scoring.
    pub fn from_profiles(profiles: &Profiles, k: usize, membership: &[usize]) -> Self {
        Self::from_profiles_with(profiles, k, membership, &RefineParams::default())
    }

    /// Statistics scored with `params.prior_count` and `params.min_gain_z`.
    pub fn from_profiles_with(
        profiles: &Profiles,
        k: usize,
        membership: &[usize],
        params: &RefineParams,
    ) -> Self {
        let m = profiles.num_features;
        let mut gene_sum = vec![0f64; k * m];
        let mut size_sum = vec![0f64; k];
        let mut count = vec![0usize; k];
        let mut pooled = vec![0f64; m];
        for (e, row) in profiles.rows.iter().enumerate() {
            let z = membership[e];
            assert!(z < k, "membership[{}]={} out of range 0..{}", e, z, k);
            for &(g, v) in row {
                gene_sum[g as usize * k + z] += v as f64;
                pooled[g as usize] += v as f64;
            }
            size_sum[z] += profiles.size_factor[e] as f64;
            count[z] += 1;
        }
        let total: f64 = pooled.iter().sum();
        let expressed = pooled.iter().filter(|&&x| x > 0.0).count() as f64;
        let alpha = (params.prior_count.max(0.0) as f64) * expressed;
        // r_e = Σ y² / π_g = Σ y² · total / pooled_g; a feature an entity
        // has a positive count in has positive pooled mass.
        let noise: Vec<f32> = profiles
            .rows
            .par_iter()
            .map(|row| {
                row.iter()
                    .filter(|&&(_, v)| v > 0.0)
                    .map(|&(g, v)| (v as f64) * (v as f64) * total / pooled[g as usize])
                    .sum::<f64>() as f32
            })
            .collect();
        let scale = if total > 0.0 { alpha / total } else { 0.0 };
        let prior: Vec<f64> = pooled.iter().map(|&x| x * scale + LOG_EPS).collect();
        let prior_mass = alpha + m as f64 * LOG_EPS;
        let mut stats = Self {
            k,
            num_features: m,
            membership: membership.to_vec(),
            gene_sum,
            size_sum,
            log_gene: vec![0f32; k * m],
            log_size_offset: vec![0f32; k],
            count,
            prior,
            prior_mass,
            noise,
            block_var: vec![0f64; k],
            min_gain_z: params.min_gain_z.max(0.0) as f64,
        };
        stats.refresh_caches();
        stats
    }

    /// `v = S / (S + α)²` for a block of mass `s`.
    #[inline(always)]
    fn var_of(&self, s: f64) -> f64 {
        let t = s + self.prior_mass;
        s / (t * t)
    }

    /// Refresh the per-block caches of block `k`.
    #[inline]
    fn refresh_block(&mut self, k: usize) {
        self.log_size_offset[k] = -fast_ln(self.size_sum[k] + self.prior_mass) as f32;
        self.block_var[k] = self.var_of(self.size_sum[k]);
    }

    /// Rebuild every cache from the running sums.
    fn refresh_caches(&mut self) {
        let k = self.k;
        for (g, &a) in self.prior.iter().enumerate() {
            let sums = &self.gene_sum[g * k..(g + 1) * k];
            let logs = &mut self.log_gene[g * k..(g + 1) * k];
            for (l, &s) in logs.iter_mut().zip(sums) {
                *l = log_total(s, a) as f32;
            }
        }
        for b in 0..k {
            self.refresh_block(b);
        }
    }

    /// Apply an entity's reassignment and refresh only the affected log entries.
    pub fn delta_move(&mut self, e: usize, k_from: usize, k_to: usize, profiles: &Profiles) {
        if k_from == k_to {
            return;
        }
        let kk = self.k;

        // Subtractions clamp at zero: the running sums accumulate rounding
        // residues, and a block drained back to (true) zero can otherwise be
        // left with a negative residue larger than its pseudo-count —
        // feeding `ln(negative) = NaN` into the log caches.
        for &(g, v) in &profiles.rows[e] {
            let base = g as usize * kk;
            let a = self.prior[g as usize];
            let (i_from, i_to) = (base + k_from, base + k_to);
            self.gene_sum[i_from] = (self.gene_sum[i_from] - v as f64).max(0.0);
            self.gene_sum[i_to] += v as f64;
            self.log_gene[i_from] = log_total(self.gene_sum[i_from], a) as f32;
            self.log_gene[i_to] = log_total(self.gene_sum[i_to], a) as f32;
        }
        let sf = profiles.size_factor[e] as f64;
        self.size_sum[k_from] = (self.size_sum[k_from] - sf).max(0.0);
        self.size_sum[k_to] += sf;
        self.count[k_from] = self.count[k_from].saturating_sub(1);
        self.count[k_to] += 1;
        self.refresh_block(k_from);
        self.refresh_block(k_to);

        self.membership[e] = k_to;
    }

    /// Full recompute from current `membership` (slow path, used for tests).
    #[cfg(test)]
    fn recompute(&mut self, profiles: &Profiles) {
        let k = self.k;
        self.gene_sum.iter_mut().for_each(|x| *x = 0.0);
        self.size_sum.iter_mut().for_each(|x| *x = 0.0);
        self.count.iter_mut().for_each(|x| *x = 0);
        for (e, row) in profiles.rows.iter().enumerate() {
            let z = self.membership[e];
            for &(g, v) in row {
                self.gene_sum[g as usize * k + z] += v as f64;
            }
            self.size_sum[z] += profiles.size_factor[e] as f64;
            self.count[z] += 1;
        }
        self.refresh_caches();
    }
}

/////////////////////
// Scoring kernels //
/////////////////////

/// Score entity `e` for one destination `k`, as documented on
/// [`compute_log_probs_restricted`], which scores all allowed blocks in one
/// pass with the same per-block summation order (so the two agree bit for
/// bit). Test reference.
#[cfg(test)]
fn score_move(e: usize, k: usize, stats: &DcPoissonStats, profiles: &Profiles) -> f64 {
    let sf = profiles.size_factor[e] as f64;
    let row = &profiles.rows[e];
    let cur = stats.membership[e];
    if cur == k {
        let loo_size = (stats.size_sum[k] - sf).max(0.0);
        let mut acc = -sf * fast_ln(loo_size + stats.prior_mass);
        for &(g, v) in row {
            let vg = v as f64;
            let loo = (stats.gene_sum[stats.idx(k, g as usize)] - vg).max(0.0);
            acc += vg * log_total(loo, stats.prior[g as usize]);
        }
        acc
    } else {
        let mut acc = sf * stats.log_size_offset[k] as f64;
        for &(g, v) in row {
            acc += v as f64 * stats.log_gene[stats.idx(k, g as usize)] as f64;
        }
        let v_cur = stats.var_of((stats.size_sum[cur] - sf).max(0.0));
        acc - stats.min_gain_z * (stats.noise[e] as f64 * (v_cur + stats.block_var[k])).sqrt()
    }
}

/// Log-probability of placing entity `e` into each of the `allowed` blocks.
///
/// `  s(e, k) = Σ_{g : y_eg > 0} y_eg · L(G⁻ᵉ[k, g], a_g) − size_factor[e] · ln(S⁻ᵉ[k] + α)`
///
/// with `L(G, a) = ln(G + a) + min(G, G²) / (2 (G + a)²)` the bias-corrected
/// log of the smoothed block total (see [`DcPoissonStats`]). `⁻ᵉ` means
/// entity `e`'s own contribution is excluded: candidate blocks never
/// contain `e` and read the log caches, while `e`'s *current* block
/// subtracts `e`'s row before taking logs (leave-one-out) — otherwise
/// "stay put" earns a self-inclusion bonus that anchors large entities.
///
/// Every block other than the current one is then charged
/// `min_gain_z · sqrt(r_e · (v_cur + v_k))`, the estimate noise of the
/// difference to the current block's score, so that comparing a slot with
/// the current block's — by argmax or by sampling — already asks for a gain
/// beyond that noise.
///
/// All blocks are scored in one pass over `e`'s nonzero features: per
/// feature, the `K` cached values sit side by side (feature-major layout),
/// so the pass touches one short contiguous run per feature. Each block's
/// sum is still accumulated in feature order, the same order as scoring the
/// blocks one at a time, so the scores match that bit for bit. Partial
/// candidate lists accumulate by *position* in `allowed`, so a block listed
/// twice gets the same, correct score in both places.
///
/// Only the `allowed` slots of `log_probs` are written, unless `allowed`
/// is as long as `K`, in which case the first `K` slots are written.
/// The rest keep whatever stale values a previous call left (the caller
/// reuses one buffer across sweeps). Downstream consumers
/// ([`sample_categorical_log_restricted`], [`argmax_log_restricted`]) read
/// the `allowed` slots exclusively.
pub fn compute_log_probs_restricted(
    e: usize,
    stats: &DcPoissonStats,
    profiles: &Profiles,
    allowed: &[usize],
    log_probs: &mut [f64],
) {
    let kk = stats.k;
    let sf = profiles.size_factor[e] as f64;
    let row = &profiles.rows[e];
    let cur = stats.membership[e];
    // Every block 0..K is scored straight into `log_probs`, so repeats in
    // `allowed` (which then misses some block) are harmless.
    let dense = allowed.len() == kk;
    let score_cur = dense || allowed.contains(&cur);

    // Leave-one-out score of the current block.
    let loo_size = (stats.size_sum[cur] - sf).max(0.0);
    let mut cur_acc = if score_cur {
        -sf * fast_ln(loo_size + stats.prior_mass)
    } else {
        0.0
    };
    let mut add_cur = |g: usize, base: usize, vg: f64| {
        if score_cur {
            let loo = (stats.gene_sum[base + cur] - vg).max(0.0);
            cur_acc += vg * log_total(loo, stats.prior[g]);
        }
    };

    // Estimate noise of a move out of the current block, per destination.
    let z = stats.min_gain_z;
    let noise = stats.noise[e] as f64;
    let v_cur = stats.var_of(loo_size);
    let charge = |k: usize| z * (noise * (v_cur + stats.block_var[k])).sqrt();

    // Cached (non-leave-one-out) scores for every allowed block, including
    // the current one, which is overwritten at the end.
    if dense {
        for (slot, &o) in log_probs[..kk].iter_mut().zip(&stats.log_size_offset) {
            *slot = sf * o as f64;
        }
        for &(g, v) in row {
            let vg = v as f64;
            let base = g as usize * kk;
            let logs = &stats.log_gene[base..base + kk];
            for (slot, &l) in log_probs[..kk].iter_mut().zip(logs) {
                *slot += vg * l as f64;
            }
            add_cur(g as usize, base, vg);
        }
        if z > 0.0 {
            for (k, slot) in log_probs[..kk].iter_mut().enumerate() {
                *slot -= charge(k);
            }
        }
    } else {
        thread_local! {
            static SCRATCH: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            scratch.clear();
            scratch.extend(
                allowed
                    .iter()
                    .map(|&k| sf * stats.log_size_offset[k] as f64),
            );
            for &(g, v) in row {
                let vg = v as f64;
                let base = g as usize * kk;
                let logs = &stats.log_gene[base..base + kk];
                for (acc, &k) in scratch.iter_mut().zip(allowed) {
                    *acc += vg * logs[k] as f64;
                }
                add_cur(g as usize, base, vg);
            }
            for (&acc, &k) in scratch.iter().zip(allowed) {
                log_probs[k] = if z > 0.0 { acc - charge(k) } else { acc };
            }
        });
    }
    if score_cur {
        log_probs[cur] = cur_acc;
    }
}

/// Unrestricted variant used only from tests.
#[cfg(test)]
fn compute_log_probs(e: usize, stats: &DcPoissonStats, profiles: &Profiles, log_probs: &mut [f64]) {
    for (k, slot) in log_probs.iter_mut().enumerate().take(stats.k) {
        *slot = score_move(e, k, stats, profiles);
    }
}

/// Gumbel-max categorical sampling over the `allowed` slots of `log_probs`,
/// skipping non-finite entries. Falls back to `current` (the entity's
/// present label — always a legal "move") when no allowed slot is finite,
/// so a degenerate score vector can never produce an out-of-range label.
pub fn sample_categorical_log_restricted(
    log_probs: &[f64],
    allowed: &[usize],
    current: usize,
    rng: &mut SmallRng,
) -> usize {
    let mut best_key = f64::NEG_INFINITY;
    let mut best_idx = current;
    for &k in allowed {
        let lp = log_probs[k];
        if lp.is_finite() {
            let u: f64 = rng.random_range(1e-12..1.0_f64);
            let g = -(-u.ln()).ln();
            let key = lp + g;
            if key > best_key {
                best_key = key;
                best_idx = k;
            }
        }
    }
    best_idx
}

/// Argmax over `allowed` indices.
pub fn argmax_log_restricted(log_probs: &[f64], allowed: &[usize]) -> usize {
    let mut best = allowed[0];
    let mut best_val = log_probs[best];
    for &k in &allowed[1..] {
        if log_probs[k] > best_val {
            best = k;
            best_val = log_probs[k];
        }
    }
    best
}

/// Score of the entity's `current` block, the bar a destination has to
/// clear. A candidate list without the current block (which a
/// [`CandidateProposer`] must not return) leaves that slot stale, so the
/// bar is then absent and any finite destination clears it.
#[inline]
fn staying_score(log_probs: &[f64], allowed: &[usize], current: usize) -> f64 {
    if allowed.contains(&current) {
        log_probs[current]
    } else {
        f64::NEG_INFINITY
    }
}

/// Gumbel-max draw among the entity's `current` block and the `allowed`
/// blocks that score strictly above it; non-finite scores are skipped.
///
/// A block that does not beat the current one is never drawn, so an entity
/// with no evidence for a move stays where it is, and a sweep of draws
/// cannot shuffle entities between blocks that score alike.
fn sample_improving_log(
    log_probs: &[f64],
    allowed: &[usize],
    current: usize,
    rng: &mut SmallRng,
) -> usize {
    let cur_val = staying_score(log_probs, allowed, current);
    let mut gumbel = |lp: f64| {
        let u: f64 = rng.random_range(1e-12..1.0_f64);
        lp - (-u.ln()).ln()
    };
    let mut best_idx = current;
    let mut best_key = if cur_val.is_finite() {
        gumbel(cur_val)
    } else {
        f64::NEG_INFINITY
    };
    for &k in allowed {
        let lp = log_probs[k];
        if k != current && lp.is_finite() && lp > cur_val {
            let key = gumbel(lp);
            if key > best_key {
                best_key = key;
                best_idx = k;
            }
        }
    }
    best_idx
}

/// Best-scoring block of `allowed`, keeping `current` unless another block
/// scores strictly higher: a tie, or a score that does not compare, is no
/// evidence for a move.
fn argmax_log_staying(log_probs: &[f64], allowed: &[usize], current: usize) -> usize {
    let mut best = current;
    let mut best_val = staying_score(log_probs, allowed, current);
    for &k in allowed {
        if k != current && log_probs[k] > best_val {
            best = k;
            best_val = log_probs[k];
        }
    }
    best
}

////////////////////////////////////
// Hierarchy / relabeling helpers //
////////////////////////////////////

/// Compact a label vector to dense `0..K` ids, preserving relative order of
/// first appearance. Returns the new labels and the new `K`. Generic over
/// the key type so it serves both plain group ids (`usize`) and composite
/// keys (e.g. `(child, parent)` pairs in `project_to_refinement`).
pub fn compact_labels<K>(labels: &[K]) -> (Vec<usize>, usize)
where
    K: Copy + Eq + std::hash::Hash,
{
    let mut map: HashMap<K, usize> = HashMap::default();
    let mut next = 0usize;
    let mut out = Vec::with_capacity(labels.len());
    for &g in labels {
        let new = *map.entry(g).or_insert_with(|| {
            let id = next;
            next += 1;
            id
        });
        out.push(new);
    }
    (out, next)
}

/// For each entity, siblings at `level` = groups at `level` sharing the
/// same parent group at `level + 1`.
///
/// `refined` follows the finest → coarsest convention: `refined[0]` is the
/// finest level, `refined[refined.len() - 1]` is the coarsest. At the
/// coarsest level (index `num_levels - 1`), the virtual root groups every
/// entity together → siblings = all groups.
pub fn compute_sibling_sets(
    refined: &[Vec<usize>],
    level: usize,
    num_groups_at_level: usize,
) -> Vec<Vec<usize>> {
    let num_levels = refined.len();
    let num_entities = refined[level].len();

    if level + 1 >= num_levels {
        let all_groups: Vec<usize> = (0..num_groups_at_level).collect();
        return vec![all_groups; num_entities];
    }

    let mut parent_to_children: HashMap<usize, Vec<usize>> = HashMap::default();
    for (child, parent) in refined[level].iter().zip(refined[level + 1].iter()) {
        let entry = parent_to_children.entry(*parent).or_default();
        if !entry.contains(child) {
            entry.push(*child);
        }
    }
    for v in parent_to_children.values_mut() {
        v.sort_unstable();
    }

    (0..num_entities)
        .map(|e| {
            let parent = refined[level + 1][e];
            parent_to_children.get(&parent).cloned().unwrap_or_default()
        })
        .collect()
}

////////////////////////////////////////////
// Candidate proposer + move guard traits //
////////////////////////////////////////////

/// Propose candidate group labels each entity may move into on one sweep.
///
/// The returned slice `candidates[e]` MUST include `labels[e]` (staying put
/// is always legal) and MUST index into the current 0..k label space.
pub trait CandidateProposer {
    fn propose(&self, labels: &[usize]) -> Vec<Vec<usize>>;
}

/// Veto individual moves after they've been picked by the sweep.
///
/// Called once per *proposed* accepted move (`from != to`) before the move
/// is applied to the sufficient statistics. The destination label is
/// intentionally not passed — guards so far only need the source cluster
/// (e.g. articulation tests ask "does removing this entity disconnect
/// `from`?"). Add a destination-aware variant if a concrete use case
/// surfaces.
pub trait MoveGuard {
    fn accept_move(&self, entity: usize, from: usize, labels: &[usize]) -> bool;
}

/// Trivial guard that accepts every move. Zero-cost default used by the
/// unguarded [`refine_with_candidates`] and [`refine_with_proposer`].
pub struct NoGuard;

impl MoveGuard for NoGuard {
    #[inline(always)]
    fn accept_move(&self, _e: usize, _from: usize, _labels: &[usize]) -> bool {
        true
    }
}

/// Intersect a per-entity sibling set with a per-entity neighbor-group set,
/// with sibling fallback when the intersection is empty and guaranteed
/// inclusion of the entity's current label so staying put is always a
/// legal move.
///
/// Shared between [`crate::alg::refine_multilevel::BbknnProposer`] and pinto's
/// `GraphProposer` — both proposers differ only in how they gather
/// `neighbor_groups`, not in how they combine with siblings.
///
/// * `siblings` — sorted, deduped sibling group list.
/// * `neighbor_groups` — sorted, deduped group list from the proposer's
///   spatial/structural neighborhood.
/// * `current` — the entity's current label; appended if the intersection
///   excluded it.
pub fn intersect_with_siblings_fallback(
    siblings: &[usize],
    neighbor_groups: &[usize],
    current: usize,
) -> Vec<usize> {
    if siblings.is_empty() {
        return Vec::new();
    }
    if siblings.len() == 1 {
        return siblings.to_vec();
    }
    let intersect: Vec<usize> = siblings
        .iter()
        .copied()
        .filter(|g| neighbor_groups.binary_search(g).is_ok())
        .collect();
    if intersect.is_empty() {
        return siblings.to_vec();
    }
    if intersect.contains(&current) {
        intersect
    } else {
        let mut c = intersect;
        c.push(current);
        c.sort_unstable();
        c
    }
}

///////////////////
// Sweep drivers //
///////////////////

/// Shared sweep-driver context bundling the readonly inputs that every
/// `refine_with_*` variant needs. Kept as a struct so the driver fns don't
/// exceed clippy's argument-count threshold.
pub struct RefineContext<'a> {
    pub profiles: &'a Profiles,
    pub k: usize,
    pub params: &'a RefineParams,
    pub level_label: &'a str,
}

/// Outcome of one sweep over all entities.
#[derive(Default, Clone, Copy)]
struct SweepCounts {
    moves: usize,
    vetoed: usize,
}

/// Member-count limits of every block, fixed from the partition a
/// refinement starts with.
///
/// With `n₀` a block's starting size and `n̄` the mean size of the occupied
/// blocks, a block is never taken below
/// `ceil(size_floor · min(n₀, n̄))` members nor above
/// `ceil(size_cap · max(n₀, n̄))`. A block larger than the mean can give
/// members down to the mean and one smaller than the mean can take them up
/// to it, so sizes may even out, but a small block cannot be drained nor a
/// large one keep growing. Any positive floor is at least one member: no
/// block is emptied.
///
/// The limits are checked against the live counts when a move is applied,
/// not when it is proposed: the proposals of a Jacobi sweep are made
/// against one snapshot and would otherwise all pass together.
struct SizeLimits {
    floor: Vec<usize>,
    cap: Vec<usize>,
}

impl SizeLimits {
    fn new(count: &[usize], params: &RefineParams) -> Self {
        let occupied = count.iter().filter(|&&c| c > 0).count().max(1);
        let n_bar = count.iter().sum::<usize>() as f64 / occupied as f64;
        let size_floor = params.size_floor as f64;
        let size_cap = params.size_cap as f64;
        let floor = count
            .iter()
            .map(|&n0| {
                if size_floor > 0.0 {
                    ((size_floor * (n0 as f64).min(n_bar)).ceil() as usize).max(1)
                } else {
                    0
                }
            })
            .collect();
        let cap = count
            .iter()
            .map(|&n0| {
                if size_cap > 0.0 {
                    (size_cap * (n0 as f64).max(n_bar)).ceil() as usize
                } else {
                    usize::MAX
                }
            })
            .collect();
        Self { floor, cap }
    }

    /// May a member leave `from` at all, given the member counts?
    #[inline]
    fn can_leave(&self, count: &[usize], from: usize) -> bool {
        count[from] > self.floor[from]
    }

    /// May a member move `from → to`, given the member counts?
    #[inline]
    fn admits(&self, count: &[usize], from: usize, to: usize) -> bool {
        self.can_leave(count, from) && count[to] < self.cap[to]
    }
}

/// What every sweep of one refinement reads.
struct SweepEnv<'a, G> {
    candidates: &'a [Vec<usize>],
    profiles: &'a Profiles,
    limits: &'a SizeLimits,
    guard: &'a G,
}

/// Buffers of a Jacobi sweep, allocated once per refinement.
struct JacobiScratch {
    /// Destination each entity proposes (its current block: no move).
    proposals: Vec<usize>,
    /// Score gain of each proposed move over staying.
    gains: Vec<f32>,
    /// Entities that proposed a move, in the order they are applied.
    movers: Vec<u32>,
    /// Blocks a move of this sweep has already changed.
    changed: Vec<bool>,
    /// Score buffer for re-checking a move against the live statistics.
    log_probs: Vec<f64>,
}

impl JacobiScratch {
    fn new(num_entities: usize, k: usize) -> Self {
        Self {
            proposals: vec![0; num_entities],
            gains: vec![0.0; num_entities],
            movers: Vec::with_capacity(num_entities),
            changed: vec![false; k],
            log_probs: vec![f64::NEG_INFINITY; k],
        }
    }
}

/// Apply the proposals of a Jacobi sweep to `stats`, largest gain first.
///
/// Every proposal was scored against the pre-sweep snapshot, so it is only
/// as good as that snapshot still is when its turn comes:
/// - the size limits are read off the live member counts;
/// - a move between blocks that an earlier move of this sweep has changed
///   is scored again on the live statistics and dropped unless it still
///   gains. Without this, entities that each want to leave the same
///   company all leave together and all come back the next sweep.
///
/// Applying in order of gain makes a mixed block shed its worst-fitting
/// members first; the members that fit are re-scored against what is left
/// and stay.
fn apply_proposals<G: MoveGuard>(
    env: &SweepEnv<G>,
    scratch: &mut JacobiScratch,
    stats: &mut DcPoissonStats,
) -> SweepCounts {
    let SweepEnv {
        profiles,
        limits,
        guard,
        ..
    } = *env;
    let mut sc = SweepCounts::default();
    let JacobiScratch {
        proposals,
        gains,
        movers,
        changed,
        log_probs,
    } = scratch;
    movers.clear();
    movers.extend(
        proposals
            .iter()
            .zip(&stats.membership)
            .enumerate()
            .filter(|(_, (new, old))| new != old)
            .map(|(e, _)| e as u32),
    );
    movers.sort_unstable_by(|&a, &b| {
        gains[b as usize]
            .total_cmp(&gains[a as usize])
            .then(a.cmp(&b))
    });
    changed.iter_mut().for_each(|c| *c = false);
    for &e in movers.iter() {
        let e = e as usize;
        let new = proposals[e];
        let old = stats.membership[e];
        if !limits.admits(&stats.count, old, new) {
            continue;
        }
        if changed[old] || changed[new] {
            compute_log_probs_restricted(e, stats, profiles, &[old, new], log_probs);
            // `partial_cmp` so that a score that does not compare keeps
            // the entity in place.
            if log_probs[new].partial_cmp(&log_probs[old]) != Some(std::cmp::Ordering::Greater) {
                continue;
            }
        }
        if guard.accept_move(e, old, &stats.membership) {
            stats.delta_move(e, old, new, profiles);
            changed[old] = true;
            changed[new] = true;
            sc.moves += 1;
        } else {
            sc.vetoed += 1;
        }
    }
    sc
}

/// Gauss–Seidel sequential sweep. For each entity visited in `order`, scores
/// the candidate destinations under the live (mutated) stats, picks one via
/// `pick(log_probs, candidates, current_label)`, and applies the move through
/// the size limits and the guard. Gibbs callers shuffle `order` and pass a
/// sampling closure; greedy callers pass `0..n` and an argmax closure.
fn sweep_sequential<G, F, I>(
    env: &SweepEnv<G>,
    stats: &mut DcPoissonStats,
    log_probs: &mut [f64],
    order: I,
    mut pick: F,
) -> SweepCounts
where
    G: MoveGuard,
    F: FnMut(&[f64], &[usize], usize) -> usize,
    I: IntoIterator<Item = usize>,
{
    let SweepEnv {
        candidates,
        profiles,
        limits,
        guard,
    } = *env;
    let mut sc = SweepCounts::default();
    for e in order {
        let cand = &candidates[e];
        let old = stats.membership[e];
        if cand.len() < 2 || !limits.can_leave(&stats.count, old) {
            continue;
        }
        compute_log_probs_restricted(e, stats, profiles, cand, log_probs);
        let new = pick(log_probs, cand, old);
        if new == old || !limits.admits(&stats.count, old, new) {
            continue;
        }
        if guard.accept_move(e, old, &stats.membership) {
            stats.delta_move(e, old, new, profiles);
            sc.moves += 1;
        } else {
            sc.vetoed += 1;
        }
    }
    sc
}

/// Jacobi sweep: every entity computes its proposal against a frozen pre-sweep
/// snapshot of `stats` (rayon `par_iter`); proposals are then applied
/// sequentially by [`apply_proposals`].
///
/// `pick(log_probs, cand, e, current)` chooses the destination given the
/// restricted log-likelihoods. For Gibbs it derives a per-entity RNG from
/// `(sweep_seed, e)` and samples; for greedy it ignores `e` and argmaxes.
fn sweep_jacobi<G: MoveGuard, F>(
    env: &SweepEnv<G>,
    stats: &mut DcPoissonStats,
    scratch: &mut JacobiScratch,
    pick: F,
) -> SweepCounts
where
    F: Fn(&[f64], &[usize], usize, usize) -> usize + Sync,
{
    let SweepEnv {
        candidates,
        profiles,
        limits,
        ..
    } = *env;
    debug_assert_eq!(scratch.proposals.len(), candidates.len());
    {
        // Immutable reborrow so the rayon closure can capture `&stats: Sync`
        // while the outer `&mut` is held across the propose / apply phases.
        let stats: &DcPoissonStats = stats;
        let k = stats.k;
        scratch
            .proposals
            .par_iter_mut()
            .zip(scratch.gains.par_iter_mut())
            .enumerate()
            .with_min_len(256)
            .for_each_init(
                || vec![f64::NEG_INFINITY; k],
                |log_probs, (e, (prop, gain))| {
                    let cand = &candidates[e];
                    let current = stats.membership[e];
                    *prop = current;
                    // A block at its floor gives no member this sweep.
                    if cand.len() < 2 || !limits.can_leave(&stats.count, current) {
                        return;
                    }
                    compute_log_probs_restricted(e, stats, profiles, cand, log_probs);
                    let new = pick(log_probs, cand, e, current);
                    if new != current {
                        *prop = new;
                        *gain = (log_probs[new] - log_probs[current]) as f32;
                    }
                },
            );
    }
    apply_proposals(env, scratch, stats)
}

/// Refine `labels` in place for one level by running Gibbs + greedy sweeps
/// over `profiles` with the given pre-built `candidates` and a custom
/// [`MoveGuard`]. Returns total accepted moves across all sweeps.
///
/// An entity moves only to a block that scores above its current one by
/// more than the noise of the two block estimates (see [`DcPoissonStats`]);
/// equal scores keep it where it is. Gibbs sweeps draw among such blocks,
/// greedy sweeps take the best, and greedy sweeps stop at the first one
/// that moves nothing. A move is applied only when the size limits of
/// [`RefineParams::size_floor`] / [`RefineParams::size_cap`] allow it and
/// `guard.accept_move(...)` returns `true`; otherwise the entity stays in
/// its current group for this sweep.
///
/// This is the lowest-level driver. Pass [`NoGuard`] to get the unguarded
/// behavior; most callers should prefer the convenience wrappers
/// [`refine_with_candidates`] / [`refine_with_proposer`] /
/// [`refine_with_proposer_guarded`].
///
/// Sweep dispatch follows `ctx.params.parallel`: Jacobi (parallel proposal,
/// sequential apply) when true, Gauss–Seidel (single-threaded) when false.
/// See [`RefineParams::parallel`] for trade-offs.
pub fn refine_with_candidates_guarded<G: MoveGuard>(
    labels: &mut [usize],
    candidates: &[Vec<usize>],
    guard: &G,
    rng: &mut SmallRng,
    ctx: &RefineContext,
) -> usize {
    let RefineContext {
        profiles,
        k,
        params,
        level_label,
    } = *ctx;
    let num_entities = labels.len();
    let mut stats = DcPoissonStats::from_profiles_with(profiles, k, labels, params);
    let limits = SizeLimits::new(&stats.count, params);
    let env = SweepEnv {
        candidates,
        profiles,
        limits: &limits,
        guard,
    };
    let mut total_moves = 0usize;
    let mut total_vetoed = 0usize;

    // Reusable buffers, hoisted so that no sweep allocates: the proposal
    // scratch for Jacobi sweeps, or the score buffer and the visiting order
    // for sequential ones (greedy sequential iterates `0..num_entities`
    // directly and ignores the order).
    let mut scratch = JacobiScratch::new(if params.parallel { num_entities } else { 0 }, k);
    let mut log_probs = vec![f64::NEG_INFINITY; if params.parallel { 0 } else { k }];
    let mut order: Vec<usize> = if !params.parallel {
        (0..num_entities).collect()
    } else {
        Vec::new()
    };

    let max_sweeps = (params.num_gibbs + params.num_greedy) as u64;
    let prog_bar = legume_numeric::matrix::progress::new_progress_bar(max_sweeps)
        .with_message(format!("{level_label} sweeps"));
    // Refresh on a timer so the bar visibly animates between sweep ticks
    // (a single sweep at large D can take several seconds — without this
    // the bar is silent until the next inc(1)).
    prog_bar.enable_steady_tick(std::time::Duration::from_millis(100));

    // Jacobi per-sweep seed base: draw a single odd u64 from `rng` so the
    // sweep loop is reproducible given `params.seed` regardless of how many
    // cells `shuffle` would have consumed in the sequential path.
    let jacobi_base_seed = rng.random::<u64>() | 1;

    if params.num_gibbs > 0 {
        let mut low_sweeps = 0usize;
        for sweep in 0..params.num_gibbs {
            let sc = if params.parallel {
                let sweep_seed = jacobi_base_seed.wrapping_mul(sweep as u64 + 1);
                sweep_jacobi(
                    &env,
                    &mut stats,
                    &mut scratch,
                    |log_probs, cand, e, current| {
                        let vertex_seed = sweep_seed ^ (e as u64).wrapping_mul(2654435761);
                        let mut rng = SmallRng::seed_from_u64(vertex_seed);
                        sample_improving_log(log_probs, cand, current, &mut rng)
                    },
                )
            } else {
                order.shuffle(rng);
                sweep_sequential(
                    &env,
                    &mut stats,
                    &mut log_probs,
                    order.iter().copied(),
                    |log_probs, cand, current| sample_improving_log(log_probs, cand, current, rng),
                )
            };
            total_moves += sc.moves;
            total_vetoed += sc.vetoed;
            prog_bar.inc(1);
            if params.gibbs_stagnation > 0.0 {
                if (sc.moves as f64) < params.gibbs_stagnation * (num_entities as f64) {
                    low_sweeps += 1;
                    if low_sweeps >= 3 {
                        break;
                    }
                } else {
                    low_sweeps = 0;
                }
            }
        }
    }

    for _sweep in 0..params.num_greedy {
        let sc = if params.parallel {
            sweep_jacobi(
                &env,
                &mut stats,
                &mut scratch,
                |log_probs, cand, _e, current| argmax_log_staying(log_probs, cand, current),
            )
        } else {
            sweep_sequential(
                &env,
                &mut stats,
                &mut log_probs,
                0..num_entities,
                argmax_log_staying,
            )
        };
        total_moves += sc.moves;
        total_vetoed += sc.vetoed;
        prog_bar.inc(1);
        if sc.moves == 0 {
            break;
        }
    }
    prog_bar.finish_and_clear();

    if total_vetoed > 0 {
        log::debug!(
            "{}: {} moves, {} vetoed by MoveGuard",
            level_label,
            total_moves,
            total_vetoed
        );
    }

    labels.copy_from_slice(&stats.membership);
    total_moves
}

/// Unguarded variant of [`refine_with_candidates_guarded`].
pub fn refine_with_candidates(
    labels: &mut [usize],
    candidates: &[Vec<usize>],
    rng: &mut SmallRng,
    ctx: &RefineContext,
) -> usize {
    refine_with_candidates_guarded(labels, candidates, &NoGuard, rng, ctx)
}

/// Generic driver: ask `proposer` for candidates, then run the guarded sweep.
pub fn refine_with_proposer_guarded<P: CandidateProposer, G: MoveGuard>(
    labels: &mut [usize],
    proposer: &P,
    guard: &G,
    rng: &mut SmallRng,
    ctx: &RefineContext,
) -> usize {
    let candidates = proposer.propose(labels);
    let moves = refine_with_candidates_guarded(labels, &candidates, guard, rng, ctx);
    info!("  {}: {} DC-Poisson moves", ctx.level_label, moves);
    moves
}

/// Unguarded generic driver: ask `proposer` for candidates, then run the
/// sweep with [`NoGuard`].
pub fn refine_with_proposer<P: CandidateProposer>(
    labels: &mut [usize],
    proposer: &P,
    rng: &mut SmallRng,
    ctx: &RefineContext,
) -> usize {
    let candidates = proposer.propose(labels);
    let moves = refine_with_candidates(labels, &candidates, rng, ctx);
    info!("  {}: {} DC-Poisson moves", ctx.level_label, moves);
    moves
}

////////////////////////////
// Tests (algorithm core) //
////////////////////////////

#[cfg(test)]
#[path = "dc_poisson_tests.rs"]
mod tests;
