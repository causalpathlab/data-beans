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
//! every block near its starting size, small blocks trading members
//! rather than losing them.
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
use std::collections::VecDeque;

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
    /// sweep), single-threaded. The statistics are still built on the
    /// rayon pool; the labels are the same on any number of threads.
    ///
    /// Jacobi is biased — within a sweep, entities can't see each other's
    /// moves — but converges fine for the multilevel DC-Poisson use case
    /// and is roughly P× faster on large pb-sample counts.
    pub parallel: bool,
    /// Pseudo-count `c` added to every block total before its log
    /// (default `0.5`; values under [`LOG_EPS`] are raised to it). A block
    /// total is read against what a block of the pooled profile and the
    /// same mass would show, so a feature a block has not seen costs
    /// little where none was expected, and an empty block scores as the
    /// pooled profile. A larger `c` holds block profiles closer to the
    /// pooled one where counts are few. See [`DcPoissonStats`].
    pub prior_count: f32,
    /// Standard deviations of block-estimate noise by which a destination
    /// has to beat the current block (default `1.5`; `0` asks for any
    /// positive gain). The noise is larger for smaller blocks, so this is
    /// what keeps a small block from losing, or taking in, members on the
    /// strength of its own sampling error. See [`DcPoissonStats`].
    pub min_gain_z: f32,
    /// Fraction of `min(starting size, mean block size)` below which no
    /// block is taken, rounded up (default `0.9`); any positive value also
    /// keeps the last member of every block. `0` lets blocks shrink and
    /// empty.
    ///
    /// The refinement is asked to make a given partition purer, not to
    /// choose the number of blocks: by the likelihood alone, the members of
    /// a mixed block all do better elsewhere, and the block dissolves.
    ///
    /// A block at its floor still exchanges members: a move the limits
    /// refuse is applied together with a move the other way between the
    /// same two blocks.
    ///
    /// Both limits are taken from the sizes a call starts with, so they
    /// bound one call. Refining the output again starts from the new sizes
    /// and may take a block a step further: at the default, a tenth of its
    /// members while it has ten or more, and none below ten unless it is
    /// larger than the mean. No positive floor empties a block, over any
    /// number of calls.
    pub size_floor: f32,
    /// Multiple of `max(starting size, mean block size)` above which no
    /// block is taken, rounded up (default `1.25`). `0` lifts the limit.
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
            prior_count: 0.5,
            min_gain_z: 1.5,
            size_floor: 0.9,
            size_cap: 1.25,
        }
    }
}

////////////////////////////////////
// Sparse row-per-entity profiles //
////////////////////////////////////

/// Pseudo-count `κ` in the observed and the expected count behind a batch
/// factor (see [`Profiles::rescale_batches`]).
pub const BATCH_FACTOR_PSEUDO_COUNT: f64 = 10.0;
/// Largest batch factor, and the reciprocal of the smallest.
pub const BATCH_FACTOR_BOUND: f32 = 8.0;
/// Alternations of the batch-factor fit.
const BATCH_FACTOR_ROUNDS: usize = 4;

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

    /// Divide each entity's features by its batch's estimated factor, then
    /// recompute the size factors.
    ///
    /// `batch[e]` is the batch of entity `e`. A batch effect multiplies a
    /// feature's rate in every entity of the batch, so entities of one
    /// batch fit a block by how much of that batch it holds, whatever its
    /// content, and a refinement on the raw profiles sorts them by batch.
    ///
    /// The factors `δ` are fitted to
    /// `y_eg ~ Poisson(s_e · λ[c(e), g] · δ[b(e), g])`, `c(e)` the stratum
    /// of the entity (`strata`; one stratum without it), by alternating
    /// `λ_cg = Σ_{e ∈ c} y_eg / Σ_b T_bc δ_bg` and
    /// `δ_bg = (O_bg + κ) / (E_bg + κ)`, with `T_bc` the mass of batch `b`
    /// in stratum `c`, `O_bg` the batch's count of the feature and
    /// `E_bg = Σ_c T_bc λ_cg` the count expected of its strata. Three
    /// things hold a factor near one, where an entity is left as it is:
    /// - the pseudo-count `κ` ([`BATCH_FACTOR_PSEUDO_COUNT`]): a feature
    ///   with a few counts in a batch says little about its factor;
    /// - the replication: a batch of `n` entities gets `δ^((n − 1) / n)`,
    ///   and a batch of one entity, whose profile cannot be told from a
    ///   factor, is not touched;
    /// - a bound of [`BATCH_FACTOR_BOUND`] either way.
    ///
    /// # What the strata are for
    ///
    /// Without strata the expected count is the pooled profile's, and the
    /// fit takes every difference between a batch and the pool for a
    /// technical factor. That holds only when the batches have the same
    /// composition. A batch that is mostly one cell type has that type's
    /// features scaled down, and the type no longer looks the same in two
    /// batches that hold different shares of it. With strata that follow
    /// the composition (a coarse grouping of the entities), a batch is
    /// compared with what its own mix of strata would show, and a type
    /// found in one batch only is compared with itself and left alone.
    /// Differences in composition within a stratum are still taken for a
    /// batch factor.
    pub fn rescale_batches(&mut self, batch: &[usize], strata: Option<&[usize]>) {
        let n = self.rows.len();
        assert_eq!(batch.len(), n, "one batch per entity");
        let nb = batch.iter().copied().max().map_or(0, |b| b + 1);
        if nb < 2 {
            return;
        }
        let single = vec![0usize; if strata.is_none() { n } else { 0 }];
        let strata = strata.unwrap_or(&single);
        assert_eq!(strata.len(), n, "one stratum per entity");
        let ns = strata.iter().copied().max().map_or(0, |c| c + 1);
        let m = self.num_features;

        // O_bg, T_bc, and the entities of each batch and stratum.
        let mut observed = vec![0f64; nb * m];
        let mut mass = vec![0f64; ns * nb];
        let mut batch_members = vec![0usize; nb];
        let mut of_stratum: Vec<Vec<u32>> = vec![Vec::new(); ns];
        for (e, row) in self.rows.iter().enumerate() {
            let (b, c) = (batch[e], strata[e]);
            for &(g, v) in row {
                observed[b * m + g as usize] += v as f64;
            }
            mass[c * nb + b] += self.size_factor[e] as f64;
            batch_members[b] += 1;
            of_stratum[c].push(e as u32);
        }

        // Σ_{e ∈ c} y_eg of each stratum, on the features it shows, and the
        // batches it holds.
        let mut dense = vec![0f64; m];
        let mut touched: Vec<u32> = Vec::new();
        let mut stratum_sums: Vec<Vec<(u32, f64)>> = Vec::with_capacity(ns);
        let mut stratum_batches: Vec<Vec<(usize, f64)>> = Vec::with_capacity(ns);
        for (c, members) in of_stratum.iter().enumerate() {
            touched.clear();
            for &e in members {
                for &(g, v) in &self.rows[e as usize] {
                    if dense[g as usize] == 0.0 {
                        touched.push(g);
                    }
                    dense[g as usize] += v as f64;
                }
            }
            touched.sort_unstable();
            touched.dedup();
            stratum_sums.push(touched.iter().map(|&g| (g, dense[g as usize])).collect());
            touched.iter().for_each(|&g| dense[g as usize] = 0.0);
            stratum_batches.push(
                (0..nb)
                    .map(|b| (b, mass[c * nb + b]))
                    .filter(|&(_, t)| t > 0.0)
                    .collect(),
            );
        }

        let mut factor = vec![1f32; nb * m];
        let mut expected = vec![0f64; nb * m];
        for _ in 0..BATCH_FACTOR_ROUNDS {
            expected.iter_mut().for_each(|x| *x = 0.0);
            for (sums, present) in stratum_sums.iter().zip(&stratum_batches) {
                for &(g, sum) in sums {
                    let g = g as usize;
                    let exposure: f64 = present
                        .iter()
                        .map(|&(b, t)| t * factor[b * m + g] as f64)
                        .sum();
                    if exposure > 0.0 {
                        let rate = sum / exposure;
                        for &(b, t) in present {
                            expected[b * m + g] += t * rate;
                        }
                    }
                }
            }
            for ((f, &o), &x) in factor.iter_mut().zip(&observed).zip(&expected) {
                *f = ((o + BATCH_FACTOR_PSEUDO_COUNT) / (x + BATCH_FACTOR_PSEUDO_COUNT)) as f32;
            }
        }
        for (b, per_feature) in factor.chunks_mut(m).enumerate() {
            let members = batch_members[b].max(1) as f32;
            let trust = (members - 1.0) / members;
            for f in per_feature {
                *f = f
                    .powf(trust)
                    .clamp(1.0 / BATCH_FACTOR_BOUND, BATCH_FACTOR_BOUND);
            }
        }

        self.rows
            .par_iter_mut()
            .zip(self.size_factor.par_iter_mut())
            .zip(batch.par_iter())
            .for_each(|((row, sf), &b)| {
                let mut new_sf = 0f32;
                for (g, v) in row.iter_mut() {
                    *v /= factor[b * m + *g as usize];
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

///////////////////////////////////////////////
// Null moments of the log of a block total //
///////////////////////////////////////////////

/// Grid step of [`NullLog`] in `ln μ`.
const NULL_STEP: f64 = 1.0 / 48.0;
/// `ln μ` below which a total is zero for every purpose here.
const NULL_LN_MIN: f64 = -20.0;
/// Mean from which the moments are taken from the expansion around `μ + c`.
const NULL_MU_MAX: f64 = 64.0;

/// Mean and variance of `ln(G + c)` for `G ~ Poisson(μ)`, tabulated on a
/// grid of `ln μ` and read by linear interpolation.
///
/// The mean is what the log of a block total comes to when the block has
/// the pooled profile; subtracting it leaves a log-ratio that is zero on
/// average for such a block at every `μ`, down to totals of zero and one,
/// where an expansion around `μ` no longer describes `ln(G + c)`.
#[derive(Clone)]
struct NullLog {
    c: f64,
    /// `ln c`, rounded as the cached block logs are.
    ln_c: f64,
    /// `E ln(G + c) − ln c` at each grid point.
    mean: Vec<f64>,
    /// `Var ln(G + c)` at each grid point.
    var: Vec<f64>,
}

impl NullLog {
    fn new(c: f64) -> Self {
        let points = ((NULL_MU_MAX.ln() - NULL_LN_MIN) / NULL_STEP).ceil() as usize + 2;
        let (mean, var) = (0..points)
            .map(|i| Self::poisson_moments((NULL_LN_MIN + i as f64 * NULL_STEP).exp(), c))
            .unzip();
        Self {
            c,
            ln_c: fast_ln(c),
            mean,
            var,
        }
    }

    /// `(E ln(1 + G / c), Var ln(G + c))` for `G ~ Poisson(mu)`, by summing
    /// over the counts that carry the mass.
    fn poisson_moments(mu: f64, c: f64) -> (f64, f64) {
        let last = (mu + 12.0 * mu.sqrt() + 25.0) as usize;
        let mut p = (-mu).exp();
        let (mut m1, mut m2) = (0f64, 0f64);
        for n in 0..=last {
            let l = (n as f64 / c).ln_1p();
            m1 += p * l;
            m2 += p * l * l;
            p *= mu / (n as f64 + 1.0);
        }
        (m1, (m2 - m1 * m1).max(0.0))
    }

    /// `(E ln(G + c), Var ln(G + c))` at mean `mu`, with `ln_mu = ln(mu)`.
    #[inline]
    fn at(&self, ln_mu: f64, mu: f64) -> (f64, f64) {
        if mu >= NULL_MU_MAX {
            // Moments of ln(m + X) for a centred Poisson X around m = μ + c:
            // central moments μ, μ, 3μ² + μ.
            let m = mu + self.c;
            let r = 1.0 / m;
            let v = mu * r * r;
            let mean =
                m.ln() - 0.5 * v + v * r / 3.0 - (3.0 * mu * mu + mu) * (r * r) * (r * r) / 4.0;
            (mean, v)
        } else if ln_mu > NULL_LN_MIN {
            let pos = (ln_mu - NULL_LN_MIN) / NULL_STEP;
            let i = pos as usize;
            let f = pos - i as f64;
            (
                self.ln_c + self.mean[i] + f * (self.mean[i + 1] - self.mean[i]),
                self.var[i] + f * (self.var[i + 1] - self.var[i]),
            )
        } else {
            (self.ln_c, 0.0)
        }
    }

    /// The moments for a block of mass `size` at each class rate.
    fn row(&self, size: f64, classes: &RateClasses, mean: &mut [f64], var: &mut [f64]) {
        let ln_size = if size > 0.0 {
            size.ln()
        } else {
            f64::NEG_INFINITY
        };
        for (((m, v), &rate), &ln_rate) in mean
            .iter_mut()
            .zip(var.iter_mut())
            .zip(&classes.rate)
            .zip(&classes.ln_rate)
        {
            (*m, *v) = self.at(ln_size + ln_rate, size * rate);
        }
    }
}

/// Width of a rate class in `ln π`: four classes per octave.
const CLASS_STEP: f64 = std::f64::consts::LN_2 / 4.0;
/// Most classes a refinement keeps (44 nats of pooled rate).
const MAX_CLASSES: usize = 256;

/// Pooled rates at which the null moments of every block are kept.
///
/// The null mean of `ln(G_kg + c)` depends on the block and the feature
/// through `S_k · π_g` only. Features are therefore binned by `ln π_g` on
/// an even grid, each feature split between its two neighbouring grid
/// points, so that a block keeps one value per grid point and an entity one
/// weight per grid point. The last class is the rate zero: features that no
/// other entity shows.
#[derive(Clone)]
struct RateClasses {
    /// `ln π` of class 0, the largest rate.
    top: f64,
    /// Rate of each class; zero for the last.
    rate: Vec<f64>,
    /// `ln` of each rate; `−∞` for the last.
    ln_rate: Vec<f64>,
}

impl RateClasses {
    /// Classes covering pooled rates from `min_entry / total` to
    /// `max_pooled / total`, with an octave to spare on either side.
    fn new(max_pooled: f64, min_entry: f64, total: f64) -> Self {
        let ln2 = std::f64::consts::LN_2;
        let (top, len) = if total > 0.0 && max_pooled > 0.0 && min_entry.is_finite() {
            let top = (max_pooled / total).ln() + ln2;
            let bottom = (min_entry / total).ln() - ln2;
            let span = ((top - bottom) / CLASS_STEP).ceil() as usize;
            (top, (span + 2).clamp(3, MAX_CLASSES))
        } else {
            (0.0, 2)
        };
        let mut ln_rate: Vec<f64> = (0..len - 1).map(|j| top - j as f64 * CLASS_STEP).collect();
        let mut rate: Vec<f64> = ln_rate.iter().map(|x| x.exp()).collect();
        ln_rate.push(f64::NEG_INFINITY);
        rate.push(0.0);
        Self { top, rate, ln_rate }
    }

    fn len(&self) -> usize {
        self.rate.len()
    }

    /// Add `weight` and `weight²` of a feature of pooled rate
    /// `pooled / total` to the class sums `mass` and `sq`.
    #[inline]
    fn add(&self, pooled: f64, total: f64, weight: f64, mass: &mut [f64], sq: &mut [f64]) {
        let zero = self.len() - 1;
        if !(pooled > 0.0 && total > 0.0) {
            mass[zero] += weight;
            sq[zero] += weight * weight;
            return;
        }
        let pos = ((self.top - fast_ln(pooled / total)) / CLASS_STEP).clamp(0.0, (zero - 1) as f64);
        let lo = pos as usize;
        let f = pos - lo as f64;
        mass[lo] += weight * (1.0 - f);
        sq[lo] += weight * weight * (1.0 - f);
        if f > 0.0 {
            mass[lo + 1] += weight * f;
            sq[lo + 1] += weight * weight * f;
        }
    }
}

/// `Σ_j w_j · t_j`, in class order.
#[inline(always)]
fn class_dot(w: &[f32], t: &[f64]) -> f64 {
    w.iter().zip(t).map(|(&w, &t)| w as f64 * t).sum()
}

///////////////////////////////////
// Poisson sufficient statistics //
///////////////////////////////////

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
/// An entity is scored against block `k` by the log-likelihood ratio of the
/// block's profile to the pooled profile `π`,
/// `Σ_g y_eg · ln(θ_kg / π_g)`, with each log-ratio estimated by
///
/// `  u_kg = ln(G_kg + c) − E₀ ln(G + c),   G ~ Poisson(S_k · π_g)`.
///
/// `c = prior_count` is a pseudo-count in every block total. The second
/// term is the mean of the first for a block that has the pooled profile
/// (`NullLog`), so `u_kg` is zero on average for such a block whatever its
/// mass, and blocks of the pooled profile score alike whatever their size.
/// A plain `ln((G + a) / (S + α))` does not have that property: its mean
/// falls with the block's mass (the log of a noisy total is biased low),
/// and on a feature the block has not seen it reads a rate proportional to
/// `1 / S_k`. With some 20,000 heavy-tailed features most totals of a
/// small block are zero or one, and either effect outweighs the evidence.
///
/// `ln((G + c) / (S_k π_g + c))` is the posterior-mean rate ratio under a
/// `Gamma(c, c)` prior on `θ_kg / π_g`: a feature expected less than `c`
/// times in the block stays near the pooled rate, one expected many times
/// reads `G / (S_k π_g)`. Blocks that share a profile other than the pooled
/// one are therefore shrunk toward the pooled profile by different amounts
/// when their masses differ, and an entity of that profile fits the larger
/// one better. A smaller `c` shrinks less and is noisier.
///
/// `π_g` is taken without the entity scored (`(P_g − y_eg) / (T − s_e)`,
/// `P` the pooled totals): a rare feature's pooled rate is otherwise made
/// of the entity's own counts. A feature nothing else shows carries no
/// evidence for any block. The block scored is not taken out of `π`: a
/// block that makes up most of the pooled mass is in part its own null,
/// and is favoured over a small block of its profile.
///
/// The null mean depends on `(k, g)` through `S_k π_g` only, so it is kept
/// per block on a grid of `ln π` (`RateClasses`), and an entity carries
/// its counts summed onto the same grid: the score costs one cached log per
/// (nonzero, block) and one short dot product per block.
///
/// What remains under the null is zero-mean noise of variance
/// `Σ_g y_eg² · Var₀ ln(G_kg + c)`, read off the same grid. A destination is
/// charged `min_gain_z` standard deviations of the difference between its
/// score and the current block's,
/// `z · sqrt(Σ_g y_eg² (Var₀[cur, g] + Var₀[k, g]))`: a move has to beat
/// what the block estimates alone would produce between two blocks of one
/// profile.
///
/// The moments are those of Poisson totals. Totals of profiles weighted by
/// [`FeatureWeighting::FisherInfoNb`] are not counts, but under the negative
/// binomial model the weights come from (`w = 1 / (1 + μ φ)`, the ratio of
/// a count's mean to its variance) a weighted total again has a variance
/// equal to its mean.
///
/// Log caches are `f32` — values live in roughly `[-21, +25]` and scoring
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
    /// Feature-major `M × K`: entry `g·K + k`. Holds `ln(G_kg + c)`.
    pub log_gene: Vec<f32>,
    /// `−ln(S_k + M · LOG_EPS)`. Not part of the score, which compares a
    /// block with the pooled profile feature by feature.
    pub log_size_offset: Vec<f32>,
    /// Members per block.
    pub count: Vec<usize>,
    /// Standard deviations of estimate noise a move has to clear.
    pub min_gain_z: f64,
    /// Pseudo-count `c` in every block total.
    prior: f64,
    null: NullLog,
    classes: RateClasses,
    /// Entity-major `N × J`: counts of each entity summed per rate class.
    class_mass: Vec<f32>,
    /// Entity-major `N × J`: squared counts per rate class.
    class_sq: Vec<f32>,
    /// Block-major `K × J`: null mean of `ln(G + c)` at each class rate.
    class_null: Vec<f64>,
    /// Block-major `K × J`: null variance of `ln(G + c)` at each class rate.
    class_var: Vec<f64>,
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
        let mut min_entry = f64::INFINITY;
        for (e, row) in profiles.rows.iter().enumerate() {
            let z = membership[e];
            assert!(z < k, "membership[{}]={} out of range 0..{}", e, z, k);
            for &(g, v) in row {
                gene_sum[g as usize * k + z] += v as f64;
                pooled[g as usize] += v as f64;
                if v > 0.0 {
                    min_entry = min_entry.min(v as f64);
                }
            }
            size_sum[z] += profiles.size_factor[e] as f64;
            count[z] += 1;
        }
        let total: f64 = pooled.iter().sum();
        let max_pooled = pooled.iter().copied().fold(0f64, f64::max);
        let classes = RateClasses::new(max_pooled, min_entry, total);
        let jj = classes.len();

        // Each entity's counts on the rate grid, the pooled rate of a
        // feature taken without the entity.
        let n = profiles.rows.len();
        let mut class_mass = vec![0f32; n * jj];
        let mut class_sq = vec![0f32; n * jj];
        class_mass
            .par_chunks_mut(jj)
            .zip(class_sq.par_chunks_mut(jj))
            .zip(profiles.rows.par_iter())
            .zip(profiles.size_factor.par_iter())
            .for_each_init(
                || (vec![0f64; jj], vec![0f64; jj]),
                |(mass, sq), (((mass_out, sq_out), row), &sf)| {
                    mass.iter_mut().for_each(|x| *x = 0.0);
                    sq.iter_mut().for_each(|x| *x = 0.0);
                    let rest = total - sf as f64;
                    for &(g, v) in row {
                        if v > 0.0 {
                            let p = pooled[g as usize];
                            let others = p - v as f64;
                            // a rounding residue of the entity's own count is no one else's
                            let others = if others > 1e-9 * p { others } else { 0.0 };
                            classes.add(others, rest, v as f64, mass, sq);
                        }
                    }
                    for (o, &x) in mass_out.iter_mut().zip(mass.iter()) {
                        *o = x as f32;
                    }
                    for (o, &x) in sq_out.iter_mut().zip(sq.iter()) {
                        *o = x as f32;
                    }
                },
            );

        let prior = (params.prior_count as f64).max(LOG_EPS);
        let mut stats = Self {
            k,
            num_features: m,
            membership: membership.to_vec(),
            gene_sum,
            size_sum,
            log_gene: vec![0f32; k * m],
            log_size_offset: vec![0f32; k],
            count,
            min_gain_z: params.min_gain_z.max(0.0) as f64,
            prior,
            null: NullLog::new(prior),
            classes,
            class_mass,
            class_sq,
            class_null: vec![0f64; k * jj],
            class_var: vec![0f64; k * jj],
        };
        stats.refresh_caches();
        stats
    }

    /// Refresh the per-block caches of block `k`.
    #[inline]
    fn refresh_block(&mut self, k: usize) {
        let jj = self.classes.len();
        self.log_size_offset[k] =
            -fast_ln(self.size_sum[k] + self.num_features as f64 * LOG_EPS) as f32;
        self.null.row(
            self.size_sum[k],
            &self.classes,
            &mut self.class_null[k * jj..(k + 1) * jj],
            &mut self.class_var[k * jj..(k + 1) * jj],
        );
    }

    /// Rebuild every cache from the running sums.
    fn refresh_caches(&mut self) {
        let c = self.prior;
        self.log_gene
            .par_iter_mut()
            .zip(self.gene_sum.par_iter())
            .with_min_len(1 << 14)
            .for_each(|(l, &s)| *l = fast_ln(s + c) as f32);
        for b in 0..self.k {
            self.refresh_block(b);
        }
    }

    /// Apply an entity's reassignment and refresh only the affected log entries.
    pub fn delta_move(&mut self, e: usize, k_from: usize, k_to: usize, profiles: &Profiles) {
        if k_from == k_to {
            return;
        }
        let kk = self.k;
        let c = self.prior;

        // Subtractions clamp at zero: the running sums accumulate rounding
        // residues, and a block drained back to (true) zero can otherwise be
        // left with a negative residue larger than its pseudo-count —
        // feeding `ln(negative) = NaN` into the log caches.
        for &(g, v) in &profiles.rows[e] {
            let base = g as usize * kk;
            let (i_from, i_to) = (base + k_from, base + k_to);
            self.gene_sum[i_from] = (self.gene_sum[i_from] - v as f64).max(0.0);
            self.gene_sum[i_to] += v as f64;
            self.log_gene[i_from] = fast_ln(self.gene_sum[i_from] + c) as f32;
            self.log_gene[i_to] = fast_ln(self.gene_sum[i_to] + c) as f32;
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

    /// The rate-class weights of entity `e`: its counts and squared counts.
    #[inline(always)]
    fn entity_classes(&self, e: usize) -> (&[f32], &[f32]) {
        let jj = self.classes.len();
        (
            &self.class_mass[e * jj..(e + 1) * jj],
            &self.class_sq[e * jj..(e + 1) * jj],
        )
    }

    /// The null moments of block `k` at each class rate.
    #[inline(always)]
    fn block_classes(&self, k: usize) -> (&[f64], &[f64]) {
        let jj = self.classes.len();
        (
            &self.class_null[k * jj..(k + 1) * jj],
            &self.class_var[k * jj..(k + 1) * jj],
        )
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
    let jj = stats.classes.len();
    let sf = profiles.size_factor[e] as f64;
    let row = &profiles.rows[e];
    let cur = stats.membership[e];
    let (mass, sq) = stats.entity_classes(e);
    let (mut loo_mean, mut loo_var) = ([0f64; MAX_CLASSES], [0f64; MAX_CLASSES]);
    let loo_size = (stats.size_sum[cur] - sf).max(0.0);
    stats.null.row(
        loo_size,
        &stats.classes,
        &mut loo_mean[..jj],
        &mut loo_var[..jj],
    );
    if cur == k {
        let mut acc = -class_dot(mass, &loo_mean[..jj]);
        for &(g, v) in row {
            let vg = v as f64;
            let loo = (stats.gene_sum[stats.idx(k, g as usize)] - vg).max(0.0);
            acc += vg * fast_ln(loo + stats.prior);
        }
        acc
    } else {
        let (null, var) = stats.block_classes(k);
        let mut acc = -class_dot(mass, null);
        for &(g, v) in row {
            acc += v as f64 * stats.log_gene[stats.idx(k, g as usize)] as f64;
        }
        if stats.min_gain_z > 0.0 {
            let noise = class_dot(sq, &loo_var[..jj]) + class_dot(sq, var);
            acc -= stats.min_gain_z * noise.sqrt();
        }
        acc
    }
}

/// Log-probability of placing entity `e` into each of the `allowed` blocks.
///
/// `  s(e, k) = Σ_{g : y_eg > 0} y_eg · (ln(G⁻ᵉ[k, g] + c) − E₀ ln(G + c)),   G ~ Poisson(S⁻ᵉ[k] · π⁻ᵉ[g])`
///
/// the log-likelihood ratio of block `k`'s profile to the pooled profile
/// `π`, each log-ratio centred on its mean for a block of the pooled
/// profile (see [`DcPoissonStats`]). `⁻ᵉ` means entity `e`'s own
/// contribution is excluded: candidate blocks never contain `e` and read
/// the log caches, while `e`'s *current* block subtracts `e`'s row before
/// taking logs (leave-one-out) — otherwise "stay put" earns a
/// self-inclusion bonus that anchors large entities.
///
/// Every block other than the current one is then charged
/// `min_gain_z` standard deviations of the estimate noise of the
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
/// reuses one buffer across sweeps). The sweeps of this module pick a block
/// from the `allowed` slots and the current block's slot only, as do the
/// public pickers [`sample_categorical_log_restricted`],
/// [`argmax_log_restricted`] and [`argmax_log_restricted_from`].
pub fn compute_log_probs_restricted(
    e: usize,
    stats: &DcPoissonStats,
    profiles: &Profiles,
    allowed: &[usize],
    log_probs: &mut [f64],
) {
    let kk = stats.k;
    let jj = stats.classes.len();
    let sf = profiles.size_factor[e] as f64;
    let row = &profiles.rows[e];
    let cur = stats.membership[e];
    let c = stats.prior;
    let (mass, sq) = stats.entity_classes(e);
    // Every block 0..K is scored straight into `log_probs`, so repeats in
    // `allowed` (which then misses some block) are harmless.
    let dense = allowed.len() == kk;
    let score_cur = dense || allowed.contains(&cur);

    // Null moments of the current block without the entity.
    let z = stats.min_gain_z;
    let (mut loo_mean, mut loo_var) = ([0f64; MAX_CLASSES], [0f64; MAX_CLASSES]);
    if score_cur || z > 0.0 {
        let loo_size = (stats.size_sum[cur] - sf).max(0.0);
        stats.null.row(
            loo_size,
            &stats.classes,
            &mut loo_mean[..jj],
            &mut loo_var[..jj],
        );
    }

    // Leave-one-out score of the current block.
    let mut cur_acc = if score_cur {
        -class_dot(mass, &loo_mean[..jj])
    } else {
        0.0
    };
    let mut add_cur = |base: usize, vg: f64| {
        if score_cur {
            let loo = (stats.gene_sum[base + cur] - vg).max(0.0);
            cur_acc += vg * fast_ln(loo + c);
        }
    };

    // Start of a block's sum, and the estimate noise of a move into it.
    let start = |k: usize| -class_dot(mass, stats.block_classes(k).0);
    let noise_cur = if z > 0.0 {
        class_dot(sq, &loo_var[..jj])
    } else {
        0.0
    };
    let charge = |k: usize| z * (noise_cur + class_dot(sq, stats.block_classes(k).1)).sqrt();

    // Cached (non-leave-one-out) scores for every allowed block, including
    // the current one, which is overwritten at the end.
    if dense {
        for (k, slot) in log_probs[..kk].iter_mut().enumerate() {
            *slot = start(k);
        }
        for &(g, v) in row {
            let vg = v as f64;
            let base = g as usize * kk;
            let logs = &stats.log_gene[base..base + kk];
            for (slot, &l) in log_probs[..kk].iter_mut().zip(logs) {
                *slot += vg * l as f64;
            }
            add_cur(base, vg);
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
            scratch.extend(allowed.iter().map(|&k| start(k)));
            for &(g, v) in row {
                let vg = v as f64;
                let base = g as usize * kk;
                let logs = &stats.log_gene[base..base + kk];
                for (acc, &k) in scratch.iter_mut().zip(allowed) {
                    *acc += vg * logs[k] as f64;
                }
                add_cur(base, vg);
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

/// Argmax over `allowed` indices that leaves `current` only for a strictly
/// better block: a tie with the current label (an entity with no evidence
/// scores every block alike) is not a move. With `current` outside
/// `allowed`, ties go to the first allowed index.
pub fn argmax_log_restricted_from(log_probs: &[f64], allowed: &[usize], current: usize) -> usize {
    let mut best = if allowed.contains(&current) {
        current
    } else {
        allowed[0]
    };
    let mut best_val = log_probs[best];
    for &k in allowed {
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
/// is applied to the sufficient statistics. For the two moves of an
/// exchange between two blocks, the second is asked about with the first
/// already applied, and the first is undone if the second is vetoed. The
/// destination label is
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
/// `max(1, ceil(size_floor · min(n₀, n̄)))` members nor above
/// `ceil(size_cap · max(n₀, n̄))`. A block larger than the mean can give
/// members down to the mean and one smaller than the mean can take them up
/// to it, so sizes may even out, but a small block cannot be drained nor a
/// large one keep growing. A floor of at least one member keeps every
/// block occupied.
///
/// Rounded up, the floor of a small block is its starting size (below ten
/// members at the default nine tenths): such a block does not shrink. It
/// still takes members up to its cap, gives one for each it has taken, and
/// trades a member for one of another block (see [`Mover`]), so a member
/// that does not fit leaves when another comes for its place.
///
/// The fractions are read in millionths and the products taken in
/// integers, so that a decimal fraction gives the limit it gives on paper
/// (`0.8 × 10` is 8, where the `f32` nearest to 0.8 widened to `f64` is a
/// little above 0.8, and rounds up to 9).
///
/// The limits are checked against the live counts when a move is applied,
/// not when it is proposed: the proposals of a Jacobi sweep are made
/// against one snapshot and would otherwise all pass together.
struct SizeLimits {
    floor: Vec<usize>,
    cap: Vec<usize>,
}

/// `x` in millionths; zero for a negative or undefined `x`.
fn millionths(x: f32) -> u128 {
    (x as f64 * 1e6).round().max(0.0) as u128
}

impl SizeLimits {
    fn new(count: &[usize], params: &RefineParams) -> Self {
        const UNIT: u128 = 1_000_000;
        // Sizes as multiples of 1 / occupied, in which the mean is `total`.
        let occupied = count.iter().filter(|&&c| c > 0).count().max(1) as u128;
        let total = count.iter().sum::<usize>() as u128;
        let size_floor = millionths(params.size_floor);
        let size_cap = millionths(params.size_cap);
        let floor = count
            .iter()
            .map(|&n0| {
                if size_floor > 0 {
                    let base = (n0 as u128 * occupied).min(total);
                    ((size_floor * base).div_ceil(UNIT * occupied) as usize).max(1)
                } else {
                    0
                }
            })
            .collect();
        let cap = count
            .iter()
            .map(|&n0| {
                if size_cap > 0 {
                    let base = (n0 as u128 * occupied).max(total);
                    (size_cap * base).div_ceil(UNIT * occupied) as usize
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

/// Buffers of the sweeps, allocated once per refinement.
struct SweepScratch {
    /// Destination each entity proposes in a Jacobi sweep (its current
    /// block: no move).
    proposals: Vec<usize>,
    /// Score gain of each proposed move over staying.
    gains: Vec<f32>,
    /// Entities that proposed a move, in the order they are applied.
    movers: Vec<u32>,
    /// Movers the size limits refused on their turn.
    deferred: Vec<u32>,
    /// Blocks a move of this sweep has already changed.
    changed: Vec<bool>,
    /// Scores of one entity, by block.
    log_probs: Vec<f64>,
    /// Refused moves by `(from, to)`, each waiting for one the other way.
    waiting: HashMap<(usize, usize), VecDeque<u32>>,
}

impl SweepScratch {
    fn new(num_entities: usize, k: usize, parallel: bool) -> Self {
        let per_entity = if parallel { num_entities } else { 0 };
        Self {
            proposals: vec![0; per_entity],
            gains: vec![0.0; per_entity],
            movers: Vec::with_capacity(per_entity),
            deferred: Vec::new(),
            changed: vec![false; k],
            log_probs: vec![f64::NEG_INFINITY; k],
            waiting: HashMap::default(),
        }
    }
}

/// What became of a move handed to [`Mover::settle`].
#[derive(PartialEq)]
enum Settled {
    /// Applied, alone or as one half of an exchange.
    Moved,
    /// Dropped: no gain left on the live statistics, or vetoed.
    Stays,
    /// Refused by the size limits and left waiting for a counterpart.
    Waits,
}

/// Applies the moves of one sweep to the live statistics, one at a time.
///
/// A move the size limits admit is applied if the guard accepts it. A move
/// they refuse waits: when a move the other way between the same two blocks
/// turns up, the two are applied together as an exchange, which changes no
/// member count. A block at its floor, as every small block is, can
/// thereby still trade a member that does not fit for one that does.
struct Mover<'a, G> {
    profiles: &'a Profiles,
    limits: &'a SizeLimits,
    guard: &'a G,
    stats: &'a mut DcPoissonStats,
    changed: &'a mut [bool],
    log_probs: &'a mut [f64],
    waiting: &'a mut HashMap<(usize, usize), VecDeque<u32>>,
    counts: SweepCounts,
}

impl<G: MoveGuard> Mover<'_, G> {
    /// Does `e`, a member of `from`, score higher in `to` on the live
    /// statistics? A score that does not compare is no gain.
    fn gains(&mut self, e: usize, from: usize, to: usize) -> bool {
        compute_log_probs_restricted(e, self.stats, self.profiles, &[from, to], self.log_probs);
        self.log_probs[to].partial_cmp(&self.log_probs[from]) == Some(std::cmp::Ordering::Greater)
    }

    /// Ask the guard about moving `e` out of `from`.
    fn accepted(&mut self, e: usize, from: usize) -> bool {
        let yes = self.guard.accept_move(e, from, &self.stats.membership);
        if !yes {
            self.counts.vetoed += 1;
        }
        yes
    }

    fn apply(&mut self, e: usize, from: usize, to: usize) {
        self.stats.delta_move(e, from, to, self.profiles);
        self.changed[from] = true;
        self.changed[to] = true;
    }

    /// Settle the move of `e` to block `to`. With `recheck`, the move was
    /// scored against statistics that have changed since, and is dropped
    /// unless it still gains.
    fn settle(&mut self, e: usize, to: usize, recheck: bool) -> Settled {
        let from = self.stats.membership[e];
        if from == to || (recheck && !self.gains(e, from, to)) {
            return Settled::Stays;
        }
        if self.limits.admits(&self.stats.count, from, to) {
            if !self.accepted(e, from) {
                return Settled::Stays;
            }
            self.apply(e, from, to);
            self.counts.moves += 1;
            return Settled::Moved;
        }
        // An exchange with a member of `to` that waits to come here: tried
        // with `e` moved, and undone unless the counterpart then gains and
        // may go.
        while let Some(f) = self
            .waiting
            .get_mut(&(to, from))
            .and_then(VecDeque::pop_front)
        {
            let f = f as usize;
            if self.stats.membership[f] != to {
                continue;
            }
            if !self.accepted(e, from) {
                self.waiting
                    .entry((to, from))
                    .or_default()
                    .push_front(f as u32);
                return Settled::Stays;
            }
            self.stats.delta_move(e, from, to, self.profiles);
            if self.gains(f, to, from) && self.accepted(f, to) {
                self.apply(f, to, from);
                self.counts.moves += 2;
                return Settled::Moved;
            }
            self.stats.delta_move(e, to, from, self.profiles);
        }
        self.waiting
            .entry((from, to))
            .or_default()
            .push_back(e as u32);
        Settled::Waits
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
///
/// A move the limits refuse on its turn is tried once more after the
/// others, when the block it leaves may have taken a member (see [`Mover`]
/// for the exchanges).
fn apply_proposals<G: MoveGuard>(
    env: &SweepEnv<G>,
    scratch: &mut SweepScratch,
    stats: &mut DcPoissonStats,
) -> SweepCounts {
    let SweepScratch {
        proposals,
        gains,
        movers,
        deferred,
        changed,
        log_probs,
        waiting,
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
    deferred.clear();
    waiting.clear();
    let mut mover = Mover {
        profiles: env.profiles,
        limits: env.limits,
        guard: env.guard,
        stats,
        changed,
        log_probs,
        waiting,
        counts: SweepCounts::default(),
    };
    for &e in movers.iter() {
        let (e, to) = (e as usize, proposals[e as usize]);
        let from = mover.stats.membership[e];
        let recheck = mover.changed[from] || mover.changed[to];
        if mover.settle(e, to, recheck) == Settled::Waits {
            deferred.push(e as u32);
        }
    }
    if mover.counts.moves > 0 {
        mover.waiting.clear();
        for &e in deferred.iter() {
            mover.settle(e as usize, proposals[e as usize], true);
        }
    }
    mover.counts
}

/// Gauss–Seidel sequential sweep. For each entity visited in `order`, scores
/// the candidate destinations under the live (mutated) stats, picks one via
/// `pick(log_probs, candidates, current_label)`, and applies the move through
/// the size limits and the guard ([`Mover`]). Gibbs callers shuffle `order`
/// and pass a sampling closure; greedy callers pass `0..n` and an argmax
/// closure.
fn sweep_sequential<G, F, I>(
    env: &SweepEnv<G>,
    stats: &mut DcPoissonStats,
    scratch: &mut SweepScratch,
    order: I,
    mut pick: F,
) -> SweepCounts
where
    G: MoveGuard,
    F: FnMut(&[f64], &[usize], usize) -> usize,
    I: IntoIterator<Item = usize>,
{
    scratch.waiting.clear();
    let mut mover = Mover {
        profiles: env.profiles,
        limits: env.limits,
        guard: env.guard,
        stats,
        changed: &mut scratch.changed,
        log_probs: &mut scratch.log_probs,
        waiting: &mut scratch.waiting,
        counts: SweepCounts::default(),
    };
    for e in order {
        let cand = &env.candidates[e];
        if cand.len() < 2 {
            continue;
        }
        let old = mover.stats.membership[e];
        compute_log_probs_restricted(e, mover.stats, env.profiles, cand, mover.log_probs);
        let new = pick(mover.log_probs, cand, old);
        // Picked on the live statistics: nothing to check again.
        mover.settle(e, new, false);
    }
    mover.counts
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
    scratch: &mut SweepScratch,
    pick: F,
) -> SweepCounts
where
    F: Fn(&[f64], &[usize], usize, usize) -> usize + Sync,
{
    let SweepEnv {
        candidates,
        profiles,
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
                    // Members of a block at its floor propose too: the
                    // block may take a member earlier in the same sweep.
                    if cand.len() < 2 {
                        return;
                    }
                    compute_log_probs_restricted(e, stats, profiles, cand, log_probs);
                    let new = pick(log_probs, cand, e, current);
                    if new != current {
                        *prop = new;
                        *gain = (log_probs[new] - staying_score(log_probs, cand, current)) as f32;
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
/// [`RefineParams::size_floor`] / [`RefineParams::size_cap`] allow it, on
/// its own or together with a move the other way between the same two
/// blocks, and `guard.accept_move(...)` returns `true`; otherwise the
/// entity stays in its current group for this sweep.
///
/// The limits are those of the sizes `labels` comes in with, so they bound
/// this call, not a sequence of calls on its own output (see
/// [`RefineParams::size_floor`]).
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

    // Reusable buffers, hoisted so that no sweep allocates anew: the
    // proposals of Jacobi sweeps, or the visiting order of sequential ones
    // (greedy sequential iterates `0..num_entities` directly and ignores
    // the order).
    let mut scratch = SweepScratch::new(num_entities, k, params.parallel);
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
                    &mut scratch,
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
                &mut scratch,
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
