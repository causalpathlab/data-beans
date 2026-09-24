//! Grouping features so a model answers for a group instead of each feature.
//!
//! **A coarsening is fixed; a module is learned.** That is the line senna
//! draws between its two kinds of feature grouping, and the reason both words
//! exist. A coarsening's membership is read off the data before training and
//! never moves, which is what lets a decoder be keyed to it and what lets a
//! continued fit inherit it verbatim. A module's membership is a parameter:
//! the masked encoder's `--gene-modules` learns centroids and re-derives
//! membership from them every step. Neither word should be used for the other,
//! and a grouping that learns its membership is a module however it is built.
//!
//! What a coarsening fixes is the assignment, not the meaning: a group's
//! embedding is the mean of its members' and moves at every step.
//!
//! A coarsening is built from the finest pseudobulks' counts by
//! [`coarsen_features`]: features with no grouping evidence form one
//! background group, and the rest are grouped by k-means on their residual
//! profiles (see that function for the method).

use clap::Args;
use legume_numeric::matrix::dmatrix_util::build_columns_par;
use legume_numeric::matrix::kmeans::kmeans_centroids_seeded;
use legume_numeric::matrix::rand_util::mix_seed;
use legume_numeric::matrix::traits::SampleOps;
use log::{debug, info};
use nalgebra::DMatrix;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

type CscMat = nalgebra_sparse::CscMatrix<f32>;

/// Maps D fine features to d coarse coarse features and back.
#[derive(Clone, Serialize, Deserialize)]
pub struct FeatureCoarsening {
    /// For each original feature, the coarse group index it belongs to.
    pub fine_to_coarse: Vec<usize>,
    /// For each coarse group, the list of original feature indices.
    pub coarse_to_fine: Vec<Vec<usize>>,
    /// Number of coarse coarse features (d).
    pub num_coarse: usize,
}

impl FeatureCoarsening {
    /// Build the two-way map from the fine → coarse assignment alone.
    ///
    /// The one place the inverse is derived, and the one place a stray group
    /// index is caught: every consumer indexes `coarse_to_fine` by the
    /// assignment, so an out-of-range entry would otherwise panic at first use.
    pub fn from_fine_to_coarse(
        fine_to_coarse: Vec<usize>,
        num_coarse: usize,
    ) -> anyhow::Result<Self> {
        let mut coarse_to_fine = vec![Vec::new(); num_coarse];
        for (f, &c) in fine_to_coarse.iter().enumerate() {
            anyhow::ensure!(
                c < num_coarse,
                "feature coarsening: feature {f} is assigned to group {c} of {num_coarse}"
            );
            coarse_to_fine[c].push(f);
        }
        Ok(Self {
            fine_to_coarse,
            coarse_to_fine,
            num_coarse,
        })
    }

    /// This coarsening carried onto a different fine axis, by name.
    ///
    /// `new_to_old[g]` is the position on this coarsening's axis of gene `g`
    /// of the new axis, or `None` for a gene it never covered. A known gene
    /// keeps its group. An unknown one joins the group whose known members it
    /// most resembles, by cosine between `unit_profiles` — one unit vector per
    /// gene of the NEW axis, in whatever reading of a profile the caller uses —
    /// and each group's centroid of its known members' vectors. A gene with no
    /// profile (a zero vector) carries nothing to place it by and goes to the
    /// group with the most known members, which perturbs the fit least. A
    /// group none of whose members survived cannot attract anything; it keeps
    /// its index and stays empty of new genes, because whatever is keyed to
    /// the groups (a decoder, say) still has a slot for it.
    ///
    /// The group count is unchanged by construction.
    pub fn grow_by_profile(
        &self,
        new_to_old: &[Option<usize>],
        unit_profiles: &[Vec<f32>],
    ) -> anyhow::Result<FeatureCoarsening> {
        let d_new = new_to_old.len();
        anyhow::ensure!(
            unit_profiles.len() == d_new,
            "feature coarsening growth: {} profiles for {d_new} features",
            unit_profiles.len(),
        );
        let k = self.num_coarse;
        let n_pb = unit_profiles.first().map_or(0, Vec::len);

        // Known genes keep their group; their unit profiles sum into the
        // group's centroid, one contiguous column per group.
        let mut fine_to_coarse = vec![usize::MAX; d_new];
        let mut centroid = DMatrix::<f32>::zeros(n_pb, k);
        let mut members = vec![0usize; k];
        for (g, old) in new_to_old.iter().enumerate() {
            let Some(p) = old else { continue };
            anyhow::ensure!(
                *p < self.fine_to_coarse.len(),
                "feature coarsening growth: feature {g} maps to {p}, beyond the {} covered",
                self.fine_to_coarse.len(),
            );
            let m = self.fine_to_coarse[*p];
            fine_to_coarse[g] = m;
            members[m] += 1;
            for (c, v) in centroid.column_mut(m).iter_mut().zip(&unit_profiles[g]) {
                *c += v;
            }
        }
        let live: Vec<usize> = (0..k).filter(|&m| members[m] > 0).collect();
        anyhow::ensure!(
            !live.is_empty(),
            "feature coarsening growth: no feature of the new axis is covered, so the groups \
             cannot be placed on it"
        );
        let fallback = live
            .iter()
            .copied()
            .max_by_key(|&m| members[m])
            .expect("a live group exists");
        let mut centroid = centroid.select_columns(&live);
        for mut c in centroid.column_iter_mut() {
            let nrm = c.norm();
            if nrm > 0.0 {
                c /= nrm;
            }
        }

        // Unknown genes: one product of their unit profiles against the live
        // centroids, then a row-wise argmax.
        let new: Vec<usize> = (0..d_new).filter(|&g| new_to_old[g].is_none()).collect();
        let u = DMatrix::from_fn(new.len(), n_pb, |i, j| unit_profiles[new[i]][j]);
        let scores = u * centroid;
        for (i, &g) in new.iter().enumerate() {
            let row = scores.row(i);
            let best = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .expect("a live group exists");
            fine_to_coarse[g] = if row.iter().all(|&v| v == 0.0) {
                fallback
            } else {
                live[best.0]
            };
        }
        debug!(
            "feature coarsening growth: {} of {d_new} features known, {} placed by profile into \
             {k} groups ({} groups had no surviving member)",
            d_new - new.len(),
            new.len(),
            k - live.len(),
        );
        Self::from_fine_to_coarse(fine_to_coarse, k)
    }

    /// Aggregate columns of an [N, D] matrix → [N, d] by summing
    /// features within each coarse group.
    pub fn aggregate_columns_nd(&self, data_nd: &DMatrix<f32>) -> DMatrix<f32> {
        let n = data_nd.nrows();
        build_columns_par(n, self.num_coarse, |c, col| {
            for &fine in &self.coarse_to_fine[c] {
                let src = data_nd.column(fine);
                for (dst, src_v) in col.iter_mut().zip(src.iter()) {
                    *dst += *src_v;
                }
            }
        })
    }

    /// Aggregate rows of a [D, S] matrix → [d, S] by summing
    /// features within each coarse group.
    pub fn aggregate_rows_ds(&self, data_ds: &DMatrix<f32>) -> DMatrix<f32> {
        let s = data_ds.ncols();
        build_columns_par(self.num_coarse, s, |j, col| {
            let src_col = data_ds.column(j);
            for (fine, &coarse) in self.fine_to_coarse.iter().enumerate() {
                col[coarse] += src_col[fine];
            }
        })
    }

    /// Expand log-probability dictionary [d, K] → [D, K].
    ///
    /// For fine feature `f` in group `c` (size `g`):
    ///   `expanded[f, k] = coarse[c, k] - ln(g)`
    ///
    /// After exponentiation, probabilities split evenly within each group:
    ///   `β[f, k] = β_coarse[c, k] / g`
    pub fn expand_log_dict_dk(&self, log_dict_dk: &DMatrix<f32>, d_fine: usize) -> DMatrix<f32> {
        let k = log_dict_dk.ncols();
        build_columns_par(d_fine, k, |kk, col| {
            let src_col = log_dict_dk.column(kk);
            for (c, fine_indices) in self.coarse_to_fine.iter().enumerate() {
                let val = src_col[c] - (fine_indices.len() as f32).ln();
                for &f in fine_indices {
                    col[f] = val;
                }
            }
        })
    }

    /// Expand a per-group `[d, K]` table to `[D, K]` by giving every fine
    /// feature its group's row unchanged.
    ///
    /// The counterpart to [`Self::expand_log_dict_dk`], for a table that is
    /// NOT a log-probability and must not be split across the group's
    /// members: factor loadings multiply a latent rather than carrying mass,
    /// so a group's loading IS each of its features' loading. Where such a
    /// model also has a per-feature offset, that offset is where the split
    /// belongs.
    pub fn expand_rows_dk(&self, table_dk: &DMatrix<f32>, d_fine: usize) -> DMatrix<f32> {
        let k = table_dk.ncols();
        build_columns_par(d_fine, k, |kk, col| {
            let src_col = table_dk.column(kk);
            for (c, fine_indices) in self.coarse_to_fine.iter().enumerate() {
                for &f in fine_indices {
                    col[f] = src_col[c];
                }
            }
        })
    }

    /// Aggregate a sparse [D, n] CSC matrix → dense [d, n] by summing
    /// rows within each coarse group. Efficient: O(nnz) work.
    pub fn aggregate_sparse_csc(&self, data_dn: &CscMat) -> DMatrix<f32> {
        let n = data_dn.ncols();
        build_columns_par(self.num_coarse, n, |j, col| {
            let src = data_dn.col(j);
            for (&row, &val) in src.row_indices().iter().zip(src.values().iter()) {
                col[self.fine_to_coarse[row]] += val;
            }
        })
    }
}

////////////////////////////////////
// Building a coarsening from counts //
////////////////////////////////////

/// A feature is informative when its homogeneity deviance exceeds its degrees
/// of freedom by this many standard deviations (`(D − df)/√(2·df)`).
const INFORMATIVE_Z: f64 = 5.0;
/// Lloyd iterations of every k-means here.
const KMEANS_ITER: usize = 30;
/// Widest residual profile k-means runs on directly; a wider one is sketched
/// down to [`SKETCH_DIM`] with a seeded Gaussian first.
const MAX_PROFILE_DIM: usize = 1024;
const SKETCH_DIM: usize = 64;

/// Per feature, whether its counts carry grouping evidence: the Poisson
/// deviance of row `g` of `counts` against the flat rate `E_gs = T_g·n_s/N`
/// exceeds its `S − 1` degrees of freedom by [`INFORMATIVE_Z`] standard
/// deviations. A feature never counted is not informative.
pub fn informative_features(counts: &DMatrix<f32>, sizes: &[f32]) -> Vec<bool> {
    let total_size: f64 = sizes.iter().map(|&n| f64::from(n)).sum();
    let live: Vec<usize> = (0..sizes.len()).filter(|&s| sizes[s] > 0.0).collect();
    if live.len() < 2 || total_size <= 0.0 {
        return vec![false; counts.nrows()];
    }
    let df = (live.len() - 1) as f64;
    (0..counts.nrows())
        .into_par_iter()
        .map(|g| {
            let t: f64 = live.iter().map(|&s| f64::from(counts[(g, s)])).sum();
            if t <= 0.0 {
                return false;
            }
            let dev: f64 = live
                .iter()
                .map(|&s| {
                    let n = f64::from(counts[(g, s)]);
                    let e = t * f64::from(sizes[s]) / total_size;
                    let log_term = if n > 0.0 { n * (n / e).ln() } else { 0.0 };
                    2.0 * (log_term - (n - e))
                })
                .sum();
            (dev - df) / (2.0 * df).sqrt() > INFORMATIVE_Z
        })
        .collect()
}

/// Unit-length Pearson residual profiles `(n − E)/√E` of the `rows` of
/// `counts`, clipped to `±√S`, sketched to [`SKETCH_DIM`] when wider than
/// [`MAX_PROFILE_DIM`].
fn residual_profiles(
    counts: &DMatrix<f32>,
    sizes: &[f32],
    rows: &[usize],
    seed: u64,
) -> DMatrix<f32> {
    let n_pb = sizes.len();
    let total_size: f32 = sizes.iter().sum();
    let clip = (n_pb as f32).sqrt();
    let mut z = DMatrix::<f32>::zeros(rows.len(), n_pb);
    for (i, &g) in rows.iter().enumerate() {
        let t: f32 = counts.row(g).sum();
        for s in 0..n_pb {
            let e = t * sizes[s] / total_size.max(f32::MIN_POSITIVE);
            if e > 0.0 {
                z[(i, s)] = ((counts[(g, s)] - e) / e.sqrt()).clamp(-clip, clip);
            }
        }
    }
    if n_pb > MAX_PROFILE_DIM {
        let basis = DMatrix::<f32>::rnorm_seeded(n_pb, SKETCH_DIM, mix_seed(seed, 0x5343_4854));
        z = (&z * basis) / (SKETCH_DIM as f32).sqrt();
    }
    unit_rows(&mut z);
    z
}

fn unit_rows(z: &mut DMatrix<f32>) {
    for mut row in z.row_iter_mut() {
        let norm = row.norm();
        if norm > 0.0 {
            row /= norm;
        }
    }
}

/// Nested feature coarsenings of pseudobulk `counts` `[D × S]` with pseudobulk
/// sizes `sizes` (cells per column), one per entry of `level_targets`
/// (coarsest → finest, non-decreasing): at most `target` groups each,
/// counting the reserved background group when uninformative features exist.
///
/// # Method
///
/// Most features of a real axis carry no grouping evidence: a feature that is
/// barely detected, or detected at one flat rate everywhere, has counts a
/// single rate explains, and grouping such features by distance chains them
/// into one group that drags informative features in with it. So every feature
/// is first tested for homogeneity ([`informative_features`]); the ones that
/// fail form ONE background group, numbered last, at every level.
///
/// The informative features are grouped by k-means on their Pearson residuals
/// `(n − E)/√E`, clipped and scaled to unit length, so a feature is placed by
/// the SHAPE of its deviation from the flat rate with count noise in
/// proportion; k-means keeps groups balanced by construction. Coarser levels
/// cluster the finest centroids, so every level nests in the one below.
/// Each level's group sizes are logged.
///
/// When a background group is reserved, the k-means budget is `target - 1`
/// informative slots (and may be zero): `target == 1` then yields a single
/// all-background coarsening rather than forcing an extra informative cluster.
pub fn coarsen_features(
    counts: &DMatrix<f32>,
    sizes: &[f32],
    level_targets: &[usize],
    seed: u64,
) -> anyhow::Result<Vec<FeatureCoarsening>> {
    let d = counts.nrows();
    anyhow::ensure!(
        counts.ncols() == sizes.len(),
        "{} count columns for {} pseudobulk sizes",
        counts.ncols(),
        sizes.len()
    );
    anyhow::ensure!(!level_targets.is_empty(), "no coarsening level requested");
    anyhow::ensure!(
        level_targets.windows(2).all(|w| w[0] <= w[1]),
        "coarsening levels must run coarsest → finest: {level_targets:?}"
    );
    let informative = informative_features(counts, sizes);
    let rows: Vec<usize> = (0..d).filter(|&g| informative[g]).collect();
    let has_background = rows.len() < d;
    // `target` is a total group budget; the background (when present) spends
    // one slot, and the rest are informative k-means clusters (possibly zero).
    let slots = |target: usize| target.max(1).saturating_sub(usize::from(has_background));
    info!(
        "feature coarsening: {} of {d} features informative, {} in the background group",
        rows.len(),
        d - rows.len()
    );

    // Finest level: k-means on the informative features' residual profiles.
    let finest = *level_targets.last().expect("checked non-empty");
    let k_fine = slots(finest).min(rows.len());
    let (fine_label, centroids) = if rows.is_empty() || k_fine == 0 {
        (Vec::new(), DMatrix::<f32>::zeros(0, 0))
    } else {
        let z = residual_profiles(counts, sizes, &rows, seed);
        let (mut c, labels) = kmeans_centroids_seeded(&z, k_fine, KMEANS_ITER, seed);
        unit_rows(&mut c);
        (labels, c)
    };
    let n_fine = centroids.nrows();

    // Coarser levels: k-means on the finest centroids, so every level nests.
    let levels = level_targets
        .iter()
        .enumerate()
        .map(|(l, &target)| {
            let k = slots(target);
            let to_level: Vec<usize> = if n_fine == 0 || k == 0 {
                Vec::new()
            } else if l + 1 == level_targets.len() || k >= n_fine {
                (0..n_fine).collect()
            } else {
                kmeans_centroids_seeded(&centroids, k, KMEANS_ITER, mix_seed(seed, l as u64)).1
            };
            let level = level_from_labels(d, &rows, &fine_label, &to_level, has_background);
            info!(
                "feature coarsening level {l} (target {target}): {}",
                group_sizes(&level, has_background)
            );
            level
        })
        .collect();
    Ok(levels)
}

/// A [`FeatureCoarsening`] from the informative rows' finest labels mapped to
/// this level, with every other feature in one background group (numbered
/// last). Group ids are compacted in order of first use.
fn level_from_labels(
    d: usize,
    rows: &[usize],
    fine_label: &[usize],
    to_level: &[usize],
    has_background: bool,
) -> FeatureCoarsening {
    let mut in_rows = vec![false; d];
    let mut compact = std::collections::HashMap::<usize, usize>::new();
    let mut fine_to_coarse = vec![0usize; d];
    for (&g, &f) in rows.iter().zip(fine_label) {
        in_rows[g] = true;
        let next = compact.len();
        fine_to_coarse[g] = *compact.entry(to_level[f]).or_insert(next);
    }
    let mut num_coarse = compact.len();
    if has_background {
        for (g, &inside) in in_rows.iter().enumerate() {
            if !inside {
                fine_to_coarse[g] = num_coarse;
            }
        }
        num_coarse += 1;
    }
    let mut coarse_to_fine = vec![Vec::new(); num_coarse];
    for (g, &c) in fine_to_coarse.iter().enumerate() {
        coarse_to_fine[c].push(g);
    }
    FeatureCoarsening {
        fine_to_coarse,
        coarse_to_fine,
        num_coarse,
    }
}

/// The informative groups' sizes, the background (numbered last) reported
/// apart: `"K informative group(s), sizes min / median / max, N singleton(s);
/// background B"`.
fn group_sizes(level: &FeatureCoarsening, has_background: bool) -> String {
    let n_inf = level.num_coarse - usize::from(has_background);
    let mut sizes: Vec<usize> = level.coarse_to_fine[..n_inf].iter().map(Vec::len).collect();
    sizes.sort_unstable();
    format!(
        "{} informative group(s), sizes min {} / median {} / max {}, {} singleton(s); \
         background {}",
        sizes.len(),
        sizes.first().copied().unwrap_or(0),
        sizes.get(sizes.len() / 2).copied().unwrap_or(0),
        sizes.last().copied().unwrap_or(0),
        sizes.iter().filter(|&&n| n == 1).count(),
        if has_background {
            level.coarse_to_fine[n_inf].len()
        } else {
            0
        }
    )
}

/// Options of [`partition_features`]; the default is the plain single-level
/// coarsening.
#[derive(Clone, Debug, Default)]
pub struct PartitionOptions {
    /// Groups of fewer features hold **scattered** features — ones whose
    /// profile matches no other, which k-means places alone. They join the
    /// background, and the slots they held split the largest groups in two
    /// (a split kept only when both halves reach this size). Re-clustering
    /// without them instead only promotes the next outliers. `0` or `1`: off.
    pub min_group_size: usize,
    /// Per feature, a BLOCK no group may cross (e.g. a genomic window);
    /// `u32::MAX` is an ordinary block id. Each block is partitioned on its
    /// own with a share of the slots proportional to its informative
    /// features (at least one). `None`: one block.
    pub block: Option<Vec<u32>>,
}

/// A single-level partition of the features (see [`partition_features`]).
#[derive(Clone, Debug)]
pub struct FeaturePartition {
    /// Group of every feature, `0..num_groups`; the background, when there is
    /// one, is numbered last.
    pub labels: Vec<usize>,
    pub num_groups: usize,
    /// The background group: flat features and scattered ones.
    pub background: Option<usize>,
    /// Per feature, [`informative_features`] (not flat), computed once here
    /// so callers need not run the test again.
    pub informative: Vec<bool>,
}

/// Salt of the k-means seed of a group split.
const SPLIT_SEED_SALT: u64 = 0x5350_4c54;

/// `k` k-means groups of `rows` by their residual profiles, as lists of row
/// ids in first-use order (empty groups dropped); `k` is clamped to the row
/// count.
fn kmeans_groups(
    counts: &DMatrix<f32>,
    sizes: &[f32],
    rows: &[usize],
    k: usize,
    seed: u64,
) -> Vec<Vec<usize>> {
    let k = k.min(rows.len());
    if k == 0 {
        return Vec::new();
    }
    let z = residual_profiles(counts, sizes, rows, seed);
    let (_, labels) = kmeans_centroids_seeded(&z, k, KMEANS_ITER, seed);
    let mut slot = vec![usize::MAX; k];
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (&r, &l) in rows.iter().zip(&labels) {
        if slot[l] == usize::MAX {
            slot[l] = groups.len();
            groups.push(Vec::new());
        }
        groups[slot[l]].push(r);
    }
    groups
}

/// The informative groups of `rows` within a budget of `n_groups` (one slot
/// kept for the background when it has members): the plain k-means, then,
/// with `min_size > 1`, scattered groups set aside and the freed slots spent
/// splitting the largest groups. Rows in no returned group are background.
fn informative_groups(
    counts: &DMatrix<f32>,
    sizes: &[f32],
    rows: &[usize],
    informative: &[bool],
    n_groups: usize,
    seed: u64,
    min_size: usize,
) -> Vec<Vec<usize>> {
    let inf: Vec<usize> = rows.iter().copied().filter(|&r| informative[r]).collect();
    let has_empty = inf.len() < rows.len();
    let slots = n_groups.max(1).saturating_sub(usize::from(has_empty));
    let groups = kmeans_groups(counts, sizes, &inf, slots, seed);
    if min_size <= 1 {
        return groups;
    }
    let (mut groups, scattered): (Vec<Vec<usize>>, Vec<Vec<usize>>) =
        groups.into_iter().partition(|g| g.len() >= min_size);
    let has_bg = has_empty || !scattered.is_empty();
    let mut free = n_groups.saturating_sub(groups.len() + usize::from(has_bg));
    let mut heap: std::collections::BinaryHeap<(usize, usize)> = groups
        .iter()
        .enumerate()
        .map(|(i, g)| (g.len(), i))
        .collect();
    while free > 0 {
        let Some((len, i)) = heap.pop() else { break };
        if len < 2 * min_size {
            break;
        }
        let mut halves = kmeans_groups(
            counts,
            sizes,
            &groups[i],
            2,
            mix_seed(seed, SPLIT_SEED_SALT + i as u64),
        );
        if halves.len() == 2 && halves.iter().all(|h| h.len() >= min_size) {
            let b = halves.pop().expect("two halves");
            groups[i] = halves.pop().expect("two halves");
            heap.push((groups[i].len(), i));
            heap.push((b.len(), groups.len()));
            groups.push(b);
            free -= 1;
        }
    }
    groups
}

/// Proportional shares of `budget` slots over blocks with `weight` informative
/// features each: at least one per block, the remainder to the blocks with the
/// most features per slot.
fn block_shares(weight: &[usize], budget: usize) -> Vec<usize> {
    let total: usize = weight.iter().sum();
    let mut alloc: Vec<usize> = weight
        .iter()
        .map(|&c| (budget * c / total.max(1)).max(1))
        .collect();
    while alloc.iter().sum::<usize>() > budget {
        let i = (0..alloc.len())
            .max_by_key(|&i| alloc[i])
            .expect("non-empty");
        alloc[i] -= 1;
    }
    while alloc.iter().sum::<usize>() < budget {
        let i = (0..alloc.len())
            .max_by(|&a, &b| (weight[a] * alloc[b]).cmp(&(weight[b] * alloc[a])))
            .expect("non-empty");
        alloc[i] += 1;
    }
    alloc
}

/// A single-level partition of `counts` `[D × S]` into at most `n_groups`
/// groups, the background counted: [`coarsen_features`] at one level, with
/// the [`PartitionOptions`] applied and the background id and informative
/// flags returned. Without options the labels are exactly the single-level
/// coarsening's.
pub fn partition_features(
    counts: &DMatrix<f32>,
    sizes: &[f32],
    n_groups: usize,
    seed: u64,
    opts: &PartitionOptions,
) -> anyhow::Result<FeaturePartition> {
    let d = counts.nrows();
    anyhow::ensure!(
        counts.ncols() == sizes.len(),
        "{} count columns for {} pseudobulk sizes",
        counts.ncols(),
        sizes.len()
    );
    let informative = informative_features(counts, sizes);
    let min_size = opts.min_group_size;
    let mut rows_of = std::collections::BTreeMap::<u32, Vec<usize>>::new();
    match &opts.block {
        Some(block) => {
            anyhow::ensure!(
                block.len() == d,
                "{} block ids for {d} features",
                block.len()
            );
            for (r, &b) in block.iter().enumerate() {
                rows_of.entry(b).or_default().push(r);
            }
        }
        None => {
            rows_of.insert(0, (0..d).collect());
        }
    }

    let groups: Vec<Vec<usize>> = if rows_of.len() <= 1 {
        let rows: Vec<usize> = (0..d).collect();
        informative_groups(counts, sizes, &rows, &informative, n_groups, seed, min_size)
    } else {
        let n_inf = |rows: &[usize]| rows.iter().filter(|&&r| informative[r]).count();
        let live: Vec<(u32, usize)> = rows_of
            .iter()
            .map(|(&b, rows)| (b, n_inf(rows)))
            .filter(|&(_, c)| c > 0)
            .collect();
        let budget = n_groups.saturating_sub(1);
        anyhow::ensure!(
            live.len() <= budget,
            "{} blocks hold informative features but only {budget} group slots: widen the blocks",
            live.len()
        );
        let weight: Vec<usize> = live.iter().map(|&(_, c)| c).collect();
        let alloc = block_shares(&weight, budget);
        live.iter()
            .zip(&alloc)
            .flat_map(|(&(b, _), &k)| {
                let rows = &rows_of[&b];
                // One more slot when the block has near-empty rows: it is spent
                // on their background, which joins the shared one.
                let has_empty = rows.len() > n_inf(rows);
                informative_groups(
                    counts,
                    sizes,
                    rows,
                    &informative,
                    k + usize::from(has_empty),
                    mix_seed(seed, u64::from(b)),
                    min_size,
                )
            })
            .collect()
    };

    // Labels: the groups in order, then the background.
    let bg = groups.len();
    let mut labels = vec![bg; d];
    for (i, g) in groups.iter().enumerate() {
        for &r in g {
            labels[r] = i;
        }
    }
    let n_bg = d - groups.iter().map(Vec::len).sum::<usize>();
    let n_flat = informative.iter().filter(|&&i| !i).count();
    let mut group_sizes: Vec<usize> = groups.iter().map(Vec::len).collect();
    group_sizes.sort_unstable();
    info!(
        "feature partition: {bg} group(s) over {} block(s), sizes min {} / median {} / max {}; \
         background {n_bg} ({n_flat} flat, {} scattered)",
        rows_of.len(),
        group_sizes.first().copied().unwrap_or(0),
        group_sizes.get(group_sizes.len() / 2).copied().unwrap_or(0),
        group_sizes.last().copied().unwrap_or(0),
        n_bg - n_flat,
    );
    Ok(FeaturePartition {
        labels,
        num_groups: bg + usize::from(n_bg > 0),
        background: (n_bg > 0).then_some(bg),
        informative,
    })
}

/// Shared CLI arg for grouping co-expressed features before training.
///
/// One declaration, one wording, one default, flattened by every command that
/// can train at reduced feature resolution — the same shape as
/// `crate::alg::hvg::HvgCliArgs`, and read alongside it: selection decides which
/// features weigh on the sketch, this decides how many distinct outputs the
/// model answers for.
#[derive(Args, Debug, Clone, Serialize, Deserialize)]
#[serde(default = "legume_numeric::matrix::clap_defaults::clap_defaults")]
pub struct FeatureCoarseningArgs {
    #[arg(
        long,
        default_value_t = 1000,
        value_name = "N",
        help = "Group co-expressed features into at most N coarse features; 0 = every feature",
        long_help = "Group co-expressed features into at most N coarse features,\n\
                     so the model answers for a group instead of for each feature.\n\
                     Groups come from the finest pseudobulk profiles, nested per\n\
                     level with log-spaced widths. The dictionary is expanded back\n\
                     to every feature on output, so what you read is unchanged.\n\
                     \n\
                     --no-feature-coarsening trains on every feature instead. So\n\
                     does 0 here, kept because recorded runs and existing scripts\n\
                     spell it that way.\n\
                     \n\
                     WHAT THE GROUPING APPLIES TO DEPENDS ON THE COMMAND:\n\
                     \n\
                     `topic` and `vae` group both sides. The encoder reads coarse\n\
                     features and every decoder answers for them, so 0 makes each\n\
                     per-step tensor as wide as the feature axis.\n\
                     \n\
                     `masked-topic`, `masked-vae` and `masked-sbp` group the decoder\n\
                     targets only. The encoder keeps its feature-level context and\n\
                     embedding either way.\n\
                     \n\
                     `joint-topic` groups per modality, or on the reference modality\n\
                     and shares it, following --decoder-type.\n\
                     \n\
                     This is a COARSENING, not a module. senna tells the two kinds\n\
                     of grouping apart by one question: does training move the\n\
                     membership? A coarsening's does not. It is read off the data\n\
                     before the first step and fixed for the life of the model,\n\
                     which is what lets every decoder be keyed to it and what lets a\n\
                     continued fit inherit it. A module's does: --gene-modules learns\n\
                     its centroids, so which features group together changes as the\n\
                     fit proceeds.\n\
                     \n\
                     What a coarsening fixes is the assignment, not the meaning. A\n\
                     group's embedding is the mean of its members' and moves at every\n\
                     step, so grouping the targets does not freeze what the model can\n\
                     learn about a feature."
    )]
    pub max_coarse_features: usize,

    #[arg(
        long,
        conflicts_with = "max_coarse_features",
        help = "Train on every feature, with no grouping",
        long_help = "Train on every feature. The named form of\n\
                     --max-coarse-features 0, which reads as a request for zero\n\
                     features when it means the opposite.\n\
                     \n\
                     Every per-step tensor is then as wide as the feature axis, so\n\
                     the fit costs considerably more at the same epoch count. It is\n\
                     also fixed for the life of a model: a continued fit inherits\n\
                     the groups the source run trained on, so a chain that starts\n\
                     ungrouped stays ungrouped and one that starts grouped cannot\n\
                     be switched over partway."
    )]
    pub no_feature_coarsening: bool,
}

impl FeatureCoarseningArgs {
    /// The cap, or `None` when the model trains on every feature.
    ///
    /// The one place either spelling of "off" is read, so no command repeats
    /// the convention: the named switch and the zero mean the same thing here
    /// and clap refuses both at once.
    #[must_use]
    pub fn cap(&self) -> Option<std::num::NonZeroUsize> {
        if self.no_feature_coarsening {
            return None;
        }
        std::num::NonZeroUsize::new(self.max_coarse_features)
    }
}

#[cfg(test)]
#[path = "feature_coarsening_tests.rs"]
mod tests;
