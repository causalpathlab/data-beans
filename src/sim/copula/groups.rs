//! Cell groups of a reference, for fitting it group by group.
//!
//! A reference of several cell types is not one population: in bone
//! marrow, the cells with the largest libraries are plasma and erythroid
//! cells, which also detect three to four times as many genes. One count
//! distribution per gene, one depth coupling and one copula over all cells
//! cannot put those cells back together, so the simulated libraries come
//! out far narrower than the reference's. As in scDesign2, the reference is
//! fitted per cell group instead, and simulated cells are drawn group by
//! group in the reference's proportions.
//!
//! The groups come from labels (`--reference-labels`), or from k-means++
//! on the leading principal components of the HVGs' `ln(1 + 10⁴·x/lib)`
//! (`--reference-groups`), the components from a randomized range finder.
//! A cluster of fewer than [`MIN_GROUP_CELLS`] cells joins its nearest
//! neighbour, and labels held by fewer cells are pooled, so every group has
//! the cells to fit its own distributions.

use crate::sim::copula::reference::SparseRef;
use legume_numeric::matrix::kmeans::{kmeans_rows_seeded, KmeansMetric, KmeansRowsOpts};
use legume_numeric::matrix::membership::Membership;
use log::info;
use nalgebra::DMatrix;
use nalgebra_sparse::CscMatrix;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

/// Fewest reference cells a group keeps; smaller groups join their nearest.
pub const MIN_GROUP_CELLS: usize = 100;

/// The group the labels of fewer than [`MIN_GROUP_CELLS`] cells are pooled in.
pub const SMALL_LABELS: &str = "small-labels";

/// Principal components the clustering works in.
const N_COMPONENTS: usize = 20;

/// Extra columns and power iterations of the randomized range finder.
const OVERSAMPLE: usize = 10;
const POWER_ITERATIONS: usize = 2;

/// Cells per block of the sparse products.
const CELL_BLOCK: usize = 1024;

/// k-means iterations and restarts.
const KMEANS_ITERATIONS: usize = 100;
const KMEANS_RESTARTS: usize = 4;

/// Seed of the clustering, so a reference always splits the same way.
const CLUSTER_SEED: u64 = 0x6705_c1a5_7e25_0001;

/// Each reference cell's group and the groups' names.
#[derive(Debug, Clone)]
pub struct CellGroups {
    pub membership: Vec<usize>,
    pub names: Vec<String>,
}

impl CellGroups {
    pub fn n_groups(&self) -> usize {
        self.names.len()
    }

    /// The cells of each group, in reference order.
    pub fn cells(&self) -> Vec<Vec<usize>> {
        members(&self.membership, self.n_groups())
    }

    /// One group of all `n` cells.
    pub fn single(n: usize) -> Self {
        Self {
            membership: vec![0; n],
            names: vec!["all".into()],
        }
    }

    /// Groups from a labels file: one label per line in the reference's
    /// column order, or `cell<TAB>label` lines matched by cell name.
    /// Unmatched cells fail the fit.
    pub fn from_labels(path: &str, sc: &SparseRef) -> anyhow::Result<Self> {
        let cells = sc.column_names()?;
        let paired = std::fs::read_to_string(path)
            .ok()
            .and_then(|t| {
                t.lines()
                    .find(|l| !l.trim().is_empty())
                    .map(|l| l.split(['\t', ',']).count() > 1)
            })
            .unwrap_or(false);
        let membership = if paired {
            Membership::from_file(path, 0, 1, true)?
        } else {
            Membership::from_file_positional(path, &cells)?
        };
        let (matched, _) = membership.match_keys(&cells);
        let labels: Vec<String> = cells
            .iter()
            .map(|c| {
                matched
                    .get(c)
                    .map(|g| g.to_string())
                    .ok_or_else(|| anyhow::anyhow!("{path} has no label for cell {c}"))
            })
            .collect::<anyhow::Result<_>>()?;
        let labels = pool_small_labels(labels);
        let mut names: Vec<String> = labels.clone();
        names.sort();
        names.dedup();
        let index: std::collections::HashMap<&str, usize> = names
            .iter()
            .enumerate()
            .map(|(k, s)| (s.as_str(), k))
            .collect();
        let membership = labels.iter().map(|l| index[l.as_str()]).collect();
        Ok(Self { membership, names })
    }

    /// `k` groups by k-means++ on the leading principal components of the
    /// HVGs' log-normalised counts; tiny groups join their nearest.
    pub fn cluster(mat: &CscMatrix<f32>, libs: &[f64], hvg: &[usize], k: usize) -> Self {
        let n = libs.len();
        if k <= 1 || n < 2 * MIN_GROUP_CELLS {
            return Self::single(n);
        }
        let x = LogNormalised::new(mat, libs, hvg);
        let pcs = principal_components(&x, N_COMPONENTS.min(hvg.len()).min(n));
        // The restart with the smallest within-group sum of squares.
        let (mut membership, mut centres) = (0..KMEANS_RESTARTS as u64)
            .map(|r| {
                let fit = kmeans_rows_seeded(
                    &pcs,
                    &KmeansRowsOpts {
                        k,
                        max_iter: KMEANS_ITERATIONS,
                        seed: CLUSTER_SEED.wrapping_add(r),
                        metric: KmeansMetric::Euclidean,
                        min_changed_frac: 0.0,
                        init_sample: 0,
                    },
                );
                let wss: f64 = (0..pcs.nrows())
                    .map(|i| distance2(&pcs, i, &fit.centroids, fit.labels[i]) as f64)
                    .sum();
                (fit.labels, fit.centroids, wss)
            })
            .min_by(|a, b| a.2.total_cmp(&b.2))
            .map(|(m, c, _)| (m, c))
            .expect("at least one restart");
        merge_small(&pcs, &mut membership, &mut centres);
        let n_groups = centres.nrows();
        let sizes = group_sizes(&membership, n_groups);
        info!("reference groups: {n_groups} by k-means (asked {k}), sizes {sizes:?}");
        Self {
            membership,
            names: (0..n_groups).map(|g| format!("cluster{g}")).collect(),
        }
    }
}

/// Labels held by fewer than [`MIN_GROUP_CELLS`] cells, pooled into one
/// [`SMALL_LABELS`] group, which joins the largest label instead when it
/// is still too small.
fn pool_small_labels(labels: Vec<String>) -> Vec<String> {
    let mut counts: std::collections::HashMap<&str, usize> = Default::default();
    for l in &labels {
        *counts.entry(l.as_str()).or_default() += 1;
    }
    let small: usize = counts.values().filter(|&&c| c < MIN_GROUP_CELLS).sum();
    if small == 0 {
        return labels;
    }
    let largest = counts
        .iter()
        .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
        .map(|(l, _)| l.to_string())
        .unwrap_or_default();
    let pooled = if small >= MIN_GROUP_CELLS {
        SMALL_LABELS.to_string()
    } else {
        largest
    };
    let n_small = counts.values().filter(|&&c| c < MIN_GROUP_CELLS).count();
    info!(
        "reference labels: {n_small} label(s) under {MIN_GROUP_CELLS} cells ({small} cells) fitted as {pooled}"
    );
    labels
        .iter()
        .map(|l| {
            if counts[l.as_str()] < MIN_GROUP_CELLS {
                pooled.clone()
            } else {
                l.clone()
            }
        })
        .collect()
}

/// Cells × HVGs `ln(1 + 10⁴·x/lib)`, kept sparse: each cell's non-zero
/// `(hvg, value)` entries, and each HVG's mean over the cells. The centred
/// matrix is `X = S − 1·μᵀ`; [`principal_components`] only multiplies by it.
struct LogNormalised {
    rows: Vec<Vec<(u32, f32)>>,
    mean: Vec<f32>,
}

impl LogNormalised {
    fn new(mat: &CscMatrix<f32>, libs: &[f64], hvg: &[usize]) -> Self {
        let mut pos = vec![u32::MAX; mat.nrows()];
        for (h, &g) in hvg.iter().enumerate() {
            pos[g] = h as u32;
        }
        let n = libs.len();
        let rows: Vec<Vec<(u32, f32)>> = (0..n)
            .into_par_iter()
            .map(|c| {
                let col = mat.col(c);
                let scale = 1e4 / libs[c].max(1.0);
                col.row_indices()
                    .iter()
                    .zip(col.values())
                    .filter(|(&g, _)| pos[g] != u32::MAX)
                    .map(|(&g, &v)| (pos[g], (v as f64 * scale).ln_1p() as f32))
                    .collect()
            })
            .collect();
        let mut sum = vec![0.0_f64; hvg.len()];
        for row in &rows {
            for &(h, v) in row {
                sum[h as usize] += v as f64;
            }
        }
        let mean = sum.iter().map(|s| (s / n.max(1) as f64) as f32).collect();
        Self { rows, mean }
    }

    fn n_cells(&self) -> usize {
        self.rows.len()
    }

    fn n_hvg(&self) -> usize {
        self.mean.len()
    }

    /// `X·W`, `W` HVGs × w: `S·W − 1·(μᵀW)`.
    fn times(&self, w: &DMatrix<f32>) -> DMatrix<f32> {
        let shift = w.transpose() * DMatrix::from_column_slice(self.n_hvg(), 1, &self.mean);
        let cols = w.ncols();
        // `wt`'s columns are `w`'s rows, contiguous.
        let wt = w.transpose();
        let rows: Vec<Vec<f32>> = self
            .rows
            .par_iter()
            .map(|row| {
                let mut out: Vec<f32> = (0..cols).map(|k| -shift[(k, 0)]).collect();
                for &(h, v) in row {
                    for (o, &x) in out.iter_mut().zip(wt.column(h as usize).iter()) {
                        *o += v * x;
                    }
                }
                out
            })
            .collect();
        DMatrix::from_fn(self.n_cells(), cols, |c, k| rows[c][k])
    }

    /// `Xᵀ·Q`, `Q` cells × w: `SᵀQ − μ·(1ᵀQ)`.
    fn transpose_times(&self, q: &DMatrix<f32>) -> DMatrix<f32> {
        let cols = q.ncols();
        let h = self.n_hvg();
        // `qt`'s columns are `q`'s rows; partial sums in cell blocks, added
        // in block order so the result does not depend on the threads.
        let qt = q.transpose();
        let partial: Vec<DMatrix<f32>> = (0..self.n_cells().div_ceil(CELL_BLOCK))
            .into_par_iter()
            .map(|b| {
                let mut acc = DMatrix::<f32>::zeros(cols, h);
                let end = ((b + 1) * CELL_BLOCK).min(self.n_cells());
                for c in b * CELL_BLOCK..end {
                    let qc = qt.column(c);
                    for &(hv, v) in &self.rows[c] {
                        acc.column_mut(hv as usize).axpy(v, &qc, 1.0);
                    }
                }
                acc
            })
            .collect();
        let mut sum = DMatrix::<f32>::zeros(cols, h);
        for p in partial {
            sum += p;
        }
        let mut out = sum.transpose();
        let col_sums: Vec<f32> = (0..cols).map(|k| q.column(k).sum()).collect();
        for h in 0..self.n_hvg() {
            for (k, &cs) in col_sums.iter().enumerate() {
                out[(h, k)] -= self.mean[h] * cs;
            }
        }
        out
    }
}

/// Cells × `r` principal component scores of the centred `x`, by a
/// randomized range finder with power iterations.
fn principal_components(x: &LogNormalised, r: usize) -> DMatrix<f32> {
    let width = (r + OVERSAMPLE).min(x.n_hvg()).min(x.n_cells());
    let mut rng = rand::rngs::StdRng::seed_from_u64(CLUSTER_SEED ^ 0x5ca1e);
    let normal = Normal::new(0.0_f32, 1.0).expect("standard normal");
    let omega = DMatrix::<f32>::from_fn(x.n_hvg(), width, |_, _| normal.sample(&mut rng));
    let mut q = x.times(&omega).qr().q();
    for _ in 0..POWER_ITERATIONS {
        let z = x.transpose_times(&q).qr().q();
        q = x.times(&z).qr().q();
    }
    // x ≈ Q·B, B = Qᵀx small; the scores are Q·U·Σ of B's SVD.
    let b = x.transpose_times(&q).transpose();
    let svd = b.svd(true, false);
    let u = svd.u.expect("left singular vectors");
    let mut order: Vec<usize> = (0..svd.singular_values.len()).collect();
    order.sort_by(|&i, &j| svd.singular_values[j].total_cmp(&svd.singular_values[i]));
    let keep: Vec<usize> = order.into_iter().take(r).collect();
    let mut scores = DMatrix::<f32>::zeros(x.n_cells(), keep.len());
    let qu = &q * u;
    for (c, &i) in keep.iter().enumerate() {
        scores.set_column(c, &(qu.column(i) * svd.singular_values[i]));
    }
    scores
}

/// Squared distance of row `i` of `pcs` to row `k` of `centres`.
fn distance2(pcs: &DMatrix<f32>, i: usize, centres: &DMatrix<f32>, k: usize) -> f32 {
    pcs.row(i)
        .iter()
        .zip(centres.row(k).iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum()
}

/// The size of each of `k` groups.
pub fn group_sizes(membership: &[usize], k: usize) -> Vec<usize> {
    let mut sizes = vec![0usize; k];
    for &g in membership {
        sizes[g] += 1;
    }
    sizes
}

/// The members of each of `k` groups, in order.
pub fn members(membership: &[usize], k: usize) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new(); k];
    for (c, &g) in membership.iter().enumerate() {
        out[g].push(c);
    }
    out
}

/// Each group's mean row; an empty group keeps its previous centre.
fn group_means(
    pcs: &DMatrix<f32>,
    membership: &[usize],
    k: usize,
    previous: &DMatrix<f32>,
) -> DMatrix<f32> {
    let mut sums = DMatrix::<f64>::zeros(k, pcs.ncols());
    let mut counts = vec![0usize; k];
    for (i, &g) in membership.iter().enumerate() {
        counts[g] += 1;
        for (s, &v) in sums.row_mut(g).iter_mut().zip(pcs.row(i).iter()) {
            *s += v as f64;
        }
    }
    DMatrix::from_fn(k, pcs.ncols(), |g, c| {
        if counts[g] == 0 {
            previous[(g, c)]
        } else {
            (sums[(g, c)] / counts[g] as f64) as f32
        }
    })
}

/// Fold every group of fewer than [`MIN_GROUP_CELLS`] cells into the
/// nearest of the others, smallest first, and renumber the groups 0..
/// largest first.
fn merge_small(pcs: &DMatrix<f32>, membership: &mut [usize], centres: &mut DMatrix<f32>) {
    loop {
        let k = centres.nrows();
        let sizes = group_sizes(membership, k);
        let small = (0..k)
            .filter(|&g| sizes[g] < MIN_GROUP_CELLS)
            .min_by_key(|&g| sizes[g]);
        let Some(g) = small.filter(|_| k > 1) else {
            break;
        };
        let into = (0..k)
            .filter(|&h| h != g)
            .min_by(|&a, &b| {
                let da = distance2(centres, g, centres, a);
                let db = distance2(centres, g, centres, b);
                da.total_cmp(&db)
            })
            .expect("another group");
        for m in membership.iter_mut() {
            if *m == g {
                *m = into;
            }
            if *m > g {
                *m -= 1;
            }
        }
        let keep: Vec<usize> = (0..k).filter(|&h| h != g).collect();
        *centres = centres.select_rows(&keep);
        *centres = group_means(pcs, membership, k - 1, centres);
    }
    // Largest group first, so the numbering does not depend on the seeding.
    let k = centres.nrows();
    let sizes = group_sizes(membership, k);
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| sizes[b].cmp(&sizes[a]).then(a.cmp(&b)));
    let mut rank = vec![0usize; k];
    for (r, &g) in order.iter().enumerate() {
        rank[g] = r;
    }
    for m in membership.iter_mut() {
        *m = rank[*m];
    }
    *centres = centres.select_rows(&order);
}

pub(crate) use crate::alg::collapse_data::select_columns;

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_sparse::CooMatrix;
    use rand_distr::Poisson;

    /// Two populations with different expression programs split cleanly.
    #[test]
    fn two_populations_split_into_two_groups() -> anyhow::Result<()> {
        let (genes, cells) = (60, 600);
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let mut coo = CooMatrix::new(genes, cells);
        let mut libs = vec![0.0; cells];
        for (c, lib) in libs.iter_mut().enumerate() {
            let a = c < 400;
            for g in 0..genes {
                let mu = if (g < 30) == a { 8.0 } else { 0.5 };
                let y: f64 = Poisson::new(mu)?.sample(&mut rng);
                if y > 0.0 {
                    coo.push(g, c, y as f32);
                    *lib += y;
                }
            }
        }
        let mat = CscMatrix::from(&coo);
        let hvg: Vec<usize> = (0..genes).collect();
        let g = CellGroups::cluster(&mat, &libs, &hvg, 2);
        assert_eq!(g.n_groups(), 2);
        // The larger population is group 0.
        assert!(g.membership[..400].iter().all(|&m| m == 0));
        assert!(g.membership[400..].iter().all(|&m| m == 1));
        // Asking for more groups than the data holds still keeps every
        // group at its minimum size.
        let many = CellGroups::cluster(&mat, &libs, &hvg, 12);
        let sizes: Vec<usize> = many.cells().iter().map(Vec::len).collect();
        assert!(sizes.iter().all(|&s| s >= MIN_GROUP_CELLS), "{sizes:?}");
        Ok(())
    }

    /// A reference of many small cells and fewer big ones, the big ones
    /// expressing another program: fitted in groups it keeps its library
    /// spread, fitted as one population it loses it.
    #[test]
    fn groups_keep_the_library_spread_of_two_cell_sizes() -> anyhow::Result<()> {
        use crate::sim::copula::reference::open_reference;
        use crate::sim::copula::sampler::{sample_reference_counts, CellPrograms, CellTerm};
        use crate::sim::copula::{fit_global_copula, GlobalCopulaArgs, GroupSpec};
        use crate::sparse_io::{create_sparse_from_triplets, SparseIoBackend};

        let (genes, cells) = (120, 1500);
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        let mut trips = Vec::new();
        let mut libs = vec![0.0_f64; cells];
        for (c, lib) in libs.iter_mut().enumerate() {
            let big = c >= 1100;
            let size = if big { 10.0 } else { 1.0 };
            for g in 0..genes {
                let mu = size * if (g < 60) != big { 4.0 } else { 0.4 };
                let y: f64 = Poisson::new(mu)?.sample(&mut rng);
                if y > 0.0 {
                    trips.push((g as u64, c as u64, y as f32));
                    *lib += y;
                }
            }
        }
        let cv = |l: &[f64]| {
            let m = l.iter().sum::<f64>() / l.len() as f64;
            (l.iter().map(|x| (x - m).powi(2)).sum::<f64>() / l.len() as f64).sqrt() / m
        };
        let reference_cv = cv(&libs);
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("ref.zarr").to_string_lossy().into_owned();
        let mut data = create_sparse_from_triplets(
            &trips,
            (genes, cells, trips.len()),
            Some(&path),
            Some(&SparseIoBackend::Zarr),
        )?;
        data.register_row_names_vec(
            &(0..genes)
                .map(|g| format!("g{g}").into())
                .collect::<Vec<_>>(),
        );
        data.register_column_names_vec(
            &(0..cells)
                .map(|c| format!("c{c}").into())
                .collect::<Vec<_>>(),
        );
        let sc = open_reference(&path)?;

        let simulated_cv = |k: usize| -> anyhow::Result<f64> {
            let fit = fit_global_copula(&GlobalCopulaArgs {
                sc: &sc,
                n_hvg: 100,
                copula_rank: 10,
                regularization: 1e-3,
                hvg_min_detection: 0.005,
                groups: GroupSpec::Cluster(k),
            })?;
            let nn = 4000;
            let zero = |_: usize| nalgebra::DVector::<f32>::zeros(genes);
            let delta = [nalgebra::DVector::<f32>::zeros(genes)];
            let membership = vec![0usize; nn];
            let programs = CellPrograms {
                term: CellTerm::PerCell(&zero),
                pve_topic: 0.0,
                cap_spread: false,
                pve_noise: 0.0,
                batch_delta: &delta,
                batch_membership: &membership,
            };
            let sample = sample_reference_counts(&fit, 0.0, &programs, true, 5, 0)?;
            let mut lib = vec![0.0_f64; nn];
            for &(_, j, x) in &sample.triplets {
                lib[j as usize] += x as f64;
            }
            Ok(cv(&lib))
        };
        let one = simulated_cv(1)?;
        let two = simulated_cv(2)?;
        assert!(
            (two / reference_cv - 1.0).abs() < 0.1,
            "two groups: library CV {two:.3} vs the reference's {reference_cv:.3}"
        );
        assert!(
            one < 0.8 * reference_cv,
            "one population: {one:.3} vs {reference_cv:.3}"
        );
        Ok(())
    }

    #[test]
    fn small_labels_are_pooled() {
        let make = |spec: &[(&str, usize)]| -> Vec<String> {
            spec.iter()
                .flat_map(|&(l, n)| std::iter::repeat_n(l.to_string(), n))
                .collect()
        };
        // 70 cells under the minimum: too few for a group of their own.
        let p = pool_small_labels(make(&[("a", 300), ("b", 50), ("c", 20), ("d", 150)]));
        assert_eq!(p.iter().filter(|l| *l == "a").count(), 370);
        // 130 cells: a group of their own.
        let p = pool_small_labels(make(&[("a", 300), ("b", 60), ("c", 70)]));
        assert_eq!(p.iter().filter(|l| *l == SMALL_LABELS).count(), 130);
        let kept = make(&[("a", 300), ("b", 100)]);
        assert_eq!(pool_small_labels(kept.clone()), kept);
    }
}
