//! How closely a reference-mode simulation follows its reference: per-cell
//! library size and detected genes, per-gene mean, variance and zero
//! fraction, and the gene-gene correlation of the densest copula genes.
//! Written next to the simulation as `{out}.fidelity.json`, and drawn by
//! `data-beans-sim view` (feature `tui`).

use nalgebra::DMatrix;
use nalgebra_sparse::CscMatrix;
use serde::{Deserialize, Serialize};

/// Copula genes whose pairwise correlation the report compares, at most.
pub const CORRELATION_GENES: usize = 200;

/// Library size each cell is scaled to before `log1p` in the correlation.
const NORMALIZED_DEPTH: f64 = 1e4;

/// One statistic of the reference and of the simulation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Paired<T> {
    pub reference: T,
    pub simulated: T,
}

/// Mean, variance and zero fraction of a set of genes over the cells.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct GeneMoments {
    pub mean: Vec<f32>,
    pub var: Vec<f32>,
    pub zero: Vec<f32>,
}

/// The reference's side of a report, computed while fitting.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Profile {
    /// Library size of each cell.
    pub library: Vec<f32>,
    /// Genes detected in each cell.
    pub detected: Vec<f32>,
    /// Moments of the report's genes ([`FidelityReport::gene_names`]).
    pub moments: GeneMoments,
    /// Upper triangle, row by row, of the Pearson correlation of
    /// `log1p(1e4 · x / library)` among [`FidelityReport::correlation_genes`].
    pub correlation: Vec<f32>,
}

/// A reference-mode simulation against its reference.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FidelityReport {
    pub reference: String,
    pub output: String,
    /// Names of the genes the moments are of (the active genes).
    pub gene_names: Vec<String>,
    /// Indices into [`FidelityReport::gene_names`] of the correlation genes.
    pub correlation_genes: Vec<usize>,
    /// The reference's statistics and the simulation's.
    pub profiles: Paired<Profile>,
}

/// Where the fidelity report of an output `prefix` goes:
/// `{prefix}.fidelity.json`. Multiome modalities pass `{out}.{modality}`.
pub fn report_path(prefix: &str) -> String {
    format!("{prefix}.fidelity.json")
}

impl FidelityReport {
    pub fn read(path: &str) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| anyhow::anyhow!("cannot open fidelity report {path}: {e}"))?;
        Ok(serde_json::from_reader(std::io::BufReader::new(file))?)
    }

    pub fn write(&self, path: &str) -> anyhow::Result<()> {
        let file = std::fs::File::create(path)?;
        serde_json::to_writer(std::io::BufWriter::new(file), self)?;
        Ok(())
    }
}

/// Genes with fewer reference counts than this over all cells are left out
/// of the mean and variance ratios and of the worst-gene lists: their
/// sampling noise is the ratio.
pub const MIN_GENE_COUNTS: f64 = 50.0;

/// How a summary statistic reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum StatKind {
    /// A count or size: cells, library sizes, detected genes.
    Count,
    /// A ratio, share or correlation.
    Ratio,
}

/// One line of [`FidelityReport::summary`]: the reference's value, when
/// the statistic has one, and the simulation's.
#[derive(Debug, Clone, Serialize)]
pub struct Stat {
    pub key: &'static str,
    pub label: &'static str,
    #[serde(skip)]
    pub kind: StatKind,
    pub reference: Option<f64>,
    pub simulated: f64,
}

/// A gene whose simulated moment is furthest from the reference's.
#[derive(Debug, Clone, Serialize)]
pub struct GeneGap {
    pub gene: String,
    pub reference: f32,
    pub simulated: f32,
}

/// A gene pair whose simulated correlation is furthest from the reference's.
#[derive(Debug, Clone, Serialize)]
pub struct PairGap {
    pub genes: [String; 2],
    pub reference: f32,
    pub simulated: f32,
}

/// The genes and pairs furthest off, `top` of each kind.
#[derive(Debug, Clone, Serialize)]
pub struct Gaps {
    /// By `|ln(sim/ref)|` of the mean.
    pub mean: Vec<GeneGap>,
    /// By `|ln(sim/ref)|` of the variance.
    pub variance: Vec<GeneGap>,
    /// By `|sim − ref|` of the zero fraction.
    pub zero_fraction: Vec<GeneGap>,
    /// By `|sim − ref|` of the correlation.
    pub correlation: Vec<PairGap>,
}

impl FidelityReport {
    /// Positions in the moment vectors of the genes with at least
    /// [`MIN_GENE_COUNTS`] reference counts.
    fn counted_genes(&self) -> Vec<usize> {
        let n = self.profiles.reference.library.len() as f64;
        let mean = &self.profiles.reference.moments.mean;
        (0..mean.len())
            .filter(|&i| mean[i] as f64 * n >= MIN_GENE_COUNTS)
            .collect()
    }

    /// The statistics the view's summary and the MCP `fidelity` tool show.
    pub fn summary(&self) -> Vec<Stat> {
        use StatKind::{Count, Ratio};
        let stat = |key, label, kind, reference, simulated| Stat {
            key,
            label,
            kind,
            reference,
            simulated,
        };
        let (r, s) = (&self.profiles.reference, &self.profiles.simulated);
        let both = |key, label, kind, f: &dyn Fn(&[f32]) -> f64, of: fn(&Profile) -> &[f32]| {
            stat(key, label, kind, Some(f(of(r))), f(of(s)))
        };
        let lib: fn(&Profile) -> &[f32] = |p| &p.library;
        let det: fn(&Profile) -> &[f32] = |p| &p.detected;
        let mut out = vec![
            stat(
                "cells",
                "cells",
                Count,
                Some(r.library.len() as f64),
                s.library.len() as f64,
            ),
            both("library_mean", "library size: mean", Count, &mean, lib),
            both("library_cv", "library size: CV", Ratio, &cv, lib),
            both(
                "library_q05",
                "library size: 5% quantile",
                Count,
                &|v| quantile(v, 0.05),
                lib,
            ),
            both(
                "library_median",
                "library size: median",
                Count,
                &|v| quantile(v, 0.5),
                lib,
            ),
            both(
                "library_q95",
                "library size: 95% quantile",
                Count,
                &|v| quantile(v, 0.95),
                lib,
            ),
            both("detected_mean", "detected genes: mean", Count, &mean, det),
            both("detected_cv", "detected genes: CV", Ratio, &cv, det),
            both(
                "detected_median",
                "detected genes: median",
                Count,
                &|v| quantile(v, 0.5),
                det,
            ),
        ];

        let (mr, ms) = (
            &self.profiles.reference.moments,
            &self.profiles.simulated.moments,
        );
        let kept = self.counted_genes();
        let log_ratio = |a: &[f32], b: &[f32]| -> Vec<f32> {
            kept.iter()
                .filter(|&&i| a[i] > 0.0 && b[i] > 0.0)
                .map(|&i| (b[i] / a[i]).ln())
                .collect()
        };
        let means = log_ratio(&mr.mean, &ms.mean);
        let vars = log_ratio(&mr.var, &ms.var);
        out.extend([
            stat(
                "mean_ratio_median",
                "gene mean: median sim/ref",
                Ratio,
                None,
                quantile(&means, 0.5).exp(),
            ),
            stat(
                "mean_log_ratio_sd",
                "gene mean: sd of ln(sim/ref)",
                Ratio,
                None,
                sd(&means),
            ),
            stat(
                "variance_ratio_median",
                "gene variance: median sim/ref",
                Ratio,
                None,
                quantile(&vars, 0.5).exp(),
            ),
            stat(
                "zero_fraction_mean",
                "zero fraction: mean over genes",
                Ratio,
                Some(mean(&mr.zero)),
                mean(&ms.zero),
            ),
            stat(
                "zero_fraction_agreement",
                "zero fraction: correlation",
                Ratio,
                None,
                pearson(&mr.zero, &ms.zero),
            ),
        ]);

        let (cr, cs) = (
            &self.profiles.reference.correlation,
            &self.profiles.simulated.correlation,
        );
        let abs = |v: &[f32]| -> Vec<f32> { v.iter().map(|x| x.abs()).collect() };
        let (ar, as_) = (abs(cr), abs(cs));
        out.extend([
            stat(
                "pairs",
                "gene pairs compared",
                Count,
                Some(cr.len() as f64),
                cs.len() as f64,
            ),
            stat(
                "pair_abs_mean",
                "pair correlation: mean |r|",
                Ratio,
                Some(mean(&ar)),
                mean(&as_),
            ),
            stat(
                "pair_abs_q95",
                "pair correlation: 95% quantile |r|",
                Ratio,
                Some(quantile(&ar, 0.95)),
                quantile(&as_, 0.95),
            ),
            stat(
                "pair_slope",
                "pair correlation: slope sim on ref",
                Ratio,
                None,
                slope(cr, cs),
            ),
            stat(
                "pair_agreement",
                "pair correlation: agreement (r)",
                Ratio,
                None,
                pearson(cr, cs),
            ),
        ]);
        out
    }

    /// The `top` genes and gene pairs furthest from the reference.
    pub fn gaps(&self, top: usize) -> Gaps {
        let (mr, ms) = (
            &self.profiles.reference.moments,
            &self.profiles.simulated.moments,
        );
        let kept = self.counted_genes();
        let genes = |a: &[f32], b: &[f32], gap: &dyn Fn(f32, f32) -> f32| -> Vec<GeneGap> {
            let mut v: Vec<(f32, usize)> = kept
                .iter()
                .map(|&i| (gap(a[i], b[i]), i))
                .filter(|(g, _)| g.is_finite())
                .collect();
            v.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
            v.into_iter()
                .take(top)
                .map(|(_, i)| GeneGap {
                    gene: self.gene_names.get(i).cloned().unwrap_or_default(),
                    reference: a[i],
                    simulated: b[i],
                })
                .collect()
        };
        let log_gap = |r: f32, s: f32| {
            // A gene never simulated is as far off as one can be.
            (s.max(f32::MIN_POSITIVE) / r.max(f32::MIN_POSITIVE))
                .ln()
                .abs()
        };
        let abs_gap = |r: f32, s: f32| (s - r).abs();

        let names: Vec<&str> = self
            .correlation_genes
            .iter()
            .map(|&i| self.gene_names.get(i).map_or("", String::as_str))
            .collect();
        let h = names.len();
        let mut pairs = Vec::with_capacity(self.profiles.reference.correlation.len());
        for i in 0..h {
            for k in (i + 1)..h {
                pairs.push((i, k));
            }
        }
        let (cr, cs) = (
            &self.profiles.reference.correlation,
            &self.profiles.simulated.correlation,
        );
        let mut order: Vec<usize> = (0..cr.len().min(cs.len()).min(pairs.len())).collect();
        order.sort_by(|&x, &y| {
            abs_gap(cr[y], cs[y])
                .total_cmp(&abs_gap(cr[x], cs[x]))
                .then(x.cmp(&y))
        });
        let correlation = order
            .into_iter()
            .take(top)
            .map(|p| {
                let (i, k) = pairs[p];
                PairGap {
                    genes: [names[i].to_string(), names[k].to_string()],
                    reference: cr[p],
                    simulated: cs[p],
                }
            })
            .collect();
        Gaps {
            mean: genes(&mr.mean, &ms.mean, &log_gap),
            variance: genes(&mr.var, &ms.var, &log_gap),
            zero_fraction: genes(&mr.zero, &ms.zero, &abs_gap),
            correlation,
        }
    }
}

fn mean(v: &[f32]) -> f64 {
    v.iter().map(|&x| x as f64).sum::<f64>() / v.len().max(1) as f64
}

fn sd(v: &[f32]) -> f64 {
    let m = mean(v);
    (v.iter().map(|&x| (x as f64 - m).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt()
}

fn cv(v: &[f32]) -> f64 {
    let m = mean(v);
    if m > 0.0 {
        sd(v) / m
    } else {
        0.0
    }
}

/// The `q` quantile of `v` (linear between order statistics); 0 if empty.
fn quantile(v: &[f32], q: f64) -> f64 {
    let mut s: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if s.is_empty() {
        return 0.0;
    }
    s.sort_by(f32::total_cmp);
    let pos = q.clamp(0.0, 1.0) * (s.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    let w = pos - lo as f64;
    s[lo] as f64 * (1.0 - w) + s[hi] as f64 * w
}

/// Centred cross and own sums of squares of `a` and `b`.
fn covariance(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    let (ma, mb) = (mean(a), mean(b));
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let (dx, dy) = (x as f64 - ma, y as f64 - mb);
        sab += dx * dy;
        saa += dx * dx;
        sbb += dy * dy;
    }
    (sab, saa, sbb)
}

fn pearson(a: &[f32], b: &[f32]) -> f64 {
    let (sab, saa, sbb) = covariance(a, b);
    if saa > 0.0 && sbb > 0.0 {
        sab / (saa * sbb).sqrt()
    } else {
        0.0
    }
}

/// Least-squares slope of `b` on `a`.
fn slope(a: &[f32], b: &[f32]) -> f64 {
    let (sab, saa, _) = covariance(a, b);
    if saa > 0.0 {
        sab / saa
    } else {
        0.0
    }
}

/// Counts of one matrix as `(gene, cell, value)` triplets.
pub trait Counts {
    fn for_each(&self, f: impl FnMut(usize, usize, f32));
}

impl Counts for CscMatrix<f32> {
    fn for_each(&self, mut f: impl FnMut(usize, usize, f32)) {
        for (c, col) in self.col_iter().enumerate() {
            for (&g, &v) in col.row_indices().iter().zip(col.values()) {
                f(g, c, v);
            }
        }
    }
}

impl Counts for [(u64, u64, f32)] {
    fn for_each(&self, mut f: impl FnMut(usize, usize, f32)) {
        for &(g, c, v) in self {
            f(g as usize, c as usize, v);
        }
    }
}

/// The [`Profile`] of `counts` (`n_genes × n_cells`) for the moment genes
/// `genes` and the correlation genes `corr_genes` (indices into `genes`).
pub fn profile<C: Counts + ?Sized>(
    counts: &C,
    n_genes: usize,
    n_cells: usize,
    genes: &[usize],
    corr_genes: &[usize],
) -> Profile {
    let mut library = vec![0.0_f64; n_cells];
    let mut detected = vec![0.0_f32; n_cells];
    let mut pos = vec![usize::MAX; n_genes];
    for (i, &g) in genes.iter().enumerate() {
        pos[g] = i;
    }
    let (mut sum, mut sum_sq, mut nnz) = (
        vec![0.0_f64; genes.len()],
        vec![0.0_f64; genes.len()],
        vec![0usize; genes.len()],
    );
    counts.for_each(|g, c, v| {
        library[c] += v as f64;
        if v > 0.0 {
            detected[c] += 1.0;
        }
        let i = pos[g];
        if i != usize::MAX {
            sum[i] += v as f64;
            sum_sq[i] += (v as f64) * (v as f64);
            nnz[i] += (v != 0.0) as usize;
        }
    });
    let n = n_cells.max(1) as f64;
    let moments = GeneMoments {
        mean: sum.iter().map(|&s| (s / n) as f32).collect(),
        var: sum
            .iter()
            .zip(&sum_sq)
            .map(|(&s, &s2)| {
                let m = s / n;
                ((s2 / n - m * m) * n / (n - 1.0).max(1.0)).max(0.0) as f32
            })
            .collect(),
        zero: nnz.iter().map(|&k| (1.0 - k as f64 / n) as f32).collect(),
    };

    let mut row = vec![usize::MAX; n_genes];
    for (h, &i) in corr_genes.iter().enumerate() {
        row[genes[i]] = h;
    }
    let mut x = DMatrix::<f64>::zeros(corr_genes.len(), n_cells);
    counts.for_each(|g, c, v| {
        let h = row[g];
        if h != usize::MAX && library[c] > 0.0 {
            x[(h, c)] = (NORMALIZED_DEPTH * v as f64 / library[c]).ln_1p();
        }
    });
    Profile {
        library: library.iter().map(|&l| l as f32).collect(),
        detected,
        moments,
        correlation: upper_correlation(x),
    }
}

/// Upper triangle of the Pearson correlation of the rows of `x`; a constant
/// row correlates 0 with everything.
fn upper_correlation(mut x: DMatrix<f64>) -> Vec<f32> {
    let n = x.ncols().max(1) as f64;
    for mut r in x.row_iter_mut() {
        let mean = r.sum() / n;
        r.add_scalar_mut(-mean);
        let norm = r.norm();
        if norm > 0.0 {
            r /= norm;
        }
    }
    let gram = &x * x.transpose();
    let h = x.nrows();
    let mut out = Vec::with_capacity(h * h.saturating_sub(1) / 2);
    for i in 0..h {
        for k in (i + 1)..h {
            out.push(gram[(i, k)] as f32);
        }
    }
    out
}

/// The correlation genes: the [`CORRELATION_GENES`] copula genes detected in
/// the most reference cells, as indices into `genes`.
pub fn correlation_genes(genes: &[usize], hvg: &[usize], zero: &[f32]) -> Vec<usize> {
    let mut pos = std::collections::HashMap::new();
    for (i, &g) in genes.iter().enumerate() {
        pos.insert(g, i);
    }
    let mut picked: Vec<usize> = hvg.iter().filter_map(|g| pos.get(g).copied()).collect();
    picked.sort_by(|&a, &b| zero[a].total_cmp(&zero[b]).then(a.cmp(&b)));
    picked.truncate(CORRELATION_GENES);
    picked.sort_unstable();
    picked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_of_triplets_matches_a_dense_count() {
        // 3 genes × 4 cells; gene 2 is not reported.
        let trips: Vec<(u64, u64, f32)> = vec![
            (0, 0, 2.0),
            (1, 0, 1.0),
            (0, 1, 4.0),
            (2, 2, 5.0),
            (1, 3, 3.0),
            (0, 3, 2.0),
        ];
        let p = profile(&trips[..], 3, 4, &[0, 1], &[0, 1]);
        assert_eq!(p.library, vec![3.0, 4.0, 5.0, 5.0]);
        assert_eq!(p.detected, vec![2.0, 1.0, 1.0, 2.0]);
        assert_eq!(p.moments.mean, vec![2.0, 1.0]);
        assert_eq!(p.moments.zero, vec![0.25, 0.5]);
        // Gene 0: 2, 4, 0, 2 → sample variance 8/3.
        assert!((p.moments.var[0] - 8.0 / 3.0).abs() < 1e-6);
        assert_eq!(p.correlation.len(), 1);
        assert!(p.correlation[0].abs() <= 1.0);
    }

    #[test]
    fn correlation_genes_are_the_densest_hvgs() {
        let genes = [3, 5, 7, 9];
        let zero = [0.9, 0.1, 0.5, 0.2];
        assert_eq!(correlation_genes(&genes, &[5, 7, 9], &zero), vec![1, 2, 3]);
        assert_eq!(correlation_genes(&genes, &[3, 4], &zero), vec![0]);
    }

    #[test]
    fn summary_and_gaps_read_the_report() {
        let report = FidelityReport {
            gene_names: vec!["a".into(), "b".into(), "c".into()],
            correlation_genes: vec![0, 1, 2],
            profiles: Paired {
                reference: Profile {
                    library: vec![100.0; 1000],
                    moments: GeneMoments {
                        mean: vec![1.0, 2.0, 0.1],
                        var: vec![2.0, 4.0, 0.2],
                        zero: vec![0.5, 0.2, 0.9],
                    },
                    correlation: vec![0.1, 0.2, 0.3],
                    ..Default::default()
                },
                simulated: Profile {
                    library: vec![100.0; 1200],
                    moments: GeneMoments {
                        mean: vec![1.0, 4.0, 0.1],
                        var: vec![2.0, 4.0, 0.1],
                        zero: vec![0.5, 0.2, 0.6],
                    },
                    correlation: vec![0.1, 0.6, 0.3],
                    ..Default::default()
                },
            },
            ..Default::default()
        };
        let summary = report.summary();
        let get = |key: &str| summary.iter().find(|s| s.key == key).unwrap();
        assert_eq!(get("cells").reference, Some(1000.0));
        assert_eq!(get("cells").simulated, 1200.0);
        assert!((get("mean_ratio_median").simulated - 1.0).abs() < 1e-9);

        let gaps = report.gaps(1);
        assert_eq!(gaps.mean[0].gene, "b");
        assert_eq!(gaps.variance[0].gene, "c");
        assert_eq!(gaps.zero_fraction[0].gene, "c");
        assert_eq!(
            gaps.correlation[0].genes,
            ["a".to_string(), "c".to_string()]
        );
        assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0], 0.5), 2.5);
        assert_eq!(quantile(&[], 0.5), 0.0);
    }

    #[test]
    fn a_report_round_trips_through_json() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("r.fidelity.json");
        let path = path.to_str().unwrap();
        let report = FidelityReport {
            reference: "ref.zarr".into(),
            gene_names: vec!["a".into(), "c".into()],
            profiles: Paired {
                reference: Profile {
                    library: vec![1.0, 2.0],
                    ..Default::default()
                },
                simulated: Profile {
                    library: vec![3.0],
                    ..Default::default()
                },
            },
            ..Default::default()
        };
        report.write(path)?;
        let back = FidelityReport::read(path)?;
        assert_eq!(back.gene_names, report.gene_names);
        assert_eq!(back.profiles.simulated.library, vec![3.0]);
        Ok(())
    }
}
