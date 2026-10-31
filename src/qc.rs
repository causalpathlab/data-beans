use crate::sparse_data_visitors::*;
use crate::sparse_io::*;
use crate::sparse_io_vector::*;

use indicatif::ParallelProgressIterator;
use legume_numeric::matrix::sparse_stat::{SparseColumnRunningStatistics, SparseRunningStatistics};
use legume_numeric::matrix::traits::RunningStatOps;
use legume_numeric::matrix::utils::partition_by_membership;
use log::warn;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

// The binary compiles its own copy of this module and reads only part of
// the axis.
#[allow(dead_code)]
pub mod axis;
pub use axis::Axis;

use rustc_hash::FxHashMap as HashMap;

#[derive(Clone)]
pub struct SqueezeCutoffs {
    pub row: usize,
    pub column: usize,
}

/// Whether a row or column with `nnz` non-zeros falls below `cutoff` and is
/// squeezed out. Every preview of a cutoff counts with this rule, so what it
/// reports matches what [`squeeze_by_nnz`] removes.
pub fn below_nnz_cutoff(nnz: f32, cutoff: usize) -> bool {
    (nnz as usize) < cutoff
}

/// squeeze out rows and columns with excessive zero values
pub fn squeeze_by_nnz(
    data: &dyn SparseIo<IndexIter = Vec<usize>>,
    cutoffs: SqueezeCutoffs,
    block_size: Option<usize>,
    preload: bool,
) -> anyhow::Result<()> {
    let col_stat = collect_column_stat(data, block_size)?;
    let row_stat = collect_row_stat(data, block_size)?;

    let file = data.get_backend_file_name();
    let backend = data.backend_type();

    let mut data = open_sparse_matrix(file, &backend)?;
    if preload {
        data.preload_columns()?;
    }

    fn nnz_index(nnz: &[f32], cutoff: usize) -> Option<Vec<usize>> {
        let ret: Vec<usize> = nnz
            .iter()
            .enumerate()
            .filter(|&(_, &x)| !below_nnz_cutoff(x, cutoff))
            .map(|(i, _)| i)
            .collect();

        (!ret.is_empty()).then_some(ret)
    }

    let row_nnz_vec = row_stat.count_positives();
    let col_nnz_vec = col_stat.count_positives();
    let row_idx = nnz_index(&row_nnz_vec, cutoffs.row);
    let col_idx = nnz_index(&col_nnz_vec, cutoffs.column);

    if row_idx.is_none() {
        warn!(
            "No rows can be kept with this cutoff {}!\n\
	     \n\
	     We will stop squeezing on the rows.\n\
	     \n",
            cutoffs.row
        );
    }

    if col_idx.is_none() {
        warn!(
            "No columns can be kept with this cutoff {}!\n\
	     \n\
	     We will stop squeezing on the columns.\n\
	     \n",
            cutoffs.column
        );
    }

    data.subset_columns_rows(col_idx.as_ref(), row_idx.as_ref())
}

/// collect row-wise sufficient statistics for Q/C
/// * `data` - `SparseIoVec` across many data matrices
/// * `block_size` - a block size for each parallelized job
pub fn collect_row_stat_across_vec(
    data: &SparseIoVec,
    block_size: Option<usize>,
) -> anyhow::Result<SparseRunningStatistics<f32>> {
    let mut row_stat = SparseRunningStatistics::new(data.num_rows());
    data.visit_columns_by_block(
        &row_stat_vec_visitor,
        &EmptyArgs {},
        &mut row_stat,
        block_size,
    )?;
    Ok(row_stat)
}

/// collect row statistics for each group of columns
/// * `data` - `SparseIo`
/// * `column_membership` - a hashmap assign columns to groups
/// * `block_size` - a block size for each parallelized job
#[allow(clippy::type_complexity)]
pub fn collect_stratified_row_stat_across_vec(
    data: &SparseIoVec,
    column_membership: &HashMap<Box<str>, Box<str>>,
    block_size: Option<usize>,
) -> anyhow::Result<(Vec<Box<str>>, Vec<SparseRunningStatistics<f32>>)> {
    let column_names = data.column_names()?;
    let default = "".to_string().into_boxed_str();
    let membership = column_names
        .into_iter()
        .map(|k| column_membership.get(&k).unwrap_or(&default).clone())
        .collect::<Vec<_>>();

    let partitions = partition_by_membership(&membership, None);
    let mut group_names = Vec::with_capacity(partitions.len());
    let mut group_stats = Vec::with_capacity(partitions.len());
    let num_features = data.num_rows();

    for (k, cols) in partitions {
        let jobs = create_jobs(cols.len(), num_features, block_size);
        let mut row_stat = SparseRunningStatistics::new(data.num_rows());
        let arc_stat = Arc::new(Mutex::new(&mut row_stat));

        jobs.par_iter()
            .progress_with(styled_progress_bar(jobs.len() as u64, "blocks"))
            .for_each(|&(lb, ub)| {
                let cols_sub = cols[lb..ub].iter().cloned();
                let csc = data
                    .read_columns_csc(cols_sub)
                    .expect("failed to read data");
                let mut stat = arc_stat.lock().expect("failed to lock row_stat");
                stat.add_csc(&csc);
            });

        group_names.push(k);
        group_stats.push(row_stat);
    }

    Ok((group_names, group_stats))
}

/// collect row-wise sufficient statistics for Q/C
/// * `data` - `SparseIo`
/// * `block_size` - a block size for each parallelized job
pub fn collect_row_stat(
    data: &dyn SparseIo<IndexIter = Vec<usize>>,
    block_size: Option<usize>,
) -> anyhow::Result<SparseRunningStatistics<f32>> {
    let nrows = data.num_rows().unwrap_or(0);
    let mut row_stat = SparseRunningStatistics::new(nrows);
    let arc_stat = Arc::new(Mutex::new(&mut row_stat));

    let jobs = create_jobs(data.num_columns().unwrap_or(0), nrows, block_size);

    jobs.par_iter()
        .progress_with(styled_progress_bar(jobs.len() as u64, "blocks"))
        .for_each(|&(lb, ub)| {
            let csc = data
                .read_columns_csc((lb..ub).collect())
                .expect("failed to read data");
            let mut stat = arc_stat.lock().expect("failed to lock row_stat");
            stat.add_csc(&csc);
        });

    Ok(row_stat)
}

/// collect column-wise sufficient statistics for Q/C
/// * `data` - `SparseIoVec` across many data matrices
/// * `select_rows` - selected row indices
/// * `block_size` - a block size for each parallelized job
pub fn collect_column_stat_across_vec(
    data: &SparseIoVec,
    select_rows: Option<&[usize]>,
    block_size: Option<usize>,
) -> anyhow::Result<SparseColumnRunningStatistics<f32>> {
    let ncols = data.num_columns();
    let nrows_total = data.num_rows();

    let row_mask: Option<Vec<bool>> = select_rows.map(|sel| {
        let mut m = vec![false; nrows_total];
        for &r in sel {
            if r < nrows_total {
                m[r] = true;
            }
        }
        m
    });
    let nrows_denom = row_mask
        .as_ref()
        .map(|m| m.iter().filter(|x| **x).count())
        .unwrap_or(nrows_total);

    let mut col_stat = SparseColumnRunningStatistics::<f32>::new(ncols, nrows_denom);
    data.visit_columns_by_block(&col_stat_visitor, &row_mask, &mut col_stat, block_size)?;
    Ok(col_stat)
}

/// collect column-wise sufficient statistics for Q/C
/// * `data` - `SparseIo`
/// * `block_size` - a block size for each parallelized job
pub fn collect_column_stat(
    data: &dyn SparseIo<IndexIter = Vec<usize>>,
    block_size: Option<usize>,
) -> anyhow::Result<SparseColumnRunningStatistics<f32>> {
    let ncols = data.num_columns().unwrap_or(0);
    let nrows = data.num_rows().unwrap_or(0);
    let mut col_stat = SparseColumnRunningStatistics::<f32>::new(ncols, nrows);
    let arc_stat = Arc::new(Mutex::new(&mut col_stat));

    let jobs = create_jobs(ncols, nrows, block_size);

    jobs.par_iter()
        .progress_with(styled_progress_bar(jobs.len() as u64, "blocks"))
        .for_each(|&(lb, ub)| {
            let csc = data
                .read_columns_csc((lb..ub).collect())
                .expect("failed to read data");
            let mut stat = arc_stat.lock().expect("failed to lock col_stat");
            stat.add_csc(&csc, lb);
        });

    Ok(col_stat)
}

struct EmptyArgs {}

fn row_stat_vec_visitor(
    job: (usize, usize),
    data: &SparseIoVec,
    _: &EmptyArgs,
    arc_stat: Arc<Mutex<&mut SparseRunningStatistics<f32>>>,
) -> anyhow::Result<()> {
    let (lb, ub) = job;
    let csc = data.read_columns_csc(lb..ub)?;

    let mut stat = arc_stat.lock().expect("failed to lock row_stat");
    stat.add_csc(&csc);
    Ok(())
}

fn col_stat_visitor(
    job: (usize, usize),
    data: &SparseIoVec,
    row_mask: &Option<Vec<bool>>,
    arc_stat: Arc<Mutex<&mut SparseColumnRunningStatistics<f32>>>,
) -> anyhow::Result<()> {
    let (lb, ub) = job;
    let csc = data.read_columns_csc(lb..ub)?;
    let mut stat = arc_stat.lock().expect("failed to lock col_stat");
    match row_mask {
        Some(mask) => stat.add_csc_masked(&csc, lb, mask),
        None => stat.add_csc(&csc, lb),
    }
    Ok(())
}

//////////////////////////////////////////////////////////////////////////////////
// Automatic nnz-cutoff selection + ASCII histogram (shared by `squeeze` and by //
// callers that want cell-calling on a per-column nnz vector).                   //
//////////////////////////////////////////////////////////////////////////////////

/// Bin width of the trough search, in natural-log units (~5% per bin).
const TROUGH_BIN: f64 = 0.05;
/// Gaussian smoothing of the binned density, in bins.
const TROUGH_SMOOTH_BINS: f64 = 3.0;
/// A cut needs the trough at most this fraction of the lower of its two peaks.
const TROUGH_MAX_DEPTH: f64 = 0.25;
/// Ambient and cell peaks sit at least a decade apart; closer modes (a doublet
/// bump, a low-complexity cell type) are not an empty↔cell boundary.
const TROUGH_MIN_PEAK_RATIO: f64 = 10.0;
/// Each side of a cut must hold at least this many columns (or 0.1% of all).
const TROUGH_MIN_SIDE: usize = 20;

/// Suggest an nnz cutoff at the **deepest trough** of the nnz distribution,
/// the gap between the ambient (empty barcode) peak and the cell peak.
/// Columns with `nnz >= cutoff` are kept.
///
/// The density lives on the log axis, where ambient and cells form two
/// peaks decades apart; in linear units the cell peak is spread so thin
/// that the trough in front of it vanishes. The bins, though, come from
/// the actual counts: each integer count `v` spreads its mass over its own
/// interval `[v - 1/2, v + 1/2)` mapped to `ln(1 + ·)`, so small counts,
/// whose log spacing exceeds a bin, leave no empty bins to pass for troughs.
///
/// A cut is proposed only when the smoothed density at the trough is at most
/// [`TROUGH_MAX_DEPTH`] of the lower of the two peaks it separates, the
/// peaks are at least [`TROUGH_MIN_PEAK_RATIO`] apart, and each side holds
/// enough columns. Unimodal data — already-called cells included — gets
/// `None`, whatever its tails look like. Deterministic, no RNG.
pub fn suggest_nnz_cutoff(nnz: &[f32]) -> Option<usize> {
    let n = nnz.len();
    let min_side = TROUGH_MIN_SIDE.max(n / 1000);
    if n < 2 * min_side {
        return None;
    }

    let mut vals: Vec<u64> = nnz.iter().map(|&x| x.max(0.0).round() as u64).collect();
    vals.sort_unstable();
    let (vmin, vmax) = (vals[0], vals[n - 1]);
    if vmin == vmax {
        return None;
    }

    // Count `v`'s interval `[v - 1/2, v + 1/2)` on the `ln(1 + ·)` axis.
    let edge = |v: u64, half: f64| (v as f64 + 1.0 + half).ln();
    let lo = edge(vmin, -0.5);
    let nbins = ((edge(vmax, 0.5) - lo) / TROUGH_BIN).ceil() as usize;
    let mut hist = vec![0.0_f64; nbins];
    let mut i = 0;
    while i < n {
        let v = vals[i];
        let mut j = i;
        while j < n && vals[j] == v {
            j += 1;
        }
        let a = (edge(v, -0.5) - lo) / TROUGH_BIN;
        let b = (edge(v, 0.5) - lo) / TROUGH_BIN;
        let mass = (j - i) as f64 / (b - a);
        for (k, h) in hist
            .iter_mut()
            .enumerate()
            .take((b.ceil() as usize).min(nbins))
            .skip(a.floor() as usize)
        {
            let overlap = b.min(k as f64 + 1.0) - a.max(k as f64);
            if overlap > 0.0 {
                *h += mass * overlap;
            }
        }
        i = j;
    }

    // Gaussian smoothing (finite ±3σ kernel, so a real gap stays exactly 0).
    let radius = (3.0 * TROUGH_SMOOTH_BINS).ceil() as isize;
    let kernel: Vec<f64> = (-radius..=radius)
        .map(|d| (-0.5 * (d as f64 / TROUGH_SMOOTH_BINS).powi(2)).exp())
        .collect();
    let smooth: Vec<f64> = (0..nbins as isize)
        .map(|k| {
            (-radius..=radius)
                .filter_map(|d| {
                    let t = k + d;
                    (0..nbins as isize)
                        .contains(&t)
                        .then(|| hist[t as usize] * kernel[(d + radius) as usize])
                })
                .sum()
        })
        .collect();

    // Mass left of each bin, and the running peaks from either end.
    let mut below = vec![0.0_f64; nbins + 1];
    for k in 0..nbins {
        below[k + 1] = below[k] + hist[k];
    }
    let mut left_peak = vec![(0.0_f64, 0usize); nbins];
    for k in 0..nbins {
        let prev = if k > 0 { left_peak[k - 1] } else { (-1.0, 0) };
        left_peak[k] = if smooth[k] > prev.0 {
            (smooth[k], k)
        } else {
            prev
        };
    }
    let mut right_peak = vec![(0.0_f64, 0usize); nbins];
    for k in (0..nbins).rev() {
        let next = if k + 1 < nbins {
            right_peak[k + 1]
        } else {
            (-1.0, k)
        };
        right_peak[k] = if smooth[k] >= next.0 {
            (smooth[k], k)
        } else {
            next
        };
    }

    let min_bins_apart = TROUGH_MIN_PEAK_RATIO.ln() / TROUGH_BIN;
    let total = below[nbins];
    let mut best: Option<(f64, usize, usize)> = None; // (depth, first, last) of the deepest run
    for t in 1..nbins.saturating_sub(1) {
        let (l_mass, r_mass) = (below[t], total - below[t + 1]);
        if l_mass < min_side as f64 || r_mass < min_side as f64 {
            continue;
        }
        let ((l_h, l_k), (r_h, r_k)) = (left_peak[t - 1], right_peak[t + 1]);
        if ((r_k - l_k) as f64) < min_bins_apart || l_h <= 0.0 || r_h <= 0.0 {
            continue;
        }
        let depth = smooth[t] / l_h.min(r_h);
        match best {
            Some((d, first, last)) if depth == d && last + 1 == t => best = Some((d, first, t)),
            Some((d, _, _)) if depth >= d => {}
            _ => best = Some((depth, t, t)),
        }
    }

    let Some((depth, first, last)) = best else {
        log::info!("nnz cell-calling: no two peaks a decade apart → unimodal, no cutoff");
        return None;
    };
    // Middle of the deepest run (an exact-zero gap is a run, not a point).
    let x = lo + ((first + last) as f64 / 2.0 + 0.5) * TROUGH_BIN;
    let cutoff = (x.exp() - 1.0).ceil().max(1.0) as usize;
    let favors_cut = depth <= TROUGH_MAX_DEPTH;
    log::info!(
        "nnz cell-calling: deepest trough at nnz {} (depth {:.3}) → {}",
        cutoff,
        depth,
        if favors_cut {
            format!("bimodal, cutoff at nnz {cutoff}")
        } else {
            "unimodal, no cutoff".to_string()
        }
    );
    favors_cut.then_some(cutoff)
}

/// `part` as a percentage of `total` (0 when there is nothing).
pub fn pct(part: usize, total: usize) -> f64 {
    if total > 0 {
        100.0 * part as f64 / total as f64
    } else {
        0.0
    }
}

/// Median of an ascending slice (0 when empty).
pub fn median_of_sorted(sorted: &[f32]) -> f32 {
    let n = sorted.len();
    match n {
        0 => 0.0,
        _ if n.is_multiple_of(2) => (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0,
        _ => sorted[n / 2],
    }
}

/// One bin of a printed histogram that holds values, with the real range
/// of the values that fell into it.
struct HistBin {
    val_min: f32,
    val_max: f32,
    count: usize,
    is_cutoff: bool,
}

/// Bins of a printed histogram across its values' range.
const PRINTED_BINS: usize = 40;

/// A histogram of `values` on the log [`Axis`] every histogram shares, its
/// bins that hold values with their real ranges. Works on any non-negative
/// statistic (nnz, sum, mean, sd); the ranges stay exact `f32` so
/// count-like stats still print as integers.
fn create_log_histogram(values: &[f32], cutoff: usize) -> Vec<HistBin> {
    let axis = Axis::fit(values, true).including(cutoff as f64);
    let mut bins: Vec<(usize, f32, f32)> =
        vec![(0, f32::INFINITY, f32::NEG_INFINITY); PRINTED_BINS];
    for &val in values.iter().filter(|v| v.is_finite()) {
        let b = &mut bins[axis.bin(val as f64, PRINTED_BINS)];
        b.0 += 1;
        b.1 = b.1.min(val);
        b.2 = b.2.max(val);
    }
    // Mark the first bin at or above the cutoff that holds values, so the
    // arrow always renders. With no cutoff (cutoff == 0, e.g. the
    // `histogram` command) nothing is marked.
    let cutoff_bin = (cutoff > 0)
        .then(|| {
            let at = axis.bin(cutoff as f64, PRINTED_BINS);
            (at..PRINTED_BINS).find(|&k| bins[k].0 > 0)
        })
        .flatten();
    bins.into_iter()
        .enumerate()
        .filter(|(_, b)| b.0 > 0)
        .map(|(k, (count, val_min, val_max))| HistBin {
            val_min,
            val_max,
            count,
            is_cutoff: Some(k) == cutoff_bin,
        })
        .collect()
}

/// Format a statistic value compactly: whole numbers (nnz, integer counts)
/// print without a decimal point; fractional values (mean, sd) get
/// `decimals` places.
pub fn fmt_stat(v: f32, decimals: usize) -> String {
    if v.fract() == 0.0 {
        (v as i64).to_string()
    } else {
        format!("{:.*}", decimals, v)
    }
}

/// A value as a printed histogram's range shows it: whole numbers as they
/// are, others to three significant digits (0.000412, 0.0731, 2.45).
fn fmt_sig(v: f32) -> String {
    if v.fract() == 0.0 {
        return (v as i64).to_string();
    }
    let places = (2 - (v.abs() as f64).log10().floor() as i32).max(0) as usize;
    format!("{:.*}", places, v)
}

/// Print summary statistics + an ASCII log-binned histogram of a per-row or
/// per-column statistic vector (`metric` names it, e.g. "nnz", "sum", "mean").
/// A non-zero `cutoff` marks the cutoff bin and reports how much it removes;
/// an optional `suggested` value reports the trough suggestion.
///
/// Used by `data-beans histogram`, whose result it is; `qc --qc-histogram`
/// prints the same on stderr ([`eprint_nnz_summary`]).
pub fn print_nnz_summary(
    label: &str,
    metric: &str,
    values: &[f32],
    cutoff: usize,
    suggested: Option<usize>,
) {
    let _ = write_nnz_summary(
        &mut std::io::stdout().lock(),
        label,
        metric,
        values,
        cutoff,
        suggested,
    );
}

/// [`print_nnz_summary`] on stderr: for a histogram that is a diagnostic
/// beside a command's result (`qc --qc-histogram`, senna's and pinto's QC),
/// not the result itself, so stdout carries only the result.
pub fn eprint_nnz_summary(
    label: &str,
    metric: &str,
    values: &[f32],
    cutoff: usize,
    suggested: Option<usize>,
) {
    let _ = write_nnz_summary(
        &mut std::io::stderr().lock(),
        label,
        metric,
        values,
        cutoff,
        suggested,
    );
}

/// [`print_nnz_summary`] to `out`.
pub fn write_nnz_summary(
    out: &mut impl std::io::Write,
    label: &str,
    metric: &str,
    values: &[f32],
    cutoff: usize,
    suggested: Option<usize>,
) -> std::io::Result<()> {
    const MAX_BAR_WIDTH: usize = 50; // Maximum width for histogram bars

    let total = values.len();
    let below_cutoff = values
        .iter()
        .filter(|&&x| below_nnz_cutoff(x, cutoff))
        .count();
    let pct_removed = pct(below_cutoff, total);

    // Calculate basic statistics
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = values.iter().sum();
    let mean = if total > 0 { sum / total as f32 } else { 0.0 };

    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f32::total_cmp);
    let median = median_of_sorted(&sorted);

    writeln!(out, "{} {} distribution:", label, metric)?;
    writeln!(out, "  Total: {}", total)?;
    writeln!(
        out,
        "  Min: {}, Max: {}, Mean: {:.2}, Median: {:.2}",
        fmt_stat(min, 2),
        fmt_stat(max, 2),
        mean,
        median
    )?;
    if cutoff > 0 {
        writeln!(
            out,
            "  Cutoff: {} (removes {} / {} = {:.2}%)",
            cutoff, below_cutoff, total, pct_removed
        )?;
    }
    if let Some(s) = suggested {
        let below_s = values.iter().filter(|&&x| below_nnz_cutoff(x, s)).count();
        let pct_s = pct(below_s, total);
        writeln!(
            out,
            "  Suggested cutoff (histogram trough of {}): {} (would remove {} / {} = {:.2}%)",
            metric, s, below_s, total, pct_s
        )?;
    }

    // Log-spaced bins, tracking the real value range per bin
    let hist = create_log_histogram(values, cutoff);

    // Scale bar width on log10(count+1) so a few outlier bins don't flatten the rest
    let max_log_count = hist
        .iter()
        .map(|b| ((b.count as f64) + 1.0).log10())
        .fold(0.0_f64, f64::max)
        .max(1e-9);

    writeln!(
        out,
        "  Histogram (x: {m} in log-spaced bins, each its range; bar: log10(count+1)):",
        m = metric
    )?;
    for b in hist {
        let marker = if b.is_cutoff { " <-- CUTOFF" } else { "" };
        let log_count = ((b.count as f64) + 1.0).log10();
        let bar_width = ((log_count / max_log_count) * MAX_BAR_WIDTH as f64).round() as usize;
        let bar_width = if b.count > 0 { bar_width.max(1) } else { 0 };
        let bar = "█".repeat(bar_width);
        let range = if b.val_min == b.val_max {
            fmt_sig(b.val_min)
        } else {
            format!("{}-{}", fmt_sig(b.val_min), fmt_sig(b.val_max))
        };
        writeln!(out, "    {:>17}: {:>6} {}{}", range, b.count, bar, marker)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
