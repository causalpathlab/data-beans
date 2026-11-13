use super::run::{run_trajectory, TrajectoryArgs};
use super::*;
use crate::sim::copula::reference::open_reference;
use crate::sim::copula::sampler::{sample_reference_counts, CellPrograms, CellTerm, Triplets};
use crate::sim::copula::{fit_global_copula, GlobalCopulaArgs};
use crate::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use clap::Parser;
use rand::SeedableRng;
use rand_distr::{Distribution, Gamma, Poisson};

fn rng(seed: u64) -> rand::rngs::StdRng {
    rand::rngs::StdRng::seed_from_u64(seed)
}

#[test]
fn trees_know_their_lineages() -> anyhow::Result<()> {
    let star = LineageTree::star(3)?;
    assert_eq!(star.n_branches(), 4);
    assert_eq!(star.leaves, vec![1, 2, 3]);
    assert!((0..3).all(|l| star.on_lineage(0, l)));
    assert!(star.on_lineage(2, 1) && !star.on_lineage(2, 0));

    let line = LineageTree::star(1)?;
    assert_eq!((line.n_branches(), line.n_lineages()), (1, 1));

    // 0 → {1, 2}, 1 → {3, 4}
    let t = LineageTree::from_parents(&[0, 0, 1, 1])?;
    assert_eq!(t.leaves, vec![2, 3, 4]);
    assert_eq!(t.depth, vec![0, 1, 1, 2, 2]);
    assert!(t.on_lineage(1, 1) && t.on_lineage(1, 2) && !t.on_lineage(1, 0));
    assert!(LineageTree::from_parents(&[1]).is_err());
    Ok(())
}

#[test]
fn curves_are_continuous_at_branch_points() -> anyhow::Result<()> {
    let tree = LineageTree::from_parents(&[0, 0, 1, 1])?;
    let mut r = rng(1);
    let cells = CellPlacement::sample(&tree, 2000, &[], (1.0, 1.0), &mut r)?;
    let mut curves = GeneCurves::sample(&tree, 50, None, 1.0, 4, &mut r)?;
    curves.standardize(&cells);
    assert_eq!(curves.dynamic.len(), 50);
    for g in 0..50 {
        for b in 1..tree.n_branches() {
            let p = tree.parent[b].unwrap();
            let gap = (curves.value(g, b, 0.0) - curves.value(g, p, 1.0)).abs();
            assert!(gap < 1e-5, "gene {g} branch {b}: gap {gap}");
        }
        // Small steps in t make small steps in the curve.
        for b in 0..tree.n_branches() {
            for i in 0..100 {
                let (t0, t1) = (i as f32 / 100.0, (i + 1) as f32 / 100.0);
                assert!((curves.value(g, b, t1) - curves.value(g, b, t0)).abs() < 0.5);
            }
        }
    }
    // Standardised over the cells.
    let g = curves.dynamic[0];
    let x: Vec<f64> = (0..cells.n_cells())
        .map(|j| curves.value(g, cells.branch[j], cells.time[j]) as f64)
        .collect();
    let m = x.iter().sum::<f64>() / x.len() as f64;
    let v = x.iter().map(|a| (a - m).powi(2)).sum::<f64>() / x.len() as f64;
    assert!(
        m.abs() < 1e-4 && (v - 1.0).abs() < 1e-3,
        "mean {m}, var {v}"
    );
    Ok(())
}

#[test]
fn one_basis_function_makes_monotone_curves() -> anyhow::Result<()> {
    let tree = LineageTree::star(2)?;
    let curves = GeneCurves::sample(&tree, 20, None, 1.0, 1, &mut rng(2))?;
    for g in 0..20 {
        for b in 0..tree.n_branches() {
            let steps: Vec<f32> = (0..20)
                .map(|i| {
                    curves.value(g, b, (i + 1) as f32 / 20.0) - curves.value(g, b, i as f32 / 20.0)
                })
                .collect();
            assert!(steps.iter().all(|&d| d >= 0.0) || steps.iter().all(|&d| d <= 0.0));
        }
    }
    Ok(())
}

#[test]
fn only_a_share_of_the_candidates_is_dynamic() -> anyhow::Result<()> {
    let tree = LineageTree::star(3)?;
    let pool: Vec<usize> = (0..100).map(|i| 2 * i).collect();
    let c = GeneCurves::sample(&tree, 200, Some(&pool), 0.25, 3, &mut rng(3))?;
    assert_eq!(c.dynamic.len(), 25);
    assert!(c.dynamic.iter().all(|g| g % 2 == 0));
    let f = c.at(2, 0.7);
    assert!((0..200)
        .filter(|g| !c.dynamic.contains(g))
        .all(|g| f[g] == 0.0));
    Ok(())
}

#[test]
fn cells_follow_branch_weights_and_time_shape() -> anyhow::Result<()> {
    let tree = LineageTree::star(2)?;
    let c = CellPlacement::sample(&tree, 30_000, &[2.0, 1.0, 1.0], (2.0, 1.0), &mut rng(4))?;
    let on_root = c.branch.iter().filter(|&&b| b == 0).count() as f64 / 30_000.0;
    assert!((on_root - 0.5).abs() < 0.02, "root share {on_root}");
    // Beta(2, 1) has mean 2/3.
    let mean_t = c.time.iter().map(|&t| t as f64).sum::<f64>() / 30_000.0;
    assert!((mean_t - 2.0 / 3.0).abs() < 0.01, "mean t {mean_t}");
    assert!(CellPlacement::sample(&tree, 5, &[1.0], (1.0, 1.0), &mut rng(4)).is_err());
    let none_last =
        CellPlacement::sample(&tree, 30_000, &[1.0, 1.0, 0.0], (1.0, 1.0), &mut rng(5))?;
    assert!(none_last.branch.iter().all(|&b| b < 2));
    Ok(())
}

fn args(extra: &[&str]) -> TrajectoryArgs {
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        args: TrajectoryArgs,
    }
    let mut v = vec!["x"];
    v.extend_from_slice(extra);
    Cli::parse_from(v).args
}

/// Dense gene × cell counts of a backend.
fn read_dense(path: &str) -> anyhow::Result<Vec<Vec<f32>>> {
    let sc = open_reference(path)?;
    let (dd, nn) = (sc.num_rows().unwrap(), sc.num_columns().unwrap());
    let csc = sc.read_columns_csc((0..nn).collect())?;
    let mut y = vec![vec![0.0_f32; nn]; dd];
    for (j, col) in csc.col_iter().enumerate() {
        for (&g, &x) in col.row_indices().iter().zip(col.values()) {
            y[g][j] = x;
        }
    }
    Ok(y)
}

/// Spearman correlation.
fn spearman(a: &[f64], b: &[f64]) -> f64 {
    fn rank(x: &[f64]) -> Vec<f64> {
        let mut idx: Vec<usize> = (0..x.len()).collect();
        idx.sort_by(|&i, &j| x[i].total_cmp(&x[j]));
        let mut r = vec![0.0; x.len()];
        for (k, &i) in idx.iter().enumerate() {
            r[i] = k as f64;
        }
        r
    }
    let (ra, rb) = (rank(a), rank(b));
    let n = a.len() as f64;
    let m = (n - 1.0) / 2.0;
    let cov: f64 = ra.iter().zip(&rb).map(|(x, y)| (x - m) * (y - m)).sum();
    let var: f64 = ra.iter().map(|x| (x - m).powi(2)).sum();
    cov / var
}

#[test]
fn synthetic_means_follow_the_curves_and_runs_repeat() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let out = dir.path().join("traj").to_string_lossy().into_owned();
    let line = [
        "--rows",
        "60",
        "--cols",
        "3000",
        "--lineages",
        "2",
        "--dynamic-genes",
        "0.5",
        "--basis",
        "1",
        "--depth",
        "6000",
        "--beta-scale",
        "0.3",
        "--rseed",
        "7",
        "-o",
    ];
    let mut a: Vec<&str> = line.to_vec();
    a.push(&out);
    run_trajectory(&args(&a))?;
    let y = read_dense(&format!("{out}.zarr.zip"))?;

    let out2 = dir.path().join("again").to_string_lossy().into_owned();
    let mut a2: Vec<&str> = line.to_vec();
    a2.push(&out2);
    run_trajectory(&args(&a2))?;
    assert_eq!(
        y,
        read_dense(&format!("{out2}.zarr.zip"))?,
        "same seed, same counts"
    );

    // Rebuild the structure the run drew: the same seed, the same draws.
    let tree = LineageTree::star(2)?;
    let mut r = rng(7);
    let cells = CellPlacement::sample(&tree, 3000, &[], (1.0, 1.0), &mut r)?;
    let mut curves = GeneCurves::sample(&tree, 60, None, 0.5, 1, &mut r)?;
    curves.standardize(&cells);
    assert_eq!(curves.dynamic.len(), 30);

    // Along each lineage, a dynamic gene's binned mean tracks exp(f̃).
    let mut agree = 0;
    for &g in &curves.dynamic {
        for l in 0..tree.n_lineages() {
            let mut bins = vec![(0.0_f64, 0.0_f64, 0usize); 10];
            for (j, &count) in y[g].iter().enumerate() {
                if !tree.on_lineage(cells.branch[j], l) {
                    continue;
                }
                let k = ((cells.tree_time(&tree, j) / 2.0 * 10.0) as usize).min(9);
                bins[k].0 += count as f64;
                bins[k].1 += curves.value(g, cells.branch[j], cells.time[j]).exp() as f64;
                bins[k].2 += 1;
            }
            let obs: Vec<f64> = bins.iter().map(|b| b.0 / b.2 as f64).collect();
            let want: Vec<f64> = bins.iter().map(|b| b.1 / b.2 as f64).collect();
            if spearman(&obs, &want) > 0.8 {
                agree += 1;
            }
        }
    }
    assert!(agree >= 54, "{agree} of 60 gene-lineage trends recovered");

    // Static genes keep one mean: their correlation with tree time is small.
    let static_g = (0..60).find(|g| !curves.dynamic.contains(g)).unwrap();
    let tt: Vec<f64> = (0..3000)
        .map(|j| cells.tree_time(&tree, j) as f64)
        .collect();
    let raw: Vec<f64> = (0..3000).map(|j| y[static_g][j] as f64).collect();
    assert!(spearman(&raw, &tt).abs() < 0.1);
    Ok(())
}

/// A reference of `dd` genes × `nn` cells, gamma-Poisson counts with
/// per-gene means and dispersions, written as a zarr.
fn write_reference(dir: &std::path::Path, dd: usize, nn: usize) -> anyhow::Result<String> {
    let mut r = rng(11);
    let mut trips: Triplets = Vec::new();
    for g in 0..dd {
        let mu = 0.2 + 3.0 * (g as f64 / dd as f64);
        let size = if g % 5 == 0 {
            1e6
        } else {
            0.5 + (g % 7) as f64
        };
        let gamma = Gamma::new(size, mu / size)?;
        for j in 0..nn {
            let l = gamma.sample(&mut r);
            let y: f64 = Poisson::new(l.max(1e-12))?.sample(&mut r);
            if y > 0.0 {
                trips.push((g as u64, j as u64, y as f32));
            }
        }
    }
    let path = dir.join("ref.zarr").to_string_lossy().into_owned();
    let names: Vec<Box<str>> = (0..dd).map(|g| format!("g{g}").into()).collect();
    let cols: Vec<Box<str>> = (0..nn).map(|j| format!("c{j}").into()).collect();
    let mut data = create_sparse_from_triplets(
        &trips,
        (dd, nn, trips.len()),
        Some(&path),
        Some(&SparseIoBackend::Zarr),
    )?;
    data.register_row_names_vec(&names);
    data.register_column_names_vec(&cols);
    Ok(path)
}

#[test]
fn reference_moments_stay_near_the_reference() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let (dd, nn) = (80, 2000);
    let path = write_reference(dir.path(), dd, nn)?;
    let sc = open_reference(&path)?;
    let fit = fit_global_copula(&GlobalCopulaArgs {
        sc: &sc,
        n_hvg: 40,
        copula_rank: 10,
        regularization: 1e-3,
        hvg_min_detection: 0.005,
        groups: crate::sim::copula::GroupSpec::Cluster(1),
    })?;
    let tree = LineageTree::star(3)?;
    let mut r = rng(5);
    let cells = CellPlacement::sample(&tree, 4000, &[], (1.0, 1.0), &mut r)?;
    let candidates = super::run::dynamic_candidates(&fit);
    let mut curves = GeneCurves::sample(&tree, dd, Some(&candidates), 0.5, 3, &mut r)?;
    curves.standardize(&cells);
    let term = |j: usize| curves.at(cells.branch[j], cells.time[j]);
    let membership = vec![0usize; 4000];
    let delta = vec![nalgebra::DVector::zeros(dd)];
    // The largest |sim/ref − 1| of each gene's variance, and the amplitudes.
    let run = |cap_spread: bool, pve_noise: f32| -> anyhow::Result<(f64, Vec<f32>)> {
        let programs = CellPrograms {
            term: CellTerm::PerCell(&term),
            pve_topic: 1.0,
            cap_spread,
            pve_noise,
            batch_delta: &delta,
            batch_membership: &membership,
        };
        let mut sample = sample_reference_counts(&fit, 0.0, &programs, true, 9, 0)?;
        let (trips, scaling) = (sample.triplets, sample.scaling.swap_remove(0));
        let mut s1 = vec![0.0_f64; dd];
        let mut s2 = vec![0.0_f64; dd];
        for &(g, _, x) in &trips {
            s1[g as usize] += x as f64;
            s2[g as usize] += (x as f64).powi(2);
        }
        let mut worst = 0.0_f64;
        for &g in &fit.active_genes {
            let m = s1[g] / 4000.0;
            let v = s2[g] / 4000.0 - m * m;
            let (rm, rv) = (fit.marginals[g].mean(), fit.marginals[g].variance());
            assert!((m / rm - 1.0).abs() < 0.12, "gene {g}: mean {m} vs {rm}");
            worst = worst.max((v / rv - 1.0).abs());
        }
        Ok((worst, scaling.amp))
    };
    let (capped, amp) = run(true, 0.2)?;
    assert!(curves.dynamic.iter().any(|&g| amp[g] > 0.5));
    assert!(amp.iter().all(|&a| a <= 1.0 + 1e-6));
    assert!(
        capped < 0.3,
        "variance off by {capped:.2} with the spread cap"
    );
    // The log-variance cap alone lets skewed curves overshoot.
    let (uncapped, _) = run(false, 0.0)?;
    assert!(uncapped > capped, "{uncapped:.2} vs {capped:.2}");
    Ok(())
}
