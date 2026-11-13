//! Lineage / trajectory structure for `data-beans-sim trajectory`: cells on
//! the branches of a lineage tree, each gene's log mean a smooth function of
//! the cell's branch and pseudotime.
//!
//! Design, after the simulators that model a trajectory without a gene
//! regulatory network:
//!
//! - scDesign3 (Song et al. 2024, Nat Biotechnol 42:247) fits each gene's
//!   marginal parameters as a GAM of pseudotime, one smooth per lineage.
//! - Splatter's "paths" (Zappia et al. 2017, Genome Biol 18:174) walk each
//!   gene's mean from a path's start to its end, a path starting where its
//!   parent path ends.
//! - PROSSTT (Papadopoulos et al. 2019, Bioinformatics 35:3517) draws each
//!   gene's relative expression along a tree as random walks of a few
//!   programs, continuous at branch points.
//! - dyngen (Cannoodt et al. 2021, Nat Commun 12:3942) simulates the
//!   regulatory kinetics behind the trajectory; it is the realistic one and
//!   the costly one, and it does not condition on a reference.
//!
//! We keep the shape the first three share, a smooth curve per gene on a
//! tree, continuous at branch points, and draw it at random:
//!
//! ```text
//! f_g(root, t)  = Σ_k c_{g,root,k} · φ_k(t)
//! f_g(b, t)     = f_g(parent(b), 1) + Σ_k c_{g,b,k} · φ_k(t)
//! φ_k(t)        = sin((k − ½)·π·t) / ((k − ½)·π),   k = 1..K
//! c_{g,b,k}     ~ N(0, 1 / Z),    Z = Σ_k φ_k(1)²
//! f̃_g          = (f_g − mean_j f_g) / sd_j f_g      over the simulated cells
//! ```
//!
//! `φ_k` are the first `K` Karhunen–Loève terms of Brownian motion, so a
//! curve is a smoothed Brownian motion down the tree: every `φ_k(0) = 0`,
//! so a child starts where its parent ends; each branch adds unit
//! log-variance by its end (`Z` normalises the truncation); `K = 1` gives
//! monotone curves, a larger `K` wigglier ones. Lineages share their path
//! from the root and diverge after it. Only a share of genes is dynamic;
//! the others have `f_g = 0`. Standardising over the cells makes
//! `--pve-trajectory` the log-variance the trajectory adds to a dynamic
//! gene, as `--pve-topic` is for topics.
//!
//! Cells: a branch per cell by `--branch-weights` (equal by default, so
//! cells are as dense per unit pseudotime on every branch), and a
//! pseudotime `t ∈ [0, 1]` within it from `Beta(a, b)` (uniform by
//! default). A cell's tree time is its branch's depth plus `t`.
//!
//! Counts:
//!
//! ```text
//! log λ_{g,j} = log(depth/G) + log β_g
//!             + √π_traj · f̃_g(b_j, t_j) − c_g            (trajectory)
//!             + √π_noise · ε_{g,j} − π_noise / 2          (noise)
//!             + √π_batch · z_{g,B(j)} − c_{g,batch}       (batch)
//! ```
//!
//! - synthetic: `y ~ Poisson(λ)`, `β_g` log-normal with log-sd
//!   `--beta-scale` and mean 1, `ε, z ~ N(0, 1)`; every `c` centres its
//!   term so a gene's mean over the cells is `(depth/G) · β_g`;
//! - reference: the baseline is the gene's reference distribution and the
//!   trajectory is the per-cell term of the reference sampler
//!   ([`crate::sim::copula::sampler`], [`CellTerm::PerCell`]) with the same
//!   noise and batch terms; trajectory and noise are capped per gene so
//!   their log-variance stays within the gene's extra-Poisson log-variance
//!   in the reference (and, so a skewed curve does not overshoot, its
//!   multiplier's spread `E[T²]/E[T]²` within the rates' `1 + cv²`), and
//!   the gene's
//!   Poisson mixture is tempered to leave it room, so every gene keeps the
//!   reference's mean and variance, depth coupling and latent copula. A
//!   gene with no extra-Poisson variation in the reference stays static.
//!
//! [`CellTerm::PerCell`]: crate::sim::copula::sampler::CellTerm::PerCell

pub mod run;

use crate::sim::copula::sampler::first_two_moments;
use nalgebra::{DMatrix, DVector};
use rand_distr::{Beta, Distribution, Normal, Uniform};

/// A tree of branches: branch 0 is the root, every other branch starts at
/// the end of its parent. A lineage is a root-to-leaf path.
#[derive(Debug, Clone)]
pub struct LineageTree {
    /// `parent[b]`, `None` for the root.
    pub parent: Vec<Option<usize>>,
    /// Branches between `b` and the root, the root's 0.
    pub depth: Vec<usize>,
    /// Lineage `l` ends at branch `leaves[l]`, in branch order.
    pub leaves: Vec<usize>,
    /// `on_lineage[b][l]`: branch `b` is on lineage `l`'s path.
    on_lineage: Vec<Vec<bool>>,
}

impl LineageTree {
    /// A root that splits into `k` lineages; one lineage is the root alone.
    pub fn star(k: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(k >= 1, "a tree needs at least one lineage");
        let parents = if k == 1 { Vec::new() } else { vec![0; k] };
        Self::from_parents(&parents)
    }

    /// The tree whose branch `i + 1` starts at the end of `parents[i]`; each
    /// parent comes before its child.
    pub fn from_parents(parents: &[usize]) -> anyhow::Result<Self> {
        let n = parents.len() + 1;
        let mut parent = vec![None];
        let mut depth = vec![0];
        for (i, &p) in parents.iter().enumerate() {
            anyhow::ensure!(
                p <= i,
                "branch {}'s parent {p} must be an earlier branch (0..={i})",
                i + 1
            );
            parent.push(Some(p));
            depth.push(depth[p] + 1);
        }
        let mut has_child = vec![false; n];
        for &p in parents {
            has_child[p] = true;
        }
        let leaves: Vec<usize> = (0..n).filter(|&b| !has_child[b]).collect();
        let mut on_lineage = vec![vec![false; leaves.len()]; n];
        for (l, &leaf) in leaves.iter().enumerate() {
            let mut b = Some(leaf);
            while let Some(x) = b {
                on_lineage[x][l] = true;
                b = parent[x];
            }
        }
        Ok(Self {
            parent,
            depth,
            leaves,
            on_lineage,
        })
    }

    pub fn n_branches(&self) -> usize {
        self.parent.len()
    }

    pub fn n_lineages(&self) -> usize {
        self.leaves.len()
    }

    pub fn on_lineage(&self, branch: usize, lineage: usize) -> bool {
        self.on_lineage[branch][lineage]
    }
}

/// Each cell's branch and pseudotime `t ∈ [0, 1]` within it.
#[derive(Debug, Clone)]
pub struct CellPlacement {
    pub branch: Vec<usize>,
    pub time: Vec<f32>,
}

impl CellPlacement {
    /// `n` cells: a branch from `weights` (one per branch; equal when empty)
    /// and `t ~ Beta(shape)`.
    pub fn sample(
        tree: &LineageTree,
        n: usize,
        weights: &[f64],
        shape: (f64, f64),
        rng: &mut impl rand::Rng,
    ) -> anyhow::Result<Self> {
        let nb = tree.n_branches();
        let weights: Vec<f64> = if weights.is_empty() {
            vec![1.0; nb]
        } else {
            weights.to_vec()
        };
        anyhow::ensure!(
            weights.len() == nb,
            "{} branch weights for {nb} branches",
            weights.len()
        );
        let total: f64 = weights.iter().sum();
        anyhow::ensure!(
            weights.iter().all(|w| w.is_finite() && *w >= 0.0) && total > 0.0,
            "branch weights must be non-negative with a positive sum"
        );
        // Rounding can leave the last cumulative share below 1.
        let last = weights.iter().rposition(|&w| w > 0.0).unwrap_or(0);
        let cumulative: Vec<f64> = weights
            .iter()
            .scan(0.0, |s, w| {
                *s += w / total;
                Some(*s)
            })
            .collect();
        let beta = Beta::new(shape.0, shape.1)
            .map_err(|e| anyhow::anyhow!("pseudotime shape {shape:?}: {e}"))?;
        let unif = Uniform::new(0.0_f64, 1.0)?;
        let mut branch = Vec::with_capacity(n);
        let mut time = Vec::with_capacity(n);
        for _ in 0..n {
            let u = unif.sample(rng);
            let b = cumulative.partition_point(|&c| c <= u).min(last);
            branch.push(b);
            time.push(beta.sample(rng) as f32);
        }
        Ok(Self { branch, time })
    }

    pub fn n_cells(&self) -> usize {
        self.branch.len()
    }

    /// Branch depth plus pseudotime: the cell's distance from the root.
    pub fn tree_time(&self, tree: &LineageTree, j: usize) -> f32 {
        tree.depth[self.branch[j]] as f32 + self.time[j]
    }
}

/// Variance over the cells below which a curve counts as flat.
const MIN_CURVE_VARIANCE: f64 = 1e-10;

/// `φ_k(t)`, the `k`-th (0-based) sine basis function.
fn basis(k: usize, t: f32) -> f32 {
    let w = (k as f32 + 0.5) * std::f32::consts::PI;
    (w * t).sin() / w
}

/// Each gene's smooth log-mean curve over the tree; zero for static genes.
#[derive(Debug, Clone)]
pub struct GeneCurves {
    pub n_basis: usize,
    /// The dynamic genes, ascending.
    pub dynamic: Vec<usize>,
    /// Gene × (branch · `n_basis` + k) coefficients `c_{g,b,k}`.
    pub coef: DMatrix<f32>,
    /// Gene × branch: `f_g(b, 0)`, the parent's value at its end.
    pub start: DMatrix<f32>,
}

impl GeneCurves {
    /// Curves for `n_genes` genes: `share` of `candidates` (all genes when
    /// `None`) dynamic, with `n_basis` basis functions and each branch
    /// adding unit log-variance by its end.
    pub fn sample(
        tree: &LineageTree,
        n_genes: usize,
        candidates: Option<&[usize]>,
        share: f32,
        n_basis: usize,
        rng: &mut impl rand::Rng,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(n_basis >= 1, "a curve needs at least one basis function");
        let pool: std::borrow::Cow<[usize]> = match candidates {
            Some(c) => c.into(),
            None => (0..n_genes).collect::<Vec<_>>().into(),
        };
        anyhow::ensure!(
            pool.iter().all(|&g| g < n_genes),
            "a dynamic-gene candidate is past the {n_genes} genes"
        );
        let n_dynamic = ((share.clamp(0.0, 1.0) as f64) * pool.len() as f64).round() as usize;
        let mut dynamic: Vec<usize> = rand::seq::index::sample(rng, pool.len(), n_dynamic)
            .into_iter()
            .map(|i| pool[i])
            .collect();
        dynamic.sort_unstable();

        let nb = tree.n_branches();
        let z: f32 = (0..n_basis).map(|k| basis(k, 1.0).powi(2)).sum();
        let sd = z.sqrt().recip();
        let normal = Normal::new(0.0_f32, 1.0)?;
        let mut coef = DMatrix::<f32>::zeros(n_genes, nb * n_basis);
        for &g in &dynamic {
            for c in 0..nb * n_basis {
                coef[(g, c)] = sd * normal.sample(rng);
            }
        }
        let mut curves = Self {
            n_basis,
            dynamic,
            coef,
            start: DMatrix::zeros(n_genes, nb),
        };
        // Parents come before children, so a parent's start is set first.
        for b in 1..nb {
            let p = tree.parent[b].expect("a non-root branch has a parent");
            for i in 0..curves.dynamic.len() {
                let g = curves.dynamic[i];
                curves.start[(g, b)] = curves.start[(g, p)] + curves.increment(g, p, 1.0);
            }
        }
        Ok(curves)
    }

    /// Rescale each dynamic gene's curve to mean 0 and variance 1 over
    /// `cells`; a gene whose curve does not vary over them becomes static.
    pub fn standardize(&mut self, cells: &CellPlacement) {
        let (m1, m2) = self.moments_over(cells, |f| f as f64);
        let mut kept = Vec::with_capacity(self.dynamic.len());
        for (i, &g) in self.dynamic.iter().enumerate() {
            let var = m2[i] - m1[i] * m1[i];
            if var <= MIN_CURVE_VARIANCE {
                self.coef.row_mut(g).fill(0.0);
                self.start.row_mut(g).fill(0.0);
                continue;
            }
            let inv = (1.0 / var.sqrt()) as f32;
            self.coef.row_mut(g).scale_mut(inv);
            // The same shift on every branch keeps the curve continuous.
            self.start.row_mut(g).add_scalar_mut(-m1[i] as f32);
            self.start.row_mut(g).scale_mut(inv);
            kept.push(g);
        }
        self.dynamic = kept;
    }

    /// `φ_k(t)` for every `k`.
    fn basis_at(&self, t: f32) -> Vec<f32> {
        (0..self.n_basis).map(|k| basis(k, t)).collect()
    }

    /// `f_g(b, t)` given `phi = φ(t)`.
    fn value_at(&self, g: usize, b: usize, phi: &[f32]) -> f32 {
        let c = b * self.n_basis;
        self.start[(g, b)]
            + phi
                .iter()
                .enumerate()
                .map(|(k, p)| self.coef[(g, c + k)] * p)
                .sum::<f32>()
    }

    /// `f_g(b, t)`.
    pub fn value(&self, g: usize, b: usize, t: f32) -> f32 {
        self.value_at(g, b, &self.basis_at(t))
    }

    /// `Σ_k c_{g,b,k} φ_k(t)`: what branch `b` adds by pseudotime `t`.
    fn increment(&self, g: usize, b: usize, t: f32) -> f32 {
        self.value(g, b, t) - self.start[(g, b)]
    }

    pub fn n_genes(&self) -> usize {
        self.coef.nrows()
    }

    /// Every gene's `f_g(b, t)`, zero for the static ones: the branch's
    /// start plus its basis columns, each contiguous over the genes.
    pub fn at(&self, b: usize, t: f32) -> DVector<f32> {
        let mut f = self.start.column(b).into_owned();
        for (k, p) in self.basis_at(t).into_iter().enumerate() {
            f.axpy(p, &self.coef.column(b * self.n_basis + k), 1.0);
        }
        f
    }

    /// Per dynamic gene, in [`Self::dynamic`]'s order, the mean over
    /// `cells` of `h(f_g)` and of its square.
    fn moments_over(
        &self,
        cells: &CellPlacement,
        h: impl Fn(f32) -> f64 + Sync,
    ) -> (Vec<f64>, Vec<f64>) {
        first_two_moments(self.dynamic.len(), cells.n_cells(), |j, out| {
            let b = cells.branch[j];
            let phi = self.basis_at(cells.time[j]);
            for (o, &g) in out.iter_mut().zip(&self.dynamic) {
                *o = h(self.value_at(g, b, &phi));
            }
        })
    }

    /// Per gene, `ln mean_j exp(amp · f_g(b_j, t_j))` over `cells`: what
    /// keeps the gene's mean when subtracted; 0 for the static genes.
    pub fn log_mean_exp(&self, cells: &CellPlacement, amp: f32) -> Vec<f32> {
        let (m1, _) = self.moments_over(cells, |f| ((amp * f) as f64).exp());
        let mut centre = vec![0.0_f32; self.n_genes()];
        for (&g, m) in self.dynamic.iter().zip(m1) {
            centre[g] = m.ln() as f32;
        }
        centre
    }
}

#[cfg(test)]
mod tests;
