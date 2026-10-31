//! Where values sit across a histogram, and its bins: linear, or log on
//! `ln(shift + x)`, the shift chosen from the data so zeros have a place
//! and values mostly below 1 (means, fractions) spread out. The printed
//! histograms, the views' and the figures' all read values on it.

/// Where values sit across a panel: linear, or log on `ln(shift + x)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Axis {
    pub lo: f64,
    pub hi: f64,
    pub log: bool,
    /// What a log axis adds before taking the log: 1 for counts, so zero
    /// has a place; less for values mostly below 1 (means, fractions).
    pub shift: f64,
}

impl Axis {
    pub fn linear(lo: f64, hi: f64) -> Self {
        Self {
            lo,
            hi,
            log: false,
            shift: 0.0,
        }
    }

    /// A log axis on `ln(1 + x)`.
    pub fn log1p(lo: f64, hi: f64) -> Self {
        Self {
            lo,
            hi,
            log: true,
            shift: 1.0,
        }
    }

    /// The finite range of `values`, widened where they are all one value.
    /// A log axis shifts by 1 where the values reach 20, else by the
    /// smallest positive value's half, so small ones spread out too.
    pub fn fit(values: &[f32], log: bool) -> Self {
        let (mut lo, mut hi) = values
            .iter()
            .filter(|v| v.is_finite())
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v as f64), hi.max(v as f64))
            });
        if !lo.is_finite() {
            (lo, hi) = (0.0, 1.0);
        }
        if log {
            lo = lo.max(0.0);
        }
        if hi <= lo {
            hi = lo + if log { lo.max(1.0) } else { 1.0 };
        }
        let shift = if !log || hi >= 20.0 {
            1.0
        } else {
            let least = values
                .iter()
                .filter(|v| v.is_finite() && **v > 0.0)
                .fold(f64::INFINITY, |m, &v| m.min(v as f64));
            (least / 2.0).clamp(hi * 1e-4, 1.0)
        };
        Self { lo, hi, log, shift }
    }

    /// This axis reaching `v` too.
    pub fn including(mut self, v: f64) -> Self {
        if v.is_finite() {
            self.lo = self.lo.min(if self.log { v.max(0.0) } else { v });
            self.hi = self.hi.max(v);
        }
        self
    }

    fn t(&self, v: f64) -> f64 {
        if self.log {
            (v.max(0.0) + self.shift).ln()
        } else {
            v
        }
    }

    /// Where `v` falls, 0 at `lo` and 1 at `hi`.
    pub fn frac(&self, v: f64) -> f64 {
        let (a, b) = (self.t(self.lo), self.t(self.hi));
        if b > a {
            (self.t(v) - a) / (b - a)
        } else {
            0.0
        }
    }

    /// The value at `f` of the way from `lo` to `hi`.
    pub fn at(&self, f: f64) -> f64 {
        let (a, b) = (self.t(self.lo), self.t(self.hi));
        let t = a + f * (b - a);
        if self.log {
            (t.exp() - self.shift).max(0.0)
        } else {
            t
        }
    }

    /// The bin of `n` that `v` falls in, clamped to the range.
    pub fn bin(&self, v: f64, n: usize) -> usize {
        ((self.frac(v) * n as f64).floor().max(0.0) as usize).min(n.saturating_sub(1))
    }

    /// How many of `values` (those `pick` takes) fall in each of `n` bins.
    pub fn counts(&self, values: &[f32], n: usize, pick: impl Fn(usize) -> bool) -> Vec<usize> {
        let mut out = vec![0; n];
        for (i, &v) in values.iter().enumerate() {
            if v.is_finite() && pick(i) {
                out[self.bin(v as f64, n)] += 1;
            }
        }
        out
    }

    /// Tick values inside the range: 1-2-5 steps on a log axis (and 0 where
    /// it starts there), round steps on a linear one.
    pub fn ticks(&self) -> Vec<f64> {
        if self.log {
            let mut t: Vec<f64> = (self.lo <= 0.0).then_some(0.0).into_iter().collect();
            for e in -3..10 {
                for m in [1.0, 2.0, 5.0] {
                    let v = m * 10f64.powi(e);
                    if v >= self.lo && v <= self.hi {
                        t.push(v);
                    }
                }
            }
            // On ln(shift + x) the small ones crowd at the start: keep a tick
            // only a twentieth of the axis past the last one kept.
            let mut kept: Vec<f64> = vec![];
            for v in t {
                if kept
                    .last()
                    .is_none_or(|&l| self.frac(v) - self.frac(l) >= 0.05)
                {
                    kept.push(v);
                }
            }
            let mut t = kept;
            // Thin to at most 8, keeping the decades.
            while t.len() > 8 {
                let decade = |v: &f64| *v == 0.0 || (v.log10() - v.log10().round()).abs() < 1e-9;
                if t.iter().all(decade) {
                    t = t.into_iter().step_by(2).collect();
                } else {
                    t.retain(decade);
                }
            }
            t
        } else {
            let span = self.hi - self.lo;
            let raw = span / 5.0;
            let mag = 10f64.powf(raw.log10().floor());
            let step = [1.0, 2.0, 2.5, 5.0, 10.0]
                .iter()
                .map(|m| m * mag)
                .find(|s| span / s <= 6.0)
                .unwrap_or(10.0 * mag);
            // Each tick k steps in, rounded to the step's decimals so 0.15
            // reads 0.15.
            let places = 10f64.powi((2 - step.log10().floor() as i32).max(0));
            let first = (self.lo / step).ceil() as i64;
            (first..)
                .map(|k| (k as f64 * step * places).round() / places)
                .take_while(|v| *v <= self.hi + 1e-9 * step)
                .map(|v| if v == 0.0 { 0.0 } else { v })
                .collect()
        }
    }
}
