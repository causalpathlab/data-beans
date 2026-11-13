//! The standard normal CDF, inverse CDF and density the copula needs,
//! as thin wrappers over `statrs`.

/// Standard normal CDF Φ; thin wrapper over `statrs::distribution::Normal`.
pub fn phi(z: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF, Normal};
    Normal::standard().cdf(z)
}

/// Standard normal inverse CDF Φ⁻¹; thin wrapper over `statrs::distribution::Normal`.
pub fn inv_phi(u: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF, Normal};
    Normal::standard().inverse_cdf(u)
}

/// Standard normal density φ; 0 at `±∞`.
pub fn std_normal_pdf(z: f64) -> f64 {
    use statrs::distribution::{Continuous, Normal};
    Normal::standard().pdf(z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_distr::Distribution;

    #[test]
    fn phi_inv_phi_round_trip() {
        for &z in &[-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0] {
            let u = phi(z);
            let z_back = inv_phi(u);
            assert!((z_back - z).abs() < 1e-9, "z={} back={}", z, z_back);
        }
    }

    #[test]
    fn phi_inv_phi_extremes() {
        // Tail behavior shouldn't blow up.
        for &u in &[1e-6, 1e-3, 0.5, 1.0 - 1e-3, 1.0 - 1e-6] {
            let z = inv_phi(u);
            assert!(z.is_finite(), "inv_phi({})={} not finite", u, z);
            let _ = rand_distr::Normal::new(0.0, 1.0)
                .unwrap()
                .sample(&mut rand::rng());
        }
    }
}
