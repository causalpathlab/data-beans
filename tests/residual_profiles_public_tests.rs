//! `residual_profiles` is part of the public API: a caller outside the crate
//! can take the profiles the feature partition clusters in and group them
//! its own way.

#![cfg(feature = "alg")]

use data_beans::alg::feature_coarsening::{informative_features, residual_profiles};
use nalgebra::DMatrix;

/// Two planted feature groups over six pseudobulks, plus one flat feature.
fn counts() -> (DMatrix<f32>, Vec<f32>) {
    let rows: [[f32; 6]; 5] = [
        [90.0, 80.0, 85.0, 5.0, 4.0, 6.0],
        [70.0, 75.0, 60.0, 3.0, 5.0, 2.0],
        [4.0, 6.0, 5.0, 80.0, 95.0, 70.0],
        [2.0, 3.0, 4.0, 60.0, 65.0, 75.0],
        [20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
    ];
    let m = DMatrix::from_fn(5, 6, |g, s| rows[g][s]);
    (m, vec![10.0; 6])
}

#[test]
fn profiles_are_unit_rows_in_the_order_asked_for() {
    let (counts, sizes) = counts();
    let rows = [2usize, 0, 3];
    let z = residual_profiles(&counts, &sizes, &rows, 1);
    assert_eq!(z.shape(), (rows.len(), sizes.len()));
    for i in 0..z.nrows() {
        assert!((z.row(i).norm() - 1.0).abs() < 1e-5);
    }
    // Rows 2 and 3 share a group; row 0 is the other one.
    assert!(z.row(0).dot(&z.row(2)) > 0.9);
    assert!(z.row(0).dot(&z.row(1)) < -0.9);
}

#[test]
fn the_flat_feature_is_the_one_without_grouping_evidence() {
    let (counts, sizes) = counts();
    assert_eq!(
        informative_features(&counts, &sizes),
        vec![true, true, true, true, false]
    );
}
