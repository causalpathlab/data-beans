use super::*;

#[test]
fn lower_edges_open_their_bin_on_every_scale() {
    for scale in [Scale::Sqrt, Scale::Linear] {
        let b = Binning::new(scale, 5000.0, true);
        for k in 1..60 {
            let e = b.lower_edge(k);
            assert!(b.key(e as f64) >= k, "{scale:?} k={k} e={e}");
            assert!(e == 0 || b.key((e - 1) as f64) < k, "{scale:?} k={k} e={e}");
        }
    }
}

#[test]
fn continuous_bins_cover_the_range() {
    for scale in [Scale::Log, Scale::Sqrt, Scale::Linear] {
        let b = Binning::new(scale, 18.0, false);
        let (lo, hi) = (b.key(0.0), b.key(18.0));
        assert!(hi > lo, "{scale:?}");
        let vals = [0.0f32, 0.5, 3.2, 18.0];
        let counts = bin_counts(vals.iter().copied(), &b, lo, (hi - lo + 1) as usize);
        assert_eq!(counts.iter().sum::<usize>(), vals.len(), "{scale:?}");
    }
}

#[test]
fn compact_labels() {
    assert_eq!(compact(0.0), "0");
    assert_eq!(compact(0.25), "0.25");
    assert_eq!(compact(950.0), "950");
    assert_eq!(compact(1234.0), "1.2k");
    assert_eq!(compact(35_000.0), "35k");
    assert_eq!(compact(1_100_000.0), "1.1M");
}
