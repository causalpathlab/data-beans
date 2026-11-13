use super::*;

#[test]
fn blosc_thread_count_saturates_instead_of_wrapping() {
    assert_eq!(blosc_threads(0), 1);
    assert_eq!(blosc_threads(1), 1);
    assert_eq!(blosc_threads(64), 64);
    assert_eq!(blosc_threads(255), 255);
    // a plain cast to `u8` would turn these into 0 and 1
    assert_eq!(blosc_threads(256), 255);
    assert_eq!(blosc_threads(257), 255);
    assert_eq!(blosc_threads(usize::MAX), 255);
}
