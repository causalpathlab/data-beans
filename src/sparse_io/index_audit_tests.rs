use super::*;

#[test]
fn clean_structure_passes() {
    // 3 columns over 4 rows: [0, 2], [], [1, 3]
    let indptr = [0u64, 2, 2, 4];
    let indices = [0u64, 2, 1, 3];
    check_compressed("t", &indptr, &indices, 4, 3, 4).unwrap();
}

#[test]
fn single_high_bit_is_caught_and_reported() {
    let indptr = [0u64, 2, 2, 4];
    let indices = [0u64, 2, (1 << 62) + 1, 3];
    let err = check_compressed("t", &indptr, &indices, 4, 3, 4)
        .unwrap_err()
        .to_string();
    assert!(err.contains("position 2"), "{err}");
    assert!(err.contains("0x4000000000000001"), "{err}");
}

#[test]
fn indptr_faults_are_caught() {
    let indices = [0u64, 1, 2, 3];
    // wrong length
    assert!(check_compressed("t", &[0, 2, 4], &indices, 4, 3, 4).is_err());
    // does not start at 0
    assert!(check_compressed("t", &[1, 2, 2, 4], &indices, 4, 3, 4).is_err());
    // decreases
    assert!(check_compressed("t", &[0, 3, 2, 4], &indices, 4, 3, 4).is_err());
    // total disagrees with the arrays
    assert!(check_compressed("t", &[0, 2, 2, 3], &indices, 4, 3, 4).is_err());
    // indices and values disagree
    assert!(check_compressed("t", &[0, 2, 2, 4], &indices, 5, 3, 4).is_err());
}

#[test]
fn slot_range_is_checked() {
    check_slot("t", 0, 2, 4, 4).unwrap();
    assert!(check_slot("t", 0, 3, 2, 4).is_err());
    assert!(check_slot("t", 0, 2, 5, 4).is_err());
}
