#![cfg(feature = "ndarray")]

//! A preload's share of the process-wide preload budget must come back when
//! the preloaded arrays go — on `clean_preloaded_*` and when the backend is
//! dropped — or a later stage of the same process is refused a preload with
//! memory free. Repeating a preload must not read (or reserve) twice.
//!
//! One test in this binary: the reservation counter is process-wide, so a
//! second test preloading in parallel would race the exact byte counts.

use data_beans::sparse_io::*;

fn exercise(backend: Option<&SparseIoBackend>) -> anyhow::Result<()> {
    let arr = ndarray::Array2::<f32>::from_elem((50, 40), 1.0);
    let cost = 50 * 40 * 12;
    let base = preload_reserved_bytes();

    let mut data = create_sparse_from_ndarray(&arr, None, backend)?;
    assert_eq!(preload_reserved_bytes(), base, "creating does not preload");

    data.preload_columns()?;
    assert!(data.csc_column_arrays().is_some());
    assert_eq!(preload_reserved_bytes(), base + cost);

    // already preloaded: a no-op, not a second read and reservation
    data.preload_columns()?;
    assert_eq!(preload_reserved_bytes(), base + cost);

    data.preload_rows()?;
    data.preload_rows()?;
    assert_eq!(preload_reserved_bytes(), base + 2 * cost);

    data.clean_preloaded_columns();
    assert!(data.csc_column_arrays().is_none());
    assert_eq!(
        preload_reserved_bytes(),
        base + cost,
        "cleaning gives it back"
    );

    // preloading again after a clean reserves again
    data.preload_columns()?;
    assert_eq!(preload_reserved_bytes(), base + 2 * cost);

    drop(data);
    assert_eq!(preload_reserved_bytes(), base, "dropping gives it back");
    Ok(())
}

#[test]
fn preload_reservations_are_given_back() -> anyhow::Result<()> {
    exercise(Some(&SparseIoBackend::Zarr))?;
    #[cfg(feature = "hdf5")]
    exercise(Some(&SparseIoBackend::HDF5))?;
    Ok(())
}
