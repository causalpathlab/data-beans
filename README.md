
# Data Backend for Expedited Acquisition and Neighbourhood Search

## Installation

```sh
make install          # data-beans + data-beans-sim, auto-detected backend and HDF5
make install-cpu      # or: install-cuda / install-metal
make help             # what was detected on this host, and the overrides
```

or from crates.io (CPU, no HDF5):

```sh
cargo install data-beans
cargo install data-beans --features sim   # data-beans-sim
```

Here, we implement a wrapper for sparse matrix backends for swift data access by rows and columns without populating everything in memory. Being inspired by [`anndata-rs`](https://github.com/kaizhang/anndata-rs), we use `hdf5` and `zarr` as the back-end storage format. The `hdf5` file or `.zarr.zip` archive (or plain `zarr` directory with `--no-zip`) is organized as follows:

```
(root)
    ├── nrow
    ├── ncol
    ├── nnz
    ├── by_column
    │   ├── data
    │   ├── indices (row indices)
    │   └── indptr (column pointers)
    └── by_row
        ├── data
        ├── indices (column indices)
        └── indptr (row pointers)
```
