
# Data Backend for Expedited Acquisition and Neighbourhood Search

## Installation

```sh
make install          # data-beans + data-beans-sim, HDF5 if detected
make install HDF5=off # without .h5/.h5ad support
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

## For an agent (MCP)

`data-beans mcp` serves the commands as [Model Context Protocol](https://modelcontextprotocol.io) tools over stdio:

```sh
claude mcp add data-beans -- data-beans mcp
```

The read tools are `info`, `names`, `statistics`, `values` and `list_contents`. `qc` runs the cell QC senna and pinto apply (near-empty floor, MAD bands on detected genes, total counts and the mito fraction): without `out` it reports the bands and what they would drop, with `out` it writes the kept cells. `plot` returns the QC figure, or a statistic's histogram, as an image. The write tools, `import`, `subset`, `subsample`, `split`, `convert` and `merge` (with `common_rows` for backends whose rows differ), build new backends. Each write goes to an `out` that must not exist yet: no tool replaces a file, and `squeeze`, which rewrites its input, is not offered.

The server keeps the project's records the way legume-mcp lays them out, after [Noble (2009), *A quick guide to organizing computational biology projects*, PLoS Comput Biol 5(7):e1000424](https://doi.org/10.1371/journal.pcbi.1000424): every step that writes goes into the day's `results/<date>/runall.sh`, which replays the session, and into `results/INDEX.tsv`; outputs the agent does not name go in `results/<date>/`; `import` notes where its input came from in `data/README.md`. `LEGUME_MCP_JOURNAL=off` turns the records off.

The same QC and figures are on the command line:

```sh
data-beans qc data.zarr.zip --qc-mito-pattern '(?i)^MT-'            # summary only
data-beans qc data.zarr.zip --qc-mito-pattern '(?i)^MT-' -o clean   # and the kept cells
data-beans plot qc data.zarr.zip --qc-mito-pattern '(?i)^MT-' -o qc.png
data-beans plot stat data.zarr.zip -s row --stat sum -o genes.png
```
