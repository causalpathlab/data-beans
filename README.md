
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

## Agents (MCP)

`data-beans mcp` serves data-beans to an AI agent as [Model Context Protocol](https://modelcontextprotocol.io) tools over stdio, for any MCP client: Claude Code, Claude Desktop, Cursor, VS Code, Gemini CLI, Codex CLI and others. The server talks only to the client that started it, over stdin and stdout; data-beans sends nothing anywhere itself. What leaves your machine is up to the agent's own client and model.

### Connect a client

The server is in the default build (the `mcp` feature; `cargo install data-beans` has it). Importing `.h5ad` and 10x `.h5` files needs `--features hdf5` too.

The server keeps its records in a project folder: its working directory, or `LEGUME_MCP_PROJECT`. Set it for clients that start servers somewhere else (Claude Desktop, say).

Most clients (Claude Desktop, Cursor, Gemini CLI) take an `mcpServers` entry:

```json
{
  "mcpServers": {
    "data-beans": {
      "command": "data-beans",
      "args": ["mcp"],
      "env": { "LEGUME_MCP_PROJECT": "/path/to/project" }
    }
  }
}
```

Claude Code, from the project folder:

```sh
claude mcp add data-beans -e LEGUME_MCP_PROJECT="$PWD" -- data-beans mcp
```

VS Code (`.vscode/mcp.json`):

```json
{
  "servers": {
    "data-beans": {
      "type": "stdio",
      "command": "data-beans",
      "args": ["mcp"],
      "env": { "LEGUME_MCP_PROJECT": "${workspaceFolder}" }
    }
  }
}
```

Codex CLI (`~/.codex/config.toml`):

```toml
[mcp_servers.data-beans]
command = "data-beans"
args = ["mcp"]
env = { LEGUME_MCP_PROJECT = "/path/to/project" }
```

### What the agent is told

The server hands these instructions to the client when it connects. Clients that do not pass MCP instructions on to the model can paste them into an `AGENTS.md` or the system prompt:

```text
data-beans stores sparse single-cell matrices, features (genes, peaks) as rows and cells as columns, in .zarr.zip, .zarr or .h5 backends.

Look before you derive:
1. info gives a backend's size and metadata; list_contents shows what an outside .zarr or .h5 holds before you import it.
2. names pages through row or column names, with a regex filter, to find the exact names to pass on.
3. statistics summarizes each row or column (nnz, tot, mu, sig) with quantiles and the top entries by a chosen statistic; values reads the counts of a few named rows or columns.

Quality control: qc runs the cell QC senna and pinto apply (a near-empty floor, then robust MAD bands on detected genes, total counts and the mito fraction). Call it without `out` first: it reports each metric's median, the band and how many cells fall outside, and writes nothing. Adjust n_mads or the caps until the drops are what the data warrant, then call it with `out` to write the kept cells as a new backend. plot shows the same settings as a figure (histograms with the band edges, counts against genes by verdict), so look at it before and after tuning.

Derive new backends with import (mtx, zarr triplets, fragments, h5ad, 10x; say in `source` where the input came from), subset, subsample, split, convert and merge. Each writes to `out`, which must not exist yet: no tool replaces a file. Leave `out` out and the output goes in today's results folder, named after its input and the tool. Outputs are .zarr.zip archives unless `out` ends in .zarr (a directory) or .h5. Paths are resolved against the server's working directory; results give the paths written. On a big store, pass `background` to these (and to qc with `out`): the call returns a job at once, with the paths it will write.

A person may be watching (`data-beans mcp watch`): a "Note from the user" in a result is theirs to you; take it into account. A call they stopped fails with "stopped by the user": do not run it again unasked.
```

After these, [legume-mcp](https://crates.io/crates/legume-mcp) adds its own notes: where the records go (below), and how to follow a background job.

### Tools

| Tool | What it does | Writes |
|---|---|---|
| `info` | a backend's size (rows, columns, non-zeros) and metadata | |
| `list_contents` | the groups and arrays in a `.zarr`, `.zarr.zip` or `.h5`, before importing it | |
| `names` | row or column names, paged, filtered by a case-insensitive regex | |
| `statistics` | per-row or per-column nnz, total, mean and sd: quantiles and the top entries | a `.parquet` when `out` is given |
| `values` | the counts of up to 100 named or indexed rows or columns | a table when `out` is given |
| `qc` | senna's and pinto's cell QC: the near-empty floor, MAD bands on genes, counts and mito fraction, caps | the kept cells when `out` is given; a dry run otherwise |
| `plot` | the QC figure, or a statistic's histogram, returned as an image | the figure |
| `import` | a backend from `.mtx`, zarr triplets, fragments, `.h5ad` or 10x `.h5`, its `source` noted | a backend |
| `subset` | some rows or columns: by `names`, a `pattern` (regex) or `indices` (ranges such as `0-19612`) | a backend |
| `subsample` | cells and/or genes drawn at random | a backend |
| `split` | cells split train/test, or into K folds; by region or group with `options` | the halves |
| `convert` | zarr ↔ h5, `.zarr` ↔ `.zarr.zip` | a backend |
| `merge` | backends side by side, batch-labelled; `common_rows` keeps the rows all share | a backend and its `.batch.gz` |
| `notebook` | the agent's plan, observations and conclusions (from legume-mcp) | the day's `notebook.md` |
| `job_status`, `job_wait`, `job_cancel` | follow or stop a step started with `background` (from legume-mcp) | |

Every tool that writes a backend (`qc` with `out`, `import`, `subset`, `subsample`, `split`, `convert`, `merge`) takes `background`: it returns at once with the `job` and the paths it will write (`written`, and merge's `batches`, qc's `report`). Wait for `job_wait` to say `done` before using them: a `.zarr` or `.h5` is there, half written, while the job runs. What the command printed (qc's medians and bands, say) is the job's `output`. Until the job ends, its outputs are no other call's to write.

No tool replaces a file: an output that is an input, or anything named after the output beside it, is refused, and `squeeze`, which rewrites its input, is not offered.

### Records

The server keeps the project's records the way [legume-mcp](https://crates.io/crates/legume-mcp) lays them out, after Noble's guide to organizing computational biology projects [1]:

- outputs the agent does not name go in `results/<YYYY-MM-DD>/`, named after their input and the tool (`pbmc.imported.qc.subset.zarr.zip`);
- every step that writes goes into the day's `results/<YYYY-MM-DD>/runall.sh`, with paths relative to the project and steps skipped when their outputs exist, so a copy of the project replays the session;
- `results/INDEX.tsv` lists each output with a one-line summary;
- `import` notes where its input came from (the agent's `source`: a URL, an accession, a version) in `data/README.md`;
- the agent keeps `results/<YYYY-MM-DD>/notebook.md` through the `notebook` tool.

`LEGUME_MCP_RECORDS=off` keeps no records.

[1] W. S. Noble, "A quick guide to organizing computational biology projects", PLoS Comput Biol 5(7): e1000424 (2009). doi:[10.1371/journal.pcbi.1000424](https://doi.org/10.1371/journal.pcbi.1000424)

### Watching

```sh
data-beans mcp watch
```

follows a running data-beans server in a detachable session: the agent's calls as they come, each one's log and result, the notebook. `s` stops the selected call, `n` leaves the agent a note (it arrives with its next result, and goes into the notebook), `o` opens a backend the call wrote in the statistics explorer, `F` follows again after you move. `--watch-only` offers no stop or note. With several servers running it asks which; `data-beans mcp watch <folder>` names one. `LEGUME_MCP_WATCH=off` keeps a server from being watched.

### The same on the command line

```sh
data-beans qc data.zarr.zip --qc-mito-pattern '(?i)^MT-'            # summary only
data-beans qc data.zarr.zip --qc-mito-pattern '(?i)^MT-' -o clean   # and the kept cells
data-beans plot qc data.zarr.zip --qc-mito-pattern '(?i)^MT-' -o qc.png
data-beans plot stat data.zarr.zip -s row --stat sum -o genes.png
```
