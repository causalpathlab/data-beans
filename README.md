
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

## Views

One rule decides, for every command with a view:

- `qc`, `stat` and `histogram` open their view on a terminal, and run straight through otherwise;
- `row-names`, `column-names`, `list-h5` and `list-zarr` open a browser on a terminal in place of printing (a file `-o` is still written);
- `merge-backend`, `merge-mtx`, `align-data`, `subsample`, `split` and every `from-*` show a plan of what they write (or wrote) first, and write only on Enter;
- `columns`, `rows`, `subset` and `subset-*` open the explorer to pick from only when the command line selects nothing;
- `--batch-process`, or `DATA_BEANS_BATCH=1` in the environment, asks for no view: put `export DATA_BEANS_BATCH=1` at the top of a script started from a terminal, whose commands would otherwise take it for yours;
- a terminal is stdin and stdout both a terminal, and not `TERM=dumb`. Without one (output redirected, say) a view is skipped with one line on stderr; a picker with nothing to pick from stops and names the flags that select instead.

Results go to stdout and nothing else does: logs, the QC histograms, the skipped-view line and errors go to stderr. The MCP server's commands always run in batch.

**`data-beans qc`** shows each cell QC rule with its edge, how many cells it fails and how many only it removes; the selected rule's histogram in three layers (every cell, the cells passing the other rules, the cells only it removes); total counts against detected genes; and the removed cells by the rules they fail. `←`/`→` move a rule along its MAD band, `m` switches it to a value of its own (`e` types one), `o` turns it off or on, `p` shows the `--qc-*` flags that decide the same cells (`c` copies them to the clipboard), `s` saves the figure as PDF and PNG, Enter writes the kept cells to `-o` (or asks where, offering a free name; it never replaces a file or rewrites the input) and `q` cancels. On leaving it prints the same run as a `--batch-process` command line.

`qc` also does what `squeeze` did: its nnz floors are `--qc-min-cell-nnz` (cells) and `--qc-feature-min-cells` (genes), and `--qc-auto-cutoff` puts both at the trough of their histograms. Several files (`qc a.zarr.zip,b.zarr.zip`) are pooled first, as `merge-backend` merges them (`--common-rows` keeps the genes all share), and QC'd as one; the output comes with their batch labels.

`stat` explores per-row or per-column statistics; `s` saves its figure (the four histograms and mean against sd, marked entries in the accent). `histogram` opens the same explorer on the `--stat` and `--dim` asked.

**Names and trees.** `row-names` and `column-names` list a matrix's names, `list-h5` and `list-zarr` a file's hierarchy (Enter or the arrows open and close a group). `/` filters (a regular expression when the text is one), Space marks, `a` marks every entry shown, `w` writes the marked entries (or, with none marked, those shown; a tree's as paths) and `y` copies them.

**Plans.** Before a command writes, its plan says what it will make: `merge-backend` and `merge-mtx` each input's rows and columns, how many rows every input has and how many only some, and the output's size (`c` keeps the common rows, `s` squeezes after); `align-data` each file's data type and column group, and the rows and columns each keeps (`r` sets the data types); `subsample` the size it draws (`n` cells, `g` genes, `s` seed); `split` each fold's train and test cells (`t` test fraction, `k` folds, `s` seed, `g` grid under `--coord`). After a `from-*` build, the import preview shows its rows' and columns' non-zeros on log bins with where the squeeze cutoffs fall (`s` squeeze, `r` and `c` the cutoffs); `q` there deletes the build. A plan that cannot be written says why, and Enter waits; elsewhere `q` writes nothing.

In kitty, iTerm2, WezTerm and Ghostty the plots are the figure itself, drawn as images; other terminals get text plots. `DATA_BEANS_TUI_IMAGES=0` keeps text, `=1` asks any terminal. `--figure` writes the same figure without the view, and the MCP `qc` and `statistics` tools return it to an agent.

Each view runs as a detachable session: Ctrl+Alt+D leaves it running (as does a dropped connection), and `data-beans attach` returns to it; `data-beans sessions` lists them and `data-beans kill` stops one as its quit key does. `--no-detach` runs a view in the terminal only.

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

Quality control: qc runs the cell QC senna and pinto apply (a near-empty floor, then robust MAD bands on detected genes, total counts and the mito fraction). Call it without `out` first: it reports each metric's median, the band and how many cells fall outside, and writes nothing. Adjust n_mads or the caps until the drops are what the data warrant, then call it with `out` to write the kept cells as a new backend. Pass `figure` to see the same settings as a figure, returned as an image: each rule's histogram with its edge, the cells it alone removes, counts against genes, and the removed cells by the rules they fail; look at it before and after tuning. statistics takes `figure` too: the histograms of nnz, sum, mean and sd, and mean against sd.

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
| `statistics` | per-row or per-column nnz, total, mean and sd: quantiles and the top entries; with `figure`, their histograms and mean against sd as an image | a `.parquet` when `out` is given; the figure |
| `values` | the counts of up to 100 named or indexed rows or columns | a table when `out` is given |
| `qc` | senna's and pinto's cell QC: the near-empty floor, MAD bands on genes, counts and mito fraction, caps, the gene floor; several backends pooled first; with `figure`, the QC figure as an image | the kept cells when `out` is given (and a pool's batch labels); a dry run otherwise; the figure |
| `import` | a backend from `.mtx`, zarr triplets, fragments, `.h5ad` or 10x `.h5`, its `source` noted | a backend |
| `subset` | some rows or columns: by `names`, a `pattern` (regex) or `indices` (ranges such as `0-19612`) | a backend |
| `subsample` | cells and/or genes drawn at random | a backend |
| `split` | cells split train/test, or into K folds; by region or group with `options` | the halves |
| `convert` | zarr ↔ h5, `.zarr` ↔ `.zarr.zip` | a backend |
| `merge` | backends side by side, batch-labelled; `common_rows` keeps the rows all share | a backend and its `.batch.gz` |
| `notebook` | the agent's plan, observations and conclusions (from legume-mcp) | the day's `notebook.md` |
| `job_status`, `job_wait`, `job_cancel` | follow or stop a step started with `background` (from legume-mcp) | |

Every tool that writes a backend (`qc` with `out`, `import`, `subset`, `subsample`, `split`, `convert`, `merge`) takes `background`: it returns at once with the `job` and the paths it will write (`written`, and merge's `batches`, qc's `report`). Wait for `job_wait` to say `done` before using them: a `.zarr` or `.h5` is there, half written, while the job runs. What the command printed (qc's medians and bands, say) is the job's `output`. Until the job ends, its outputs are no other call's to write.

No tool replaces a file: an output that is an input, or anything named after the output beside it, is refused.

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

follows a running data-beans server in a detachable session: the agent's calls as they come, each one's log and result, the notebook. `s` stops the selected call, `n` leaves the agent a note (it arrives with its next result, and goes into the notebook), `o` opens a `qc` call in the QC view, on the data the agent looked at with its settings (try others, `c` copies their flags for a note back; Enter writes only where you say), and a backend another call wrote in the statistics explorer, `F` follows again after you move. `--watch-only` offers no stop or note. With several servers running it asks which; `data-beans mcp watch <folder>` names one. `LEGUME_MCP_WATCH=off` keeps a server from being watched.

### The same on the command line

```sh
data-beans qc data.zarr.zip                            # the view, on a terminal
data-beans qc data.zarr.zip --batch-process            # summary only
data-beans qc data.zarr.zip --batch-process -o clean   # and the kept cells
data-beans qc data.zarr.zip --batch-process --figure qc        # qc.pdf and qc.png
data-beans qc a.zarr.zip,b.zarr.zip --batch-process -o both    # pooled, with both.batch.gz
data-beans stat data.zarr.zip -s row --figure genes.png        # the statistics figure
```

What `squeeze -r R -c C -o out` did, without the MAD bands or the mito metric:

```sh
data-beans qc data.zarr.zip --batch-process -o out \
  --qc-feature-min-cells R --qc-min-cell-nnz C \
  --qc-mad-on-genes=false --qc-mad-on-counts=false --qc-mito-pattern=
```

`qc` starts from `--qc-mito-pattern '(?i)^MT-'`; an empty `--qc-mito-pattern=` turns the mito metric off.
