//! `data-beans mcp`: data-beans' inspection and backend-building commands as
//! Model Context Protocol tools, served over stdio with `legume-mcp`, so an
//! agent can look inside a backend and derive new ones from it.
//!
//! Every tool runs as a child `data-beans` process, as from the shell:
//! stdout carries the protocol, a child's log lines come back as progress,
//! and a cancelled call kills its child. Reads return what the child printed,
//! parsed and cut to a page. Writes take an `out` that must not exist yet:
//! no tool replaces a file.
//!
//! The project's records are legume-mcp's, laid out after Noble (2009), "A
//! quick guide to organizing computational biology projects", PLoS Comput
//! Biol 5(7):e1000424: each writing step goes into the day's replayable
//! `results/<date>/runall.sh` with its outputs (skipped on replay when they
//! exist) and a line in `results/INDEX.tsv`; an output the agent does not
//! name goes in that day's folder; `import` notes its input's source in
//! `data/README.md`.

use crate::handlers::merging::{batch_file, same_file};
use crate::hdf5_io::strip_backend_suffix;
use crate::sparse_io::SparseIoBackend;
use crate::utilities::io_helpers::{index_count, parse_index_spec};
use crate::utilities::name_matching::{match_by_substring, names_matching};
use crate::zarr_io::{output_target, OutputTarget};
use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use legume_mcp::{Call, Content, Handler, ServerInfo, ToolSpec};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

#[derive(Args, Debug)]
pub struct McpArgs {
    #[command(subcommand)]
    pub then: Option<McpCommand>,
}

#[derive(Subcommand, Debug)]
pub enum McpCommand {
    /// Watch a data-beans MCP server while an agent uses it: its calls as
    /// they come, each one's log and result, the notebook; s stops the
    /// selected call, n leaves the agent a note, o opens what a call wrote
    /// in the statistics explorer
    Watch(WatchArgs),
}

#[derive(Args, Debug)]
pub struct WatchArgs {
    /// the server's watch folder (as `ls $XDG_RUNTIME_DIR/legume-mcp`
    /// lists them); without it, the one live server, or a pick among them
    pub server: Option<PathBuf>,

    /// watch only: no stop, no notes
    #[arg(long)]
    pub watch_only: bool,
}

const INSTRUCTIONS: &str = "\
data-beans stores sparse single-cell matrices, features (genes, peaks) as \
rows and cells as columns, in .zarr.zip, .zarr or .h5 backends.

Look before you derive:
1. info gives a backend's size and metadata; list_contents shows what an \
outside .zarr or .h5 holds before you import it.
2. names pages through row or column names, with a regex filter, to find \
the exact names to pass on.
3. statistics summarizes each row or column (nnz, tot, mu, sig) with \
quantiles and the top entries by a chosen statistic; values reads the \
counts of a few named rows or columns.

Quality control: qc runs the cell QC senna and pinto apply (a near-empty \
floor, then robust MAD bands on detected genes, total counts and the mito \
fraction). Call it without `out` first: it reports each metric's median, the \
band and how many cells fall outside, and writes nothing. Adjust n_mads or \
the caps until the drops are what the data warrant, then call it with `out` \
to write the kept cells as a new backend. Pass `figure` to see the same \
settings as a figure, returned as an image: each rule's histogram with its \
edge, the cells it alone removes, counts against genes, and the removed \
cells by the rules they fail; look at it before and after tuning. \
statistics takes `figure` too: the histograms of nnz, sum, mean and sd, and \
mean against sd.

Derive new backends with import (mtx, zarr triplets, fragments, h5ad, 10x; \
say in `source` where the input came from), subset, subsample, split, \
convert and merge. Each writes to `out`, which must not exist yet: no tool \
replaces a file. Leave `out` out and the output goes in today's results \
folder, named after its input and the tool. Outputs are .zarr.zip archives \
unless `out` ends in .zarr (a directory) or .h5. Paths are resolved against \
the server's working directory; results give the paths written. On a big \
store, pass `background` to these (and to qc with `out`): the call returns \
a job at once, with the paths it will write.

A person may be watching (`data-beans mcp watch`): a \"Note from the user\" \
in a result is theirs to you; take it into account. A call they stopped \
fails with \"stopped by the user\": do not run it again unasked.";

/// The most entries a read returns, and the most a `values` call takes.
const MAX_LIMIT: usize = 1000;
const MAX_TAKEN: usize = 100;

fn schema(properties: Value, required: &[&str]) -> Value {
    json!({ "type": "object", "properties": properties, "required": required })
}

/// The `import` sources this build can read.
fn import_commands() -> Vec<&'static str> {
    let mut v = vec!["from-mtx", "from-zarr", "from-fragments"];
    if cfg!(feature = "hdf5") {
        v.extend(["from-h5ad", "from-10x-matrix", "from-10x-molecule"]);
    }
    v
}

/// The cell-QC settings `qc` takes, and `data`.
fn qc_properties(data: &Value) -> Value {
    json!({
        "data": data,
        "mito_pattern": { "type": "string", "description": "Regex over feature names for mitochondrial genes (default (?i)^MT-; empty turns the mito metric off)" },
        "n_mads": { "type": "number", "exclusiveMinimum": 0, "description": "Band half-width in MADs (default 5; smaller drops more)" },
        "min_cell_nnz": { "type": "integer", "minimum": 0, "description": "Near-empty floor on detected features (default 2)" },
        "mad_on_genes": { "type": "boolean", "description": "Band on detected genes (default true)" },
        "mad_on_counts": { "type": "boolean", "description": "Band on total counts (default true)" },
        "mad_on_mito": { "type": "boolean", "description": "Band on the mito fraction (default true); false leaves only mito_max_frac, if given" },
        "mito_max_frac": { "type": "number", "description": "Hard cap on the mito fraction, 0..1" },
        "ribo_pattern": { "type": "string", "description": "Regex for ribosomal genes, e.g. (?i)^RP[SL]" },
        "ribo_max_frac": { "type": "number", "description": "Hard cap on the ribo fraction, 0..1" },
        "feature_min_cells": { "type": "integer", "minimum": 0, "description": "Also drop features detected in fewer cells (default 0, off)" },
        "auto_cutoff": { "type": "boolean", "description": "Also drop cells below the trough between ambient and real cells' detected genes" },
    })
}

fn tools() -> Vec<ToolSpec> {
    let data = json!({ "type": "string", "description": "Backend: .zarr.zip, .zarr or .h5" });
    let axis = json!({ "type": "string", "enum": ["rows", "columns"] });
    let out = json!({
        "type": "string",
        "description": "Output path; must not exist. A .zarr.zip archive unless it ends in .zarr or .h5. Omitted, a new name in today's results folder"
    });
    let options = json!({
        "type": "array", "items": { "type": "string" },
        "description": "Further command-line options, as `data-beans <command> --help` lists them"
    });
    let background = json!({
        "type": "boolean",
        "description": "Run in the background: the job's id comes back at once, and job_status / job_wait follow it (for big stores)"
    });
    let limit = json!({ "type": "integer", "minimum": 1, "maximum": MAX_LIMIT, "description": "Entries to return (default 100)" });
    let figure = json!({ "type": ["boolean", "string"], "description": "Also the figure, returned as an image: true for a PNG in today's results folder, or a new .png, .pdf or .svg to write" });
    vec![
        ToolSpec::new(
            "info",
            "A backend's number of rows, columns and non-zeros, and its metadata.",
            schema(json!({ "data": data }), &["data"]),
        ),
        ToolSpec::new(
            "list_contents",
            "The groups and arrays inside a .zarr, .zarr.zip or .h5 file, to find the fields `import` needs.",
            schema(json!({ "file": { "type": "string" } }), &["file"]),
        ),
        ToolSpec::new(
            "names",
            "A page of a backend's row or column names, those matching `pattern` (a case-insensitive regex) when given. Returns the total, the number matched and the page with each name's index.",
            schema(
                json!({
                    "data": data, "axis": axis,
                    "pattern": { "type": "string" },
                    "offset": { "type": "integer", "minimum": 0 },
                    "limit": limit,
                }),
                &["data", "axis"],
            ),
        ),
        ToolSpec::new(
            "statistics",
            "Per-row or per-column statistics (nnz, tot, mu, sig) over one or more backends: quantiles of each, and the top entries by `sort_by`. With `out` (.parquet), the full table is written there instead. With `figure`, also the histograms of nnz, sum, mean and sd and mean against sd (with the Poisson line), as an image.",
            schema(
                json!({
                    "data": { "type": "array", "items": { "type": "string" }, "minItems": 1 },
                    "axis": axis,
                    "row_pattern": { "type": "string", "description": "Column statistics count only rows matching this case-insensitive regex, e.g. ^MT-" },
                    "sort_by": { "type": "string", "enum": ["nnz", "tot", "mu", "sig"] },
                    "ascending": { "type": "boolean" },
                    "limit": limit,
                    "out": { "type": "string", "description": "A .parquet file for the full table; must not exist" },
                    "figure": figure,
                }),
                &["data", "axis"],
            ),
        ),
        ToolSpec::new(
            "values",
            "The counts of at most 100 named (or indexed) rows or columns, as a dense table: one line per entry of the other axis, all-zero lines left out unless `keep_zeros`, at most `limit` lines (default 100, up to 1000) returned. With `out` (.parquet or .tsv), the full table is written there instead.",
            schema(
                json!({
                    "data": data, "axis": axis,
                    "names": { "type": "array", "items": { "type": "string" }, "description": "Names to take (substring matching)" },
                    "indices": { "type": "string", "description": "0-based indices and inclusive ranges, e.g. 0-9,20" },
                    "keep_zeros": { "type": "boolean" },
                    "limit": limit,
                    "out": { "type": "string" },
                }),
                &["data", "axis"],
            ),
        ),
        ToolSpec::new(
            "qc",
            "Cell QC as senna and pinto apply it: cells below the near-empty floor, or outside `median ± n_mads·MAD` on log detected genes and log total counts (from below) or on the mito fraction (from above), are dropped; genes detected in fewer than `feature_min_cells` cells too. The nnz floors (`min_cell_nnz` on cells, `feature_min_cells` on genes) are what squeeze used to cut. Several backends in `data` are pooled first (merged, all rows or `common_rows`) and QC'd as one; the output then has their batch labels. Returns the counts kept and dropped and, per metric, its median, band and the cells outside it. Without `out` nothing is written; with it, the kept cells go to a new backend. With `figure`, also the QC figure as an image: each rule's histogram (all cells, those passing the other rules, those only it removes) with its edge, counts against genes coloured by mito with removed cells ringed, cells per gene, and the removed cells by the rules they fail.",
            schema(
                {
                    let mut p = qc_properties(&data);
                    p["data"] = json!({ "type": ["string", "array"], "items": { "type": "string" }, "description": "A backend, or several to pool and QC as one" });
                    p["common_rows"] = json!({ "type": "boolean", "description": "Pooling several backends, keep only the rows (genes) all have" });
                    p["report"] = json!({ "type": "string", "description": "A .tsv(.gz) for the per-cell metrics and verdicts; must not exist" });
                    p["out"] = json!({ "type": "string", "description": "Output for the kept cells; must not exist. Omit for a dry run" });
                    p["figure"] = figure.clone();
                    p["background"] = background.clone();
                    p
                },
                &["data"],
            ),
        ),
        ToolSpec::new(
            "import",
            "Build a backend from another format with `data-beans <command> <input>`; from-mtx takes the feature and barcode files as options (-r, -c).",
            schema(
                json!({
                    "command": { "type": "string", "enum": import_commands() },
                    "input": { "type": "string" },
                    "source": { "type": "string", "description": "Where the input came from (a URL or accession, and its version), kept in the project's data/README.md" },
                    "out": out, "options": options, "background": background,
                }),
                &["command", "input"],
            ),
        ),
        ToolSpec::new(
            "subset",
            "A new backend with only some rows or columns, chosen by one of: `names` (exact unless `prefix_match`), `pattern` (a case-insensitive regex over the names, e.g. /count/spliced$), or `indices` (0-based, as a list or as text with ranges, e.g. \"0-19612,20000\").",
            schema(
                json!({
                    "data": data, "axis": axis,
                    "names": { "type": "array", "items": { "type": "string" } },
                    "pattern": { "type": "string" },
                    "indices": { "description": "0-based indices: a list, or text with ranges (0-9,20)", "anyOf": [
                        { "type": "array", "items": { "type": "integer", "minimum": 0 } },
                        { "type": "string" }
                    ] },
                    "prefix_match": { "type": "boolean" },
                    "out": out, "options": options, "background": background,
                }),
                &["data", "axis"],
            ),
        ),
        ToolSpec::new(
            "subsample",
            "A new backend of randomly drawn cells (columns) and/or genes (rows), by count or fraction; an unset axis is kept whole.",
            schema(
                json!({
                    "data": data,
                    "cells": { "type": "integer", "minimum": 1 },
                    "cell_frac": { "type": "number" },
                    "genes": { "type": "integer", "minimum": 1 },
                    "gene_frac": { "type": "number" },
                    "seed": { "type": "integer", "minimum": 0 },
                    "out": out, "background": background,
                }),
                &["data"],
            ),
        ),
        ToolSpec::new(
            "split",
            "Split cells into train/test halves ({out}.train, {out}.test) or K folds ({out}.fold{k}.train/test). On spatial data pass --coord/--grid in `options`; --groups keeps a donor or sample on one side. `out` is a prefix here.",
            schema(
                json!({
                    "data": data,
                    "test_frac": { "type": "number" },
                    "folds": { "type": "integer", "minimum": 2 },
                    "seed": { "type": "integer", "minimum": 0 },
                    "out": { "type": "string", "description": "Output prefix; nothing named {out}.train*, {out}.test* or {out}.fold* may exist. Omitted, a new prefix in today's results folder" },
                    "options": options, "background": background,
                }),
                &["data"],
            ),
        ),
        ToolSpec::new(
            "convert",
            "Copy a backend into another format: zarr <-> h5, .zarr <-> .zarr.zip; the format follows `out`.",
            schema(
                json!({ "data": data, "out": out, "background": background }),
                &["data"],
            ),
        ),
        ToolSpec::new(
            "merge",
            "Merge backends into one: their columns side by side, each tagged with its batch, and the batch labels written beside it (`data-beans merge-backend`). The backends must have the same rows in the same order, unless `common_rows`, which keeps the rows all of them have, in the first one's order.",
            schema(
                json!({
                    "data": { "type": "array", "items": { "type": "string" }, "minItems": 2 },
                    "common_rows": { "type": "boolean" },
                    "out": out, "options": options, "background": background,
                }),
                &["data"],
            ),
        ),
    ]
}

/// What o opens on a call: a `qc` call in the QC view, on the data it
/// looked at with the agent's settings, for the person to try others on
/// (its Enter writes only where they say; `c` copies the flags, for a note
/// back); else a backend the call wrote, in the statistics explorer. On the
/// watcher's terminal either view opens by itself.
fn open(call: &legume_mcp::watch::view::CallView) -> Option<Command> {
    let qc = match call.tool.as_str() {
        "qc" => qc_view_argv(&call.args, &call.text),
        _ => None,
    };
    let words = match qc {
        Some(words) => words,
        None => {
            let written = written_backend(&call.text)?;
            argv(["statistics", &written, "-s", "column"])
        }
    };
    let mut cmd = Command::new(own_exe().ok()?);
    cmd.args(words);
    Some(cmd)
}

/// The `qc` command line of the view on what a `qc` call looked at: the
/// data as its result names them (absolute, so the watcher may be anywhere),
/// the settings as the call gave them, and no output. `None` for a call
/// that did not come to a result.
fn qc_view_argv(args: &Value, text: &str) -> Option<Vec<String>> {
    let v: Value = serde_json::Deserializer::from_str(text)
        .into_iter()
        .next()?
        .ok()?;
    let data: Vec<&str> = match &v["data"] {
        Value::String(s) => vec![s.as_str()],
        Value::Array(a) => a.iter().filter_map(Value::as_str).collect(),
        _ => return None,
    };
    if data.is_empty() {
        return None;
    }
    let a = Arguments(args.clone());
    let mut argv = argv(["qc", &data.join(",")]);
    if data.len() > 1 && a.flag("common_rows").ok()? {
        argv.push("--common-rows".into());
    }
    argv.extend(qc_flags(&a).ok()?);
    Some(argv)
}

/// The backend a result says it wrote (`written`, or the first of a list),
/// when it is one. The feed cuts a result at 4 KB: past that, nothing opens.
fn written_backend(text: &str) -> Option<String> {
    // The first value: a note the person left may follow it.
    let v: Value = serde_json::Deserializer::from_str(text)
        .into_iter()
        .next()?
        .ok()?;
    let w = &v["written"];
    let path = w.as_str().or_else(|| w.get(0)?.as_str())?;
    // A job's output is not there until it is done; a .zarr.zip only
    // appears then, but a .zarr or .h5 is there while it is written.
    let done = v.get("job").is_none() || Path::new(path).exists();
    (strip_backend_suffix(path) != path && done).then(|| path.to_string())
}

pub fn run_mcp(args: &McpArgs) -> Result<()> {
    if let Some(McpCommand::Watch(w)) = &args.then {
        let tool = &crate::interactive::TOOL;
        return legume_mcp::watch::view::run(tool, w.server.clone(), !w.watch_only, open);
    }
    let info = ServerInfo {
        name: "data-beans".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        instructions: Some(INSTRUCTIONS.into()),
    };
    legume_mcp::serve(info, DataBeans)
}

struct DataBeans;

impl Handler for DataBeans {
    fn tools(&self) -> Vec<ToolSpec> {
        tools()
    }

    fn call(&mut self, name: &str, args: Value, cx: &Call) -> Result<Vec<Content>> {
        let a = Arguments(args);
        let v = match name {
            "info" => summary(cx, &a.path("data")?)?,
            "list_contents" => return list_contents(&a, cx),
            "names" => names(&a, cx)?,
            "statistics" => return statistics(&a, cx),
            "values" => values(&a, cx)?,
            "qc" => return qc(&a, cx),
            "import" => import(&a, cx)?,
            "subset" => subset(&a, cx)?,
            "subsample" => {
                let data = a.path("data")?;
                let out = a.backend_out(cx, &named(&data, "subsample"))?;
                let mut argv = argv(["subsample", &data]);
                argv.extend(a.numbers(&["cells", "cell_frac", "genes", "gene_frac", "seed"])?);
                let what = format!("cells and genes drawn at random from {data}");
                written(cx, a.flag("background")?, argv, &out, &what, &[data])?
            }
            "split" => split(&a, cx)?,
            "convert" => {
                let data = a.path("data")?;
                let out = a.backend_out(cx, &named(&data, "converted"))?;
                let what = format!("{data} in another format");
                written(
                    cx,
                    a.flag("background")?,
                    argv(["convert", &data]),
                    &out,
                    &what,
                    &[data],
                )?
            }
            "merge" => merge(&a, cx)?,
            _ => anyhow::bail!("no tool {name}"),
        };
        Ok(vec![Content::Json(v)])
    }
}

/// A tool's arguments.
struct Arguments(Value);

impl Arguments {
    fn str(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(Value::as_str)
    }

    fn req(&self, key: &str) -> Result<String> {
        self.str(key)
            .map(str::to_string)
            .with_context(|| format!("`{key}` is required"))
    }

    /// A path, which goes on the command line where a leading `-` would
    /// read as an option.
    fn path(&self, key: &str) -> Result<String> {
        let p = self.req(key)?;
        anyhow::ensure!(!p.starts_with('-'), "`{key}` {p}: write it as ./{p}");
        // Commands that take several split them at commas.
        anyhow::ensure!(
            !p.contains(','),
            "`{key}` {p}: a path with a comma; rename it"
        );
        Ok(p)
    }

    /// Paths for a command that takes a list of them, which it also splits
    /// at commas.
    fn paths(&self, key: &str) -> Result<Vec<String>> {
        let v = self.strings(key)?;
        if let Some(p) = v.iter().find(|p| p.starts_with('-')) {
            anyhow::bail!("`{key}` {p}: write it as ./{p}");
        }
        if let Some(p) = v.iter().find(|p| p.contains(',')) {
            anyhow::bail!("`{key}` {p}: a path with a comma cannot go in a list; rename it");
        }
        Ok(v)
    }

    /// `key`'s value, when it is given and not null.
    fn given(&self, key: &str) -> Option<&Value> {
        self.0.get(key).filter(|v| !v.is_null())
    }

    fn parse<T: serde::de::DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        self.given(key)
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()
            .with_context(|| format!("reading `{key}`"))
    }

    fn flag(&self, key: &str) -> Result<bool> {
        Ok(self.parse(key)?.unwrap_or(false))
    }

    fn strings(&self, key: &str) -> Result<Vec<String>> {
        Ok(self.parse(key)?.unwrap_or_default())
    }

    /// The numbers given among `keys`, as `--key-name value`.
    fn numbers(&self, keys: &[&str]) -> Result<Vec<String>> {
        let mut v = vec![];
        for k in keys {
            match self.given(k) {
                None => {}
                Some(n @ Value::Number(_)) => {
                    v.extend([format!("--{}", k.replace('_', "-")), n.to_string()])
                }
                Some(other) => anyhow::bail!("`{k}` is a number, not {other}"),
            }
        }
        Ok(v)
    }

    fn rows(&self) -> Result<bool> {
        match self.req("axis")?.as_str() {
            "rows" | "row" => Ok(true),
            "columns" | "column" => Ok(false),
            other => anyhow::bail!("`axis` is rows or columns, not {other}"),
        }
    }

    fn limit(&self) -> Result<usize> {
        Ok(self.parse("limit")?.unwrap_or(100).clamp(1, MAX_LIMIT))
    }

    /// `out`, or, when the agent names none, `{name}{ext}` in today's
    /// results folder, `-2`, `-3`… added to `name` (before `ext`) while
    /// `taken` says a name is in use, or a running job writes it.
    fn out_or(
        &self,
        cx: &Call,
        (name, ext): (&str, &str),
        taken: impl Fn(&str) -> bool,
    ) -> Result<String> {
        if self.given("out").is_some() {
            return self.path("out");
        }
        cx.results_dir()
            .context("`out` is required: this server keeps no results folder")?;
        results_name(cx, (name, ext), taken)
    }

    /// Where `figure` asks a tool's figure to go: the new .png, .pdf or .svg
    /// it names, or, for `true`, `{name}.png` in today's results folder; a
    /// server keeping no results folder shows that one to the agent only.
    fn figure(&self, cx: &Call, name: &str) -> Result<Option<Figure>> {
        match self.given("figure") {
            None | Some(Value::Bool(false)) => Ok(None),
            Some(Value::Bool(true)) => Ok(Some(match cx.results_dir() {
                Some(_) => Figure {
                    path: results_name(cx, (name, ".png"), |c| Path::new(c).exists())?,
                    kept: true,
                    _temp: None,
                },
                None => {
                    let temp = tempfile::Builder::new()
                        .suffix(".png")
                        .tempfile()?
                        .into_temp_path();
                    Figure {
                        path: temp.to_string_lossy().into_owned(),
                        kept: false,
                        _temp: Some(temp),
                    }
                }
            })),
            Some(Value::String(_)) => {
                let path = self.path("figure")?;
                anyhow::ensure!(
                    [".png", ".pdf", ".svg"].iter().any(|e| path.ends_with(e)),
                    "`figure` {path}: a .png, .pdf or .svg"
                );
                fresh(&path)?;
                Ok(Some(Figure {
                    path,
                    kept: true,
                    _temp: None,
                }))
            }
            Some(other) => anyhow::bail!("`figure` is true or a file name, not {other}"),
        }
    }
}

/// A free `{name}{ext}` in today's results folder: `-2`, `-3`… added to
/// `name` (before `ext`) while `taken` says a name is in use, or a running
/// job writes it.
fn results_name(
    cx: &Call,
    (name, ext): (&str, &str),
    taken: impl Fn(&str) -> bool,
) -> Result<String> {
    let dir = cx
        .results_dir()
        .context("this server keeps no results folder")?;
    (1..10_000)
        .map(|k| match k {
            1 => dir.join(format!("{name}{ext}")),
            k => dir.join(format!("{name}-{k}{ext}")),
        })
        .map(|p| p.to_string_lossy().into_owned())
        .find(|p| !held(cx, &[p]) && !taken(p))
        .context("no free name in the results folder")
}

/// A figure a tool writes: where, and whether it is kept (else a temporary
/// file, gone with this).
struct Figure {
    path: String,
    kept: bool,
    _temp: Option<tempfile::TempPath>,
}

impl Figure {
    /// The figure for the agent: the image itself where it is a PNG, and
    /// where it is kept.
    fn contents(&self, v: &mut Map<String, Value>) -> Result<Vec<Content>> {
        if self.kept {
            v.insert("figure".into(), json!(absolute(&self.path)));
        }
        let mut out = vec![];
        if self.path.ends_with(".png") {
            out.push(Content::Png(std::fs::read(&self.path)?));
        }
        Ok(out)
    }
}

impl Arguments {
    /// [`Self::out_or`] for a backend: a name is taken when what the command
    /// would write for it exists.
    fn backend_out(&self, cx: &Call, name: &str) -> Result<String> {
        self.out_or(cx, (name, ""), |c| {
            backend_target(c).map_or(true, |t| {
                let stem = strip_backend_suffix(&t.effective);
                held(cx, &[&t.effective, &t.found])
                    || being_written(stem).is_some()
                    || outputs_beside(stem).map_or(true, |v| !v.is_empty())
            })
        })
    }

    /// `options`, refusing those that would name the output, change its
    /// format or open a view. Short options stand alone (`-r X`, not
    /// `-rX`), so none hides inside a cluster.
    fn options(&self) -> Result<Vec<String>> {
        let v = self.strings("options")?;
        const TAKEN: &[&str] = &[
            "-o",
            "--output",
            "--output-directory",
            "--qc-report",
            "--save-histogram",
            "--backend",
            "--no-zip",
            "--",
        ];
        for o in &v {
            let name = o.split('=').next().unwrap_or(o);
            let cluster = o.starts_with('-')
                && !o.starts_with("--")
                && o.len() > 2
                && o.chars().nth(1).is_some_and(|c| c.is_ascii_alphabetic());
            anyhow::ensure!(
                !cluster,
                "{o}: give a short option apart from its value (`-r X`)"
            );
            anyhow::ensure!(
                !TAKEN.contains(&name),
                "{o} is not taken in `options`: outputs go to `out`, their format follows it, and no view opens here"
            );
        }
        Ok(v)
    }
}

fn argv<const N: usize>(words: [&str; N]) -> Vec<String> {
    words.map(String::from).into()
}

/// `data-beans` itself, with `argv`. A binary replaced under the running
/// server (`cargo install`) is run as the process's own image.
fn command(argv: &[String]) -> Result<Command> {
    let mut cmd = Command::new(own_exe()?);
    // `-v` so its log lines come back as progress.
    cmd.arg("-v").args(argv);
    // No backtrace after an error: a failure quotes the last lines of the
    // child's stderr, and one would push the error itself out of them.
    cmd.env("RUST_BACKTRACE", "0")
        .env("RUST_LIB_BACKTRACE", "0");
    // Never a view: an agent has no terminal to work one.
    cmd.env(crate::interactive::BATCH_ENV, "1");
    Ok(cmd)
}

/// The data-beans binary this is: the process's own image when the file was
/// replaced under it (`cargo install`).
fn own_exe() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("finding the data-beans binary")?;
    let own = Path::new("/proc/self/exe");
    Ok(if !exe.exists() && own.exists() {
        own.to_path_buf()
    } else {
        exe
    })
}

/// `data-beans` run with `argv`, a step that writes nothing, to its end;
/// its stdout. The records keep the call, not the command.
fn data_beans(cx: &Call, argv: &[String]) -> Result<String> {
    cx.run_reading(command(argv)?)
}

/// [`data_beans`] for a step that writes `outputs`: the project's records
/// list each with `what` (what was made, from what), and a replay skips the
/// step when they all exist.
fn data_beans_making(
    cx: &Call,
    argv: &[String],
    outputs: &[impl AsRef<Path>],
    what: &str,
) -> Result<String> {
    let outputs: Vec<PathBuf> = outputs.iter().map(|p| p.as_ref().to_path_buf()).collect();
    cx.run_making(command(argv)?, &outputs, what)
}

/// [`data_beans_making`] as a background job: its id comes back at once,
/// with `written`, where its output will be once it is done.
fn data_beans_job(
    cx: &Call,
    argv: &[String],
    outputs: &[impl AsRef<Path>],
    what: &str,
    written: Value,
) -> Result<Map<String, Value>> {
    let outputs: Vec<PathBuf> = outputs.iter().map(|p| p.as_ref().to_path_buf()).collect();
    let job = cx.spawn_job(command(argv)?, &outputs, what)?;
    let mut v = Map::new();
    v.insert("note".into(), json!(format!("{job} writes `written`")));
    v.insert("job".into(), json!(job));
    v.insert("written".into(), written);
    Ok(v)
}

/// `{stem}.{tool}`: `data`'s name without its folder and backend suffix,
/// then the tool, so a chain of unnamed outputs spells its lineage
/// (`pbmc.imported.qc.subset`). The name an output takes when the agent
/// gives none.
fn named(data: &str, tool: &str) -> String {
    let stem = Path::new(strip_backend_suffix(data))
        .file_name()
        .map_or_else(|| data.to_string(), |n| n.to_string_lossy().into_owned());
    format!("{stem}.{tool}")
}

/// Where a backend-writing command given `-o out` writes, as it resolves
/// it under its default `--backend zarr`; zipped unless `out` names a
/// `.zarr` directory.
fn backend_target(out: &str) -> Result<OutputTarget> {
    output_target(out, SparseIoBackend::Zarr, !out.ends_with(".zarr"))
}

fn absolute(p: &str) -> String {
    std::path::absolute(p)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| p.to_string())
}

/// A file a tool is about to write: refused if it exists.
fn fresh(path: &str) -> Result<()> {
    anyhow::ensure!(
        !Path::new(path).exists(),
        "{path} exists; not replacing it: choose another `out`"
    );
    Ok(())
}

/// The stems being written (all `{stem}.*` is theirs): by a call still
/// running, or by a background job for as long as the server runs. Until
/// a command has made its outputs, only this says they are spoken for.
static WRITING: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Whether a running job writes one of `paths`, however spelled.
fn held(cx: &Call, paths: &[&str]) -> bool {
    let paths: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();
    cx.ensure_free(&paths).is_err()
}

/// Whether two stems name some file alike: one is the other, or a file
/// named after it.
fn overlap(a: &str, b: &str) -> bool {
    let after = |a: &str, b: &str| a.strip_prefix(b).is_some_and(|r| r.starts_with('.'));
    a == b || after(a, b) || after(b, a)
}

/// The stem being written that `stem` would touch, if any.
fn being_written(stem: &str) -> Option<String> {
    let stem = absolute(stem);
    let writing = WRITING.lock().unwrap_or_else(|e| e.into_inner());
    writing.iter().find(|w| overlap(w, &stem)).cloned()
}

/// `stems` spoken for until the claim drops, or, [`Claim::keep`]ed, for
/// good; refused when another call or job has one.
struct Claim(Vec<String>);

impl Claim {
    fn new(stems: &[&str]) -> Result<Self> {
        let stems: Vec<String> = stems.iter().map(|s| absolute(s)).collect();
        let mut writing = WRITING.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(w) = writing.iter().find(|w| stems.iter().any(|s| overlap(w, s))) {
            anyhow::bail!(
                "{w} is being written by another call or a background job: choose another `out`"
            );
        }
        writing.extend(stems.iter().cloned());
        Ok(Self(stems))
    }

    /// For a background job: the server cannot tell when it is done.
    fn keep(self) {
        std::mem::forget(self);
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        let mut writing = WRITING.lock().unwrap_or_else(|e| e.into_inner());
        writing.retain(|w| !self.0.contains(w));
    }
}

/// Run `argv`, a command that writes the backend `out` (made of `what`)
/// from `inputs`, with `-o`; say where the backend is and how big.
fn written(
    cx: &Call,
    background: bool,
    argv: Vec<String>,
    out: &str,
    what: &str,
    inputs: &[String],
) -> Result<Value> {
    Ok(Value::Object(
        written_with(cx, background, argv, out, what, inputs, &[])?.0,
    ))
}

/// [`written`], with `also` written beside it; and what the command
/// printed. The output must not be an input, and nothing else named after
/// it (`{stem}.*`) may exist beside it: data-beans clears the backend and
/// its working copy before writing, and writes its side files (batch
/// labels, an importer's metadata) over whatever is there. The side files
/// it wrote come back as `also`.
fn written_with(
    cx: &Call,
    background: bool,
    mut argv: Vec<String>,
    out: &str,
    what: &str,
    inputs: &[String],
    also: &[&str],
) -> Result<(Map<String, Value>, String)> {
    let target = backend_target(out)?;
    let is_input = |p: &str| inputs.iter().any(|i| same_file(i, p));
    for p in [&target.effective, &target.working, &target.found] {
        anyhow::ensure!(!is_input(p), "{p} is an input: choose another `out`");
    }
    let stem = strip_backend_suffix(&target.effective).to_string();
    let before: Vec<String> = outputs_beside(&stem)?
        .into_iter()
        .filter(|p| !is_input(p))
        .collect();
    anyhow::ensure!(
        before.is_empty(),
        "{} exist beside {out}; not replacing them: choose another `out`",
        before.join(", ")
    );
    // The full name, so the command cannot resolve it otherwise (`subset`
    // keeps an extensionless output in its input's format).
    argv.extend(["-o".into(), target.effective.to_string()]);
    // A `.zarr` directory asked for, as `backend_target` resolved it.
    if out.ends_with(".zarr") {
        argv.push("--no-zip".into());
    }
    let path: &str = &target.found;
    let mut outputs = vec![path];
    outputs.extend(also);
    let claim = Claim::new(&[&[stem.as_str()], also].concat())?;
    if background {
        let job = data_beans_job(cx, &argv, &outputs, what, json!(absolute(path)))?;
        claim.keep();
        return Ok((job, String::new()));
    }
    let printed = data_beans_making(cx, &argv, &outputs, what)?;
    let written = absolute(path);
    // Nothing was there before, so the rest there now was written too.
    let side: Vec<String> = outputs_beside(&stem)?
        .into_iter()
        .filter(|p| *p != written && !is_input(p))
        .collect();
    let mut v = Map::new();
    v.insert("written".into(), json!(written));
    if !side.is_empty() {
        v.insert("also".into(), json!(side));
    }
    v.insert("info".into(), summary(cx, path)?);
    Ok((v, printed))
}

/// Run `argv`, a command that writes the table `out` (made of `what`).
fn to_file(cx: &Call, mut argv: Vec<String>, out: &str, what: &str) -> Result<Value> {
    fresh(out)?;
    argv.extend(["-o".into(), out.into()]);
    data_beans_making(cx, &argv, &[out], what)?;
    Ok(json!({ "written": absolute(out) }))
}

fn summary(cx: &Call, data: &str) -> Result<Value> {
    let text = data_beans(cx, &argv(["info", data]))?;
    Ok(Value::Object(key_values(&text.replace("number_of_", ""))))
}

/// `key:\tvalue` lines as an object, numbers as numbers; a dotted key
/// (`meta.sample`, `n_genes.median`) nests.
fn key_values(text: &str) -> Map<String, Value> {
    let mut v = Map::new();
    for line in text.lines() {
        let Some((key, value)) = line.split_once(":\t") else {
            continue;
        };
        let value = match (value.parse::<u64>(), value.parse::<f64>()) {
            (Ok(n), _) => json!(n),
            (_, Ok(x)) if x.is_finite() => json!(x),
            _ => json!(value),
        };
        let mut at = &mut v;
        let mut parts = key.split('.').peekable();
        while let Some(part) = parts.next() {
            if parts.peek().is_none() {
                at.insert(part.into(), value);
                break;
            }
            let next = at.entry(part).or_insert_with(|| json!({}));
            if !next.is_object() {
                *next = json!({});
            }
            at = next.as_object_mut().unwrap();
        }
    }
    v
}

fn list_contents(a: &Arguments, cx: &Call) -> Result<Vec<Content>> {
    let file = a.path("file")?;
    let command = if file.ends_with(".h5") || file.ends_with(".h5ad") || file.ends_with(".hdf5") {
        "list-h5"
    } else {
        "list-zarr"
    };
    Ok(vec![Content::Text(data_beans(
        cx,
        &argv([command, &file]),
    )?)])
}

/// The command listing a backend's row or column names.
fn names_command(rows: bool) -> &'static str {
    if rows {
        "row-names"
    } else {
        "column-names"
    }
}

fn names(a: &Arguments, cx: &Call) -> Result<Value> {
    let text = data_beans(cx, &argv([names_command(a.rows()?), &a.path("data")?]))?;
    let lines: Vec<&str> = text.lines().collect();
    // Which names the pattern finds, as `subset --pattern` would.
    let found: Option<std::collections::HashSet<usize>> = match a.str("pattern") {
        Some(p) => {
            let names: Vec<Box<str>> = lines.iter().map(|&n| n.into()).collect();
            Some(names_matching(&names, p)?.into_iter().collect())
        }
        None => None,
    };
    let offset: usize = a.parse("offset")?.unwrap_or(0);
    let limit = a.limit()?;
    let (mut total, mut matched, mut page) = (0, 0, vec![]);
    for (i, n) in lines.iter().enumerate() {
        total += 1;
        if found.as_ref().is_none_or(|f| f.contains(&i)) {
            if (offset..offset.saturating_add(limit)).contains(&matched) {
                page.push(json!({ "index": i, "name": n }));
            }
            matched += 1;
        }
    }
    Ok(json!({ "total": total, "matched": matched, "offset": offset, "names": page }))
}

/// One line of a tab-separated table as an object: its first field, a
/// name, as text; the rest numbers where they parse.
fn record<'a>(header: &[&str], line: impl Iterator<Item = &'a str>) -> Value {
    let m: Map<String, Value> = header
        .iter()
        .zip(line)
        .enumerate()
        .map(|(j, (h, x))| {
            let v = match x.parse::<f64>() {
                Ok(n) if j > 0 && n.is_finite() => json!(n),
                _ => json!(x),
            };
            (h.to_string(), v)
        })
        .collect();
    Value::Object(m)
}

/// A table's header; `statistics` heads its table `#name`.
fn header(line: Option<&str>) -> Vec<&str> {
    line.unwrap_or_default()
        .trim_start_matches('#')
        .split('\t')
        .collect()
}

fn statistics(a: &Arguments, cx: &Call) -> Result<Vec<Content>> {
    let data = a.paths("data")?;
    anyhow::ensure!(!data.is_empty(), "`data` is required");
    let dim = if a.rows()? { "row" } else { "column" };
    let mut argv = argv(["statistics", &data.join(","), "-s", dim, "--batch-process"]);
    if let Some(p) = a.str("row_pattern") {
        argv.push(format!("--row-name-pattern={p}"));
    }
    let figure = a.figure(cx, &named(&data[0], &format!("{dim}_statistics")))?;
    if let Some(f) = &figure {
        argv.push(format!("--figure={}", f.path));
    }
    let what = format!("{dim} statistics of {}", data.join(", "));
    if let Some(out) = a.str("out") {
        anyhow::ensure!(out.ends_with(".parquet"), "`out` is a .parquet file");
        let Value::Object(mut v) = to_file(cx, argv, out, &what)? else {
            unreachable!("to_file answers with an object");
        };
        let images = figure
            .as_ref()
            .map(|f| f.contents(&mut v))
            .transpose()?
            .unwrap_or_default();
        return Ok(std::iter::once(Content::Json(Value::Object(v)))
            .chain(images)
            .collect());
    }
    let text = match figure.as_ref().filter(|f| f.kept) {
        Some(f) => data_beans_making(cx, &argv, &[&f.path], &format!("the figure of the {what}"))?,
        None => data_beans(cx, &argv)?,
    };
    let mut lines = text.lines();
    let header = header(lines.next());
    let stats = header.get(1..).unwrap_or_default();
    let sort_by = a.str("sort_by").unwrap_or("tot");
    let col = stats
        .iter()
        .position(|h| *h == sort_by)
        .with_context(|| format!("no `{sort_by}` among {stats:?}"))?;
    // Each line once: itself, and its statistics.
    let mut rows: Vec<(&str, Vec<f64>)> = lines
        .map(|l| {
            let v = l.split('\t').skip(1);
            (l, v.map(|x| x.parse().unwrap_or(f64::NAN)).collect())
        })
        .collect();
    let mut quantiles = Map::new();
    for (j, h) in stats.iter().enumerate() {
        let mut xs: Vec<f64> = rows
            .iter()
            .filter_map(|(_, v)| v.get(j).copied())
            .filter(|x| x.is_finite())
            .collect();
        if xs.is_empty() {
            continue;
        }
        xs.sort_by(f64::total_cmp);
        let q = |p: f64| xs[((xs.len() - 1) as f64 * p).round() as usize];
        quantiles.insert(
            h.to_string(),
            json!({ "min": q(0.0), "q25": q(0.25), "median": q(0.5), "q75": q(0.75), "max": q(1.0) }),
        );
    }
    let count = rows.len();
    let ascending = a.flag("ascending")?;
    let key = |r: &(&str, Vec<f64>)| r.1.get(col).copied().unwrap_or(f64::NAN);
    // A statistic that is not a number goes last either way.
    let order = |x: &(&str, Vec<f64>), y: &(&str, Vec<f64>)| {
        let (kx, ky) = (key(x), key(y));
        let by = if ascending {
            kx.total_cmp(&ky)
        } else {
            ky.total_cmp(&kx)
        };
        kx.is_nan().cmp(&ky.is_nan()).then(by)
    };
    // The top `limit` picked, then only those sorted.
    let k = a.limit()?.min(count);
    if k > 0 && k < count {
        rows.select_nth_unstable_by(k - 1, order);
        rows.truncate(k);
    }
    rows.sort_by(order);
    let top: Vec<Value> = rows
        .iter()
        .map(|(l, _)| record(&header, l.split('\t')))
        .collect();
    let mut v = Map::new();
    v.insert("count".into(), json!(count));
    v.insert("quantiles".into(), Value::Object(quantiles));
    v.insert("sort_by".into(), json!(sort_by));
    v.insert("top".into(), json!(top));
    let images = figure
        .as_ref()
        .map(|f| f.contents(&mut v))
        .transpose()?
        .unwrap_or_default();
    Ok(std::iter::once(Content::Json(Value::Object(v)))
        .chain(images)
        .collect())
}

fn values(a: &Arguments, cx: &Call) -> Result<Value> {
    let rows = a.rows()?;
    let data = a.path("data")?;
    // Resolved to indices first, so the cap holds before anything is read.
    let names = a.strings("names")?;
    let taken: Vec<usize> = match (names.is_empty(), a.str("indices")) {
        (false, None) => {
            let all: Vec<Box<str>> = data_beans(cx, &argv([names_command(rows), &data]))?
                .lines()
                .map(Box::from)
                .collect();
            let queries: Vec<Box<str>> = names.iter().map(|n| n.as_str().into()).collect();
            match_by_substring(&all, &queries, if rows { "row" } else { "column" })?.0
        }
        (true, Some(spec)) => {
            // Counted before it is expanded: a range can be any size.
            let count = index_count(spec)?;
            anyhow::ensure!(
                count <= MAX_TAKEN,
                "{count} entries taken; at most {MAX_TAKEN}"
            );
            parse_index_spec(spec)?
        }
        _ => anyhow::bail!("give `names` or `indices` (as text, e.g. 0-9,20), one of them"),
    };
    anyhow::ensure!(
        taken.len() <= MAX_TAKEN,
        "{} entries taken; at most {MAX_TAKEN}: name them more exactly",
        taken.len()
    );
    let spec: Vec<String> = taken.iter().map(usize::to_string).collect();
    let command = if rows { "rows" } else { "columns" };
    let argv = argv([command, &data, "-i", &spec.join(",")]);
    if let Some(out) = a.str("out") {
        let what = format!("the counts of {} {command} of {data}", taken.len());
        return to_file(cx, argv, out, &what);
    }
    let text = data_beans(cx, &argv)?;
    let mut lines = text.lines();
    let header = header(lines.next());
    let keep_zeros = a.flag("keep_zeros")?;
    let limit = a.limit()?;
    let nonzero = |x: &str| x != "0" && x.parse::<f64>().is_ok_and(|v| v != 0.0);
    let (mut total, mut kept, mut page) = (0, 0, vec![]);
    for l in lines {
        total += 1;
        if keep_zeros || l.split('\t').skip(1).any(nonzero) {
            if page.len() < limit {
                page.push(record(&header, l.split('\t')));
            }
            kept += 1;
        }
    }
    Ok(json!({
        "taken": header.get(1..).unwrap_or_default(),
        "lines": total,
        "kept": kept,
        "values": page,
    }))
}

/// The `--qc-*` flags for the cell-QC settings among `a`.
fn qc_flags(a: &Arguments) -> Result<Vec<String>> {
    let mut v = vec![];
    // The command starts from (?i)^MT-; an empty pattern turns it off.
    if let Some(mito) = a.str("mito_pattern") {
        v.push(format!("--qc-mito-pattern={mito}"));
    }
    // `--flag=value`, so a value cannot read as an option.
    if let Some(ribo) = a.str("ribo_pattern") {
        v.push(format!("--qc-ribo-pattern={ribo}"));
    }
    let numbers = [
        ("n_mads", "--qc-mads"),
        ("min_cell_nnz", "--qc-min-cell-nnz"),
        ("mito_max_frac", "--qc-mito-max-frac"),
        ("ribo_max_frac", "--qc-ribo-max-frac"),
        ("feature_min_cells", "--qc-feature-min-cells"),
    ];
    for (key, flag) in numbers {
        match a.given(key) {
            None => {}
            Some(n @ Value::Number(_)) => v.push(format!("{flag}={n}")),
            Some(other) => anyhow::bail!("`{key}` is a number, not {other}"),
        }
    }
    for (key, flag) in [
        ("mad_on_genes", "--qc-mad-on-genes"),
        ("mad_on_counts", "--qc-mad-on-counts"),
        ("mad_on_mito", "--qc-mad-on-mito"),
    ] {
        if let Some(on) = a.parse::<bool>(key)? {
            v.push(format!("{flag}={on}"));
        }
    }
    if a.flag("auto_cutoff")? {
        v.push("--qc-auto-cutoff".into());
    }
    Ok(v)
}

fn qc(a: &Arguments, cx: &Call) -> Result<Vec<Content>> {
    // One backend, or several to pool.
    let inputs = match a.given("data") {
        Some(Value::Array(_)) => a.paths("data")?,
        _ => vec![a.path("data")?],
    };
    anyhow::ensure!(!inputs.is_empty(), "`data` is required");
    let data = inputs.join(",");
    let mut argv = argv(["qc", &data, "--batch-process"]);
    if inputs.len() > 1 && a.flag("common_rows")? {
        argv.push("--common-rows".into());
    }
    argv.extend(qc_flags(a)?);
    let report = a.str("report");
    if let Some(report) = report {
        fresh(report)?;
        argv.push(format!("--qc-report={report}"));
    }
    let background = a.flag("background")?;
    anyhow::ensure!(
        a.given("out").is_some() || !background,
        "`background` runs a qc that writes `out`; a dry run answers in the call"
    );
    let figure = a.figure(cx, &named(&inputs[0], "qc"))?;
    anyhow::ensure!(
        figure.is_none() || !background,
        "`figure` comes back with the call: leave out `background`, or ask for the figure on a dry run first"
    );
    if let Some(f) = &figure {
        argv.push(format!("--figure={}", f.path));
    }
    // Besides `out`: the report and a kept figure.
    let mut also: Vec<&str> = report.into_iter().collect();
    if let Some(f) = figure.as_ref().filter(|f| f.kept) {
        also.push(&f.path);
    }
    let what = format!("the cells of {data} that cell QC keeps");
    // Without `out` a dry run: only the summary, and the report and figure
    // if asked.
    let (mut v, printed) = match a.given("out") {
        Some(_) => {
            let out = a.path("out")?;
            // A pool's output comes with its batch labels.
            let batches = batch_file(&backend_target(&out)?.effective);
            if inputs.len() > 1 {
                also.push(&batches);
            }
            written_with(cx, background, argv, &out, &what, &inputs, &also)?
        }
        None if also.is_empty() => (Map::new(), data_beans(cx, &argv)?),
        None => {
            let what = format!("cell QC of {data}: its report or figure");
            (Map::new(), data_beans_making(cx, &argv, &also, &what)?)
        }
    };
    v.extend(key_values(&printed));
    // What it looked at, absolute: for the watcher's `o`, wherever it runs.
    let looked: Vec<String> = inputs.iter().map(|p| absolute(p)).collect();
    v.insert(
        "data".into(),
        match looked.len() {
            1 => json!(looked[0]),
            _ => json!(looked),
        },
    );
    if let Some(report) = report {
        v.insert("report".into(), json!(absolute(report)));
    }
    let images = figure
        .as_ref()
        .map(|f| f.contents(&mut v))
        .transpose()?
        .unwrap_or_default();
    Ok(std::iter::once(Content::Json(Value::Object(v)))
        .chain(images)
        .collect())
}

fn import(a: &Arguments, cx: &Call) -> Result<Value> {
    let command = a.req("command")?;
    anyhow::ensure!(
        import_commands().contains(&command.as_str()),
        "`command` is one of {:?}",
        import_commands()
    );
    let input = a.path("input")?;
    // Its provenance is noted before a background job has read it.
    anyhow::ensure!(Path::new(&input).exists(), "{input} does not exist");
    // The input's own suffixes (`.mtx.gz`, `.h5ad`) go.
    let file = Path::new(&input).file_name().map(|n| n.to_string_lossy());
    let stem = file
        .as_deref()
        .and_then(|n| n.split('.').next())
        .unwrap_or("data");
    let out = a.backend_out(cx, &format!("{stem}.imported"))?;
    let what = format!("{input} as a backend ({command})");
    let mut argv = argv([&command, &input]);
    argv.extend(a.options()?);
    let v = written(
        cx,
        a.flag("background")?,
        argv,
        &out,
        &what,
        std::slice::from_ref(&input),
    )?;
    // Where the input came from, in the project's data/README.md, once it
    // is in (or, in the background, once its job has started).
    let source = a.str("source").unwrap_or("a local file, source not given");
    cx.provenance(Path::new(&input), source);
    Ok(v)
}

/// `indices` as the CLI takes them: text (`0-9,20`) as given, a list with
/// its ascending runs folded into ranges.
fn index_spec(v: &Value) -> Result<String> {
    if let Value::String(spec) = v {
        return Ok(spec.clone());
    }
    let ix: Vec<usize> = serde_json::from_value(v.clone()).context("reading `indices`")?;
    // Runs folded as they come: the order asked for is the order written.
    let mut runs: Vec<(usize, usize)> = vec![];
    for i in ix {
        match runs.last_mut() {
            Some((_, end)) if *end + 1 == i => *end = i,
            _ => runs.push((i, i)),
        }
    }
    let spec: Vec<String> = runs
        .iter()
        .map(|&(a, b)| {
            if a == b {
                a.to_string()
            } else {
                format!("{a}-{b}")
            }
        })
        .collect();
    Ok(spec.join(","))
}

fn subset(a: &Arguments, cx: &Call) -> Result<Value> {
    let rows = a.rows()?;
    let (command, side) = if rows {
        ("subset-rows", "rows")
    } else {
        ("subset-columns", "columns")
    };
    let data = a.path("data")?;
    let out = a.backend_out(cx, &named(&data, "subset"))?;
    let mut argv = argv([command, &data]);
    // An empty list of names is no choice.
    let names = a.strings("names")?;
    anyhow::ensure!(
        a.given("pattern").is_none_or(Value::is_string),
        "`pattern` is a regex, as text"
    );
    let given = [
        !names.is_empty(),
        a.given("pattern").is_some(),
        a.given("indices").is_some(),
    ];
    anyhow::ensure!(
        given.iter().filter(|&&g| g).count() == 1,
        "give one of `names`, `pattern` or `indices`"
    );
    let background = a.flag("background")?;
    // The names go to the command in a file, kept until it has run; a
    // job's is left in the temp folder, as the job reads it later.
    let file;
    let what = if let Some(p) = a.str("pattern") {
        // `--flag=value`, so a pattern cannot read as an option.
        argv.push(format!("--pattern={p}"));
        format!("the {side} of {data} matching {p}")
    } else if let Some(v) = a.given("indices") {
        let spec = index_spec(v)?;
        argv.push(format!("--{}-indices={spec}", &side[..side.len() - 1]));
        format!("{} {side} of {data}", index_count(&spec)?)
    } else {
        file = tempfile::NamedTempFile::new()?;
        // Through the call, so the session script carries the names too.
        cx.write_file(file.path(), &(names.join("\n") + "\n"))?;
        argv.extend(["-f".into(), file.path().to_string_lossy().into_owned()]);
        if background {
            file.keep()?;
        }
        if !a.flag("prefix_match")? {
            argv.push("--exact-names".into());
        }
        format!("{} named {side} of {data}", names.len())
    };
    argv.extend(a.options()?);
    written(cx, background, argv, &out, &what, &[data])
}

fn merge(a: &Arguments, cx: &Call) -> Result<Value> {
    let data = a.paths("data")?;
    anyhow::ensure!(data.len() >= 2, "`data` takes two or more backends");
    let out = a.backend_out(cx, &named(&data[0], "merged"))?;
    // merge-backend's batch labels, beside the backend.
    let batches = batch_file(&backend_target(&out)?.effective);
    let what = format!("{} merged", data.join(" + "));
    let mut argv = argv(["merge-backend"]);
    argv.extend(data.iter().cloned());
    if a.flag("common_rows")? {
        argv.push("--common-rows".into());
    }
    argv.extend(a.options()?);
    let mut v = written_with(
        cx,
        a.flag("background")?,
        argv,
        &out,
        &what,
        &data,
        &[&batches],
    )?
    .0;
    v.insert("batches".into(), json!(absolute(&batches)));
    Ok(Value::Object(v))
}

/// The entries beside `stem` named after it (`{stem}.*`): what a command
/// writing under that name leaves there (a backend and its working copy,
/// batch labels, an importer's metadata, split's halves).
fn outputs_beside(stem: &str) -> Result<Vec<String>> {
    let name = Path::new(stem)
        .file_name()
        .context("`out` names no file")?
        .to_string_lossy()
        .into_owned();
    let parent = Path::new(stem)
        .parent()
        .filter(|d| !d.as_os_str().is_empty());
    let dir = std::path::absolute(parent.unwrap_or(Path::new(".")))?;
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
        Err(e) => return Err(e.into()),
    };
    let mut v: Vec<String> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| {
            n.strip_prefix(name.as_str())
                .is_some_and(|r| r.starts_with('.'))
        })
        .map(|n| dir.join(n).to_string_lossy().into_owned())
        .collect();
    v.sort();
    Ok(v)
}

fn split(a: &Arguments, cx: &Call) -> Result<Value> {
    let data = a.path("data")?;
    let prefix = a.out_or(cx, (&named(&data, "split"), ""), |c| {
        being_written(c).is_some() || outputs_beside(c).map_or(true, |v| !v.is_empty())
    })?;
    let before = outputs_beside(&prefix)?;
    anyhow::ensure!(
        before.is_empty(),
        "{} exist beside {prefix}; not replacing them: choose another `out`",
        before.join(", ")
    );
    let folds: Option<usize> = a.parse("folds")?;
    // What split will write: two halves, or two per fold, all zipped.
    let halves: Vec<String> = match folds {
        Some(k) => (0..k)
            .flat_map(|f| ["train", "test"].map(|h| format!("{prefix}.fold{f}.{h}.zarr.zip")))
            .collect(),
        None => ["train", "test"]
            .map(|h| format!("{prefix}.{h}.zarr.zip"))
            .into(),
    };
    let mut argv = argv(["split", &data, "-o", &prefix]);
    argv.extend(a.numbers(&["test_frac", "folds", "seed"])?);
    argv.extend(a.options()?);
    let what = format!("cells of {data} split for testing");
    let claim = Claim::new(&[&prefix])?;
    if a.flag("background")? {
        let written: Vec<String> = halves.iter().map(|h| absolute(h)).collect();
        let job = data_beans_job(cx, &argv, &halves, &what, json!(written))?;
        claim.keep();
        return Ok(Value::Object(job));
    }
    data_beans_making(cx, &argv, &halves, &what)?;
    // Nothing was there before, so all there is now was written.
    Ok(json!({ "written": outputs_beside(&prefix)? }))
}

#[cfg(test)]
mod tests {
    use super::{qc_view_argv, written_backend};
    use serde_json::json;

    #[test]
    fn o_on_a_qc_call_opens_its_view_with_the_agent_settings() {
        let args =
            json!({ "data": "pbmc.zarr.zip", "n_mads": 3, "mito_pattern": "^mt-", "out": "kept" });
        let text = r#"{ "cells": 10, "data": "/p/pbmc.zarr.zip" }"#;
        assert_eq!(
            qc_view_argv(&args, text).unwrap(),
            [
                "qc",
                "/p/pbmc.zarr.zip",
                "--qc-mito-pattern=^mt-",
                "--qc-mads=3"
            ]
        );
        // A pool, its rows as the call aligned them; never the call's out.
        let args = json!({ "data": ["a", "b"], "common_rows": true });
        let text = r#"{ "data": ["/p/a", "/p/b"] }"#;
        assert_eq!(
            qc_view_argv(&args, text).unwrap(),
            ["qc", "/p/a,/p/b", "--common-rows"]
        );
        // A call that failed has no result to open.
        assert_eq!(
            qc_view_argv(&json!({ "data": "x" }), "Error: no such file"),
            None
        );
    }

    /// The README gives the instructions for clients that do not pass them
    /// on: they must be the server's, word for word.
    #[test]
    fn the_readme_says_what_the_server_tells_the_agent() {
        let readme = include_str!("../README.md");
        assert!(
            readme.contains(super::INSTRUCTIONS),
            "README.md's \"What the agent is told\" block differs from mcp.rs INSTRUCTIONS"
        );
    }

    #[test]
    fn a_result_opens_the_backend_it_wrote() {
        let one = r#"{ "written": "/p/results/x.subset.zarr.zip", "info": {} }"#;
        assert_eq!(
            written_backend(one).as_deref(),
            Some("/p/results/x.subset.zarr.zip")
        );
        let halves = r#"{ "written": ["/p/cv.test.zarr.zip", "/p/cv.train.zarr.zip"] }"#;
        assert_eq!(
            written_backend(halves).as_deref(),
            Some("/p/cv.test.zarr.zip")
        );
        let job = r#"{ "job": "job-1", "written": "/p/results/x.subset.zarr.zip" }"#;
        assert_eq!(written_backend(job), None);
        // A note the person left follows the result.
        let noted = format!("{one}\nNote from the user: use exact names");
        assert_eq!(
            written_backend(&noted).as_deref(),
            Some("/p/results/x.subset.zarr.zip")
        );
        // A figure, a table, a read: nothing to open in the explorer.
        assert_eq!(written_backend(r#"{ "written": "/p/qc.png" }"#), None);
        assert_eq!(written_backend(r#"{ "rows": 3 }"#), None);
        assert_eq!(written_backend("[image/png, 3 bytes]"), None);
    }
}
