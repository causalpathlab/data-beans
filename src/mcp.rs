//! `data-beans mcp`: data-beans' inspection and backend-building commands as
//! Model Context Protocol tools, served over stdio with `legume-mcp`, so an
//! agent can look inside a backend and derive new ones from it.
//!
//! Every tool runs as a child `data-beans` process, as from the shell:
//! stdout carries the protocol, a child's log lines come back as progress,
//! and a cancelled call kills its child. Reads return what the child printed,
//! parsed and cut to a page. Writes take an `out` that must not exist yet:
//! no tool replaces a file, and `squeeze`, which rewrites its input, is not
//! offered.
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
use crate::utilities::io_helpers::parse_index_spec;
use crate::utilities::name_matching::match_by_substring;
use crate::zarr_io::{output_target, OutputTarget};
use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use legume_mcp::{Call, Content, Handler, ServerInfo, ToolSpec};
use regex::RegexBuilder;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use std::process::Command;

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
    #[cfg(feature = "tui")]
    Watch(WatchArgs),
}

#[cfg(feature = "tui")]
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
to write the kept cells as a new backend. plot shows the same settings as a \
figure (histograms with the band edges, counts against genes by verdict), \
so look at it before and after tuning.

Derive new backends with import (mtx, zarr triplets, fragments, h5ad, 10x; \
say in `source` where the input came from), subset, subsample, split, \
convert and merge. Each writes to `out`, which must not exist yet: no tool \
replaces a file. Leave `out` out and the output goes in today's results \
folder, named after its input and the tool. Outputs are .zarr.zip archives \
unless `out` ends in .zarr (a directory) or .h5. Paths are resolved against \
the server's working directory; results give the paths written.

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

/// The cell-QC settings `qc` and `plot` share, and `data`.
fn qc_properties(data: &Value) -> Value {
    json!({
        "data": data,
        "mito_pattern": { "type": "string", "description": "Regex over feature names for mitochondrial genes (default (?i)^MT-; empty turns the mito metric off)" },
        "n_mads": { "type": "number", "exclusiveMinimum": 0, "description": "Band half-width in MADs (default 5; smaller drops more)" },
        "min_cell_nnz": { "type": "integer", "minimum": 0, "description": "Near-empty floor on detected features (default 2)" },
        "mad_on_genes": { "type": "boolean", "description": "Band on detected genes (default true)" },
        "mad_on_counts": { "type": "boolean", "description": "Band on total counts (default true)" },
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
    let limit = json!({ "type": "integer", "minimum": 1, "maximum": MAX_LIMIT, "description": "Entries to return (default 100)" });
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
            "Per-row or per-column statistics (nnz, tot, mu, sig) over one or more backends: quantiles of each, and the top entries by `sort_by`. With `out` (.parquet), the full table is written there instead.",
            schema(
                json!({
                    "data": { "type": "array", "items": { "type": "string" }, "minItems": 1 },
                    "axis": axis,
                    "row_pattern": { "type": "string", "description": "Column statistics count only rows matching this case-insensitive regex, e.g. ^MT-" },
                    "sort_by": { "type": "string", "enum": ["nnz", "tot", "mu", "sig"] },
                    "ascending": { "type": "boolean" },
                    "limit": limit,
                    "out": { "type": "string", "description": "A .parquet file for the full table; must not exist" },
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
            "Cell QC as senna and pinto apply it: cells below the near-empty floor, or outside `median ± n_mads·MAD` on log detected genes and log total counts (from below) or on the mito fraction (from above), are dropped. Returns the counts kept and dropped and, per metric, its median, band and the cells outside it. Without `out` nothing is written; with it, the kept cells go to a new backend.",
            schema(
                {
                    let mut p = qc_properties(&data);
                    p["report"] = json!({ "type": "string", "description": "A .tsv(.gz) for the per-cell metrics and verdicts; must not exist" });
                    p["out"] = json!({ "type": "string", "description": "Output for the kept cells; must not exist. Omit for a dry run" });
                    p
                },
                &["data"],
            ),
        ),
        #[cfg(feature = "plot")]
        ToolSpec::new(
            "plot",
            "A figure, returned as an image (unless `out` is an .svg or .pdf). kind qc (default): each QC metric's histogram, stacked kept/outlier/near-empty, with the edge QC held it to, and total counts against detected genes by verdict; it takes the qc settings, so plot before and after tuning them. kind stat: the histogram of one statistic over rows or columns, on a log10(x+1) axis. With `out` (.png, .svg or .pdf) the figure is kept there too.",
            schema(
                {
                    let mut p = qc_properties(&data);
                    p["kind"] = json!({ "type": "string", "enum": ["qc", "stat"] });
                    p["axis"] = json!({ "type": "string", "enum": ["rows", "columns"], "description": "For kind stat (default columns)" });
                    p["stat"] = json!({ "type": "string", "enum": ["nnz", "tot", "mu", "sig"], "description": "For kind stat (default nnz)" });
                    p["out"] = json!({ "type": "string", "description": "A .png, .svg or .pdf to keep; must not exist" });
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
                    "out": out, "options": options,
                }),
                &["command", "input"],
            ),
        ),
        ToolSpec::new(
            "subset",
            "A new backend with only the named (or indexed) rows or columns. Names match exactly unless `prefix_match`.",
            schema(
                json!({
                    "data": data, "axis": axis,
                    "names": { "type": "array", "items": { "type": "string" } },
                    "indices": { "type": "array", "items": { "type": "integer", "minimum": 0 } },
                    "prefix_match": { "type": "boolean" },
                    "out": out, "options": options,
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
                    "out": out,
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
                    "options": options,
                }),
                &["data"],
            ),
        ),
        ToolSpec::new(
            "convert",
            "Copy a backend into another format: zarr <-> h5, .zarr <-> .zarr.zip; the format follows `out`.",
            schema(json!({ "data": data, "out": out }), &["data"]),
        ),
        ToolSpec::new(
            "merge",
            "Merge backends into one: their columns side by side, each tagged with its batch, and the batch labels written beside it (`data-beans merge-backend`). The backends must have the same rows in the same order, unless `common_rows`, which keeps the rows all of them have, in the first one's order.",
            schema(
                json!({
                    "data": { "type": "array", "items": { "type": "string" }, "minItems": 2 },
                    "common_rows": { "type": "boolean" },
                    "out": out, "options": options,
                }),
                &["data"],
            ),
        ),
    ]
}

/// What o opens on a call: a backend it wrote, in the statistics explorer.
#[cfg(feature = "tui")]
fn open(call: &legume_mcp::watch::view::CallView) -> Option<Command> {
    let written = written_backend(&call.text)?;
    let mut cmd = Command::new(std::env::current_exe().ok()?);
    cmd.args(["statistics", &written, "-s", "column", "--interactive"]);
    Some(cmd)
}

/// The backend a result says it wrote (`written`, or the first of a list),
/// when it is one.
#[cfg_attr(not(feature = "tui"), allow(dead_code))]
fn written_backend(text: &str) -> Option<String> {
    let v: Value = serde_json::from_str(text).ok()?;
    let w = &v["written"];
    let path = w.as_str().or_else(|| w.get(0)?.as_str())?;
    [".zarr.zip", ".zarr", ".h5"]
        .iter()
        .any(|s| path.ends_with(s))
        .then(|| path.to_string())
}

pub fn run_mcp(args: &McpArgs) -> Result<()> {
    #[cfg(feature = "tui")]
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
            "statistics" => statistics(&a, cx)?,
            "values" => values(&a, cx)?,
            "qc" => qc(&a, cx)?,
            #[cfg(feature = "plot")]
            "plot" => return plot(&a, cx),
            "import" => import(&a, cx)?,
            "subset" => subset(&a, cx)?,
            "subsample" => {
                let data = a.path("data")?;
                let out = a.backend_out(cx, &named(&data, "subsample"))?;
                let mut argv = argv(["subsample", &data]);
                argv.extend(a.numbers(&["cells", "cell_frac", "genes", "gene_frac", "seed"])?);
                let what = format!("cells and genes drawn at random from {data}");
                written(cx, argv, &out, &what, &[data])?
            }
            "split" => split(&a, cx)?,
            "convert" => {
                let data = a.path("data")?;
                let out = a.backend_out(cx, &named(&data, "converted"))?;
                let what = format!("{data} in another format");
                written(cx, argv(["convert", &data]), &out, &what, &[data])?
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
    /// `taken` says a name is in use.
    fn out_or(
        &self,
        cx: &Call,
        (name, ext): (&str, &str),
        taken: impl Fn(&str) -> bool,
    ) -> Result<String> {
        if self.given("out").is_some() {
            return self.path("out");
        }
        let dir = cx
            .results_dir()
            .context("`out` is required: this server keeps no results folder")?;
        (1..10_000)
            .map(|k| match k {
                1 => dir.join(format!("{name}{ext}")),
                k => dir.join(format!("{name}-{k}{ext}")),
            })
            .map(|p| p.to_string_lossy().into_owned())
            .find(|p| !taken(p))
            .context("no free name in the results folder")
    }

    /// [`Self::out_or`] for a backend: a name is taken when what the command
    /// would write for it exists.
    fn backend_out(&self, cx: &Call, name: &str) -> Result<String> {
        self.out_or(cx, (name, ""), |c| {
            backend_target(c).map_or(true, |t| {
                outputs_beside(strip_backend_suffix(&t.effective)).map_or(true, |v| !v.is_empty())
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
            "-I",
            "--interactive",
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
    let exe = std::env::current_exe().context("finding the data-beans binary")?;
    let own = Path::new("/proc/self/exe");
    let mut cmd = Command::new(if !exe.exists() && own.exists() {
        own
    } else {
        &exe
    });
    // `-v` so its log lines come back as progress.
    cmd.arg("-v").args(argv);
    Ok(cmd)
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

/// Run `argv`, a command that writes the backend `out` (made of `what`)
/// from `inputs`, with `-o`; say where the backend is and how big.
fn written(
    cx: &Call,
    argv: Vec<String>,
    out: &str,
    what: &str,
    inputs: &[String],
) -> Result<Value> {
    Ok(Value::Object(
        written_with(cx, argv, out, what, inputs, &[])?.0,
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
    let pattern = a
        .str("pattern")
        .map(|p| RegexBuilder::new(p).case_insensitive(true).build())
        .transpose()
        .context("reading `pattern`")?;
    let offset: usize = a.parse("offset")?.unwrap_or(0);
    let limit = a.limit()?;
    let (mut total, mut matched, mut page) = (0, 0, vec![]);
    for (i, n) in text.lines().enumerate() {
        total += 1;
        if pattern.as_ref().is_none_or(|re| re.is_match(n)) {
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

fn statistics(a: &Arguments, cx: &Call) -> Result<Value> {
    let data = a.paths("data")?;
    anyhow::ensure!(!data.is_empty(), "`data` is required");
    let dim = if a.rows()? { "row" } else { "column" };
    let mut argv = argv(["statistics", &data.join(","), "-s", dim]);
    if let Some(p) = a.str("row_pattern") {
        argv.push(format!("--row-name-pattern={p}"));
    }
    if let Some(out) = a.str("out") {
        anyhow::ensure!(out.ends_with(".parquet"), "`out` is a .parquet file");
        let what = format!("{dim} statistics of {}", data.join(", "));
        return to_file(cx, argv, out, &what);
    }
    let text = data_beans(cx, &argv)?;
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
    Ok(json!({ "count": count, "quantiles": quantiles, "sort_by": sort_by, "top": top }))
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

fn qc(a: &Arguments, cx: &Call) -> Result<Value> {
    let data = a.path("data")?;
    let mut argv = argv(["qc", &data]);
    argv.extend(qc_flags(a)?);
    let report = a.str("report");
    if let Some(report) = report {
        fresh(report)?;
        argv.push(format!("--qc-report={report}"));
    }
    let what = format!("the cells of {data} that cell QC keeps");
    // Without `out` a dry run: only the summary, and the report if asked.
    let (mut v, printed) = match (a.given("out"), report) {
        (Some(_), _) => {
            let out = a.path("out")?;
            written_with(
                cx,
                argv,
                &out,
                &what,
                std::slice::from_ref(&data),
                report.as_slice(),
            )?
        }
        (None, Some(r)) => {
            let what = format!("per-cell QC metrics and verdicts of {data}");
            (Map::new(), data_beans_making(cx, &argv, &[r], &what)?)
        }
        (None, None) => (Map::new(), data_beans(cx, &argv)?),
    };
    v.extend(key_values(&printed));
    if let Some(report) = report {
        v.insert("report".into(), json!(absolute(report)));
    }
    Ok(Value::Object(v))
}

/// How many indices an index spec (`0-9,20`) names, without listing them.
fn index_count(spec: &str) -> Result<usize> {
    let mut n = 0usize;
    for token in spec.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let width = match token.split_once('-') {
            Some((lo, hi)) => {
                let (lo, hi): (usize, usize) = (lo.trim().parse()?, hi.trim().parse()?);
                hi.checked_sub(lo).map_or(0, |d| d.saturating_add(1))
            }
            None => 1,
        };
        n = n.saturating_add(width);
    }
    Ok(n)
}

#[cfg(feature = "plot")]
fn plot(a: &Arguments, cx: &Call) -> Result<Vec<Content>> {
    let data = a.path("data")?;
    let kind = a.str("kind").unwrap_or("qc");
    let mut argv = match kind {
        "qc" => {
            let mut v = argv(["plot", "qc", &data]);
            v.extend(qc_flags(a)?);
            v
        }
        "stat" => {
            // Columns unless `axis` says rows.
            let rows = a.given("axis").is_some() && a.rows()?;
            let dim = if rows { "row" } else { "column" };
            // The names `statistics` reports them by, which `--stat` takes too.
            let stat = a.str("stat").unwrap_or("nnz");
            anyhow::ensure!(
                ["nnz", "tot", "mu", "sig"].contains(&stat),
                "`stat` is nnz, tot, mu or sig, not {stat}"
            );
            argv(["plot", "stat", &data, "-s", dim, "--stat", stat])
        }
        other => anyhow::bail!("`kind` is qc or stat, not {other}"),
    };
    // A figure in `out`'s format, else a PNG in the results folder; a
    // server keeping no records shows it to the agent only.
    let temp;
    let kept = a.given("out").is_some() || cx.results_dir().is_some();
    let file = if kept {
        let file = a.out_or(cx, (&named(&data, kind), ".png"), |c| Path::new(c).exists())?;
        fresh(&file)?;
        file
    } else {
        temp = tempfile::Builder::new().suffix(".png").tempfile()?;
        temp.path().to_string_lossy().into_owned()
    };
    argv.extend(["-o".into(), file.clone()]);
    if kept {
        data_beans_making(cx, &argv, &[&file], &format!("the {kind} figure of {data}"))?;
    } else {
        data_beans(cx, &argv)?;
    }
    let mut v = vec![];
    if file.ends_with(".png") {
        v.push(Content::Png(std::fs::read(&file)?));
    }
    if kept {
        v.push(Content::Json(json!({ "written": absolute(&file) })));
    }
    Ok(v)
}

fn import(a: &Arguments, cx: &Call) -> Result<Value> {
    let command = a.req("command")?;
    anyhow::ensure!(
        import_commands().contains(&command.as_str()),
        "`command` is one of {:?}",
        import_commands()
    );
    let input = a.path("input")?;
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
    let v = written(cx, argv, &out, &what, std::slice::from_ref(&input))?;
    // Where the input came from, in the project's data/README.md, once it
    // is in.
    let source = a.str("source").unwrap_or("a local file, source not given");
    cx.provenance(Path::new(&input), source);
    Ok(v)
}

fn subset(a: &Arguments, cx: &Call) -> Result<Value> {
    let rows = a.rows()?;
    let command = if rows {
        "subset-rows"
    } else {
        "subset-columns"
    };
    let data = a.path("data")?;
    let out = a.backend_out(cx, &named(&data, "subset"))?;
    let mut argv = argv([command, &data]);
    let names = a.strings("names")?;
    let indices: Vec<usize> = a.parse("indices")?.unwrap_or_default();
    // The names go to the command in a file, kept until it has run.
    let file;
    match (names.is_empty(), indices.is_empty()) {
        (false, true) => {
            file = tempfile::NamedTempFile::new()?;
            // Through the call, so the session script carries the names too.
            cx.write_file(file.path(), &(names.join("\n") + "\n"))?;
            argv.extend(["-f".into(), file.path().to_string_lossy().into_owned()]);
        }
        (true, false) => {
            let i: Vec<String> = indices.iter().map(usize::to_string).collect();
            argv.extend(["-i".into(), i.join(",")]);
        }
        _ => anyhow::bail!("give `names` or `indices`, one of them"),
    }
    if !a.flag("prefix_match")? {
        argv.push("--exact-names".into());
    }
    let side = if rows { "rows" } else { "columns" };
    let n = names.len().max(indices.len());
    argv.extend(a.options()?);
    written(cx, argv, &out, &format!("{n} {side} of {data}"), &[data])
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
    let mut v = written_with(cx, argv, &out, &what, &data, &[&batches])?.0;
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
        outputs_beside(c).map_or(true, |v| !v.is_empty())
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
    data_beans_making(
        cx,
        &argv,
        &halves,
        &format!("cells of {data} split for testing"),
    )?;
    // Nothing was there before, so all there is now was written.
    Ok(json!({ "written": outputs_beside(&prefix)? }))
}

#[cfg(test)]
mod tests {
    use super::written_backend;

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
        // A figure, a table, a read: nothing to open in the explorer.
        assert_eq!(written_backend(r#"{ "written": "/p/qc.png" }"#), None);
        assert_eq!(written_backend(r#"{ "rows": 3 }"#), None);
        assert_eq!(written_backend("[image/png, 3 bytes]"), None);
    }
}
