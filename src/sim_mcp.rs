//! `data-beans-sim mcp`: the simulators and their fidelity diagnosis as
//! Model Context Protocol tools, served over stdio with `legume-mcp`.
//!
//! Every simulation runs as a child `data-beans-sim` process, as from the
//! shell: its log lines come back as progress, and a cancelled call kills
//! it. Outputs go to `out`, a prefix no existing file starts with; without
//! one, a new name in the day's results folder. `fidelity` reads the report
//! a reference-mode run writes.

use anyhow::{Context, Result};
use clap::Parser;
use data_beans::hdf5_io::strip_backend_suffix;
use data_beans::sim::copula::fidelity::{self, FidelityReport};
use legume_mcp::shown_path;
use legume_mcp::{Call, Content, Handler, ServerInfo, ToolSpec};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;

const INSTRUCTIONS: &str = "\
data-beans-sim simulates sparse count matrices (features as rows, cells as \
columns) in the backends data-beans reads.

The usual loop:
1. simulate_from_reference fits a real reference (.zarr.zip, .zarr or .h5) \
and simulates cells from it: each gene keeps its reference count distribution, \
cells vary in depth as the reference's do, and the copula genes keep their \
latent correlation. Topics (`factors`, `pve_topic`) and batches (`batches`, \
`pve_batch`) add known structure on top. It returns the written files and \
the fidelity summary.
2. fidelity reads a run's `{out}.fidelity.json`: library size, detected \
genes, gene mean, variance and zero fraction, and the gene-gene correlation \
of the densest copula genes, each against the reference, and the genes and \
pairs furthest off. Look at it before using a simulation as ground truth.
3. simulate_trajectory puts cells on a lineage tree with a pseudotime each \
and moves a share of genes along smooth curves, synthetic or from a reference; \
the cells' branch, pseudotime and lineages are written as ground truth.
4. simulate runs any simulator (topic, trajectory, multimodal, multiome, bulk, \
faba) with command-line `options`; command_help lists a command's options.

Outputs go to `out`, a prefix no existing file starts with; leave it out and \
a new name in today's results folder is used. Pass `background` for a big \
run: a job id comes back at once, followed with job_status / job_wait.";

/// Gene and pair gaps `fidelity` returns per kind at most.
const MAX_TOP: usize = 100;

const COMMANDS: [&str; 6] = [
    "topic",
    "trajectory",
    "multimodal",
    "multiome",
    "bulk",
    "faba",
];

fn schema(properties: Value, required: &[&str]) -> Value {
    json!({ "type": "object", "properties": properties, "required": required })
}

fn tools() -> Vec<ToolSpec> {
    let out = json!({
        "type": "string",
        "description": "Output prefix; no existing file may start with it. Omitted, a new name in today's results folder"
    });
    let options = json!({
        "type": "array", "items": { "type": "string" },
        "description": "Further command-line options, as command_help lists them (not the output, which is `out`)"
    });
    let background = json!({
        "type": "boolean",
        "description": "Run in the background: the job's id comes back at once, and job_status / job_wait follow it"
    });
    let number = |d: &str| json!({ "type": "number", "description": d });
    let integer = |d: &str| json!({ "type": "integer", "minimum": 0, "description": d });
    vec![
        ToolSpec::new(
            "simulate_from_reference",
            "Fit a real reference and simulate cells from it (`topic --reference`). Returns the files written and the fidelity summary against the reference.",
            schema(
                json!({
                    "reference": { "type": "string", "description": "Reference backend: .zarr.zip, .zarr or .h5" },
                    "cols": integer("Cells to simulate (default 1000)"),
                    "depth": integer("Mean library size; omitted, the reference's"),
                    "factors": integer("Topics K (default 1)"),
                    "pve_topic": number("Topic log-variance, capped by each gene's reference dispersion (default 1; 0 for no topics)"),
                    "pve_noise": number("iid log-mean noise, capped likewise (default 0)"),
                    "batches": integer("Batches (default 1)"),
                    "pve_batch": number("Batch effect size (default 1; no effect with one batch)"),
                    "n_hvg": integer("Copula genes (default 2000)"),
                    "hvg_min_detection": number("Smallest share of reference cells a copula gene is detected in (default 0.005, at least 50 cells)"),
                    "fixed_depth": { "type": "boolean", "description": "Same expected depth for every cell (default false)" },
                    "seed": integer("Random seed (default 42)"),
                    "out": out,
                    "options": options,
                    "background": background,
                }),
                &["reference"],
            ),
        ),
        ToolSpec::new(
            "simulate_trajectory",
            "Simulate cells along a lineage tree (`trajectory`): a root that splits into `lineages`, each cell at a pseudotime on its branch, a share of genes following smooth curves. With a `reference`, genes keep its count distributions and the fidelity summary comes back. Ground truth: `{out}.cells.parquet` (branch, pseudotime, tree_time, batch, lineage_*), `{out}.genes.parquet`, `{out}.curves.parquet`, `{out}.tree.tsv`.",
            schema(
                json!({
                    "reference": { "type": "string", "description": "Reference backend (.zarr.zip, .zarr or .h5); omitted, synthetic counts with `rows` genes" },
                    "rows": integer("Genes in synthetic mode (required without a reference)"),
                    "cols": integer("Cells to simulate (default 1000)"),
                    "lineages": integer("Lineages from one root (default 3; 1 is a single linear branch)"),
                    "tree": { "type": "string", "description": "Parent of each branch after the root, e.g. \"0,0,1,1\"; overrides `lineages`" },
                    "dynamic_genes": number("Share of genes that change along the trajectory (default 0.2)"),
                    "pve_trajectory": number("Log-variance of a dynamic gene's curve, capped by its reference dispersion (default 1)"),
                    "pve_noise": number("iid log-mean noise, capped likewise (default 0)"),
                    "batches": integer("Batches (default 1)"),
                    "pve_batch": number("Batch effect size (default 1; no effect with one batch)"),
                    "depth": integer("Mean library size; omitted, the reference's (synthetic: 1000)"),
                    "seed": integer("Random seed (default 42)"),
                    "out": out,
                    "options": options,
                    "background": background,
                }),
                &[],
            ),
        ),
        ToolSpec::new(
            "fidelity",
            "A reference-mode run against its reference, from its `{out}.fidelity.json` (or the run's `out`): summary statistics, and the `top` genes and gene pairs furthest off in mean, variance, zero fraction and correlation.",
            schema(
                json!({
                    "report": { "type": "string", "description": "The .fidelity.json, or the run's output prefix" },
                    "top": { "type": "integer", "minimum": 0, "maximum": MAX_TOP, "description": "Genes and pairs per kind (default 10)" },
                }),
                &["report"],
            ),
        ),
        ToolSpec::new(
            "simulate",
            "Run one simulator with command-line `options`: topic (log-normal topic model, or `--reference` mode), trajectory (lineage tree with pseudotime), multimodal, multiome (paired ATAC + RNA), bulk (deconvolution pseudo-bulk) or faba (RNA modification tracks). Returns the files written.",
            schema(
                json!({
                    "command": { "type": "string", "enum": COMMANDS },
                    "options": options,
                    "out": out,
                    "background": background,
                }),
                &["command"],
            ),
        ),
        ToolSpec::new(
            "command_help",
            "A simulator's full option list (`data-beans-sim <command> --help`).",
            schema(json!({ "command": { "type": "string", "enum": COMMANDS } }), &["command"]),
        ),
    ]
}

pub fn run_mcp() -> Result<()> {
    let info = ServerInfo {
        name: "data-beans-sim".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        instructions: Some(INSTRUCTIONS.into()),
    };
    legume_mcp::serve(info, Sim)
}

struct Sim;

impl Handler for Sim {
    fn tools(&self) -> Vec<ToolSpec> {
        tools()
    }

    fn call(&mut self, name: &str, args: Value, cx: &Call) -> Result<Vec<Content>> {
        let v = match name {
            "simulate_from_reference" => simulate_from_reference(&args, cx)?,
            "simulate_trajectory" => simulate_trajectory(&args, cx)?,
            "fidelity" => fidelity(&args)?,
            "simulate" => {
                let command = command_arg(&args)?;
                let options = options(&args)?;
                run(
                    cx,
                    &args,
                    command,
                    options,
                    &format!("{command} simulation"),
                )?
            }
            "command_help" => {
                let command = command_arg(&args)?;
                json!({ "help": cx.run_reading(sim_command(&[command, "--help"])?)? })
            }
            _ => anyhow::bail!("no tool {name}"),
        };
        Ok(vec![Content::Json(v)])
    }
}

fn command_arg(args: &Value) -> Result<&'static str> {
    let c = args
        .get("command")
        .and_then(Value::as_str)
        .context("`command` is required")?;
    COMMANDS
        .into_iter()
        .find(|&k| k == c)
        .with_context(|| format!("no simulator {c}; one of {COMMANDS:?}"))
}

/// `options` as given; [`run`] checks them with the simulator's own parser.
fn options(args: &Value) -> Result<Vec<String>> {
    match args.get("options") {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(o) => serde_json::from_value(o.clone()).context("`options` is a list of strings"),
    }
}

/// `topic --reference` with the typed settings, then the fidelity summary.
fn simulate_from_reference(args: &Value, cx: &Call) -> Result<Value> {
    let reference = args
        .get("reference")
        .and_then(Value::as_str)
        .context("`reference` is required")?;
    anyhow::ensure!(Path::new(reference).exists(), "no reference at {reference}");
    let mut argv = vec!["--reference".to_string(), shown_path(Path::new(reference))];
    default_cols(args, &mut argv);
    numeric_flags(
        args,
        &[
            ("cols", "--cols"),
            ("depth", "--depth"),
            ("factors", "--factors"),
            ("pve_topic", "--pve-topic"),
            ("pve_noise", "--pve-noise"),
            ("batches", "--batches"),
            ("pve_batch", "--pve-batch"),
            ("n_hvg", "--n-hvg"),
            ("hvg_min_detection", "--hvg-min-detection"),
            ("seed", "--rseed"),
        ],
        &mut argv,
    )?;
    if args.get("fixed_depth").and_then(Value::as_bool) == Some(true) {
        argv.push("--fixed-depth".into());
    }
    argv.extend(options(args)?);
    let what = format!("cells simulated from {reference}");
    with_fidelity_summary(run(cx, args, "topic", argv, &what)?)
}

/// `trajectory` with the typed settings, then the fidelity summary of a
/// reference-mode run.
fn simulate_trajectory(args: &Value, cx: &Call) -> Result<Value> {
    let mut argv = Vec::new();
    let reference = args.get("reference").and_then(Value::as_str);
    if let Some(r) = reference {
        anyhow::ensure!(Path::new(r).exists(), "no reference at {r}");
        argv.extend(["--reference".to_string(), shown_path(Path::new(r))]);
    }
    default_cols(args, &mut argv);
    numeric_flags(
        args,
        &[
            ("rows", "--rows"),
            ("cols", "--cols"),
            ("lineages", "--lineages"),
            ("dynamic_genes", "--dynamic-genes"),
            ("pve_trajectory", "--pve-trajectory"),
            ("pve_noise", "--pve-noise"),
            ("batches", "--batches"),
            ("pve_batch", "--pve-batch"),
            ("depth", "--depth"),
            ("seed", "--rseed"),
        ],
        &mut argv,
    )?;
    if let Some(t) = args.get("tree").filter(|v| !v.is_null()) {
        let t = t.as_str().context("`tree` is a comma-separated list")?;
        argv.extend(["--tree".to_string(), t.to_string()]);
    }
    argv.extend(options(args)?);
    let what = match reference {
        Some(r) => format!("trajectory cells simulated from {r}"),
        None => "synthetic trajectory cells".to_string(),
    };
    with_fidelity_summary(run(cx, args, "trajectory", argv, &what)?)
}

/// A finished run's result with its fidelity summary, when it wrote one.
fn with_fidelity_summary(mut v: Value) -> Result<Value> {
    if v.get("job").is_none() {
        if let Some(path) = v.get("fidelity_report").and_then(Value::as_str) {
            v["summary"] = json!(FidelityReport::read(path)?.summary());
        }
    }
    Ok(v)
}

/// Cells the typed tools simulate when `cols` is not given.
const DEFAULT_COLS: &str = "1000";

/// `--cols 1000` unless `cols` is given (the simulators require it).
fn default_cols(args: &Value, argv: &mut Vec<String>) {
    if args.get("cols").is_none_or(Value::is_null) {
        argv.extend(["--cols".to_string(), DEFAULT_COLS.to_string()]);
    }
}

/// `flag value` for each `key` given in `args`, which must be a number.
fn numeric_flags(args: &Value, keys: &[(&str, &str)], argv: &mut Vec<String>) -> Result<()> {
    for &(key, flag) in keys {
        if let Some(v) = args.get(key).filter(|v| !v.is_null()) {
            anyhow::ensure!(v.is_number(), "`{key}` is a number");
            argv.extend([flag.to_string(), v.to_string()]);
        }
    }
    Ok(())
}

/// Run `command` with `argv` and `-o` the output prefix; what it wrote, or
/// the background job writing it.
fn run(cx: &Call, args: &Value, command: &str, argv: Vec<String>, what: &str) -> Result<Value> {
    let out = output_prefix(cx, args, command, &argv)?;
    let mut full = vec![command.to_string()];
    full.extend(argv);
    full.extend(["-o".to_string(), out.clone()]);
    // The simulator's own parser checks the line before anything runs: an
    // unknown flag, a bad value, or a second output (`-o` in `options`)
    // comes back as its error.
    crate::Cli::try_parse_from(
        std::iter::once("data-beans-sim").chain(full.iter().map(String::as_str)),
    )
    .map_err(|e| anyhow::anyhow!("{}", e.render()))?;
    // What a replay finds done: the topic or trajectory backend in its
    // default format.
    // Other simulators write per-modality files, and `--backend` or
    // `--no-zip` change the name, so nothing is skipped for those.
    let default_format = !full
        .iter()
        .any(|a| a == "--no-zip" || a.starts_with("--backend"));
    let planned: Vec<PathBuf> = if matches!(command, "topic" | "trajectory") && default_format {
        vec![PathBuf::from(format!("{out}.zarr.zip"))]
    } else {
        Vec::new()
    };
    let cmd = sim_command(&full)?;
    if args.get("background").and_then(Value::as_bool) == Some(true) {
        let job = cx.spawn_job(cmd, &planned, what)?;
        return Ok(json!({
            "job": job,
            "out": out,
            "note": format!("{job} writes files starting with {out}."),
        }));
    }
    cx.run_making(cmd, &planned, what)?;
    let written = written_with(&out)?;
    // Topic writes `{out}.fidelity.json`, multiome one per modality.
    let reports: Vec<&String> = written
        .iter()
        .filter(|f| f.ends_with(".fidelity.json"))
        .collect();
    let mut v = json!({ "out": out, "written": written });
    if let [one] = reports.as_slice() {
        v["fidelity_report"] = json!(one);
    } else if !reports.is_empty() {
        v["fidelity_reports"] = json!(reports);
    }
    Ok(v)
}

/// The output prefix: `out` as given (refused when a file already starts
/// with it), or a new name in the results folder after the reference or the
/// command.
fn output_prefix(cx: &Call, args: &Value, command: &str, argv: &[String]) -> Result<String> {
    if let Some(out) = args.get("out").and_then(Value::as_str) {
        let out = shown_path(Path::new(out));
        let taken = written_with(&out)?;
        anyhow::ensure!(
            taken.is_empty(),
            "{out} is taken: {} already exist(s); choose another `out`",
            taken.join(", ")
        );
        return Ok(out);
    }
    let base = argv
        .iter()
        .position(|a| a == "--reference")
        .and_then(|i| argv.get(i + 1))
        .map(|r| format!("{}.sim", stem(r)))
        .unwrap_or_else(|| format!("{command}.sim"));
    let plan = |p: &str| Ok(vec![PathBuf::from(format!("{p}.zarr.zip"))]);
    let taken = |p: &str| written_with(p).map_or(true, |f| !f.is_empty());
    // Today's results folder, skipping names a running job holds; without
    // records, the working directory.
    let (out, _) = if cx.results_dir().is_some() {
        cx.free_prefix(&base, plan, taken)?
    } else {
        let dir = std::env::current_dir().context("finding a folder for the output")?;
        legume_mcp::free_prefix(&dir, &base, plan, taken)?
    };
    Ok(out)
}

/// Files in `prefix`'s folder whose names start with its name and a dot,
/// sorted; none when the folder does not exist yet.
fn written_with(prefix: &str) -> Result<Vec<String>> {
    let path = Path::new(prefix);
    let (Some(dir), Some(name)) = (path.parent(), path.file_name()) else {
        return Ok(Vec::new());
    };
    let dir = if dir.as_os_str().is_empty() {
        Path::new(".")
    } else {
        dir
    };
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e).with_context(|| format!("reading {}", dir.display())),
    };
    let lead = format!("{}.", name.to_string_lossy());
    let mut found: Vec<String> = entries
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with(&lead))
        .map(|e| e.path().to_string_lossy().into_owned())
        .collect();
    found.sort();
    Ok(found)
}

/// A reference's name without folder and backend suffix.
fn stem(path: &str) -> String {
    let stripped = strip_backend_suffix(path);
    Path::new(stripped).file_name().map_or_else(
        || stripped.to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

/// `fidelity`: a report's summary and gaps.
fn fidelity(args: &Value) -> Result<Value> {
    let given = args
        .get("report")
        .and_then(Value::as_str)
        .context("`report` is required")?;
    let path = if given.ends_with(".json") {
        given.to_string()
    } else {
        // A topic run's report, or the one report of a multiome run.
        let topic = fidelity::report_path(given);
        let found: Vec<String> = written_with(given)?
            .into_iter()
            .filter(|f| f.ends_with(".fidelity.json"))
            .collect();
        match found.as_slice() {
            _ if Path::new(&topic).exists() => topic,
            [one] => one.clone(),
            [] => anyhow::bail!("no fidelity report starts with {given}"),
            many => anyhow::bail!("{given} has several fidelity reports; name one: {many:?}"),
        }
    };
    let top = match args.get("top") {
        None | Some(Value::Null) => 10,
        Some(t) => t
            .as_u64()
            .context("`top` is a whole number")?
            .min(MAX_TOP as u64) as usize,
    };
    let report = FidelityReport::read(&path)?;
    Ok(json!({
        "report": path,
        "reference": report.reference,
        "output": report.output,
        "summary": report.summary(),
        "furthest": report.gaps(top),
    }))
}

/// `data-beans-sim` itself with `argv`, logging (`-v`) so its lines come
/// back as progress. No view opens: `--view` is refused in `options`.
fn sim_command<S: AsRef<str>>(argv: &[S]) -> Result<Command> {
    let mut cmd = legume_mcp::own_command(["-v"])?;
    cmd.args(argv.iter().map(AsRef::as_ref));
    cmd.env("RUST_BACKTRACE", "0")
        .env("RUST_LIB_BACKTRACE", "0");
    Ok(cmd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_simulators_parser_refuses_a_second_output() {
        let line = |extra: &[&str]| {
            let mut v = vec!["data-beans-sim", "topic"];
            v.extend_from_slice(extra);
            v.extend(["-o", "x"]);
            crate::Cli::try_parse_from(v)
        };
        assert!(line(&["--cols", "10"]).is_ok());
        assert!(line(&["-o", "y"]).is_err());
        assert!(line(&["--no-such-flag"]).is_err());
        assert!(options(&json!({})).unwrap().is_empty());
    }

    #[test]
    fn prefixes_find_what_starts_with_them() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let p = dir.path().join("run").to_string_lossy().into_owned();
        assert!(written_with(&p)?.is_empty());
        std::fs::write(format!("{p}.zarr.zip"), b"")?;
        std::fs::write(dir.path().join("runner.txt"), b"")?;
        assert_eq!(written_with(&p)?, vec![format!("{p}.zarr.zip")]);
        assert!(written_with("/no/such/folder/x")?.is_empty());
        assert_eq!(stem("/a/b/pbmc.zarr.zip"), "pbmc");
        Ok(())
    }

    #[test]
    fn every_tool_has_an_object_schema() {
        for t in tools() {
            assert_eq!(t.input_schema["type"], "object", "{}", t.name);
        }
        assert!(command_arg(&json!({ "command": "topic" })).is_ok());
        assert!(command_arg(&json!({ "command": "trajectory" })).is_ok());
        let mut argv = Vec::new();
        default_cols(&json!({}), &mut argv);
        assert_eq!(argv, ["--cols", DEFAULT_COLS]);
        default_cols(&json!({ "cols": 5 }), &mut argv);
        assert_eq!(argv.len(), 2);
        assert!(command_arg(&json!({ "command": "rm" })).is_err());
    }
}
