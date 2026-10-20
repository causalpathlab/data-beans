//! `data-beans mcp` driven over stdio, as an agent's client drives it: a
//! small backend is built, read through the tools, and derived from, and a
//! write onto an existing file is refused.

// The binary needs `tui`.
#![cfg(all(feature = "mcp", feature = "tui"))]

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

const EXE: &str = env!("CARGO_BIN_EXE_data-beans");

struct Client {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next: u64,
}

impl Client {
    fn start(dir: &Path) -> Self {
        let mut child = Command::new(EXE)
            .arg("mcp")
            .current_dir(dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Self {
            child,
            stdin,
            stdout,
            next: 0,
        }
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        self.next += 1;
        let id = self.next;
        let msg = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        writeln!(self.stdin, "{msg}").unwrap();
        loop {
            let mut line = String::new();
            assert!(
                self.stdout.read_line(&mut line).unwrap() > 0,
                "server closed"
            );
            let v: Value = serde_json::from_str(&line).unwrap();
            if v["id"] == json!(id) {
                return v;
            }
        }
    }

    /// A tool's result: its structured content, or its error text.
    fn call(&mut self, name: &str, args: Value) -> Result<Value, String> {
        let r = self.request("tools/call", json!({ "name": name, "arguments": args }));
        let r = &r["result"];
        if r["isError"] == json!(true) {
            Err(r["content"][0]["text"].as_str().unwrap().to_string())
        } else {
            Ok(r["structuredContent"].clone())
        }
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A 6 x 4 backend `base.zarr.zip` in `dir`: rows MT-CO1, MT-ND1, G2..G5;
/// row i is non-zero in column (i mod 4), with count i + 1.
fn backend(dir: &Path) {
    let mut mtx = String::from("%%MatrixMarket matrix coordinate integer general\n6 4 6\n");
    for i in 0..6 {
        mtx += &format!("{} {} {}\n", i + 1, i % 4 + 1, i + 1);
    }
    std::fs::write(dir.join("m.mtx"), mtx).unwrap();
    std::fs::write(dir.join("rows.tsv"), "MT-CO1\nMT-ND1\nG2\nG3\nG4\nG5\n").unwrap();
    std::fs::write(dir.join("cols.tsv"), "C0\nC1\nC2\nC3\n").unwrap();
    let status = Command::new(EXE)
        .args([
            "from-mtx", "m.mtx", "-r", "rows.tsv", "-c", "cols.tsv", "-o", "base",
        ])
        .current_dir(dir)
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert!(status.success());
}

#[test]
fn tools_read_derive_and_refuse_to_replace() {
    let dir = tempfile::tempdir().unwrap();
    backend(dir.path());
    let mut c = Client::start(dir.path());

    let init = c.request("initialize", json!({ "protocolVersion": "2025-06-18" }));
    assert_eq!(init["result"]["serverInfo"]["name"], "data-beans");
    let tools = c.request("tools/list", json!({}));
    let names: Vec<&str> = tools["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"statistics") && !names.contains(&"squeeze"));

    let info = c.call("info", json!({ "data": "base.zarr.zip" })).unwrap();
    assert_eq!(info["rows"], 6);
    assert_eq!(info["columns"], 4);
    assert_eq!(info["nonzeros"], 6);

    let mt = c
        .call(
            "names",
            json!({ "data": "base.zarr.zip", "axis": "rows", "pattern": "^mt-" }),
        )
        .unwrap();
    assert_eq!(mt["matched"], 2);
    assert_eq!(mt["names"][1], json!({ "index": 1, "name": "MT-ND1" }));

    // Row tot is i + 1: G5 (6) tops it, MT-CO1 (1) is last.
    let stat = c
        .call(
            "statistics",
            json!({ "data": ["base.zarr.zip"], "axis": "rows", "sort_by": "tot", "limit": 1 }),
        )
        .unwrap();
    assert_eq!(stat["count"], 6);
    assert_eq!(stat["top"][0]["name"], "G5");
    assert_eq!(stat["quantiles"]["tot"]["min"], 1.0);

    let vals = c
        .call(
            "values",
            json!({ "data": "base.zarr.zip", "axis": "rows", "names": ["G4"] }),
        )
        .unwrap();
    // G4 is row 4: count 5 in column C0, the only line kept.
    assert_eq!(vals["kept"], 1);
    assert_eq!(vals["values"][0], json!({ "column_name": "C0", "G4": 5.0 }));

    let sub = c
        .call(
            "subset",
            json!({ "data": "base.zarr.zip", "axis": "rows", "names": ["MT-CO1", "G2"], "out": "sub" }),
        )
        .unwrap();
    assert!(sub["written"].as_str().unwrap().ends_with("sub.zarr.zip"));
    assert_eq!(sub["info"]["rows"], 2);

    let again = c.call(
        "subset",
        json!({ "data": "base.zarr.zip", "axis": "rows", "names": ["G3"], "out": "sub" }),
    );
    assert!(again.unwrap_err().contains("not replacing"));

    let dirout = c
        .call(
            "convert",
            json!({ "data": "base.zarr.zip", "out": "copy.zarr" }),
        )
        .unwrap();
    assert!(dir.path().join("copy.zarr").is_dir());
    assert_eq!(dirout["info"]["nonzeros"], 6);

    let merged = c
        .call(
            "merge",
            json!({ "data": ["base.zarr.zip", "copy.zarr"], "out": "both" }),
        )
        .unwrap();
    assert!(dir.path().join("both.zarr.zip").exists());
    assert_eq!(merged["info"]["columns"], 8);
    let mismatched = c.call(
        "merge",
        json!({ "data": ["base.zarr.zip", "sub.zarr.zip"], "out": "x" }),
    );
    assert!(mismatched.unwrap_err().contains("other row names"));

    // `out` is the only way to name an output.
    let refused = c.call(
        "import",
        json!({ "command": "from-mtx", "input": "m.mtx", "out": "y", "options": ["--output=z"] }),
    );
    assert!(refused.unwrap_err().contains("not taken"));
    // Nor hidden in a cluster of short options.
    let refused = c.call(
        "import",
        json!({ "command": "from-mtx", "input": "m.mtx", "out": "y", "options": ["-Io"] }),
    );
    assert!(refused.unwrap_err().contains("apart from its value"));
    assert!(!dir.path().join("y.zarr.zip").exists());
}

#[test]
fn qc_plot_and_merge_on_common_rows() {
    let dir = tempfile::tempdir().unwrap();
    backend(dir.path());
    let mut c = Client::start(dir.path());
    c.request("initialize", json!({}));

    // C2 and C3 hold one entry each, under the near-empty floor of 2; C0
    // and C1 hold two, with MT-CO1 1 of 6 counts and MT-ND1 2 of 8.
    let dry = c.call("qc", json!({ "data": "base.zarr.zip" })).unwrap();
    assert_eq!(dry["cells"], 4);
    assert_eq!(dry["near_empty"], 2);
    assert_eq!(dry["cells_kept"], 2);
    assert!(dry.get("written").is_none());
    assert!(dry["mito_frac"]["median"].is_number());
    let kept = c
        .call(
            "qc",
            json!({ "data": "base.zarr.zip", "out": "kept", "report": "qc.tsv" }),
        )
        .unwrap();
    assert_eq!(kept["info"]["columns"], 2);
    assert!(dir.path().join("qc.tsv").exists());

    if cfg!(feature = "plot") {
        let r = c.request(
            "tools/call",
            json!({ "name": "plot", "arguments": { "data": "base.zarr.zip", "out": "qc.png" } }),
        );
        let content = &r["result"]["content"];
        assert_eq!(content[0]["type"], "image");
        assert_eq!(content[0]["mimeType"], "image/png");
        assert!(dir.path().join("qc.png").exists());
    }

    let sub = c
        .call(
            "subset",
            json!({ "data": "base.zarr.zip", "axis": "rows", "names": ["G2", "MT-CO1"], "out": "sub" }),
        )
        .unwrap();
    assert_eq!(sub["info"]["rows"], 2);
    let merged = c
        .call(
            "merge",
            json!({ "data": ["base.zarr.zip", "sub.zarr.zip"], "out": "both", "common_rows": true }),
        )
        .unwrap();
    assert_eq!(merged["info"]["rows"], 2);
    assert_eq!(merged["info"]["columns"], 8);
    // Each batch keeps its two rows' entries: MT-CO1 (1) and G2 (3).
    assert_eq!(merged["info"]["nonzeros"], 4);
    assert!(merged["batches"]
        .as_str()
        .unwrap()
        .ends_with("both.batch.gz"));
}

#[test]
fn unnamed_outputs_go_to_the_results_folder_and_the_records() {
    let dir = tempfile::tempdir().unwrap();
    backend(dir.path());
    let mut c = Client::start(dir.path());
    c.request("initialize", json!({}));

    let imported = c
        .call(
            "import",
            json!({ "command": "from-mtx", "input": "m.mtx", "options": ["-r", "rows.tsv", "-c", "cols.tsv"], "source": "the test's fixture" }),
        )
        .unwrap();
    let path = imported["written"].as_str().unwrap();
    assert!(path.contains("/results/") && path.ends_with("m.imported.zarr.zip"));
    // A second unnamed output of the same kind takes the next free name.
    let first = c
        .call("subsample", json!({ "data": path, "cells": 2 }))
        .unwrap();
    let second = c
        .call("subsample", json!({ "data": path, "cells": 3 }))
        .unwrap();
    assert!(first["written"]
        .as_str()
        .unwrap()
        .ends_with("m.imported.subsample.zarr.zip"));
    assert!(second["written"]
        .as_str()
        .unwrap()
        .ends_with("m.imported.subsample-2.zarr.zip"));

    let index = std::fs::read_to_string(dir.path().join("results/INDEX.tsv")).unwrap();
    assert!(index.contains("m.imported.subsample-2.zarr.zip"));
    let readme = std::fs::read_to_string(dir.path().join("data/README.md")).unwrap();
    assert!(readme.contains("the test's fixture"));
}
