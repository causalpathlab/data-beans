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
        Self::start_with(dir, &[])
    }

    fn start_with(dir: &Path, env: &[(&str, &Path)]) -> Self {
        let mut child = Command::new(EXE)
            .arg("mcp")
            .current_dir(dir)
            .envs(env.iter().copied())
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

/// A server in a fresh directory holding the `backend` fixture, past its
/// handshake.
fn setup() -> (tempfile::TempDir, Client) {
    let dir = tempfile::tempdir().unwrap();
    backend(dir.path());
    let mut c = Client::start(dir.path());
    c.request("initialize", json!({ "protocolVersion": "2025-06-18" }));
    (dir, c)
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
    let (dir, mut c) = setup();

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
    let (dir, mut c) = setup();

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

    // A figure's number goes before its extension.
    if cfg!(feature = "plot") {
        for want in ["m.imported.qc.png", "m.imported.qc-2.png"] {
            let r = c.request(
                "tools/call",
                json!({ "name": "plot", "arguments": { "data": path } }),
            );
            let text = r["result"]["content"][1]["text"]
                .as_str()
                .unwrap_or_default();
            assert!(text.contains(want), "{r}");
        }
    }

    let index = std::fs::read_to_string(dir.path().join("results/INDEX.tsv")).unwrap();
    assert!(index.contains("m.imported.subsample-2.zarr.zip"));
    let readme = std::fs::read_to_string(dir.path().join("data/README.md")).unwrap();
    assert!(readme.contains("the test's fixture"));
}

/// What a call should come to.
enum Want {
    /// Not an error, and the check holds on its structured content.
    Ok(fn(&Value) -> bool),
    /// A tool error (not a protocol one) whose text has this in it.
    Refused(&'static str),
}

#[test]
fn every_tool_answers_or_refuses_as_it_should() {
    use Want::{Ok, Refused};
    let (_dir, mut c) = setup();

    // Every schema is an object whose required keys are among its properties.
    let tools = c.request("tools/list", json!({}));
    for t in tools["result"]["tools"].as_array().unwrap() {
        let s = &t["inputSchema"];
        assert_eq!(s["type"], "object", "{}", t["name"]);
        for k in s["required"].as_array().into_iter().flatten() {
            assert!(
                s["properties"].get(k.as_str().unwrap()).is_some(),
                "{}: {k}",
                t["name"]
            );
        }
    }

    let cases: Vec<(&str, Value, Want)> = vec![
        (
            "info",
            json!({ "data": "missing.zarr.zip" }),
            Refused("failed"),
        ),
        ("info", json!({ "data": "-x" }), Refused("write it as ./-x")),
        (
            "list_contents",
            json!({ "file": "base.zarr.zip" }),
            Ok(|_| true),
        ),
        (
            "names",
            json!({ "data": "base.zarr.zip", "axis": "columns", "offset": 3, "limit": 5 }),
            Ok(|v| v["names"].as_array().unwrap().len() == 1),
        ),
        (
            "names",
            json!({ "data": "base.zarr.zip", "axis": "sideways" }),
            Refused("rows or columns"),
        ),
        (
            "names",
            json!({ "data": "base.zarr.zip", "axis": "rows", "pattern": "(" }),
            Refused("pattern"),
        ),
        (
            "statistics",
            json!({ "data": ["base.zarr.zip"], "axis": "columns", "limit": 0 }),
            Ok(|v| v["top"].as_array().unwrap().len() == 1),
        ),
        (
            "statistics",
            json!({ "data": ["base.zarr.zip"], "axis": "rows", "sort_by": "median" }),
            Refused("median"),
        ),
        (
            "statistics",
            json!({ "data": ["base.zarr.zip"], "axis": "rows", "out": "s.parquet" }),
            Ok(|v| v["written"].as_str().unwrap().ends_with("s.parquet")),
        ),
        (
            "statistics",
            json!({ "data": ["base.zarr.zip"], "axis": "rows", "out": "s.parquet" }),
            Refused("not replacing"),
        ),
        (
            "values",
            json!({ "data": "base.zarr.zip", "axis": "rows", "indices": "0-100000000000" }),
            Refused("at most 100"),
        ),
        (
            "values",
            json!({ "data": "base.zarr.zip", "axis": "rows" }),
            Refused("one of them"),
        ),
        (
            "qc",
            json!({ "data": "base.zarr.zip", "n_mads": "three" }),
            Refused("`n_mads` is a number"),
        ),
        (
            "import",
            json!({ "command": "rm-rf", "input": "m.mtx" }),
            Refused("one of"),
        ),
        // m.mtx sits beside `m`, but it is the input, and stays.
        (
            "import",
            json!({ "command": "from-mtx", "input": "m.mtx", "out": "m" }),
            Ok(|v| v["written"].as_str().unwrap().ends_with("m.zarr.zip")),
        ),
        // The same again: m.zarr.zip is there now, and is no input.
        (
            "import",
            json!({ "command": "from-mtx", "input": "m.mtx", "out": "m" }),
            Refused("not replacing"),
        ),
        // Rows by pattern, and by a range of indices.
        (
            "subset",
            json!({ "data": "base.zarr.zip", "axis": "rows", "pattern": "^g[23]$", "out": "g23" }),
            Ok(|v| v["info"]["rows"] == 2),
        ),
        (
            "subset",
            json!({ "data": "base.zarr.zip", "axis": "rows", "indices": "0-2", "out": "first3" }),
            Ok(|v| v["info"]["rows"] == 3),
        ),
        (
            "subset",
            json!({ "data": "base.zarr.zip", "axis": "rows", "pattern": "^nope$", "out": "none" }),
            Refused("no rows match"),
        ),
        // A selection flag of its own: the command refuses a second one.
        (
            "subset",
            json!({ "data": "base.zarr.zip", "axis": "rows", "indices": [0], "out": "x1", "options": ["-i", "1,2"] }),
            Refused("cannot be used multiple times"),
        ),
        // An output that is the input itself.
        (
            "convert",
            json!({ "data": "base.zarr.zip", "out": "base.zarr.zip" }),
            Refused("is an input"),
        ),
        // Beside its input under the same name, in another format.
        (
            "convert",
            json!({ "data": "base.zarr.zip", "out": "base.zarr" }),
            Ok(|v| v["written"].as_str().unwrap().ends_with("base.zarr")),
        ),
        (
            "import",
            json!({ "command": "from-mtx", "input": "m.mtx", "out": "y", "options": ["--qc-report=z"] }),
            Refused("not taken"),
        ),
        (
            "subset",
            json!({ "data": "base.zarr.zip", "axis": "columns", "indices": [0, 1, 2], "out": "three" }),
            Ok(|v| v["info"]["columns"] == 3),
        ),
        (
            "subsample",
            json!({ "data": "base.zarr.zip", "cells": "2", "out": "s2" }),
            Refused("is a number"),
        ),
        (
            "subsample",
            json!({ "data": "base.zarr.zip", "cells": 2, "out": "s2" }),
            Ok(|v| v["info"]["columns"] == 2),
        ),
        (
            "split",
            json!({ "data": "base.zarr.zip", "folds": 2, "out": "cv" }),
            Ok(|v| v["written"].as_array().unwrap().len() == 4),
        ),
        (
            "split",
            json!({ "data": "base.zarr.zip", "folds": 2, "out": "cv" }),
            Refused("not replacing"),
        ),
        (
            "merge",
            json!({ "data": ["base.zarr.zip"] }),
            Refused("two or more"),
        ),
        (
            "notebook",
            json!({ "kind": "observation", "text": "every tool checked" }),
            Ok(|_| true),
        ),
    ];
    for (tool, args, want) in cases {
        let r = c.request("tools/call", json!({ "name": tool, "arguments": args }));
        // A tool's failure is a result, never a protocol error.
        assert!(r.get("error").is_none(), "{tool} {args}: {r}");
        let r = &r["result"];
        let text = r["content"][0]["text"].as_str().unwrap_or_default();
        match want {
            Ok(check) => {
                assert_eq!(r["isError"], false, "{tool} {args}: {text}");
                assert!(check(&r["structuredContent"]), "{tool} {args}: {r}");
            }
            Refused(why) => {
                assert_eq!(r["isError"], true, "{tool} {args}: {r}");
                assert!(text.contains(why), "{tool} {args}: {text}");
            }
        }
    }

    if cfg!(feature = "plot") {
        let r = c.call("plot", json!({ "data": "base.zarr.zip", "kind": "violin" }));
        assert!(r.unwrap_err().contains("qc or stat"));
    }

    // An unknown tool is a protocol error.
    let r = c.request(
        "tools/call",
        json!({ "name": "no_such_tool", "arguments": {} }),
    );
    assert!(r.get("error").is_some(), "{r}");
}

#[test]
fn a_cancelled_call_gets_no_answer_and_the_server_goes_on() {
    let (_dir, mut c) = setup();
    let id = 1000;
    let call = json!({ "jsonrpc": "2.0", "id": id, "method": "tools/call",
        "params": { "name": "info", "arguments": { "data": "base.zarr.zip" } } });
    let cancel = json!({ "jsonrpc": "2.0", "method": "notifications/cancelled",
        "params": { "requestId": id } });
    writeln!(c.stdin, "{call}\n{cancel}").unwrap();
    // The ping is answered; whatever comes before it is not the call's
    // answer unless the call finished before the cancel arrived.
    let pong = c.request("ping", json!({}));
    assert!(pong.get("result").is_some(), "{pong}");
}

#[test]
fn a_person_follows_the_server_and_leaves_the_agent_a_note() {
    use legume_mcp::watch::{self, view::AgentPane, view::Handled};
    use ratatui::{backend::TestBackend, crossterm::event::KeyCode, Terminal};

    let dir = tempfile::tempdir().unwrap();
    backend(dir.path());
    let base = dir.path().join("watch");
    let mut c = Client::start_with(dir.path(), &[("LEGUME_MCP_WATCH", &base)]);
    c.request("initialize", json!({}));
    c.call("info", json!({ "data": "base.zarr.zip" })).unwrap();

    let server = watch::live_in(&base)
        .pop()
        .expect("the server's watch folder");
    assert_eq!(server.meta.server, "data-beans");
    let mut pane = AgentPane::new(&server.dir, true).unwrap();
    let screen = |pane: &mut AgentPane| {
        let mut t = Terminal::new(TestBackend::new(160, 30)).unwrap();
        t.draw(|f| pane.render(f.area(), f.buffer_mut())).unwrap();
        t.backend().to_string()
    };
    let shows = |pane: &mut AgentPane, want: &str| {
        for _ in 0..200 {
            pane.poll();
            if screen(pane).contains(want) {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        panic!("the pane never showed {want}:\n{}", screen(pane));
    };
    shows(&mut pane, "info");

    // The person leaves a note; the agent's next result carries it.
    pane.key(KeyCode::Char('n').into());
    for ch in "use exact names".chars() {
        pane.key(KeyCode::Char(ch).into());
    }
    pane.key(KeyCode::Enter.into());
    let r = c.request(
        "tools/call",
        json!({ "name": "subset", "arguments": { "data": "base.zarr.zip", "axis": "rows", "names": ["G2"], "out": "g2" } }),
    );
    assert!(
        r.to_string()
            .contains("Note from the user: use exact names"),
        "{r}"
    );
    shows(&mut pane, "subset");

    // Opened, the subset hands over what it wrote.
    match pane.key(KeyCode::Char('o').into()) {
        Handled::Open(call) => assert!(call.text.contains("g2.zarr.zip"), "{}", call.text),
        _ => panic!("o opens the subset"),
    }
}
