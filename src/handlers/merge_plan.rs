//! The plans `merge-backend`, `merge-mtx` and `align-data` show before they
//! write: each input's rows and columns, how many rows every input has and
//! how many only one does, and the output's size, from the inputs' names
//! and counts alone.

use super::*;
use crate::handlers::explore::off_screen;
use crate::interactive::plan_tui::{plan_in, Control, Plan, PlanView, Planner, Setting};
use crate::interactive::ui::Tui;
use rustc_hash::FxHashSet as HashSet;

/// One input, as read before anything is written.
struct Input {
    /// What the plan calls it: a batch name, or the file.
    label: String,
    path: String,
    rows: Vec<Box<str>>,
    columns: usize,
    nnz: Option<usize>,
}

/// An input backend's names and counts.
fn read_backend(label: String, path: &str) -> anyhow::Result<Input> {
    let (backend, file) = resolve_backend_file(path, None)?;
    let data = open_sparse_matrix(&file, &backend)?;
    Ok(Input {
        label,
        path: path.to_string(),
        rows: data.row_names()?,
        columns: data.num_columns().unwrap_or(0),
        nnz: data.num_non_zeros(),
    })
}

/// Rows of each input not among `common`, each name once.
fn only_here(inputs: &[&Input], common: &[Box<str>]) -> Vec<usize> {
    let common: HashSet<&str> = common.iter().map(|n| n.as_ref()).collect();
    inputs
        .iter()
        .map(|i| {
            i.rows
                .iter()
                .map(|n| n.as_ref())
                .filter(|n| !common.contains(n))
                .collect::<HashSet<_>>()
                .len()
        })
        .collect()
}

/// The table of `inputs`: rows, columns, how many rows every input has
/// and how many only some, and the entries when known.
fn inputs_table(inputs: &[&Input], common: &[Box<str>], plan: &mut Plan) {
    plan.columns = [
        "input",
        "file",
        "rows",
        "columns",
        "in all",
        "not in all",
        "entries",
    ]
    .map(String::from)
    .to_vec();
    let others = only_here(inputs, common);
    plan.rows = inputs
        .iter()
        .zip(others)
        .map(|(i, other)| {
            vec![
                i.label.clone(),
                i.path.clone(),
                i.rows.len().to_string(),
                i.columns.to_string(),
                common.len().to_string(),
                other.to_string(),
                i.nnz.map_or("?".into(), |n| n.to_string()),
            ]
        })
        .collect();
}

fn squeeze_note(squeeze: bool, rows: usize, columns: usize) -> String {
    match squeeze {
        true => format!(
            "then squeezed: rows under {rows} and columns under {columns} non-zeros dropped"
        ),
        false => "not squeezed (s)".into(),
    }
}

fn cancelled(command: &str) -> anyhow::Result<()> {
    log::info!("{command}: cancelled; nothing written");
    Ok(())
}

pub(super) fn merge_backend(args: &MergeBackendArgs) -> anyhow::Result<()> {
    let Some(session) = args.view.session("merge-backend")? else {
        return super::merge_backend(args, args.common_rows, args.do_squeeze);
    };
    let mut tui = Tui::new(&session);
    let labels = generate_unique_batch_names(&args.data_files)?;
    let inputs = off_screen(Some(&mut tui), || {
        labels
            .iter()
            .zip(&args.data_files)
            .map(|(l, f)| read_backend(l.to_string(), f))
            .collect::<anyhow::Result<Vec<_>>>()
    })?;
    let lists: Vec<Vec<Box<str>>> = inputs.iter().map(|i| i.rows.clone()).collect();
    let common = common_names(&lists);
    let same = lists.windows(2).all(|w| w[0] == w[1]);
    let output = crate::zarr_io::output_path(&args.output, args.zip, &args.backend);
    let columns: usize = inputs.iter().map(|i| i.columns).sum();
    let entries: Option<usize> = inputs.iter().map(|i| i.nnz).sum();
    let planner: Planner = Box::new(|s: &[Setting]| {
        let (common_rows, squeeze) = (s[0].flag(), s[1].flag());
        let refs: Vec<&Input> = inputs.iter().collect();
        let mut plan = Plan::default();
        inputs_table(&refs, &common, &mut plan);
        let rows = if common_rows {
            common.len()
        } else {
            inputs[0].rows.len()
        };
        let entries = entries.map_or("?".into(), |n| n.to_string());
        plan.notes = vec![
            format!("{output}: {rows} rows × {columns} columns, from {entries} input entries"),
            match (common_rows, same) {
                (true, true) => "every input has the same rows".into(),
                (true, false) => format!("the {rows} rows every input has (c)"),
                (false, _) => "every row, as the inputs have them in order (c)".into(),
            },
            squeeze_note(squeeze, args.row_nnz_cutoff, args.column_nnz_cutoff),
        ];
        plan.blocked = if !common_rows && !same {
            Some(format!(
                "the inputs' rows differ: press c to keep the {} rows every input has",
                common.len()
            ))
        } else if common_rows && common.is_empty() {
            Some("the inputs share no row".into())
        } else {
            None
        };
        Ok(plan)
    });
    let view = PlanView::new(
        "merge",
        &args.data_files.join(", "),
        "merge",
        vec![
            Control::flag('c', "common rows", args.common_rows),
            Control::flag('s', "squeeze", args.do_squeeze),
        ],
        planner,
    );
    let Some(s) = plan_in(&mut tui, view)? else {
        return cancelled("merge-backend");
    };
    off_screen(Some(&mut tui), || {
        super::merge_backend(args, s[0].flag(), s[1].flag())
    })
}

pub(super) fn merge_mtx(args: &MergeMtxArgs) -> anyhow::Result<()> {
    let session = args.view.session("merge-mtx")?;
    let sets = find_mtx_sets(args)?;
    let Some(session) = session else {
        return super::merge_mtx(args, sets, args.do_squeeze);
    };
    let mut tui = Tui::new(&session);
    let inputs = off_screen(Some(&mut tui), || {
        (0..sets.batch_names.len())
            .map(|b| {
                let rows = read_row_names(sets.row_files[b].clone(), args.num_feature_name_words)?;
                let columns =
                    read_col_names(sets.col_files[b].clone(), args.num_barcode_name_words)?.len();
                Ok(Input {
                    label: sets.batch_names[b].clone(),
                    path: sets.mtx_files[b].to_string(),
                    rows,
                    columns,
                    nnz: None,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()
    })?;
    // merge-mtx keeps the rows every set has, sorted by name.
    let mut common = common_names(&inputs.iter().map(|i| i.rows.clone()).collect::<Vec<_>>());
    common.sort();
    let output = crate::zarr_io::output_path(&args.output, args.zip, &args.backend);
    let columns: usize = inputs.iter().map(|i| i.columns).sum();
    let planner: Planner = Box::new(|s: &[Setting]| {
        let refs: Vec<&Input> = inputs.iter().collect();
        let mut plan = Plan::default();
        inputs_table(&refs, &common, &mut plan);
        plan.notes = vec![
            format!(
                "{output}: the {} rows every set has, × {columns} columns",
                common.len()
            ),
            squeeze_note(s[0].flag(), args.row_nnz_cutoff, args.column_nnz_cutoff),
        ];
        plan.blocked = if inputs.is_empty() {
            Some(format!(
                "no set of {}, {} and {} under the directories given",
                args.mtx_file_name, args.feature_file_name, args.barcode_file_name
            ))
        } else if common.is_empty() {
            Some("the sets share no row".into())
        } else {
            None
        };
        Ok(plan)
    });
    let view = PlanView::new(
        "merge",
        &args.data_directories.join(", "),
        "merge",
        vec![Control::flag('s', "squeeze", args.do_squeeze)],
        planner,
    );
    let Some(s) = plan_in(&mut tui, view)? else {
        return cancelled("merge-mtx");
    };
    off_screen(Some(&mut tui), || super::merge_mtx(args, sets, s[0].flag()))
}

/// What `align-data` does with `inputs` as `types` data types: the files
/// are read row-major, a data type per row of the grid and a group of
/// columns per column; each type keeps the rows its files share and each
/// group the columns its files share.
fn align_plan(inputs: &[(Input, Vec<Box<str>>)], types: usize, out: &str) -> Plan {
    let n = inputs.len();
    let mut plan = Plan {
        columns: [
            "file",
            "type",
            "group",
            "rows",
            "columns",
            "rows kept",
            "columns kept",
        ]
        .map(String::from)
        .to_vec(),
        ..Plan::default()
    };
    if types == 0 || !n.is_multiple_of(types) {
        plan.blocked = Some(format!(
            "{n} files do not split into {types} data types (r)"
        ));
        return plan;
    }
    let groups = n / types;
    let kept = |lists: Vec<Vec<Box<str>>>| common_names(&lists).len();
    let rows_kept: Vec<usize> = (0..types)
        .map(|t| {
            kept(
                (t * groups..(t + 1) * groups)
                    .map(|i| inputs[i].0.rows.clone())
                    .collect(),
            )
        })
        .collect();
    let cols_kept: Vec<usize> = (0..groups)
        .map(|g| {
            kept(
                (g..n)
                    .step_by(groups)
                    .map(|i| inputs[i].1.clone())
                    .collect(),
            )
        })
        .collect();
    plan.rows = inputs
        .iter()
        .enumerate()
        .map(|(i, (input, cols))| {
            let (t, g) = (i / groups, i % groups);
            vec![
                input.path.clone(),
                (t + 1).to_string(),
                (g + 1).to_string(),
                input.rows.len().to_string(),
                cols.len().to_string(),
                rows_kept[t].to_string(),
                cols_kept[g].to_string(),
            ]
        })
        .collect();
    plan.notes = vec![
        format!("{out}/: {types} data type(s) × {groups} group(s), each file aligned on its type's rows and its group's columns"),
    ];
    let dropped: Vec<String> = (0..groups)
        .filter(|&g| cols_kept[g] == 0)
        .map(|g| (g + 1).to_string())
        .collect();
    if !dropped.is_empty() {
        plan.notes.push(format!(
            "group(s) {} share no column and are left out",
            dropped.join(", ")
        ));
    }
    if let Some(t) = rows_kept.iter().position(|&k| k == 0) {
        plan.blocked = Some(format!("data type {} shares no row", t + 1));
    }
    plan
}

pub(super) fn align_data(args: &AlignDataArgs) -> anyhow::Result<()> {
    let Some(session) = args.view.session("align-data")? else {
        return super::align_data(args, args.num_data_types);
    };
    let mut tui = Tui::new(&session);
    let inputs = off_screen(Some(&mut tui), || {
        args.data_files
            .iter()
            .map(|f| {
                let input = read_backend(f.to_string(), f)?;
                let (backend, file) = resolve_backend_file(f, None)?;
                let cols = open_sparse_matrix(&file, &backend)?.column_names()?;
                Ok((input, cols))
            })
            .collect::<anyhow::Result<Vec<_>>>()
    })?;
    let planner: Planner = Box::new(|s: &[Setting]| {
        let types = s[0].number().unwrap_or(0.0);
        anyhow::ensure!(
            types >= 1.0 && types.fract() == 0.0,
            "the data types (r) are a whole number, at least 1"
        );
        Ok(align_plan(&inputs, types as usize, &args.output_directory))
    });
    let view = PlanView::new(
        "align",
        &args.data_files.join(", "),
        "align",
        vec![Control::number(
            'r',
            "data types",
            Some(args.num_data_types as f64),
        )],
        planner,
    );
    let Some(s) = plan_in(&mut tui, view)? else {
        return cancelled("align-data");
    };
    let types = s[0].number().unwrap_or(1.0) as usize;
    off_screen(Some(&mut tui), || super::align_data(args, types))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(path: &str, rows: &[&str], columns: usize) -> Input {
        Input {
            label: path.into(),
            path: path.into(),
            rows: rows.iter().map(|&r| r.into()).collect(),
            columns,
            nnz: Some(columns * 2),
        }
    }

    #[test]
    fn the_table_counts_what_every_input_has() {
        let a = input("a", &["G1", "G2", "G3"], 4);
        let b = input("b", &["G2", "G3", "G4", "G4"], 3);
        let common = common_names(&[a.rows.clone(), b.rows.clone()]);
        let mut plan = Plan::default();
        inputs_table(&[&a, &b], &common, &mut plan);
        assert_eq!(
            plan.rows[0][2..],
            ["3", "4", "2", "1", "8"].map(String::from)
        );
        // A name repeated counts once.
        assert_eq!(
            plan.rows[1][2..],
            ["4", "3", "2", "1", "6"].map(String::from)
        );
    }

    #[test]
    fn align_plans_rows_by_type_and_columns_by_group() {
        let cols = |c: &[&str]| -> Vec<Box<str>> { c.iter().map(|&x| x.into()).collect() };
        // Two types (RNA, ATAC) by two groups (donors).
        let inputs = vec![
            (input("rna1", &["G1", "G2"], 2), cols(&["c1", "c2"])),
            (input("rna2", &["G2", "G3"], 2), cols(&["c3"])),
            (input("atac1", &["P1"], 2), cols(&["c1"])),
            (input("atac2", &["P1", "P2"], 2), cols(&["c9"])),
        ];
        let plan = align_plan(&inputs, 2, "out");
        assert!(plan.blocked.is_none(), "{plan:?}");
        // rna1: type 1, group 1; rows kept G2, columns kept c1.
        assert_eq!(
            plan.rows[0][1..],
            ["1", "1", "2", "2", "1", "1"].map(String::from)
        );
        // Group 2 (rna2, atac2) shares no column: left out.
        assert!(plan.notes[1].contains("group(s) 2"), "{:?}", plan.notes);
        assert!(align_plan(&inputs, 3, "out").blocked.is_some());
        let disjoint = vec![
            (input("a", &["G1"], 1), cols(&["c1"])),
            (input("b", &["G2"], 1), cols(&["c1"])),
        ];
        assert_eq!(
            align_plan(&disjoint, 1, "out").blocked.as_deref(),
            Some("data type 1 shares no row")
        );
    }
}
