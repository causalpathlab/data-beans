//! The plan `split` shows before it writes: each fold's train and test
//! cells, from the test fraction or folds, the seed and, under `--coord`,
//! the grid, each on a key.

use super::*;
use crate::interactive::plan_tui::{plan_in, Control, Plan, PlanView, Planner, Setting};

/// A whole number of at least `min` from a setting, or `None` unset.
fn whole(s: &Setting, what: &str, min: usize) -> anyhow::Result<Option<usize>> {
    match s.number() {
        None => Ok(None),
        Some(v) if v.fract() == 0.0 && v >= min as f64 => Ok(Some(v as usize)),
        Some(v) => anyhow::bail!("{what} is a whole number, at least {min}; not {v}"),
    }
}

/// The split the settings say: test fraction, folds, seed, then the grid
/// when there is one; `base` for what they leave.
fn choice_of(s: &[Setting], base: Choice) -> anyhow::Result<Choice> {
    let test_frac = s[0].number();
    let folds = whole(&s[1], "folds (k)", 2)?;
    anyhow::ensure!(
        !(test_frac.is_some() && folds.is_some()),
        "set the test fraction (t) or the folds (k), not both: clear one"
    );
    let seed = whole(&s[2], "the seed (s)", 0)?.unwrap_or(base.seed as usize) as u64;
    let grid = match s.get(3) {
        Some(g) => whole(g, "the grid (g)", 2)?.unwrap_or(base.grid),
        None => base.grid,
    };
    Ok(Choice {
        test_frac,
        folds,
        grid,
        seed,
    })
}

/// What `choice` writes from the cells `names`.
fn plan_of(args: &SplitArgs, names: &[Box<str>], choice: Choice) -> anyhow::Result<Plan> {
    let groups = column_groups(args, names, choice.grid)?.len();
    let splits = split_columns(args, names, choice)?;
    let n = names.len().max(1) as f64;
    let rows = splits
        .iter()
        .enumerate()
        .map(|(fold, (train, test))| {
            vec![
                half_name(args, choice, fold, "{train,test}"),
                train.len().to_string(),
                test.len().to_string(),
                format!("{:.1}%", 100.0 * test.len() as f64 / n),
            ]
        })
        .collect();
    let unit = match (args.coord.as_deref(), args.groups.as_deref()) {
        (Some(path), _) => format!(
            "{groups} non-empty tiles of a {g}×{g} grid over {path}",
            g = choice.grid
        ),
        (_, Some(path)) => format!("{groups} groups labelled in {path}"),
        _ => "one cell each".into(),
    };
    Ok(Plan {
        columns: ["output", "train cells", "test cells", "test share"]
            .map(String::from)
            .to_vec(),
        rows,
        notes: vec![
            format!(
                "{} cells, shuffled and assigned in units of {unit}",
                names.len()
            ),
            format!("seed {}", choice.seed),
        ],
        blocked: None,
    })
}

/// Show the plan of `choice` on `tui`; the split it ended on, or `None`
/// when cancelled.
pub(super) fn choose(
    tui: &mut Tui<'_>,
    args: &SplitArgs,
    names: &[Box<str>],
    choice: Choice,
) -> anyhow::Result<Option<Choice>> {
    let mut controls = vec![
        Control::number('t', "test fraction", choice.test_frac),
        Control::number('k', "folds", choice.folds.map(|k| k as f64)),
        Control::number('s', "seed", Some(choice.seed as f64)),
    ];
    if args.coord.is_some() {
        controls.push(Control::number('g', "grid", Some(choice.grid as f64)));
    }
    let planner: Planner = Box::new(move |s: &[Setting]| {
        let c = choice_of(s, choice)?;
        plan_of(args, names, c)
    });
    let view = PlanView::new("split", &args.data_file, "split", controls, planner);
    match plan_in(tui, view)? {
        Some(s) => choice_of(&s, choice).map(Some),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Choice {
        Choice {
            test_frac: Some(0.2),
            folds: None,
            grid: 8,
            seed: 42,
        }
    }

    #[test]
    fn settings_read_as_a_split() {
        let n = |v: Option<f64>| Setting::Number(v);
        let c = choice_of(&[n(None), n(Some(5.0)), n(Some(7.0))], base()).unwrap();
        assert_eq!(
            (c.test_frac, c.folds, c.seed, c.grid),
            (None, Some(5), 7, 8)
        );
        let both = choice_of(&[n(Some(0.3)), n(Some(5.0)), n(Some(1.0))], base());
        assert!(both.unwrap_err().to_string().contains("not both"));
        assert!(choice_of(&[n(None), n(Some(1.0)), n(Some(1.0))], base()).is_err());
        assert!(choice_of(&[n(Some(0.2)), n(None), n(Some(1.5))], base()).is_err());
        let g = choice_of(&[n(Some(0.2)), n(None), n(Some(1.0)), n(Some(4.0))], base()).unwrap();
        assert_eq!(g.grid, 4);
    }
}
