use super::{prepare_output_beside, squeeze_in_place, ReorderRowsArgs};
use super::{SubsetArgs, SubsetColumnsArgs, SubsetRowsArgs};
use crate::column_subset::{stream_mapped_selection, RowMap};
use crate::handlers::explore::{off_screen, open_session, pick_entries, pick_rows_and_columns};
use crate::hdf5_io::*;
use crate::interactive::picks;
use crate::interactive::stat_tui::Side;
use crate::interactive::ui::Tui;
use crate::sparse_io::*;
use crate::utilities::io_helpers::{
    parse_index_spec_within, read_col_names, read_row_names, MAX_COLUMN_NAME_IDX, MAX_ROW_NAME_IDX,
};
use crate::utilities::name_matching::names_matching;
use crate::zarr_io::{finalize_zarr_output, make_stage};

use legume_numeric::matrix::common_io::*;
use legume_numeric::matrix::membership::Membership;
use log::info;

/// The backend [`write_subset`] reads.
type Source = dyn SparseIo<IndexIter = Vec<usize>>;

/// A selection of one side must name something, each entry once and within
/// the `n` there are.
fn distinct_within(selection: &[usize], n: usize, what: &str) -> anyhow::Result<()> {
    anyhow::ensure!(!selection.is_empty(), "subset: empty {what} selection");
    let mut seen = vec![false; n];
    for &i in selection {
        let slot = seen
            .get_mut(i)
            .ok_or_else(|| anyhow::anyhow!("subset: {what} {i} is past the {n} {what}s"))?;
        anyhow::ensure!(
            !std::mem::replace(slot, true),
            "subset: the {what} selection repeats {what} {i}"
        );
    }
    Ok(())
}

/// The rows `selection` of `data`, in that order: where each row goes, and
/// the names of those kept.
fn rows_kept(data: &Source, selection: &[usize]) -> anyhow::Result<(RowMap, Vec<Box<str>>)> {
    let names = data.row_names()?;
    distinct_within(selection, names.len(), "row")?;
    let kept = selection.iter().map(|&r| names[r].clone()).collect();
    Ok((RowMap::keeping(selection, names.len())?, kept))
}

/// Stream the `columns` of `data` (the backend at `input`; all of them when
/// `None`), in that order, into the backend `output` names, with the rows
/// `rows` maps and names when given. The input is only read; the output is
/// a fresh backend of the input's format, unless its name gives another.
///
/// `zip` is applied here, the same way every `from-*` builder applies it, so a
/// bare `-o out` lands on `out.zarr.zip` whichever subcommand produced it.
///
/// Returns `(effective_output, working_path)`: what the user will find, and
/// the file actually written (for a `.zarr.zip` output, a `.zarr` directory
/// in the hidden stage beside it). Pair with [`finalize_zarr_output`] at the
/// end of the handler so `.zarr.zip` output paths get zipped.
fn write_subset(
    input: &str,
    data: &Source,
    columns: Option<&[usize]>,
    rows: Option<(RowMap, Vec<Box<str>>)>,
    output: &str,
    zip: bool,
) -> anyhow::Result<(Box<str>, Box<str>)> {
    let all_names = data.column_names()?;
    let every_column: Vec<usize>;
    let columns = match columns {
        Some(columns) => {
            distinct_within(columns, all_names.len(), "column")?;
            columns
        }
        None => {
            every_column = (0..all_names.len()).collect();
            &every_column
        }
    };
    let column_names: Vec<Box<str>> = columns.iter().map(|&c| all_names[c].clone()).collect();
    let (rows, row_names) = match rows {
        Some((map, names)) => (Some(map), names),
        None => (None, data.row_names()?),
    };

    let (backend_in, _) = resolve_backend_file(input, None)?;
    let (effective_output, backend, output_file) =
        prepare_output_beside(input, output, backend_in, zip)?;
    make_stage(&output_file)?;
    if let Some(out_dir) = dirname(&output_file) {
        mkdir(&out_dir)?;
    }

    let (nrow, ncol, nnz) = stream_mapped_selection(
        data,
        columns,
        rows.as_ref(),
        &row_names,
        &column_names,
        &output_file,
        &backend,
    )?;
    info!("wrote {nrow} rows x {ncol} columns, {nnz} non-zeros to {output_file}");
    Ok((effective_output, output_file))
}

/// Interactive runs write the same container as the input: a `.zarr`
/// directory stays a directory, a `.zarr.zip` stays zipped, and HDF5 stays
/// HDF5 (which [`write_subset`] already keeps). An explicit `.zarr` or
/// `.zarr.zip` output name still wins.
fn zip_like_input(input: &str, output: &str) -> bool {
    input.ends_with(".zarr.zip") && !output.ends_with(".zarr")
}

/// Subset columns from a sparse matrix data file
///
/// This function takes a subset of columns from the input data file and writes
/// them to a new output file. Columns can be specified either by indices or by
/// a file containing column names. Optionally, the output can be squeezed to
/// remove rows/columns with too few non-zero entries.
pub fn subset_columns(args: &SubsetColumnsArgs) -> anyhow::Result<()> {
    // The picker's session first: hosted, the rest runs in the session.
    // Nothing selected: picked in the explorer.
    let selected =
        args.column_indices.is_some() || args.name_file.is_some() || args.pattern.is_some();
    let options = "--column-indices, --name-file or --pattern";
    let picking = picks("subset-columns", args.batch_process, selected, options)?;
    let session = picking
        .then(|| open_session("subset-columns", &args.session))
        .transpose()?;
    let mut tui = session.as_ref().map(Tui::new);
    // Pick before writing, so cancelling writes nothing.
    let columns_indices = match tui.as_mut() {
        Some(tui) => match pick_entries(tui, &args.data_file, Side::Columns, "subset")? {
            Some(picked) => Some(picked),
            None => return Ok(()),
        },
        // By index or pattern: resolved before anything is written.
        None => early_columns(
            args.column_indices.as_deref(),
            args.pattern.as_deref(),
            &args.data_file,
        )?,
    };

    // Written off the screen: a `kill` meanwhile lets it finish.
    off_screen(tui.as_mut(), || {
        let column_name_file = args.name_file.clone();

        let zip = if picking {
            zip_like_input(&args.data_file, &args.output)
        } else {
            args.zip
        };
        let data = open_sparse_matrix_by_path(&args.data_file)?;

        let original_ncol = data.num_columns().unwrap_or(0);
        info!("original data: {} columns", original_ncol);

        let selected_columns = if let Some(idx) = columns_indices {
            info!("using {} column indices directly", idx.len());
            idx
        } else if let Some(column_file) = column_name_file {
            let col_names = read_col_names(column_file, MAX_COLUMN_NAME_IDX)?;
            if col_names.is_empty() {
                return Err(anyhow::anyhow!("Empty column file"));
            }
            info!("read {} column names from file", col_names.len());

            let data_col_names = data
                .column_names()
                .expect("column names not found in data file");

            let membership = Membership::from_pairs(
                col_names
                    .iter()
                    .enumerate()
                    .map(|(i, name)| (name.clone(), i.to_string().into_boxed_str())),
                args.allow_prefix && !args.exact_names,
            )
            .with_delimiter(args.delimiter);

            let (matched, stats) = membership.match_keys(&data_col_names);

            info!(
                "Column matching: {} exact + {} base_key + {} prefix = {}/{} matched",
                stats.exact,
                stats.base_key,
                stats.prefix,
                stats.total_matched(),
                data_col_names.len()
            );

            if matched.is_empty() {
                let file_sample: Vec<_> = col_names.iter().take(3).collect();
                let data_sample: Vec<_> = data_col_names.iter().take(3).collect();
                info!("column names from file (sample): {:?}", file_sample);
                info!("column names in data (sample): {:?}", data_sample);
                return Err(anyhow::anyhow!(
                    "Found empty columns - no matching column names"
                ));
            }

            let idx: Vec<usize> = data_col_names
                .iter()
                .enumerate()
                .filter_map(|(i, name)| matched.get(name).map(|_| i))
                .collect();

            info!("subsetting to {} columns", idx.len());
            idx
        } else {
            return Err(anyhow::anyhow!(
                "either `column-indices`, `name-file` or `pattern` must be provided"
            ));
        };

        let (effective_output, output_file) = write_subset(
            &args.data_file,
            data.as_ref(),
            Some(&selected_columns),
            None,
            &args.output,
            zip,
        )?;
        info!("after subset: {} columns", selected_columns.len());
        drop(data);

        if args.do_squeeze {
            squeeze_in_place(
                &output_file,
                args.row_nnz_cutoff,
                args.column_nnz_cutoff,
                None,
            )?;
        }

        finalize_zarr_output(&output_file, &effective_output)?;
        info!("done");
        Ok(())
    })
}

/// Subset rows from a sparse matrix data file
///
/// This function takes a subset of rows from the input data file and writes
/// them to a new output file. Rows can be specified either by indices or by
/// a file containing row names. Optionally, the output can be squeezed to
/// remove rows/columns with too few non-zero entries.
pub fn subset_rows(args: &SubsetRowsArgs) -> anyhow::Result<()> {
    // The picker's session first: hosted, the rest runs in the session.
    // Nothing selected: picked in the explorer.
    let selected = args.row_indices.is_some() || args.name_file.is_some() || args.pattern.is_some();
    let options = "--row-indices, --name-file or --pattern";
    let picking = picks("subset-rows", args.batch_process, selected, options)?;
    let session = picking
        .then(|| open_session("subset-rows", &args.session))
        .transpose()?;
    let mut tui = session.as_ref().map(Tui::new);
    // Pick before writing, so cancelling writes nothing.
    let row_indices = match tui.as_mut() {
        Some(tui) => match pick_entries(tui, &args.data_file, Side::Rows, "subset")? {
            Some(picked) => Some(picked),
            None => return Ok(()),
        },
        // By index or pattern: resolved before anything is written.
        None => early_rows(
            args.row_indices.as_deref(),
            args.pattern.as_deref(),
            &args.data_file,
        )?,
    };

    // Written off the screen: a `kill` meanwhile lets it finish.
    off_screen(tui.as_mut(), || {
        let row_name_file = args.name_file.clone();

        let zip = if picking {
            zip_like_input(&args.data_file, &args.output)
        } else {
            args.zip
        };
        let data = open_sparse_matrix_by_path(&args.data_file)?;

        let original_nrow = data.num_rows().unwrap_or(0);
        info!("original data: {} rows", original_nrow);

        let selected_rows = if let Some(idx) = row_indices {
            info!("using {} row indices directly", idx.len());
            idx
        } else if let Some(row_file) = row_name_file {
            let row_names = read_row_names(row_file, MAX_ROW_NAME_IDX)?;
            if row_names.is_empty() {
                return Err(anyhow::anyhow!("Empty row file"));
            }
            info!("read {} row names from file", row_names.len());

            let data_row_names = data.row_names().expect("row names not found in data file");

            let membership = Membership::from_pairs(
                row_names
                    .iter()
                    .enumerate()
                    .map(|(i, name)| (name.clone(), i.to_string().into_boxed_str())),
                args.allow_prefix && !args.exact_names,
            )
            .with_delimiter(args.delimiter);

            let (matched, stats) = membership.match_keys(&data_row_names);

            info!(
                "Row matching: {} exact + {} base_key + {} prefix = {}/{} matched",
                stats.exact,
                stats.base_key,
                stats.prefix,
                stats.total_matched(),
                data_row_names.len()
            );

            if matched.is_empty() {
                let file_sample: Vec<_> = row_names.iter().take(3).collect();
                let data_sample: Vec<_> = data_row_names.iter().take(3).collect();
                info!("row names from file (sample): {:?}", file_sample);
                info!("row names in data (sample): {:?}", data_sample);
                return Err(anyhow::anyhow!("Found empty rows - no matching row names"));
            }

            let idx: Vec<usize> = data_row_names
                .iter()
                .enumerate()
                .filter_map(|(i, name)| matched.get(name).map(|_| i))
                .collect();

            info!("subsetting to {} rows", idx.len());
            idx
        } else {
            return Err(anyhow::anyhow!(
                "either `row-indices`, `name-file` or `pattern` must be provided"
            ));
        };

        let rows = rows_kept(data.as_ref(), &selected_rows)?;
        let (effective_output, output_file) = write_subset(
            &args.data_file,
            data.as_ref(),
            None,
            Some(rows),
            &args.output,
            zip,
        )?;
        info!("after subset: {} rows", selected_rows.len());
        drop(data);

        if args.do_squeeze {
            squeeze_in_place(
                &output_file,
                args.row_nnz_cutoff,
                args.column_nnz_cutoff,
                None,
            )?;
        }

        finalize_zarr_output(&output_file, &effective_output)?;
        info!("done");
        Ok(())
    })
}

/// Subset rows and columns picked in the explorer
///
/// Opens the explorer on the columns (Tab for the rows); the user marks the
/// entries to keep on either side, and Enter writes a new backend of the
/// same format as the input with just those. A side with nothing marked is
/// kept whole.
pub fn subset(args: &SubsetArgs) -> anyhow::Result<()> {
    // It only picks: with nothing to pick in, the commands that take the
    // selection on the command line are named.
    let options = "the rows and columns to subset-rows and subset-columns instead";
    picks("subset", args.batch_process, false, options)?;
    // The picker's session first: hosted, the rest runs in the session.
    let session = open_session("subset", &args.session)?;
    let mut tui = Tui::new(&session);
    // Pick before writing, so cancelling writes nothing.
    let Some(picked) = pick_rows_and_columns(&mut tui, &args.data_file, "subset")? else {
        return Ok(());
    };
    // Written off the screen: a `kill` meanwhile lets it finish.
    off_screen(Some(&mut tui), || {
        let zip = zip_like_input(&args.data_file, &args.output);
        let data = open_sparse_matrix_by_path(&args.data_file)?;

        let rows = (!picked.rows.is_empty()).then_some(picked.rows);
        let columns = (!picked.columns.is_empty()).then_some(picked.columns);
        info!(
            "subsetting to {} rows x {} columns",
            rows.as_ref()
                .map_or(data.num_rows().unwrap_or(0), |r| r.len()),
            columns
                .as_ref()
                .map_or(data.num_columns().unwrap_or(0), |c| c.len())
        );
        let rows = rows
            .map(|rows| rows_kept(data.as_ref(), &rows))
            .transpose()?;
        let (effective_output, output_file) = write_subset(
            &args.data_file,
            data.as_ref(),
            columns.as_deref(),
            rows,
            &args.output,
            zip,
        )?;
        drop(data);

        finalize_zarr_output(&output_file, &effective_output)?;
        info!("done");
        Ok(())
    })
}

/// Reorder rows in a sparse matrix data file
///
/// This function reorders the rows of the input data file according to the
/// order specified in a row name file, and writes the result to a new output file.
pub fn reorder_rows(args: &ReorderRowsArgs) -> anyhow::Result<()> {
    let row_names_order: Vec<Box<str>> = read_row_names(args.row_file.clone(), MAX_ROW_NAME_IDX)?;

    let data = open_sparse_matrix_by_path(&args.data_file)?;

    // Each row of the data goes where the file names it; a row it does not
    // name is dropped, and a name the data lacks is an empty row.
    let place = build_name2index_map(&row_names_order);
    let mut taken = vec![false; row_names_order.len()];
    let to = data
        .row_names()?
        .iter()
        .map(|name| {
            let Some(&new) = place.get(name) else {
                return Ok(None);
            };
            anyhow::ensure!(
                !std::mem::replace(&mut taken[new], true),
                "{}: the row name {name} is on more than one row, so which of \
                 them goes to its place is not determined",
                args.data_file
            );
            Ok(Some(new as u64))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let rows = RowMap {
        to,
        nrow: row_names_order.len(),
    };

    let (effective_output, output_file) = write_subset(
        &args.data_file,
        data.as_ref(),
        None,
        Some((rows, row_names_order)),
        &args.output,
        args.zip,
    )?;
    drop(data);

    finalize_zarr_output(&output_file, &effective_output)?;
    info!("done");
    Ok(())
}

#[cfg(test)]
#[path = "subset/tests.rs"]
mod tests;

/// The columns `-i` or `--pattern` selects in `data_file`, read without
/// staging anything; none when neither is given.
fn early_columns(
    spec: Option<&str>,
    pattern: Option<&str>,
    data_file: &str,
) -> anyhow::Result<Option<Vec<usize>>> {
    early(spec, pattern, data_file, Side::Columns)
}

/// [`early_columns`] for rows.
fn early_rows(
    spec: Option<&str>,
    pattern: Option<&str>,
    data_file: &str,
) -> anyhow::Result<Option<Vec<usize>>> {
    early(spec, pattern, data_file, Side::Rows)
}

fn early(
    spec: Option<&str>,
    pattern: Option<&str>,
    data_file: &str,
    side: Side,
) -> anyhow::Result<Option<Vec<usize>>> {
    if spec.is_none() && pattern.is_none() {
        return Ok(None);
    }
    let data = open_sparse_matrix_by_path(data_file)?;
    let (what, n) = match side {
        Side::Columns => ("columns", data.num_columns().unwrap_or(0)),
        Side::Rows => ("rows", data.num_rows().unwrap_or(0)),
    };
    let idx = if let Some(spec) = spec {
        parse_index_spec_within(spec, n)?
    } else {
        let pattern = pattern.unwrap_or_default();
        let names = match side {
            Side::Columns => data.column_names()?,
            Side::Rows => data.row_names()?,
        };
        let idx = names_matching(&names, pattern)?;
        anyhow::ensure!(!idx.is_empty(), "no {what} match `{pattern}`");
        idx
    };
    info!("selected {} {what}", idx.len());
    Ok(Some(idx))
}
