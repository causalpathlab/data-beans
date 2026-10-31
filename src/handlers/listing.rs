//! `list-h5` and `list-zarr`: a file's hierarchy, browsed as a tree on a
//! terminal and printed otherwise.

use crate::handlers::explore::open_session;
#[cfg(feature = "hdf5")]
use crate::hdf5_io::read_hdf5_strings;
use crate::interactive::list_tui::{browse_in, Item, ListView};
use crate::interactive::ui::Tui;
use clap::Args;
use legume_numeric::matrix::common_io::{basename, file_ext};
use log::info;
#[cfg(feature = "hdf5")]
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Args, Debug)]
pub struct ListH5Args {
    pub h5_file: Box<str>,

    #[command(flatten)]
    pub view: ListViewArgs,
}

#[derive(Args, Debug)]
pub struct ListZarrArgs {
    pub zarr_file: Box<str>,

    #[command(flatten)]
    pub view: ListViewArgs,
}

/// How a listing is shown.
#[derive(Args, Debug)]
pub struct ListViewArgs {
    #[arg(
        long,
        default_value_t = false,
        help = "Print the hierarchy and stop: no browser, even on a terminal",
        long_help = "On a terminal, the hierarchy opens in a browser instead of being printed:\n\
                     Enter or the arrows open and close groups, `/` filters, `w` writes the\n\
                     marked (or shown) paths, `y` copies them. --batch-process (or no\n\
                     terminal) prints it and stops."
    )]
    pub batch_process: bool,

    #[command(flatten)]
    pub session: legume_tui::cli::SessionArgs,
}

/// A file's hierarchy: its entries for the browser, lines to show under
/// them, and the text printed in batch.
struct Listing {
    items: Vec<Item>,
    notes: Vec<String>,
    text: String,
}

/// Browse `file`'s `listing` on a terminal, else print it.
fn show(
    command: &str,
    file: &str,
    args: &ListViewArgs,
    listing: impl FnOnce() -> anyhow::Result<Listing>,
) -> anyhow::Result<()> {
    if !crate::interactive::opens_view(command, args.batch_process) {
        print!("{}", listing()?.text);
        return Ok(());
    }
    // The session first: hosted, the file is read there.
    let session = open_session(command, &args.session)?;
    let Listing { items, notes, .. } = listing()?;
    let stem = basename(file).unwrap_or_else(|_| file.into());
    let notes = notes.into_iter().filter(|n| !n.trim().is_empty()).collect();
    let view = ListView::new("tree", file, items, &format!("{stem}.paths.txt")).with_notes(notes);
    let mut tui = Tui::new(&session);
    browse_in(&mut tui, view)
}

/// A `.zarr` store's tree as zarrs draws it: two spaces a level, an
/// array's shape and type after its name.
fn zarr_items(tree: &str) -> Vec<Item> {
    tree.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let name = line.trim_start();
            let depth = (line.len() - name.len()) / 2;
            match name.split_once(' ') {
                Some((name, detail)) => Item::new(name, depth, detail),
                None => Item::new(name, depth, ""),
            }
        })
        .collect()
}

/// A zip's entry paths as a tree: each directory once, before what it
/// holds, in the order the entries come.
fn zip_items(paths: &[String]) -> Vec<Item> {
    let mut items = Vec::new();
    let mut open: Vec<&str> = Vec::new();
    for path in paths {
        let is_dir = path.ends_with('/');
        let parts: Vec<&str> = path.trim_end_matches('/').split('/').collect();
        let (dirs, leaf) = parts.split_at(parts.len() - 1);
        let same = open.iter().zip(dirs).take_while(|(a, b)| a == b).count();
        open.truncate(same);
        for (depth, d) in dirs.iter().enumerate().skip(same) {
            items.push(Item::new(format!("{d}/"), depth, ""));
            open.push(d);
        }
        if is_dir {
            items.push(Item::new(format!("{}/", leaf[0]), dirs.len(), ""));
            open.push(leaf[0]);
        } else {
            items.push(Item::new(leaf[0], dirs.len(), ""));
        }
    }
    items
}

fn zarr_listing(file: &str) -> anyhow::Result<Listing> {
    use zarrs::config::MetadataRetrieveVersion;
    use zarrs::filesystem::FilesystemStore;
    use zarrs::node::Node;
    info!(".zarr file: {file}");
    let store = Arc::new(FilesystemStore::new(file)?);
    let node = Node::open_opt(&store, "/", &MetadataRetrieveVersion::Default)?;
    let tree = node.hierarchy_tree();
    Ok(Listing {
        items: zarr_items(&tree),
        notes: Vec::new(),
        text: format!("hierarchy_tree:\n{tree}\n"),
    })
}

fn zip_listing(file: &str) -> anyhow::Result<Listing> {
    info!("zipped .zarr file: {file}");
    let reader = std::io::BufReader::new(std::fs::File::open(file)?);
    let archive = zip::ZipArchive::new(reader)?;
    let mut names = Vec::with_capacity(archive.len());
    for i in 0..archive.len() {
        let entry = archive.by_index_data(i)?;
        names.push(match entry.name() {
            Ok(name) => name.to_string(),
            Err(_) => String::from_utf8_lossy(entry.name_raw()).into_owned(),
        });
    }
    let text = names.iter().map(|n| format!("{n}\n")).collect();
    Ok(Listing {
        items: zip_items(&names),
        notes: Vec::new(),
        text,
    })
}

/// List contents of a Zarr file (either .zarr directory or .zip archive)
pub fn list_zarr(cmd_args: &ListZarrArgs) -> anyhow::Result<()> {
    let file = cmd_args.zarr_file.as_ref();
    let ext = file_ext(file)?;
    match ext.as_ref() {
        "zarr" => show("list-zarr", file, &cmd_args.view, || zarr_listing(file)),
        "zip" => show("list-zarr", file, &cmd_args.view, || zip_listing(file)),
        _ => {
            info!("unknown extension '{}'", ext);
            Ok(())
        }
    }
}

/// Stub when data-beans is built without the `hdf5` feature; produces a
/// clear runtime error if a user invokes `list-h5` on a CPU-only build.
#[cfg(not(feature = "hdf5"))]
pub fn list_h5(_cmd_args: &ListH5Args) -> anyhow::Result<()> {
    anyhow::bail!(
        "`list-h5` requires the `hdf5` feature; this build was compiled without it. \
         Reinstall with `--features hdf5` (requires libhdf5) to inspect .h5 files."
    )
}

/// List contents of an HDF5 file.
///
/// The group/dataset hierarchy (with shapes for datasets) and, for every
/// dataset whose name suggests it holds 10x-style feature types
/// (`feature_type` / `feature_types`), a histogram of distinct values so
/// the caller knows what to pass to `from-h5 --select-row-type`.
#[cfg(feature = "hdf5")]
pub fn list_h5(cmd_args: &ListH5Args) -> anyhow::Result<()> {
    let file = cmd_args.h5_file.as_ref();
    show("list-h5", file, &cmd_args.view, || h5_listing(file))
}

#[cfg(feature = "hdf5")]
fn h5_listing(data_file: &str) -> anyhow::Result<Listing> {
    let file = hdf5::File::open(data_file)?;
    info!("Opened {data_file}");

    fn walk(
        group: &hdf5::Group,
        path: &str,
        depth: usize,
        items: &mut Vec<Item>,
        feature_type_paths: &mut Vec<String>,
    ) -> hdf5::Result<()> {
        for member in group.member_names()? {
            let child_path = if path.is_empty() {
                member.clone()
            } else {
                format!("{}/{}", path, member)
            };

            if let Ok(subgroup) = group.group(&member) {
                items.push(Item::new(format!("{member}/"), depth, ""));
                walk(&subgroup, &child_path, depth + 1, items, feature_type_paths)?;
            } else if let Ok(ds) = group.dataset(&member) {
                let shape = ds.shape();
                let dtype = ds
                    .dtype()
                    .ok()
                    .and_then(|d| d.to_descriptor().ok())
                    .map(|d| format!("{:?}", d))
                    .unwrap_or_else(|| "?".to_string());
                if member == "feature_type" || member == "feature_types" {
                    feature_type_paths.push(child_path);
                }
                items.push(Item::new(member, depth, format!("[{:?} {}]", shape, dtype)));
            } else {
                items.push(Item::new(member, depth, ""));
            }
        }
        Ok(())
    }

    let mut items = Vec::new();
    let mut feature_type_paths: Vec<String> = Vec::new();
    walk(&file, "", 0, &mut items, &mut feature_type_paths)?;

    let mut notes = Vec::new();
    if let Ok(shape_ds) = file.dataset("matrix/shape") {
        if let Ok(s) = shape_ds.read_1d::<i64>() {
            notes.push(String::new());
            notes.push(format!(
                "matrix/shape = {:?}  (features x barcodes)",
                s.to_vec()
            ));
        }
    }

    for ft_path in &feature_type_paths {
        notes.push(String::new());
        notes.push(format!("feature types at /{}:", ft_path));
        match file
            .dataset(ft_path)
            .map_err(anyhow::Error::from)
            .and_then(read_hdf5_strings)
        {
            Ok(values) => {
                let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
                for v in &values {
                    *counts.entry(v.as_ref()).or_insert(0) += 1;
                }
                for (ty, n) in &counts {
                    notes.push(format!("  {:>10}  {}", n, ty));
                }
                notes.push(format!("  (total: {})", values.len()));
            }
            Err(e) => {
                notes.push(format!("  (failed to read: {})", e));
            }
        }
    }

    // As it always printed: two spaces a level, a dataset's shape and type
    // after two more.
    let mut text = String::new();
    for it in &items {
        let indent = 2 * it.depth;
        match it.detail.is_empty() {
            true => text.push_str(&format!("{:indent$}{}\n", "", it.name)),
            false => text.push_str(&format!("{:indent$}{}  {}\n", "", it.name, it.detail)),
        }
    }
    for n in &notes {
        text.push_str(n);
        text.push('\n');
    }
    Ok(Listing { items, notes, text })
}

#[cfg(test)]
mod tests {
    use super::{zarr_items, zip_items};
    use crate::interactive::list_tui::Item;

    #[test]
    fn a_zarr_tree_reads_as_items() {
        let items = zarr_items("/\n  matrix\n    data [100] float32\n  shape [2] int64\n");
        assert_eq!(
            items,
            [
                Item::new("/", 0, ""),
                Item::new("matrix", 1, ""),
                Item::new("data", 2, "[100] float32"),
                Item::new("shape", 1, "[2] int64"),
            ]
        );
    }

    #[test]
    fn zip_entries_read_as_a_tree() {
        let paths: Vec<String> = ["zarr.json", "m/", "m/zarr.json", "m/c/0", "m/c/1", "n/x"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let shown: Vec<(String, usize)> = zip_items(&paths)
            .into_iter()
            .map(|i| (i.name, i.depth))
            .collect();
        assert_eq!(
            shown,
            [
                ("zarr.json".into(), 0),
                ("m/".into(), 0),
                ("zarr.json".into(), 1),
                ("c/".into(), 1),
                ("0".into(), 2),
                ("1".into(), 2),
                ("n/".into(), 0),
                ("x".into(), 1),
            ]
        );
    }
}
