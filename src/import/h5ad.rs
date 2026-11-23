use super::{log_feature_type_histogram, Output, Staged};
use crate::hdf5_io::*;
use crate::sparse_io::*;
use crate::sparse_util::*;
use crate::utilities::name_matching::{
    colon_peak_names, compose_id_name, filter_row_indices_by_type, make_names_unique,
};

use legume_numeric::matrix::common_io::*;
use log::{info, warn};

/// An AnnData file, and where in it [`from_h5ad`] finds the names.
#[derive(Clone, Debug)]
pub struct H5adImport {
    /// The `.h5ad` file (CELLxGENE schema v7): `X/{data,indices,indptr}`,
    /// `obs/`, `var/`; `raw/X` is read over `X` when there is one.
    pub h5ad_file: Box<str>,
    /// `var/` fields to try, in order, for the feature id; `_index` is the
    /// index of `var/`, under whatever name the file stores it.
    pub row_id_field: Vec<Box<str>>,
    /// `var/` columns to try, in order, for the gene symbol, joined with the
    /// id as `id_name`.
    pub row_name_field: Vec<Box<str>>,
    /// `obs/` fields to try, in order, for the cell barcode.
    pub col_name_field: Vec<Box<str>>,
    /// Keep features whose biotype contains this; empty keeps all.
    pub select_row_type: Box<str>,
    /// Drop features whose biotype contains this; empty drops none.
    pub remove_row_type: Box<str>,
    /// `layers/` to read instead of `X`, stacked as rows
    /// `{feature}/count/{layer}` (`spliced,unspliced`: the two tracks RNA
    /// velocity reads). Empty reads `X`.
    pub layers: Vec<Box<str>>,
}

impl H5adImport {
    /// `h5ad_file` with the CLI's defaults: ids from `_index,gene_id`,
    /// symbols from `feature_name,gene_name`, barcodes from `_index,cell`,
    /// every feature kept.
    pub fn new(h5ad_file: &str) -> Self {
        let fields = |xs: &[&str]| xs.iter().map(|&x| Box::from(x)).collect();
        Self {
            h5ad_file: h5ad_file.into(),
            row_id_field: fields(&["_index", "gene_id"]),
            row_name_field: fields(&["feature_name", "gene_name"]),
            col_name_field: fields(&["_index", "cell"]),
            select_row_type: "".into(),
            remove_row_type: "".into(),
            layers: Vec::new(),
        }
    }
}

/// Build the backend `out` names from the AnnData file `h5ad`; the path
/// written. Beside it go `{stem}.cell_metadata.tsv.gz` (every `obs`
/// column, by barcode) and, when `obs` has a `donor_id`, cells named
/// `barcode@donor`, `{stem}.barcode_to_donor.tsv.gz` and
/// `{stem}.sample_metadata.tsv.gz`.
pub fn from_h5ad(h5ad: &H5adImport, out: &Output) -> anyhow::Result<Box<str>> {
    stage_h5ad(h5ad, out)?.finish()
}

/// [`from_h5ad`] short of squeezing and putting the backend in place.
pub fn stage_h5ad(args: &H5adImport, out: &Output) -> anyhow::Result<Staged> {
    let file = hdf5::File::open(args.h5ad_file.to_string())?;
    info!("Opened AnnData h5ad file: {}", args.h5ad_file);

    let (staged, backend) = out.stage()?;
    let backend_file = staged.working();
    let effective_output = &staged.effective;

    // The layers asked for, on var/; else raw/X (raw counts, `raw.X` in
    // older anndata) over X (processed/normalized).
    let layers = layer_names(&args.layers)?;
    let layer_paths: Vec<String> = layers.iter().map(|l| format!("layers/{l}")).collect();
    let (x_paths, var_path): (Vec<&str>, &str) = if !layer_paths.is_empty() {
        info!("Using layers {layers:?}");
        (layer_paths.iter().map(String::as_str).collect(), "var")
    } else if let Some((x, var)) = [("raw/X", "raw/var"), ("raw.X", "raw.var")]
        .into_iter()
        .find(|(x, _)| file.group(x).is_ok())
    {
        info!("Using {x} (raw counts)");
        (vec![x], var)
    } else {
        info!("Using X");
        (vec!["X"], "var")
    };
    let x_path = x_paths[0];

    let x_groups = x_paths
        .iter()
        .map(|p| match file.group(p) {
            Ok(g) => Ok(g),
            Err(_) if file.dataset(p).is_ok() => anyhow::bail!(
                "{}: {p} is a dense array; only sparse matrices are read",
                args.h5ad_file
            ),
            Err(e) => anyhow::bail!("{}: no {p}: {e}", args.h5ad_file),
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let x_group = &x_groups[0];
    let var_frame = Frame::open(&file, var_path)?;
    let obs_frame = Frame::open(&file, "obs")?;

    // Feature IDs, names and types from var/, barcodes from obs/ (each
    // list of candidates tried in order). Types are var/feature_type or
    // var/feature_types (scanpy, muon), often categorical.
    let row_ids = var_frame.resolve_index(&args.row_id_field, "feature IDs");
    let row_names = var_frame.resolve(&args.row_name_field, "gene symbols");
    let typed = var_frame
        .column("feature_type")
        .or_else(|_| var_frame.column("feature_types"))
        .ok();
    let column_names = obs_frame.resolve_index(&args.col_name_field, "barcodes");

    // AnnData stores (obs x var) = (cells x features), but data-beans
    // convention is (features x cells). We swap the pointer type to
    // transpose: CSR -> Column pointers, CSC -> Row pointers.
    // Every matrix (each layer) told CSR from CSC alike: by its own
    // attributes, else by the first's declared shape or the names' counts.
    let (n_obs, n_var) = declared_shape(x_group).unzip();
    let (obs_hint, var_hint) = (
        n_obs.or(column_names.as_ref().map(Vec::len)),
        n_var.or(row_ids.as_ref().map(Vec::len)),
    );

    let pointer_types = x_groups
        .iter()
        .zip(&x_paths)
        .map(|(g, p)| sparse_pointer_type(g, p, obs_hint, var_hint))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let pointer_type = pointer_types[0];

    // Read triplets
    let read = |g: &hdf5::Group, pointer_type| -> anyhow::Result<CooTripletsShape> {
        let values = g.dataset("data")?.read_1d::<f32>()?;
        let indices = g.dataset("indices")?.read_1d::<u64>()?;
        let indptr = g.dataset("indptr")?.read_1d::<u64>()?;
        ValuesIndicesPointers {
            values: values.as_slice().expect("values not contiguous"),
            indices: indices.as_slice().expect("indices not contiguous"),
            indptr: indptr.as_slice().expect("indptr not contiguous"),
        }
        .to_coo(pointer_type)
    };
    let CooTripletsShape {
        mut triplets,
        shape,
    } = read(x_group, pointer_type)?;
    let mut nnz = shape.nnz;
    info!(
        "Read {} non-zero elements reaching {} x {}",
        nnz, shape.nrows, shape.ncols
    );

    // The matrix is as large as the file says, so features and cells with
    // no count past the last one that has any are kept.
    let nrows = crate::convert::axis_len(
        n_var,
        row_ids.as_ref().map(Vec::len),
        shape.nrows,
        pointer_type == IndexPointerType::Row,
        "features",
    )?;
    let ncols = crate::convert::axis_len(
        n_obs,
        column_names.as_ref().map(Vec::len),
        shape.ncols,
        pointer_type == IndexPointerType::Column,
        "cells",
    )?;
    info!("Matrix: {} x {}", nrows, ncols);
    let listed = |what: &str, names: &[Box<str>], n: usize| {
        anyhow::ensure!(
            names.len() == n,
            "{} lists {} {what} but {x_path} has {n}",
            args.h5ad_file,
            names.len()
        );
        Ok(())
    };
    let index =
        |n: usize| -> Vec<Box<str>> { (0..n).map(|x| x.to_string().into_boxed_str()).collect() };

    let mut row_ids = row_ids.unwrap_or_else(|| {
        warn!("No feature ID column found in {var_path}/; using numeric IDs");
        index(nrows)
    });
    listed("feature IDs", &row_ids, nrows)?;
    let mut row_names = row_names.unwrap_or_else(|| {
        info!("No gene symbol column found in {var_path}/; using index only");
        vec![Box::from(""); nrows]
    });
    listed("gene symbols", &row_names, nrows)?;
    let has_types = typed.is_some();
    let row_types: Vec<Box<str>> = typed.unwrap_or_else(|| vec![Box::from(""); nrows]);
    listed("feature types", &row_types, nrows)?;

    // Peak rows in chr:start-end form, then composite row names: id_name
    let n_peaks = colon_peak_names(
        &mut row_ids,
        &mut row_names,
        has_types.then_some(&row_types[..]),
    );
    if n_peaks > 0 {
        info!("{n_peaks} peak names rewritten in chr:start-end form");
    }
    let mut row_ids = compose_id_name(row_ids, row_names);
    make_names_unique(&mut row_ids);

    // Every further layer's triplets below the first's, and every row
    // named for its layer: `{feature}/count/{layer}`.
    let (nrows, row_types) = if layer_paths.is_empty() {
        (nrows, row_types)
    } else {
        // Room for every layer at once: no copy of the first as they grow.
        let mut all = 0;
        for g in &x_groups {
            all += g.dataset("data")?.size();
        }
        triplets.reserve(all.saturating_sub(triplets.len()));
        for (k, (g, p)) in x_groups.iter().zip(&x_paths).enumerate().skip(1) {
            let more = read(g, pointer_types[k])?;
            anyhow::ensure!(
                more.shape.nrows <= nrows && more.shape.ncols <= ncols,
                "{p} reaches {} x {}, past X's {nrows} x {ncols}",
                more.shape.nrows,
                more.shape.ncols
            );
            let offset = (k * nrows) as u64;
            nnz += more.shape.nnz;
            triplets.extend(
                more.triplets
                    .into_iter()
                    .map(|(r, c, v)| (r + offset, c, v)),
            );
        }
        row_ids = layers
            .iter()
            .flat_map(|layer| {
                // `aux::feature_rows::feature_row(id, COUNT, layer, None)`,
                // which the importers, built without `aux`, cannot call.
                row_ids
                    .iter()
                    .map(move |id| Box::<str>::from(format!("{id}/count/{layer}")))
            })
            .collect();
        let k = layers.len();
        (
            nrows * k,
            row_types.iter().cloned().cycle().take(nrows * k).collect(),
        )
    };

    let column_names = column_names.unwrap_or_else(|| {
        warn!("No barcode column found in obs/; using numeric IDs");
        index(ncols)
    });
    listed("barcodes", &column_names, ncols)?;
    info!("Read {} barcodes", column_names.len());

    // The features to keep, settled before anything is written: a backend
    // without a row is of no use.
    log_feature_type_histogram(args.h5ad_file.as_ref(), &row_types);
    let select_rows =
        filter_row_indices_by_type(&row_types, &args.select_row_type, &args.remove_row_type);
    crate::convert::ensure_rows_kept(
        select_rows.len(),
        nrows,
        has_types.then_some(&row_types[..]),
        (&args.select_row_type, &args.remove_row_type),
        None,
    )?;

    // Create sparse backend
    let mut data = create_sparse_from_triplets_owned(
        triplets,
        (nrows, ncols, nnz),
        Some(backend_file),
        Some(&backend),
    )?;
    info!("created sparse matrix: {}", backend_file);
    data.register_row_names_vec(&row_ids);

    // Read all obs metadata columns
    let obs_df = obs_frame.into_dataframe()?;
    info!(
        "Read {} obs columns: {:?}",
        obs_df.col_names.len(),
        obs_df.col_names
    );

    // Output stem for metadata files (strip .zarr/.h5)
    let output_stem = crate::hdf5_io::strip_backend_suffix(effective_output);

    // Handle donor_id: barcode@donor_id column names + donor mapping files
    let donor_idx = obs_df
        .col_names
        .iter()
        .position(|c| c.as_ref() == "donor_id");

    if let Some(di) = donor_idx {
        let donors = &obs_df.col_data[di];
        let new_col_names: Vec<Box<str>> = column_names
            .iter()
            .zip(donors.iter())
            .map(|(b, d)| format!("{}@{}", b, d).into_boxed_str())
            .collect();
        data.register_column_names_vec(&new_col_names);
        info!("Column names set to barcode@donor_id");

        // barcode_to_donor.tsv.gz
        let donor_file = format!("{}.barcode_to_donor.tsv.gz", output_stem);
        let mut lines: Vec<Box<str>> = Vec::with_capacity(column_names.len() + 1);
        lines.push("barcode\tdonor_id".into());
        for (b, d) in column_names.iter().zip(donors.iter()) {
            lines.push(format!("{}\t{}", b, d).into());
        }
        write_lines(&lines, &donor_file)?;
        info!("Wrote {}", donor_file);

        // sample_metadata.tsv.gz (one row per unique donor)
        let sample_file = format!("{}.sample_metadata.tsv.gz", output_stem);
        write_sample_metadata(&sample_file, &obs_df.col_names, &obs_df.col_data, di)?;
        info!("Wrote {}", sample_file);
    } else {
        data.register_column_names_vec(&column_names);
    }

    if select_rows.len() < nrows {
        info!(
            "filtering features: {} -> {} of `{}` type",
            nrows,
            select_rows.len(),
            args.select_row_type
        );
        data.subset_columns_rows(None, Some(&select_rows))?;
    }

    // cell_metadata.tsv.gz — all obs columns indexed by barcode
    let meta_file = format!("{}.cell_metadata.tsv.gz", output_stem);
    let mut lines: Vec<Box<str>> = Vec::with_capacity(column_names.len() + 1);
    // header
    let header = std::iter::once("barcode")
        .chain(obs_df.col_names.iter().map(|s| s.as_ref()))
        .collect::<Vec<&str>>()
        .join("\t");
    lines.push(header.into());
    // rows
    for i in 0..column_names.len() {
        let row = std::iter::once(column_names[i].as_ref())
            .chain(obs_df.col_data.iter().map(|col| col[i].as_ref()))
            .collect::<Vec<&str>>()
            .join("\t");
        lines.push(row.into());
    }
    write_lines(&lines, &meta_file)?;
    info!("Wrote {}", meta_file);

    drop(data);
    Ok(staged)
}

/// An AnnData dataframe (`obs`, `var`): a group, as anndata 0.7 and later
/// write it, or one compound dataset, as older anndata did (its `index`
/// member the index, every other member a column, a categorical's codes
/// named through `uns/{column}_categories`).
enum Frame {
    Group(hdf5::Group),
    Table(H5adDataFrame),
}

impl Frame {
    fn open(file: &hdf5::File, path: &str) -> anyhow::Result<Self> {
        if let Ok(group) = file.group(path) {
            return Ok(Self::Group(group));
        }
        let table = file
            .dataset(path)
            .map_err(|e| anyhow::anyhow!("{path} is neither a group nor a table: {e}"))?;
        info!("{path}: an older AnnData table");
        Ok(Self::Table(read_compound_frame(
            &table,
            file.group("uns").ok().as_ref(),
        )?))
    }

    /// The column `name`.
    fn column(&self, name: &str) -> anyhow::Result<Vec<Box<str>>> {
        match self {
            Self::Group(g) => read_h5ad_column(g, name),
            Self::Table(t) => t
                .col_names
                .iter()
                .position(|c| c.as_ref() == name)
                .map(|j| t.col_data[j].clone())
                .ok_or_else(|| anyhow::anyhow!("no column {name}")),
        }
    }

    /// The first of `fields` there is, logged as `label`.
    fn resolve(&self, fields: &[Box<str>], label: &str) -> Option<Vec<Box<str>>> {
        fields
            .iter()
            .map(|f| f.trim())
            .filter(|f| !f.is_empty())
            .find_map(|f| {
                let v = self.column(f).ok()?;
                info!("Using '{f}' for {label} ({} entries)", v.len());
                Some(v)
            })
    }

    /// [`Self::resolve`] with `_index` the dataframe's index: the dataset a
    /// group's `_index` attribute names, a table's `index` member.
    fn resolve_index(&self, fields: &[Box<str>], label: &str) -> Option<Vec<Box<str>>> {
        let fields: Vec<Box<str>> = match self {
            Self::Group(g) => index_candidates(g, fields),
            Self::Table(_) => fields
                .iter()
                .map(|f| match f.trim() {
                    "_index" => "index".into(),
                    _ => f.clone(),
                })
                .collect(),
        };
        self.resolve(&fields, label)
    }

    /// Every column but the index.
    fn into_dataframe(self) -> anyhow::Result<H5adDataFrame> {
        match self {
            Self::Group(g) => read_h5ad_dataframe(&g),
            Self::Table(t) => {
                let (col_names, col_data) = t
                    .col_names
                    .into_iter()
                    .zip(t.col_data)
                    .filter(|(n, _)| n.as_ref() != "index")
                    .unzip();
                Ok(H5adDataFrame {
                    col_names,
                    col_data,
                })
            }
        }
    }
}

/// The `--layers` names, trimmed: none empty, none twice.
pub(super) fn layer_names(layers: &[Box<str>]) -> anyhow::Result<Vec<Box<str>>> {
    let mut out: Vec<Box<str>> = Vec::with_capacity(layers.len());
    for layer in layers {
        let layer = layer.trim();
        anyhow::ensure!(!layer.is_empty(), "--layers names an empty layer");
        anyhow::ensure!(
            !out.iter().any(|l| l.as_ref() == layer),
            "--layers names {layer} twice"
        );
        out.push(layer.into());
    }
    Ok(out)
}

/// An older AnnData dataframe, one compound dataset, as text columns by
/// member name: strings as they are, numbers printed, an integer member
/// with `uns/{member}_categories` (and every code in range, `-1` for none)
/// read as codes into those categories.
fn read_compound_frame(
    table: &hdf5::Dataset,
    uns: Option<&hdf5::Group>,
) -> anyhow::Result<H5adDataFrame> {
    use hdf5::types::TypeDescriptor;
    // In memory at C alignment: the file's records are often packed.
    let desc = table.dtype()?.to_descriptor()?.to_c_repr();
    let TypeDescriptor::Compound(compound) = &desc else {
        anyhow::bail!("{} is not a table: {desc:?}", table.name());
    };
    let (n, size) = (table.size(), desc.size());
    let memory = hdf5::Datatype::from_descriptor(&desc)?;
    // Words, so every record (a multiple of its alignment, at most 8) and
    // every member in it are aligned.
    let mut words = vec![0u64; (n * size).div_ceil(8)];
    // SAFETY: `words` holds `n` records of the memory type.
    let status = unsafe {
        hdf5_sys::h5d::H5Dread(
            table.id(),
            memory.id(),
            hdf5_sys::h5s::H5S_ALL,
            hdf5_sys::h5s::H5S_ALL,
            hdf5_sys::h5p::H5P_DEFAULT,
            words.as_mut_ptr().cast(),
        )
    };
    anyhow::ensure!(status >= 0, "reading {} failed", table.name());
    // SAFETY: the words' bytes, as H5Dread wrote them.
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), n * size) };
    let record = |i: usize, offset: usize| &bytes[i * size + offset..];

    let mut names = Vec::with_capacity(compound.fields.len());
    let mut cols = Vec::with_capacity(compound.fields.len());
    for field in &compound.fields {
        let at = |i: usize| record(i, field.offset);
        let categories = uns
            .and_then(|u| u.dataset(&format!("{}_categories", field.name)).ok())
            .and_then(|d| read_hdf5_strings(d).ok());
        let named = categories.and_then(|categories| {
            // Codes, `-1` for none; text only when every one is in range.
            (0..n)
                .map(|i| {
                    let code = integer_member(&field.ty, at(i))?;
                    match usize::try_from(code) {
                        Ok(c) => categories.get(c).cloned(),
                        Err(_) => (code == -1).then(|| "".into()),
                    }
                })
                .collect::<Option<Vec<Box<str>>>>()
        });
        let col = named.unwrap_or_else(|| (0..n).map(|i| member_text(&field.ty, at(i))).collect());
        // The variable-length strings H5Dread allocated for this member.
        if matches!(
            field.ty,
            TypeDescriptor::VarLenUnicode | TypeDescriptor::VarLenAscii
        ) {
            for i in 0..n {
                let ptr: *mut std::ffi::c_void = read_member(at(i));
                if !ptr.is_null() {
                    // SAFETY: allocated by HDF5 for this read, freed once.
                    unsafe { hdf5_sys::h5::H5free_memory(ptr) };
                }
            }
        }
        names.push(field.name.as_str().into());
        cols.push(col);
    }
    Ok(H5adDataFrame {
        col_names: names,
        col_data: cols,
    })
}

/// A plain value of type `T` at the start of `at`.
fn read_member<T: Copy>(at: &[u8]) -> T {
    assert!(at.len() >= std::mem::size_of::<T>());
    // SAFETY: in bounds, as checked; read unaligned.
    unsafe { at.as_ptr().cast::<T>().read_unaligned() }
}

/// An integer member of type `ty` at the start of `at`; `None` for any
/// other type.
fn integer_member(ty: &hdf5::types::TypeDescriptor, at: &[u8]) -> Option<i64> {
    use hdf5::types::{IntSize, TypeDescriptor as T};
    Some(match *ty {
        T::Integer(IntSize::U1) => read_member::<i8>(at).into(),
        T::Integer(IntSize::U2) => read_member::<i16>(at).into(),
        T::Integer(IntSize::U4) => read_member::<i32>(at).into(),
        T::Integer(IntSize::U8) => read_member::<i64>(at),
        T::Unsigned(IntSize::U1) => read_member::<u8>(at).into(),
        T::Unsigned(IntSize::U2) => read_member::<u16>(at).into(),
        T::Unsigned(IntSize::U4) => read_member::<u32>(at).into(),
        T::Unsigned(IntSize::U8) => i64::try_from(read_member::<u64>(at)).ok()?,
        _ => return None,
    })
}

/// A compound member of type `ty` at the start of `at`, as text.
fn member_text(ty: &hdf5::types::TypeDescriptor, at: &[u8]) -> Box<str> {
    use hdf5::types::{FloatSize, IntSize, TypeDescriptor as T};
    match *ty {
        T::VarLenUnicode | T::VarLenAscii => {
            let ptr: *const std::ffi::c_char = read_member(at);
            if ptr.is_null() {
                "".into()
            } else {
                // SAFETY: a NUL-terminated string HDF5 wrote for this read.
                unsafe { std::ffi::CStr::from_ptr(ptr) }
                    .to_string_lossy()
                    .into()
            }
        }
        T::FixedUnicode(len) | T::FixedAscii(len) => {
            let raw = &at[..len];
            let end = raw.iter().position(|&b| b == 0).unwrap_or(len);
            String::from_utf8_lossy(&raw[..end]).into()
        }
        T::Unsigned(IntSize::U8) => read_member::<u64>(at).to_string().into(),
        T::Integer(_) | T::Unsigned(_) => integer_member(ty, at)
            .map_or_else(String::new, |x| x.to_string())
            .into(),
        T::Float(FloatSize::U4) => read_member::<f32>(at).to_string().into(),
        T::Float(FloatSize::U8) => read_member::<f64>(at).to_string().into(),
        T::Boolean => (at[0] != 0).to_string().into(),
        _ => "".into(),
    }
}

/// A string attribute of an h5ad group, however the writer encoded it.
fn str_attr(loc: &hdf5::Location, name: &str) -> Option<String> {
    use hdf5::types::{FixedAscii, FixedUnicode, VarLenAscii, VarLenUnicode};
    let attr = loc.attr(name).ok()?;
    attr.read_scalar::<VarLenUnicode>()
        .map(|s| s.to_string())
        .or_else(|_| attr.read_scalar::<VarLenAscii>().map(|s| s.to_string()))
        .or_else(|_| {
            attr.read_scalar::<FixedUnicode<64>>()
                .map(|s| s.to_string())
        })
        .or_else(|_| attr.read_scalar::<FixedAscii<64>>().map(|s| s.to_string()))
        .ok()
}

/// Which axis the sparse group `x` (at `x_path`) is compressed along, as
/// data-beans sees it: its `encoding-type` (`csr_matrix` / `csc_matrix`),
/// or the `h5sparse_format` older anndata wrote. AnnData's CSR, pointers
/// over cells, has pointers over the columns here; CSC over the rows.
///
/// A group that names no format is read by its number of compressed
/// vectors, one less than the length of `indptr`: CSR when that is `n_obs`
/// (the cells) and not `n_var` (the features), CSC when it is `n_var` and
/// not `n_obs`. An error when the count matches both or neither, or the
/// group names a format that is neither.
fn sparse_pointer_type(
    x: &hdf5::Group,
    x_path: &str,
    n_obs: Option<usize>,
    n_var: Option<usize>,
) -> anyhow::Result<IndexPointerType> {
    let named = str_attr(x, "encoding-type").or_else(|| str_attr(x, "h5sparse_format"));
    let format = match named {
        Some(format) => format.to_ascii_lowercase(),
        None => {
            let vectors = x.dataset("indptr")?.size().checked_sub(1);
            let over = |n: Option<usize>| vectors.is_some() && vectors == n;
            match (over(n_obs), over(n_var)) {
                (true, false) => "csr".into(),
                (false, true) => "csc".into(),
                _ => anyhow::bail!(
                    "{x_path} has no `encoding-type` or `h5sparse_format` attribute to tell \
                     CSR from CSC, and its {} compressed vectors do not tell either for \
                     {} cells and {} features",
                    vectors.map_or("no".into(), |n| n.to_string()),
                    n_obs.map_or("unknown".into(), |n| n.to_string()),
                    n_var.map_or("unknown".into(), |n| n.to_string()),
                ),
            }
        }
    };
    if format.contains("csr") {
        info!("Sparse format: CSR -> transposing to (features x cells)");
        Ok(IndexPointerType::Column)
    } else if format.contains("csc") {
        info!("Sparse format: CSC -> transposing to (features x cells)");
        Ok(IndexPointerType::Row)
    } else {
        anyhow::bail!("{x_path} is stored as `{format}`, neither CSR nor CSC")
    }
}

/// `(n_obs, n_var)` as the sparse group declares it: its `shape`
/// attribute, or the `h5sparse_shape` older anndata wrote.
fn declared_shape(x: &hdf5::Group) -> Option<(usize, usize)> {
    ["shape", "h5sparse_shape"].iter().find_map(|name| {
        let shape: Vec<u64> = x.attr(name).ok()?.read_raw().ok()?;
        (shape.len() == 2).then(|| (shape[0] as usize, shape[1] as usize))
    })
}

/// `fields` with `_index` read as the index of the dataframe `group`: the
/// dataset its `_index` attribute names (anndata stores a named index under
/// that name), ahead of one literally called `_index`.
fn index_candidates(group: &hdf5::Group, fields: &[Box<str>]) -> Vec<Box<str>> {
    let named = str_attr(group, "_index").filter(|name| name != "_index");
    fields
        .iter()
        .flat_map(|field| {
            let index = named
                .as_deref()
                .filter(|_| field.trim() == "_index")
                .map(Box::from);
            index.into_iter().chain([field.clone()])
        })
        .collect()
}

/// Write sample-level metadata: one row per unique donor_id.
fn write_sample_metadata(
    path: &str,
    col_names: &[Box<str>],
    col_data: &[Vec<Box<str>>],
    donor_col_idx: usize,
) -> anyhow::Result<()> {
    use std::collections::BTreeMap;

    let donors = &col_data[donor_col_idx];
    let mut donor_first_row: BTreeMap<&str, usize> = BTreeMap::new();
    for (i, d) in donors.iter().enumerate() {
        donor_first_row.entry(d.as_ref()).or_insert(i);
    }

    let mut lines: Vec<Box<str>> = Vec::with_capacity(donor_first_row.len() + 1);

    // header: donor_id + all other columns
    let header = std::iter::once("donor_id")
        .chain(
            col_names
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != donor_col_idx)
                .map(|(_, s)| s.as_ref()),
        )
        .collect::<Vec<&str>>()
        .join("\t");
    lines.push(header.into());

    // one row per unique donor
    for (donor_id, &row_i) in &donor_first_row {
        let row = std::iter::once(*donor_id)
            .chain(
                col_data
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != donor_col_idx)
                    .map(|(_, col)| col[row_i].as_ref()),
            )
            .collect::<Vec<&str>>()
            .join("\t");
        lines.push(row.into());
    }

    write_lines(&lines, path)?;
    Ok(())
}
