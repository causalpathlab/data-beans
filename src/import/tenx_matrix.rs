use super::{Output, Staged};
use crate::convert::{build_from_h5_matrix, MatrixLayout};

/// A Cell Ranger feature-barcode matrix `.h5`, as [`from_10x_matrix`] reads
/// it.
#[derive(Clone, Debug)]
pub struct TenxMatrixImport {
    pub h5_file: Box<str>,
    /// The group the matrix is under; `layout`'s fields are relative to it.
    pub root_group: Box<str>,
    /// Where the matrix and its names are, and which rows to keep.
    pub layout: MatrixLayout,
}

impl TenxMatrixImport {
    /// `h5_file` laid out as Cell Ranger writes it
    /// ([`MatrixLayout::cell_ranger_h5`]).
    pub fn new(h5_file: &str) -> Self {
        Self {
            h5_file: h5_file.into(),
            root_group: MatrixLayout::H5_ROOT.into(),
            layout: MatrixLayout::cell_ranger_h5(),
        }
    }
}

/// Build the backend `out` names from the 10x matrix `h5`; the path written.
pub fn from_10x_matrix(h5: &TenxMatrixImport, out: &Output) -> anyhow::Result<Box<str>> {
    stage_10x_matrix(h5, out)?.finish()
}

/// [`from_10x_matrix`] short of squeezing and putting the backend in place.
pub fn stage_10x_matrix(h5: &TenxMatrixImport, out: &Output) -> anyhow::Result<Staged> {
    let (staged, backend) = out.stage()?;
    build_from_h5_matrix(
        &h5.h5_file,
        &h5.root_group,
        &h5.layout,
        staged.working(),
        &backend,
    )?;
    Ok(staged)
}
