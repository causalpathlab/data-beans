//! Metadata: short strings stored with a backend that say what its data is,
//! so a reader need not infer it from the file name. Any key may be used;
//! the ones below are shared between the programs that write and read
//! backends, and mean the same everywhere.

// The shared keys are for the library's users; the binary copies metadata
// through without naming them.
#![allow(dead_code)]

use std::collections::BTreeMap;

/// Key → value, in key order.
pub type Metadata = BTreeMap<String, String>;

/// The root attribute that holds the metadata, in both backends.
pub(crate) const ATTR: &str = "meta";

/// The sample or batch the data comes from, e.g. `s1`. Merging backends of
/// different samples drops it.
pub const SAMPLE: &str = "sample";

/// The program and command that wrote the data, e.g. `faba count`.
pub const PRODUCER: &str = "producer";

/// What the rows hold, e.g. [`GENE_COUNT`].
pub const CONTENT: &str = "content";

/// [`CONTENT`] of a gene count matrix, its rows named
/// `{gene}/count/{spliced|unspliced}`.
pub const GENE_COUNT: &str = "gene_count";

/// The metadata all of `metas` agree on: each key that every one has, with
/// the same value. Used where backends are merged, so a merge of samples
/// keeps no [`SAMPLE`] while a merge of one sample's parts keeps it.
pub fn common_metadata(metas: &[Metadata]) -> Metadata {
    let Some((first, rest)) = metas.split_first() else {
        return Metadata::new();
    };
    first
        .iter()
        .filter(|(k, v)| rest.iter().all(|m| m.get(*k) == Some(*v)))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

#[cfg(test)]
mod tests;
