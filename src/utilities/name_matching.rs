use crate::sparse_io::ROW_SEP;
use genomic_data::coordinates::{import_interval, is_locus, locus_key};
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;

/// Make duplicate names unique by appending `-1`, `-2`, etc. to repeated entries.
/// Similar to scanpy's `var_names_make_unique()`.
pub fn make_names_unique(names: &mut [Box<str>]) -> usize {
    let mut counts: HashMap<Box<str>, usize> = HashMap::default();
    let mut num_duped = 0usize;
    for name in names.iter_mut() {
        if let Some(count) = counts.get_mut(name.as_ref()) {
            if *count == 1 {
                num_duped += 1;
            }
            *name = format!("{}-{}", name, count).into_boxed_str();
            *count += 1;
        } else {
            counts.insert(name.clone(), 1);
        }
    }
    if num_duped > 0 {
        log::warn!(
            "{} names had duplicates and were made unique with -N suffixes",
            num_duped
        );
    }
    num_duped
}

/// The composite name of one row: `id{ROW_SEP}name`, or the ID alone when
/// the name is empty or already equals it (e.g. 10x ATAC peaks, where both
/// `features/id` and `features/name` are `chr1:1000-2000`).
pub fn id_name(id: &str, name: &str) -> Box<str> {
    if name.is_empty() || name == id {
        id.into()
    } else {
        format!("{id}{ROW_SEP}{name}").into_boxed_str()
    }
}

/// [`id_name`] of each row.
pub fn compose_id_name(ids: Vec<Box<str>>, names: Vec<Box<str>>) -> Vec<Box<str>> {
    ids.iter()
        .zip(&names)
        .map(|(id, name)| id_name(id, name))
        .collect()
}

/// Import boundary for feature rows: peak rows get their id and name
/// rewritten in the colon locus form, whatever spelling the producer used
/// (`chr1-100-200` and `chr1_100_200` become `chr1:100-200`). With feature
/// types, a row is a peak when its type names peaks or ATAC; without them,
/// the file is a peak list only when every id reads as an interval. A gene
/// id that happens to end in two numbers is therefore left alone. Returns
/// how many rows were rewritten.
pub fn colon_peak_names(
    ids: &mut [Box<str>],
    names: &mut [Box<str>],
    types: Option<&[Box<str>]>,
) -> usize {
    let is_peak = |i: usize| match types {
        Some(types) => {
            contains_ignore_ascii_case(&types[i], "peak")
                || contains_ignore_ascii_case(&types[i], "atac")
        }
        None => true,
    };
    if types.is_none() {
        let n_read = ids
            .iter()
            .filter(|id| import_interval(id).is_some())
            .count();
        if n_read < ids.len() {
            if n_read * 2 >= ids.len() {
                log::info!(
                    "{n_read} of {} untyped rows read as intervals, but not all; \
                     the names are kept as written",
                    ids.len()
                );
            }
            return 0;
        }
    }
    let mut n = 0;
    for (i, (id, name)) in ids.iter_mut().zip(names.iter_mut()).enumerate() {
        if !is_peak(i) {
            continue;
        }
        let mut changed = false;
        for s in [id, name] {
            if let Some(l) = import_interval(s) {
                let colon = l.to_string().into_boxed_str();
                changed |= *s != colon;
                *s = colon;
            }
        }
        n += usize::from(changed);
    }
    n
}

/// Inverse of [`compose_id_name`]: split a composite `id{ROW_SEP}name` display
/// name back into `(id, name)` on the first `ROW_SEP`. When there is no
/// separator (a bare symbol, or an id-only composite where name was empty or
/// equalled id) both parts are the whole string, so a 10x `features.tsv` still
/// gets a non-empty gene name.
pub fn split_id_name(composite: &str) -> (&str, &str) {
    match composite.split_once(ROW_SEP) {
        Some((id, name)) if !name.is_empty() => (id, name),
        _ => (composite, composite),
    }
}

/// Comma-separated case-insensitive substring filter, parsed once and matched
/// many times. Used by `--select-row-type` / `--remove-row-type` /
/// `--hto-row-type` so callers can pass e.g. `"gene,peak"` to match either
/// "Gene Expression" or "Peaks".
pub struct RowTypeFilter {
    patterns: Vec<Box<str>>,
}

impl RowTypeFilter {
    pub fn parse(s: &str) -> Self {
        let patterns = s
            .split(',')
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .map(|p| p.to_ascii_lowercase().into_boxed_str())
            .collect();
        Self { patterns }
    }

    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// True if any pattern is an ASCII-case-insensitive substring of `s`.
    /// Bytewise scan — does not allocate, so callers can pass row types
    /// straight from the backend without an intermediate lowercase copy.
    pub fn matches(&self, s: &str) -> bool {
        self.patterns
            .iter()
            .any(|p| contains_ignore_ascii_case(s, p))
    }
}

/// Bytewise case-insensitive substring search. ASCII only; non-ASCII bytes
/// compare verbatim. Allocation-free.
pub fn contains_ignore_ascii_case(haystack: &str, needle: &str) -> bool {
    let n = needle.len();
    if n == 0 {
        return true;
    }
    let h = haystack.as_bytes();
    if h.len() < n {
        return false;
    }
    h.windows(n)
        .any(|w| w.eq_ignore_ascii_case(needle.as_bytes()))
}

/// Return indices of rows whose type passes select/remove filtering.
/// - `select`: comma-separated patterns; row passes if any pattern is a
///   case-insensitive substring of the row type. Empty keeps all rows.
/// - `remove`: comma-separated patterns; row is dropped if any pattern matches.
pub fn filter_row_indices_by_type(
    row_types: &[Box<str>],
    select: &str,
    remove: &str,
) -> Vec<usize> {
    let sel = RowTypeFilter::parse(select);
    let rem = RowTypeFilter::parse(remove);
    if sel.is_empty() && rem.is_empty() {
        return (0..row_types.len()).collect();
    }
    row_types
        .iter()
        .enumerate()
        .filter_map(|(i, x)| {
            let selected = sel.is_empty() || sel.matches(x);
            let removed = !rem.is_empty() && rem.matches(x);
            if selected && !removed {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}

/// Flexible gene name matching (case-insensitive, underscore-delimited)
/// Returns true if `query` matches `target` with these rules:
/// - Exact match (case-insensitive)
/// - Suffix match: target ends with `_query`
/// - Prefix match: target starts with `query_`
/// - Segment match: target contains `_query_`
///
/// Example: "CD8A" matches "ENSG00000153563_CD8A", "CD8A_variant1", "chr1_CD8A_isoform2"
#[allow(dead_code)]
pub fn flexible_name_match(query: &str, target: &str) -> bool {
    let q = query.to_lowercase();
    let t = target.to_lowercase();
    t == q
        || t.ends_with(&format!("_{}", q))
        || t.starts_with(&format!("{}_", q))
        || t.contains(&format!("_{}_", q))
}

/// Heuristic: a lower-cased Ensembl-style stable id (`ensg…`, `ensmusg…`,
/// `enst…`). Used to index/look up the *leading* id segment of an
/// `ENSG…_SYMBOL` name so bare-`ENSG` and `ENSG_SYMBOL` forms reconcile both
/// ways without a linear scan.
fn is_ensembl_id(s: &str) -> bool {
    s.len() >= 8 && s.starts_with("ens") && s.bytes().any(|b| b.is_ascii_digit())
}

/// Curated HGNC **old symbol → current symbol** renames, lower-cased.
///
/// A symbol match is exact, so a marker panel written against an older HGNC release
/// silently loses every gene HGNC has since renamed — the gene is in the matrix under its
/// new name, but the panel asks for the old one and gets nothing back. That is invisible in
/// the output: the type just scores on fewer genes (or is dropped entirely). This table is
/// what closes the gap.
///
/// It is **curated, not exhaustive** — the families that actually recur in single-cell
/// marker panels (histones, the `MARCH`/`SEPT` families that Excel also mangles into dates,
/// the selenoproteins, and the well-known one-off renames). Systematic families are handled
/// by rule in [`alias_candidates`] rather than enumerated here. Entries are one-directional
/// in this table but matched **both ways** at lookup, so it does not matter whether the
/// matrix or the panel is the one carrying the old name.
static HGNC_RENAMES: &[(&str, &str)] = &[
    // Histones — HGNC's 2019 systematic renaming; heavily used as cell-cycle / S-phase
    // markers, so a stale panel loses much of its S-phase signature.
    ("h1f0", "h1-0"),
    ("h1fx", "h1-10"),
    ("hist1h1b", "h1-5"),
    ("hist1h1c", "h1-2"),
    ("hist1h1d", "h1-3"),
    ("hist1h1e", "h1-4"),
    ("hist1h2ac", "h2ac6"),
    ("hist1h2bk", "h2bc12"),
    ("hist1h4c", "h4c3"),
    ("hist2h2be", "h2bc21"),
    ("hist3h2a", "h2ac25"),
    ("h2afx", "h2ax"),
    ("h2afv", "h2az2"),
    ("h2afz", "h2az1"),
    ("h2afy", "macroh2a1"),
    ("h3f3a", "h3-3a"),
    ("h3f3b", "h3-3b"),
    // Mitochondrial amidoxime-reducing components (note: NOT the MARCH family below).
    ("marc1", "mtarc1"),
    ("marc2", "mtarc2"),
    // Selenoproteins.
    ("sepp1", "selenop"),
    ("selt", "selenot"),
    ("sepw1", "selenow"),
    // One-off renames common in immune / proliferation panels.
    ("fam129a", "niban1"),
    ("fam129b", "niban2"),
    ("fam129c", "niban3"),
    ("rarres3", "plaat4"),
    ("fyb", "fyb1"),
    ("cd97", "adgre5"),
    ("gpr56", "adgrg1"),
    ("kiaa0101", "pclaf"),
    ("c10orf54", "vsir"),
    ("tmem66", "saraf"),
    ("atpif1", "atp5if1"),
    ("fam46c", "tent5c"),
    ("whsc1", "nsd2"),
];

/// `HGNC_RENAMES` as a lookup: old → new (`fwd`) and new → old (`rev`).
///
/// Kept as two maps rather than one seeded in both directions. The table's key sets happen to
/// be disjoint today, so one map would give identical answers — but the moment someone adds a
/// *chained* rename (`A→B` alongside an existing `B→C`), a single map has two entries for `B`
/// and silently keeps whichever was inserted last. Two maps cannot lose that way.
struct RenameMaps {
    fwd: HashMap<&'static str, &'static str>,
    rev: HashMap<&'static str, &'static str>,
}

/// The lazily-built rename lookups. Built once; every `match_gene` miss consults them.
fn rename_maps() -> &'static RenameMaps {
    static MAPS: std::sync::OnceLock<RenameMaps> = std::sync::OnceLock::new();
    MAPS.get_or_init(|| RenameMaps {
        fwd: HGNC_RENAMES.iter().copied().collect(),
        rev: HGNC_RENAMES.iter().map(|&(o, n)| (n, o)).collect(),
    })
}

/// The numeric suffix of a `{prefix}{n}` symbol (`numeric_suffix("march12", "march") == 12`),
/// or `None` if `sym` does not have exactly that shape.
fn numeric_suffix(sym: &str, prefix: &str) -> Option<u32> {
    sym.strip_prefix(prefix)
        .filter(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|d| d.parse().ok())
}

/// Alternative HGNC symbols for `sym` (already lower-cased): the table above in both
/// directions, plus the two rule-based families whose members are too numerous to enumerate
/// and whose rename is purely mechanical — `MARCH{n}` ↔ `MARCHF{n}` (membrane-associated
/// ring-CH E3 ligases) and `SEPT{n}` ↔ `SEPTIN{n}` (septins). Both families were renamed
/// precisely because spreadsheets kept coercing them to dates, so panels in the wild carry
/// either form.
///
/// `MARCH1`/`MARC1` do not collide: `MARC1` is in the table (→ `MTARC1`) and the `march`
/// rule only fires on the literal `march` prefix.
fn alias_candidates(sym: &str) -> Vec<String> {
    let maps = rename_maps();
    let mut out = Vec::new();
    if let Some(&new) = maps.fwd.get(sym) {
        out.push(new.to_string());
    }
    if let Some(&old) = maps.rev.get(sym) {
        out.push(old.to_string());
    }
    for (old, new) in [("march", "marchf"), ("sept", "septin")] {
        if let Some(n) = numeric_suffix(sym, old) {
            out.push(format!("{new}{n}"));
        }
        if let Some(n) = numeric_suffix(sym, new) {
            out.push(format!("{old}{n}"));
        }
    }
    out
}

/// Pre-built index over a gene-name vocabulary for fast marker→row matching.
/// Resolves a query gene in tiers, returning the first matching row:
///   0. a locus query (`chr:start-end`) matches only a locus row, by its
///      locus key with the chromosome case kept, and stops here: loci never
///      go through the case-insensitive or fuzzy tiers below,
///   1. exact (case-insensitive) full-name match,
///   2. last `_`-segment symbol match (`CD8A` ↔ `ENSG…_CD8A`),
///   3. leading Ensembl-id segment match (`ENSG…` ↔ `ENSG…_CD8A`),
///   4. decompose a combined `ENSG…_SYMBOL` query and retry tiers 2–3 per part,
///   5. HGNC alias retry ([`alias_candidates`]: `HIST1H4C` ↔ `H4C3`, `MARCH2` ↔ `MARCHF2`),
///   6. fallback to the general [`flexible_name_match`] (prefix / `_x_` segment).
///
/// Tiers 1–5 are O(1) hash lookups; only the rare fallback scans the
/// vocabulary. Build once, match many — replaces the O(genes × markers)
/// `.position(flexible_name_match)` scan. Tier 1 preferring an exact match
/// over an earlier-indexed suffix match is the one intended refinement vs a
/// pure positional scan. Tiers 3–4 make HGNC / ENSG / `ENSG_HGNC` reconcile
/// in either direction (gene-set sources mix these conventions), and tier 5
/// does the same across HGNC *releases* — the matrix and the marker panel are
/// routinely built against different ones.
#[allow(dead_code)] // consumed by downstream crates (geu, senna), not the data-beans bin
pub struct GeneIndex {
    lowered: Vec<String>,
    exact: HashMap<String, usize>,
    symbol: HashMap<String, usize>,
    ensg: HashMap<String, usize>,
    /// Rows with a locus part ([`locus_part`]), by their name as written,
    /// then by locus key plus the rest of the name, then (for rows with a
    /// rest) by the bare locus key. Loci match only here, case kept.
    locus_raw: HashMap<Box<str>, usize>,
    locus: HashMap<Box<str>, usize>,
}

/// A name's locus part and the rest after it: the whole name when it is a
/// locus, its `/`-core (`chr1:1-2/count/spliced`), or the locus before an
/// `id{ROW_SEP}name` join (`chr1:1-2_GENE1`). `None` when no part is a locus.
fn locus_part(name: &str) -> Option<(&str, &str)> {
    if is_locus(name) {
        return Some((name, ""));
    }
    let core = name.split('/').next().unwrap_or(name);
    let cut = if is_locus(core) {
        core.len()
    } else {
        let colon = core.find(':')?;
        colon + 1 + core[colon + 1..].find(ROW_SEP)?
    };
    let (part, rest) = name.split_at(cut);
    is_locus(part).then_some((part, rest))
}

/// The key a name with a locus part is matched by: the locus key, then the
/// rest of the name as written.
fn locus_match_key(part: &str, rest: &str) -> Option<Box<str>> {
    let key = locus_key(part)?;
    Some(if rest.is_empty() {
        key
    } else {
        format!("{key}{rest}").into_boxed_str()
    })
}

#[allow(dead_code)] // consumed by downstream crates (geu, senna), not the data-beans bin
impl GeneIndex {
    /// Build the index from the dictionary's gene-name order. The first row
    /// wins on duplicate keys (matching positional-scan semantics).
    #[must_use]
    pub fn build(gene_names: &[Box<str>]) -> Self {
        // A whole-locus row gets an empty lowered name, which keeps it out
        // of every gene tier, the fallback scan included. A row with a
        // locus part plus a rest stays in the gene tiers for its full name.
        let lowered: Vec<String> = gene_names
            .par_iter()
            .map(|g| {
                if is_locus(g) {
                    String::new()
                } else {
                    g.to_lowercase()
                }
            })
            .collect();
        let mut locus_raw: HashMap<Box<str>, usize> = HashMap::default();
        let mut locus: HashMap<Box<str>, usize> = HashMap::default();
        let mut bare: Vec<(Box<str>, usize)> = Vec::new();
        for (i, g) in gene_names.iter().enumerate() {
            let Some((part, rest)) = locus_part(g) else {
                continue;
            };
            locus_raw.entry(g.clone()).or_insert(i);
            if let Some(key) = locus_match_key(part, rest) {
                locus.entry(key).or_insert(i);
            }
            if !rest.is_empty() {
                bare.extend(locus_key(part).map(|k| (k, i)));
            }
        }
        // A whole-locus row wins its key over a row that only starts with it.
        for (key, i) in bare {
            locus.entry(key).or_insert(i);
        }
        let mut exact: HashMap<String, usize> = HashMap::default();
        let mut symbol: HashMap<String, usize> = HashMap::default();
        let mut ensg: HashMap<String, usize> = HashMap::default();
        for (i, low) in lowered.iter().enumerate() {
            if low.is_empty() {
                continue;
            }
            exact.entry(low.clone()).or_insert(i);
            // Strip a faba-style aux suffix first (`SYMBOL/count/spliced` →
            // symbol is the leading `/`-segment), then an Ensembl-style prefix
            // (`ENSG…_CD8A` → symbol is the trailing `_`-segment). Handles
            // either convention, or both combined (`ENSG…_CD8A/count/spliced`).
            let core = low.split('/').next().unwrap_or(low);
            if let Some(sym) = core.rsplit('_').next() {
                symbol.entry(sym.to_string()).or_insert(i);
            }
            // Also index the *leading* segment when it is an Ensembl id, so a
            // bare `ENSG…` query resolves to an `ENSG…_SYMBOL` row (and back).
            let head = core.split('_').next().unwrap_or(core);
            if is_ensembl_id(head) {
                ensg.entry(head.to_string()).or_insert(i);
            }
        }
        Self {
            lowered,
            exact,
            symbol,
            ensg,
            locus_raw,
            locus,
        }
    }

    /// Row index for `gene`, or `None` if unmatched (tiers above).
    #[must_use]
    pub fn match_gene(&self, gene: &str) -> Option<usize> {
        // A name with a locus part matches only a row with one: the same
        // name as written first, then by locus key plus the rest. Strictly:
        // no case folding, aliasing or prefix fallback.
        if let Some((part, rest)) = locus_part(gene) {
            if let Some(&i) = self.locus_raw.get(gene) {
                return Some(i);
            }
            return locus_match_key(part, rest).and_then(|k| self.locus.get(&k).copied());
        }
        let gl = gene.to_lowercase();
        if let Some(&i) = self.exact.get(&gl) {
            return Some(i);
        }
        if let Some(&i) = self.symbol.get(&gl) {
            return Some(i);
        }
        if let Some(&i) = self.ensg.get(&gl) {
            return Some(i);
        }
        // Decompose a combined `ENSG…_SYMBOL[/aux]` query: match its trailing
        // symbol or leading Ensembl id against the per-part indices.
        let core = gl.split('/').next().unwrap_or(&gl);
        if let Some(sym) = core.rsplit('_').next() {
            if sym != gl {
                if let Some(&i) = self.symbol.get(sym) {
                    return Some(i);
                }
            }
        }
        let head = core.split('_').next().unwrap_or(core);
        if is_ensembl_id(head) {
            if let Some(&i) = self.ensg.get(head) {
                return Some(i);
            }
        }
        // HGNC alias retry: the query and the vocabulary can be built against different HGNC
        // releases (`HIST1H4C` in the panel, `H4C3` in the matrix, or the reverse). Retry the
        // exact/symbol tiers under each alternative symbol before falling back to the scan.
        let sym = core.rsplit('_').next().unwrap_or(core);
        for alias in alias_candidates(sym) {
            if let Some(&i) = self.exact.get(&alias).or_else(|| self.symbol.get(&alias)) {
                return Some(i);
            }
        }
        // Allocation-free flexible fallback: `flexible_name_match` re-lowercases
        // both sides and builds three `format!` needles *per comparison*, which
        // is catastrophic when scanning a 30k+ vocabulary for each of thousands
        // of unmatched gene-set genes. The vocabulary is already lowercased and
        // `gl` is lowercase, so build the needles once and scan with plain
        // byte-level `ends_with`/`starts_with`/`contains`.
        // (an exact `*t == gl` match is already handled by the `exact` tier above)
        let suffix = format!("_{gl}");
        let prefix = format!("{gl}_");
        let middle = format!("_{gl}_");
        self.lowered
            .iter()
            .position(|t| t.ends_with(&suffix) || t.starts_with(&prefix) || t.contains(&middle))
    }
}

/// Inverse-document-frequency marker weight `ln(C / df)`: a gene claimed by
/// all `C` types gets weight 0 (removed from scoring), a type-exclusive gene
/// the maximum `ln(C)`.
#[allow(dead_code)] // consumed by downstream crates (geu, senna), not the data-beans bin
#[must_use]
pub fn idf_weight(n_types: usize, df: usize) -> f32 {
    (n_types as f32 / df.max(1) as f32).ln()
}

/// Match names by substring queries and return matched indices and names
///
/// # Arguments
/// * `all_names` - All available names to search through
/// * `queries` - Substring queries to match against
/// * `entity_type` - Description of what's being matched (e.g., "column", "row") for error messages
///
/// # Returns
/// A tuple of (matched_indices, matched_names)
pub fn match_by_substring(
    all_names: &[Box<str>],
    queries: &[Box<str>],
    entity_type: &str,
) -> anyhow::Result<(Vec<usize>, Vec<Box<str>>)> {
    let mut matched_indices = Vec::new();

    for query in queries.iter() {
        for (idx, name) in all_names.iter().enumerate() {
            if name.contains(query.as_ref()) {
                matched_indices.push(idx);
            }
        }
    }

    if matched_indices.is_empty() {
        return Err(anyhow::anyhow!(
            "No {} names matched the provided queries",
            entity_type
        ));
    }

    let matched_names: Vec<Box<str>> = matched_indices
        .iter()
        .map(|&i| all_names[i].clone())
        .collect();

    Ok((matched_indices, matched_names))
}

#[cfg(test)]
mod tests;
