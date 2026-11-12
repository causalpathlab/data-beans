#![allow(dead_code)]

use super::*;

impl SparseIoVec {
    /// Add a backend's columns to the vector.
    ///
    /// * `data`: `Arc` to the backend [`SparseData`].
    /// * `data_name`: under [`ColumnAlignment::Disjoint`], appended as the
    ///   `@<data_name>` display disambiguator for barcodes shared across
    ///   files. **Ignored under [`ColumnAlignment::Union`]**, where cells
    ///   glue by raw barcode — use [`push_with_barcode_suffix`] to attach a
    ///   per-cell sample tag that participates in the Union merge.
    ///
    /// [`push_with_barcode_suffix`]: Self::push_with_barcode_suffix
    pub fn push(
        &mut self,
        data: Arc<SparseData>,
        data_name: Option<Box<str>>,
    ) -> anyhow::Result<()> {
        self.push_with_barcode_suffix(data, data_name, None)
    }

    /// Like [`push`](Self::push) but, under [`ColumnAlignment::Union`], tags
    /// every barcode of this backend with `{COLUMN_SEP}{barcode_suffix}`
    /// **before** the canonical-merge step. Two backends that share a barcode
    /// merge into one global cell only if they carry the SAME suffix, so
    /// callers can encode per-file sample identity (e.g. `rep1_wt`):
    /// same-sample modalities merge, different samples stay distinct. The
    /// tagged name also becomes the displayed column name. `None` reproduces
    /// [`push`](Self::push) exactly. The `data_name` Disjoint disambiguator is
    /// orthogonal: it still applies under `Disjoint`, and `barcode_suffix`
    /// only under `Union` (the two alignments are mutually exclusive).
    pub fn push_with_barcode_suffix(
        &mut self,
        data: Arc<SparseData>,
        data_name: Option<Box<str>>,
        barcode_suffix: Option<&str>,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.column_scales.is_none(),
            "push after set_column_scales: the new columns would have no scale"
        );
        let Some(ncol_data) = data.num_columns() else {
            return Err(anyhow::anyhow!("data file has no columns"));
        };
        debug_assert_eq!(self.col_to_data.len(), self.offset);
        let didx = self.data_vec.len();
        let didx_u32: u32 = didx
            .try_into()
            .map_err(|_| anyhow::anyhow!("backend count overflows u32"))?;

        // Everything that can fail runs before the first change to `self`, so
        // a refused backend leaves the vector exactly as it was and the next
        // push starts from a consistent state.
        let raw_col_names = data.column_names()?;
        debug_assert_eq!(raw_col_names.len(), ncol_data);
        anyhow::ensure!(u32::try_from(ncol_data).is_ok(), "local col overflows u32");
        // `Union` keys its global columns by `u32`.
        anyhow::ensure!(
            self.column_alignment == ColumnAlignment::Disjoint
                || u32::try_from(self.offset + ncol_data).is_ok(),
            "global col overflows u32"
        );

        // Per-backend modality namespacing: `/{suffix}` goes after
        // canonicalization so the gene/locus rule still applies to the
        // bare name and only the modality tag distinguishes the row.
        let row_suffix = match self.per_backend_row_suffix.as_ref() {
            Some(suffixes) => Some(suffixes.get(didx).ok_or_else(|| {
                anyhow::anyhow!(
                    "per_backend_row_suffix has {} entries but backend index is {}",
                    suffixes.len(),
                    didx,
                )
            })?),
            None => None,
        };
        let row_keys: Vec<Box<str>> = data
            .row_names()?
            .iter()
            .map(|row| {
                let key: Box<str> = match self.row_canonicalizer.as_ref() {
                    Some(canon) => canon(row),
                    None => row.clone(),
                };
                match row_suffix {
                    Some(suffix) => format!("{key}/{suffix}").into_boxed_str(),
                    None => key,
                }
            })
            .collect();

        // `Union` only: each column's (merge key, displayed name), refused
        // when two columns of this backend share a key — that would put two
        // entries from the same backend on one global col.
        let union_keys: Vec<(Box<str>, Box<str>)> = match self.column_alignment {
            ColumnAlignment::Disjoint => Vec::new(),
            ColumnAlignment::Union => {
                let mut seen_in_this_push: HashMap<Box<str>, usize> =
                    HashMap::with_capacity_and_hasher(ncol_data, Default::default());
                let mut keys = Vec::with_capacity(ncol_data);
                for (loc, raw) in raw_col_names.iter().enumerate() {
                    // Tag the barcode with the per-file suffix (sample id)
                    // before canonicalization so the merge key — and the
                    // displayed name — carry sample identity. Same suffix
                    // across backends ⇒ merge; different ⇒ distinct cells.
                    let tagged: Box<str> = match barcode_suffix {
                        Some(s) => format!("{raw}{COLUMN_SEP}{s}").into_boxed_str(),
                        None => raw.clone(),
                    };
                    let canon: Box<str> = match self.column_canonicalizer.as_ref() {
                        Some(c) => c(&tagged),
                        None => tagged.clone(),
                    };
                    if let Some(prev_loc) = seen_in_this_push.insert(canon.clone(), loc) {
                        return Err(anyhow::anyhow!(
                            "ColumnAlignment::Union: backend {} has duplicate canonical \
                             barcode `{}` (local cols {} and {}) — Union cannot fold a \
                             cell with itself within one backend",
                            didx,
                            canon,
                            prev_loc,
                            loc,
                        ));
                    }
                    keys.push((canon, tagged));
                }
                keys
            }
        };

        // `Disjoint`: every push appends fresh global columns + the
        // `@<basename>` disambiguator. `Union`: match each pushed
        // barcode against the existing global cell pool by canonical
        // name so matched barcodes share a global column across
        // backends (reads merge their nonzeros into one output col).
        let data_to_cells_loc_to_glob: Vec<usize> = match self.column_alignment {
            ColumnAlignment::Disjoint => {
                let local_to_glob: Vec<usize> = (self.offset..self.offset + ncol_data).collect();
                self.col_to_data
                    .extend((0..ncol_data as u32).map(|local_col| {
                        vec![BackendLocation {
                            backend: didx_u32,
                            local_col,
                        }]
                    }));
                let data_tag = match data_name.as_deref() {
                    Some(x) => COLUMN_SEP.to_string() + x,
                    None => String::new(),
                };
                self.column_names_with_data_tag.extend(
                    raw_col_names
                        .iter()
                        .map(|x| (x.to_string() + data_tag.as_str()).into_boxed_str()),
                );
                self.offset += ncol_data;
                local_to_glob
            }
            ColumnAlignment::Union => {
                let mut local_to_glob = Vec::with_capacity(ncol_data);
                for (loc, (canon, tagged)) in union_keys.into_iter().enumerate() {
                    let here = BackendLocation {
                        backend: didx_u32,
                        local_col: loc as u32,
                    };
                    let glob = match self.col_name_position.get(&canon).copied() {
                        Some(g) => {
                            self.col_to_data[g as usize].push(here);
                            g as usize
                        }
                        None => {
                            let new_g = self.offset;
                            self.col_name_position.insert(canon, new_g as u32);
                            self.col_to_data.push(vec![here]);
                            // Tagged barcode (`raw` when no suffix) becomes
                            // the displayed name; subsequent backends
                            // contributing to this cell don't relabel it.
                            self.column_names_with_data_tag.push(tagged);
                            self.offset += 1;
                            new_g
                        }
                    };
                    local_to_glob.push(glob);
                }
                local_to_glob
            }
        };

        // `data_to_cols[didx][loc] = glob`. Stored separately from
        // `col_to_data` because callers (e.g. `rows_triplets`) need the
        // forward map per backend.
        self.data_to_cols.insert(didx, data_to_cells_loc_to_glob);

        let mut local_to_global = Vec::with_capacity(row_keys.len());
        for key in row_keys {
            let glob_row = match self.row_name_position.get(&key) {
                Some(&g) => g,
                None => {
                    let next_global = self.row_names_by_global.len();
                    self.row_name_position.insert(key.clone(), next_global);
                    self.row_names_by_global.push(key);
                    self.row_count_by_global.push(0);
                    next_global
                }
            };
            local_to_global.push(glob_row);
        }
        // Count per-dataset *presence*, not per-local-row hits: when the
        // row canonicalizer collapses several local rows in one file to
        // the same global key, this dataset should still contribute
        // exactly 1 to that global's count. Otherwise the
        // `RowAlignment::Intersect` admit-rule (`count >= n_datasets`)
        // silently excludes every collapsed row.
        let mut counted: HashSet<usize> = HashSet::default();
        for &g in &local_to_global {
            if counted.insert(g) {
                self.row_count_by_global[g] += 1;
            }
        }
        // Flag canonicalizer-induced intra-file row merges so the read
        // paths can sum duplicate (row, col) entries instead of emitting
        // them twice (which breaks `from_nonzero_triplets` strictness).
        let has_intra_merges = counted.len() != local_to_global.len();
        self.data_has_intra_row_merges.push(has_intra_merges);
        let mut g2l: Vec<(usize, usize)> = local_to_global
            .iter()
            .enumerate()
            .map(|(l, &g)| (g, l))
            .collect();
        g2l.sort_unstable();
        self.data_global_to_local_row.push(g2l);
        self.data_local_to_global_row.push(local_to_global);

        self.data_vec.push(data.clone());

        self.cached_num_columns = self.offset;
        self.recompute_row_mapping();

        info!(
            "Added {} columns ({} total); row {} = {}",
            ncol_data,
            self.offset,
            match self.row_alignment {
                RowAlignment::Intersect => "intersection",
                RowAlignment::Union => "union",
            },
            self.cached_num_rows
        );
        Ok(())
    }

    /// Recompute the global → compact row mapping under the current
    /// [`RowAlignment`] mode. Intersection keeps only rows present in
    /// every backend; Union keeps every row that any backend contains.
    /// Surviving rows get a compact index in raw-global order;
    /// everything else maps to `None`.
    fn recompute_row_mapping(&mut self) {
        let n_datasets = self.data_local_to_global_row.len();
        let n_global = self.row_names_by_global.len();
        let min_count = match self.row_alignment {
            RowAlignment::Intersect => n_datasets,
            RowAlignment::Union => 1,
        };

        self.global_to_compact_row.clear();
        self.global_to_compact_row.resize(n_global, None);
        self.compact_to_global_row.clear();

        let mut next_compact = 0usize;
        for g in 0..n_global {
            if self.row_count_by_global[g] >= min_count {
                self.global_to_compact_row[g] = Some(next_compact);
                self.compact_to_global_row.push(g);
                next_compact += 1;
            }
        }
        self.cached_num_rows = next_compact;
    }

    pub fn num_columns_by_data(&self) -> anyhow::Result<Vec<usize>> {
        Ok(self
            .data_vec
            .iter()
            .map(|d| d.num_columns().unwrap_or(0_usize))
            .collect())
    }

    pub fn remove_backend_file(&mut self) -> anyhow::Result<()> {
        for dat in self.data_vec.iter() {
            dat.remove_backend_file()?;
        }
        Ok(())
    }
}
