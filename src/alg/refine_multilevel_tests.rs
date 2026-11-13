use super::*;

#[test]
fn test_build_candidate_sets_fallback() {
    // pbsamp 0 has siblings [0,1] but BBKNN lands only on a non-sibling group.
    let siblings = vec![vec![0usize, 1], vec![0, 1]];
    let bbknn = vec![vec![1usize], vec![0]];
    let pbsamp_to_group = vec![0usize, 1];
    // Neighbor of sc0 is sc1 (group 1) → intersection {1}.
    // sc0 is in group 0; intersection lacks 0 so 0 is appended.
    let cand = build_candidate_sets(&siblings, &bbknn, &pbsamp_to_group, None);
    assert_eq!(cand[0], vec![0, 1]);
    assert_eq!(cand[1], vec![0, 1]);
}

#[test]
fn child_offset_within_parent_is_bounded_per_parent() {
    // Two parents; child codes nest in them. Offsets are local, first-seen
    // indices within each parent (so they're bounded by children-per-parent,
    // not by the global child-code count).
    let parent = vec![0usize, 0, 0, 1, 1, 1];
    let child = vec![10usize, 10, 11, 22, 23, 22];
    let off = child_offset_within_parent(&child, &parent);
    // parent 0: 10→0, 11→1 ; parent 1: 22→0, 23→1
    assert_eq!(off, vec![0, 0, 1, 0, 1, 0]);

    // Subdividing a *scrambled* refined parent by the offset stays bounded
    // by #parent_groups × max_children_per_parent — no cross-product blowup.
    let refined_parent = vec![3usize, 3, 3, 8, 8, 8];
    let (_lab, k) = project_to_refinement(&off, &refined_parent);
    assert!(k <= 4, "expected ≤ 2 parents × 2 offsets, got {k}");
}

#[test]
fn test_project_to_refinement_splits_straddling_groups() {
    // child group 0 straddles parents A=0 and B=1; group 1 sits wholly
    // under parent B. Re-projection must split group 0 into two and
    // produce a strict refinement of `parent`.
    let child = vec![0usize, 0, 0, 1, 1];
    let parent = vec![0usize, 0, 1, 1, 1];
    let (reproj, k) = project_to_refinement(&child, &parent);

    // 3 distinct (child, parent) pairs: (0,0), (0,1), (1,1).
    assert_eq!(k, 3);
    // Entities 0,1 → (0,0); entity 2 → (0,1); entities 3,4 → (1,1).
    assert_eq!(reproj[0], reproj[1]);
    assert_ne!(reproj[0], reproj[2]);
    assert_eq!(reproj[3], reproj[4]);
    assert_ne!(reproj[2], reproj[3]);

    // Strict refinement: every reprojected group maps to one parent.
    let mut group_parent = std::collections::HashMap::new();
    for (&g, &p) in reproj.iter().zip(parent.iter()) {
        assert_eq!(*group_parent.entry(g).or_insert(p), p);
    }
}

/// With strata, a pb-sample only gets groups of its own stratum: neither the
/// sibling fallback (no BBKNN neighbors) nor a neighbor's group can take it
/// across, and a group that already mixes strata is nobody's candidate.
#[test]
fn candidate_sets_stay_within_the_stratum() {
    // Groups: 0 = {pb 0} (stratum 3), 1 = {pb 1} (stratum 0),
    // 2 = {pb 2} (stratum 0), 3 = {pb 3, pb 4} (strata 0 and 3, mixed).
    let labels = vec![0usize, 1, 2, 3, 3];
    let strata = vec![3usize, 0, 0, 0, 3];
    let siblings = vec![vec![0usize, 1, 2, 3]; 5];
    // pb 0 has no cross-batch match of its stratum; pb 1's match sits in group 2.
    let bbknn = vec![vec![], vec![2usize], vec![1], vec![1], vec![]];

    let unstratified = build_candidate_sets(&siblings, &bbknn, &labels, None);
    assert_eq!(unstratified[0], vec![0, 1, 2, 3]);

    let cand = build_candidate_sets(&siblings, &bbknn, &labels, Some(&strata));
    assert_eq!(cand[0], vec![0], "private stratum stays put");
    assert_eq!(cand[1], vec![1, 2]);
    assert_eq!(cand[2], vec![1, 2]);
    // Members of the mixed group may leave for a pure group of their stratum
    // or stay.
    assert_eq!(cand[3], vec![1, 3]);
    assert_eq!(cand[4], vec![0, 3]);
}

/// Two batches of pb-samples over four well-separated states, two pb-samples
/// (three cells each) per state and batch.
struct Toy {
    layout: PbSampleLayout,
    gene_sums: GeneSums,
    pb_sample_to_cells: Vec<Vec<usize>>,
    batch_knn: Vec<legume_numeric::matrix::knn_match::ColumnDict<usize>>,
    state: Vec<usize>,
}

const TOY_STATES: usize = 4;
const TOY_GENES: usize = 8;

fn toy() -> Toy {
    toy_with(60.0, 1.0)
}

/// The toy with `marker` counts on a state's two marker genes, and the
/// even genes of batch 1 multiplied by `batch_factor`.
fn toy_with(marker: f32, batch_factor: f32) -> Toy {
    use legume_numeric::matrix::knn_match::ColumnDict;
    use nalgebra::DMatrix;
    let num_pb = 2 * TOY_STATES * 2;
    let mut centroids = DMatrix::<f32>::zeros(TOY_STATES, num_pb);
    let mut pb_sample_to_batch = Vec::new();
    let mut state = Vec::new();
    let mut cell_to_pbsamp = Vec::new();
    let mut pb_sample_to_cells = Vec::new();
    let mut gene_sums: GeneSums = Vec::new();
    let mut per_batch: Vec<Vec<(usize, Vec<f32>)>> = vec![Vec::new(); 2];
    for (batch, cells_of_batch) in per_batch.iter_mut().enumerate() {
        for k in 0..TOY_STATES {
            for rep in 0..2 {
                let pb = pb_sample_to_batch.len();
                let mut cells = Vec::new();
                for i in 0..3 {
                    let c = cell_to_pbsamp.len();
                    let mut feat = vec![0f32; TOY_STATES];
                    feat[k] = 1.0 + 0.01 * (rep * 3 + i) as f32;
                    centroids[(k, pb)] += feat[k] / 3.0;
                    cells_of_batch.push((c, feat));
                    cell_to_pbsamp.push(pb);
                    cells.push(c);
                }
                pb_sample_to_cells.push(cells);
                pb_sample_to_batch.push(batch);
                state.push(k);
                // Two marker genes per state over a flat background.
                gene_sums.push(
                    (0..TOY_GENES)
                        .map(|g| {
                            let v = if g / 2 == k { marker } else { 3.0 } + rep as f32;
                            let factor = if batch == 1 && g % 2 == 0 {
                                batch_factor
                            } else {
                                1.0
                            };
                            (g, v * factor)
                        })
                        .collect(),
                );
            }
        }
    }
    let batch_knn = per_batch
        .into_iter()
        .map(|cells| {
            let mut mat = DMatrix::<f32>::zeros(TOY_STATES, cells.len());
            let mut names = Vec::new();
            for (col, (c, feat)) in cells.into_iter().enumerate() {
                mat.column_mut(col).copy_from_slice(&feat);
                names.push(c);
            }
            ColumnDict::<usize>::from_dmatrix(mat, names)
        })
        .collect();
    let layout = PbSampleLayout {
        centroids,
        cell_counts: vec![3.0; num_pb],
        pb_sample_to_batch,
        pb_sample_to_group: (0..num_pb).collect(),
        bg_to_pbsamp: Default::default(),
        cell_to_pbsamp,
        singleton_col: vec![None; num_pb],
        bulk_batches: Vec::new(),
        pb_sample_to_stratum: None,
    };
    Toy {
        layout,
        gene_sums,
        pb_sample_to_cells,
        batch_knn,
        state,
    }
}

fn toy_params() -> RefineParams {
    RefineParams {
        num_gibbs: 5,
        num_greedy: 3,
        seed: 7,
        ..Default::default()
    }
}

/// Handing the refinement the pooled match table gives the assignment it
/// would have computed from the lookups itself.
#[test]
fn refinement_with_a_given_match_table_is_the_same_refinement() {
    let t = toy();
    // One finest group per state, with a pb-sample of state 0 and one of
    // state 1 exchanged; states 0 and 1 share a parent, as do 2 and 3.
    let num_pb = t.state.len();
    let mut finest = t.state.clone();
    finest.swap(1, 3);
    assert_ne!(finest, t.state);
    let coarse: Vec<usize> = finest.iter().map(|&g| g / 2).collect();
    let levels = vec![finest, coarse];
    let inputs = RefineInputs {
        layout: &t.layout,
        gene_sums: &t.gene_sums,
        num_genes: TOY_GENES,
        pb_sample_to_cells: &t.pb_sample_to_cells,
        batch_knn_lookup: &t.batch_knn,
        k_per_batch: 3,
        initial_sc_to_group_per_level: &levels,
        reproject_offsets_per_level: &[],
    };
    let own = refine_assignments(&inputs, &toy_params()).unwrap();
    let matches = per_batch_sc_neighbors(&t.layout, &t.batch_knn, 3, None).unwrap();
    assert!(matches.iter().all(|m| !m.is_empty()));
    let given = refine_assignments_with_matches(&inputs, &toy_params(), Some(&matches)).unwrap();
    assert_eq!(own.pbsamp_to_group, given.pbsamp_to_group);
    assert_eq!(own.num_groups_per_level, given.num_groups_per_level);
    // The sweeps put the two pb-samples back: the finest groups are the
    // states (the result is relabelled densely, in order of appearance).
    assert_eq!(num_pb, own.pbsamp_to_group[0].len());
    assert_eq!(own.pbsamp_to_group[0], compact_labels(&t.state).0);
}

/// A stratum present in one batch only has no cross-batch match, so its
/// pb-samples fall back to their siblings. Refinement must still leave every
/// group of the finest level within one stratum.
#[test]
fn refinement_never_merges_a_private_stratum() {
    let mut t = toy();
    let num_pb = t.state.len();
    // Batch 0's two pb-samples of state 0 form a private stratum whose
    // profile is that of the stratum-0 pb-samples of the same state.
    let strata: Vec<usize> = (0..num_pb)
        .map(|p| usize::from(t.layout.pb_sample_to_batch[p] == 0 && t.state[p] == 0))
        .collect();
    t.layout.pb_sample_to_stratum = Some(strata.clone());

    // Start stratum-pure: one finest group per (state, stratum), all groups
    // siblings under one coarse parent.
    let finest: Vec<usize> = (0..num_pb).map(|p| t.state[p] * 2 + strata[p]).collect();
    let levels = vec![finest, vec![0; num_pb]];
    let inputs = RefineInputs {
        layout: &t.layout,
        gene_sums: &t.gene_sums,
        num_genes: TOY_GENES,
        pb_sample_to_cells: &t.pb_sample_to_cells,
        batch_knn_lookup: &t.batch_knn,
        k_per_batch: 3,
        initial_sc_to_group_per_level: &levels,
        reproject_offsets_per_level: &[],
    };
    for seed in 0..8 {
        let params = RefineParams {
            seed,
            ..toy_params()
        };
        let refined = refine_assignments(&inputs, &params).unwrap();
        let finest = &refined.pbsamp_to_group[0];
        for p in 0..num_pb {
            for q in 0..num_pb {
                assert!(
                    finest[p] != finest[q] || strata[p] == strata[q],
                    "seed {seed}: pb-samples {p} and {q} of different strata share group {}",
                    finest[p]
                );
            }
        }
    }
}

/// Batch 1 carries a per-gene factor larger than the difference between
/// two states, and the finest groups start as one batch each, two states
/// mixed. Refinement sorts the pb-samples by state: every group ends with
/// both batches and one state.
#[test]
fn refinement_groups_by_state_across_batches() {
    let t = toy_with(60.0, 8.0);
    let num_pb = t.state.len();
    let batch = &t.layout.pb_sample_to_batch;
    // States 0, 1 under one parent and 2, 3 under the other; below each
    // parent, one group per batch.
    let coarse: Vec<usize> = t.state.iter().map(|&k| k / 2).collect();
    let finest: Vec<usize> = (0..num_pb).map(|p| 2 * coarse[p] + batch[p]).collect();
    let levels = vec![finest, coarse];
    let inputs = RefineInputs {
        layout: &t.layout,
        gene_sums: &t.gene_sums,
        num_genes: TOY_GENES,
        pb_sample_to_cells: &t.pb_sample_to_cells,
        batch_knn_lookup: &t.batch_knn,
        k_per_batch: 3,
        initial_sc_to_group_per_level: &levels,
        reproject_offsets_per_level: &[],
    };
    // Every gene of the toy marks a state, which the NB trend reads as
    // overdispersion and weights away; the counts are scored as they are.
    let params = RefineParams {
        feature_weighting: crate::alg::dc_poisson::FeatureWeighting::None,
        ..toy_params()
    };
    let refined = refine_assignments(&inputs, &params).unwrap();
    let groups = &refined.pbsamp_to_group[0];
    assert_eq!(refined.num_groups_per_level[0], TOY_STATES);
    for g in 0..TOY_STATES {
        let members: Vec<usize> = (0..num_pb).filter(|&p| groups[p] == g).collect();
        assert_eq!(members.len(), 4, "group {g}: {members:?}");
        assert!(
            members.iter().all(|&p| t.state[p] == t.state[members[0]]),
            "group {g} mixes states: {members:?}"
        );
        assert_eq!(
            members.iter().filter(|&&p| batch[p] == 0).count(),
            2,
            "group {g} is not half of each batch: {members:?}"
        );
    }
}
