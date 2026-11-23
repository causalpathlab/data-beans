use super::*;

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| Box::from(*s)).collect()
}

#[test]
fn a_row_is_named_by_id_alone_when_its_name_adds_nothing() {
    assert_eq!(id_name("FID1", "GENE1"), Box::from("FID1_GENE1"));
    assert_eq!(id_name("FID2", "FID2"), Box::from("FID2"));
    assert_eq!(id_name("FID3", ""), Box::from("FID3"));
}

#[test]
fn peak_rows_import_in_colon_form_and_gene_rows_stay() {
    let mut ids = names(&["chr1-100-200", "GENE_1_2", "chrUn_CTG1v1_5_10", "chr2:1-9"]);
    let mut nm = names(&["chr1-100-200", "GENE_1_2", "", "chr2:1-9"]);
    let types = names(&["Peaks", "Gene Expression", "Peaks", "Peaks"]);
    colon_peak_names(&mut ids, &mut nm, Some(&types));
    assert_eq!(
        ids,
        names(&["chr1:100-200", "GENE_1_2", "chrUn_CTG1v1:5-10", "chr2:1-9"])
    );
    assert_eq!(nm, names(&["chr1:100-200", "GENE_1_2", "", "chr2:1-9"]));
}

#[test]
fn an_untyped_list_converts_only_when_every_row_is_an_interval() {
    let mut ids = names(&["chr1-100-200", "chr2_5_9"]);
    let mut nm = names(&["", ""]);
    assert_eq!(colon_peak_names(&mut ids, &mut nm, None), 2);
    assert_eq!(ids, names(&["chr1:100-200", "chr2:5-9"]));
    let mut ids = names(&["chr1-100-200", "GENE1"]);
    let mut nm = names(&["", ""]);
    assert_eq!(colon_peak_names(&mut ids, &mut nm, None), 0);
    assert_eq!(ids, names(&["chr1-100-200", "GENE1"]));
}

#[test]
fn a_suffixed_name_never_collides_with_a_name_in_the_list() {
    let mut v = names(&["ABC", "ABC-1", "ABC", "ABC", "X", "ABC-1"]);
    assert_eq!(make_names_unique(&mut v), 2);
    assert_eq!(
        v,
        names(&["ABC", "ABC-1", "ABC-2", "ABC-3", "X", "ABC-1-1"])
    );
    let distinct: HashSet<&str> = v.iter().map(|n| n.as_ref()).collect();
    assert_eq!(distinct.len(), v.len());

    // Plain duplicates number as before.
    let mut v = names(&["A", "B", "A", "A", "B"]);
    assert_eq!(make_names_unique(&mut v), 2);
    assert_eq!(v, names(&["A", "B", "A-1", "A-2", "B-1"]));

    let mut v = names(&["A", "B"]);
    assert_eq!(make_names_unique(&mut v), 0);
    assert_eq!(v, names(&["A", "B"]));
}

#[test]
fn a_locus_id_is_not_split_inside_its_contig_name() {
    for id in [
        "chrUn_KI270742v1:100-200",
        "chr1_KI270706v1_random:5-9",
        "chr1:100-200",
        "chrUn_KI270742v1:100-200/count/spliced",
    ] {
        assert_eq!(split_id_name(id), (id, id), "{id}");
        assert_eq!(split_id_name(&id_name(id, id)), (id, id), "{id}");
    }
    assert_eq!(
        split_id_name(&id_name("chrUn_KI270742v1:100-200", "GENE1")),
        ("chrUn_KI270742v1:100-200", "GENE1")
    );
    // A composite that is one locus as a whole stays whole: a genome-tagged
    // contig of a multi-genome reference, and equally an id joined to a
    // locus name, which reads the same.
    for id in [
        "GRCh38_chr1:100-200",
        "mm10_chr1:100-200",
        "peak1_chr1:100-200",
    ] {
        assert_eq!(split_id_name(id), (id, id), "{id}");
    }
    // Gene rows split on the first separator, as composed.
    assert_eq!(
        split_id_name("ENSG00000153563_CD8A"),
        ("ENSG00000153563", "CD8A")
    );
    assert_eq!(split_id_name("CD8A"), ("CD8A", "CD8A"));
}

#[test]
fn substring_matches_come_back_once_and_blank_queries_match_nothing() {
    let all = names(&["CD3E", "CD3D", "CD8A", "MS4A1"]);

    // A blank line of a name file is not a query for every name.
    let (idx, got) = match_by_substring(&all, &names(&["CD8A", ""]), "row").unwrap();
    assert_eq!(idx, vec![2]);
    assert_eq!(got, names(&["CD8A"]));
    assert!(match_by_substring(&all, &names(&[""]), "row").is_err());

    // Overlapping queries: each name once, in first-matched order.
    let (idx, got) = match_by_substring(&all, &names(&["CD3E", "CD3", "MS4"]), "row").unwrap();
    assert_eq!(idx, vec![0, 1, 3]);
    assert_eq!(got, names(&["CD3E", "CD3D", "MS4A1"]));
}
