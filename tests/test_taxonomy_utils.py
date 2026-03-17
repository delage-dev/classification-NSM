import pytest
from taxonomy_utils import parse_taxonomy_from_filename, TaxonomyTree

def test_parse_pattern_1():
    fname = "cordylidae_cham-aenea_26_1-c1.vtk"
    res = parse_taxonomy_from_filename(fname)
    
    assert res is not None
    assert res['family'] == 'cordylidae'
    # Wait, the implemented logic splits by hyphen or underscore.
    # Parts: cordylidae, cham, aenea, 26, 1
    # Actually the split might yield:
    # family=cordylidae, genus=cham, species=aenea, specimen_id=26_1
    # Let's see how the function would evaluate it:
    # "cordylidae_cham-aenea_26_1-c1"
    # match_end -> -c1. base="cordylidae_cham-aenea_26_1". pos=Cervical, num=1.
    # match_filenum -> base ends with _1. base="cordylidae_cham-aenea_26".
    # wait: `re.search(r'[-_]\d+$', base)` matches `_1` exactly at the end.
    # Then `parts`: ['cordylidae', 'cham', 'aenea', '26']
    # If length = 4 -> family='cordylidae', genus='cham', species='aenea', specimen_id='26'
    
    # Let's adjust expected based on how we want it to work:
    assert res['family'] == 'cordylidae'
    assert res['genus'] == 'cham'
    assert res['species'] == 'aenea'
    
    # Check position
    assert res['position'] == 'Cervical'
    assert res['vertebra_number'] == 1

def test_parse_pattern_2_cordylidae_cham():
    # Another pattern from count_vertebrae.py:
    fname = "cordylidae_chamseura_aenea_specimen4_3-t12.vtk"
    res = parse_taxonomy_from_filename(fname)
    
    assert res is not None
    assert res['family'] == 'cordylidae'
    assert res['genus'] == 'chamseura'
    assert res['species'] == 'aenea'
    assert res['specimen_id'] == 'specimen4'
    assert res['position'] == 'Thoracic'
    assert res['vertebra_number'] == 12

def test_parse_missing_taxonomy():
    fname = "scincidae_specimen7-l5.vtk"
    res = parse_taxonomy_from_filename(fname)
    
    assert res is not None
    assert res['family'] == 'scincidae'
    assert res['position'] == 'Lumbar'
    assert res['vertebra_number'] == 5

def test_taxonomic_distance():
    true_dict = {'family': 'F1', 'genus': 'G1', 'species': 'S1'}
    
    pred_exact = {'family': 'F1', 'genus': 'G1', 'species': 'S1'}
    assert TaxonomyTree.get_taxonomic_distance(pred_exact, true_dict) == 0
    
    pred_genus_right = {'family': 'F1', 'genus': 'G1', 'species': 'S2'}
    assert TaxonomyTree.get_taxonomic_distance(pred_genus_right, true_dict) == 1
    
    pred_family_right = {'family': 'F1', 'genus': 'G2', 'species': 'S2'}
    assert TaxonomyTree.get_taxonomic_distance(pred_family_right, true_dict) == 2
    
    pred_wrong = {'family': 'F2', 'genus': 'G2', 'species': 'S2'}
    assert TaxonomyTree.get_taxonomic_distance(pred_wrong, true_dict) == 3
