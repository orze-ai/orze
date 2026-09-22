"""Regression checks for hidden errors and misleading selected-sample rates."""
import json
import pytest
from orze.research.development_profile import classification_profile


def test_correct_prefix_cannot_hide_population_errors():
    labels = [0]*16+[1]*4
    predictions = [0]*16+[0]*4
    profile = classification_profile(labels, predictions, ['a', 'b'], seed=4)
    assert profile['error_rate'] == .2
    assert sum(row['sample_stratum'] == 'error' for row in profile['examples']) == 4
    assert profile['largest_confusions'] == [{'label': 1, 'prediction': 0, 'rows': 4}]
    assert len({row['row'] for row in profile['examples']}) == len(profile['examples'])
    assert labels[:12] == predictions[:12]


def test_absent_class_unknown_and_empty_error_stratum_not_fabricated():
    profile = classification_profile([0, 0], [0, 0], ['present', 'absent'], seed=4)
    assert profile['per_class'][1]['error_rate'] is None
    assert len(profile['examples']) == 2
    assert all(row['sample_stratum'] == 'correct' for row in profile['examples'])
    assert profile['largest_confusions'] == []
    json.dumps(profile, allow_nan=False)


def test_text_is_observed_truncated_and_sample_reproducible():
    texts = ['x'*500, 'counterexample']; labels = [0, 1]; predictions = [1, 1]
    a = classification_profile(labels, predictions, ['a', 'b'], seed=7, texts=texts)
    b = classification_profile(labels, predictions, ['a', 'b'], seed=7, texts=texts)
    assert a == b and a['examples'][0]['excerpt_is_truncated']
    assert len(a['examples'][0]['observed_text_excerpt']) == 400
    assert a['examples'][1]['observed_text_excerpt'] == texts[1]


@pytest.mark.parametrize('labels,predictions', [([], []), ([0], []), ([0], [.5]), ([0], [float('nan')]), ([0], [2])])
def test_invalid_observations_are_not_described_as_results(labels, predictions):
    with pytest.raises(ValueError): classification_profile(labels, predictions, ['a', 'b'], seed=4)
