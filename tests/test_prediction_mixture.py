import pytest
from examples.research_comparison.prediction_mixture import best_pair, mix


def record(name, values):
    return {'id': name, 'row_ids': ['a', 'b'], 'prediction': values}


def test_complementary_predictions_have_an_exact_interior_optimum():
    result = best_pair(['a', 'b'], [1., 1.], [record('left', [0., 2.]), record('right', [2., 0.])])
    assert result['left_weight'] == .5
    assert result['loss'] == 0
    assert result['prediction'] == [1., 1.]


def test_endpoint_and_duplicate_predictions_keep_the_earliest_single():
    result = best_pair(['a', 'b'], [0., 1.], [record('a', [0., 1.]), record('b', [0., 1.]), record('c', [4., 5.])])
    assert result['left_id'] == result['right_id'] == 'a'
    assert result['pairs_evaluated'] == 3
    assert mix([1e-12, 1e12], [1e12, 1e-12], 1) == [1e-12, 1e12]


def test_row_permutation_requires_explicit_alignment():
    bad = record('b', [1., 0.]);bad['row_ids'] = ['b', 'a']
    with pytest.raises(ValueError, match='row identities'):
        best_pair(['a', 'b'], [0., 1.], [bad])


@pytest.mark.parametrize('bad', [[float('nan'), 1.], [True, 1.]])
def test_nonfinite_or_boolean_predictions_are_not_measurements(bad):
    with pytest.raises(ValueError):
        best_pair(['a', 'b'], [0., 1.], [record('bad', bad)])
