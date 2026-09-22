import numpy as np
import pytest

from orze.research.data_splits import shuffled_group_split


def test_group_isolation_also_removes_preserved_class_order():
    # Same failure mode as corpus files sorted by class, with duplicate groups.
    groups = np.repeat(np.arange(200), 2)
    labels = np.repeat([0, 1], 200)
    parts = shuffled_group_split(groups, [.5, .25, .25], seed=7219)
    assert sorted(np.concatenate(parts).tolist()) == list(range(400))
    assert [len(p) for p in parts] == [200, 100, 100]
    for i, part in enumerate(parts):
        assert all(not set(groups[part]) & set(groups[other]) for other in parts[i+1:])
        # The old flatnonzero-only splitter leaves exactly one transition.
        ordered_labels = labels[part]
        assert np.count_nonzero(ordered_labels[1:] != ordered_labels[:-1]) > len(part) // 4


def test_reproducibility_and_independent_seeds():
    groups = [f'person-{i // 3}' for i in range(90)]
    first = shuffled_group_split(groups, [.5, .3, .2], seed=71)
    repeat = shuffled_group_split(groups, [.5, .3, .2], seed=71)
    other = shuffled_group_split(groups, [.5, .3, .2], seed=72)
    assert all(np.array_equal(a, b) for a, b in zip(first, repeat))
    assert all(not np.array_equal(a, b) for a, b in zip(first, other))


@pytest.mark.parametrize('fractions', [[.4, .4], [1., 0.], [float('nan'), .5]])
def test_invalid_protocol_fails_before_materializing_splits(fractions):
    with pytest.raises(ValueError):
        shuffled_group_split(np.arange(20), fractions, seed=1)


def test_small_group_population_cannot_silently_produce_empty_holdout():
    with pytest.raises(ValueError, match='not enough groups'):
        shuffled_group_split(['a', 'a', 'b', 'b'], [.8, .1, .1], seed=1)


def test_missing_group_ids_cannot_silently_drop_rows():
    with pytest.raises(ValueError, match='complete membership'):
        shuffled_group_split([0., 1., 2., float('nan')], [.5, .5], seed=1)
