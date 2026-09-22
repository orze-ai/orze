"""Group-disjoint random splits for exchangeable research examples.

Group isolation alone does not hide labels encoded by source row order. Shuffle
rows within each output as well as assigning groups. For ordered forecasting or
sequence tasks, use the task's own temporal protocol instead.
"""


def shuffled_group_split(groups, fractions, *, seed):
    """Return reproducible index arrays with intact groups and shuffled rows.

    Use a data-preparation seed separate from the model's public training seed.
    Record it in the evaluator's frozen manifest; do not put it in model input.
    This removes preserved source ordering, not all possible data shortcuts.
    """
    import numpy as np

    groups = np.asarray(groups)
    fractions = np.asarray(fractions, dtype=float)
    if groups.ndim != 1 or len(groups) == 0:
        raise ValueError('groups must be a nonempty one-dimensional array')
    if (fractions.ndim != 1 or len(fractions) < 2 or
            not np.isfinite(fractions).all() or (fractions <= 0).any() or
            not np.isclose(fractions.sum(), 1., rtol=0., atol=1e-12)):
        raise ValueError('positive split fractions must sum to one')
    unique = np.unique(groups)
    counts = np.floor(len(unique) * fractions[:-1]).astype(int)
    counts = np.append(counts, len(unique) - counts.sum())
    if (counts == 0).any():
        raise ValueError('not enough groups for nonempty splits')
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    boundaries = np.cumsum(np.r_[0, counts])
    parts = [rng.permutation(np.flatnonzero(np.isin(groups, unique[a:b])))
             for a, b in zip(boundaries, boundaries[1:])]
    if sum(len(part) for part in parts) != len(groups):
        raise ValueError('group values must support complete membership matching')
    return parts
