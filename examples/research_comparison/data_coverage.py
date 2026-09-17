"""Measured development support and group errors for a research project.

Supply training and development features only, with an explicit grouping.
This descriptive adapter neither reads heldout data nor chooses a model.
Marginal ranges cannot establish joint support or independent sample size.
"""
from collections import Counter
import math
from numbers import Real


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("finite numeric values required")
    return float(value)


def _groups(values, rows):
    if len(values) != rows or any(type(v) is not str or not v for v in values):
        raise ValueError("one nonempty string group per row required")
    counts = Counter(values)
    return {
        "rows": rows,
        "groups": len(counts),
        "largest_group_rows": max(counts.values()),
        "row_weight_effective_groups": rows * rows / sum(n * n for n in counts.values()),
    }


def coverage_summary(train_X, development_X, feature_names, train_groups, development_groups,
                     categorical_features=()):
    """Count unsupported development values and unequal group representation.

    ``row_weight_effective_groups`` is 1/sum(group row fractions squared).
    It describes group-size concentration, not a statistical effective N.
    Categorical columns must be declared by name; no dtype-based guessing.
    """
    names = list(feature_names)
    if not names or any(type(n) is not str or not n for n in names) or len(set(names)) != len(names):
        raise ValueError("unique nonempty feature names required")
    categories = set(categorical_features)
    if not categories <= set(names):
        raise ValueError("unknown categorical feature")
    matrices = []
    for X in [train_X, development_X]:
        if not len(X) or any(len(row) != len(names) for row in X):
            raise ValueError("nonempty rectangular feature matrices required")
        matrices.append([[_number(v) for v in row] for row in X])
    train, dev = matrices
    training = _groups(train_groups, len(train))
    development = _groups(development_groups, len(dev))
    columns = []
    unsupported_rows = set()
    for j, name in enumerate(names):
        observed = [row[j] for row in train]
        evaluated = [row[j] for row in dev]
        lo, hi = min(observed), max(observed)
        below = sum(v < lo for v in evaluated)
        above = sum(v > hi for v in evaluated)
        column = {"feature": name, "training_min": lo, "training_max": hi,
                  "development_min": min(evaluated), "development_max": max(evaluated),
                  "below_training_range_rows": below, "above_training_range_rows": above}
        unsupported_rows.update(i for i, v in enumerate(evaluated) if v < lo or v > hi)
        if name in categories:
            levels = set(observed)
            novel = {v for v in evaluated if v not in levels}
            column.update(training_levels=sorted(levels), unseen_development_levels=sorted(novel),
                          unseen_level_rows=sum(v in novel for v in evaluated))
            unsupported_rows.update(i for i, v in enumerate(evaluated) if v in novel)
        columns.append(column)
    return {"training": training, "development": development,
            "overlapping_groups": len(set(train_groups) & set(development_groups)),
            "development_rows_outside_marginal_support": len(unsupported_rows),
            "features": columns,
            "scope": "Training and development only. Marginal support and group-size concentration "
                     "are descriptive; they do not establish joint support, statistical independence "
                     "or future generalization. No default clipping or selection change follows."}


def group_error_summary(targets, predictions, groups):
    """Retain every declared development group, with row-weighted MSE."""
    if not len(targets) or len(targets) != len(predictions):
        raise ValueError("one prediction per nonempty target row required")
    _groups(groups, len(targets))
    errors = {}
    for target, prediction, group in zip(targets, predictions, groups):
        squared = (_number(prediction) - _number(target)) ** 2
        errors.setdefault(group, []).append(_number(squared))
    return [{"group": group, "rows": len(values), "mse": math.fsum(values) / len(values)}
            for group, values in sorted(errors.items())]
