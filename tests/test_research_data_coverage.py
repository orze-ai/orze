"""Hand-computed support and unequal-group examples, independent of model fitting."""
import math
import pytest
from examples.research_comparison.data_coverage import coverage_summary, group_error_summary


def test_marginal_ranges_do_not_hide_new_categories_inside_the_range():
    result = coverage_summary([[0, 1], [2, 3], [1, 1]], [[3, 2], [1, 1]],
                              ["x", "category"], ["a", "a", "b"], ["c", "d"],
                              categorical_features=["category"])
    assert result["training"]["row_weight_effective_groups"] == 9 / 5
    assert result["overlapping_groups"] == 0
    assert result["development_rows_outside_marginal_support"] == 1
    assert result["features"][0]["above_training_range_rows"] == 1
    assert result["features"][1]["above_training_range_rows"] == 0
    assert result["features"][1]["unseen_development_levels"] == [2]


def test_group_errors_preserve_row_weights_and_all_groups():
    result = group_error_summary([0, 0, 0, 0], [1, 1, 1, 3], ["a", "a", "a", "b"])
    assert result == [{"group": "a", "rows": 3, "mse": 1.0},
                      {"group": "b", "rows": 1, "mse": 9.0}]
    assert math.fsum(r["rows"] * r["mse"] for r in result) / 4 == 3.0
    assert group_error_summary([0] * 4, [3, 1, 1, 1], ["b", "a", "a", "a"]) == result


@pytest.mark.parametrize("X", [[], [[1, 2]], [[float("nan")]], [[True]]])
def test_bad_features_cannot_become_research_facts(X):
    with pytest.raises(ValueError):
        coverage_summary([[0]], X, ["x"], ["a"], ["b"])


def test_group_misalignment_is_rejected():
    with pytest.raises(ValueError):
        group_error_summary([0, 1], [0, 1], ["a"])
