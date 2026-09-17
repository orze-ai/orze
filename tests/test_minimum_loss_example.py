import pytest

from examples.research_comparison.loss_selection import MinimumLoss


def test_select_keeps_the_original_first_candidate_on_ties():
    first = {"loss": 0.0, "id": "earlier"}
    later = {"loss": 0, "id": "later", "runtime": 0.001}
    rule = MinimumLoss("error rate", lower_bound=0)
    assert rule.select([{"loss": 0.2}, first, later]) is first
    assert rule.at_declared_bound([first, later])
    assert "exact ties retain the earlier candidate" in rule.instructions()


def test_observed_minimum_and_approximate_zero_do_not_establish_a_bound():
    assert not MinimumLoss("loss").at_declared_bound([{"loss": 0}])
    assert not MinimumLoss("error rate", 0).at_declared_bound([{"loss": 1e-15}])
    assert MinimumLoss("shifted loss", -2).at_declared_bound([{"loss": -2}])


@pytest.mark.parametrize("value", [True, "0", float("nan"), float("inf"), -0.1])
def test_bad_or_contradictory_measurements_cannot_trigger_a_stop(value):
    with pytest.raises(ValueError):
        MinimumLoss("error rate", 0).at_declared_bound([{"loss": 0}, {"loss": value}])


def test_empty_candidates_and_nonfinite_bounds_are_rejected():
    with pytest.raises(ValueError):
        MinimumLoss("loss").select([])
    with pytest.raises(ValueError):
        MinimumLoss("loss", float("inf"))
