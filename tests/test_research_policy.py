"""Autonomous research admission contracts."""

import pytest

from orze.core.research_policy import validate_batch_decision_contract


def _cfg(sort="ascending"):
    return {
        "report": {"primary_metric": "avg_wer", "sort": sort},
        "research_policy": {
            "require_batch_decision_contract": True,
            "max_decision_batch": 3,
            "min_decision_effect": 0.02,
        },
    }


def _contract(**overrides):
    contract = {
        "uncertainty": "Whether architecture changes improve the public proxy.",
        "metric": "avg_wer",
        "baseline": 5.4,
        "comparator": "lt",
        "threshold": 5.3,
        "on_failure": "redirect_family",
        "max_experiments": 3,
    }
    contract.update(overrides)
    return contract


def test_valid_batch_decision_contract_is_accepted():
    assert validate_batch_decision_contract(
        _contract(), _cfg(), idea_count=3, qualified_best=5.4) is None


def test_minimum_effect_boundary_tolerates_float_representation():
    assert validate_batch_decision_contract(
        _contract(threshold=5.38), _cfg(), idea_count=3,
        qualified_best=5.4) is None


@pytest.mark.parametrize(
    "contract,reason",
    [
        (None, "batch_decision_contract_missing"),
        (_contract(extra="field"), "batch_decision_contract_fields_invalid"),
        (_contract(uncertainty="try it"),
         "batch_decision_contract_uncertainty_invalid"),
        (_contract(metric="private_rank"),
         "batch_decision_contract_metric_mismatch"),
        (_contract(baseline=5.5),
         "batch_decision_contract_baseline_mismatch"),
        (_contract(comparator="gt"),
         "batch_decision_contract_direction_mismatch"),
        (_contract(threshold=True),
         "batch_decision_contract_threshold_invalid"),
        (_contract(threshold=float("nan")),
         "batch_decision_contract_threshold_invalid"),
        (_contract(threshold=5.39),
         "batch_decision_contract_effect_too_small"),
        (_contract(threshold=5.4),
         "batch_decision_contract_effect_too_small"),
        (_contract(on_failure="keep_trying"),
         "batch_decision_contract_failure_action_invalid"),
        (_contract(max_experiments=4),
         "batch_decision_contract_budget_invalid"),
        (_contract(max_experiments=2),
         "batch_decision_contract_count_mismatch"),
    ],
)
def test_invalid_batch_decision_contract_fails_closed(contract, reason):
    assert validate_batch_decision_contract(
        contract, _cfg(), idea_count=3, qualified_best=5.4) == reason


@pytest.mark.parametrize("qualified_best", [True, float("nan"), float("inf")])
def test_invalid_authoritative_baseline_fails_closed(qualified_best):
    assert validate_batch_decision_contract(
        _contract(), _cfg(), idea_count=3,
        qualified_best=qualified_best,
    ) == "batch_decision_contract_authoritative_baseline_invalid"


@pytest.mark.parametrize("idea_count", [True, 0, 2])
def test_invalid_or_mismatched_parsed_count_fails_closed(idea_count):
    assert validate_batch_decision_contract(
        _contract(), _cfg(), idea_count=idea_count,
        qualified_best=5.4,
    ) == "batch_decision_contract_count_mismatch"


def test_descending_metric_requires_improvement_in_correct_direction():
    contract = _contract(
        baseline=0.8, comparator="gte", threshold=0.9)
    assert validate_batch_decision_contract(
        contract, _cfg(sort="descending"), idea_count=3,
        qualified_best=0.8) is None


def test_contract_is_optional_unless_project_requires_it():
    assert validate_batch_decision_contract(
        None, {"research_policy": {}}, idea_count=3) is None


def test_first_batch_requires_explicit_null_baseline():
    contract = _contract(baseline=None)
    assert validate_batch_decision_contract(
        contract, _cfg(), idea_count=3, qualified_best=None) is None
