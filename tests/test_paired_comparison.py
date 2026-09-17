import copy

import pytest

from orze.research.paired_comparison import compare


def records():
    a = {"id": "baseline", "comparison_scope": "source:protocol:split", "metric": "loss",
         "direction": "minimize", "configuration": {"lr": 0.1, "l2": 1},
         "units": [{"id": str(i), "group": "small" if i == 0 else "large", "value": 2.0} for i in range(4)]}
    b = copy.deepcopy(a)
    b.update(id="candidate", configuration={"lr": 0.2, "l2": 3})
    for row, value in zip(b["units"], [0.0, 3.0, 3.0, 3.0]):
        row["value"] = value
    return {"reference": a, "candidate": b, "prior_uses": None}


def test_pairs_by_identity_and_retains_different_aggregation_directions():
    doc = records()
    doc["candidate"]["units"].reverse()
    result = compare(doc)
    effect = result["paired_effect"]
    assert effect["candidate_minus_reference_mean"] == 0.25
    assert effect["equal_group_candidate_minus_reference_mean"] == -0.5
    assert (effect["candidate_better_units"], effect["candidate_worse_units"]) == (1, 3)
    assert [x["path"] for x in result["configuration_changes"]] == [["l2"], ["lr"]]
    assert result["prior_use_overlap"] is None


@pytest.mark.parametrize("change", ["scope", "units", "groups"])
def test_incomplete_comparison_never_reports_subset_effect(change):
    doc = records()
    if change == "scope":
        doc["candidate"]["comparison_scope"] = "other-data"
    elif change == "units":
        doc["candidate"]["units"][0]["id"] = "different"
    else:
        doc["candidate"]["units"][0]["group"] = "other-group"
    result = compare(doc)
    assert result["paired_effect"] is None
    assert not result["pairing"]["complete"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_non_measurements_are_rejected(value):
    doc = records()
    doc["candidate"]["units"][0]["value"] = value
    with pytest.raises(ValueError):
        compare(doc)


def test_duplicate_unit_is_not_double_weighted():
    doc = records()
    doc["candidate"]["units"].append(doc["candidate"]["units"][0])
    with pytest.raises(ValueError, match="duplicate measurement"):
        compare(doc)


def test_direction_zero_reference_typed_changes_and_declared_reuse():
    doc = records()
    for record in [doc["reference"], doc["candidate"]]:
        record["direction"] = "maximize"
    for row in doc["reference"]["units"]:
        row["value"] = 0
    doc["candidate"]["configuration"] = {"lr": 0.1, "l2": 1.0, "new": None}
    doc["prior_uses"] = [{"id": "earlier-selection", "unit_ids": ["0", "3", "outside"]}]
    result = compare(doc)
    assert result["paired_effect"]["improvement"] > 0
    assert result["paired_effect"]["relative_improvement"] is None
    assert [x["path"] for x in result["configuration_changes"]] == [["l2"], ["new"]]
    assert result["prior_use_overlap"][0]["candidate_evaluation_overlap"] == 2
