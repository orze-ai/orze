import hashlib
import json

import pytest

from examples.research_comparison.existing_selection import select_existing


def canonical(value):
    if type(value) is not dict or set(value) != {"alpha"} or type(value["alpha"]) not in (int, float):
        raise ValueError("complete configuration required")
    return {"alpha": float(value["alpha"])}


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def choose(selection, history):
    return select_existing(selection, history, canonicalize=canonical, identity=identity)


def test_complete_candidate_and_id_preserve_explicit_nonminimum_choice():
    history = [{"id": identity(canonical({"alpha": a})), "candidate": {"alpha": a}, "loss": loss}
               for a, loss in [(1., .1), (2., .3)]]
    selected = history[1]
    assert choose({"kind": "select", "candidate_id": selected["id"]}, history) is selected
    assert choose({"kind": "select", "candidate": {"alpha": 2}}, history) is selected


def test_absent_configuration_does_not_select_nearest_or_fit_a_new_candidate():
    history = [{"id": identity(canonical({"alpha": 1})), "candidate": {"alpha": 1}}]
    with pytest.raises(ValueError, match="absent"):
        choose({"kind": "select", "candidate": {"alpha": 1.01}}, history)
    assert len(history) == 1


def test_stored_identity_is_checked_for_both_selection_forms():
    cid = identity(canonical({"alpha": 1}))
    history = [{"id": cid, "candidate": {"alpha": 2}}]
    for selection in [{"kind": "select", "candidate_id": cid}, {"kind": "select", "candidate": {"alpha": 1}}]:
        with pytest.raises(ValueError, match="identity"):
            choose(selection, history)


@pytest.mark.parametrize("selection", [
    {"kind": "select", "candidate": {}},
    {"kind": "select", "candidate": {"alpha": True}},
    {"kind": "select", "candidate_id": "x", "candidate": {"alpha": 1}},
    {"kind": "select", "candidate_id": ""},
    {"kind": "select", "candidate_id": True},
])
def test_incomplete_or_ambiguous_choice_is_rejected(selection):
    with pytest.raises(ValueError):
        choose(selection, [])
