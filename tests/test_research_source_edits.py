"""Check source fidelity and data boundaries, not scientific capability."""
import copy

import pytest

from examples.research_comparison.open_experiment import check_source, identity
from examples.research_comparison.regression_experiment import execute
from examples.research_comparison.source_edits import resolve


def measured():
    action = {"kind": "method", "source": "def fit_predict(train, inputs, seed):\n    return [2.0] * len(inputs['X'])\n"}
    return {"action": action, "action_id": identity(action), "valid": True}


def test_exact_edits_keep_unmentioned_source_and_parent_intact():
    row = measured()
    history = [row]
    original = copy.deepcopy(history)
    action = {"kind": "method", "parent_id": row["action_id"],
              "edits": [{"old": "[2.0]", "new": "[3.0]"}]}
    result = resolve(action, history)
    assert result["source"] == row["action"]["source"].replace("[2.0]", "[3.0]")
    assert history == original
    data = {k: {"X": [[1], [2]], "feature_names": ["x"], "y": [2., 4.],
                "groups": [0, 1], "row_ids": [0, 1]} for k in ["train", "development"]}
    facts, pred = execute(result, data, history, input_fields=("X", "feature_names"))
    assert pred["prediction"] == [3., 3.] and facts["loss"] == 1.


@pytest.mark.parametrize("change", ["missing", "tampered", "invalid", "analysis"])
def test_unverified_or_changed_parent_is_not_guessed(change):
    row = measured()
    action = {"kind": "method", "parent_id": row["action_id"],
              "edits": [{"old": "[2.0]", "new": "[3.0]"}]}
    history = [row]
    if change == "missing": history = []
    if change == "tampered": row["action"]["source"] += "\n# changed"
    if change == "invalid": row["valid"] = False
    if change == "analysis": row["action"]["kind"] = "analyze"
    with pytest.raises(ValueError): resolve(action, history)


@pytest.mark.parametrize("old,new", [("absent", "x"), (" ", "x"), ("", "x"), ("[2.0]", "x" * 32768)])
def test_missing_ambiguous_or_oversized_edits_fail(old, new):
    row = measured()
    with pytest.raises(ValueError):
        resolve({"kind": "method", "parent_id": row["action_id"],
                 "edits": [{"old": old, "new": new}]}, [row])


def test_in_memory_json_and_counting_supported_but_io_still_rejected():
    check_source("import json\nfrom collections import Counter\ndef analyze(data, history, seed):\n    return json.loads(json.dumps(dict(Counter([1, 1]))))\n", "analyze")
    for source in ["import os\ndef analyze(*args): return 1", "import json\ndef analyze(*args): return json.dump({}, None)"]:
        with pytest.raises(ValueError): check_source(source, "analyze")


def test_analysis_and_method_cannot_receive_confirmation_labels():
    action = {"kind": "analyze", "source": "def analyze(data, history, seed):\n    return 0\n"}
    with pytest.raises(ValueError, match="early-data"):
        execute(action, {"train": {}, "development": {}, "confirmation": {}}, [])
    with pytest.raises(ValueError, match="labels"):
        execute(measured()["action"], {}, [], input_fields=("X", "y"))
