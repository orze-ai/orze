"""Recorded execution and source survive failed follow-up actions together."""
import copy
import json

import pytest

from examples.research_comparison.open_experiment import check_source, identity
from examples.research_comparison.program_evidence import context
from examples.research_comparison.regression_experiment import execute


def row(task, value, valid=True):
    action = {"kind": "method", "source": f"def fit_predict(train, inputs, seed):\n    return [{value!r}] * len(inputs['X'])\n"}
    return {"task_id": task, "action": action, "action_id": identity(action),
            "valid": valid, "facts": {"findings": {"chosen": value}}}


def test_failed_followup_keeps_actual_previous_method_and_leader():
    history = [row("leader", 2.), row("actual-mean", 3.), row("failed", None, False)]
    original = copy.deepcopy(history)
    packet = json.loads(context(history, history[0]["action_id"]))
    assert [r["task_id"] for r in packet["records"]] == ["failed", "actual-mean", "leader"]
    for actual, expected in zip(packet["records"], reversed(history)):
        assert actual["action"] == expected["action"] and actual["facts"] == expected["facts"]
    assert not packet["omitted_task_ids"] and history == original


def test_same_record_combines_roles_and_capacity_never_splits_source_from_facts():
    h = [row("leader", 2.)]
    p = json.loads(context(h, h[0]["action_id"]))
    assert len(p["records"]) == 1 and len(p["records"][0]["roles"]) == 3
    h.append(row("large-failure", "中" * 1000, False))
    raw = context(h, h[0]["action_id"], max_bytes=1000)
    p = json.loads(raw)
    assert len(raw.encode()) <= 1000 and p["omitted_task_ids"] == ["large-failure"]
    assert p["records"][0]["action"] == h[0]["action"]
    h[-1]["action"]["source"] += "# tampered"
    with pytest.raises(ValueError): context(h, h[0]["action_id"])


def test_local_helpers_and_exception_names_execute_without_private_internals():
    source = '''class Method:
    def _predict(self, inputs):
        try:
            raise ValueError('diagnostic')
        except Exception as exc:
            return {'prediction': [1.] * len(inputs['X']), 'findings': {'error_type': type(exc).__name__}}
    def predict(self, inputs):
        return self._predict(inputs)
def fit_predict(train, inputs, seed):
    return Method().predict(inputs)
'''
    data = {k: {"X": [[0]], "y": [1.], "groups": [0], "row_ids": [0],
                "feature_names": ["x"]} for k in ["train", "development"]}
    facts, _ = execute({"kind": "method", "source": source}, data, [],
                       report_findings=True, input_fields=("X", "feature_names"))
    assert facts["loss"] == 0 and facts["findings"]["error_type"] == "ValueError"


@pytest.mark.parametrize("expression", ["type(data).__dict__", "np._core", "data.__class__", "self._unknown()"])
def test_other_private_access_is_still_rejected(expression):
    with pytest.raises(ValueError):
        check_source(f"import numpy as np\ndef analyze(data, history, seed):\n    return {expression}\n", "analyze")
