"""One declared CPU action is not a Cartesian training sweep."""
from copy import deepcopy

import pytest

from orze.core.research_policy import validate_idea_against_research_policy


CFG = {"research_policy": {"require_batch_decision_contract": True}}


def task(domain=False):
    request = {"version": 1, "purpose": "inspect inline input", "inputs": {"items": [1, 2, 3]},
               "timeout_seconds": 1, "outputs": {}}
    if domain:
        request.update(input_artifact_ids=["artifact-one"], payload={"command": ["python3", "check.py"]})
        return {"kind": "native_cpu_action", "domain_request": request}
    request.update(adapter="command", command=["python3", "check.py"])
    return {"kind": "native_cpu_action", "action": request}


@pytest.mark.parametrize("domain", [False, True])
def test_exact_cpu_task_with_inline_lists_is_one_proposal(domain):
    configured = task(domain)
    before = deepcopy(configured)
    assert validate_idea_against_research_policy(configured, CFG) is None
    assert configured == before


@pytest.mark.parametrize("kind", ["train", "native_cpu_actioN", None])
def test_action_shaped_training_config_cannot_skip_sweep_guard(kind):
    configured = task()
    if kind is None:
        configured.pop("kind")
    else:
        configured["kind"] = kind
    assert validate_idea_against_research_policy(configured, CFG) == "batch_decision_contract_implicit_sweep_forbidden"


@pytest.mark.parametrize("mutation", ["extra_root", "both_envelopes", "missing_envelope", "command_string", "empty_argv", "non_string_arg", "version_bool", "bad_timeout", "unbounded_inputs", "bad_output"])
def test_cpu_label_requires_an_exact_valid_action(mutation):
    configured = task()
    action = configured["action"]
    if mutation == "extra_root":
        configured["learning_rate"] = [.1, .2]
    elif mutation == "both_envelopes":
        configured["domain_request"] = task(True)["domain_request"]
    elif mutation == "missing_envelope":
        configured.pop("action")
    elif mutation == "command_string":
        action["command"] = "python3 check.py"
    elif mutation == "empty_argv":
        action["command"] = []
    elif mutation == "non_string_arg":
        action["command"] = ["python3", 2]
    elif mutation == "version_bool":
        action["version"] = True
    elif mutation == "bad_timeout":
        action["timeout_seconds"] = float("nan")
    elif mutation == "unbounded_inputs":
        action["inputs"]["text"] = "x" * 70000
    elif mutation == "bad_output":
        action["outputs"] = {"x": {"path": "../escape", "max_bytes": 5}}
    assert validate_idea_against_research_policy(configured, CFG) == "batch_decision_contract_native_cpu_config_invalid"


@pytest.mark.parametrize("mutation", ["missing_field", "duplicate_ref", "too_many_refs", "bad_timeout", "bad_payload", "extra_root"])
def test_domain_envelope_is_validated_before_policy_acceptance(mutation):
    configured = task(True)
    request = configured["domain_request"]
    if mutation == "missing_field":
        request.pop("outputs")
    elif mutation == "duplicate_ref":
        request["input_artifact_ids"] *= 2
    elif mutation == "too_many_refs":
        request["input_artifact_ids"] = [f"artifact-{i}" for i in range(33)]
    elif mutation == "bad_timeout":
        request["timeout_seconds"] = -1
    elif mutation == "bad_payload":
        request["payload"] = ["command"]
    else:
        configured["action"] = task()["action"]
    assert validate_idea_against_research_policy(configured, CFG) == "batch_decision_contract_native_cpu_config_invalid"


def test_other_research_policy_restrictions_still_apply_to_cpu_data():
    configured = task()
    configured["action"]["inputs"]["ensemble_members"] = ["one", "two"]
    cfg = deepcopy(CFG)
    cfg["research_policy"]["model_form"] = "single_model_single_pass"
    assert validate_idea_against_research_policy(configured, cfg) is not None


@pytest.mark.parametrize("configured", [{"learning_rate": [.1, .2]}, {"training": {"seed": [1, 2]}}])
def test_ordinary_training_sweeps_still_rejected(configured):
    assert validate_idea_against_research_policy(configured, CFG) == "batch_decision_contract_implicit_sweep_forbidden"
