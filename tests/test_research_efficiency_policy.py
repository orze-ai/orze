"""New policy mechanisms and direct existing-policy controls, not old Core bugs."""
import copy
import json

import pytest

from examples.acceptance.common import digest
from examples.acceptance.policy import CommonPolicy, TASKS
from examples.research_efficiency.policy import DominancePruningPolicy


PROTOCOL = "a" * 64


def declaration(protocols=None):
    return {"version": 1, "kind": "dominance_pruning", "idle": "stop", "wait_seconds": .05,
            "config": {"validation_only_protocols": [PROTOCOL] if protocols is None else protocols}}


def result(key, cost, status, *, spec=None):
    task = TASKS.get(key, key)
    return {"ref": {"task_id": task, "attempt_id": task + "-attempt"},
            "outcome": "completed",
            "artifact_records": [{"logical_name": "result", "spec_fingerprint": spec or task,
                                  "artifact_id": task + "-artifact"}],
            "observation_records": [{"name": "objective", "values": {
                "cost": cost, "dataset_sha256": "dataset"},
                "validation": {"status": status}, "protocol_fingerprint": PROTOCOL,
                "comparison_scope": "scope"}]}


def inputs():
    snapshot = {"recorded_evidence": {"unavailable": False, "more_available": False,
        "results": [result("baseline", 10, "valid"), result("challenger", 0, "invalid"),
                    result("unchecked", 20, "unknown")]},
        "recorded_proposals": {"results": []}, "queue": [], "active": [], "now": 100.0}
    budget = {"stopped": False, "active_reservations": [], "remaining_wall_seconds": 10}
    return snapshot, budget


def ordinary():
    d = declaration(); d["config"] = {}
    return CommonPolicy(d)


def test_baseline_really_proposes_analysis_and_new_policy_requests_real_replica():
    s, b = inputs()
    assert ordinary()._decide(s, b)["task_id"] == TASKS["analysis"]
    decision = DominancePruningPolicy(declaration())._decide(s, b)
    assert decision["kind"] == "Replicate"
    assert decision["source_ref"] == s["recorded_evidence"]["results"][0]["ref"]


@pytest.mark.parametrize("mode", [
    "equal", "better", "invalid", "no_capability", "different_protocol",
    "different_scope", "different_dataset", "incumbent_unknown", "active",
    "queue", "stopped", "analysis_exists", "failed", "bad_proposal",
])
def test_nonqualifying_conditions_preserve_exact_baseline_behavior(mode):
    s, b = inputs(); config = declaration()
    rows = s["recorded_evidence"]["results"]; u = rows[2]["observation_records"][0]
    if mode == "equal": u["values"]["cost"] = 10
    elif mode == "better": u["values"]["cost"] = 1
    elif mode == "invalid": u["validation"]["status"] = "invalid"
    elif mode == "no_capability": config = declaration([])
    elif mode == "different_protocol": u["protocol_fingerprint"] = "b" * 64
    elif mode == "different_scope": u["comparison_scope"] = "other"
    elif mode == "different_dataset": u["values"]["dataset_sha256"] = "other"
    elif mode == "incumbent_unknown": rows[0]["observation_records"][0]["validation"]["status"] = "unknown"
    elif mode == "active": s["active"] = [{"opaque": "active"}]
    elif mode == "queue": s["queue"] = [{"idea_id": "queued", "request": {"timeout_seconds": 2}}]
    elif mode == "stopped": b["stopped"] = True
    elif mode == "analysis_exists": rows.append(result("analysis", 20, "valid"))
    elif mode == "failed": rows[1]["outcome"] = "failed"
    elif mode == "bad_proposal": s["recorded_proposals"]["results"] = [{"status": "rejected"}]
    before = copy.deepcopy((s, b))
    assert DominancePruningPolicy(config)._decide(s, b) == ordinary()._decide(s, b)
    assert (s, b) == before


@pytest.mark.parametrize("mode", ["unavailable", "more_available", "duplicate", "bool", "negative", "nan"])
def test_baseline_rejections_are_not_swallowed(mode):
    s, b = inputs()
    if mode in ("unavailable", "more_available"): s["recorded_evidence"][mode] = True
    elif mode == "duplicate": s["recorded_evidence"]["results"].append(copy.deepcopy(s["recorded_evidence"]["results"][0]))
    else:
        s["recorded_evidence"]["results"][2]["observation_records"][0]["values"]["cost"] = {
            "bool": True, "negative": -1, "nan": float("nan")}[mode]
    for policy in (ordinary(), DominancePruningPolicy(declaration())):
        with pytest.raises(ValueError):
            policy._decide(s, b)


@pytest.mark.parametrize("value", [None, {}, {"validation_only_protocols": "a" * 64},
    {"validation_only_protocols": ["x"]}, {"validation_only_protocols": [PROTOCOL, PROTOCOL]},
    {"validation_only_protocols": [True]}, {"validation_only_protocols": [], "other": 1}])
def test_invalid_configuration_is_rejected(value):
    d = declaration(); d["config"] = value
    with pytest.raises(ValueError):
        DominancePruningPolicy(d)


@pytest.mark.parametrize("status,cost,expected", [
    ("valid", 10, "confirmed_selection"), ("valid", 11, "confirmation_failed"),
    ("unknown", 10, "confirmation_failed"), ("invalid", 10, "confirmation_failed")])
def test_recorded_replica_is_required_and_requalified(status, cost, expected):
    s, b = inputs()
    s["recorded_evidence"]["results"].append(
        result("replica", cost, status, spec=TASKS["baseline"]))
    decision = DominancePruningPolicy(declaration())._decide(s, b)
    assert decision == {"kind": "Stop", "reason": expected, "wakeup": None}


def test_real_snapshot_and_unknown_preserved_in_diagnostic(capsys):
    s, b = inputs(); before = copy.deepcopy((s, b))
    DominancePruningPolicy(declaration()).decide(s, b)
    records = capsys.readouterr().out.splitlines()
    trace = json.loads(next(line.removeprefix("ACCEPTANCE_DECISION=")
                            for line in records if line.startswith("ACCEPTANCE_DECISION=")))
    pruning = json.loads(next(line.removeprefix("PRUNING_DECISION=")
                              for line in records if line.startswith("PRUNING_DECISION=")))
    assert (s, b) == before
    assert trace["snapshot"] == s
    assert trace["snapshot_sha256"] == pruning["snapshot_sha256"] == digest(s)
    assert TASKS["analysis"] not in [r["ref"]["task_id"] for r in trace["snapshot"]["recorded_evidence"]["results"]]
    assert trace["snapshot"]["recorded_evidence"]["results"][2]["observation_records"][0]["validation"]["status"] == "unknown"
