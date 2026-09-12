"""Independent policy-only checks using historical actual CLI metadata.

Reading these archived snapshots does not execute a worker, replay a campaign,
or create new evidence. Counterfactual metadata edits below are unit inputs,
not published artifacts or observations. Formal paired experiments are separate.
"""
import copy
import json
from pathlib import Path

import pytest

from examples.acceptance.common import digest
from examples.acceptance.policy import CommonPolicy, TASKS
from examples.research_efficiency.policy import DominancePruningPolicy


ARCHIVE = (Path(__file__).resolve().parents[1] / "docs" / "evidence" / "runs"
           / "2026-09-12-c3-core-full-raw")


def _run(domain, variant="default"):
    path = ARCHIVE / ("acceptance-" + domain + "-" + variant
                      + "0__acceptance-report.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _analysis_entry(run):
    return next(entry for entry in run["trace"]
                if entry["decision"].get("task_id") == TASKS["analysis"])


def _record(snapshot, task):
    return next(row for row in snapshot["recorded_evidence"]["results"]
                if row["ref"]["task_id"] == task)


def _observation(snapshot, task):
    return _record(snapshot, task)["observation_records"][0]


def _declaration(protocols):
    return {"version": 1, "kind": "dominance_pruning", "idle": "stop",
            "wait_seconds": .05,
            "config": {"validation_only_protocols": list(protocols)}}


def _ordinary():
    declaration = _declaration([])
    declaration["config"] = {}
    return CommonPolicy(declaration)


@pytest.mark.parametrize("domain", ["sorting", "compression"])
def test_actual_preanalysis_snapshot_prunes_without_rewriting_recorded_evidence(domain, capsys):
    entry = copy.deepcopy(_analysis_entry(_run(domain)))
    snapshot, budget = entry["snapshot"], entry["budget"]
    before = copy.deepcopy(entry)
    unknown = _observation(snapshot, TASKS["unchecked"])
    incumbent = _record(snapshot, TASKS["baseline"])
    policy = DominancePruningPolicy(_declaration([unknown["protocol_fingerprint"]]))

    assert _ordinary()._decide(snapshot, budget) == entry["decision"]
    selected = policy.decide(snapshot, budget)
    output = capsys.readouterr().out.splitlines()
    trace = json.loads(next(line.removeprefix("ACCEPTANCE_DECISION=")
                            for line in output if line.startswith("ACCEPTANCE_DECISION=")))
    pruning = json.loads(next(line.removeprefix("PRUNING_DECISION=")
                              for line in output if line.startswith("PRUNING_DECISION=")))
    assert selected["kind"] == "Replicate" and selected["source_ref"] == incumbent["ref"]
    assert entry == before
    assert trace["snapshot"] == snapshot and trace["budget"] == budget
    assert trace["snapshot_sha256"] == pruning["snapshot_sha256"] == digest(snapshot)
    assert trace["decision"] == pruning["decision"] == selected
    assert unknown["validation"]["status"] == "unknown"
    assert TASKS["analysis"] not in {r["ref"]["task_id"]
                                     for r in trace["snapshot"]["recorded_evidence"]["results"]}


@pytest.mark.parametrize("domain", ["sorting", "compression"])
@pytest.mark.parametrize("variant", ["default", "counterfactual"])
def test_every_other_actual_trajectory_decision_is_exactly_the_existing_policy(domain, variant, capsys):
    run = _run(domain, variant)
    protocol = run["observations"][0]["protocol_fingerprint"]
    policy = DominancePruningPolicy(_declaration([protocol]))
    checked = 0
    for original in run["trace"]:
        if original["decision"].get("task_id") == TASKS["analysis"]:
            continue
        entry = copy.deepcopy(original)
        assert _ordinary()._decide(entry["snapshot"], entry["budget"]) == original["decision"]
        assert policy._decide(entry["snapshot"], entry["budget"]) == original["decision"]
        assert entry == original
        checked += 1
    assert checked >= 7
    assert "PRUNING_DECISION=" not in capsys.readouterr().out


@pytest.mark.parametrize("domain", ["sorting", "compression"])
def test_equal_unknown_cost_keeps_analysis_that_can_change_stable_tie_break(domain, capsys):
    run = _run(domain)
    entry = copy.deepcopy(_analysis_entry(run))
    snapshot, budget = entry["snapshot"], entry["budget"]
    unknown = _observation(snapshot, TASKS["unchecked"])
    incumbent = _observation(snapshot, TASKS["baseline"])
    # Deliberate counterfactual, not a changed stored measurement.
    unknown["values"]["cost"] = incumbent["values"]["cost"]
    policy = DominancePruningPolicy(_declaration([unknown["protocol_fingerprint"]]))
    before = copy.deepcopy(snapshot)
    assert policy._decide(snapshot, budget) == _ordinary()._decide(snapshot, budget)
    assert policy._decide(snapshot, budget)["task_id"] == TASKS["analysis"]
    assert snapshot == before
    assert "PRUNING_DECISION=" not in capsys.readouterr().out

    # Once analysis really would have a valid tied result, the existing stable
    # task-ID tie break prefers idea-analysis over idea-baseline. This explains
    # why equality is not a safe optimization, rather than checking only > syntax.
    later = next(row for row in run["trace"] if row["decision"]["kind"] == "Replicate")
    analysis = copy.deepcopy(_record(later["snapshot"], TASKS["analysis"]))
    analysis["observation_records"][0]["values"]["cost"] = incumbent["values"]["cost"]
    snapshot["recorded_evidence"]["results"].append(analysis)
    original_choice = _ordinary()._decide(snapshot, budget)
    assert original_choice["source_ref"] == analysis["ref"]
    assert policy._decide(snapshot, budget) == original_choice


def test_mutating_caller_configuration_cannot_add_a_protocol_capability(capsys):
    entry = _analysis_entry(_run("sorting"))
    protocol = _observation(entry["snapshot"], TASKS["unchecked"])["protocol_fingerprint"]
    declaration = _declaration([])
    policy = DominancePruningPolicy(declaration)
    declaration["config"]["validation_only_protocols"].append(protocol)
    assert policy.validation_only_protocols == frozenset()
    assert policy._decide(entry["snapshot"], entry["budget"]) == entry["decision"]
    assert "PRUNING_DECISION=" not in capsys.readouterr().out


@pytest.mark.parametrize("field", ["protocol_fingerprint", "comparison_scope", "dataset_sha256"])
def test_matched_action_replica_with_incomparable_observation_cannot_confirm(field):
    run = _run("sorting")
    entry = copy.deepcopy(_analysis_entry(run))
    snapshot = entry["snapshot"]
    final = run["trace"][-1]["snapshot"]
    replica = copy.deepcopy(next(row for row in final["recorded_evidence"]["results"]
                                 if row["ref"]["task_id"] not in TASKS.values()))
    observation = replica["observation_records"][0]
    if field == "dataset_sha256":
        observation["values"][field] = "0" * 64
    else:
        observation[field] = "0" * 64
    snapshot["recorded_evidence"]["results"].append(replica)
    before = copy.deepcopy(snapshot)
    protocol = _observation(snapshot, TASKS["unchecked"])["protocol_fingerprint"]
    decision = DominancePruningPolicy(_declaration([protocol]))._decide(snapshot, entry["budget"])
    assert decision == {"kind": "Stop", "reason": "confirmation_failed", "wakeup": None}
    assert snapshot == before
