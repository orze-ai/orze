"""Actual registered Domain terminal recovery; no application edits or fake tree.

Each parameter runs two actual native CPU actions in a private project. Domain
callbacks and READY handles are transparently observed in fresh controllers.
A recovery-only controller forbids Domain callbacks, not normal CLI lifecycle.
"""
from contextlib import closing
import json
import os
from pathlib import Path
import sys

import pytest
import yaml

from cpu_terminal_recovery_helpers import canonical, run_cli, save_report, snapshot
from orze.idea_lake import IdeaLake

pytest_plugins = ["cpu_terminal_recovery_helpers"]


def _events(root):
    path = root / "domain-events.jsonl"
    return [json.loads(row) for row in path.read_text().splitlines()] if path.exists() else []


def _admit(project, task, domain_request):
    before = snapshot(project, "domain-admit:" + task + ":before")
    raw = canonical({"kind": "native_cpu_action", "domain_request": domain_request}).decode()
    with closing(IdeaLake(project["root"] / "lake.db")) as lake:
        result = lake.insert(task, task, raw, "", status="queued",
                             kind="native_cpu_action", if_absent=True)
    after = snapshot(project, "domain-admit:" + task + ":after")
    project["admissions"].append({"task_id": task, "config": raw, "result": result,
                                  "before": before["label"], "after": after["label"]})
    save_report(project)
    assert result["status"] == "inserted"


@pytest.fixture
def domain_project(tmp_path, request):
    root = tmp_path / "domain-project"
    root.mkdir()
    instance = json.loads((Path(__file__).resolve().parents[1] / "examples/holdout/instance.json").read_bytes())
    cfg = {"execution": {"version": 1, "resource": "cpu", "slots": 1, "wall_budget_seconds": 6},
           "results_dir": str(root / "results"), "idea_lake_db": str(root / "lake.db"),
           "ideas_file": str(root / "ideas.md"), "min_disk_gb": 0,
           "action_domain": {"version": 1, "kind": "schedule_holdout", "config": {"instance": instance}},
           "action_policy": {"version": 1, "kind": "queue", "idle": "wait", "wait_seconds": .05}}
    (root / "orze.yaml").write_text(yaml.safe_dump(cfg))
    project = {"root": root, "cfg": cfg, "calls": [], "snapshots": [], "admissions": []}
    try:
        yield project
    finally:
        project["domain_events"] = _events(root)
        reports = getattr(request.config, "_cpu_recovery_reports", None)
        if reports is None:
            reports = request.config._cpu_recovery_reports = []
        reports.append({**save_report(project), "test": request.node.nodeid})


@pytest.mark.parametrize("protocol,status", [
    ("schedule-feasibility-v1", "valid"), ("schedule-feasibility-v2", "invalid")])
def test_fresh_cli_resumes_confirmed_domain_without_reinterpretation(domain_project, monkeypatch, protocol, status):
    from examples.holdout.scheduling import make_request
    project = domain_project
    module = "test_cpu_terminal_recovery_domain"
    _admit(project, "idea-producer", make_request("produce", candidate="challenger"))
    run_cli(project, "domain-producer", child_module=module)
    produced = project["snapshots"][-1]
    assert len(produced["database"]["research_artifacts"]) == 1
    source = json.loads(produced["database"]["research_artifacts"][0]["record_json"])
    assert source["producer"]["task_id"] == "idea-producer"
    assert produced["database"]["research_observations"] == []
    _admit(project, "idea-evaluator", make_request("evaluate", protocol=protocol,
                                                  source_id=source["artifact_id"]))
    run_cli(project, "domain-evaluator-crash", crash=True, expected=86, child_module=module)
    before = project["snapshots"][-1]
    db = before["database"]
    assert len(db["execution_attempts"]) == len(db["research_artifacts"]) == 2
    assert len(db["research_observations"]) == 1
    observation = json.loads(db["research_observations"][0]["record_json"])
    assert observation["validation"]["status"] == status
    assert observation["input_artifact_ids"] == [source["artifact_id"]]
    assert observation["input_artifact_bindings"][source["artifact_id"]] == {
        key: source[key] for key in ("producer", "spec_fingerprint", "content_sha256")}
    if status == "invalid":
        assert observation["validation"]["reason_code"] == "capacity_overload"
        assert "value" not in observation["values"]
    evaluator = next(row for row in db["execution_attempts"] if row["task_id"] == "idea-evaluator")
    terminal = json.loads(evaluator["terminal_json"])
    assert evaluator["state"] == "TERMINAL" and terminal["outcome"] == "completed"
    assert terminal["observation_ids"] == [observation["observation_id"]]
    assert {r["task_id"]: r["state"] for r in db["cpu_action_reservations"]} == {
        "idea-producer": "SETTLED", "idea-evaluator": "BOUND"}
    evidence = project["calls"][-1]["metadata"][1]
    assert evidence["ref"]["task_id"] == "idea-evaluator" and evidence["effect_confirmed"]
    old_events = _events(project["root"])
    ready = [event for event in old_events if event["event"] == "native_ready"]
    interpreted = [event for event in old_events if event["event"] == "interpret"]
    assert len(ready) == len(interpreted) == 2
    assert {event["operation"] for event in interpreted} == {"produce", "evaluate"}
    assert len({(event["binding"]["worker"]["pid"], event["binding"]["worker"]["start_ticks"])
                for event in ready}) == 2

    with monkeypatch.context() as patch:
        patch.setenv("ORZE_F_RECOVERY_DOMAIN_NO_CALLBACKS", "1")
        run_cli(project, "domain-recover-only", child_module=module)
        run_cli(project, "domain-recover-idle-again", child_module=module)
    after = project["snapshots"][-1]
    for table in ("ideas", "idea_state", "idea_transitions", "idea_stage_state", "idea_stage_transitions",
                  "execution_attempts", "research_artifacts", "research_observations"):
        assert after["database"][table] == db[table]
    assert after["files"] == before["files"]
    assert _events(project["root"]) == old_events
    assert all(row["state"] == "SETTLED" for row in after["database"]["cpu_action_reservations"])
    assert sum(int(json.loads(row["permit_json"])["reserved_nanoseconds"])
               for row in after["database"]["cpu_action_reservations"]) == 4000000000
    for old, new in zip(db["cpu_action_reservations"], after["database"]["cpu_action_reservations"]):
        assert {k: old[k] for k in ("reservation_id", "permit_json", "ref_json")} == {
            k: new[k] for k in ("reservation_id", "permit_json", "ref_json")}
    assert len(project["calls"]) == 4
    assert len({(c["controller_binding"]["worker"]["pid"], c["controller_binding"]["worker"]["start_ticks"])
                for c in project["calls"]}) == 4
    with closing(IdeaLake(project["root"] / "lake.db")) as lake:
        row = lake.conn.execute("SELECT state,summary_json FROM cpu_action_recovery WHERE scope=?",
                               (project["cfg"]["results_dir"],)).fetchone()
        assert row[0] == "COMPLETE"
        summary = json.loads(row[1])
        assert summary["settled"] == [next(r["reservation_id"] for r in db["cpu_action_reservations"]
                                           if r["task_id"] == "idea-evaluator")]


def _domain_child():
    from unittest.mock import patch
    from cpu_terminal_recovery_helpers import _child_main
    from examples.holdout.scheduling import SchedulingDomain
    from orze.core.research_interfaces import register_domain
    from orze.engine import native_cpu_action as native
    original_prepare, original_interpret = SchedulingDomain.prepare, SchedulingDomain.interpret
    actual_prepare = native.prepare_supervised
    forbidden = os.environ.get("ORZE_F_RECOVERY_DOMAIN_NO_CALLBACKS") == "1"

    def audit(record):
        path = Path.cwd() / "domain-events.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def prepare(self, request, sources):
        if forbidden:
            raise AssertionError("recovery must not call Domain.prepare")
        return original_prepare(self, request, sources)

    def interpret(self, prepared, envelope):
        if forbidden:
            raise AssertionError("recovery must not call Domain.interpret")
        result = original_interpret(self, prepared, envelope)
        audit({"event": "interpret", "operation": prepared["action"]["inputs"]["payload"]["operation"]})
        return result

    def observed_prepare(*args, **kwargs):
        handle = actual_prepare(*args, **kwargs)
        try:
            audit({"event": "native_ready", "binding": handle.binding})
        except BaseException:
            handle.stop(timeout=10)
            raise
        return handle

    register_domain("schedule_holdout", "acceptance.schedule.v1", SchedulingDomain)
    with patch.object(SchedulingDomain, "prepare", prepare), patch.object(SchedulingDomain, "interpret", interpret), \
            patch.object(native, "prepare_supervised", observed_prepare):
        return _child_main()


if __name__ == "__main__":
    raise SystemExit(_domain_child())
