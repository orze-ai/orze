"""Same-request CPU replicas through existing CLI and actual Orze loop.

Private supervised CPU programs only. No alternate runner, training shim,
configuration salt, GPU/provider or fake successful completion.
"""
import json
import sqlite3
import sys

import pytest
import yaml

from test_cpu_product_loop import project, task
from test_cpu_domain_product import request, submit


def read(root, query):
    with sqlite3.connect(root / "lake.db") as conn:
        return conn.execute(query).fetchall()


def source(project, *, domain):
    root, cfg, run = project
    cfg["execution"]["wall_budget_seconds"] = 12
    if domain:
        cfg["action_domain"] = {"version": 1, "kind": "json_observations", "config": {}}
        claims = [{"name": "minimum", "values": {"value": -1},
                   "validation": {"status": "valid", "reason_code": "checked"},
                   "comparison_scope": "same-protocol"},
                  {"name": "unknown", "values": {},
                   "validation": {"status": "unknown", "reason_code": "not_measured"},
                   "comparison_scope": "same-protocol"}]
        program = "from pathlib import Path\nPath('result.json').write_text(" + repr(
            json.dumps({"version": 1, "observations": claims})) + ")"
        submit(root, "idea-0001", request(program, outputs={
            "result": {"path": "result.json", "max_bytes": 4096}}, observation=True))
    else:
        task(root, outputs={}, program="pass")
    assert run() == 0
    return root, cfg, run


def replicate(root, monkeypatch, capsys, *, key="repeat-1", reason="explicit repeat"):
    import orze.cli as cli
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["orze", "replicate", "idea-0001",
        "--request-id", key, "--reason", reason, "-c", str(root / "orze.yaml")])
    result = cli.main()
    output = json.loads(capsys.readouterr().out)
    assert result == 0, output
    return output


@pytest.mark.parametrize("domain", [False, True])
def test_same_request_cli_replica_retains_spec_and_creates_new_occurrences(
        project, monkeypatch, capsys, domain):
    root, cfg, run = source(project, domain=domain)
    original = read(root, "SELECT config FROM ideas WHERE idea_id='idea-0001'")[0][0]
    first = replicate(root, monkeypatch, capsys)
    assert first["status"] == "created" and first["task_id"] != "idea-0001"
    assert read(root, "SELECT COUNT(*) FROM execution_attempts") == [(1,)]
    assert read(root, "SELECT COUNT(*) FROM cpu_action_reservations") == [(1,)]
    again = replicate(root, monkeypatch, capsys)
    assert again == {**first, "status": "already_requested"}
    assert run() == 0
    assert replicate(root, monkeypatch, capsys) == again
    second = replicate(root, monkeypatch, capsys, key="repeat-2")
    assert second["task_id"] != first["task_id"]
    assert run() == 0
    tasks = read(root, "SELECT idea_id,kind,config,status FROM ideas")
    attempts = read(root, "SELECT task_id,attempt_id,binding_json,terminal_json FROM execution_attempts")
    assert len(tasks) == len(attempts) == 3
    assert all(row[1:] == ("native_cpu_action", original, "completed") for row in tasks)
    assert len({row[1] for row in attempts}) == 3
    bindings = [json.loads(row[2]) for row in attempts]
    assert len({row["action_sha256"] for row in bindings}) == 1
    assert len({row["artifact_publication"]["spec_fingerprint"] for row in bindings}) == 1
    assert read(root, "SELECT state FROM cpu_action_reservations") == [("SETTLED",)] * 3
    assert (root / "ideas.md").read_text().strip() == ""
    if domain:
        observations = [json.loads(row[0]) for row in read(root, "SELECT record_json FROM research_observations")]
        assert len(observations) == len({row["observation_id"] for row in observations}) == 6
        assert len({row["evaluator"]["attempt_id"] for row in observations}) == 3
        assert len({row["spec_fingerprint"] for row in observations}) == 1
        assert len({row["protocol_fingerprint"] for row in observations}) == 1
        assert sorted(row["validation"]["status"] for row in observations) == ["unknown"] * 3 + ["valid"] * 3
        assert len(read(root, "SELECT artifact_id FROM research_artifacts")) == 3


def test_policy_can_request_one_identical_cpu_replica_then_execute_it(project, monkeypatch):
    from orze.core import research_interfaces as api
    root, cfg, run = source(project, domain=True)
    monkeypatch.setattr(api, "_POLICIES", dict(api._POLICIES))
    decisions = []

    class Policy:
        def __init__(self, declaration):
            pass

        def decide(self, snapshot, budget):
            results = snapshot["recorded_evidence"]["results"]
            if len(results) == 2:
                decision = {"kind": "Stop", "reason": "explicit_repeat_recorded", "wakeup": None}
            elif snapshot["queue"]:
                decision = {"kind": "Execute", "task_id": snapshot["queue"][0]["idea_id"]}
            else:
                decision = {"kind": "Replicate", "source_ref": results[0]["ref"],
                            "request_id": "policy-repeat", "reason": "check same request"}
            decisions.append(decision)
            return decision

    api.register_policy("fixture_replica_policy", "fixture.replica_policy.v1", Policy)
    cfg["action_policy"]["kind"] = "fixture_replica_policy"
    assert run(once=False) == 0
    assert [item["kind"] for item in decisions] == ["Replicate", "Execute", "Stop"]
    assert read(root, "SELECT COUNT(*) FROM replication_requests") == [(1,)]
    assert read(root, "SELECT COUNT(*) FROM execution_attempts") == [(2,)]
    assert read(root, "SELECT COUNT(DISTINCT config) FROM ideas") == [(1,)]
    assert read(root, "SELECT state FROM cpu_action_reservations") == [("SETTLED",)] * 2


def test_ordinary_identical_proposal_still_stays_unadmitted(project):
    root, cfg, run = source(project, domain=True)
    original = read(root, "SELECT config FROM ideas WHERE idea_id='idea-0001'")[0][0]
    text = "## idea-ordinary-copy: ordinary duplicate\n\n\x60\x60\x60yaml\n" + original + "\x60\x60\x60\n"
    (root / "ideas.md").write_text(text)
    assert run() == 0
    assert (root / "ideas.md").read_text() == text
    assert read(root, "SELECT idea_id FROM ideas") == [("idea-0001",)]
    assert read(root, "SELECT COUNT(*) FROM execution_attempts") == [(1,)]
    assert read(root, "SELECT COUNT(*) FROM cpu_action_reservations") == [(1,)]


def test_cpu_replication_cli_accepts_explicitly_absent_training_paths(project, monkeypatch, capsys):
    root, cfg, run = source(project, domain=False)
    cfg.update(train_script=None, base_config=None)
    (root / "orze.yaml").write_text(yaml.safe_dump(cfg))
    result = replicate(root, monkeypatch, capsys)
    assert result["status"] == "created"
    assert read(root, "SELECT COUNT(*) FROM execution_attempts") == [(1,)]
    assert read(root, "SELECT COUNT(*) FROM ideas") == [(2,)]


@pytest.mark.parametrize("fault", ["request_record", "domain_implementation"])
def test_cpu_replication_dispatch_rechecks_before_claim_and_budget(project, monkeypatch, capsys, fault):
    from orze.core import research_interfaces as api
    root, cfg, run = source(project, domain=True)
    result = replicate(root, monkeypatch, capsys)
    if fault == "request_record":
        with sqlite3.connect(root / "lake.db") as conn:
            conn.execute("UPDATE replication_requests SET record_json='{}'")
    else:
        monkeypatch.setattr(api, "_DOMAINS", dict(api._DOMAINS))
        api._DOMAINS["json_observations"] = ("fixture.new_domain_version", api.JsonObservationDomain)
    assert run() == 75
    assert read(root, "SELECT COUNT(*) FROM execution_attempts") == [(1,)]
    assert read(root, "SELECT COUNT(*) FROM cpu_action_reservations") == [(1,)]
    assert read(root, "SELECT status FROM ideas WHERE idea_id!='idea-0001'") == [("queued",)]
    assert not (root / "results" / result["task_id"] / "claim.json").exists()


def test_explicit_cpu_replication_still_requires_its_own_wall_allowance(project, monkeypatch, capsys):
    root, cfg, run = project
    cfg["execution"]["wall_budget_seconds"] = 2
    task(root, outputs={}, program="pass")
    assert run() == 0
    result = replicate(root, monkeypatch, capsys)
    assert run() == 0
    assert read(root, "SELECT COUNT(*) FROM execution_attempts") == [(1,)]
    assert read(root, "SELECT state FROM cpu_action_reservations") == [("SETTLED",)]
    assert read(root, "SELECT status FROM ideas WHERE idea_id!='idea-0001'") == [("queued",)]
    stop = json.loads(read(root, "SELECT stop_json FROM cpu_action_scopes")[0][0])
    assert stop == {"kind": "Stop", "reason": "wall_envelope_exhausted", "wakeup": None}
    assert not (root / "results" / result["task_id"] / "claim.json").exists()
    assert run() == 75


def test_cli_same_key_changed_reason_is_conflict_without_another_task(project, monkeypatch, capsys):
    import orze.cli as cli
    root, cfg, run = source(project, domain=False)
    replicate(root, monkeypatch, capsys)
    monkeypatch.setattr(sys, "argv", ["orze", "replicate", "idea-0001",
        "--request-id", "repeat-1", "--reason", "different reason", "-c", str(root / "orze.yaml")])
    assert cli.main() == 2
    assert "conflict" in json.loads(capsys.readouterr().out)["error"]
    assert read(root, "SELECT COUNT(*) FROM replication_requests") == [(1,)]
    assert read(root, "SELECT COUNT(*) FROM ideas") == [(2,)]
    assert read(root, "SELECT COUNT(*) FROM execution_attempts") == [(1,)]
