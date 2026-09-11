"""Registered domains and policies through the actual foreground CLI loop."""
import json
import sqlite3
import sys

import pytest
import yaml

from test_cpu_product_loop import project


def submit(root, identity, request):
    (root / "ideas.md").write_text("## " + identity + ": Domain action\n\n```yaml\n" +
        yaml.safe_dump({"kind": "native_cpu_action", "domain_request": request}) + "```\n")


def request(program, *, outputs=None, inputs=None, sources=None, observation=False):
    payload = {"command": [sys.executable, "-c", program]}
    if observation:
        payload.update(specification={"subject": "declared generic fixture"},
                       protocol={"measurement": "explicit JSON claims"}, result_output="result")
    return {"version": 1, "purpose": "exercise replaceable autoresearch domain",
            "inputs": inputs or {}, "timeout_seconds": 2,
            "outputs": outputs or {}, "input_artifact_ids": sources or [], "payload": payload}


def rows(root):
    with sqlite3.connect(root / "lake.db") as conn:
        return conn.execute("SELECT task_id,state,binding_json,terminal_json FROM execution_attempts ORDER BY task_id").fetchall()


@pytest.mark.parametrize("count", [None, 0, 2])
def test_cli_selects_real_domains_and_explicit_observation_counts(project, count):
    root, cfg, run = project
    kind = "command" if count is None else "json_observations"
    cfg["action_domain"] = {"version": 1, "kind": kind, "config": {}}
    claims = [{"name": "measure-" + str(i), "values": {"value": i},
               "validation": {"status": "valid", "reason_code": "fixture_checked"},
               "comparison_scope": "fixture-v1"} for i in range(count or 0)]
    body = "pass" if count is None else "from pathlib import Path\nPath('result.json').write_text(" + repr(json.dumps({"version": 1, "observations": claims})) + ")"
    outputs = {} if count is None else {"result": {"path": "result.json", "max_bytes": 4096}}
    submit(root, "idea-domain", request(body, outputs=outputs, observation=count is not None))
    assert run() == 0
    row = rows(root)[0]
    binding, terminal = json.loads(row[2]), json.loads(row[3])
    assert row[:2] == ("idea-domain", "TERMINAL")
    assert binding["domain_run"]["domain_kind"] == kind
    assert terminal["outcome"] == "completed"
    assert len(terminal["observation_ids"]) == (count or 0)
    assert terminal["process_tree"]["wait_proof"] == "ECHILD_WALL"
    with sqlite3.connect(root / "lake.db") as conn:
        if count:
            records = [json.loads(r[0]) for r in conn.execute("SELECT record_json FROM research_observations")]
            assert len(records) == count
            assert all(r["schema"] == 2 and r["evaluator"]["phase"] == "action" for r in records)
        assert conn.execute("SELECT state FROM cpu_action_reservations").fetchall() == [("SETTLED",)]


def test_missing_analysis_source_returns_hold_without_claim_or_reservation(project):
    root, cfg, run = project
    cfg["action_domain"] = {"version": 1, "kind": "command", "config": {}}
    submit(root, "idea-missing", request("raise AssertionError('must not run')", sources=["missing-artifact"]))
    assert run() == 75
    with sqlite3.connect(root / "lake.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM cpu_action_reservations").fetchone()[0] == 0
        assert conn.execute("SELECT status FROM ideas").fetchall() == [("queued",)]
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='execution_attempts'").fetchall() == []
    assert not (root / "results" / "idea-missing" / "claim.json").exists()


def test_registered_policy_can_stop_a_nonempty_queue_without_execution(project, monkeypatch):
    from orze.core import research_interfaces as api
    root, cfg, run = project
    monkeypatch.setattr(api, "_POLICIES", dict(api._POLICIES))
    seen = []

    class StopPolicy:
        def __init__(self, declaration):
            assert declaration["config"] == {"reason": "declared_goal_satisfied"}

        def decide(self, snapshot, budget):
            seen.append(snapshot)
            return {"kind": "Stop", "reason": "declared_goal_satisfied", "wakeup": None}

    api.register_policy("fixture_stop", "fixture.stop_policy.v1", StopPolicy)
    cfg["action_policy"].update(kind="fixture_stop", config={"reason": "declared_goal_satisfied"})
    cfg["action_domain"] = {"version": 1, "kind": "command", "config": {}}
    submit(root, "idea-do-not-run", request("raise AssertionError('must not run')"))
    assert run() == 0
    assert len(seen) == 1 and seen[0]["queue"][0]["idea_id"] == "idea-do-not-run"
    assert seen[0]["recorded_evidence"]["results"] == []
    with sqlite3.connect(root / "lake.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM cpu_action_reservations").fetchone()[0] == 0
        assert json.loads(conn.execute("SELECT stop_json FROM cpu_action_scopes").fetchone()[0])["reason"] == "declared_goal_satisfied"


def test_unavailable_policy_view_returns_hold_before_any_execute(project, monkeypatch):
    from orze.engine import cpu_policy_evidence as evidence
    root, cfg, run = project
    cfg["action_domain"] = {"version": 1, "kind": "command", "config": {}}
    submit(root, "idea-no-view", request("raise AssertionError('must not run')"))
    actual = evidence.recorded_evidence
    reached = []

    def open_transaction(lake, results, **kwargs):
        lake.conn.execute("BEGIN IMMEDIATE")
        try:
            reached.append(True)
            return actual(lake, results, **kwargs)
        finally:
            lake.conn.rollback()

    monkeypatch.setattr(evidence, "recorded_evidence", open_transaction)
    assert run() == 75
    assert reached == [True]
    with sqlite3.connect(root / "lake.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM cpu_action_reservations").fetchone()[0] == 0
        assert conn.execute("SELECT status FROM ideas").fetchall() == [("queued",)]


@pytest.mark.parametrize("component", ["policy", "domain"])
def test_registered_callback_exception_returns_hold_before_execution(project, monkeypatch, component):
    from orze.core import research_interfaces as api
    root, cfg, run = project
    monkeypatch.setattr(api, "_POLICIES", dict(api._POLICIES))
    monkeypatch.setattr(api, "_DOMAINS", dict(api._DOMAINS))

    class BrokenPolicy:
        def __init__(self, declaration):
            pass

        def decide(self, snapshot, budget):
            raise RuntimeError("private plugin fixture failure")

    class BrokenDomain(api.CommandDomain):
        def prepare(self, request, sources):
            raise RuntimeError("private plugin fixture failure")

    cfg["action_domain"] = {"version": 1, "kind": "command", "config": {}}
    if component == "policy":
        api.register_policy("fixture_broken_policy", "fixture.broken_policy.v1", BrokenPolicy)
        cfg["action_policy"]["kind"] = "fixture_broken_policy"
    else:
        api.register_domain("fixture_broken_domain", "fixture.broken_domain.v1", BrokenDomain)
        cfg["action_domain"]["kind"] = "fixture_broken_domain"
    submit(root, "idea-broken-plugin", request("raise AssertionError('must not run')"))
    assert run() == 75
    with sqlite3.connect(root / "lake.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM cpu_action_reservations").fetchone()[0] == 0
        assert conn.execute("SELECT status FROM ideas").fetchall() == [("queued",)]


def test_policy_reads_confirmed_evidence_and_analyzes_two_distinct_specs(project, monkeypatch):
    from orze.core import research_interfaces as api
    root, cfg, run = project
    cfg["execution"]["wall_budget_seconds"] = 12
    cfg["action_domain"] = {"version": 1, "kind": "command", "config": {}}
    for suffix, value in [("a", 3), ("b", 5)]:
        submit(root, "idea-source-" + suffix, request(
            "from pathlib import Path\nPath('value.json').write_text(" + repr(str(value)) + ")",
            outputs={"value": {"path": "value.json", "max_bytes": 64}}))
        assert run() == 0
    source_rows = rows(root)
    source_ids = [json.loads(r[3])["artifact_ids"][0] for r in source_rows]
    with sqlite3.connect(root / "lake.db") as conn:
        source_records = [json.loads(r[0]) for r in conn.execute("SELECT record_json FROM research_artifacts")]
    assert len({r["spec_fingerprint"] for r in source_records}) == 2
    observed = []
    monkeypatch.setattr(api, "_POLICIES", dict(api._POLICIES))

    class AnalyzePolicy:
        def __init__(self, declaration):
            pass

        def decide(self, snapshot, budget):
            observed.append(snapshot["recorded_evidence"])
            available = {a["artifact_id"] for r in snapshot["recorded_evidence"]["results"]
                         for a in r["artifact_records"]}
            assert set(source_ids) <= available
            return {"kind": "Execute", "task_id": snapshot["queue"][0]["idea_id"]}

    api.register_policy("fixture_analysis", "fixture.analysis_policy.v1", AnalyzePolicy)
    cfg["action_policy"].update(kind="fixture_analysis")
    cfg["action_domain"]["kind"] = "json_observations"
    program = """import json, os
from pathlib import Path
fds = json.loads(os.environ['ORZE_ACTION_SOURCE_FDS'])
values = [json.loads(os.read(fd, 64)) for fd in fds.values()]
assert sorted(values) == [3, 5]
claims = [{'name': name, 'values': {'value': value},
           'validation': {'status': 'valid', 'reason_code': 'declared_arithmetic'},
           'comparison_scope': 'sum-and-count-v1'}
          for name, value in [('sum', sum(values)), ('count', len(values))]]
Path('result.json').write_text(json.dumps({'version': 1, 'observations': claims}))
"""
    declared = request(program, sources=source_ids, observation=True,
                       outputs={"result": {"path": "result.json", "max_bytes": 4096}})
    for suffix in ("a", "b"):
        submit(root, "idea-analysis-" + suffix, declared)
        assert run() == 0
    assert len(observed) == 2
    with sqlite3.connect(root / "lake.db") as conn:
        observations = [json.loads(r[0]) for r in conn.execute("SELECT record_json FROM research_observations")]
        assert conn.execute("SELECT COUNT(*) FROM cpu_action_reservations WHERE state='SETTLED'").fetchone()[0] == 4
    assert len(observations) == 4
    assert len({r["observation_id"] for r in observations}) == 4
    assert len({r["evaluator"]["attempt_id"] for r in observations}) == 2
    assert len({r["spec_fingerprint"] for r in observations}) == 1
    for record in observations:
        assert record["input_artifact_ids"] == source_ids
        assert record["values"]["value"] == (8 if record["name"] == "sum" else 2)
        for source in source_records:
            actual = record["input_artifact_bindings"][source["artifact_id"]]
            assert actual == {key: source[key] for key in ("producer", "spec_fingerprint", "content_sha256")}
            assert actual["spec_fingerprint"] != record["spec_fingerprint"]
