"""Actual Policy-generated CPU experiments and source-bound analysis, no inbox producer."""
import copy
import json

from test_cpu_domain_product import request
from test_cpu_product_loop import project
from test_cpu_replication_product import read


def proposal(identity, declared):
    return {"kind": "Propose", "request_id": "plan-" + identity, "task_id": identity,
            "reason": "answer the next explicit research question", "domain_request": declared}


def register(monkeypatch, cfg, implementation):
    from orze.core import research_interfaces as api
    monkeypatch.setattr(api, "_POLICIES", dict(api._POLICIES))
    api.register_policy("proposal_product_fixture", "proposal.product.v1", implementation)
    cfg["action_policy"]["kind"] = "proposal_product_fixture"
    cfg["action_domain"] = {"version": 1, "kind": "command", "config": {}}


def test_cli_policy_proposal_only_queues_and_replays_after_restarting(project, monkeypatch):
    root, cfg, run = project
    selected = proposal("idea-new", request("pass"))
    seen = []

    class Policy:
        def __init__(self, declaration):
            pass

        def decide(self, snapshot, budget):
            seen.append(copy.deepcopy(snapshot))
            return selected

    register(monkeypatch, cfg, Policy)
    assert run() == 0
    assert read(root, "SELECT idea_id,status FROM ideas") == [("idea-new", "queued")]
    assert read(root, "SELECT COUNT(*) FROM cpu_action_reservations") == [(0,)]
    assert read(root, "SELECT name FROM sqlite_master WHERE name='execution_attempts'") == []
    assert not (root / "results" / "idea-new" / "claim.json").exists()
    assert run() == 0
    assert read(root, "SELECT idea_id,status FROM ideas") == [("idea-new", "queued")]
    assert read(root, "SELECT COUNT(*) FROM cpu_action_reservations") == [(0,)]
    assert len(seen[1]["recorded_proposals"]["results"]) == 1
    assert seen[1]["recorded_proposals"]["results"][0]["request_id"] == "plan-idea-new"
    assert seen[1]["recorded_proposals"]["results"][0]["status"] == "inserted"


def test_policy_builds_experiment_then_analysis_from_new_artifact_ids(project, monkeypatch):
    from orze.core import research_interfaces as api
    root, cfg, run = project
    cfg["execution"]["wall_budget_seconds"] = 6
    decisions, seen = [], []
    source_request = request("from pathlib import Path\nPath('value.json').write_text('7')",
        outputs={"value": {"path": "value.json", "max_bytes": 64}})
    analysis_program = """import json, os
from pathlib import Path
fds = json.loads(os.environ['ORZE_ACTION_SOURCE_FDS'])
values = [json.loads(os.read(fd, 64)) for fd in fds.values()]
assert values == [7]
record = {'version': 1, 'observations': [{'name': 'square',
    'values': {'value': values[0] ** 2},
    'validation': {'status': 'valid', 'reason_code': 'explicit_arithmetic'},
    'comparison_scope': 'square-v1'}]}
Path('result.json').write_text(json.dumps(record))
"""

    class Domain(api.CommandDomain):
        def prepare(self, declared, sources):
            if "result_output" in declared["payload"]:
                return api.JsonObservationDomain.prepare(self, declared, sources)
            return super().prepare(declared, sources)

        def interpret(self, prepared, envelope):
            if prepared["observation"] is None:
                return ()
            return api.JsonObservationDomain.interpret(self, prepared, envelope)

    class Policy:
        def __init__(self, declaration):
            pass

        def decide(self, snapshot, budget):
            seen.append(copy.deepcopy(snapshot))
            results = snapshot["recorded_evidence"]["results"]
            if snapshot["queue"]:
                decision = {"kind": "Execute", "task_id": snapshot["queue"][0]["idea_id"]}
            elif not results:
                decision = proposal("idea-source", source_request)
            elif len(results) == 1:
                ids = [item["artifact_id"] for item in results[0]["artifact_records"]]
                declared = request(analysis_program, sources=ids, observation=True,
                    outputs={"result": {"path": "result.json", "max_bytes": 4096}})
                declared["purpose"] = "analyze the actual newly produced value"
                decision = proposal("idea-analysis", declared)
            else:
                decision = {"kind": "Stop", "reason": "analysis_recorded", "wakeup": None}
            decisions.append(copy.deepcopy(decision))
            return decision

    register(monkeypatch, cfg, Policy)
    monkeypatch.setattr(api, "_DOMAINS", dict(api._DOMAINS))
    api.register_domain("proposal_mixed_fixture", "proposal.mixed.v1", Domain)
    cfg["action_domain"]["kind"] = "proposal_mixed_fixture"
    assert run(once=False) == 0
    assert [item["kind"] for item in decisions] == ["Propose", "Execute", "Propose", "Execute", "Stop"]
    assert read(root, "SELECT idea_id,status FROM ideas ORDER BY idea_id") == [
        ("idea-analysis", "completed"), ("idea-source", "completed")]
    assert read(root, "SELECT state FROM cpu_action_reservations") == [("SETTLED",)] * 2
    assert read(root, "SELECT COUNT(*) FROM execution_attempts") == [(2,)]
    assert len(seen[-1]["recorded_proposals"]["results"]) == 2
    records = [json.loads(row[0]) for row in read(root, "SELECT record_json FROM research_observations")]
    assert len(records) == 1
    assert records[0]["values"] == {"value": 49}
    assert records[0]["validation"]["status"] == "valid"
    source = next(item for item in seen[-1]["recorded_evidence"]["results"]
                  if item["ref"]["task_id"] == "idea-source")
    original = source["artifact_records"][0]
    assert records[0]["input_artifact_ids"] == [original["artifact_id"]]
    assert records[0]["input_artifact_bindings"][original["artifact_id"]] == {
        key: original[key] for key in ("producer", "spec_fingerprint", "content_sha256")}
    assert not (root / "ideas.md").exists() or (root / "ideas.md").read_text().strip() == ""
