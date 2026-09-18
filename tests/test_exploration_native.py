"""The exploration driver uses the existing admitted/supervised CPU executor."""
import json
import sqlite3

from orze.research.exploration import replay, run_online
from orze.research.exploration_policies import ParallelRefine
from test_cpu_product_loop import project
from test_cpu_domain_product import request, submit
from test_exploration import spec


def test_branch_context_roundtrip_through_native_cpu_and_replay(project):
    root, cfg, run = project
    cfg["execution"]["wall_budget_seconds"] = 20
    cfg["action_domain"] = {"version": 1, "kind": "json_observations", "config": {}}
    completed = []

    def execute(contexts):
        outcomes = {}
        for context in contexts:
            action = context["action"]
            task_id = "idea-" + action["id"]
            code = """import json, os
from pathlib import Path
context = json.load(os.fdopen(os.dup(int(os.environ['ORZE_ACTION_INPUT_FD']))))
assert len(context['history']) == context['action']['step']
assert all(n['branch'] == context['action']['branch'] for n in context['history'])
value = context['action']['step'] + 1
Path('result.json').write_text(json.dumps({'version': 1, 'observations': [
 {'name': 'branch-value', 'values': {'quality': value},
  'validation': {'status': 'valid', 'reason_code': 'fixture_checked'},
  'comparison_scope': 'branch-fixture-v1'}]}))
"""
            submit(root, task_id, request(code, inputs=context, observation=True,
                                         outputs={"result": {"path": "result.json", "max_bytes": 4096}}))
            assert run() == 0
            with sqlite3.connect(root / "lake.db") as db:
                row = db.execute("SELECT terminal_json FROM execution_attempts WHERE task_id=?",
                                 (task_id,)).fetchone()
                terminal = json.loads(row[0])
                assert terminal["outcome"] == "completed"
                assert terminal["process_tree"]["wait_proof"] == "ECHILD_WALL"
                records = [json.loads(r[0]) for r in db.execute("SELECT record_json FROM research_observations")]
                observation = next(r for r in records if r["evaluator"]["task_id"] == task_id)
                assert observation["validation"]["status"] == "valid"
            completed.append(task_id)
            outcomes[action["id"]] = {"score": observation["values"]["quality"], "status": "ok",
                "artifact": {"artifact_ids": terminal["artifact_ids"]},
                "feedback": {"observation": observation}, "cost": 0, "seconds": 0}
        return outcomes

    trace = run_online(spec(branches=2, depth=2, calls=4, workers=1), ParallelRefine(), execute,
                       root / "discovery")
    assert len(completed) == 4
    with sqlite3.connect(root / "lake.db") as db:
        assert db.execute("SELECT state FROM cpu_action_reservations").fetchall() == [("SETTLED",)] * 4
    assert replay(trace, ParallelRefine())["metrics"] == trace["metrics"]
    assert len(completed) == 4  # Replay did not invoke the executor.
