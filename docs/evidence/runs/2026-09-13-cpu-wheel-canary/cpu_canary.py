"""One actual Core-only installed-wheel CPU canary, not an efficiency trial."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

import yaml
import orze
from orze.idea_lake import IdeaLake
from orze.engine.supervised_process import prepare_supervised, SupervisionUncertain

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "cpu-canary"
assert sys.prefix == str(ROOT / "core-only-venv")
assert importlib.util.find_spec("orze_pro") is None
assert Path(orze.__file__).is_relative_to(ROOT / "core-only-venv")
PROJECT.mkdir()
cfg = {"execution": {"version": 2, "resource": "cpu", "slots": 1, "wall_budget_seconds": None},
       "results_dir": str(PROJECT / "results"), "idea_lake_db": str(PROJECT / "lake.db"),
       "ideas_file": str(PROJECT / "ideas.md"), "min_disk_gb": 0,
       "action_policy": {"version": 1, "kind": "queue", "idle": "wait", "wait_seconds": .05}}
config_path = PROJECT / "orze.yaml"
config_path.write_text(yaml.safe_dump(cfg))
worker = """import json
from pathlib import Path
Path('answer.json').write_text(json.dumps({'sum':sum(range(1,11)),'count':10}),encoding='utf-8')
"""
action = {"version": 1, "adapter": "command", "purpose": "installed wheel CPU canary",
          "inputs": {}, "command": [sys.executable, "-c", worker], "timeout_seconds": 2,
          "outputs": {"answer": {"path": "answer.json", "max_bytes": 128}}}
lake = IdeaLake(PROJECT / "lake.db")
try:
    admitted = lake.insert("idea-wheel-canary", "wheel canary",
        json.dumps({"kind": "native_cpu_action", "action": action}, sort_keys=True),
        "", status="queued", kind="native_cpu_action", if_absent=True)
finally:
    lake.close()
assert admitted["status"] == "inserted"
env = {"PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin", "LANG": "C.UTF-8",
       "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
argv = [str(ROOT / "core-only-venv" / "bin" / "orze"), "-c", str(config_path), "--once"]
process = None
started = time.monotonic()
with (PROJECT / "cli.stdout").open("wb") as stdout, (PROJECT / "cli.stderr").open("wb") as stderr:
    try:
        try:
            process = prepare_supervised(argv, identity={"scope": str(PROJECT), "release_canary": True},
                cwd=str(PROJECT), env=env, stdout=stdout, stderr=stderr)
        except SupervisionUncertain as exc:
            process = exc.process
            raise
        process.start()
        code = process.wait(timeout=20)
        binding = process.binding
        closure = process.closure_receipt()
    finally:
        if process is not None and process.poll() is None:
            process.stop(timeout=10)
finished = time.monotonic()
conn = sqlite3.connect((PROJECT / "lake.db").as_uri()+"?mode=ro", uri=True)
conn.row_factory = sqlite3.Row
try:
    tables = {}
    names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for name in ("ideas", "idea_state", "idea_transitions", "execution_attempts",
                 "research_artifacts", "research_observations", "cpu_action_scopes",
                 "cpu_action_reservations", "cpu_action_decisions", "cpu_action_recovery"):
        tables[name] = ([dict(r) for r in conn.execute("SELECT * FROM main." + name + " ORDER BY rowid")]
                        if name in names else [])
finally:
    conn.close()
report = {"scope": "one installed Core-only command-action canary; no experimental comparison",
          "python": sys.executable, "orze_module": orze.__file__, "version": importlib.metadata.version("orze"), "cfg": cfg, "environment": env,
          "pro_present": False, "argv": argv, "cwd": str(PROJECT), "exit_code": code,
          "controller_binding": binding, "controller_closure": closure,
          "wall_seconds": finished-started, "admission": admitted, "database": tables,
          "stdout": (PROJECT/"cli.stdout").read_text(), "stderr": (PROJECT/"cli.stderr").read_text(),
          "license_bypassed": False, "production_deployed": False}
(ROOT / "cpu-canary-report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
assert code == 0 and closure["event"] == "TREE_CLOSED" and closure["wait_proof"] == "ECHILD_WALL"
assert closure["forced_cleanup"] is False and closure["stop_requested"] is False
assert len(tables["execution_attempts"]) == len(tables["cpu_action_reservations"]) == 1
attempt = tables["execution_attempts"][0]
terminal = json.loads(attempt["terminal_json"])
assert attempt["state"] == "TERMINAL" and terminal["outcome"] == "completed"
assert terminal["process_tree"]["event"] == "TREE_CLOSED"
assert terminal["process_tree"]["wait_proof"] == "ECHILD_WALL"
assert terminal["process_tree"]["binding"]["protocol"] == "orze.linux_subreaper.v2"
assert terminal["runtime_lease"]["status"] == "authorized"
assert tables["cpu_action_reservations"][0]["state"] == "SETTLED"
permit = json.loads(tables["cpu_action_reservations"][0]["permit_json"])
assert permit["reserved_nanoseconds"] == "2000000000"
assert len(tables["research_artifacts"]) == 1 and tables["research_observations"] == []
artifact = json.loads(tables["research_artifacts"][0]["record_json"])
artifact_path = Path(artifact["path"])
assert artifact_path.is_relative_to(PROJECT)
payload = artifact_path.read_bytes()
assert json.loads(payload) == {"sum": 55, "count": 10}
assert hashlib.sha256(payload).hexdigest() == artifact["content_sha256"]
effect_path = PROJECT / "results" / "idea-wheel-canary" / "_execution_effects" / attempt["attempt_id"] / "committed.json"
effect_raw = effect_path.read_bytes()
report.update(passed=True, terminal=terminal, artifact=artifact,
              artifact_bytes=payload.decode(), effect_committed_bytes=effect_raw.decode(),
              effect_committed_sha256=hashlib.sha256(effect_raw).hexdigest(),
              effect_guard_absent=not (effect_path.parents[2] / "_attempt_effect.lock").exists())
assert report["effect_guard_absent"] is True
from orze.core import cpu_action_budget
scope_row, = tables["cpu_action_scopes"]
scope = json.loads(scope_row["binding_json"])
lake = IdeaLake(PROJECT / "lake.db")
try:
    budget = cpu_action_budget.snapshot(lake, scope)
finally:
    lake.close()
assert scope["declaration"] == cfg["execution"]
assert scope_row["stop_json"] is None and tables["cpu_action_decisions"] == []
assert budget["remaining_wall_seconds"] is None and budget["reserved_wall_seconds"] == 2
assert budget["active_reservations"] == 0 and budget["free_slots"] == 1
prepared_raw = (effect_path.parent / "prepared.json").read_bytes()
committed = json.loads(effect_raw)
ref = {key: attempt[key] for key in ("task_id", "phase", "attempt_id", "generation")}
assert all(committed[key] == ref[key] for key in ref)
assert hashlib.sha256(prepared_raw).hexdigest() == terminal["effect_receipt_sha256"] == committed["prepared_sha256"]
assert committed["event"] == "effect_committed"
assert json.loads(tables["cpu_action_reservations"][0]["ref_json"]) == ref
report.update(budget_after=budget, effect_prepared_bytes=prepared_raw.decode(),
              guardian_seconds=20, guardian_is_fault_safety_not_research_deadline=True,
              stop_reason="--once dispatched and drained one action; not scientific convergence")
(ROOT / "cpu-canary-report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
print(json.dumps({"passed":True,"exit_code":code,"native_actions":1,"artifact_result":json.loads(payload),
                  "native_lease_status":terminal["runtime_lease"]["status"],"budget_state":"SETTLED",
                  "reserved_nanoseconds":permit["reserved_nanoseconds"],
                  "wall_seconds":finished-started,"report":str(ROOT/"cpu-canary-report.json")},sort_keys=True))
