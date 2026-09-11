"""Actual foreground CLI, Orze loop, SQLite and private supervised CPU jobs."""
import json
from pathlib import Path
import signal
import sqlite3
import sys
import threading
import time

import pytest
import yaml


@pytest.fixture
def project(tmp_path, monkeypatch):
    import orze.cli as cli
    import orze.engine.orchestrator as engine
    import orze.engine.gpu_slots as slots
    monkeypatch.chdir(tmp_path)
    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}

    def forbidden(*args, **kwargs):
        pytest.fail("CPU product path invoked GPU/legacy side effects")

    monkeypatch.setattr(cli, "detect_all_gpus", forbidden)
    monkeypatch.setattr(slots, "GpuSlotManager", forbidden)
    monkeypatch.setattr(engine, "acquire_gpu_leases", forbidden)
    monkeypatch.setattr(engine, "startup_canary", forbidden)
    monkeypatch.setattr(engine.Orze, "_kill_orphans", forbidden)
    monkeypatch.setattr("orze.extensions.has_pro", forbidden)
    monkeypatch.setattr("orze.extensions._find_pro_key", forbidden)
    cfg = {"execution": {"version": 1, "resource": "cpu", "slots": 1,
                        "wall_budget_seconds": 5},
           "results_dir": str(tmp_path / "results"),
           "ideas_file": str(tmp_path / "ideas.md"),
           "idea_lake_db": str(tmp_path / "lake.db"), "min_disk_gb": 0,
           "action_policy": {"version": 1, "kind": "queue", "idle": "wait", "wait_seconds": 0.05}}

    def run(*, once=True):
        path = tmp_path / "orze.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        monkeypatch.setattr("sys.argv", ["orze", "-c", str(path)] + (["--once"] if once else []))
        return cli.main()

    try:
        yield tmp_path, cfg, run
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


def task(root, *, outputs, program):
    action = {"version": 1, "adapter": "command", "purpose": "verify generic CPU data flow",
              "inputs": {"values": [3, 1, 2]}, "command": [sys.executable, "-c", program],
              "timeout_seconds": 2, "outputs": outputs}
    text = "## idea-0001: CPU contract probe\n\n```yaml\n" + yaml.safe_dump(
        {"kind": "native_cpu_action", "action": action}) + "```\n"
    (root / "ideas.md").write_text(text, encoding="utf-8")


@pytest.mark.parametrize("with_artifact", [False, True])
def test_cli_runs_real_cpu_action_without_training_or_gpu(project, with_artifact):
    root, cfg, run = project
    program = """import json, os
from pathlib import Path
fd = int(os.environ['ORZE_ACTION_INPUT_FD'])
value = json.loads(os.read(fd, 65536))
assert value == {'values': [3, 1, 2]}
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
try:
    os.write(fd, b'changed')
except OSError:
    pass
else:
    raise AssertionError('input descriptor was writable')
Path('answer.json').write_text(json.dumps(sorted(value['values'])))
"""
    outputs = {"answer": {"path": "answer.json", "max_bytes": 128}} if with_artifact else {}
    task(root, outputs=outputs, program=program)
    assert run() in (None, 0)
    with sqlite3.connect(root / "lake.db") as conn:
        row = conn.execute("SELECT phase,state,binding_json,terminal_json FROM execution_attempts").fetchone()
        stages = conn.execute("SELECT stage,current_state FROM idea_stage_state").fetchall()
        slots = conn.execute("SELECT state FROM cpu_action_reservations").fetchall()
        state = conn.execute("SELECT kind,status FROM ideas").fetchone()
    assert row[:2] == ("action", "TERMINAL")
    binding, terminal = json.loads(row[2]), json.loads(row[3])
    assert binding["resource"] == "cpu"
    assert "gpu" not in binding
    assert terminal["outcome"] == "completed"
    assert terminal["process_tree"]["wait_proof"] == "ECHILD_WALL"
    assert terminal["observation_ids"] == []
    assert len(terminal["artifact_ids"]) == int(with_artifact)
    assert stages == [("action", "COMPLETE")]
    assert state == ("native_cpu_action", "completed")
    assert slots == [("SETTLED",)]
    assert not (root / "results" / "idea-0001" / "idea_config.yaml").exists()
    claim = json.loads((root / "results" / "idea-0001" / "claim.json").read_text())
    assert claim["resource"] == "cpu" and claim["gpu"] is None
    assert (root / "ideas.md").read_text().strip() == ""
    if with_artifact:
        content = root / ".orze" / "artifacts" / terminal["artifact_ids"][0] / "content"
        assert json.loads(content.read_text()) == [1, 2, 3]


@pytest.mark.parametrize("idle", ["wait", "stop"])
def test_empty_cpu_policy_is_durable_without_claim_attempt_or_charge(project, idle):
    root, cfg, run = project
    cfg["action_policy"]["idle"] = idle
    assert run() in (None, 0)
    with sqlite3.connect(root / "lake.db") as conn:
        decisions = conn.execute("SELECT record_json FROM cpu_action_decisions").fetchall()
        assert conn.execute("SELECT COUNT(*) FROM cpu_action_reservations").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM ideas").fetchone()[0] == 0
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='execution_attempts'").fetchall() == []
    assert len(decisions) == 1
    decision = json.loads(decisions[0][0])
    assert decision["kind"] == idle.title()
    assert bool(decision["reason"])
    assert (decision["wakeup"] is not None) == (idle == "wait")
    assert list((root / "results").glob("idea-*/claim.json")) == []


def test_cpu_budget_stop_does_not_claim_or_start_unaffordable_task(project):
    root, cfg, run = project
    cfg["execution"]["wall_budget_seconds"] = 1
    task(root, outputs={}, program="raise AssertionError('must not execute')")
    assert run() in (None, 0)
    with sqlite3.connect(root / "lake.db") as conn:
        record = json.loads(conn.execute("SELECT record_json FROM cpu_action_decisions").fetchone()[0])
        assert conn.execute("SELECT COUNT(*) FROM cpu_action_reservations").fetchone()[0] == 0
        assert conn.execute("SELECT status FROM ideas").fetchone()[0] == "queued"
    assert record["kind"] == "Stop" and record["reason"] == "wall_envelope_exhausted"
    assert not (root / "results" / "idea-0001" / "claim.json").exists()
    # Sticky Stop is not silently reset by a fresh actual CLI invocation.
    assert run() == 75


def test_cpu_daemon_drains_actual_action_then_stops_by_policy(project):
    root, cfg, run = project
    cfg["action_policy"]["idle"] = "stop"
    task(root, outputs={}, program="pass")
    assert run(once=False) in (None, 0)
    with sqlite3.connect(root / "lake.db") as conn:
        assert conn.execute("SELECT status FROM ideas").fetchone()[0] == "completed"
        stop = json.loads(conn.execute("SELECT stop_json FROM cpu_action_scopes").fetchone()[0])
        assert conn.execute("SELECT COUNT(*) FROM execution_attempts").fetchone()[0] == 1
        assert conn.execute("SELECT state FROM cpu_action_reservations").fetchone()[0] == "SETTLED"
    assert stop == {"kind": "Stop", "reason": "queue_drained", "wakeup": None}


def test_cpu_wait_wakes_for_new_proposal_without_preclaiming(project, monkeypatch):
    from orze.core import cpu_action_budget as budget
    root, cfg, run = project
    ready = threading.Event()
    errors = []
    real_record = budget.record_decision

    def record(*args):
        result = real_record(*args)
        if args[-1]["kind"] == "Wait" and args[-1]["reason"] == "queue_empty":
            ready.set()
        return result

    monkeypatch.setattr(budget, "record_decision", record)

    def producer():
        try:
            assert ready.wait(5), "actual durable Wait was not reached"
            with sqlite3.connect(root / "lake.db") as conn:
                assert conn.execute("SELECT COUNT(*) FROM cpu_action_reservations").fetchone()[0] == 0
                assert conn.execute("SELECT name FROM sqlite_master WHERE name='execution_attempts'").fetchall() == []
            task(root, outputs={}, program="pass")
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with sqlite3.connect(root / "lake.db") as conn:
                    row = conn.execute("SELECT status FROM ideas WHERE idea_id='idea-0001'").fetchone()
                if row == ("completed",):
                    break
                threading.Event().wait(0.01)
            else:
                raise AssertionError("proposal did not complete after Wait")
        except BaseException as exc:
            errors.append(exc)
        finally:
            (root / "results" / ".orze_stop_all").write_text("fixture-owned stop\n")

    worker = threading.Thread(target=producer)
    worker.start()
    try:
        assert run(once=False) in (None, 0)
    finally:
        worker.join(timeout=7)
    assert not worker.is_alive()
    assert errors == []
    with sqlite3.connect(root / "lake.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM execution_attempts").fetchone()[0] == 1
        assert conn.execute("SELECT state FROM cpu_action_reservations").fetchone()[0] == "SETTLED"
        stop = json.loads(conn.execute("SELECT stop_json FROM cpu_action_scopes").fetchone()[0])
    assert stop["reason"] == "operator_stop"
