"""Caller HOLD propagation and new legacy-only repair admission requirements.

No provider, GPU, or worker is executed. The public fixer/monitor/dispatch
callers are real; only OS/repair/relaunch boundaries are controlled. New native
admission constraints are not historical missing-API regressions.
"""
import json
import sqlite3
import subprocess
import time
from types import SimpleNamespace

import pytest

from orze.core.execution_attempts import create_attempt
from orze.engine import failure, launcher, phases
from orze.engine.process import TrainingProcess
from orze.engine.termination_hold import TerminationUnconfirmed
from test_training_launch_termination_handoff import scenario


@pytest.fixture
def repair_case(tmp_path, monkeypatch):
    results = tmp_path / "results"
    folder = results / "idea-fix-authority"
    folder.mkdir(parents=True)
    (folder / "train_output.log").write_text("synthetic failure\n")
    cfg = {"_project_root": str(tmp_path), "max_fix_attempts": 1,
           "agent_tool_policy": {"enabled": True}, "sops": {"failure_feedback": False}}
    calls = []

    def run(cmd, **kwargs):
        calls.append("executor")
        return subprocess.CompletedProcess(cmd, 0, stdout="FIX_APPLIED\n", stderr="")

    monkeypatch.setattr(failure, "_run_bounded_executor", run)
    return SimpleNamespace(results=results, folder=folder, idea=folder.name,
                           cfg=cfg, calls=calls, fixes={})


def _repair(c):
    return failure._try_executor_fix(c.idea, "synthetic failure", c.results,
                                     c.cfg, c.fixes, exit_code=7)


def test_wrapper_does_not_turn_unknown_tree_into_false_or_count(repair_case, monkeypatch):
    c = repair_case

    def unknown(*args, **kwargs):
        c.calls.append("executor")
        raise TerminationUnconfirmed("synthetic_executor_tree_unknown")

    monkeypatch.setattr(failure, "_run_bounded_executor", unknown)
    with pytest.raises(TerminationUnconfirmed):
        _repair(c)
    assert c.calls == ["executor"]
    assert c.fixes == {}
    assert list((c.results / "_fix_logs").glob("*.log")) == []


@pytest.mark.parametrize("branch", ["timeout", "stall", "zombie", "fatal", "metrics_failed", "exit"])
def test_six_legacy_relaunch_callers_propagate_hold(repair_case, monkeypatch, branch):
    c = repair_case
    events = []
    proc = SimpleNamespace(pid=871234, returncode=None)
    proc.poll = lambda: proc.returncode
    proc.wait = lambda timeout=None: proc.returncode
    proc.returncode = 0 if branch == "metrics_failed" else 7 if branch == "exit" else None
    tp = TrainingProcess(c.idea, 0, proc, time.time() - 180,
                         c.folder / "train_output.log", timeout=1 if branch == "timeout" else 3600,
                         attempt_id="legacy-authority-attempt")
    if branch == "metrics_failed":
        (c.folder / "metrics.json").write_text('{"status":"FAILED","error":"synthetic failure"}\n')
    active, failures = {0: tp}, {}
    monkeypatch.setattr(launcher, "notify", lambda *a, **k: None)
    monkeypatch.setattr("orze.engine.health._adaptive_stall_minutes", lambda *a: 1)
    monkeypatch.setattr("orze.engine.health.check_stalled", lambda *a: branch == "stall")
    monkeypatch.setattr("orze.engine.health.detect_fatal_in_log", lambda *a: "fatal" if branch == "fatal" else None)
    monkeypatch.setattr(launcher, "_detect_zombie", lambda *a: branch == "zombie")
    monkeypatch.setattr(launcher, "_watchdog_check", lambda *a: False)

    def stop(*args):
        events.append("original_stop")
        proc.returncode = -15
        return True

    def reset(*args, **kwargs):
        events.append("reset_before_relaunch")

    def relaunch(*args, **kwargs):
        events.append("relaunch_hold")
        raise TerminationUnconfirmed("synthetic_new_launch_unknown")

    monkeypatch.setattr(launcher, "_terminate_training", stop)
    monkeypatch.setattr(failure, "_reset_idea_for_retry", reset)
    monkeypatch.setattr(launcher, "launch", relaunch)
    monkeypatch.setattr(launcher, "_write_failure", lambda *a, **k: events.append("failure_write"))
    monkeypatch.setattr("orze.engine.failure_analysis.write_failure_analysis", lambda *a, **k: events.append("failure_analysis"))
    monkeypatch.setattr("orze.engine.accounting.record_compute_terminal", lambda *a, **k: events.append("original_terminal"))
    with pytest.raises(TerminationUnconfirmed):
        launcher.check_active(active, c.results, c.cfg, failures, c.fixes)
    assert active == {0: tp}
    assert failures == {}
    assert c.calls == ["executor"]
    assert c.fixes == {c.idea: 1}
    assert events.count("reset_before_relaunch") == 1
    assert events[-1] == "relaunch_hold"
    assert "failure_write" not in events and "failure_analysis" not in events


def test_phase_legacy_pre_failure_hold_does_not_reset_or_publish(scenario, monkeypatch):
    runner, child, popen, old_fixer = scenario
    events = []
    monkeypatch.setattr(phases, "run_pre_script", lambda *a, **k: False)

    def unknown(*args, **kwargs):
        events.append("fixer_hold")
        raise TerminationUnconfirmed("synthetic_executor_tree_unknown")

    monkeypatch.setattr(phases, "_try_executor_fix", unknown)
    monkeypatch.setattr(phases, "_reset_idea_for_retry", lambda *a, **k: events.append("reset"))
    monkeypatch.setattr(phases, "_write_failure", lambda *a, **k: events.append("failure"))
    with pytest.raises(TerminationUnconfirmed):
        phases.OrzePhaseMixin._launch_training(runner, ["idea-handoff"], True,
            {"idea-handoff": {"title": "Fixture", "config": {"seed": 1}}})
    folder = runner.results_dir / "idea-handoff"
    assert events == ["fixer_hold"]
    assert runner.failure_counts == {} and runner.fix_counts == {}
    assert runner.lake.get_fsm_state("idea-handoff") == "CLAIMED"
    assert (folder / "claim.json").exists()
    assert not (folder / "metrics.json").exists()
    assert list(folder.glob("_compute_receipts/*/terminal.json")) == []
    popen.assert_not_called()


@pytest.mark.parametrize(("route", "phase"), [
    ("declared", "training"), ("claim", "pre_script"),
    ("configured_db_only", "artifact_preflight"), ("default_db_only", "pre_script"),
])
def test_native_ownership_cannot_directly_authorize_legacy_repair(repair_case, route, phase):
    c = repair_case
    db = c.results.parent / (".orze/idea_lake.db" if route == "default_db_only" else "native.db")
    db.parent.mkdir(exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.execute("BEGIN")
        create_attempt(conn, c.idea, phase, "native-fix-source", {"origin": "native_" + phase})
    if route == "declared":
        (c.folder / "_execution_catalog.json").write_text(json.dumps({"schema": 1,
            "task_id": c.idea, "database": str(db)}, sort_keys=True, separators=(",", ":")) + "\n")
    elif route == "claim":
        (c.folder / "claim.json").write_text(json.dumps({"attempt_id": "claim-fix-source", "lifecycle_db": str(db)}))
    elif route == "configured_db_only":
        c.cfg["idea_lake_db"] = str(db)
    before = db.read_bytes()
    with pytest.raises(TerminationUnconfirmed):
        _repair(c)
    assert c.calls == [] and c.fixes == {}
    assert db.read_bytes() == before
    assert not (c.results / "_fix_logs").exists()


@pytest.mark.parametrize("database", ["pre_native", "other_task"])
def test_genuinely_legacy_database_or_other_task_does_not_block(repair_case, database):
    c = repair_case
    db = c.results.parent / "legacy.db"
    with sqlite3.connect(db) as conn:
        if database == "pre_native":
            conn.execute("CREATE TABLE ideas (idea_id TEXT PRIMARY KEY)")
        else:
            conn.execute("BEGIN")
            create_attempt(conn, "unrelated-task", "training", "unrelated-attempt", {"origin": "native_training"})
    c.cfg["idea_lake_db"] = str(db)
    before = db.read_bytes()
    assert _repair(c) is True
    assert c.calls == ["executor"] and c.fixes == {c.idea: 1}
    assert db.read_bytes() == before


@pytest.mark.parametrize("database", ["missing", "corrupt"])
def test_explicit_unavailable_database_never_downgrades(repair_case, database):
    c = repair_case
    db = c.results.parent / "unavailable.db"
    if database == "corrupt":
        db.write_bytes(b"not a SQLite catalog")
    c.cfg["idea_lake_db"] = str(db)
    before = db.read_bytes() if db.exists() else None
    with pytest.raises(TerminationUnconfirmed):
        _repair(c)
    assert c.calls == [] and c.fixes == {}
    assert (db.read_bytes() if db.exists() else None) == before
