"""New fixer admission/output requirements; no provider, model or GPU work.

Output and invalid-budget cases use an explicit CompletedProcess boundary.
READY cases use the real Linux supervisor and a tiny local CPU executable.
These are new requirements, not additional historical red regressions.
"""
import json
import os
import signal
import sqlite3
import subprocess
import sys

import pytest

from orze.core.execution_attempts import create_attempt
from orze.engine import bounded_executor, failure
from orze.engine.termination_hold import TerminationUnconfirmed
from test_executor_fix_caller_authority import repair_case, _repair
from test_native_training_tree_completion import _alive, _wait_dead


@pytest.mark.parametrize("budget", [True, False, 0, -1, float("inf"), float("nan"), "3", None])
def test_wrapper_rejects_invalid_raw_timeout_before_runner(repair_case, budget):
    c = repair_case
    c.cfg["executor_fix"] = {"timeout": budget}
    assert _repair(c) is False
    assert c.calls == [] and c.fixes == {}


@pytest.mark.parametrize(("output", "complete", "stopped", "accepted"), [
    ("FIX_APPLIED\n", False, False, False),
    ("FIX_APPLIED\n", True, True, False),
    ("UNFIXABLE: no verified change\n" + "x" * 6000 + "\nFIX_APPLIED\n", True, False, False),
    ("FIX_APPLIED\n" + "diagnostic\n" * 600, True, False, True),
    ("FIX_APPLIED\n", True, False, True),
])
def test_entire_complete_unstopped_response_qualifies_marker(
        repair_case, monkeypatch, output, complete, stopped, accepted):
    c = repair_case

    def result(cmd, **kwargs):
        c.calls.append("executor")
        value = subprocess.CompletedProcess(cmd, 0, stdout=output, stderr="")
        value.output_complete, value.stopped = complete, stopped
        return value

    monkeypatch.setattr(failure, "_run_bounded_executor", result)
    assert _repair(c) is accepted
    assert c.calls == ["executor"] and c.fixes == {c.idea: 1}
    assert len((c.results / "_fix_logs" / f"{c.idea}_attempt1.log").read_bytes()) < 6000


@pytest.mark.parametrize("shortcut", ["disabled", "schema_invalid", "queue_revalidation"])
def test_native_scope_gate_precedes_ordinary_false_and_failure_write(
        repair_case, shortcut):
    c = repair_case
    c.cfg["max_fix_attempts"] = 0
    db = c.results.parent / "registered.db"
    with sqlite3.connect(db) as conn:
        conn.execute("BEGIN")
        create_attempt(conn, c.idea, "pre_script", "imported-history",
                       {"origin": "legacy_import"})
    c.cfg["idea_lake_db"] = str(db)
    error = ("error: unrecognized arguments: --wrong" if shortcut == "schema_invalid"
             else "queue_revalidation_rejected" if shortcut == "queue_revalidation"
             else "synthetic failure")
    before = db.read_bytes()
    with pytest.raises(TerminationUnconfirmed):
        failure._try_executor_fix(c.idea, error, c.results, c.cfg, c.fixes,
                                  exit_code=2 if shortcut == "schema_invalid" else 7)
    assert c.calls == [] and c.fixes == {}
    assert db.read_bytes() == before
    assert not (c.results / "_fix_logs").exists()


@pytest.mark.parametrize("change", ["original_database", "catalog", "project_root", "max_fix", "unchanged"])
def test_real_ready_rechecks_legacy_admission_before_go(tmp_path, monkeypatch, change):
    if sys.platform != "linux" or not hasattr(os, "pidfd_open"):
        pytest.skip("requires Linux pidfds for exact own-process cleanup")
    results = tmp_path / "results"
    folder = results / "idea-ready-fix"
    folder.mkdir(parents=True)
    (folder / "train_output.log").write_text("synthetic failure\n")
    marker = tmp_path / "executed"
    cli = tmp_path / "cpu-fixer"
    cli.write_text("#!" + sys.executable + "\nfrom pathlib import Path\n"
                   + "Path(" + repr(str(marker)) + ").write_text('CPU only')\n"
                   + "print('FIX_APPLIED')\n")
    cli.chmod(0o700)
    db = tmp_path / "original.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE ideas (idea_id TEXT PRIMARY KEY)")
    cfg = {"_project_root": str(tmp_path), "max_fix_attempts": 1,
           "idea_lake_db": str(db), "agent_tool_policy": {"enabled": True},
           "executor_fix": {"claude_bin": str(cli), "timeout": 5}}
    counts, handles, owned = {}, [], []
    real_prepare = bounded_executor.prepare_supervised

    def prepare(*args, **kwargs):
        child = real_prepare(*args, **kwargs)
        handles.append(child)
        for pid in (child.pid, child.supervisor_pid):
            owned.append((pid, os.pidfd_open(pid)))
        assert not marker.exists()
        if change == "original_database":
            with sqlite3.connect(db) as conn:
                conn.execute("BEGIN")
                create_attempt(conn, folder.name, "artifact_preflight", "new-history",
                               {"origin": "legacy_import"})
            cfg.pop("idea_lake_db")
        elif change == "catalog":
            (folder / "_execution_catalog.json").write_text(json.dumps({
                "schema": 1, "task_id": folder.name, "database": str(db)},
                sort_keys=True, separators=(",", ":")) + "\n")
        elif change == "project_root":
            replacement = tmp_path / "replacement"
            replacement.mkdir()
            cfg["_project_root"] = str(replacement)
        elif change == "max_fix":
            cfg["max_fix_attempts"] = 0
        return child

    monkeypatch.setattr(bounded_executor, "prepare_supervised", prepare)
    try:
        if change == "unchanged":
            assert failure._try_executor_fix(folder.name, "synthetic failure",
                                              results, cfg, counts) is True
            assert marker.read_text() == "CPU only"
            assert counts == {folder.name: 1}
        else:
            with pytest.raises(TerminationUnconfirmed):
                failure._try_executor_fix(folder.name, "synthetic failure",
                                           results, cfg, counts)
            assert not marker.exists() and counts == {}
            assert list((results / "_fix_logs").glob("*.log")) == []
        assert len(handles) == 1
        assert all(not _alive(fd) for _, fd in owned)
        tree = handles[0].closure_receipt()
        assert tree["event"] == "TREE_CLOSED" and tree["wait_proof"] == "ECHILD_WALL"
        assert tree["stop_requested"] is (change != "unchanged")
    finally:
        for _, fd in owned:
            if _alive(fd):
                signal.pidfd_send_signal(fd, signal.SIGKILL)
            _wait_dead(fd)
        for pid, fd in owned:
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
            os.close(fd)
        for child in handles:
            child._close_descriptors()
