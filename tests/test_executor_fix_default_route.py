"""Actual blocked READY must retain a configured implicit database route.

The injected boundary changes configuration and registers real SQLite history
after READY. This is a candidate admission-capture regression, not an old
release native-fixer test or an actual concurrent pathname race.
"""
import os
import signal
import sqlite3
import sys

import pytest

from orze.core.execution_attempts import create_attempt
from orze.engine import bounded_executor, failure
from orze.engine.termination_hold import TerminationUnconfirmed
from test_native_training_tree_completion import _alive, _wait_dead


def test_real_ready_cannot_erase_original_orze_dir_database(tmp_path, monkeypatch):
    if sys.platform != "linux" or not hasattr(os, "pidfd_open"):
        pytest.skip("requires Linux pidfds for exact own-process cleanup")
    folder = tmp_path / "results" / "idea-default-route"
    folder.mkdir(parents=True)
    (folder / "train_output.log").write_text("synthetic failure\n")
    marker = tmp_path / "executed"
    cli = tmp_path / "cpu-fixer"
    cli.write_text("#!" + sys.executable + "\nfrom pathlib import Path\n"
                   + "Path(" + repr(str(marker)) + ").write_text('CPU only')\n"
                   + "print('FIX_APPLIED')\n")
    cli.chmod(0o700)
    original = tmp_path / "authority-a"
    original.mkdir()
    db = original / "idea_lake.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE ideas (idea_id TEXT PRIMARY KEY)")
    cfg = {"_project_root": str(tmp_path), "_orze_dir": str(original),
           "max_fix_attempts": 1, "agent_tool_policy": {"enabled": True},
           "executor_fix": {"claude_bin": str(cli), "timeout": 5}}
    handles, owned, counts = [], [], {}
    real_prepare = bounded_executor.prepare_supervised

    def prepare(*args, **kwargs):
        child = real_prepare(*args, **kwargs)
        handles.append(child)
        for pid in (child.pid, child.supervisor_pid):
            owned.append((pid, os.pidfd_open(pid)))
        assert not marker.exists()
        with sqlite3.connect(db) as conn:
            conn.execute("BEGIN")
            create_attempt(conn, folder.name, "artifact_preflight", "new-native-history",
                           {"origin": "native_artifact_preflight"})
        cfg["_orze_dir"] = str(tmp_path / "authority-b")
        return child

    monkeypatch.setattr(bounded_executor, "prepare_supervised", prepare)
    try:
        with pytest.raises(TerminationUnconfirmed):
            failure._try_executor_fix(folder.name, "synthetic failure", folder.parent, cfg, counts)
        assert not marker.exists() and counts == {}
        assert list((folder.parent / "_fix_logs").glob("*.log")) == []
        assert len(handles) == 1
        assert all(not _alive(fd) for _, fd in owned)
        tree = handles[0].closure_receipt()
        assert tree["event"] == "TREE_CLOSED" and tree["wait_proof"] == "ECHILD_WALL"
        assert tree["stop_requested"] is True
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
