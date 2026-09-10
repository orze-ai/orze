"""Native shutdown mechanism with real launch, SQLite, receipts and stop gate."""
from dataclasses import replace
import json
from pathlib import Path
import sqlite3

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import lifecycle, process
from orze.idea_lake import IdeaLake
from test_native_training_caller_boundaries import case, _launch


def _files(folder):
    return {str(path.relative_to(folder)): path.read_bytes()
            for path in folder.rglob("*") if path.is_file()}


def _stop_boundary(c, monkeypatch, *, confirmed=True):
    stops = []

    def stop(child, *args, **kwargs):
        assert child is c.child
        stops.append(child)
        child.returncode = -15
        return confirmed

    monkeypatch.setattr(process, "_terminate_and_reap", stop)
    return stops


def _invoke(c, tracked, entry, *, lake=None):
    active = {0: tracked}
    if entry == "atexit":
        lifecycle.atexit_cleanup(active, {}, {}, c.results)
    else:
        lifecycle.graceful_shutdown(c.results, c.cfg, active, {}, {}, 1, {},
            c.lake if lake is None else lake, "test-only", "test-only",
            kill_all=True, managed=True)
    return active


def _state(c):
    conn = sqlite3.connect(c.lake.db_path)
    try:
        row = current_attempt(conn, c.idea, "training")
        state = conn.execute("SELECT current_state FROM idea_state WHERE idea_id=?", (c.idea,)).fetchone()[0]
        return row, state
    finally:
        conn.close()


@pytest.mark.parametrize("entry", ["graceful", "atexit"])
def test_confirmed_native_training_shutdown_closes_exact_attempt(case, monkeypatch, entry):
    c = case
    tracked = _launch(c)
    claim = (c.folder / "claim.json").read_bytes()
    stops = _stop_boundary(c, monkeypatch)
    try:
        assert _invoke(c, tracked, entry) == {}
        row, state = _state(c)
        assert stops == [c.child]
        assert row["state"] == "TERMINAL" and row["terminal"]["outcome"] == "interrupted"
        assert state == "FAILED"
        receipt = json.loads((c.folder / "_compute_receipts" / tracked.attempt_id / "terminal.json").read_text())
        assert receipt["phase"] == "training" and receipt["return_code"] == -15
        assert (c.folder / "_execution_effects" / tracked.attempt_id / "committed.json").exists()
        assert (c.folder / "claim.json").read_bytes() == claim
        assert not (c.folder / "metrics.json").exists()
        interruption = c.folder / "interruption.json"
        if entry == "atexit":
            assert not interruption.exists(), "no config cannot invent checkpoint resumability"
        else:
            assert json.loads(interruption.read_text())["resume_eligible"] is False
    finally:
        tracked.close_log()


class RejectCommit(sqlite3.Connection):
    def commit(self):
        state = self.execute("SELECT state FROM execution_attempts").fetchone()
        if state is not None and state[0] == "TERMINAL":
            raise sqlite3.OperationalError("synthetic shutdown commit failure")
        return super().commit()


@pytest.mark.parametrize("fault", ["stop_false", "sql_ignore", "commit_failure"])
def test_native_shutdown_failure_keeps_handle_and_durable_hold(case, monkeypatch, fault):
    c = case
    tracked = _launch(c)
    stops = _stop_boundary(c, monkeypatch, confirmed=fault != "stop_false")
    if fault == "sql_ignore":
        c.lake.conn.execute("CREATE TRIGGER ignore_shutdown BEFORE UPDATE ON idea_state "
            "WHEN NEW.current_state='FAILED' BEGIN SELECT RAISE(IGNORE); END")
        c.lake.conn.commit()
    elif fault == "commit_failure":
        c.lake.conn.close()
        c.lake.conn = sqlite3.connect(c.lake.db_path, factory=RejectCommit)
        c.lake.conn.row_factory = sqlite3.Row
    try:
        assert _invoke(c, tracked, "graceful").get(0) is tracked
        assert stops == [c.child]
        row, state = _state(c)
        assert row["state"] == "RUNNING" and state == "IN_PROGRESS"
        assert not (c.folder / "_execution_effects" / tracked.attempt_id / "committed.json").exists()
        if fault == "stop_false":
            assert tracked._termination_unconfirmed is True
            assert not (c.folder / "_compute_receipts" / tracked.attempt_id / "terminal.json").exists()
            assert list((c.folder / "_execution_stops").glob("*/requested.json"))
        else:
            assert (c.folder / "_attempt_effect.lock").is_dir()
            assert (c.folder / "_execution_effects" / tracked.attempt_id / "prepared.json").exists()
    finally:
        tracked.close_log()


@pytest.mark.parametrize("fault", ["missing_route_db", "wrong_scope", "wrong_ref"])
def test_unknown_shutdown_authority_does_not_stop_or_create_database(case, monkeypatch, fault):
    c = case
    tracked = _launch(c)
    stops = _stop_boundary(c, monkeypatch)
    missing = c.results.parent / "never-create.sqlite"
    peer = None
    if fault == "missing_route_db":
        declaration = c.folder / "_execution_catalog.json"
        payload = json.loads(declaration.read_text())
        payload["database"] = str(missing)
        declaration.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    elif fault == "wrong_scope":
        peer = IdeaLake(c.results.parent / "different-scope.sqlite")
    else:
        tracked.attempt_ref = replace(tracked.attempt_ref, task_id="idea-other-task")
    before = _files(c.folder)
    try:
        active = _invoke(c, tracked, "graceful" if peer is not None else "atexit", lake=peer)
        assert active.get(0) is tracked
        assert stops == [] and c.child.returncode is None
        assert _files(c.folder) == before
        assert _state(c)[0]["state"] == "RUNNING"
        assert not missing.exists()
    finally:
        tracked.close_log()
        if peer is not None:
            peer.close()


def test_native_training_detach_retains_persisted_ownership_without_claiming_resume(case, monkeypatch):
    c = case
    tracked = _launch(c)
    stops = _stop_boundary(c, monkeypatch)
    before = _files(c.folder)
    active = {0: tracked}
    lifecycle.graceful_shutdown(c.results, c.cfg, active, {}, {}, 1, {}, c.lake,
        "test-only", "test-only", kill_all=False, managed=True)
    assert active == {}, "detached handle must not be killed by subsequent atexit"
    assert stops == [] and c.child.returncode is None
    assert _files(c.folder) == before
    assert _state(c)[0]["state"] == "RUNNING"
    claim = json.loads((c.folder / "claim.json").read_text())
    assert claim["attempt_id"] == tracked.attempt_id and claim["trainer_pid"] == c.child.pid
    assert (c.folder / "_compute_receipts" / tracked.attempt_id / "start.json").exists()


def test_atexit_connection_close_failure_keeps_handle_without_escaping_cleanup(case, monkeypatch):
    c = case
    tracked = _launch(c)
    stops = _stop_boundary(c, monkeypatch)
    real_connect = sqlite3.connect
    closed = []

    class CloseFailure(sqlite3.Connection):
        def close(self):
            super().close()
            closed.append(True)
            raise OSError("synthetic shutdown connection close failure")

    def connect(database, *args, **kwargs):
        if str(database).endswith("?mode=rw"):
            kwargs["factory"] = CloseFailure
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)
    try:
        active = _invoke(c, tracked, "atexit")
        assert closed and stops == [c.child]
        assert active.get(0) is tracked
        # The commit itself was verified; preserve its truthful terminal and
        # do not pretend a connection-close error rolled SQLite back.
        assert _state(c)[0]["state"] == "TERMINAL"
    finally:
        tracked.close_log()
