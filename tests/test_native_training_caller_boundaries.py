"""Native training caller acceptance and explicitly captured D2 draft flaws."""
import copy
import json
import sqlite3
from types import SimpleNamespace

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import launcher
from orze.engine.scheduler import claim
from orze.engine.termination_hold import TerminationUnconfirmed
from orze.idea_lake import IdeaLake


class Child:
    def __init__(self, pid=741201, ret=None):
        self.pid = pid
        self.returncode = ret
        self.polls = 0

    def poll(self):
        self.polls += 1
        return self.returncode


class FaultConnection(sqlite3.Connection):
    fail_started_commit = False

    def commit(self):
        if self.fail_started_commit:
            state = self.execute("SELECT current_state FROM idea_state").fetchone()
            if state is not None and state[0] == "IN_PROGRESS":
                raise sqlite3.OperationalError("synthetic started commit failure")
        return super().commit()


@pytest.fixture
def case(tmp_path, monkeypatch):
    results = tmp_path / "results"
    lake = IdeaLake(tmp_path / "lake.db")
    idea = "idea-native-caller"
    lake.insert(idea, "Native caller", "seed: 13", "", status="queued")
    assert claim(idea, results, 0, lake=lake)
    folder = results / idea
    train, base, ideas = [tmp_path / name for name in ("train.py", "base.yaml", "ideas.md")]
    train.write_text("# Never executed\n")
    base.write_text("{}\n")
    ideas.write_text("")
    cfg = {"train_script": str(train), "base_config": str(base),
           "ideas_file": str(ideas), "python": "python3", "max_fix_attempts": 0,
           "sops": {"failure_feedback": False}, "stall_minutes": 0}
    child = Child()
    c = SimpleNamespace(lake=lake, results=results, folder=folder, idea=idea,
                        cfg=cfg, child=child, popen_calls=[], stops=[], before_popen=None)

    def popen(*args, **kwargs):
        c.popen_calls.append(True)
        if c.before_popen:
            c.before_popen()
        return child

    def reap(proc, idea_id, **kwargs):
        assert proc is child and idea_id == idea
        c.stops.append(True)
        child.returncode = -15
        return True

    monkeypatch.setattr(launcher, "_verify_gpu_free", lambda *a, **k: None)
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher, "capture_process_identity", lambda pid: {
        "pid": pid, "pgid": pid, "start_ticks": 12001})
    monkeypatch.setattr(launcher, "_terminate_and_reap", reap)
    monkeypatch.setattr(launcher, "notify", lambda *a, **k: None)
    try:
        yield c
    finally:
        lake.close()


def _launch(c):
    return launcher.launch(c.idea, 0, c.results, c.cfg, lake=c.lake)


def _terminals(c):
    return list(c.folder.glob("_compute_receipts/*/terminal.json"))


def test_real_launch_publishes_intent_to_peer_before_popen_and_binds_started(case):
    c = case
    seen = []

    def peer_read():
        peer = sqlite3.connect(c.lake.db_path)
        try:
            row = current_attempt(peer, c.idea, "training")
            assert row["state"] == "LAUNCHING"
            assert peer.execute("SELECT current_state FROM idea_state").fetchone()[0] == "CLAIMED"
            stages = peer.execute("SELECT current_state FROM idea_stage_state WHERE stage='training'").fetchall()
            assert all(row[0] in ("NOT_STARTED", "PENDING") for row in stages)
            assert not _terminals(c)
            seen.append(row["attempt_id"])
        finally:
            peer.close()

    c.before_popen = peer_read
    tp = _launch(c)
    assert seen == [tp.attempt_id] and c.popen_calls == [True]
    row = current_attempt(c.lake.conn, c.idea, "training")
    claim_value = json.loads((c.folder / "claim.json").read_text())
    assert row["state"] == "RUNNING" and row["binding"]["process_pid"] == c.child.pid
    assert claim_value["trainer_pid"] == c.child.pid
    assert claim_value["trainer_start_ticks"] == 12001
    assert c.lake.get_fsm_state(c.idea) == "IN_PROGRESS"
    tp.close_log()


def test_native_constructor_failure_closes_known_stopped_attempt_not_task(case, monkeypatch):
    c = case

    def fail_constructor(*args, **kwargs):
        raise ValueError("synthetic TrainingProcess construction failure")

    monkeypatch.setattr(launcher, "TrainingProcess", fail_constructor)
    with pytest.raises(ValueError, match="construction failure"):
        _launch(c)
    assert c.popen_calls == c.stops == [True]
    assert c.lake.get_fsm_state(c.idea) == "CLAIMED"
    assert current_attempt(c.lake.conn, c.idea, "training")["state"] == "TERMINAL"
    assert len(_terminals(c)) == 1
    assert not (c.folder / "metrics.json").exists()


@pytest.mark.parametrize("fault", ["sql_ignore", "commit_failure"])
def test_started_claim_publication_then_sql_failure_retains_hold(case, fault):
    c = case
    if fault == "sql_ignore":
        c.lake.conn.execute(
            "CREATE TRIGGER ignore_started BEFORE UPDATE ON idea_state "
            "WHEN NEW.current_state='IN_PROGRESS' BEGIN SELECT RAISE(IGNORE); END")
        c.lake.conn.commit()
    else:
        c.lake.conn.close()
        c.lake.conn = sqlite3.connect(c.lake.db_path, factory=FaultConnection)
        c.lake.conn.row_factory = sqlite3.Row
        c.lake.conn.fail_started_commit = True
    with pytest.raises(TerminationUnconfirmed):
        _launch(c)
    assert c.popen_calls == c.stops == [True]
    assert json.loads((c.folder / "claim.json").read_text())["trainer_pid"] == c.child.pid
    assert current_attempt(c.lake.conn, c.idea, "training")["state"] == "LAUNCHING"
    assert c.lake.get_fsm_state(c.idea) == "CLAIMED"
    assert (c.folder / "_attempt_effect.lock").is_dir()
    assert not _terminals(c)
    assert not (c.folder / "metrics.json").exists()


def _assert_monitor_does_not_publish(c, tp, lake):
    metrics = c.folder / "metrics.json"
    metrics.write_text('{"status":"IN_PROGRESS","step":7}')
    before = metrics.read_bytes()
    active, failures = {0: tp}, {}
    try:
        result = launcher.check_active(active, c.results, c.cfg, failures, lake=lake)
    except TerminationUnconfirmed:
        result = []
    assert result == []
    assert active.get(0) is tp
    assert failures == {}
    assert metrics.read_bytes() == before
    assert not _terminals(c)
    assert c.lake.get_fsm_state(c.idea) == "IN_PROGRESS"
    assert current_attempt(c.lake.conn, c.idea, "training")["state"] == "RUNNING"


def test_monitor_rejects_native_handle_with_different_process_identity(case):
    c = case
    tp = _launch(c)
    wrong = copy.copy(tp)
    wrong.process = Child(pid=c.child.pid + 1, ret=1)
    try:
        _assert_monitor_does_not_publish(c, wrong, c.lake)
        assert wrong.process.polls == 0
    finally:
        tp.close_log()


def test_monitor_omitted_lake_cannot_downgrade_native_handle_to_legacy_write(case):
    c = case
    tp = _launch(c)
    c.child.returncode = 1
    try:
        _assert_monitor_does_not_publish(c, tp, None)
    finally:
        tp.close_log()
