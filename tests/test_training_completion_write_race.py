"""D2 public completion races, frozen against D1 250ad8a.

All attempts, claims, resets, state transitions and compute receipts are real.
The second connection rotates A to B at existing observer/write boundaries.
A future bounded write guard may reject that rotation; the fixture then proves
the producer can retry after the outer operation releases its guard. Every
injected seam must execute, so moving production IO requires an explicit test
migration instead of silently turning these regressions green.

D2 migration: only the FSM injection target moves from the public transaction
wrapper to the actual caller-owned SQL writer. All business assertions and the
other three real IO seams are unchanged; the original fixture is archived.
"""
import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from orze.engine import launcher
from orze.engine.accounting import record_compute_start
from orze.engine.failure import _reset_idea_for_retry
from orze.engine.process import TrainingProcess
from orze.engine.scheduler import claim
from orze.idea_lake import IdeaLake


class Child:
    def __init__(self, pid, returncode):
        self.pid = pid
        self.returncode = returncode
        self.on_poll = None

    def poll(self):
        callback, self.on_poll = self.on_poll, None
        if callback:
            callback()
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture
def case(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    db = tmp_path / "ideas.db"
    lake = IdeaLake(str(db))
    peer = IdeaLake(str(db))
    peer.conn.execute("PRAGMA busy_timeout=1")
    idea_id = "idea-write-race"
    lake.insert(idea_id, "Synthetic current attempt", "seed: 13", "", status="queued")
    assert claim(idea_id, results, 0, lake=lake)
    assert lake.record_state_transition(idea_id, "CLAIMED", "IN_PROGRESS")
    folder = results / idea_id
    log = folder / "train_output.log"
    log.write_text("")
    attempt_id = json.loads((folder / "claim.json").read_text())["attempt_id"]
    tp = TrainingProcess(idea_id, 0, Child(881101, 0), time.time() - 1,
                         log, 3600, attempt_id=attempt_id)
    record_compute_start(tp, folder)
    metrics = folder / "metrics.json"
    metrics.write_text('{"status":"IN_PROGRESS","owner":"A","step":3}\n')
    monkeypatch.setattr(launcher, "notify", lambda *a, **k: None)
    c = SimpleNamespace(
        results=results, folder=folder, metrics=metrics, lake=lake, peer=peer,
        tp=tp, active={0: tp}, failures={}, fixes={}, current=None,
        cfg={"max_fix_attempts": 0, "sops": {"failure_feedback": False}},
        seam_calls=0, rotation_accepted=None,
    )
    try:
        yield c
    finally:
        peer.close()
        lake.close()


def _poll(c, *, active=None, lake=None):
    return launcher.check_active(
        c.active if active is None else active, c.results, c.cfg,
        c.failures, c.fixes, lake=c.lake if lake is None else lake,
    )


def _files(c):
    return {name: ((c.folder / name).read_bytes()
                   if (c.folder / name).exists() else None)
            for name in ("claim.json", "metrics.json", "train_output.log")}


def _rotate_to_b(c):
    """Real second-connection lifecycle, with an explicit bounded busy probe."""
    before = _files(c)
    before_history = c.peer.get_fsm_history(c.tp.idea_id)
    try:
        # Do not let the legacy busy-retry loop block pytest for minutes if
        # production legitimately owns its write transaction at this seam.
        c.peer.conn.execute("BEGIN IMMEDIATE")
        c.peer.conn.rollback()
        _reset_idea_for_retry(c.folder, release_claim=True)
    except (RuntimeError, sqlite3.OperationalError):
        assert _files(c) == before, "a rejected rotation must not partly clear A"
        assert c.peer.get_fsm_history(c.tp.idea_id) == before_history
        return False

    state = c.peer.get_fsm_state(c.tp.idea_id)
    if state != "FAILED":
        assert c.peer.record_state_transition(
            c.tp.idea_id, state, "FAILED", "fixture A finished")
    assert c.peer.record_state_transition(
        c.tp.idea_id, "FAILED", "QUEUED", "fixture next attempt")
    assert claim(c.tp.idea_id, c.results, 0, lake=c.peer)
    assert c.peer.record_state_transition(
        c.tp.idea_id, "CLAIMED", "IN_PROGRESS", "fixture B started")
    attempt_id = json.loads((c.folder / "claim.json").read_text())["attempt_id"]
    assert attempt_id != c.tp.attempt_id
    current = TrainingProcess(
        c.tp.idea_id, 0, Child(881102, None), time.time(), c.tp.log_path,
        3600, attempt_id=attempt_id,
    )
    record_compute_start(current, c.folder)
    c.metrics.write_text(json.dumps({
        "status": "IN_PROGRESS", "owner": "B", "step": 99,
        "attempt_id": attempt_id,
    }))
    c.current = current
    c.b_files = _files(c)
    c.b_history = c.peer.get_fsm_history(c.tp.idea_id)
    c.b_stages = c.peer.get_stage_history(c.tp.idea_id)
    return True


def _at_seam(c):
    if c.seam_calls:
        return
    c.seam_calls += 1
    c.rotation_accepted = _rotate_to_b(c)


@pytest.mark.parametrize("seam", ["metrics_read", "invalid_rename", "failure_write", "fsm_write"])
def test_rollover_at_actual_read_or_write_boundary_cannot_publish_into_b(
        case, monkeypatch, seam):
    c = case
    if seam in {"failure_write", "fsm_write"}:
        c.tp.process.returncode = 1
    if seam == "failure_write":
        c.metrics.unlink()

    if seam == "metrics_read":
        original = Path.read_text

        def read_text(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path == c.metrics:
                _at_seam(c)
            return result

        monkeypatch.setattr(Path, "read_text", read_text)
    elif seam == "invalid_rename":
        original = launcher.os.replace

        def replace(source, destination, *args, **kwargs):
            if (Path(source) == c.metrics
                    and Path(destination).name.startswith("metrics.invalid.")):
                _at_seam(c)
            return original(source, destination, *args, **kwargs)

        monkeypatch.setattr(launcher.os, "replace", replace)
    elif seam == "failure_write":
        original = launcher.atomic_write

        def write(path, *args, **kwargs):
            if Path(path) == c.metrics:
                _at_seam(c)
            return original(path, *args, **kwargs)

        monkeypatch.setattr(launcher, "atomic_write", write)
    else:
        original = c.lake._record_state_transition_in_tx

        def transition(idea_id, from_state, to_state, *args, **kwargs):
            if idea_id == c.tp.idea_id and to_state == "FAILED":
                _at_seam(c)
            return original(idea_id, from_state, to_state, *args, **kwargs)

        monkeypatch.setattr(c.lake, "_record_state_transition_in_tx", transition)

    _poll(c)
    assert c.seam_calls == 1, "the captured read/write race must actually execute"
    if c.rotation_accepted:
        assert c.failures == {}, "a stale completion must not spend B's failure allowance"
    else:
        # Correct serialization may finish A first. B must remain retryable
        # afterward instead of being lost or stranded by the rejected race.
        assert _rotate_to_b(c)
    assert _files(c) == c.b_files, "A must not rename or overwrite B artifacts"
    assert c.peer.get_fsm_state(c.tp.idea_id) == "IN_PROGRESS"
    assert c.peer.get_fsm_history(c.tp.idea_id) == c.b_history
    assert c.peer.get_stage_history(c.tp.idea_id) == c.b_stages


def test_duplicate_terminal_callback_from_second_connection_counts_once(case):
    c = case
    c.tp.process.returncode = 1
    first = _poll(c)
    assert first == [(c.tp.idea_id, 0)]
    assert c.failures == {c.tp.idea_id: 1}
    history = c.peer.get_fsm_history(c.tp.idea_id)
    replay = TrainingProcess(
        c.tp.idea_id, 0, Child(c.tp.process.pid, 1), c.tp.start_time,
        c.tp.log_path, c.tp.timeout, attempt_id=c.tp.attempt_id,
    )
    second = _poll(c, active={0: replay}, lake=c.peer)
    assert second == [], "the same attempt's terminal is not a second finished event"
    assert c.failures == {c.tp.idea_id: 1}
    assert c.peer.get_fsm_history(c.tp.idea_id) == history


def test_late_handler_does_not_delete_a_replaced_active_slot(case):
    c = case
    c.tp.process.returncode = 1
    other_id = "idea-slot-successor"
    c.peer.insert(other_id, "Other running task", "seed: 29", "", status="queued")
    assert claim(other_id, c.results, 0, lake=c.peer)
    assert c.peer.record_state_transition(other_id, "CLAIMED", "IN_PROGRESS")
    other_dir = c.results / other_id
    other_attempt = json.loads((other_dir / "claim.json").read_text())["attempt_id"]
    successor = TrainingProcess(
        other_id, 0, Child(881103, None), time.time(),
        other_dir / "train_output.log", 3600, attempt_id=other_attempt,
    )
    record_compute_start(successor, other_dir)
    c.tp.process.on_poll = lambda: c.active.__setitem__(0, successor)
    _poll(c)
    assert c.active.get(0) is successor, "retire A by identity, not by a reused slot key"
    assert c.peer.get_fsm_state(other_id) == "IN_PROGRESS"
    assert other_id not in c.failures


@pytest.mark.parametrize("status,expected", [("IN_PROGRESS", "FAILED"), ("COMPLETED", "COMPLETE")])
def test_uncontended_current_attempt_retains_existing_terminal_semantics(case, status, expected):
    c = case
    c.metrics.write_text(json.dumps({"status": status, "score": 0}))
    assert _poll(c) == [(c.tp.idea_id, 0)]
    assert c.lake.get_fsm_state(c.tp.idea_id) == expected
    assert c.active == {}
    assert c.failures == ({c.tp.idea_id: 1} if expected == "FAILED" else {})
