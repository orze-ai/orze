"""Draft nested claim/reset boundaries must acknowledge effects before return."""
import errno
import os
from pathlib import Path

import pytest

from orze.core.execution_attempts import create_attempt, current_attempt, finish_attempt, mark_running
from orze.engine import failure, scheduler
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt, attempt_effect_lock
from orze.engine.execution_authority import execution_transaction, lifecycle_fence
from orze.idea_lake import IdeaLake


@pytest.fixture
def case(tmp_path):
    lake = IdeaLake(tmp_path / "lake.db")
    idea = "idea-reset-durable"
    lake.insert(idea, "Reset durability", "{}", "", status="queued")
    assert scheduler.claim(idea, tmp_path, 0, lake=lake)
    assert lake.record_state_transition(idea, "CLAIMED", "IN_PROGRESS")
    folder = tmp_path / idea
    (folder / "metrics.json").write_text('{"status":"FAILED"}')
    (folder / "train_output.log").write_text("preserved training log")
    lake.conn.execute("BEGIN IMMEDIATE")
    ref = create_attempt(lake.conn, idea, "training", "attempt-A", {})
    mark_running(lake.conn, ref)
    lake.conn.commit()
    try:
        yield lake, folder, ref
    finally:
        lake.close()


def _close_for_reset(tx, lake, ref):
    digest = tx.prepare(ref, {"operation": "terminal_then_reset"})
    assert lake._record_state_transition_in_tx(ref.task_id, "IN_PROGRESS", "FAILED")
    assert lake._record_state_transition_in_tx(ref.task_id, "FAILED", "QUEUED")
    assert finish_attempt(lake.conn, ref, {
        "effect_receipt_sha256": digest,
        "lifecycle": lifecycle_fence(lake, ref.task_id, ref.phase),
    }) == "committed"


def _directory_sync_probe(monkeypatch, folder, *, fail):
    identity = folder.stat()
    real = os.fsync
    probe = {"armed": False, "seen": []}

    def fsync(fd):
        actual = os.fstat(fd)
        if probe["armed"] and (actual.st_dev, actual.st_ino) == (identity.st_dev, identity.st_ino):
            probe["seen"].append("idea_directory")
            if fail:
                raise OSError(errno.EIO, "synthetic reset directory durability failure")
        return real(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    return probe


def test_owned_reset_syncs_directory_before_returning_to_sql_commit(case, monkeypatch):
    lake, folder, ref = case
    probe = _directory_sync_probe(monkeypatch, folder, fail=False)
    with execution_transaction(lake, folder) as tx:
        _close_for_reset(tx, lake, ref)
        probe["armed"] = True
        failure._reset_idea_for_retry(folder, release_claim=True, lake=lake, effect_lease=tx.lease)
        synced_at_return = bool(probe["seen"])
    assert synced_at_return, "outer unlock after commit cannot replace reset's own durability receipt"
    assert current_attempt(lake.conn, ref.task_id, ref.phase)["state"] == "TERMINAL"


def test_owned_reset_directory_sync_failure_holds_before_sql_or_confirmation(case, monkeypatch):
    lake, folder, ref = case
    probe = _directory_sync_probe(monkeypatch, folder, fail=True)
    returned = False
    with pytest.raises(AttemptEffectInDoubt):
        with execution_transaction(lake, folder) as tx:
            _close_for_reset(tx, lake, ref)
            probe["armed"] = True
            failure._reset_idea_for_retry(folder, release_claim=True, lake=lake, effect_lease=tx.lease)
            returned = True
    assert probe["seen"]
    assert not returned, "uncertain reset must stop its caller before SQL acceptance"
    assert current_attempt(lake.conn, ref.task_id, ref.phase)["state"] == "RUNNING"
    assert lake.get_fsm_state(ref.task_id) == "IN_PROGRESS"
    assert (folder / "_attempt_effect.lock").is_dir()
    assert not (folder / "_execution_effects" / ref.attempt_id / "committed.json").exists()


def test_nested_claim_must_propagate_uncertain_rollback_instead_of_false(tmp_path, monkeypatch):
    lake = IdeaLake(tmp_path / "lake.db")
    idea = "idea-claim-uncertain"
    folder = tmp_path / idea
    lake.insert(idea, "Claim rollback", "{}", "", status="queued")
    lake.conn.execute(
        "CREATE TRIGGER reject_claim BEFORE UPDATE ON idea_state "
        "WHEN NEW.current_state='CLAIMED' BEGIN SELECT RAISE(IGNORE); END")
    lake.conn.commit()
    real_unlink = Path.unlink

    def unlink(path, *args, **kwargs):
        if path == folder / "claim.json":
            raise OSError(errno.EIO, "synthetic claim rollback failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    returned = False
    try:
        with pytest.raises(AttemptEffectInDoubt):
            with attempt_effect_lock(folder) as lease:
                scheduler.claim(idea, tmp_path, 0, lake=lake, effect_lease=lease)
                returned = True
        assert not returned
        assert (folder / "claim.json").exists()
        assert (folder / "_attempt_effect.lock").is_dir()
        assert lake.get_fsm_state(idea) == "QUEUED"
    finally:
        lake.close()
