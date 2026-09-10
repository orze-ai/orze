"""New native-attempt routing/lease mechanism; no old-API absence reds."""
import json
import os
from pathlib import Path

import pytest

from orze.core.execution_attempts import create_attempt, finish_attempt, hold_attempt, mark_running
from orze.engine import failure, scheduler
from orze.engine.attempt_effect_lock import attempt_effect_lock
from orze.engine.termination_hold import TerminationUnconfirmed
from orze.idea_lake import IdeaLake


@pytest.fixture
def project(tmp_path, monkeypatch):
    lake = IdeaLake(tmp_path / "custom-catalog.sqlite")
    idea = "idea-native-claim"
    lake.insert(idea, "Native claim", "{}", "", status="queued")
    monkeypatch.setattr(scheduler, "capture_process_identity", lambda pid: {"start_ticks": 13})
    try:
        yield tmp_path, lake, idea
    finally:
        lake.close()


def _files(folder):
    return {str(path.relative_to(folder)): path.read_bytes()
            for path in folder.rglob("*") if path.is_file()}


def _claimed(project):
    root, lake, idea = project
    assert scheduler.claim(idea, root, 0, lake=lake)
    folder = root / idea
    (folder / "metrics.json").write_bytes(b'{"status":"FAILED","quality":0}')
    (folder / "train_output.log").write_bytes(b"training evidence")
    return folder


def _attempt(lake, idea, state, phase="training"):
    lake.conn.execute("BEGIN IMMEDIATE")
    ref = create_attempt(lake.conn, idea, phase, "native-attempt-A", {})
    if state in {"RUNNING", "TERMINAL"}:
        mark_running(lake.conn, ref)
    if state == "TERMINAL":
        assert finish_attempt(lake.conn, ref, {"outcome": "failed"}) == "committed"
    if state == "IN_DOUBT":
        hold_attempt(lake.conn, ref, "unknown execution")
    lake.conn.commit()
    return ref


@pytest.mark.parametrize("state", ["LAUNCHING", "RUNNING", "IN_DOUBT"])
@pytest.mark.parametrize("phase", ["training", "evaluation"])
def test_reset_without_lake_uses_bound_db_and_preserves_open_attempt_evidence(project, state, phase):
    _, lake, idea = project
    folder = _claimed(project)
    assert json.loads((folder / "claim.json").read_bytes())["lifecycle_db"] == str(Path(lake.db_path).absolute())
    _attempt(lake, idea, state, phase)
    before = _files(folder)
    with pytest.raises(TerminationUnconfirmed):
        failure._reset_idea_for_retry(folder)
    assert _files(folder) == before


def test_closed_native_attempt_allows_renewal_and_keeps_absolute_scope(project):
    _, lake, idea = project
    folder = _claimed(project)
    _attempt(lake, idea, "TERMINAL")
    original = json.loads((folder / "claim.json").read_bytes())
    failure._reset_idea_for_retry(folder)
    renewed = json.loads((folder / "claim.json").read_bytes())
    assert renewed["lifecycle_db"] == original["lifecycle_db"]
    assert renewed["attempt_id"] != original["attempt_id"]
    assert not (folder / "metrics.json").exists()
    archive = list(folder.glob("claim.retry.*.json"))
    assert len(archive) == 1 and json.loads(archive[0].read_bytes()) == original


@pytest.mark.parametrize("binding", [None, "relative.db", "missing", "corrupt", "symlink"])
def test_invalid_explicit_db_binding_never_downgrades_to_legacy_or_creates_db(project, binding):
    root, _, _ = project
    folder = _claimed(project)
    claim = folder / "claim.json"
    value = json.loads(claim.read_bytes())
    missing = root / "never-created.db"
    if binding == "missing":
        value["lifecycle_db"] = str(missing)
    elif binding == "corrupt":
        corrupt = root / "corrupt.sqlite"
        corrupt.write_bytes(b"not SQLite")
        value["lifecycle_db"] = str(corrupt)
    elif binding == "symlink":
        redirected = root / "redirect.sqlite"
        redirected.symlink_to(value["lifecycle_db"])
        value["lifecycle_db"] = str(redirected)
    else:
        value["lifecycle_db"] = binding
    claim.write_text(json.dumps(value))
    before = _files(folder)
    with pytest.raises(TerminationUnconfirmed):
        failure._reset_idea_for_retry(folder, release_claim=True)
    assert _files(folder) == before
    assert not missing.exists()


def test_claim_rejects_open_native_attempt_even_when_no_claim_or_metrics_exist(project):
    root, lake, idea = project
    _attempt(lake, idea, "LAUNCHING")
    assert scheduler.claim(idea, root, 0, lake=lake) is False
    assert not (root / idea / "claim.json").exists()
    assert lake.get_fsm_state(idea) == "QUEUED"


def test_orphan_cleanup_without_lake_keeps_native_open_claim(project, monkeypatch):
    root, lake, idea = project
    folder = _claimed(project)
    _attempt(lake, idea, "RUNNING")
    for name in ("claim.json", "train_output.log"):
        os.utime(folder / name, (1, 1))
    monkeypatch.setattr(scheduler, "process_is_running", lambda *args: False)
    before = _files(folder)
    assert scheduler.cleanup_orphans(root, 1) == 0
    assert _files(folder) == before


def test_exact_explicit_effect_lease_allows_claim_and_reset_without_implicit_reentry(project):
    root, lake, idea = project
    folder = root / idea
    with attempt_effect_lock(folder) as lease:
        assert scheduler.claim(idea, root, 0, lake=lake) is False
        assert scheduler.claim(idea, root, 0, lake=lake, effect_lease=lease)
        (folder / "metrics.json").write_bytes(b'{"status":"FAILED"}')
        failure._reset_idea_for_retry(folder, effect_lease=lease)
        assert not (folder / "metrics.json").exists()
        assert (folder / "claim.json").exists()


@pytest.mark.parametrize("rollback", [False, True], ids=["commit", "rollback"])
def test_explicit_owned_transaction_may_reset_its_just_closed_attempt_without_committing(project, rollback):
    from contextlib import nullcontext
    from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
    from orze.engine.execution_authority import execution_transaction, lifecycle_fence

    _, lake, idea = project
    folder = _claimed(project)
    assert lake.record_state_transition(idea, "CLAIMED", "IN_PROGRESS")
    ref = _attempt(lake, idea, "RUNNING")
    peer = IdeaLake(lake.db_path)
    try:
        expectation = pytest.raises(AttemptEffectInDoubt) if rollback else nullcontext()
        with expectation:
            with execution_transaction(lake, folder) as tx:
                digest = tx.prepare(ref, {"operation": "training_requeue"})
                assert lake._record_state_transition_in_tx(idea, "IN_PROGRESS", "FAILED", "fixture requeue")
                assert lake._record_state_transition_in_tx(idea, "FAILED", "QUEUED", "fixture requeue")
                assert finish_attempt(lake.conn, ref, {
                    "outcome": "requeued", "effect_receipt_sha256": digest,
                    "lifecycle": lifecycle_fence(lake, idea, "training"),
                }) == "committed"
                failure._reset_idea_for_retry(
                    folder, release_claim=True, lake=lake, effect_lease=tx.lease)
                assert lake.conn.in_transaction
                assert peer.conn.execute("SELECT state FROM main.execution_attempts").fetchone()[0] == "RUNNING"
                assert not (folder / "metrics.json").exists()
                if rollback:
                    raise ValueError("fixture caller rejects after prepared file effect")
        assert not lake.conn.in_transaction
        assert peer.conn.execute("SELECT state FROM main.execution_attempts").fetchone()[0] == (
            "RUNNING" if rollback else "TERMINAL")
        assert peer.get_fsm_state(idea) == ("IN_PROGRESS" if rollback else "QUEUED")
        attempt = folder / "_execution_effects" / ref.attempt_id
        assert (attempt / "prepared.json").exists()
        assert (attempt / "committed.json").exists() is not rollback
        if rollback:
            assert (folder / "_attempt_effect.lock" / "lock.json").exists()
    finally:
        peer.close()
