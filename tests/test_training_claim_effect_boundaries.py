"""Existing scheduler/reset entrypoints must respect D2 effect ownership.

These are draft cross-protocol behavioral regressions, not claims that the
new effect-lock API existed in the original release. No GPU/provider is run.
"""
import json
import os
import socket

import pytest

from orze.engine import failure, scheduler
from orze.engine.attempt_effect_lock import attempt_effect_lock
from orze.engine.termination_hold import TerminationUnconfirmed


def _files(folder):
    return {str(path.relative_to(folder)): path.read_bytes()
            for path in folder.rglob("*") if path.is_file()}


def _claimed(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "capture_process_identity", lambda pid: {"start_ticks": 13})
    assert scheduler.claim("idea-claim-boundary", tmp_path, 0)
    folder = tmp_path / "idea-claim-boundary"
    (folder / "metrics.json").write_bytes(b'{"status":"FAILED","quality":-1}')
    (folder / "train_output.log").write_bytes(b"old training log")
    return folder


def test_claim_cannot_publish_inside_another_effect_owner(tmp_path):
    folder = tmp_path / "idea-claim-boundary"
    with attempt_effect_lock(folder):
        before = _files(folder)
        assert scheduler.claim(folder.name, tmp_path, 0) is False
        assert _files(folder) == before


def test_reset_cannot_delete_metrics_or_rotate_claim_inside_another_effect_owner(tmp_path, monkeypatch):
    folder = _claimed(tmp_path, monkeypatch)
    with attempt_effect_lock(folder):
        before = _files(folder)
        with pytest.raises(TerminationUnconfirmed):
            failure._reset_idea_for_retry(folder)
        assert _files(folder) == before


def test_orphan_cleanup_cannot_archive_another_effect_owners_claim(tmp_path, monkeypatch):
    folder = _claimed(tmp_path, monkeypatch)
    claim = folder / "claim.json"
    value = json.loads(claim.read_bytes())
    value.update(claimed_by=socket.gethostname(), pid=112233, owner_start_ticks=13)
    claim.write_text(json.dumps(value))
    for path in (claim, folder / "train_output.log"):
        os.utime(path, (1, 1))
    monkeypatch.setattr(scheduler, "process_is_running", lambda *args: False)
    with attempt_effect_lock(folder):
        before = _files(folder)
        assert scheduler.cleanup_orphans(tmp_path, 1) == 0
        assert _files(folder) == before


def test_invalid_claim_is_validated_before_reset_deletes_training_evidence(tmp_path, monkeypatch):
    folder = _claimed(tmp_path, monkeypatch)
    (folder / "claim.json").write_bytes(b"{invalid claim")
    before = _files(folder)
    with pytest.raises(RuntimeError):
        failure._reset_idea_for_retry(folder)
    assert _files(folder) == before
