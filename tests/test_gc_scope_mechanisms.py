"""New explicit-scope/quarantine mechanisms, not old missing-API regressions."""
from pathlib import Path
import os

import pytest

from orze.agents.orze_gc import gc_checkpoints, gc_results, archive_to_cold_storage
from orze.engine.gc_safety import gc_scope, GCRefused


def project(tmp_path):
    results, checkpoints, archive = (tmp_path / name for name in ("results", "checkpoints", "archive"))
    task = results / "idea-disposable"
    task.mkdir(parents=True)
    (task / "metrics.json").write_bytes(b'{"status":"FAILED"}')
    checkpoint = checkpoints / task.name
    checkpoint.mkdir(parents=True)
    (checkpoint / "weights.pt").write_bytes(b"checkpoint bytes")
    (task / "scratch.pt").write_bytes(b"result bytes")
    cfg = {"_project_root": str(tmp_path), "results_dir": str(results)}
    scope = gc_scope(results, cfg, checkpoints_dir=checkpoints, archive_dir=archive)
    return scope, task, checkpoint


@pytest.mark.parametrize("operation", ["checkpoints", "results", "archive"])
def test_explicit_scope_permits_real_legacy_disposable_operation(tmp_path, operation):
    scope, task, checkpoint = project(tmp_path)
    if operation == "checkpoints":
        stats = gc_checkpoints(scope.checkpoints_dir, set(), scope=scope)
        assert stats["deleted"] == 1 and not checkpoint.exists()
        assert (task / "scratch.pt").is_file()
    elif operation == "results":
        stats = gc_results(scope.results_dir, set(), scope=scope)
        assert stats["deleted_files"] == 1 and not (task / "scratch.pt").exists()
        assert checkpoint.is_dir()
    else:
        stats = archive_to_cold_storage(scope.results_dir, scope.archive_dir, set(), scope=scope)
        assert stats["archived_files"] == 1 and not (task / "scratch.pt").exists()
        assert (scope.archive_dir / task.name / "scratch.pt").read_bytes() == b"result bytes"
        assert stats["moved_bytes"] == len(b"result bytes") and stats["freed_bytes"] == 0
    assert stats["errors"] == 0
    assert (task / "metrics.json").read_bytes() == b'{"status":"FAILED"}'


def test_scoped_checkpoint_rejects_nested_redirect_before_detach(tmp_path):
    scope, task, checkpoint = project(tmp_path)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "data.bin").write_bytes(b"other data")
    (checkpoint / "borrowed").symlink_to(foreign, target_is_directory=True)
    stats = gc_checkpoints(scope.checkpoints_dir, set(), scope=scope)
    assert stats["deleted"] == 0 and stats["errors"] == 1
    assert checkpoint.is_dir() and (foreign / "data.bin").read_bytes() == b"other data"
    assert not (scope.checkpoints_dir / "_orze_gc_quarantine").exists()


def test_scope_cannot_be_reused_for_different_checkpoint_root(tmp_path):
    scope, task, checkpoint = project(tmp_path)
    other = tmp_path / "foreign-checkpoints"
    other.mkdir()
    with pytest.raises(GCRefused, match="gc_operation_scope_mismatch"):
        gc_checkpoints(other, set(), scope=scope)
    assert checkpoint.is_dir() and list(other.iterdir()) == []


def test_scoped_archive_retains_existing_destination_and_source(tmp_path):
    scope, task, checkpoint = project(tmp_path)
    target = scope.archive_dir / task.name / "scratch.pt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"already archived")
    stats = archive_to_cold_storage(scope.results_dir, scope.archive_dir, set(), scope=scope)
    assert stats["archived_files"] == 0 and stats["errors"] == 1
    assert target.read_bytes() == b"already archived"
    assert (task / "scratch.pt").read_bytes() == b"result bytes"
    assert not (scope.results_dir / "_orze_gc_quarantine").exists()


def test_uncertain_detach_retains_effect_owner_and_never_counts_success(tmp_path, monkeypatch):
    import orze.engine.gc_safety as safety
    scope, task, checkpoint = project(tmp_path)
    original = safety.rename_no_replace
    def uncertain(source, target):
        original(source, target)
        raise OSError("selected lost acknowledgement after rename")
    monkeypatch.setattr(safety, "rename_no_replace", uncertain)
    stats = gc_checkpoints(scope.checkpoints_dir, set(), scope=scope)
    assert stats["deleted"] == 0 and stats["errors"] == 1
    assert not checkpoint.exists()
    held = list((scope.checkpoints_dir / "_orze_gc_quarantine" / task.name).glob("*/content/weights.pt"))
    assert len(held) == 1 and held[0].read_bytes() == b"checkpoint bytes"
    assert (task / "_attempt_effect.lock").is_dir()
    assert not list((scope.checkpoints_dir / "_orze_gc_quarantine").rglob("completed.json"))


def test_reclamation_failure_leaves_quarantine_without_automatic_retry(tmp_path, monkeypatch):
    scope, task, checkpoint = project(tmp_path)
    original = os.unlink
    def denied(path, *args, **kwargs):
        if path == "weights.pt":
            raise PermissionError("selected quarantined payload")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(os, "unlink", denied)
    stats = gc_checkpoints(scope.checkpoints_dir, set(), scope=scope)
    assert stats["deleted"] == 0 and stats["errors"] == 1
    held = list((scope.checkpoints_dir / "_orze_gc_quarantine" / task.name).glob("*/content/weights.pt"))
    assert len(held) == 1 and held[0].read_bytes() == b"checkpoint bytes"
    assert not checkpoint.exists()
    second = gc_checkpoints(scope.checkpoints_dir, set(), scope=scope)
    assert second["deleted"] == 0
    assert held[0].read_bytes() == b"checkpoint bytes"
