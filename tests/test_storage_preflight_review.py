"""Independent storage denial boundaries, on new private local directories.

Real rename primitives and filesystem contents are used. Transparent hooks
inject the stated path race or primitive failure; they do not grant GC/role
authority. No worker, provider, GPU or deployment is started. The one actual
unsupported-filesystem control skips if that precise local feature is absent.
"""
from pathlib import Path
import os

import pytest

from orze.engine import gc_safety, gc_tree, storage_preflight as storage
from orze.engine.attempt_effect_lock import attempt_effect_lock
from orze.engine.gc_tree import GCRefused
from test_gc_scope_mechanisms import project
from test_storage_preflight import unsupported_root


def test_path_replacement_with_same_probe_names_cannot_move_foreign_source(tmp_path, monkeypatch):
    root = tmp_path / "root"
    retained = tmp_path / "retained-original-root"
    root.mkdir()
    actual = storage.rename_no_replace_at
    directory = storage._Probe.directory
    parents = {}
    foreign = {}

    def capture_directory(probe, name):
        fd = directory(probe, name)
        parents[fd] = probe.path / name
        return fd

    def swap_before_real_rename(source_fd, source_name, target_fd, target_name):
        source, target = parents[source_fd] / source_name, parents[target_fd] / target_name
        # A real replacement exactly between the preflight's path/FD check and
        # the real primitive. Only our newly created private roots are changed.
        if not foreign:
            root.rename(retained)
            root.mkdir()
            source.parent.mkdir(parents=True)
            source.write_bytes(b"foreign source must retain its name and inode")
            foreign.update(source=source, target=target, inode=source.stat().st_ino)
        return actual(source_fd, source_name, target_fd, target_name)

    monkeypatch.setattr(storage._Probe, "directory", capture_directory)
    monkeypatch.setattr(storage, "rename_no_replace_at", swap_before_real_rename)
    with pytest.raises(storage.StoragePreflightError):
        storage.require_atomic_rename_support(root)
    assert foreign, "the controlled replacement window was not reached"
    assert foreign["source"].is_file(), "preflight renamed an uncaptured replacement source"
    assert foreign["source"].stat().st_ino == foreign["inode"]
    assert foreign["source"].read_bytes() == b"foreign source must retain its name and inode"
    assert not foreign["target"].exists()


def test_cleanup_retains_untracked_entry_inside_original_probe(tmp_path, monkeypatch):
    keep = tmp_path / "user-file"
    keep.write_bytes(b"outside the probe")
    actual = storage.rename_no_replace_at
    directory = storage._Probe.directory
    parents = {}
    foreign = []

    def capture_directory(probe, name):
        fd = directory(probe, name)
        parents[fd] = probe.path / name
        return fd

    def insert_after_real_rename(source_fd, source_name, target_fd, target_name):
        result = actual(source_fd, source_name, target_fd, target_name)
        target = parents[target_fd] / target_name
        if not foreign:
            entry = target.parent / "foreign-entry"
            entry.write_bytes(b"not created by probe")
            foreign.append((entry, entry.stat().st_ino))
        return result

    monkeypatch.setattr(storage._Probe, "directory", capture_directory)
    monkeypatch.setattr(storage, "rename_no_replace_at", insert_after_real_rename)
    with pytest.raises(storage.StoragePreflightError) as caught:
        storage.require_atomic_rename_support(tmp_path)
    assert foreign and Path(caught.value.probe).is_dir()
    entry, inode = foreign[0]
    assert entry.read_bytes() == b"not created by probe" and entry.stat().st_ino == inode
    assert keep.read_bytes() == b"outside the probe"


def test_deployment_role_route_refuses_intermediate_symlink_before_creating_locks(tmp_path):
    control = tmp_path / "control"
    foreign = tmp_path / "foreign"
    control.mkdir()
    foreign.mkdir()
    (foreign / "keep").write_bytes(b"unrelated administration")
    (control / "redirect").symlink_to(foreign, target_is_directory=True)
    cfg = {"_orze_dir": str(control / "redirect" / "nested-admin"),
           "roles": {"worker": {"mode": "script", "script": "not-executed.py"}}}
    with pytest.raises(storage.StoragePreflightError):
        storage.require_deployment_storage(cfg, tmp_path)
    assert sorted(p.name for p in foreign.iterdir()) == ["keep"]
    assert (foreign / "keep").read_bytes() == b"unrelated administration"
    assert (control / "redirect").is_symlink()


def test_real_existing_effect_owner_prevents_archive_creation(tmp_path):
    scope, task, _ = project(tmp_path)
    source = task / "scratch.pt"
    original = source.read_bytes()
    with attempt_effect_lock(task):
        lock = task / "_attempt_effect.lock"
        before = {str(p.relative_to(lock)): p.read_bytes() for p in lock.rglob("*") if p.is_file()}
        stats = gc_safety.collect(scope, scope.results_dir, set(), mode="archive")
        assert stats["errors"] == 1 and stats["archived_files"] == 0
        assert source.read_bytes() == original
        assert not scope.archive_dir.exists()
        assert not (scope.results_dir / "_orze_gc_quarantine").exists()
        assert {str(p.relative_to(lock)): p.read_bytes() for p in lock.rglob("*") if p.is_file()} == before


def test_successful_probe_is_not_permission_when_final_gc_rename_refuses(tmp_path, monkeypatch):
    scope, task, checkpoint = project(tmp_path)
    storage.require_atomic_rename_support(scope.checkpoints_dir)
    attempted = []

    def unsupported_final_rename(source, target):
        attempted.append((source, target))
        raise GCRefused("gc_atomic_rename_unsupported")

    # Only the final destructive operation fails; preflight still uses the
    # real primitive successfully. This simulates a later capability failure.
    monkeypatch.setattr(gc_safety, "rename_no_replace", unsupported_final_rename)
    stats = gc_safety.collect(scope, scope.checkpoints_dir, set(), mode="checkpoints")
    assert len(attempted) == 1 and attempted[0][0] == checkpoint
    assert stats["deleted"] == 0 and stats["errors"] == 1
    assert (checkpoint / "weights.pt").read_bytes() == b"checkpoint bytes"
    assert (task / "_attempt_effect.lock").is_dir()
    assert not list((scope.checkpoints_dir / "_orze_gc_quarantine").rglob("completed.json"))


def test_actual_unsupported_role_route_does_not_create_a_role_owner(unsupported_root):
    cfg = {"_orze_dir": str(unsupported_root / "control")}
    role = {"mode": "research", "backend": "not-invoked"}
    with pytest.raises(storage.StoragePreflightError, match="storage_atomic_rename_unsupported"):
        storage.require_role_storage(cfg, role)
    locks = unsupported_root / "control" / "locks"
    assert locks.is_dir() and list(locks.iterdir()) == []
    assert not (locks / "research").exists()


@pytest.mark.parametrize("name", ["", "..", "nested/name"])
def test_fd_rename_rejects_invalid_basename_without_touching_entries(tmp_path, name):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"source stays")
    target.write_bytes(b"target stays")
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        with pytest.raises(GCRefused):
            gc_tree.rename_no_replace_at(fd, name, fd, "target")
    finally:
        os.close(fd)
    assert source.read_bytes() == b"source stays"
    assert target.read_bytes() == b"target stays"


def test_fd_rename_rejects_regular_file_descriptor_without_touching_entries(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"source stays")
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    file_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        with pytest.raises(GCRefused):
            gc_tree.rename_no_replace_at(file_fd, "source", fd, "target")
    finally:
        os.close(file_fd)
        os.close(fd)
    assert source.read_bytes() == b"source stays"
    assert not (tmp_path / "target").exists()
