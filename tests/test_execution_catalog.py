"""Bounded native routing declarations, not capabilities or DB commit proof."""
import errno
import json
import os
from pathlib import Path
import stat

import pytest

from orze.engine import execution_catalog as catalog
from orze.engine.attempt_effect_lock import (
    AttemptEffectBusy, AttemptEffectInDoubt, attempt_effect_lock,
)
from orze.engine.termination_hold import TerminationUnconfirmed
from orze.idea_lake import IdeaLake


@pytest.fixture
def project(tmp_path):
    folder = tmp_path / "idea-routing"
    lake = IdeaLake(tmp_path / "catalog.sqlite")
    try:
        yield folder, lake
    finally:
        lake.close()


def _bind(folder, lake):
    with attempt_effect_lock(folder) as lease:
        catalog.bind_catalog(lake, folder, lease)
    return folder / catalog.CATALOG_FILE


def test_absent_declaration_is_readonly_and_does_not_create_parent(tmp_path):
    folder = tmp_path / "absent" / "idea-routing"
    assert catalog.declared_catalog(folder) is None
    assert not folder.parent.exists()


def test_actual_binding_and_same_scope_replay_are_readonly_and_canonical(project):
    folder, lake = project
    path = _bind(folder, lake)
    raw, info = path.read_bytes(), path.stat()
    assert catalog.declared_catalog(folder) == str(Path(lake.db_path).absolute())
    assert set(json.loads(raw)) == {"schema", "task_id", "database"}
    assert not lake.conn.in_transaction
    with attempt_effect_lock(folder) as lease:
        catalog.bind_catalog(lake, folder, lease)
    assert path.read_bytes() == raw
    assert path.stat() == info


def test_other_database_cannot_rebind_existing_route(project, tmp_path):
    folder, lake = project
    path = _bind(folder, lake)
    before = path.read_bytes()
    other = IdeaLake(tmp_path / "other.sqlite")
    try:
        with attempt_effect_lock(folder) as lease:
            with pytest.raises(AttemptEffectBusy):
                catalog.bind_catalog(other, folder, lease)
        assert path.read_bytes() == before
    finally:
        other.close()


def test_other_task_cannot_borrow_a_lease_for_publication(project, tmp_path):
    folder, lake = project
    other = tmp_path / "idea-other"
    with attempt_effect_lock(folder) as lease:
        with pytest.raises(AttemptEffectInDoubt):
            catalog.bind_catalog(lake, other, lease)
    assert not other.exists()


@pytest.mark.parametrize("damage", ["schema-bool", "extra", "relative-db", "oversize", "invalid-utf8"])
def test_malformed_declarations_are_refused_without_repair(project, damage):
    folder, lake = project
    path = _bind(folder, lake)
    value = json.loads(path.read_bytes())
    if damage == "schema-bool":
        value["schema"] = True
    elif damage == "extra":
        value["ignored"] = "unsafe extension"
    elif damage == "relative-db":
        value["database"] = "relative.sqlite"
    if damage == "oversize":
        raw = b" " * 8193
    elif damage == "invalid-utf8":
        raw = b"\xff"
    else:
        raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(raw)
    with pytest.raises(TerminationUnconfirmed):
        catalog.declared_catalog(folder)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("redirect", ["file-symlink", "file-hardlink", "parent-symlink"])
def test_redirected_declarations_never_supply_native_routing(project, tmp_path, redirect):
    folder, lake = project
    path = _bind(folder, lake)
    raw = path.read_bytes()
    if redirect == "parent-symlink":
        parent = tmp_path / "linked-parent"
        parent.mkdir()
        alias = parent / folder.name
        alias.symlink_to(folder, target_is_directory=True)
        observed_folder = alias
    else:
        outside = tmp_path / "outside.json"
        if redirect == "file-hardlink":
            os.link(path, outside)
        else:
            path.rename(outside)
            path.symlink_to(outside)
        observed_folder = folder
    with pytest.raises(TerminationUnconfirmed):
        catalog.declared_catalog(observed_folder)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("fault", ["lstat", "open", "read", "close"])
def test_reader_io_uncertainty_is_a_dedicated_hold(project, monkeypatch, fault):
    folder, lake = project
    path = _bind(folder, lake)
    before = path.read_bytes()
    original_lstat, original_close = Path.lstat, os.close

    def fail(*args, **kwargs):
        raise OSError(errno.EIO, "fixture catalog read uncertainty")

    def fail_lstat(item, *args, **kwargs):
        if item == path:
            return fail()
        return original_lstat(item, *args, **kwargs)

    def fail_close(fd):
        original_close(fd)
        return fail()

    with monkeypatch.context() as faults:
        if fault == "lstat":
            faults.setattr(Path, "lstat", fail_lstat)
        else:
            faults.setattr(os, fault, fail_close if fault == "close" else fail)
        with pytest.raises(TerminationUnconfirmed):
            catalog.declared_catalog(folder)
    assert path.read_bytes() == before


def test_same_scope_replay_rechecks_lease_after_reading_declaration(project, monkeypatch):
    folder, lake = project
    path = _bind(folder, lake)
    open_fd = os.open
    observed = []
    with attempt_effect_lock(folder) as lease:
        owner = lease.owner.lock_dir / "lock.json"
        original = owner.read_bytes()

        def change_owner_before_route_read(item, flags, *args, **kwargs):
            if Path(item) == path and not observed:
                owner.write_bytes(b'{"owner_nonce":"different"}')
                observed.append(True)
            return open_fd(item, flags, *args, **kwargs)

        try:
            with monkeypatch.context() as faults:
                faults.setattr(os, "open", change_owner_before_route_read)
                with pytest.raises(AttemptEffectInDoubt):
                    catalog.bind_catalog(lake, folder, lease)
        finally:
            # Restore only this fixture's owner metadata for context cleanup;
            # the assertion above concerns bind_catalog's own return boundary.
            owner.write_bytes(original)
        assert observed == [True]


@pytest.mark.parametrize("reported", [False, True])
def test_silent_atomic_create_cannot_claim_publication(project, monkeypatch, reported):
    folder, lake = project
    with pytest.raises(AttemptEffectInDoubt):
        with attempt_effect_lock(folder) as lease:
            monkeypatch.setattr(catalog, "atomic_create", lambda *args: reported)
            catalog.bind_catalog(lake, folder, lease)
    assert not (folder / catalog.CATALOG_FILE).exists()
    assert (folder / "_attempt_effect.lock" / "lock.json").exists()


@pytest.mark.parametrize("target", ["file", "directory"])
def test_actual_publication_fsync_failure_retains_uncertain_owner(project, monkeypatch, target):
    folder, lake = project
    fsync = os.fsync
    calls = []

    def fail_target_fsync(fd):
        kind = "file" if stat.S_ISREG(os.fstat(fd).st_mode) else "directory"
        if kind == target:
            calls.append(kind)
            raise OSError(errno.ENOSPC, "fixture publication durability failure")
        return fsync(fd)

    with pytest.raises(AttemptEffectInDoubt):
        with attempt_effect_lock(folder) as lease:
            with monkeypatch.context() as faults:
                faults.setattr(os, "fsync", fail_target_fsync)
                catalog.bind_catalog(lake, folder, lease)
    assert calls
    assert (folder / "_attempt_effect.lock" / "lock.json").exists()
