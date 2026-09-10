"""New filesystem-intent mechanism; no old API-absence red claims.

Real temporary files and the actual effect/source lock are used. Faults target
only OS file publication boundaries, never ownership or receipt validation.
No database commit, provider execution, or filesystem/SQLite atomicity is claimed.
"""
import errno
import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from orze.core.execution_attempts import AttemptRef
from orze.engine import attempt_effect_receipts as receipts
from orze.engine.attempt_effect_lock import (
    AttemptEffectBusy, AttemptEffectInDoubt, attempt_effect_lock,
)


@pytest.fixture
def project(tmp_path):
    folder = tmp_path / "idea-effects"
    ref = AttemptRef(folder.name, "evaluation", "attempt-1", 1)
    return folder, ref


def _paths(folder, ref):
    attempt = folder / "_execution_effects" / ref.attempt_id
    return attempt / "prepared.json", attempt / "committed.json"


def _bytes(folder):
    return {str(path.relative_to(folder)): path.read_bytes()
            for path in folder.rglob("*") if path.is_file()}


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _closed(folder, ref):
    with attempt_effect_lock(folder) as lease:
        digest = receipts.prepare_effect(lease, ref, {"file_sha256": "a" * 64, "value": 0})
        receipts.confirm_effect(lease, ref, digest)
    return digest


def test_absent_legacy_tree_is_readonly_and_not_created(tmp_path):
    folder = tmp_path / "not-created" / "idea-effects"
    receipts.require_closed_effects(folder)
    assert not folder.parent.exists()


@pytest.mark.parametrize("phase", ["evaluation", "training", "custom-phase"])
def test_complete_history_exact_confirmation_is_idempotent_and_readonly(project, phase):
    folder, original = project
    ref = AttemptRef(original.task_id, phase, original.attempt_id, 1)
    with attempt_effect_lock(folder) as lease:
        digest = receipts.prepare_effect(lease, ref, {"finite": -0.5, "explicit": True})
        prepared, committed = _paths(folder, ref)
        assert digest == hashlib.sha256(prepared.read_bytes()).hexdigest()
        assert not committed.exists()
        with pytest.raises(AttemptEffectInDoubt):
            receipts.require_closed_effects(folder)
        receipts.confirm_effect(lease, ref, digest)
        before = _bytes(folder)
        identities = {p: p.stat() for p in (prepared, committed)}
        receipts.confirm_effect(lease, ref, digest)
        receipts.require_closed_effects(folder)
        assert _bytes(folder) == before
        assert all(p.stat() == prior for p, prior in identities.items())
    with attempt_effect_lock(folder):
        receipts.require_closed_effects(folder)


def test_prepare_never_replays_or_overwrites_existing_intent(project):
    folder, ref = project
    with attempt_effect_lock(folder) as lease:
        digest = receipts.prepare_effect(lease, ref, {"revision": 1})
        for confirmed in (False, True):
            if confirmed:
                receipts.confirm_effect(lease, ref, digest)
            before = _bytes(folder)
            with pytest.raises(AttemptEffectInDoubt):
                receipts.prepare_effect(lease, ref, {"revision": 2})
            assert _bytes(folder) == before


@pytest.mark.parametrize("change", ["task", "phase", "attempt", "generation", "hash", "hash-type"])
def test_confirmation_requires_exact_reference_and_raw_hash(project, change):
    folder, ref = project
    with attempt_effect_lock(folder) as lease:
        digest = receipts.prepare_effect(lease, ref, {})
        wrong = ref
        if change in {"task", "phase", "attempt", "generation"}:
            args = [ref.task_id, ref.phase, ref.attempt_id, ref.generation]
            index = {"task": 0, "phase": 1, "attempt": 2, "generation": 3}[change]
            args[index] = 2 if index == 3 else "different"
            wrong = AttemptRef(*args)
        if change == "hash":
            digest = "0" * 64
        if change == "hash-type":
            digest = True
        before = _bytes(folder)
        with pytest.raises(AttemptEffectInDoubt):
            receipts.confirm_effect(lease, wrong, digest)
        assert _bytes(folder) == before


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("generation", True), ("generation", 1.0),
    ("phase", False), ("task_id", "different"), ("attempt_id", "different"),
])
def test_closed_receipt_identity_types_cannot_be_repaired_by_rehashing(project, field, value):
    folder, ref = project
    _closed(folder, ref)
    prepared, committed = _paths(folder, ref)
    request = json.loads(prepared.read_bytes())
    confirmation = json.loads(committed.read_bytes())
    request[field] = confirmation[field] = value
    raw = _json_bytes(request)
    prepared.write_bytes(raw)
    confirmation["prepared_sha256"] = hashlib.sha256(raw).hexdigest()
    committed.write_bytes(_json_bytes(confirmation))
    before = _bytes(folder)
    with pytest.raises(AttemptEffectInDoubt):
        receipts.require_closed_effects(folder)
    assert _bytes(folder) == before


@pytest.mark.parametrize("damage", [
    "empty-root", "partial", "extra-file", "extra-key", "symlink-root",
    "hardlink-file", "oversize", "duplicate-json-key",
])
def test_partial_redirected_or_malformed_history_is_held_readonly(project, tmp_path, damage):
    folder, ref = project
    if damage == "empty-root":
        (folder / "_execution_effects").mkdir(parents=True)
    else:
        _closed(folder, ref)
        prepared, committed = _paths(folder, ref)
        if damage == "partial":
            committed.unlink()
        elif damage == "extra-file":
            (prepared.parent / "unexpected").write_bytes(b"unresolved")
        elif damage == "extra-key":
            value = json.loads(committed.read_bytes())
            value["ignored"] = True
            committed.write_bytes(_json_bytes(value))
        elif damage == "symlink-root":
            root = folder / "_execution_effects"
            outside = tmp_path / "redirected-effects"
            root.rename(outside)
            root.symlink_to(outside, target_is_directory=True)
        elif damage == "hardlink-file":
            os.link(prepared, tmp_path / "outside-prepared")
        elif damage == "oversize":
            committed.write_bytes(b" " * (receipts.MAX_JSON_BYTES + 1))
        else:
            raw = committed.read_bytes()
            committed.write_bytes(b'{"schema_version":1,' + raw[1:])
    before = _bytes(folder)
    with pytest.raises(AttemptEffectInDoubt):
        receipts.require_closed_effects(folder)
    assert _bytes(folder) == before


def _bad_plan(kind):
    if kind == "nonfinite":
        return {"value": float("nan")}
    if kind == "oversize":
        return {"value": "x" * receipts.MAX_JSON_BYTES}
    if kind == "nodes":
        return {"value": [0] * receipts.MAX_JSON_NODES}
    if kind == "depth":
        value = {}
        for _ in range(receipts.MAX_JSON_DEPTH + 1):
            value = {"child": value}
        return value
    if kind == "recursive":
        value = {}
        value["self"] = value
        return value
    return {1: "not a string key"}


@pytest.mark.parametrize("kind", ["nonfinite", "oversize", "nodes", "depth", "recursive", "key"])
def test_plan_limits_fail_before_creating_effect_history(project, kind):
    folder, ref = project
    with attempt_effect_lock(folder) as lease:
        with pytest.raises(AttemptEffectInDoubt):
            receipts.prepare_effect(lease, ref, _bad_plan(kind))
        assert not (folder / "_execution_effects").exists()


def test_history_budget_applies_to_read_and_new_prepare(project, monkeypatch):
    folder, first = project
    assert receipts.MAX_EFFECTS == 1024
    monkeypatch.setattr(receipts, "MAX_EFFECTS", 2)
    _closed(folder, first)
    second = AttemptRef(first.task_id, first.phase, "attempt-2", 2)
    _closed(folder, second)
    with attempt_effect_lock(folder) as lease:
        before = _bytes(folder)
        third = AttemptRef(first.task_id, first.phase, "attempt-3", 3)
        with pytest.raises(AttemptEffectInDoubt):
            receipts.prepare_effect(lease, third, {})
        assert _bytes(folder) == before
    monkeypatch.setattr(receipts, "MAX_EFFECTS", 1)
    with pytest.raises(AttemptEffectInDoubt):
        receipts.require_closed_effects(folder)


def test_foreign_stale_and_path_traversal_refs_never_write(project):
    folder, ref = project
    with attempt_effect_lock(folder) as lease:
        for wrong in (AttemptRef("other-task", ref.phase, ref.attempt_id, 1),
                      AttemptRef(ref.task_id, ref.phase, "..", 1)):
            with pytest.raises(AttemptEffectInDoubt):
                receipts.prepare_effect(lease, wrong, {})
        assert not (folder / "_execution_effects").exists()
    with pytest.raises(AttemptEffectInDoubt):
        receipts.prepare_effect(lease, ref, {})
    assert not (folder / "_execution_effects").exists()


def test_positive_short_writes_are_completed_before_prepare_returns(project, monkeypatch):
    folder, ref = project
    write = os.write
    calls = []
    with attempt_effect_lock(folder) as lease:
        def short_write(fd, data):
            calls.append(len(data))
            return write(fd, data[:7])
        with monkeypatch.context() as faults:
            faults.setattr(os, "write", short_write)
            digest = receipts.prepare_effect(lease, ref, {"metadata": "small"})
        assert len(calls) > 1
        receipts.confirm_effect(lease, ref, digest)
    receipts.require_closed_effects(folder)


@pytest.mark.parametrize("fault", ["file-fsync", "parent-fsync", "zero-write", "confirm-open", "confirm-fsync"])
def test_publication_failures_keep_uncertain_ownership_without_overwrite(project, monkeypatch, fault):
    folder, ref = project
    fsync, write, open_fd = os.fsync, os.write, os.open
    prepared, committed = _paths(folder, ref)
    injected = []

    def fail_fsync(fd):
        info = os.fstat(fd)
        is_file = stat.S_ISREG(info.st_mode)
        is_attempt_dir = (prepared.parent.exists()
                          and (info.st_dev, info.st_ino) ==
                          (prepared.parent.stat().st_dev, prepared.parent.stat().st_ino))
        if ((fault in {"file-fsync", "confirm-fsync"} and is_file)
                or (fault == "parent-fsync" and is_attempt_dir)):
            injected.append("fsync")
            raise OSError(errno.ENOSPC, "fixture durability failure")
        return fsync(fd)

    def fail_write(fd, data):
        if injected:
            return 0
        injected.append("write")
        return write(fd, data[:1])

    def fail_open(path, flags, *args, **kwargs):
        if Path(path) == committed and flags & os.O_CREAT:
            injected.append("open")
            raise OSError(errno.ENOSPC, "fixture confirmation creation failure")
        return open_fd(path, flags, *args, **kwargs)

    with pytest.raises(AttemptEffectInDoubt):
        with attempt_effect_lock(folder) as lease:
            if fault.startswith("confirm"):
                digest = receipts.prepare_effect(lease, ref, {"unchanged": True})
                original = prepared.read_bytes()
            with monkeypatch.context() as faults:
                if fault == "zero-write":
                    faults.setattr(os, "write", fail_write)
                elif fault == "confirm-open":
                    faults.setattr(os, "open", fail_open)
                else:
                    faults.setattr(os, "fsync", fail_fsync)
                if fault.startswith("confirm"):
                    try:
                        receipts.confirm_effect(lease, ref, digest)
                    finally:
                        assert prepared.read_bytes() == original
                else:
                    receipts.prepare_effect(lease, ref, {"unchanged": True})
    assert injected
    assert (folder / "_attempt_effect.lock" / "lock.json").is_file()
    with pytest.raises((AttemptEffectBusy, AttemptEffectInDoubt)):
        with attempt_effect_lock(folder):
            pytest.fail("uncertain publication automatically reacquired ownership")
