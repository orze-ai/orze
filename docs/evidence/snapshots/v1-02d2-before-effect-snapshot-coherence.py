"""Bounded filesystem intents for one controller publication per attempt.

CALLING SPEC: acquire the short task effect guard, BEGIN IMMEDIATE, and verify
the current AttemptRef in SQLite before prepare_effect. Commit the database
transaction before confirm_effect. A committed.json file only records that
the caller reported this ordering; it is NOT independent database-commit proof.
These functions never access SQLite, run a process, or undo partial effects.
The coordinator must retain ownership on uncertain commit/publication outcomes.
An existing prepared intent cannot be replayed, even when already confirmed.
Only exact confirmation of that intent is idempotent. No automatic recovery.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import stat

from orze.core.execution_attempts import AttemptRef
from orze.engine.attempt_effect_lock import (
    AttemptEffectInDoubt, AttemptEffectLease, require_effect_lease,
)

MAX_EFFECTS = 1024
MAX_JSON_BYTES = 65536
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 2048
_TREE = "_execution_effects"
_TOKEN = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_IDENTITY = {"schema_version", "task_id", "phase", "attempt_id", "generation"}
_ERRORS = (OSError, ValueError, TypeError, UnicodeError, RecursionError, OverflowError)


def _encoded(value: dict) -> bytes:
    if type(value) is not dict:
        raise ValueError("effect_json_object_required")
    nodes, string_bytes = 0, 0
    active = set()

    def visit(item, depth):
        nonlocal nodes, string_bytes
        nodes += 1
        if nodes > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
            raise ValueError("effect_json_complexity_limit")
        kind = type(item)
        if kind in (dict, list):
            if id(item) in active:
                raise ValueError("effect_json_recursive")
            active.add(id(item))
            if kind is dict:
                for key, child in item.items():
                    if type(key) is not str:
                        raise ValueError("effect_json_key_invalid")
                    visit(key, depth + 1)
                    visit(child, depth + 1)
            else:
                for child in item:
                    visit(child, depth + 1)
            active.remove(id(item))
        elif kind is str:
            if len(item) > MAX_JSON_BYTES:
                raise ValueError("effect_json_byte_limit")
            string_bytes += len(item.encode("utf-8"))
            if string_bytes > MAX_JSON_BYTES:
                raise ValueError("effect_json_byte_limit")
        elif kind is float:
            if not math.isfinite(item):
                raise ValueError("effect_json_nonfinite")
        elif kind not in (int, bool, type(None)):
            raise ValueError("effect_json_value_invalid")

    visit(value, 0)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("effect_json_byte_limit")
    return raw


def _decode(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("effect_json_duplicate_key")
            result[key] = value
        return result

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    if _encoded(value) != raw:
        raise ValueError("effect_json_noncanonical")
    return value


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _directory(path: Path, *, missing_ok=False):
    if ".." in path.parts:
        raise ValueError("effect_directory_traversal")
    found = None
    for index, current in enumerate((path, *path.parents)):
        try:
            info = current.lstat()
        except FileNotFoundError:
            if not missing_ok:
                raise
            continue
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("effect_directory_redirected")
        if index == 0:
            found = info
    return found


def _read(path: Path) -> bytes:
    _directory(path.parent)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or not 0 < before.st_size <= MAX_JSON_BYTES):
            raise ValueError("effect_file_invalid")
        chunks, size = [], 0
        while size <= MAX_JSON_BYTES:
            chunk = os.read(fd, MAX_JSON_BYTES + 1 - size)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(fd)
        if (size != before.st_size or _identity(before) != _identity(after)
                or _identity(after) != _identity(path.lstat())):
            raise ValueError("effect_file_changed")
        return raw
    finally:
        os.close(fd)


def _sync(path: Path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish(path: Path, raw: bytes):
    _directory(path.parent)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        remaining = memoryview(raw)
        while remaining:
            count = os.write(fd, remaining)
            if type(count) is not int or not 0 < count <= len(remaining):
                raise OSError("effect_write_incomplete")
            remaining = remaining[count:]
        os.fsync(fd)
    finally:
        os.close(fd)
    _sync(path.parent)
    if _read(path) != raw:
        raise OSError("effect_publication_readback_failed")


def _ref_fields(ref: AttemptRef, idea_dir: Path):
    if not isinstance(ref, AttemptRef):
        raise ValueError("effect_reference_invalid")
    fields = {"task_id": ref.task_id, "phase": ref.phase, "attempt_id": ref.attempt_id}
    if any(type(value) is not str or not _TOKEN.fullmatch(value)
           or value in (".", "..") for value in fields.values()):
        raise ValueError("effect_reference_token_invalid")
    if (ref.task_id != idea_dir.name or type(ref.generation) is not int
            or not 0 < ref.generation <= 2**63 - 1):
        raise ValueError("effect_reference_scope_invalid")
    return {"schema_version": 1, **fields, "generation": ref.generation}


def _validate_identity(payload, idea_dir, attempt_id):
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("effect_schema_invalid")
    # Validate types before constructing the core ref so all malformed storage
    # is reported as this filesystem contract's dedicated uncertainty.
    for key in ("task_id", "phase", "attempt_id"):
        if type(payload.get(key)) is not str or not _TOKEN.fullmatch(payload[key]):
            raise ValueError("effect_stored_identity_invalid")
    generation = payload.get("generation")
    if type(generation) is not int or not 0 < generation <= 2**63 - 1:
        raise ValueError("effect_stored_generation_invalid")
    ref = AttemptRef(payload["task_id"], payload["phase"], payload["attempt_id"], generation)
    identity = _ref_fields(ref, idea_dir)
    if ref.attempt_id != attempt_id:
        raise ValueError("effect_stored_attempt_mismatch")
    return identity


def _scan(idea_dir: Path, *, pending: tuple[AttemptRef, str] | None = None):
    root = idea_dir / _TREE
    before = _directory(root, missing_ok=True)
    if before is None:
        if pending is not None:
            raise ValueError("effect_prepared_missing")
        return {}
    attempts = list(itertools.islice(root.iterdir(), MAX_EFFECTS + 1))
    if not attempts or len(attempts) > MAX_EFFECTS:
        raise ValueError("effect_history_limit")
    result = {}
    for attempt in attempts:
        initial = _directory(attempt)
        if not _TOKEN.fullmatch(attempt.name) or attempt.name in (".", ".."):
            raise ValueError("effect_attempt_directory_invalid")
        names = {entry.name for entry in itertools.islice(attempt.iterdir(), 3)}
        permitted_pending = pending is not None and attempt.name == pending[0].attempt_id
        if names != {"prepared.json", "committed.json"} and not (
                permitted_pending and names == {"prepared.json"}):
            raise ValueError("effect_unconfirmed_or_extra_files")
        raw = _read(attempt / "prepared.json")
        prepared = _decode(raw)
        identity = _validate_identity(prepared, idea_dir, attempt.name)
        if (set(prepared) != _IDENTITY | {"event", "plan"}
                or prepared["event"] != "effect_prepared" or type(prepared["plan"]) is not dict):
            raise ValueError("effect_prepared_invalid")
        digest = hashlib.sha256(raw).hexdigest()
        if permitted_pending and (identity != _ref_fields(pending[0], idea_dir)
                                  or digest != pending[1]):
            raise ValueError("effect_confirmation_reference_mismatch")
        committed = "committed.json" in names
        if committed:
            actual = _decode(_read(attempt / "committed.json"))
            _validate_identity(actual, idea_dir, attempt.name)
            expected = {**identity, "event": "effect_committed", "prepared_sha256": digest}
            if actual != expected:
                raise ValueError("effect_confirmation_invalid")
        if _identity(initial) != _identity(_directory(attempt)):
            raise ValueError("effect_directory_changed")
        result[attempt.name] = (digest, committed)
    if _identity(before) != _identity(_directory(root)):
        raise ValueError("effect_history_changed")
    if pending is not None and pending[0].attempt_id not in result:
        raise ValueError("effect_prepared_missing")
    return result


def require_closed_effects(idea_dir: Path) -> None:
    """Read-only bounded gate. An entirely absent legacy tree is compatible."""
    try:
        _scan(Path(idea_dir).absolute())
    except _ERRORS as exc:
        raise AttemptEffectInDoubt("attempt_effect_history_unconfirmed") from exc


def prepare_effect(lease: AttemptEffectLease, ref: AttemptRef, plan: dict) -> str:
    """Create one durable intent after the caller's current-token SQL check."""
    try:
        if not isinstance(lease, AttemptEffectLease):
            raise ValueError("effect_lease_invalid")
        idea_dir = lease.idea_dir
        require_effect_lease(lease, idea_dir)
        identity = _ref_fields(ref, idea_dir)
        if type(plan) is not dict:
            raise ValueError("effect_plan_invalid")
        raw = _encoded({**identity, "event": "effect_prepared", "plan": plan})
        history = _scan(idea_dir)
        if ref.attempt_id in history or len(history) >= MAX_EFFECTS:
            raise ValueError("effect_prepare_replay_or_limit")
        root = idea_dir / _TREE
        root.mkdir(exist_ok=True)
        _sync(idea_dir)
        attempt = root / ref.attempt_id
        attempt.mkdir(exist_ok=False)
        _sync(root)
        _publish(attempt / "prepared.json", raw)
        require_effect_lease(lease, idea_dir)
        return hashlib.sha256(raw).hexdigest()
    except _ERRORS as exc:
        raise AttemptEffectInDoubt("attempt_effect_prepare_unconfirmed") from exc


def confirm_effect(lease: AttemptEffectLease, ref: AttemptRef, prepared_sha256: str) -> None:
    """Confirm the exact intent, only after the caller verified SQL commit."""
    try:
        if not isinstance(lease, AttemptEffectLease):
            raise ValueError("effect_lease_invalid")
        idea_dir = lease.idea_dir
        require_effect_lease(lease, idea_dir)
        identity = _ref_fields(ref, idea_dir)
        if type(prepared_sha256) is not str or not _SHA.fullmatch(prepared_sha256):
            raise ValueError("effect_confirmation_hash_invalid")
        history = _scan(idea_dir, pending=(ref, prepared_sha256))
        if not history[ref.attempt_id][1]:
            raw = _encoded({**identity, "event": "effect_committed",
                            "prepared_sha256": prepared_sha256})
            _publish(idea_dir / _TREE / ref.attempt_id / "committed.json", raw)
        _scan(idea_dir, pending=(ref, prepared_sha256))
        require_effect_lease(lease, idea_dir)
    except _ERRORS as exc:
        raise AttemptEffectInDoubt("attempt_effect_confirmation_unconfirmed") from exc
