"""Fail-closed admission for byte-for-byte equivalent training launches.

The completed-config cache is useful for idea generation, but it cannot stop
two equivalent queued ideas from reaching different GPUs concurrently.  This
module provides the final, project-scoped admission boundary.  It deliberately
stores only a SHA-256 identity plus ownership metadata; input material and
environment values are never serialized to the registry.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
import socket
import time
import secrets
from pathlib import Path
from typing import Optional

import yaml

from orze.core.fs import _fs_lock, _fs_unlock, atomic_write
from orze.core.integrity import canonical_config_for_execution


class DuplicateExecutionError(RuntimeError):
    """Raised when an equivalent execution is active or already completed."""


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise DuplicateExecutionError("execution_identity_input_unreadable") from exc


def _canonical_yaml(path: Path):
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise DuplicateExecutionError("execution_identity_config_unreadable") from exc
    if not isinstance(value, dict):
        raise DuplicateExecutionError("execution_identity_config_not_mapping")
    return value


def compute_execution_identity(
    *,
    config_path: Path,
    base_config_path: Path,
    train_script: Path,
    python: str,
    train_extra_args: list,
    train_extra_env: dict,
    data_boundaries: dict,
    data_separation: Optional[dict] = None,
) -> str:
    """Hash semantic launch inputs without returning or persisting them.

    Idea IDs, result paths, GPU IDs, and attempt IDs are intentionally absent:
    changing any of those does not create independent model-quality evidence.
    A seed in the training config is included naturally and therefore denotes a
    distinct execution.
    """
    if not isinstance(train_extra_env, dict):
        raise DuplicateExecutionError("execution_identity_env_not_mapping")
    payload = {
        "schema_version": 1,
        "config": canonical_config_for_execution(
            _canonical_yaml(Path(config_path))
        ),
        "base_config_sha256": _sha256_file(Path(base_config_path)),
        "train_script_sha256": _sha256_file(Path(train_script)),
        "python": str(python),
        "train_extra_args": [str(value) for value in train_extra_args],
        # Values affect execution identity but are present only in this
        # in-memory preimage.  The registry receives the outer digest alone.
        "train_extra_env": {
            str(key): str(value)
            for key, value in sorted(train_extra_env.items(), key=lambda item: str(item[0]))
        },
        "data_boundaries": data_boundaries,
    }
    if (isinstance(data_separation, dict)
            and data_separation.get("enabled") is True):
        # Enabling a new data contract intentionally creates a distinct
        # execution identity; disabled projects retain their prior identity.
        payload["schema_version"] = 2
        payload["data_separation"] = data_separation
    try:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DuplicateExecutionError("execution_identity_not_canonical") from exc
    return hashlib.sha256(encoded).hexdigest()


def _registry_root(results_dir: Path, cfg: dict) -> Path:
    if cfg.get("_orze_dir"):
        root = Path(cfg["_orze_dir"]) / "state" / "execution_identities"
    else:
        root = Path(results_dir) / "_execution_identities"
    # A redirected registry could split or overwrite admission state.  Treat
    # an existing symlink at any registry level as an integrity failure.
    current = root
    existing = []
    while True:
        if current.exists() or current.is_symlink():
            existing.append(current)
        if current == current.parent:
            break
        current = current.parent
    for path in existing:
        if path.is_symlink():
            raise DuplicateExecutionError("execution_identity_registry_symlink")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _read_terminal(results_dir: Path, owner: dict) -> Optional[dict]:
    idea_id = owner.get("idea_id")
    attempt_id = owner.get("attempt_id")
    if not isinstance(idea_id, str) or not isinstance(attempt_id, str):
        raise DuplicateExecutionError("execution_identity_owner_invalid")
    path = (Path(results_dir) / idea_id / "_compute_receipts" /
            attempt_id / "terminal.json")
    if not path.exists():
        return None
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DuplicateExecutionError("execution_identity_terminal_invalid") from exc
    if (not isinstance(receipt, dict)
            or receipt.get("idea_id") != idea_id
            or receipt.get("attempt_id") != attempt_id
            or receipt.get("event") != "terminal"):
        raise DuplicateExecutionError("execution_identity_terminal_invalid")
    return receipt


def reserve_execution_identity(
    results_dir: Path,
    cfg: dict,
    identity: str,
    idea_id: str,
    attempt_id: str,
) -> None:
    """Atomically reserve an execution identity before GPU inspection.

    Non-completed terminal attempts may be replaced, which preserves normal
    repair/resume behavior.  Active, unresolved, or completed ownership blocks
    an equivalent launch.  Corrupt registry evidence also blocks fail-closed.
    """
    if (len(identity) != 64
            or any(char not in "0123456789abcdef" for char in identity)):
        raise DuplicateExecutionError("execution_identity_digest_invalid")
    root = _registry_root(Path(results_dir), cfg)
    gate = root / f"{identity}.lock"
    if not _fs_lock(gate, stale_seconds=300):
        raise DuplicateExecutionError("execution_identity_admission_busy")
    record_path = root / f"{identity}.json"
    try:
        owner = None
        if record_path.exists() or record_path.is_symlink():
            if record_path.is_symlink():
                raise DuplicateExecutionError("execution_identity_record_symlink")
            try:
                owner = json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DuplicateExecutionError("execution_identity_owner_invalid") from exc
            if not isinstance(owner, dict):
                raise DuplicateExecutionError("execution_identity_owner_invalid")

        if owner:
            same_attempt = (
                owner.get("idea_id") == idea_id
                and owner.get("attempt_id") == attempt_id
            )
            if same_attempt:
                return
            terminal = _read_terminal(Path(results_dir), owner)
            if terminal is None:
                raise DuplicateExecutionError(
                    f"exact_execution_already_reserved:{owner.get('idea_id', 'unknown')}"
                )
            if terminal.get("outcome") == "completed":
                raise DuplicateExecutionError(
                    f"exact_execution_already_completed:{owner.get('idea_id', 'unknown')}"
                )

        record = {
            "schema_version": 1,
            "execution_identity": identity,
            "idea_id": idea_id,
            "attempt_id": attempt_id,
            "reserved_at_epoch": round(time.time(), 6),
            "reserved_by_host": socket.gethostname(),
            "reserved_by_pid": os.getpid(),
        }
        atomic_write(
            record_path,
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        )
    finally:
        _fs_unlock(gate)


def release_execution_identity(
    results_dir: Path,
    cfg: dict,
    identity: str,
    idea_id: str,
    attempt_id: str,
) -> None:
    """Release only this exact pre-allocation reservation.

    Once a process starts, callers retain the record and terminal compute
    receipts determine whether a later retry is admissible.
    """
    root = _registry_root(Path(results_dir), cfg)
    gate = root / f"{identity}.lock"
    if not _fs_lock(gate, stale_seconds=300):
        raise DuplicateExecutionError("execution_identity_release_busy")
    record_path = root / f"{identity}.json"
    try:
        if not record_path.exists() or record_path.is_symlink():
            return
        try:
            owner = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return
        if (isinstance(owner, dict)
                and owner.get("idea_id") == idea_id
                and owner.get("attempt_id") == attempt_id):
            record_path.unlink()
    finally:
        _fs_unlock(gate)


@dataclass(frozen=True)
class ReplicaReservation:
    """Captured occurrence slot; not authorization to create a process."""
    path: Path
    record_sha256: str
    identity: str
    idea_id: str
    attempt_id: str
    request_id: str


def _replica_bytes(value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False) + "\n").encode()
    if len(raw) > 16384:
        raise ValueError("replication_slot_metadata_limit")
    return raw


def _replica_read(path):
    # Reuse the bounded stable, single-link receipt reader; this neither
    # creates directories nor regards a routing/slot record as authority.
    from orze.engine.training_attempts import _read
    value, digest = _read(path, 16384)
    if hashlib.sha256(_replica_bytes(value)).hexdigest() != digest:
        raise ValueError("replication_slot_noncanonical")
    return value, digest


def _replica_sync(folder):
    fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def _replica_gate(path):
    """No age takeover, and an uncertain publication retains its owner."""
    from orze.core.idea_source_lock import (
        SourceLockInDoubt, idea_source_lock, idea_source_lock_owned,
    )
    from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
    body_error = None
    try:
        with idea_source_lock(path.with_suffix(".lock")) as lease:
            if lease is None:
                raise AttemptEffectBusy("replication_slot_busy")
            try:
                yield lambda: idea_source_lock_owned(lease)
            except AttemptEffectInDoubt as exc:
                raise SourceLockInDoubt("replication_slot_unconfirmed") from exc
            except BaseException as exc:
                body_error = exc
        if body_error is not None:
            raise body_error
    except OSError as exc:
        raise AttemptEffectInDoubt("replication_slot_storage_unconfirmed") from exc


def _replace_replica(path, record, owned, *, replacing):
    from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
    raw = _replica_bytes(record)
    temporary = path.with_name(path.name + "." + secrets.token_hex(16) + ".tmp")
    try:
        if not owned():
            raise OSError("replication_slot_owner_changed")
        target = temporary if replacing else path
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            remaining = memoryview(raw)
            while remaining:
                n = os.write(fd, remaining)
                if type(n) is not int or not 0 < n <= len(remaining):
                    raise OSError("replication_slot_short_write")
                remaining = remaining[n:]
            os.fsync(fd)
        finally:
            # Close is attempted once; an uncertain close cannot cause a
            # second close of an fd that another thread may have reused.
            closing, fd = fd, None
            os.close(closing)
        if not owned():
            raise OSError("replication_slot_owner_changed")
        if replacing:
            os.replace(temporary, path)
        _replica_sync(path.parent)
        actual, digest = _replica_read(path)
        if actual != record or digest != hashlib.sha256(raw).hexdigest() or not owned():
            raise OSError("replication_slot_readback_failed")
        return digest
    except BaseException as exc:
        # Partial bytes/rename/fsync are never interpreted as a refundable
        # reservation. Leave both the selected record and gate for recovery.
        raise AttemptEffectInDoubt("replication_slot_publication_unconfirmed") from exc


def _require_closed_replica(lake, results_dir, owner, authorization):
    from orze.core.execution_attempts import AttemptRef, current_attempt
    from orze.engine.completion_events import CompletionEvent, require_completion
    from orze.engine.execution_authority import canonical_identity_equal
    from orze.engine.attempt_effect_lock import AttemptEffectBusy
    row = current_attempt(lake.conn, owner["idea_id"], "training")
    if (row is None or row["attempt_id"] != owner["attempt_id"]
            or row["state"] not in ("TERMINAL", "NOT_STARTED")
            or not canonical_identity_equal(row["binding"].get("replication"), authorization)):
        raise AttemptEffectBusy("replication_slot_previous_attempt_unclosed")
    ref = AttemptRef(row["task_id"], row["phase"], row["attempt_id"], row["generation"])
    closed = require_completion(CompletionEvent(ref.task_id, 0, ref), lake, results_dir,
                                phase="training")
    if closed["terminal"].get("outcome") not in ("failed", "not_started"):
        raise AttemptEffectBusy("replication_slot_previous_attempt_not_retryable")


def reserve_replica_execution_identity(results_dir, cfg, identity, idea_id, attempt_id,
                                       *, lake, authorization):
    """Reserve a DB-authorized occurrence without changing semantic identity.

    The flat default owner is never read, replaced or deleted. Existing
    occurrence slots need a confirmed failed/not-started *native* attempt;
    missing intent, open work, completed work and uncertain storage all hold.
    """
    from orze.engine.replication import replication_authorization
    from orze.engine.execution_authority import canonical_identity_equal
    from orze.engine.attempt_effect_lock import AttemptEffectBusy
    actual = replication_authorization(lake, idea_id, Path(results_dir) / idea_id,
                                       cfg, identity, claim_id=attempt_id)
    if actual is None or not canonical_identity_equal(actual, authorization):
        raise AttemptEffectBusy("replication_slot_authorization_changed")
    request_id = actual["request_id"]
    from orze.core.replication_requests import token
    token(request_id)
    root = _registry_root(Path(results_dir), cfg).absolute()
    folder = root / (identity + ".replicas")
    if folder.is_symlink():
        raise AttemptEffectBusy("replication_slot_directory_redirected")
    folder.mkdir(exist_ok=True)
    _replica_sync(root)
    path = folder / (request_id + ".json")
    auth_sha = hashlib.sha256(_replica_bytes(actual)).hexdigest()
    with _replica_gate(path) as owned:
        old = None
        if path.exists() or path.is_symlink():
            old, _ = _replica_read(path)
            if (old.get("schema_version") != 2 or type(old.get("schema_version")) is not int
                    or old.get("execution_identity") != identity or old.get("idea_id") != idea_id
                    or old.get("request_id") != request_id or old.get("authorization_sha256") != auth_sha):
                raise AttemptEffectBusy("replication_slot_owner_mismatch")
            _require_closed_replica(lake, results_dir, old, actual)
        record = {"schema_version": 2, "execution_identity": identity,
                  "idea_id": idea_id, "attempt_id": attempt_id,
                  "request_id": request_id, "authorization_sha256": auth_sha}
        digest = _replace_replica(path, record, owned, replacing=old is not None)
    return ReplicaReservation(path, digest, identity, idea_id, attempt_id, request_id)


def release_replica_execution_identity(reservation):
    """Release only the exact captured pre-Popen slot, never a cfg-selected path."""
    from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
    if not isinstance(reservation, ReplicaReservation):
        raise AttemptEffectBusy("replication_slot_capture_required")
    path = reservation.path
    with _replica_gate(path) as owned:
        try:
            value, digest = _replica_read(path)
            if (digest != reservation.record_sha256 or value.get("idea_id") != reservation.idea_id
                    or value.get("attempt_id") != reservation.attempt_id
                    or value.get("request_id") != reservation.request_id or not owned()):
                raise AttemptEffectBusy("replication_slot_capture_changed")
            path.unlink()
            _replica_sync(path.parent)
            if path.exists() or path.is_symlink() or not owned():
                raise OSError("replication_slot_release_unconfirmed")
        except AttemptEffectBusy:
            raise
        except BaseException as exc:
            raise AttemptEffectInDoubt("replication_slot_release_unconfirmed") from exc
