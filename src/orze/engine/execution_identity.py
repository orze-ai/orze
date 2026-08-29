"""Fail-closed admission for byte-for-byte equivalent training launches.

The completed-config cache is useful for idea generation, but it cannot stop
two equivalent queued ideas from reaching different GPUs concurrently.  This
module provides the final, project-scoped admission boundary.  It deliberately
stores only a SHA-256 identity plus ownership metadata; input material and
environment values are never serialized to the registry.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import time
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
