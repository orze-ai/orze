"""Bounded, content-free train/evaluation manifest separation audit."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import sqlite3
import stat
import tempfile
import time
from pathlib import Path
from typing import Mapping

from orze.core.fs import _fs_lock, _fs_unlock, atomic_write


_FIELDS = ("sample", "speaker", "source")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEADER_KEYS = {
    "schema_version",
    "role",
    "fingerprint_algorithm",
    "fingerprint_namespace_sha256",
    "normalization_contract_sha256",
    "fields",
}


class DataSeparationError(RuntimeError):
    """A stable, content-free data-separation rejection."""


def _is_nonnegative_int(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def validate_data_separation_config(cfg: Mapping) -> list[str]:
    """Return configuration errors for the optional separation contract."""
    spec = cfg.get("data_separation", {})
    prefix = "data_separation"
    if not isinstance(spec, Mapping):
        return [f"{prefix}: must be a mapping"]
    errors = []
    enabled = spec.get("enabled", False)
    if not isinstance(enabled, bool):
        errors.append(f"{prefix}.enabled: must be true or false")

    fields = spec.get("fields", ["sample"])
    if (not isinstance(fields, list) or not fields
            or any(field not in _FIELDS for field in fields)
            or len(fields) != len(set(fields))):
        errors.append(
            f"{prefix}.fields: must be a unique non-empty subset of "
            "sample, speaker, source"
        )
        fields = []
    elif "sample" not in fields:
        errors.append(f"{prefix}.fields: sample is required")

    overlap = spec.get("max_overlap", {"sample": 0})
    if (not isinstance(overlap, Mapping)
            or set(overlap) != set(fields)
            or any(not _is_nonnegative_int(value)
                   for value in overlap.values())):
        errors.append(
            f"{prefix}.max_overlap: must give a non-negative integer for "
            "every configured field and no others"
        )

    for key in ("max_records", "max_bytes", "max_line_bytes"):
        value = spec.get(key, {
            "max_records": 10_000_000,
            "max_bytes": 2 * 1024 * 1024 * 1024,
            "max_line_bytes": 4096,
        }[key])
        if not _is_nonnegative_int(value) or value == 0:
            errors.append(f"{prefix}.{key}: must be a positive integer")

    if enabled:
        for role in ("train", "evaluation"):
            path = spec.get(f"{role}_manifest")
            if (not isinstance(path, str) or not path
                    or not Path(path).is_absolute()):
                errors.append(
                    f"{prefix}.{role}_manifest: must be an absolute path"
                )
            elif ":" in path or any(ord(char) < 32 for char in path):
                errors.append(
                    f"{prefix}.{role}_manifest: path contains unsafe characters"
                )
            digest = spec.get(f"{role}_manifest_sha256")
            if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
                errors.append(
                    f"{prefix}.{role}_manifest_sha256: expected lowercase SHA-256"
                )
        for key in (
            "fingerprint_namespace_sha256",
            "normalization_contract_sha256",
        ):
            value = spec.get(key)
            if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
                errors.append(f"{prefix}.{key}: expected lowercase SHA-256")
    return errors


def _safe_manifest(path_text: str, max_bytes: int) -> tuple[Path, os.stat_result]:
    path = Path(path_text)
    if (not path.is_absolute()
            or os.path.abspath(path_text) != os.path.realpath(path_text)):
        raise DataSeparationError("data_separation_manifest_redirected")
    try:
        info = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise DataSeparationError(
            "data_separation_manifest_unavailable") from exc
    if not stat.S_ISREG(info.st_mode):
        raise DataSeparationError("data_separation_manifest_not_regular")
    if info.st_size <= 0 or info.st_size > max_bytes:
        raise DataSeparationError("data_separation_manifest_size_invalid")
    return path, info


def _metadata_signature(items: list[tuple[Path, os.stat_result]]) -> str:
    payload = [
        [str(path), info.st_dev, info.st_ino, info.st_mode, info.st_size,
         info.st_mtime_ns, info.st_ctime_ns]
        for path, info in items
    ]
    return hashlib.sha256(json.dumps(
        payload, separators=(",", ":"), sort_keys=False,
    ).encode("utf-8")).hexdigest()


def _policy(spec: Mapping) -> dict:
    fields = list(spec.get("fields", ["sample"]))
    return {
        "fields": fields,
        "max_overlap": {
            field: int(spec.get("max_overlap", {"sample": 0})[field])
            for field in fields
        },
        "train_manifest_sha256": spec["train_manifest_sha256"],
        "evaluation_manifest_sha256": spec["evaluation_manifest_sha256"],
        "fingerprint_namespace_sha256": spec[
            "fingerprint_namespace_sha256"],
        "normalization_contract_sha256": spec[
            "normalization_contract_sha256"],
        "max_records": int(spec.get("max_records", 10_000_000)),
        "max_bytes": int(spec.get("max_bytes", 2 * 1024 * 1024 * 1024)),
        "max_line_bytes": int(spec.get("max_line_bytes", 4096)),
    }


def _policy_hash(policy: Mapping) -> str:
    return hashlib.sha256(json.dumps(
        policy, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _state_path(cfg: Mapping) -> Path:
    root = Path(cfg.get("_orze_dir") or (
        Path(cfg.get("_project_root", ".")) / ".orze"))
    current = root
    while current != current.parent:
        if current.is_symlink():
            raise DataSeparationError("data_separation_state_redirected")
        current = current.parent
    state = root / "state"
    state.mkdir(parents=True, exist_ok=True)
    if state.is_symlink():
        raise DataSeparationError("data_separation_state_redirected")
    return state / "data_separation.json"


def _payload_hash(payload: Mapping) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def data_separation_receipt_sha256(payload: Mapping) -> str:
    """Return the canonical digest used to bind a verified receipt onward."""
    if not isinstance(payload, Mapping) or payload.get("status") != "passed":
        raise DataSeparationError("data_separation_receipt_not_passed")
    return _payload_hash(payload)


def _receipt_payload_valid(payload, policy: Mapping, policy_hash: str,
                           metadata_signature: str) -> bool:
    expected_keys = {
        "schema_version", "status", "verified_at", "policy_sha256",
        "metadata_signature", "train_manifest_sha256",
        "evaluation_manifest_sha256", "fingerprint_namespace_sha256",
        "normalization_contract_sha256", "fields", "records",
        "unique_fingerprints", "overlap", "max_overlap",
        "fingerprints_persisted", "rank_claim_proven",
    }
    if (not isinstance(payload, dict) or set(payload) != expected_keys
            or payload.get("schema_version") != 1
            or payload.get("status") != "passed"
            or not isinstance(payload.get("verified_at"), str)
            or payload.get("policy_sha256") != policy_hash
            or payload.get("metadata_signature") != metadata_signature
            or payload.get("train_manifest_sha256")
            != policy["train_manifest_sha256"]
            or payload.get("evaluation_manifest_sha256")
            != policy["evaluation_manifest_sha256"]
            or payload.get("fingerprint_namespace_sha256")
            != policy["fingerprint_namespace_sha256"]
            or payload.get("normalization_contract_sha256")
            != policy["normalization_contract_sha256"]
            or payload.get("fields") != policy["fields"]
            or payload.get("max_overlap") != policy["max_overlap"]
            or payload.get("fingerprints_persisted") is not False
            or payload.get("rank_claim_proven") is not False):
        return False
    records = payload.get("records")
    unique = payload.get("unique_fingerprints")
    overlap = payload.get("overlap")
    if (not isinstance(records, dict) or set(records) != {"train", "evaluation"}
            or any(not _is_nonnegative_int(value) or value == 0
                   for value in records.values())
            or not isinstance(unique, dict)
            or set(unique) != {"train", "evaluation"}
            or not isinstance(overlap, dict)
            or set(overlap) != set(policy["fields"])):
        return False
    for role in ("train", "evaluation"):
        values = unique.get(role)
        if (not isinstance(values, dict)
                or set(values) != set(policy["fields"])
                or any(not _is_nonnegative_int(value)
                       or value > records[role] for value in values.values())
                or values.get("sample") != records[role]):
            return False
    return all(
        _is_nonnegative_int(overlap[field])
        and overlap[field] <= policy["max_overlap"][field]
        and overlap[field] <= unique["train"][field]
        and overlap[field] <= unique["evaluation"][field]
        for field in policy["fields"]
    )


def _cached_receipt(path: Path, policy: Mapping, policy_hash: str,
                    metadata_signature: str) -> dict | None:
    try:
        if (path.is_symlink() or not path.is_file()
                or path.stat().st_size > 1024 * 1024):
            return None
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(envelope, dict) or set(envelope) != {
            "payload", "payload_sha256"}:
        return None
    payload = envelope.get("payload")
    if (not isinstance(payload, dict)
            or envelope.get("payload_sha256") != _payload_hash(payload)
            or not _receipt_payload_valid(
                payload, policy, policy_hash, metadata_signature)):
        return None
    return payload


def _read_manifest(handle, *, role: str, policy: Mapping, connection,
                   side: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    max_line = policy["max_line_bytes"]
    fields = policy["fields"]
    expected_header = {
        "schema_version": 1,
        "role": role,
        "fingerprint_algorithm": "hmac-sha256",
        "fingerprint_namespace_sha256": policy[
            "fingerprint_namespace_sha256"],
        "normalization_contract_sha256": policy[
            "normalization_contract_sha256"],
        "fields": fields,
    }
    first = handle.readline(max_line + 1)
    if not first or len(first) > max_line:
        raise DataSeparationError("data_separation_manifest_header_invalid")
    digest.update(first)
    try:
        header = json.loads(first)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataSeparationError(
            "data_separation_manifest_header_invalid") from exc
    if (not isinstance(header, dict) or set(header) != _HEADER_KEYS
            or header != expected_header):
        raise DataSeparationError("data_separation_manifest_header_invalid")

    count = 0
    batch = []
    expected_keys = set(fields)
    while True:
        raw = handle.readline(max_line + 1)
        if not raw:
            break
        if len(raw) > max_line:
            raise DataSeparationError("data_separation_manifest_line_too_large")
        digest.update(raw)
        try:
            record = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DataSeparationError(
                "data_separation_manifest_record_invalid") from exc
        if (not isinstance(record, dict) or set(record) != expected_keys
                or any(not isinstance(record[field], str)
                       or _HEX64.fullmatch(record[field]) is None
                       for field in fields)):
            raise DataSeparationError(
                "data_separation_manifest_record_invalid")
        count += 1
        if count > policy["max_records"]:
            raise DataSeparationError("data_separation_manifest_record_limit")
        batch.extend(
            (side, index, bytes.fromhex(record[field]))
            for index, field in enumerate(fields)
        )
        if len(batch) >= 10_000:
            connection.executemany(
                "INSERT OR IGNORE INTO fingerprints VALUES (?, ?, ?)", batch)
            batch.clear()
    if batch:
        connection.executemany(
            "INSERT OR IGNORE INTO fingerprints VALUES (?, ?, ?)", batch)
    if count == 0:
        raise DataSeparationError("data_separation_manifest_empty")
    return count, digest.hexdigest()


def ensure_data_separation(cfg: Mapping, *, _lock_held: bool = False) -> dict:
    """Return a verified receipt or raise before GPU telemetry.

    Only keyed fingerprints enter the temporary SQLite index. The durable
    receipt contains aggregate counts and hashes, never fingerprint values.
    """
    spec = cfg.get("data_separation", {})
    if not isinstance(spec, Mapping):
        raise DataSeparationError("data_separation_policy_not_mapping")
    if not spec.get("enabled", False):
        return {"status": "disabled"}
    errors = validate_data_separation_config(cfg)
    if errors:
        raise DataSeparationError("data_separation_policy_invalid")
    policy = _policy(spec)
    policy_sha256 = _policy_hash(policy)
    train = _safe_manifest(spec["train_manifest"], policy["max_bytes"])
    evaluation = _safe_manifest(
        spec["evaluation_manifest"], policy["max_bytes"])
    if train[0] == evaluation[0]:
        raise DataSeparationError("data_separation_manifests_not_distinct")
    metadata = _metadata_signature([train, evaluation])
    receipt_path = _state_path(cfg)
    cached = _cached_receipt(receipt_path, policy, policy_sha256, metadata)
    if cached is not None:
        return cached
    if not _lock_held:
        lock_dir = receipt_path.parent / "data_separation.lock"
        if lock_dir.is_symlink():
            raise DataSeparationError("data_separation_state_redirected")
        deadline = time.monotonic() + 600
        while not _fs_lock(lock_dir, stale_seconds=600):
            if time.monotonic() >= deadline:
                raise DataSeparationError("data_separation_audit_locked")
            time.sleep(0.1)
        try:
            # Recheck every input and the cache under the project-wide lock.
            # Concurrent nodes then perform one audit rather than one per GPU.
            return ensure_data_separation(cfg, _lock_held=True)
        finally:
            _fs_unlock(lock_dir)

    state_dir = receipt_path.parent
    with tempfile.TemporaryDirectory(
            prefix="data-separation-", dir=state_dir) as temp_dir:
        db_path = Path(temp_dir) / "fingerprints.sqlite3"
        connection = sqlite3.connect(str(db_path))
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                "CREATE TABLE fingerprints ("
                "side INTEGER NOT NULL, dimension INTEGER NOT NULL, "
                "value BLOB NOT NULL, PRIMARY KEY(side, dimension, value)"
                ") WITHOUT ROWID"
            )
            # Keep the disk-backed index bounded without leaving even keyed
            # fingerprints after a crash. Linux retains the open inode for
            # SQLite while removing its directory entry immediately.
            try:
                os.unlink(db_path)
            except OSError as exc:
                raise DataSeparationError(
                    "data_separation_ephemeral_index_unavailable") from exc
            with train[0].open("rb") as handle:
                train_count, train_digest = _read_manifest(
                    handle, role="train", policy=policy,
                    connection=connection, side=0)
            with evaluation[0].open("rb") as handle:
                eval_count, eval_digest = _read_manifest(
                    handle, role="evaluation", policy=policy,
                    connection=connection, side=1)
            if (train_digest != policy["train_manifest_sha256"]
                    or eval_digest != policy["evaluation_manifest_sha256"]):
                raise DataSeparationError(
                    "data_separation_manifest_digest_mismatch")
            current = [
                _safe_manifest(str(train[0]), policy["max_bytes"]),
                _safe_manifest(str(evaluation[0]), policy["max_bytes"]),
            ]
            if _metadata_signature(current) != metadata:
                raise DataSeparationError(
                    "data_separation_manifest_changed_during_audit")

            unique = {"train": {}, "evaluation": {}}
            overlap = {}
            for index, field in enumerate(policy["fields"]):
                for side, label in ((0, "train"), (1, "evaluation")):
                    unique[label][field] = connection.execute(
                        "SELECT COUNT(*) FROM fingerprints "
                        "WHERE side=? AND dimension=?", (side, index),
                    ).fetchone()[0]
                overlap[field] = connection.execute(
                    "SELECT COUNT(*) FROM fingerprints AS train "
                    "JOIN fingerprints AS evaluation "
                    "ON train.dimension=evaluation.dimension "
                    "AND train.value=evaluation.value "
                    "WHERE train.side=0 AND evaluation.side=1 "
                    "AND train.dimension=?", (index,),
                ).fetchone()[0]
            if (unique["train"]["sample"] != train_count
                    or unique["evaluation"]["sample"] != eval_count):
                raise DataSeparationError(
                    "data_separation_duplicate_sample_fingerprint")
            exceeded = [
                field for field in policy["fields"]
                if overlap[field] > policy["max_overlap"][field]
            ]
            if exceeded:
                raise DataSeparationError(
                    "data_separation_overlap_exceeded:" + exceeded[0])
        finally:
            connection.close()

    payload = {
        "schema_version": 1,
        "status": "passed",
        "verified_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "policy_sha256": policy_sha256,
        "metadata_signature": metadata,
        "train_manifest_sha256": train_digest,
        "evaluation_manifest_sha256": eval_digest,
        "fingerprint_namespace_sha256": policy[
            "fingerprint_namespace_sha256"],
        "normalization_contract_sha256": policy[
            "normalization_contract_sha256"],
        "fields": policy["fields"],
        "records": {"train": train_count, "evaluation": eval_count},
        "unique_fingerprints": unique,
        "overlap": overlap,
        "max_overlap": policy["max_overlap"],
        "fingerprints_persisted": False,
        "rank_claim_proven": False,
    }
    envelope = {"payload": payload, "payload_sha256": _payload_hash(payload)}
    if receipt_path.is_symlink():
        raise DataSeparationError("data_separation_state_redirected")
    atomic_write(receipt_path, json.dumps(
        envelope, sort_keys=True, indent=2) + "\n")
    return payload
