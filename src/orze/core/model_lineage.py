"""Managed training-to-evaluation artifact lineage receipts."""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import re
import select
import stat
from pathlib import Path
from typing import Mapping

from orze.core.data_separation import (
    data_separation_receipt_sha256,
    ensure_data_separation,
)
from orze.core.fs import atomic_write


LINEAGE_FILE = "_model_lineage.json"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_TOKEN = re.compile(r"[A-Za-z0-9_.:-]{1,128}")


class ModelLineageError(RuntimeError):
    """A stable, content-free managed-lineage rejection."""


def _positive_int(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def validate_model_lineage_config(cfg: Mapping) -> list[str]:
    spec = cfg.get("model_lineage", {})
    prefix = "model_lineage"
    if not isinstance(spec, Mapping):
        return [f"{prefix}: must be a mapping"]
    errors = []
    enabled = spec.get("enabled", False)
    if not isinstance(enabled, bool):
        errors.append(f"{prefix}.enabled: must be true or false")
    for key, default in (
        ("max_files", 100_000),
        ("max_bytes", 100 * 1024 * 1024 * 1024),
    ):
        if not _positive_int(spec.get(key, default)):
            errors.append(f"{prefix}.{key}: must be a positive integer")
    timeout = spec.get("attestation_timeout", 10)
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout)) or timeout <= 0
            or timeout > 60):
        errors.append(
            f"{prefix}.attestation_timeout: must be finite and in (0, 60]"
        )
    if enabled is True:
        artifact = spec.get("artifact")
        if (not isinstance(artifact, str) or not artifact
                or Path(artifact).is_absolute() or Path(artifact) == Path(".")
                or ".." in Path(artifact).parts):
            errors.append(
                f"{prefix}.artifact: must be a relative path inside each idea"
            )
        boundaries = cfg.get("data_boundaries", {})
        if (not isinstance(boundaries, Mapping)
                or not boundaries.get("forbidden_in_training")
                or boundaries.get("training_network") != "deny"):
            errors.append(
                f"{prefix}: requires non-empty hard forbidden paths and "
                "data_boundaries.training_network: deny"
            )
        separation = cfg.get("data_separation", {})
        if (not isinstance(separation, Mapping)
                or separation.get("enabled") is not True):
            errors.append(f"{prefix}: requires data_separation.enabled: true")
    return errors


def _canonical_hash(value: Mapping) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ModelLineageError("model_lineage_policy_not_canonical") from exc
    return hashlib.sha256(encoded).hexdigest()


def _idea_path(idea_dir: Path, relative_name: str) -> Path:
    idea_dir = Path(idea_dir)
    relative = Path(str(relative_name))
    if (idea_dir.is_symlink() or relative.is_absolute()
            or relative == Path(".") or ".." in relative.parts):
        raise ModelLineageError("model_lineage_path_redirected")
    current = idea_dir
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ModelLineageError("model_lineage_path_redirected")
    return current


def _write_envelope_once(path: Path, payload: dict) -> dict:
    encoded_payload = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    envelope = {
        "payload": payload,
        "payload_sha256": hashlib.sha256(
            encoded_payload.encode("utf-8")).hexdigest(),
    }
    encoded = (json.dumps(
        envelope, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ModelLineageError("model_lineage_receipt_redirected")
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise ModelLineageError("model_lineage_receipt_preexisting") from exc
    except OSError as exc:
        raise ModelLineageError("model_lineage_receipt_write_failed") from exc
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short receipt write")
            view = view[written:]
        os.fsync(fd)
    except OSError as exc:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise ModelLineageError("model_lineage_receipt_write_failed") from exc
    finally:
        os.close(fd)
    return envelope


def _read_envelope(path: Path, expected_keys: set[str]) -> tuple[dict, str]:
    try:
        if (path.is_symlink() or not path.is_file()
                or path.stat().st_size > 1024 * 1024):
            raise OSError("unsafe receipt")
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelLineageError("model_lineage_receipt_invalid") from exc
    if (not isinstance(envelope, dict)
            or set(envelope) != {"payload", "payload_sha256"}
            or not isinstance(envelope.get("payload"), dict)
            or set(envelope["payload"]) != expected_keys):
        raise ModelLineageError("model_lineage_receipt_invalid")
    payload = envelope["payload"]
    actual = _canonical_hash(payload)
    if envelope.get("payload_sha256") != actual:
        raise ModelLineageError("model_lineage_receipt_invalid")
    return payload, actual


def _receipt_dir(idea_dir: Path, attempt_id: str) -> Path:
    if not isinstance(attempt_id, str) or _TOKEN.fullmatch(attempt_id) is None:
        raise ModelLineageError("model_lineage_attempt_id_invalid")
    path = _idea_path(
        idea_dir, str(Path("_compute_receipts") / attempt_id))
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ModelLineageError("model_lineage_path_redirected")
    return path


def prepare_model_lineage_launch(
    *, idea_id: str, attempt_id: str, execution_identity: str,
    idea_dir: Path, cfg: Mapping, separation_receipt: Mapping,
) -> dict | None:
    """Create a private one-shot pipe for post-mount child attestation."""
    spec = cfg.get("model_lineage", {})
    if not isinstance(spec, Mapping) or not spec.get("enabled", False):
        return None
    if validate_model_lineage_config(cfg):
        raise ModelLineageError("model_lineage_policy_invalid")
    if (not isinstance(idea_id, str) or _TOKEN.fullmatch(idea_id) is None
            or not isinstance(execution_identity, str)
            or _HEX64.fullmatch(execution_identity) is None):
        raise ModelLineageError("model_lineage_launch_identity_invalid")
    separation_sha256 = data_separation_receipt_sha256(separation_receipt)
    boundary_sha256 = _canonical_hash(dict(cfg.get("data_boundaries") or {}))
    receipt_path = _receipt_dir(idea_dir, attempt_id) / "boundary.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise ModelLineageError("model_lineage_boundary_receipt_preexisting")
    read_fd, write_fd = os.pipe()
    nonce = os.urandom(32).hex()
    return {
        "read_fd": read_fd,
        "write_fd": write_fd,
        "nonce": nonce,
        "receipt_path": receipt_path,
        "timeout": float(spec.get("attestation_timeout", 10)),
        "idea_id": idea_id,
        "attempt_id": attempt_id,
        "execution_identity_sha256": execution_identity,
        "data_boundary_policy_sha256": boundary_sha256,
        "data_separation_receipt_sha256": separation_sha256,
        "env": {
            "ORZE_BOUNDARY_ATTEST_FD": str(write_fd),
            "ORZE_BOUNDARY_ATTEST_NONCE": nonce,
        },
    }


def close_model_lineage_attestation(context: dict | None) -> None:
    if not context:
        return
    for key in ("read_fd", "write_fd"):
        fd = context.get(key)
        if isinstance(fd, int) and fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
            context[key] = -1


def receive_model_lineage_attestation(
    context: dict | None, *, process_pid: int,
) -> tuple[dict | None, str | None]:
    """Receive the nonce only after the child established kernel isolation."""
    if context is None:
        return None, None
    write_fd = context.get("write_fd", -1)
    if isinstance(write_fd, int) and write_fd >= 0:
        os.close(write_fd)
        context["write_fd"] = -1
    read_fd = context.get("read_fd", -1)
    try:
        ready, _, _ = select.select(
            [read_fd], [], [], float(context["timeout"]))
        if not ready:
            raise ModelLineageError("model_lineage_boundary_attestation_timeout")
        observed = os.read(read_fd, 256)
    except (OSError, ValueError) as exc:
        raise ModelLineageError(
            "model_lineage_boundary_attestation_failed") from exc
    finally:
        if isinstance(read_fd, int) and read_fd >= 0:
            try:
                os.close(read_fd)
            except OSError:
                pass
            context["read_fd"] = -1
    if observed != (context["nonce"] + "\n").encode("ascii"):
        raise ModelLineageError("model_lineage_boundary_attestation_invalid")
    payload = {
        "schema_version": 1,
        "idea_id": context["idea_id"],
        "attempt_id": context["attempt_id"],
        "execution_identity_sha256": context["execution_identity_sha256"],
        "data_boundary_policy_sha256": context[
            "data_boundary_policy_sha256"],
        "data_separation_receipt_sha256": context[
            "data_separation_receipt_sha256"],
        "kernel_path_isolation": True,
        "training_network_denied": True,
        "process_pid": int(process_pid),
        "activated_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
    }
    envelope = _write_envelope_once(context["receipt_path"], payload)
    return payload, envelope["payload_sha256"]


def _stat_identity(info) -> tuple:
    return (
        info.st_dev, info.st_ino, info.st_mode, info.st_size,
        info.st_mtime_ns, info.st_ctime_ns,
    )


def _hash_stable_file(path: Path, before, digest_state) -> None:
    """Hash one exact inode without following a last-moment symlink swap."""
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
        opened = os.fstat(fd)
        if (not stat.S_ISREG(opened.st_mode)
                or _stat_identity(opened) != _stat_identity(before)):
            raise ModelLineageError(
                "model_lineage_artifact_changed_during_hash")
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest_state.update(chunk)
        after_fd = os.fstat(fd)
    except ModelLineageError:
        raise
    except OSError as exc:
        raise ModelLineageError("model_lineage_artifact_unreadable") from exc
    finally:
        if "fd" in locals():
            os.close(fd)
    try:
        after_path = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ModelLineageError(
            "model_lineage_artifact_changed_during_hash") from exc
    if (_stat_identity(after_fd) != _stat_identity(before)
            or _stat_identity(after_path) != _stat_identity(before)):
        raise ModelLineageError(
            "model_lineage_artifact_changed_during_hash")


def _artifact_files(root: Path, max_files: int, max_bytes: int):
    if root.is_symlink():
        raise ModelLineageError("model_lineage_artifact_redirected")
    try:
        root_info = os.stat(root, follow_symlinks=False)
    except OSError as exc:
        raise ModelLineageError("model_lineage_artifact_missing") from exc
    if stat.S_ISREG(root_info.st_mode):
        if root_info.st_size > max_bytes:
            raise ModelLineageError("model_lineage_artifact_limit_exceeded")
        if root_info.st_size == 0:
            raise ModelLineageError("model_lineage_artifact_empty")
        return (
            "file", [(Path(root.name), root, root_info)],
            root_info.st_size, root_info,
        )
    if not stat.S_ISDIR(root_info.st_mode):
        raise ModelLineageError("model_lineage_artifact_type_unsupported")
    files = []
    total = 0
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs.sort()
        names.sort()
        directory_path = Path(directory)
        for name in [*dirs, *names]:
            candidate = directory_path / name
            if candidate.is_symlink():
                raise ModelLineageError("model_lineage_artifact_redirected")
        for name in names:
            path = directory_path / name
            try:
                info = os.stat(path, follow_symlinks=False)
            except OSError as exc:
                raise ModelLineageError(
                    "model_lineage_artifact_unreadable") from exc
            if not stat.S_ISREG(info.st_mode):
                raise ModelLineageError(
                    "model_lineage_artifact_type_unsupported")
            files.append((path.relative_to(root), path, info))
            total += info.st_size
            if len(files) > max_files or total > max_bytes:
                raise ModelLineageError(
                    "model_lineage_artifact_limit_exceeded")
    if not files:
        raise ModelLineageError("model_lineage_artifact_empty")
    return "directory_tree_v1", files, total, root_info


def _artifact_digest_once(root: Path, max_files: int, max_bytes: int) -> dict:
    kind, files, total, root_before = _artifact_files(
        root, max_files, max_bytes)
    if kind == "file":
        digest_state = hashlib.sha256()
        _hash_stable_file(root, root_before, digest_state)
        digest = digest_state.hexdigest()
    else:
        digest_state = hashlib.sha256(b"orze-model-tree-v1\0")
        for relative, path, before in files:
            encoded_name = relative.as_posix().encode("utf-8")
            digest_state.update(len(encoded_name).to_bytes(8, "big"))
            digest_state.update(encoded_name)
            digest_state.update(before.st_size.to_bytes(8, "big"))
            _hash_stable_file(path, before, digest_state)
        digest = digest_state.hexdigest()
        _, after_files, after_total, root_after = _artifact_files(
            root, max_files, max_bytes)
        if ([item[0] for item in files] != [item[0] for item in after_files]
                or total != after_total
                or _stat_identity(root_before) != _stat_identity(root_after)):
            raise ModelLineageError(
                "model_lineage_artifact_changed_during_hash")
    return {
        "artifact_sha256": digest,
        "artifact_kind": kind,
        "artifact_files": len(files),
        "artifact_bytes": total,
    }


def _artifact_digest(root: Path, max_files: int, max_bytes: int) -> dict:
    """Require two identical full reads to rule out an unstable snapshot."""
    first = _artifact_digest_once(root, max_files, max_bytes)
    second = _artifact_digest_once(root, max_files, max_bytes)
    if first != second:
        raise ModelLineageError(
            "model_lineage_artifact_changed_during_hash")
    return first


_BOUNDARY_KEYS = {
    "schema_version", "idea_id", "attempt_id",
    "execution_identity_sha256", "data_boundary_policy_sha256",
    "data_separation_receipt_sha256", "kernel_path_isolation",
    "training_network_denied", "process_pid", "activated_at",
}
_LINEAGE_KEYS = {
    "schema_version", "idea_id", "attempt_id",
    "execution_identity_sha256", "boundary_receipt_sha256",
    "data_separation_receipt_sha256", "artifact_sha256", "artifact_kind",
    "artifact_files", "artifact_bytes", "managed_training",
    "rank_claim_proven", "finalized_at",
}


def finalize_model_lineage(tp, idea_dir: Path, cfg: Mapping) -> dict:
    spec = cfg.get("model_lineage", {})
    if not isinstance(spec, Mapping) or not spec.get("enabled", False):
        return {"status": "disabled"}
    if validate_model_lineage_config(cfg):
        raise ModelLineageError("model_lineage_policy_invalid")
    attempt_id = str(getattr(tp, "attempt_id", ""))
    execution_identity = getattr(tp, "execution_identity", None)
    if (not isinstance(execution_identity, str)
            or _HEX64.fullmatch(execution_identity) is None):
        raise ModelLineageError("model_lineage_execution_identity_missing")
    receipt_dir = _receipt_dir(idea_dir, attempt_id)
    boundary, boundary_sha256 = _read_envelope(
        receipt_dir / "boundary.json", _BOUNDARY_KEYS)
    separation = ensure_data_separation(cfg)
    separation_sha256 = data_separation_receipt_sha256(separation)
    expected_boundary = {
        "schema_version": 1,
        "idea_id": getattr(tp, "idea_id", None),
        "attempt_id": attempt_id,
        "execution_identity_sha256": execution_identity,
        "data_boundary_policy_sha256": _canonical_hash(
            dict(cfg.get("data_boundaries") or {})),
        "data_separation_receipt_sha256": separation_sha256,
        "kernel_path_isolation": True,
        "training_network_denied": True,
    }
    if any(boundary.get(key) != value
           for key, value in expected_boundary.items()):
        raise ModelLineageError("model_lineage_boundary_receipt_mismatch")
    start_path = receipt_dir / "start.json"
    try:
        start = json.loads(start_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelLineageError("model_lineage_compute_start_invalid") from exc
    if (not isinstance(start, dict)
            or start.get("idea_id") != getattr(tp, "idea_id", None)
            or start.get("attempt_id") != attempt_id
            or start.get("phase") != "training"
            or start.get("event") != "start"
            or start.get("outcome") != "started"):
        raise ModelLineageError("model_lineage_compute_start_invalid")
    artifact_path = _idea_path(idea_dir, str(spec["artifact"]))
    artifact = _artifact_digest(
        artifact_path, int(spec.get("max_files", 100_000)),
        int(spec.get("max_bytes", 100 * 1024 * 1024 * 1024)))
    payload = {
        "schema_version": 1,
        "idea_id": getattr(tp, "idea_id", None),
        "attempt_id": attempt_id,
        "execution_identity_sha256": execution_identity,
        "boundary_receipt_sha256": boundary_sha256,
        "data_separation_receipt_sha256": separation_sha256,
        **artifact,
        "managed_training": True,
        "rank_claim_proven": False,
        "finalized_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
    }
    return _write_envelope_once(
        _idea_path(idea_dir, LINEAGE_FILE), payload)["payload"]


def validate_model_lineage_for_evaluation(
    idea_dir: Path, cfg: Mapping,
) -> tuple[dict, str]:
    """Validate current managed artifact and terminal attempt evidence."""
    spec = cfg.get("model_lineage", {})
    if not isinstance(spec, Mapping) or not spec.get("enabled", False):
        raise ModelLineageError("model_lineage_disabled")
    if validate_model_lineage_config(cfg):
        raise ModelLineageError("model_lineage_policy_invalid")
    lineage, lineage_sha256 = _read_envelope(
        _idea_path(idea_dir, LINEAGE_FILE), _LINEAGE_KEYS)
    if (lineage.get("schema_version") != 1
            or lineage.get("idea_id") != Path(idea_dir).name
            or lineage.get("managed_training") is not True
            or lineage.get("rank_claim_proven") is not False
            or not isinstance(lineage.get("attempt_id"), str)
            or _TOKEN.fullmatch(lineage["attempt_id"]) is None
            or not isinstance(lineage.get("execution_identity_sha256"), str)
            or _HEX64.fullmatch(lineage["execution_identity_sha256"]) is None):
        raise ModelLineageError("model_lineage_receipt_invalid")
    receipt_dir = _receipt_dir(idea_dir, lineage["attempt_id"])
    boundary, boundary_sha256 = _read_envelope(
        receipt_dir / "boundary.json", _BOUNDARY_KEYS)
    if (boundary_sha256 != lineage.get("boundary_receipt_sha256")
            or boundary.get("idea_id") != lineage["idea_id"]
            or boundary.get("attempt_id") != lineage["attempt_id"]
            or boundary.get("execution_identity_sha256")
            != lineage["execution_identity_sha256"]
            or boundary.get("kernel_path_isolation") is not True
            or boundary.get("training_network_denied") is not True):
        raise ModelLineageError("model_lineage_boundary_receipt_mismatch")
    separation = ensure_data_separation(cfg)
    separation_sha256 = data_separation_receipt_sha256(separation)
    if (lineage.get("data_separation_receipt_sha256") != separation_sha256
            or boundary.get("data_separation_receipt_sha256")
            != separation_sha256
            or boundary.get("data_boundary_policy_sha256") != _canonical_hash(
                dict(cfg.get("data_boundaries") or {}))):
        raise ModelLineageError("model_lineage_policy_evidence_mismatch")
    terminal_path = receipt_dir / "terminal.json"
    try:
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelLineageError("model_lineage_terminal_receipt_invalid") from exc
    if (not isinstance(terminal, dict)
            or terminal.get("idea_id") != lineage["idea_id"]
            or terminal.get("attempt_id") != lineage["attempt_id"]
            or terminal.get("phase") != "training"
            or terminal.get("event") != "terminal"
            or terminal.get("outcome") != "completed"
            or terminal.get("reason_code") != "trainer_completed"
            or terminal.get("return_code") != 0):
        raise ModelLineageError("model_lineage_terminal_receipt_invalid")
    artifact = _artifact_digest(
        _idea_path(idea_dir, str(spec["artifact"])),
        int(spec.get("max_files", 100_000)),
        int(spec.get("max_bytes", 100 * 1024 * 1024 * 1024)))
    if any(lineage.get(key) != value for key, value in artifact.items()):
        raise ModelLineageError("model_lineage_artifact_drift")
    return lineage, lineage_sha256


def model_lineage_evidence_paths(idea_dir: Path, cfg: Mapping) -> list[Path]:
    """Return metadata-tracked paths that can invalidate local lineage.

    This is a cache invalidation aid, not a validator.  It deliberately avoids
    reading model bytes; the full validator hashes them after any metadata
    change and before evaluation.
    """
    spec = cfg.get("model_lineage", {})
    if not isinstance(spec, Mapping) or spec.get("enabled") is not True:
        return []
    idea_dir = Path(idea_dir)
    lineage_path = idea_dir / LINEAGE_FILE
    paths = [lineage_path]
    try:
        if not lineage_path.is_symlink():
            envelope = json.loads(lineage_path.read_text(encoding="utf-8"))
            attempt_id = envelope.get("payload", {}).get("attempt_id")
            if isinstance(attempt_id, str) and _TOKEN.fullmatch(attempt_id):
                receipt_dir = idea_dir / "_compute_receipts" / attempt_id
                paths.extend(receipt_dir / name for name in (
                    "boundary.json", "start.json", "terminal.json"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        pass
    state_root = Path(cfg.get("_orze_dir") or (
        Path(cfg.get("_project_root", ".")) / ".orze"))
    paths.append(state_root / "state" / "data_separation.json")
    artifact_name = spec.get("artifact")
    if not isinstance(artifact_name, str):
        return paths
    try:
        artifact = _idea_path(idea_dir, artifact_name)
    except ModelLineageError:
        return paths
    paths.append(artifact)
    if artifact.is_symlink() or not artifact.is_dir():
        return paths
    raw_maximum = spec.get("max_files", 100_000)
    maximum = raw_maximum if _positive_int(raw_maximum) else 100_000
    observed = 0
    try:
        for directory, dirs, names in os.walk(artifact, followlinks=False):
            directory_path = Path(directory)
            paths.append(directory_path)
            for name in [*dirs, *names]:
                paths.append(directory_path / name)
                observed += 1
                if observed > maximum:
                    return paths
    except OSError:
        pass
    return paths
