"""Bounded native proposal results, bound to one captured role invocation.

This is operational yield, not measurement validity or a replay permission.
Legacy scripts and direct integer-cycle callers need not use this protocol.
The controller supplies the reference; a result never chooses its own attempt.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import uuid


CONTEXT_ENV = "ORZE_RESEARCH_RESULT_CONTEXT"
MAX_BYTES = 64 * 1024
MAX_IDS = 512
STATUSES = frozenset({"accepted", "partial", "empty", "rejected", "blocked", "error"})
_REF_KEYS = frozenset({
    "schema", "attempt_id", "role_name", "project_root", "results_dir",
    "ideas_file", "nonce_sha256", "result_path",
})
_RESULT_KEYS = frozenset({
    "status", "reason", "accepted_ids", "accepted_count", "rejected_count",
    "rejection_reasons",
})
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_CODE = re.compile(r"[a-z][a-z0-9_]{0,95}\Z")
_IDEA = re.compile(r"idea-[a-z0-9][a-z0-9-]{0,119}\Z")


class NativeResultError(ValueError):
    """A native result cannot authorize productive completion."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise NativeResultError("native_result_duplicate_key")
        value[key] = item
    return value


def _decode(raw):
    if len(raw) > MAX_BYTES:
        raise NativeResultError("native_result_too_large")
    try:
        return json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise NativeResultError("native_result_malformed") from exc


def _nonce_hash(nonce):
    if not isinstance(nonce, str) or not re.fullmatch(r"[0-9a-f]{64}", nonce):
        raise NativeResultError("native_result_nonce_invalid")
    return hashlib.sha256(nonce.encode("ascii")).hexdigest()


def validate_native_result_ref(ref, *, process_nonce):
    if (not isinstance(ref, dict) or set(ref) != _REF_KEYS
            or type(ref.get("schema")) is not int or ref["schema"] != 1):
        raise NativeResultError("native_result_reference_invalid")
    for key in ("attempt_id", "role_name"):
        if not isinstance(ref[key], str) or not _TOKEN.fullmatch(ref[key]):
            raise NativeResultError("native_result_identity_invalid")
    for key in ("project_root", "results_dir", "ideas_file", "result_path"):
        value = ref[key]
        if (not isinstance(value, str) or not value or len(value.encode("utf-8")) > 4096
                or "\0" in value or not Path(value).is_absolute()
                or str(Path(value)) != value or ".." in Path(value).parts):
            raise NativeResultError("native_result_scope_invalid")
    if ref["nonce_sha256"] != _nonce_hash(process_nonce):
        raise NativeResultError("native_result_nonce_mismatch")
    return dict(ref)


def make_native_result_ref(*, attempt_id, role_name, project_root, results_dir,
                           ideas_file, result_path, process_nonce):
    ref = {
        "schema": 1, "attempt_id": attempt_id, "role_name": role_name,
        "project_root": str(Path(project_root).absolute()),
        "results_dir": str(Path(results_dir).absolute()),
        "ideas_file": str(Path(ideas_file).absolute()),
        "result_path": str(Path(result_path).absolute()),
        "nonce_sha256": _nonce_hash(process_nonce),
    }
    return validate_native_result_ref(ref, process_nonce=process_nonce)


def native_result_context(*, project_root, results_dir, ideas_file):
    """Read the explicit child contract, checking the actual CLI scope."""
    raw = os.environ.get(CONTEXT_ENV)
    if raw is None:
        return None
    ref = validate_native_result_ref(
        _decode(raw.encode("utf-8")),
        process_nonce=os.environ.get("ORZE_ROLE_PROCESS_NONCE"),
    )
    for key, actual in (("project_root", project_root), ("results_dir", results_dir),
                        ("ideas_file", ideas_file)):
        if ref[key] != str(Path(actual).absolute()):
            raise NativeResultError("native_result_scope_mismatch")
    return ref


def validate_result(result):
    if (not isinstance(result, dict) or set(result) != _RESULT_KEYS
            or not isinstance(result["status"], str) or result["status"] not in STATUSES
            or not isinstance(result["reason"], str) or not _CODE.fullmatch(result["reason"])):
        raise NativeResultError("native_result_payload_invalid")
    ids, reasons = result["accepted_ids"], result["rejection_reasons"]
    if (not isinstance(ids, list) or len(ids) > MAX_IDS
            or any(not isinstance(item, str) or not _IDEA.fullmatch(item) for item in ids)
            or len(set(ids)) != len(ids)
            or type(result["accepted_count"]) is not int or result["accepted_count"] != len(ids)
            or not isinstance(reasons, dict) or len(reasons) > 32
            or any(not isinstance(key, str) or not _CODE.fullmatch(key)
                   or type(count) is not int or not 1 <= count <= 1_000_000
                   for key, count in reasons.items())
            or type(result["rejected_count"]) is not int
            or result["rejected_count"] != sum(reasons.values())):
        raise NativeResultError("native_result_yield_invalid")
    accepted, rejected = len(ids), result["rejected_count"]
    status = result["status"]
    if ((status == "accepted" and (not accepted or rejected))
            or (status == "partial" and (not accepted or not rejected))
            or (status not in {"accepted", "partial"} and accepted)
            or (status == "rejected" and not rejected)
            or (status == "empty" and rejected)):
        raise NativeResultError("native_result_status_invalid")
    encoded = _canonical(result)
    if len(encoded) > MAX_BYTES:
        raise NativeResultError("native_result_too_large")
    return json.loads(encoded)


def make_result(status, reason, *, accepted_ids=(), rejection_reasons=None):
    ids, reasons = list(accepted_ids), dict(rejection_reasons or {})
    return validate_result({
        "status": status, "reason": reason, "accepted_ids": ids,
        "accepted_count": len(ids), "rejected_count": sum(reasons.values()),
        "rejection_reasons": reasons,
    })


def _identity(ref):
    return {key: value for key, value in ref.items() if key not in {"schema", "result_path"}}


def _safe_path(path):
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise NativeResultError("native_result_redirected")
    if path.exists():
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise NativeResultError("native_result_not_regular")


def _read_bytes(path):
    _safe_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_size > MAX_BYTES):
            raise NativeResultError("native_result_not_bounded_regular")
        chunks, size = [], 0
        while size <= MAX_BYTES:
            chunk = os.read(fd, min(8192, MAX_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        after, named = os.fstat(fd), path.stat(follow_symlinks=False)
        fields = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if fields(before) != fields(after) or fields(after) != fields(named):
            raise NativeResultError("native_result_changed_during_read")
        raw = b"".join(chunks)
        if len(raw) > MAX_BYTES:
            raise NativeResultError("native_result_too_large")
        return raw
    finally:
        os.close(fd)


def read_native_result(ref, *, process_nonce):
    ref = validate_native_result_ref(ref, process_nonce=process_nonce)
    payload = _decode(_read_bytes(Path(ref["result_path"])))
    if (not isinstance(payload, dict) or set(payload) != {"schema", "identity", "result"}
            or type(payload["schema"]) is not int or payload["schema"] != 1
            or _canonical(payload["identity"]) != _canonical(_identity(ref))):
        raise NativeResultError("native_result_identity_mismatch")
    return validate_result(payload["result"])


def publish_native_result(ref, result, *, process_nonce):
    """Publish once after temporary-file close, then fsync/read back exactly.

    Any uncertainty raises. The native CLI must exit nonzero even when the
    final file already exists; its bytes alone cannot override process failure.
    """
    ref = validate_native_result_ref(ref, process_nonce=process_nonce)
    result = validate_result(result)
    raw = _canonical({"schema": 1, "identity": _identity(ref), "result": result})
    if len(raw) > MAX_BYTES:
        raise NativeResultError("native_result_too_large")
    path = Path(ref["result_path"])
    _safe_path(path)
    if path.exists():
        if _read_bytes(path) != raw:
            raise NativeResultError("native_result_already_published")
        return
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            written = 0
            while written < len(raw):
                count = os.write(fd, raw[written:])
                if count <= 0:
                    raise OSError("native_result_short_write")
                written += count
            os.fsync(fd)
        finally:
            os.close(fd)
        os.link(tmp, path)
        tmp.unlink()
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        if _read_bytes(path) != raw:
            raise NativeResultError("native_result_readback_mismatch")
    finally:
        tmp.unlink(missing_ok=True)
