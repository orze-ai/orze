"""Durable, content-safe accounting for repeated watchdog launch failures."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
import time
from pathlib import Path
from typing import Iterable, Optional


SCHEMA_VERSION = 1
DEFAULT_ALERT_THRESHOLD = 2
DEFAULT_ALERT_COOLDOWN_SECONDS = 6 * 3600
MIN_ALERT_COOLDOWN_SECONDS = 5 * 60
MAX_STATE_BYTES = 64 * 1024
MAX_COUNTER = 2**63 - 1
_CODE_RE = re.compile(r"^[a-z0-9_]{1,64}$")
_STATE_KEYS = {
    "schema_version", "host", "active", "failure_code", "fingerprint",
    "consecutive_count", "first_seen_epoch", "last_seen_epoch",
    "last_alert_epoch", "alert_count", "recovered_from_invalid_state",
    "resolved_at_epoch", "resolution_code",
}
_ACTIVE_STATE_KEYS = _STATE_KEYS - {"resolved_at_epoch", "resolution_code"}


def _safe_code(value: object, fallback: str) -> str:
    text = str(value or "")
    return text if _CODE_RE.fullmatch(text) else fallback


def _safe_host(hostname: object) -> str:
    text = str(hostname or "unknown")
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    return safe[:128] or "unknown"


def failure_fingerprint(
    failure_code: str,
    identity_parts: Optional[Iterable[object]] = None,
) -> str:
    """Hash a canonical failure identity without retaining its raw inputs."""
    code = _safe_code(failure_code, "unclassified_failure")
    parts = sorted(str(part) for part in (identity_parts or ()))
    payload = json.dumps(
        {"failure_code": code, "identity_parts": parts},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _paths(results_dir: Path, hostname: str) -> tuple[Path, Path]:
    safe_host = _safe_host(hostname)
    return (
        results_dir / f".orze_watchdog_failures_{safe_host}.json",
        results_dir / f".orze_watchdog_failures_{safe_host}.lock",
    )


def _open_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = (
        os.O_RDWR | os.O_CREAT | os.O_NONBLOCK
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = os.open(str(path), flags, 0o600)
    info = os.fstat(fd)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_uid != os.geteuid()
    ):
        os.close(fd)
        raise OSError("unsafe watchdog failure lock")
    os.fchmod(fd, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _read_state(path: Path) -> tuple[dict, bool]:
    """Return state and whether invalid prior content had to be discarded."""
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(path), flags)
    except FileNotFoundError:
        return {}, False
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return {}, True
        size = info.st_size
        if size < 0 or size > MAX_STATE_BYTES:
            return {}, True
        chunks = []
        remaining = MAX_STATE_BYTES + 1
        while remaining > 0:
            chunk = os.read(fd, min(remaining, 8192))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
    finally:
        os.close(fd)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}, True
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        return {}, True
    active = value.get("active")
    allowed_keys = _ACTIVE_STATE_KEYS if active is True else _STATE_KEYS
    required_keys = _ACTIVE_STATE_KEYS
    fingerprint = value.get("fingerprint")
    failure_code = value.get("failure_code")
    count = value.get("consecutive_count")
    alert_count = value.get("alert_count")
    epochs = [value.get("first_seen_epoch"), value.get("last_seen_epoch")]
    if value.get("last_alert_epoch") is not None:
        epochs.append(value.get("last_alert_epoch"))
    if (
        not isinstance(active, bool)
        or not required_keys.issubset(value)
        or not set(value).issubset(allowed_keys)
        or not isinstance(value.get("host"), str)
        or _safe_host(value.get("host")) != value.get("host")
        or not isinstance(value.get("recovered_from_invalid_state"), bool)
        or not isinstance(fingerprint, str)
        or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
        or not isinstance(failure_code, str)
        or _safe_code(failure_code, "") != failure_code
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or count > MAX_COUNTER
        or isinstance(alert_count, bool)
        or not isinstance(alert_count, int)
        or alert_count < 0
        or alert_count > MAX_COUNTER
        or any(
            isinstance(epoch, bool)
            or not isinstance(epoch, (int, float))
            or not math.isfinite(float(epoch))
            or float(epoch) < 0
            for epoch in epochs
        )
    ):
        return {}, True
    if active is False:
        resolved_at = value.get("resolved_at_epoch")
        resolution_code = value.get("resolution_code")
        if (
            isinstance(resolved_at, bool)
            or not isinstance(resolved_at, (int, float))
            or not math.isfinite(float(resolved_at))
            or float(resolved_at) < 0
            or not isinstance(resolution_code, str)
            or _safe_code(resolution_code, "") != resolution_code
        ):
            return {}, True
    return value, False


def _atomic_write_state(path: Path, state: dict) -> None:
    encoded = (json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    if len(encoded) > MAX_STATE_BYTES:
        raise ValueError("watchdog failure state exceeds size limit")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = os.open(str(tmp), flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        offset = 0
        while offset < len(encoded):
            written = os.write(fd, encoded[offset:])
            if written <= 0:
                raise OSError("short write while persisting watchdog failure state")
            offset += written
        os.fsync(fd)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    finally:
        os.close(fd)
    try:
        os.replace(str(tmp), str(path))
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def record_failure(
    results_dir: Path,
    hostname: str,
    failure_code: str,
    identity_parts: Optional[Iterable[object]] = None,
    *,
    now: Optional[float] = None,
    alert_cooldown_seconds: float = DEFAULT_ALERT_COOLDOWN_SECONDS,
) -> dict:
    """Record one failure and return safe state plus an ephemeral alert flag."""
    now = float(time.time() if now is None else now)
    if not math.isfinite(now) or now < 0:
        raise ValueError("watchdog failure timestamp must be finite and nonnegative")
    threshold = DEFAULT_ALERT_THRESHOLD
    try:
        cooldown = float(alert_cooldown_seconds)
    except (TypeError, ValueError, OverflowError):
        cooldown = DEFAULT_ALERT_COOLDOWN_SECONDS
    if (
        isinstance(alert_cooldown_seconds, bool)
        or not math.isfinite(cooldown)
        or cooldown < MIN_ALERT_COOLDOWN_SECONDS
    ):
        cooldown = DEFAULT_ALERT_COOLDOWN_SECONDS
    code = _safe_code(failure_code, "unclassified_failure")
    fingerprint = failure_fingerprint(code, identity_parts)
    results = Path(results_dir)
    state_path, lock_path = _paths(results, hostname)
    lock_fd = _open_lock(lock_path)
    try:
        previous, recovered = _read_state(state_path)
        same = (
            previous.get("active") is True
            and previous.get("fingerprint") == fingerprint
        )
        count = (
            min(MAX_COUNTER, int(previous.get("consecutive_count", 0)) + 1)
            if same else 1
        )
        first_seen = previous.get("first_seen_epoch") if same else now
        if not isinstance(first_seen, (int, float)):
            first_seen = now
        last_alert = previous.get("last_alert_epoch") if same else None
        alert_count = int(previous.get("alert_count", 0)) if same else 0
        alert_due = count >= threshold and (
            not isinstance(last_alert, (int, float)) or now - last_alert >= cooldown
        )
        if alert_due:
            last_alert = now
            alert_count = min(MAX_COUNTER, alert_count + 1)
        state = {
            "schema_version": SCHEMA_VERSION,
            "host": _safe_host(hostname),
            "active": True,
            "failure_code": code,
            "fingerprint": fingerprint,
            "consecutive_count": count,
            "first_seen_epoch": float(first_seen),
            "last_seen_epoch": now,
            "last_alert_epoch": last_alert,
            "alert_count": alert_count,
            "recovered_from_invalid_state": bool(
                recovered or (same and previous.get("recovered_from_invalid_state"))
            ),
        }
        _atomic_write_state(state_path, state)
        result = dict(state)
        result["alert_due"] = alert_due
        return result
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def record_resolution(
    results_dir: Path,
    hostname: str,
    resolution_code: str,
    *,
    now: Optional[float] = None,
) -> bool:
    """Close an active loop without deleting its audit state."""
    now = float(time.time() if now is None else now)
    if not math.isfinite(now) or now < 0:
        raise ValueError("watchdog resolution timestamp must be finite and nonnegative")
    results = Path(results_dir)
    state_path, lock_path = _paths(results, hostname)
    if not state_path.exists():
        return False
    lock_fd = _open_lock(lock_path)
    try:
        state, recovered = _read_state(state_path)
        if not state or state.get("active") is not True:
            return False
        state["active"] = False
        state["resolved_at_epoch"] = now
        state["resolution_code"] = _safe_code(
            resolution_code, "watchdog_loop_resolved"
        )
        state["recovered_from_invalid_state"] = bool(
            recovered or state.get("recovered_from_invalid_state")
        )
        _atomic_write_state(state_path, state)
        return True
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def read_failure_state(results_dir: Path, hostname: str) -> Optional[dict]:
    """Read the safe operator view, or None when no state has been recorded."""
    results = Path(results_dir)
    state_path, lock_path = _paths(results, hostname)
    if not state_path.exists():
        return None
    lock_fd = _open_lock(lock_path)
    try:
        state, invalid = _read_state(state_path)
        if invalid:
            return {
                "schema_version": SCHEMA_VERSION,
                "valid": False,
                "error_code": "watchdog_failure_state_invalid",
            }
        if not state:
            return None
        result = dict(state)
        result["valid"] = True
        return result
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
