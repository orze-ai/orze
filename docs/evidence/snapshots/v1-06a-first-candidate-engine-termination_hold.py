"""Durable, conservative authority for forced execution termination.

A stopped group leader is not proof that its writers stopped. Persist a stop
request before signaling; only an exact True from the tree reaper plus an
integer exit code may confirm it. An incomplete request holds the whole idea
across controller restart. No age/PID-based automatic resolution is provided.
This is not an execution lease, normal-exit descendant proof, or attempt CAS.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import secrets
import socket
import stat
from pathlib import Path

from orze.core.fs import atomic_create

_TREE = "_execution_stops"
_PHASES = {"training", "posthoc", "evaluation", "post_script", "pre_script", "artifact_preflight", "action"}
_TOKEN = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_MAX_STOPS = 1024


class TerminationUnconfirmed(RuntimeError):
    """Do not treat this as a repairable task failure or free its slot."""


def _directory(path: Path) -> None:
    for parent in (path, *path.parents):
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise OSError("execution_stop_directory_invalid")


def _read(path: Path) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or not 0 < before.st_size <= 8192):
            raise OSError("execution_stop_file_invalid")
        data = os.read(fd, 8193)
        after = os.fstat(fd)
        identity = lambda info: (info.st_dev, info.st_ino, info.st_size,
                                 info.st_mtime_ns, info.st_ctime_ns)
        if (len(data) != before.st_size or identity(before) != identity(after)):
            raise OSError("execution_stop_file_changed")
        return data
    finally:
        os.close(fd)


def _sync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _encoded(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"))
            + "\n").encode("utf-8")


def _publish(path: Path, payload: dict) -> bytes:
    encoded = _encoded(payload)
    if not atomic_create(path, encoded.decode("utf-8")):
        raise OSError("execution_stop_receipt_already_exists")
    if _read(path) != encoded:
        raise OSError("execution_stop_receipt_readback_failed")
    return encoded


def require_no_unconfirmed_stop(idea_dir: Path) -> None:
    """Read-only bounded gate; missing legacy evidence remains compatible."""
    idea_dir = Path(idea_dir)
    root = idea_dir / _TREE
    try:
        _directory(root)
        if not root.exists():
            return
        attempts = list(itertools.islice(root.iterdir(), _MAX_STOPS + 1))
        if not attempts or len(attempts) > _MAX_STOPS:
            raise ValueError("execution_stop_history_limit")
        for attempt in attempts:
            _directory(attempt)
            if not attempt.is_dir() or not _TOKEN.fullmatch(attempt.name):
                raise ValueError("execution_stop_attempt_invalid")
            # Partial publication/unknown files are not silently discarded.
            names = {entry.name for entry in itertools.islice(attempt.iterdir(), 3)}
            if names != {"requested.json", "confirmed.json"}:
                raise ValueError("execution_stop_unconfirmed")
            raw = _read(attempt / "requested.json")
            request = json.loads(raw)
            confirmed = json.loads(_read(attempt / "confirmed.json"))
            if (not isinstance(request, dict) or not isinstance(confirmed, dict)
                    or request.get("schema_version") != 1
                    or request.get("idea_id") != idea_dir.name
                    or request.get("attempt_id") != attempt.name
                    or request.get("phase") not in _PHASES
                    or request.get("event") != "stop_requested"
                    or not isinstance(request.get("nonce"), str)
                    or re.fullmatch(r"[0-9a-f]{32}", request["nonce"]) is None
                    or confirmed.get("schema_version") != 1
                    or confirmed.get("idea_id") != request["idea_id"]
                    or confirmed.get("attempt_id") != request["attempt_id"]
                    or confirmed.get("phase") != request["phase"]
                    or confirmed.get("event") != "stop_confirmed"
                    or confirmed.get("request_sha256")
                    != hashlib.sha256(raw).hexdigest()
                    or confirmed.get("tree_stopped") is not True
                    or type(confirmed.get("return_code")) is not int):
                raise ValueError("execution_stop_confirmation_invalid")
    except (OSError, ValueError, TypeError, UnicodeDecodeError, RecursionError) as exc:
        raise TerminationUnconfirmed("execution_termination_unconfirmed") from exc


def terminate_execution(tp, idea_dir: Path, *, phase: str, reaper,
                        timeout: float = 10) -> int:
    """Stop once, durably confirm, or permanently latch this handle in doubt.

    ``reaper`` is injected by the owning execution module so OS boundaries
    remain independently testable. A failed stop is never retried by rediscovering
    a now-dead leader: that could lose identities of escaped descendants.
    """
    from orze.engine.accounting import ensure_attempt_id

    error = f"{phase}_termination_unconfirmed"
    if getattr(tp, "_termination_unconfirmed", False) is True:
        raise TerminationUnconfirmed(error)
    idea_dir = Path(idea_dir)
    try:
        if phase not in _PHASES or getattr(tp, "idea_id", None) != idea_dir.name:
            raise ValueError("execution_stop_identity_invalid")
        require_no_unconfirmed_stop(idea_dir)
        attempt_id = ensure_attempt_id(tp)
        root = idea_dir / _TREE
        _directory(root)
        root.mkdir(exist_ok=True)
        _sync(idea_dir)
        attempt = root / attempt_id
        attempt.mkdir(exist_ok=False)
        _sync(root)
        request = {
            "schema_version": 1, "idea_id": tp.idea_id,
            "attempt_id": attempt_id, "phase": phase,
            "event": "stop_requested", "nonce": secrets.token_hex(16),
            "host": socket.gethostname(), "controller_pid": os.getpid(),
        }
        raw = _publish(attempt / "requested.json", request)
        # No effect or allocation terminal is authorized by merely returning
        # from the reaper, a truthy mock, or an exited group leader.
        stopped = reaper(tp.process, tp.idea_id, timeout=timeout)
        return_code = tp.process.poll()
        if stopped is not True or type(return_code) is not int:
            raise ValueError("execution_stop_not_proven")
        _publish(attempt / "confirmed.json", {
            "schema_version": 1, "idea_id": tp.idea_id,
            "attempt_id": attempt_id, "phase": phase,
            "event": "stop_confirmed", "tree_stopped": True,
            "request_sha256": hashlib.sha256(raw).hexdigest(),
            "return_code": return_code,
        })
        return return_code
    except Exception as exc:
        tp._termination_unconfirmed = True
        raise TerminationUnconfirmed(error) from exc
