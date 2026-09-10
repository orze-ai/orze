"""Bounded compute publication checks against explicit execution context.

This verifies observed allocation facts, not process ownership or scientific
qualification. The caller must own the native attempt transaction. Legacy
accounting writers retain their existing compatibility semantics.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import stat

from orze.core.execution_attempts import AttemptAuthorityError, _json
from orze.engine import accounting
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
from orze.engine.execution_authority import canonical_identity_equal

MAX_RECEIPT_BYTES = 65536


def _encoded(payload):
    # Validate depth, node count and JSON types before encoding the wire form.
    _json(payload)
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      allow_nan=False) + "\n").encode()
    if len(raw) > MAX_RECEIPT_BYTES:
        raise ValueError("compute receipt oversized")
    return raw


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _parents(path):
    for parent in path.parents:
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise ValueError("compute receipt parent redirected")


def _read(path, *, expected=None):
    _parents(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or not 0 < before.st_size <= MAX_RECEIPT_BYTES):
            raise ValueError("compute receipt invalid file")
        raw = bytearray()
        while len(raw) <= MAX_RECEIPT_BYTES:
            chunk = os.read(fd, min(8192, MAX_RECEIPT_BYTES + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        raw = bytes(raw)
        if (len(raw) != before.st_size or _identity(os.fstat(fd)) != _identity(before)
                or _identity(path.lstat()) != _identity(before)):
            raise ValueError("compute receipt changed")
        value = json.loads(raw)
        if _encoded(value) != raw or (expected is not None and raw != expected):
            raise ValueError("compute receipt readback mismatch")
        os.fsync(fd)
    finally:
        os.close(fd)
    return value, raw, _identity(before)


def _context(process, idea_dir, phase, event, outcome):
    idea = getattr(process, "idea_id", None)
    attempt = getattr(process, "attempt_id", None)
    if (type(idea) is not str or idea != idea_dir.name
            or type(attempt) is not str or attempt in ("", ".", "..")
            or event not in ("start", "terminal")):
        raise ValueError("compute execution context invalid")
    accounting._token(attempt, "attempt_id")
    start = getattr(process, "start_time", None)
    if type(start) not in (int, float) or not math.isfinite(start) or start < 0:
        raise ValueError("compute start clock invalid")
    pid = getattr(getattr(process, "process", None), "pid", None)
    if pid is not None and (type(pid) is not int or pid <= 0):
        raise ValueError("compute process identity invalid")
    expected = accounting._base(process, phase, event, outcome)
    expected.update(started_at_epoch=round(float(start), 6), process_pid=pid)
    return expected


def _matches(payload, expected):
    if (type(payload) is not dict or not set(expected).issubset(payload)
            or not canonical_identity_equal({key: payload[key] for key in expected}, expected)
            or ("execution_identity_sha256" not in expected
                and payload.get("execution_identity_sha256") is not None)):
        raise ValueError("compute receipt context mismatch")
    duration = payload.get("allocated_gpu_seconds")
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
        raise ValueError("compute receipt duration invalid")


def verify_compute_receipt(idea_dir, payload, *, process, phase, event, outcome,
                           reason_code=None, return_code=None, require_start=True):
    """Check exact caller facts, persisted start and actual terminal bytes.

    RUNNING callers require the immutable start. A confirmed stopped process
    whose LAUNCHING registration failed may lack that file, but any existing
    start must still agree. Missing/partial receipts are never synthesized.
    """
    try:
        if type(require_start) is not bool:
            raise ValueError("compute start policy invalid")
        idea_dir = Path(idea_dir).absolute()
        if ".." in idea_dir.parts:
            raise ValueError("compute directory invalid")
        expected = _context(process, idea_dir, phase, event, outcome)
        initial_context = dict(expected)
        if event == "terminal":
            if (type(reason_code) is not str or not reason_code
                    or (return_code is not None and type(return_code) is not int)):
                raise ValueError("compute terminal context invalid")
            accounting._token(reason_code, "reason_code")
            expected.update(reason_code=reason_code, return_code=return_code)
        elif outcome != "started" or reason_code is not None or return_code is not None:
            raise ValueError("compute start context invalid")
        _matches(payload, expected)
        wire = _encoded(payload)
        folder = idea_dir / "_compute_receipts" / expected["attempt_id"]
        start_snapshot = None
        if event == "terminal":
            start_path = folder / "start.json"
            try:
                start_snapshot = _read(start_path)
            except FileNotFoundError:
                if require_start:
                    raise
            if start_snapshot is not None:
                start_expected = _context(process, idea_dir, phase, "start", "started")
                _matches(start_snapshot[0], start_expected)
        path = folder / (event + ".json")
        _read(path, expected=wire)
        if start_snapshot is not None:
            final_start = _read(folder / "start.json", expected=start_snapshot[1])
            if final_start[2] != start_snapshot[2]:
                raise ValueError("compute start changed during terminal publication")
        if not canonical_identity_equal(
                _context(process, idea_dir, phase, event, outcome), initial_context):
            raise ValueError("compute execution context changed during readback")
        for parent in (folder, folder.parent, idea_dir):
            fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, RecursionError,
            AttemptAuthorityError, accounting.ComputeAccountingError) as exc:
        raise AttemptEffectInDoubt("native_compute_publication_unconfirmed") from exc
