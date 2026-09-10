"""Training adapter identities, including an explicit pre-native import.

Legacy import consumes an already observed execution's claim/start receipts
inside its completion transaction. It is not a pre-Popen intent or permission
to start another process. Native history never permits tokenless enrollment.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

from orze.core.execution_attempts import (
    AttemptAuthorityError, AttemptRef, StaleAttempt, create_attempt,
    current_attempt, mark_running, require_current,
)
from orze.engine.attempt_effect_lock import AttemptEffectBusy
from orze.engine.execution_authority import lifecycle_fence


def _read(path, limit=65536):
    """Read one stable regular receipt without creating or repairing it."""
    path = Path(path)
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise AttemptEffectBusy("training_identity_redirected")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    identity = lambda st: (st.st_dev, st.st_ino, st.st_mode, st.st_nlink,
                           st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
            raise AttemptEffectBusy("training_identity_invalid")
        raw = os.read(fd, limit + 1)
        if (len(raw) > limit or identity(os.fstat(fd)) != identity(before)
                or identity(path.lstat()) != identity(before)):
            raise AttemptEffectBusy("training_identity_changed")
    finally:
        os.close(fd)
    try:
        def pairs(items):
            obj = {}
            for key, value in items:
                if key in obj:
                    raise ValueError("duplicate key")
                obj[key] = value
            return obj
        value = json.loads(raw, object_pairs_hook=pairs)
        if type(value) is not dict:
            raise ValueError("object required")
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise AttemptEffectBusy("training_identity_json_invalid") from exc
    return value, hashlib.sha256(raw).hexdigest()


def _claim(tp, idea_dir):
    claim, digest = _read(idea_dir / "claim.json", 8192)
    if claim.get("attempt_id") != tp.attempt_id:
        raise StaleAttempt("training_claim_replaced")
    if type(claim.get("gpu")) is not type(tp.gpu) or claim.get("gpu") != tp.gpu:
        raise AttemptEffectBusy("training_claim_resource_mismatch")
    return claim, digest


def _legacy_start(tp, idea_dir):
    start, digest = _read(idea_dir / "_compute_receipts" / tp.attempt_id / "start.json")
    expected = {
        "schema_version": 1, "event": "start", "outcome": "started",
        "idea_id": tp.idea_id, "attempt_id": tp.attempt_id, "phase": "training",
        "physical_gpu": tp.gpu, "process_pid": tp.process.pid,
    }
    if any(type(start.get(key)) is not type(value) or start.get(key) != value
           for key, value in expected.items()):
        raise AttemptEffectBusy("training_legacy_start_identity_mismatch")
    # The old watchdog may adjust an in-memory clock; persisted start remains
    # authoritative for identity, while PID alone is never an ownership grant.
    if (getattr(tp, "execution_identity", None) is not None
            and start.get("execution_identity_sha256") != tp.execution_identity):
        raise AttemptEffectBusy("training_legacy_execution_identity_mismatch")
    return digest


def current(lake, tp, idea_dir):
    """Return a current row/legacy candidate; stale or closed returns False."""
    try:
        row = current_attempt(lake.conn, tp.idea_id, "training")
        ref = getattr(tp, "attempt_ref", None)
        if ref is not None:
            if (not isinstance(ref, AttemptRef) or ref.phase != "training"
                    or ref.task_id != tp.idea_id or ref.attempt_id != tp.attempt_id):
                raise AttemptEffectBusy("training_attempt_identity_invalid")
            if row is None:
                raise AttemptEffectBusy("training_attempt_record_missing")
            if (row["attempt_id"], row["generation"]) != (ref.attempt_id, ref.generation):
                return False
        elif row is not None:
            if row["binding"].get("origin") != "legacy_import":
                raise AttemptEffectBusy("training_native_token_required")
            if row["attempt_id"] == tp.attempt_id:
                if row["state"] in ("TERMINAL", "NOT_STARTED"):
                    return False
                raise AttemptEffectBusy("training_legacy_import_unclosed")
            if row["state"] not in ("TERMINAL", "NOT_STARTED"):
                raise AttemptEffectBusy("training_previous_attempt_unclosed")
        if row is not None and ref is not None and row["state"] in ("TERMINAL", "NOT_STARTED"):
            return False
        claim, claim_sha = _claim(tp, idea_dir)
        if ref is None:
            start_sha = _legacy_start(tp, idea_dir)
            terminal = idea_dir / "_compute_receipts" / tp.attempt_id / "terminal.json"
            if terminal.exists() or terminal.is_symlink():
                # A pre-native terminal cannot be newly delivered as an
                # imported completion, regardless of its domain outcome.
                return False
            fence = lifecycle_fence(lake, tp.idea_id, "training")
            if fence["global_state"] != "IN_PROGRESS" or fence["phase_state"] != "IN_PROGRESS":
                return False
            return {"legacy": True, "claim_sha256": claim_sha,
                    "start_sha256": start_sha, "lifecycle": fence}
        row = require_current(lake.conn, ref)
        if lifecycle_fence(lake, tp.idea_id, "training") != row["binding"]["lifecycle"]:
            return False
        return row
    except StaleAttempt:
        return False
    except OSError as exc:
        raise AttemptEffectBusy("training_authority_unreadable") from exc


def import_for_completion(tx, tp, candidate):
    """Import only inside the terminal transaction, never advertise RUNNING."""
    if not candidate.get("legacy"):
        return tp.attempt_ref
    ref = create_attempt(tx.conn, tp.idea_id, "training", tp.attempt_id, {
        "origin": "legacy_import", "claim_sha256": candidate["claim_sha256"],
        "start_sha256": candidate["start_sha256"], "lifecycle": candidate["lifecycle"],
    })
    mark_running(tx.conn, ref)
    return ref
