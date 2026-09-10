"""Immutable completion values and read-only downstream source checks.

An accepted terminal is not permission to replay work against another attempt.
Keep the value through consumers, recheck its exact current closed row and
confirmed effect, and pin it with watch_dependency when admitting a new action.
This is source validation, not durable per-consumer delivery acknowledgement.
"""
from __future__ import annotations

import logging
import hashlib
from contextlib import contextmanager
from pathlib import Path

from orze.core.execution_attempts import AttemptAuthorityError, AttemptRef, StaleAttempt, current_attempt
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
from orze.engine.execution_authority import canonical_identity_equal

logger = logging.getLogger("orze")


@contextmanager
def completion_cache_guard(event, lake, results_dir):
    """Serialize small source-owned caches with authorized retry/reset.

    This is neither an observation nor a delivery ACK. Read/parse configuration
    and send external notifications before entering. Cache callbacks may be
    best-effort; the guard does not assert that their writes succeeded.
    """
    from orze.engine.attempt_effect_lock import attempt_effect_lock
    with attempt_effect_lock(_folder(event, results_dir)):
        require_completion(event, lake, results_dir)
        yield


class CompletionEvent:
    """Two-item iteration/equality compatibility without a mutable __dict__."""
    __slots__ = ("_idea_id", "_resource", "_attempt_ref")

    def __init__(self, idea_id, resource, attempt_ref):
        if not isinstance(attempt_ref, AttemptRef) or idea_id != attempt_ref.task_id:
            raise ValueError("completion_reference_invalid")
        object.__setattr__(self, "_idea_id", idea_id)
        object.__setattr__(self, "_resource", resource)
        object.__setattr__(self, "_attempt_ref", attempt_ref)

    def __setattr__(self, name, value):
        raise AttributeError("CompletionEvent is immutable")

    def __delattr__(self, name):
        raise AttributeError("CompletionEvent is immutable")

    idea_id = property(lambda self: self._idea_id)
    resource = property(lambda self: self._resource)
    attempt_ref = property(lambda self: self._attempt_ref)

    def __iter__(self):
        return iter((self.idea_id, self.resource))

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return (self.idea_id, self.resource)[index]

    def __eq__(self, other):
        if isinstance(other, (tuple, list, CompletionEvent)):
            return tuple(self) == tuple(other)
        return NotImplemented

    __hash__ = None


def _scope(lake, folder):
    from orze.engine.execution_catalog import declared_catalog
    from orze.engine.claim_authority import read_claim, _lake_path
    declared = declared_catalog(folder)
    claim = read_claim(folder / "claim.json")
    claimed = (claim or {}).get("lifecycle_db")
    actual = _lake_path(lake)
    if any(path is not None and path != actual for path in (declared, claimed)):
        raise AttemptEffectBusy("completion_catalog_scope_mismatch")
    return actual


def _folder(event, results_dir):
    if not isinstance(event, (tuple, list, CompletionEvent)) or len(event) != 2:
        raise AttemptEffectBusy("completion_event_invalid")
    idea_id = event[0]
    if (type(idea_id) is not str or not idea_id or idea_id in (".", "..")
            or Path(idea_id).parts != (idea_id,)):
        raise AttemptEffectBusy("completion_event_identity_invalid")
    return Path(results_dir).absolute() / idea_id


def require_completion(event, lake, results_dir, *, phase=None):
    """Return a native row, or None only for an explicitly pre-native event."""
    from orze.engine.attempt_effect_receipts import _scan, _read, _decode, _ref_fields
    from orze.engine.termination_hold import require_no_unconfirmed_stop
    folder = _folder(event, results_dir)
    _scope(lake, folder)
    require_no_unconfirmed_stop(folder)
    ref = getattr(event, "attempt_ref", None)
    if ref is None:
        if lake is not None and any(current_attempt(lake.conn, event[0], p) is not None
                                    for p in ("training", "evaluation")):
            raise StaleAttempt("completion_native_reference_required")
        return None
    if (type(event) is not CompletionEvent or not isinstance(ref, AttemptRef)
            or ref.task_id != event[0] or ref.phase not in ("training", "evaluation")
            or (phase is not None and ref.phase != phase) or lake is None):
        raise AttemptEffectBusy("completion_reference_scope_invalid")
    row = current_attempt(lake.conn, ref.task_id, ref.phase)
    if (row is None or row["attempt_id"] != ref.attempt_id
            or row["generation"] != ref.generation or row["state"] not in ("TERMINAL", "NOT_STARTED")):
        raise StaleAttempt("completion_source_not_current_terminal")
    terminal = row["terminal"]
    digest = terminal.get("effect_receipt_sha256") if isinstance(terminal, dict) else None
    try:
        confirmed = _scan(folder).get(ref.attempt_id)
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise AttemptEffectInDoubt("completion_effect_unconfirmed") from exc
    if not isinstance(digest, str) or confirmed != (digest, True):
        raise AttemptEffectInDoubt("completion_effect_confirmation_missing")
    raw = _read(folder / "_execution_effects" / ref.attempt_id / "prepared.json")
    prepared = _decode(raw)
    fields = _ref_fields(ref, folder)
    if (hashlib.sha256(raw).hexdigest() != digest or not canonical_identity_equal(
            {key: prepared.get(key) for key in fields}, fields)):
        raise AttemptEffectInDoubt("completion_effect_reference_mismatch")
    # Read after filesystem checks too: a peer may rotate while receipts read.
    if not canonical_identity_equal(row, current_attempt(lake.conn, ref.task_id, ref.phase)):
        raise StaleAttempt("completion_source_changed")
    return row


def completion_is_current(event, lake, results_dir, *, phase=None):
    try:
        require_completion(event, lake, results_dir, phase=phase)
        return True
    except (AttemptAuthorityError, AttemptEffectBusy, AttemptEffectInDoubt,
            OSError, ValueError, TypeError, RuntimeError) as exc:
        logger.debug("Completion delivery deferred: %s", type(exc).__name__)
        return False


def filter_completions(events, lake, results_dir, *, phase=None):
    return [event for event in events
            if completion_is_current(event, lake, results_dir, phase=phase)]


def training_source(idea_id, resource, lake, results_dir):
    """Bind new pending/backlog work to current accepted training, if native.

    No training history means explicit legacy compatibility, not a fabricated
    training terminal. A native running/failed source cannot grant eval work.
    """
    folder = _folder((idea_id, resource), results_dir)
    _scope(lake, folder)
    if lake is None:
        return None
    row = current_attempt(lake.conn, idea_id, "training")
    if row is None:
        return None
    ref = AttemptRef(idea_id, "training", row["attempt_id"], row["generation"])
    event = CompletionEvent(idea_id, resource, ref)
    row = require_completion(event, lake, results_dir, phase="training")
    if row["terminal"].get("outcome") != "completed":
        raise StaleAttempt("evaluation_training_source_not_completed")
    return event


def accepted_eval_event(idea_id, resource, lake, results_dir):
    """Read a source for a new downstream action, never invent delivery."""
    if lake is None:
        return None
    row = current_attempt(lake.conn, idea_id, "evaluation")
    if row is None:
        return None
    event = CompletionEvent(idea_id, resource, AttemptRef(
        idea_id, "evaluation", row["attempt_id"], row["generation"]))
    require_completion(event, lake, results_dir, phase="evaluation")
    return event
