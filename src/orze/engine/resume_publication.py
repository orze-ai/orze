"""Bounded resume publication after lock-free checkpoint validation.

Legacy re-admission never adopts a native catalog/history. Native started
callers may consume only their prepared request under an explicit effect
lease; their outer attempt transaction supplies actual execution authority.
No checkpoint hashing, process probe, provider or OS wait occurs here.
"""
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time

from orze.core.evaluation_retry_state import open_existing_lake
from orze.core.execution_attempts import current_attempt
from orze.core.fs import atomic_write
from orze.engine.attempt_effect_lock import (
    AttemptEffectInDoubt, attempt_effect_lock, require_effect_lease,
)
from orze.engine.execution_authority import canonical_identity_equal, execution_transaction
from orze.engine.execution_catalog import declared_catalog
from orze.engine.legacy_recovery import legacy_recovery_allowed


def _reject(reason):
    from orze.engine.resume import ResumeValidationError
    raise ResumeValidationError(reason)


def read_small(path, *, missing=False):
    from orze.engine.training_attempts import _read
    try:
        value, digest = _read(Path(path))
        if not canonical_identity_equal(value, value):
            _reject("resume_publication_json_invalid")
        return value, digest
    except FileNotFoundError:
        if missing:
            return None, None
        _reject("resume_publication_file_missing")


def _sync(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _archive_path(folder, label):
    for suffix in ("", *range(1, 1025)):
        stem = label if suffix == "" else f"{label}.{suffix}"
        candidate = folder / f"{stem}.json"
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
    _reject("resume_archive_limit")


def _open_catalog(folder, cfg, claim):
    if declared_catalog(folder) is not None:
        _reject("native_resume_admission_required")
    configured = cfg.get("idea_lake_db")
    bound = (claim or {}).get("lifecycle_db")
    paths = []
    for value in (configured, bound):
        if value is None:
            continue
        if type(value) is not str or not value:
            _reject("resume_catalog_unavailable")
        path = Path(value)
        if not path.is_absolute():
            from orze.engine.resume import _project_root
            path = _project_root(cfg, folder.parent) / path
        paths.append(path.absolute())
    if paths and any(path != paths[0] for path in paths[1:]):
        _reject("resume_catalog_scope_mismatch")
    # A loader-injected, still-absent default is not an explicit catalog.
    # The loader recomputes this provenance; a real binding or declaration
    # always wins, and existing/default-invalid databases still fail closed.
    if (paths and cfg.get("_idea_lake_db_defaulted") is True and bound is None
            and not paths[0].exists() and not paths[0].is_symlink()):
        return None
    if not paths:
        from orze.reporting.evidence import report_lifecycle_db_path
        candidate = report_lifecycle_db_path(folder.parent, cfg)
        if candidate.exists() or candidate.is_symlink():
            paths.append(candidate.absolute())
    if not paths:
        return None
    try:
        return open_existing_lake(paths[0])
    except (ValueError, OSError, sqlite3.Error):
        _reject("resume_catalog_unavailable")


def _require_legacy(lake, folder, lease):
    if declared_catalog(folder) is not None:
        _reject("native_resume_admission_required")
    if lake is not None and not legacy_recovery_allowed(lake, folder, lease=lease):
        _reject("native_resume_admission_required")


@contextmanager
def legacy_publication(folder, cfg):
    """Close owned catalog handles before releasing publication ownership."""
    claim, _ = read_small(folder / "claim.json", missing=True)
    lake = _open_catalog(folder, cfg, claim)
    closed = lake is None
    failure = None

    def close(primary=None):
        nonlocal closed
        closed = True
        if lake is not None:
            try:
                lake.close()
            except BaseException as cleanup:
                error = AttemptEffectInDoubt("resume_catalog_close_unconfirmed")
                error.cleanup_error = cleanup
                raise error from (primary if primary is not None else cleanup)

    try:
        with attempt_effect_lock(folder) as lease:
            try:
                _require_legacy(lake, folder, lease)
                yield lake, lease
            except BaseException as primary:
                close(primary)
                raise
            else:
                close()
    except BaseException as primary:
        failure = primary
        raise
    finally:
        if not closed:
            close(failure)


def publish_admission(folder, cfg, request, expected_claim_sha, expected_receipt_sha):
    """Publish a legacy request; native restart adoption is intentionally absent."""
    folder = Path(folder).absolute()
    with legacy_publication(folder, cfg) as (lake, lease):
            _require_legacy(lake, folder, lease)
            transaction = execution_transaction(lake, folder, lease=lease) if lake else nullcontext()
            with transaction:
                _require_legacy(lake, folder, lease)
                _, claim_sha = read_small(folder / "claim.json", missing=True)
                _, receipt_sha = read_small(folder / "interruption.json")
                if claim_sha != expected_claim_sha or receipt_sha != expected_receipt_sha:
                    _reject("resume_admission_evidence_changed")
                archives = []
                stamp = int(time.time())
                for label in ("metrics", "claim"):
                    path = folder / f"{label}.json"
                    value, digest = read_small(path, missing=True)
                    if value is not None:
                        archives.append((path, _archive_path(folder, f"{label}.interrupted.{stamp}"), digest))
                try:
                    if lake is not None:
                        state = lake.get_fsm_state(folder.name)
                        if state != "QUEUED" and not lake._record_state_transition_in_tx(
                                folder.name, state, "QUEUED", reason="legacy_resume_requested"):
                            _reject("idea_lake_requeue_failed")
                    request_path = folder / "resume_request.json"
                    expected = json.dumps(request, indent=2, sort_keys=True) + "\n"
                    atomic_write(request_path, expected)
                    if read_small(request_path)[1] != hashlib.sha256(expected.encode()).hexdigest():
                        _reject("resume_request_publication_unconfirmed")
                    for source, target, digest in archives:
                        if read_small(source)[1] != digest:
                            _reject("resume_admission_evidence_changed")
                        os.replace(source, target)
                        if read_small(target)[1] != digest or source.exists():
                            _reject("resume_archive_unconfirmed")
                    _sync(folder)
                    _require_legacy(lake, folder, lease)
                except BaseException as exc:
                    raise AttemptEffectInDoubt("resume_admission_publication_unconfirmed") from exc
    return request


def _native_context(folder, context, claim):
    declared = declared_catalog(folder)
    routes = [value for value in (declared, context.get("catalog_path"), claim.get("lifecycle_db")) if value is not None]
    if any(type(value) is not str or not Path(value).is_absolute() for value in routes):
        _reject("resume_catalog_scope_mismatch")
    if routes and any(Path(value).absolute() != Path(routes[0]).absolute() for value in routes[1:]):
        _reject("resume_catalog_scope_mismatch")
    if declared is not None:
        return True
    route = context.get("catalog_path") or claim.get("lifecycle_db")
    if route is None:
        return False
    try:
        lake = open_existing_lake(route)
    except (ValueError, OSError, sqlite3.Error):
        _reject("resume_catalog_unavailable")
    try:
        current_attempt(lake.conn, folder.name, "training")
        exists = lake.conn.execute("SELECT 1 FROM main.sqlite_master WHERE name='execution_attempts'").fetchone()
        return bool(exists and lake.conn.execute(
            "SELECT 1 FROM main.execution_attempts WHERE task_id=? COLLATE BINARY LIMIT 1",
            (folder.name,)).fetchone())
    finally:
        try:
            lake.close()
        except BaseException as exc:
            raise AttemptEffectInDoubt("resume_catalog_close_unconfirmed") from exc


def require_legacy_interruption(tp, folder, cfg):
    """Read-only early rejection before the legacy full-checkpoint hash."""
    if getattr(tp, "attempt_ref", None) is not None:
        _reject("native_interruption_publication_required")
    claim, _ = read_small(folder / "claim.json", missing=True)
    lake = _open_catalog(folder, cfg, claim)
    try:
        _require_legacy(lake, folder, None)
        owner = folder / "_attempt_effect.lock"
        if owner.exists() or owner.is_symlink():
            _reject("legacy_interruption_effect_held")
        from orze.engine.attempt_effect_receipts import require_closed_effects
        require_closed_effects(folder)
    finally:
        if lake is not None:
            try:
                lake.close()
            except BaseException as exc:
                raise AttemptEffectInDoubt("resume_catalog_close_unconfirmed") from exc


def publish_legacy_interruption(tp, folder, cfg, receipt, reason, return_code):
    """Legacy full-hash compatibility; actual writes share native's guard."""
    from orze.engine.resume import _interruption_reason_code
    from orze.engine.accounting import record_compute_terminal
    if getattr(tp, "attempt_ref", None) is not None:
        _reject("native_interruption_publication_required")
    with legacy_publication(folder, cfg) as (lake, lease):
        try:
            encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
            atomic_write(folder / "interruption.json", encoded)
            if read_small(folder / "interruption.json")[1] != hashlib.sha256(encoded.encode()).hexdigest():
                _reject("legacy_interruption_write_unconfirmed")
            record_compute_terminal(tp, folder, "interrupted", _interruption_reason_code(reason),
                                    phase="posthoc" if getattr(tp, "is_posthoc", False) else "training",
                                    return_code=return_code)
            _sync(folder)
            _require_legacy(lake, folder, lease)
        except BaseException as exc:
            raise AttemptEffectInDoubt("legacy_interruption_publication_unconfirmed") from exc


def consume_request(context, claim_path, *, effect_lease=None):
    claim_path = Path(claim_path).absolute()
    request_path = Path(context["request_path"]).absolute()
    folder = claim_path.parent
    if (claim_path.name != "claim.json" or request_path.name != "resume_request.json"
            or request_path.parent != folder or context.get("idea_id") != folder.name):
        _reject("resume_launch_scope_mismatch")
    expected_sha = context.get("request_sha256")
    if type(expected_sha) is not str or not re.fullmatch("[a-f0-9]{64}", expected_sha):
        _reject("resume_request_identity_missing")
    claim, _ = read_small(claim_path)
    if effect_lease is None and _native_context(folder, context, claim):
        _reject("native_resume_effect_lease_required")
    with attempt_effect_lock(folder, lease=effect_lease) as lease:
        claim, claim_sha = read_small(claim_path)
        native = _native_context(folder, context, claim)
        if effect_lease is None and native:
            _reject("native_resume_effect_lease_required")
        if native and (not isinstance(context.get("claim_attempt_id"), str)
                       or context["claim_attempt_id"] != claim.get("attempt_id")):
            _reject("resume_launch_claim_identity_changed")
        request, digest = read_small(request_path)
        if (digest != expected_sha or request.get("idea_id") != folder.name
                or request.get("interruption_receipt_sha256") != context.get("receipt_sha256")):
            _reject("resume_request_identity_changed")
        from orze.engine.resume import _stored_path
        root = context.get("project_root")
        if (not isinstance(root, str) or not Path(root).is_absolute()
                or str(_stored_path(request.get("checkpoint"), Path(root))) != context.get("checkpoint")):
            _reject("resume_checkpoint_context_changed")
        consumed = _archive_path(folder, "resume_request.consumed")
        claim.update({"resume_checkpoint": context["checkpoint"],
                      "resume_receipt_sha256": context["receipt_sha256"]})
        expected = json.dumps(claim, indent=2) + "\n"
        try:
            if read_small(claim_path)[1] != claim_sha:
                _reject("resume_launch_claim_identity_changed")
            atomic_write(claim_path, expected)
            if read_small(claim_path)[1] != hashlib.sha256(expected.encode()).hexdigest():
                _reject("resume_claim_publication_unconfirmed")
            if read_small(request_path)[1] != expected_sha:
                _reject("resume_request_identity_changed")
            os.replace(request_path, consumed)
            if read_small(consumed)[1] != expected_sha or request_path.exists():
                _reject("resume_request_consumption_unconfirmed")
            _sync(folder)
            require_effect_lease(lease, folder)
        except BaseException as exc:
            raise AttemptEffectInDoubt("resume_launch_publication_unconfirmed") from exc
