"""Bind process completion to its captured trigger attempt, not role_state.

CALLING SPEC:
    delivery_reference(rp) -> dict
        Hash/ID-only association for the existing nonce-bound process receipt.
    settle_role_delivery(rp, outcome, exit_code, cleanup_verified) -> bool
        Persist the matching attempt result before releasing its process receipt.
        For owned v2 roles, True means the exact lock was already released.
        Legacy callers still release their v1 lock; False preserves uncertainty.

Process outcomes are operational, not scientific success. A stopped process
does not prove it had no external side effects and never authorizes replay.
"""
from __future__ import annotations

import hashlib
import json
import logging


logger = logging.getLogger("orze")
_REFERENCE_FIELDS = (
    "delivery_id", "scope", "role_name", "generation", "attempt_id",
    "nonce_sha256", "command_sha256",
)


def delivery_reference(rp):
    launch = rp.trigger_launch
    if not isinstance(launch, dict):
        raise ValueError("trigger_launch_invalid")
    reference = {key: launch[key] for key in _REFERENCE_FIELDS}
    if (reference["role_name"] != rp.role_name
            or type(reference["generation"]) is not int
            or reference["generation"] < 1
            or any(not isinstance(reference[key], str) or not reference[key]
                   for key in _REFERENCE_FIELDS if key != "generation")
            or not isinstance(rp.process_nonce, str)
            or reference["nonce_sha256"] != hashlib.sha256(
                rp.process_nonce.encode("ascii")).hexdigest()):
        raise ValueError("trigger_process_binding_invalid")
    return reference


def _remove_bound_receipt(rp):
    path = rp.lock_dir / "role-process.json"
    if path.is_symlink():
        return False
    try:
        if path.stat().st_size > 256 * 1024:
            return False
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return True
    except (OSError, UnicodeError):
        return False
    try:
        receipt = json.loads(raw)
        expected = delivery_reference(rp)
        if (receipt.get("role_name") != rp.role_name
                or receipt.get("nonce_sha256") != expected["nonce_sha256"]
                or receipt.get("trigger_delivery") != expected):
            return False
        path.unlink()
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def settle_role_delivery(rp, outcome, exit_code, cleanup_verified):
    from orze.engine.role_supervision import supervised_role_owner
    try:
        owner = supervised_role_owner(rp)
        if owner is not None:
            if cleanup_verified is not True or getattr(rp, "is_pending_role", False):
                return False
            owner.require_closed(exit_code)
            db_path, launch = owner.delivery_authority()
            if launch is not None:
                from orze.engine.trigger_delivery import record_terminal
                if record_terminal(
                        db_path, launch, outcome=outcome,
                        exit_code=exit_code, cleanup_verified=True) is not True:
                    return False
            return owner.release(outcome=outcome, exit_code=exit_code) is True
    except Exception as exc:
        logger.error("Role completion HOLD: %s", type(exc).__name__)
        return False
    launch = getattr(rp, "trigger_launch", None)
    if launch is None:
        return cleanup_verified
    try:
        delivery_reference(rp)
    except (ValueError, KeyError, TypeError, AttributeError):
        logger.error("Role %s trigger/process association is invalid", rp.role_name)
        return False
    db_path = getattr(rp, "trigger_delivery_db", None)
    if not db_path:
        logger.error("Role %s has no captured trigger database", rp.role_name)
        return False
    try:
        from orze.engine.trigger_delivery import record_in_doubt, record_terminal
        recorded = record_terminal(
            db_path, launch, outcome=outcome, exit_code=exit_code,
            cleanup_verified=cleanup_verified,
        )
        if not recorded:
            record_in_doubt(db_path, launch, "terminal_receipt_unconfirmed")
        if not recorded or not cleanup_verified:
            logger.error("Role %s trigger completion is unresolved", rp.role_name)
            return False
    except Exception as exc:
        # LAUNCHING/STARTED is itself non-replayable if storage is unavailable.
        logger.error("Role %s trigger completion storage failed: %s",
                     rp.role_name, type(exc).__name__)
        return False
    return _remove_bound_receipt(rp)


def stop_owned_role(rp, *, timeout=10):
    """Stop/settle one owned role; None denotes the unchanged legacy path.

    Pending admission and uncertain owners stay held. This is role-local
    closure, never a whole-controller shutdown acknowledgement.
    """
    from orze.engine.role_supervision import supervised_role_owner
    try:
        owner = supervised_role_owner(rp)
        if owner is None:
            return None
        if getattr(rp, "is_pending_role", False):
            return False
        closure = owner.abort(timeout=timeout)
        if closure is None:
            return False
        rp.close_log()
        return settle_role_delivery(
            rp, "interrupted", closure["worker_returncode"], True)
    except Exception as exc:
        logger.error("Role shutdown HOLD: %s", type(exc).__name__)
        return False
