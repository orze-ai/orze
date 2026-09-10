"""Bind process completion to its captured trigger attempt, not role_state.

CALLING SPEC:
    delivery_reference(rp) -> dict
        Hash/ID-only association for the existing nonce-bound process receipt.
    settle_role_delivery(rp, outcome, exit_code, cleanup_verified) -> bool
        Persist the matching attempt result before releasing its process receipt.
        True permits lock release; false preserves uncertain delivery evidence.

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
