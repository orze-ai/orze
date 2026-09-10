"""Native shutdown closes an owned execution, never invents an observation.

OS termination and optional checkpoint preparation occur outside the short
publication transaction. Config-free atexit records allocation/lifecycle only;
it does not invent a resumable checkpoint or use default evaluator output paths.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
import socket
import sys

from orze.core.execution_attempts import (
    AttemptAuthorityError, AttemptRef, current_attempt, finish_attempt,
)
from orze.engine.attempt_effect_lock import AttemptEffectBusy
from orze.engine.execution_authority import execution_transaction, lifecycle_fence

logger = logging.getLogger("orze")


def _current(lake, tracked, folder, phase, cfg):
    from orze.engine.posthoc_attempts import require_catalog as posthoc_catalog
    posthoc_catalog(lake, folder, cfg, handle=tracked)
    if phase == "training":
        from orze.engine.training_attempts import require_catalog, current
        require_catalog(lake, folder, cfg, handle=tracked)
    elif phase == "posthoc":
        from orze.engine.posthoc_attempts import current
    else:
        from orze.engine.native_evaluation import require_catalog, is_current as current
        require_catalog(lake, folder.name, folder.parent, cfg, handle=tracked)
    row = current_attempt(lake.conn, folder.name, phase)
    ref = getattr(tracked, "attempt_ref", None)
    if row is None and ref is None:
        return None
    if (not isinstance(ref, AttemptRef) or ref.task_id != folder.name
            or ref.phase != phase or ref.attempt_id != tracked.attempt_id):
        raise AttemptEffectBusy("shutdown_native_reference_required")
    if row is None:
        raise AttemptEffectBusy("shutdown_native_attempt_missing")
    if (row["attempt_id"], row["generation"]) != (ref.attempt_id, ref.generation):
        return False
    if row["state"] in ("TERMINAL", "NOT_STARTED"):
        return False
    accepted = current(lake, tracked, folder) if phase in ("training", "posthoc") else current(lake, tracked)
    if not accepted or row["state"] != "RUNNING":
        raise AttemptEffectBusy("shutdown_native_attempt_not_owned")
    if phase == "evaluation":
        from orze.engine.evaluation_supervision import bound_binding
        bound_binding(tracked, row, folder)
    elif phase == "posthoc":
        from orze.engine.posthoc_supervision import bound_binding
        bound_binding(tracked, row, folder)
    elif row["binding"].get("origin") == "native_training":
        from orze.engine.training_supervision import bound_binding
        bound_binding(tracked, row, folder)
    return True


def handle_shutdown(tracked, results_dir, phase, stop, *, lake=None, cfg=None):
    """Return None for pre-native compatibility, else whether ownership closes.

    False preserves the handle on uncertainty. True includes stale deliveries:
    an already closed/replaced attempt must not signal or mutate another one.
    The catalog declaration is only a route; the exact attempt is revalidated.
    """
    if results_dir is None or phase not in ("training", "evaluation", "posthoc"):
        if getattr(tracked, "attempt_ref", None) is not None:
            return False
        return None
    folder = Path(results_dir) / tracked.idea_id
    opened = None
    try:
        from orze.engine.execution_catalog import declared_catalog
        route = declared_catalog(folder)
        if lake is None:
            if route is None:
                if getattr(tracked, "attempt_ref", None) is not None:
                    raise AttemptEffectBusy("shutdown_native_catalog_required")
                return None
            from orze.core.evaluation_retry_state import open_existing_lake
            opened = lake = open_existing_lake(route)
        disposition = _current(lake, tracked, folder, phase, cfg or {})
        if disposition is None:
            return None
        if disposition is False:
            return True
        if not stop(tracked, Path(results_dir), phase):
            return False
        ret = tracked.process.poll()
        if type(ret) is not int:
            raise AttemptEffectBusy("shutdown_exit_unconfirmed")
        closure = None
        if phase == "evaluation":
            from orze.engine.evaluation_supervision import require_closed
            closure = require_closed(tracked, current_attempt(lake.conn, folder.name, phase),
                                     folder, ret)
        elif phase == "training":
            from orze.engine.training_supervision import require_closed
            closure = require_closed(tracked, current_attempt(lake.conn, folder.name, phase),
                                     folder, ret)
        elif phase == "posthoc":
            from orze.engine.posthoc_supervision import require_closed
            closure = require_closed(tracked, current_attempt(lake.conn, folder.name, phase),
                                     folder, ret)
        interruption = None
        reason = phase + "_controller_shutdown"
        if phase == "training" and cfg is not None:
            from orze.engine.interruption_publication import prepare_interruption
            from orze.engine.resume import _interruption_reason_code
            interruption = prepare_interruption(tracked, Path(results_dir), cfg,
                                                "orze_stop", "SIGTERM", ret)
            reason = _interruption_reason_code("orze_stop")
        with execution_transaction(lake, folder) as tx:
            if _current(lake, tracked, folder, phase, cfg or {}) is not True:
                return True
            if closure is not None:
                from orze.engine.execution_authority import canonical_identity_equal
                if not canonical_identity_equal(closure, require_closed(
                        tracked, current_attempt(lake.conn, folder.name, phase), folder, ret)):
                    raise AttemptEffectBusy("shutdown_process_tree_receipt_changed")
            ref = tracked.attempt_ref
            plan = {"operation": "controller_shutdown", "outcome": "interrupted",
                    "return_code": ret, "reason_code": reason}
            if closure is not None:
                plan["process_tree"] = closure
            digest = tx.prepare(ref, plan)
            from orze.engine.compute_publication import verify_compute_receipt
            if interruption is not None:
                from orze.engine.interruption_publication import publish_interruption
                from orze.engine.training_attempts import _read
                publish_interruption(interruption, tracked, Path(results_dir), cfg)
                receipt, _ = _read(folder / "_compute_receipts" / ref.attempt_id / "terminal.json")
            else:
                from orze.engine.accounting import record_compute_terminal
                receipt = record_compute_terminal(tracked, folder, "interrupted", reason,
                                                  phase=phase, return_code=ret)
            verify_compute_receipt(folder, receipt, process=tracked, phase=phase,
                                   event="terminal", outcome="interrupted",
                                   reason_code=reason, return_code=ret)
            if not lake._record_state_transition_in_tx(
                    folder.name, "IN_PROGRESS", "FAILED", reason=reason,
                    host=socket.gethostname(), pid=os.getpid(), sop_type="training"):
                raise AttemptAuthorityError("shutdown_lifecycle_rejected")
            terminal = {"outcome": "interrupted", "reason_code": reason,
                        "return_code": ret, "effect_receipt_sha256": digest,
                        "lifecycle": lifecycle_fence(lake, folder.name,
                                                     "training" if phase == "posthoc" else phase)}
            if phase == "posthoc":
                terminal["lifecycle_phase"] = "training"
            if closure is not None:
                terminal["process_tree"] = closure
            if finish_attempt(tx.conn, ref, terminal) != "committed":
                raise AttemptAuthorityError("shutdown_terminal_not_new")
        tracked.close_log()
        return True
    except Exception as exc:
        logger.error("[SHUTDOWN-HOLD] %s: %s", tracked.idea_id, type(exc).__name__)
        return False
    finally:
        if opened is not None:
            propagating_error = sys.exc_info()[0] is not None
            try:
                opened.close()
            except Exception as exc:
                logger.error("[SHUTDOWN-HOLD] %s: catalog close failed: %s",
                             tracked.idea_id, type(exc).__name__)
                # Preserve a pending exception (including a second interrupt),
                # otherwise retain the handle without aborting other cleanup.
                if not propagating_error:
                    return False
