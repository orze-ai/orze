"""Explicit controller-control status, never a process-closure capability.

The current refusal slice cannot establish controller/tree closure. Even the
reserved ``confirmed`` status is only a typed label; constructing this object
does not authorize restart, reallocation, adoption, or clearing a stop marker.
The start gate reads existing markers only and never clears or creates state.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


STOP_SENTINELS = (".orze_disabled", ".orze_stop_all", ".orze_shutdown")
_STATUSES = frozenset({"requested", "hold", "confirmed"})
_REASON = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


class ControllerStopHOLD(RuntimeError):
    """Controller stop/start authority is unresolved; do not continue work."""


@dataclass(frozen=True)
class StopOutcome:
    """A result label, not evidence that any process tree has stopped."""

    status: str
    reason_code: str

    def __post_init__(self):
        if type(self.status) is not str or self.status not in _STATUSES:
            raise ValueError("stop_outcome_status_invalid")
        if type(self.reason_code) is not str or _REASON.fullmatch(self.reason_code) is None:
            raise ValueError("stop_outcome_reason_invalid")

    def __bool__(self):
        raise TypeError("StopOutcome has no implicit truth value; inspect its status")


def require_controller_start_allowed(results_dir) -> None:
    """Refuse any old stop marker, including dangling links and unreadable paths.

    Absence permits only continuation to the caller's other admission checks;
    it is not stop confirmation or permission to adopt an existing execution.
    Missing directories are not created. No marker is interpreted as stale.
    """
    try:
        root = Path(results_dir)
        # Once a scope has entered registered ownership, another framework
        # controller (including a legacy-mode invocation) cannot start there.
        # Persistent namespace presence denies except for the exact ACTIVE
        # context or a strongly validated pending successor. No age takeover.
        namespace = root / "_controller_registration.lock.source-lock"
        try:
            namespace.lstat()
        except FileNotFoundError:
            pass
        else:
            from orze.engine.controller_control import ControllerHOLD, current_controller
            ctx = current_controller()
            if ctx is None:
                from orze.engine.controller_handoff import require_pending_start
                try:
                    require_pending_start(root.absolute())
                except ControllerHOLD:
                    raise ControllerStopHOLD("controller_start_blocked_by_registration") from None
            elif ctx.scope != root.absolute():
                raise ControllerStopHOLD("controller_start_blocked_by_registration")
            else:
                ctx.check_admission()
        for name in STOP_SENTINELS:
            try:
                (root / name).lstat()
            except FileNotFoundError:
                continue
            raise ControllerStopHOLD("controller_start_blocked_by_sentinel:" + name)
    except ControllerStopHOLD:
        raise
    except (OSError, TypeError, ValueError):
        raise ControllerStopHOLD("controller_stop_state_unverifiable") from None
