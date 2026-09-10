"""Observe/stop outside the short current-attempt publication transaction."""
from __future__ import annotations

import time

from orze.engine.termination_hold import TerminationUnconfirmed, terminate_execution
from orze.engine.training_attempts import current
from orze.engine.training_completion import finish


def poll(lake, tp, slot, idea_dir, cfg, ret, elapsed, failure_counts, stall_minutes):
    """Handle a still-running training process; None leaves its slot owned."""
    from orze.engine import launcher
    from orze.engine.health import check_stalled, detect_fatal_in_log
    from orze.engine.interruption_publication import prepare_interruption
    from orze.engine.resume import _interruption_reason_code
    reason = None
    detail = ""
    if elapsed > tp.timeout:
        reason, detail = "timeout", "Timed out"
    elif check_stalled(tp, stall_minutes):
        reason, detail = "stall", f"Stalled (no output for {stall_minutes}m)"
    elif elapsed > 120 and launcher._detect_zombie(tp):
        reason, detail = "zombie", "Process stuck (zombie: no CPU/GPU activity)"
    elif launcher._watchdog_check(tp):
        reason, detail = "watchdog", "stuck_no_progress"
    else:
        fatal = detect_fatal_in_log(tp)
        if fatal and tp.process.poll() is None:
            reason, detail = "fatal_log", f"Process hung after fatal error: {fatal[:500]}"
        elif (idea_dir / ".kill").exists():
            reason, detail = "admin_kill", "Killed by admin"
    if reason is None:
        return None
    # The stop itself is never authorized by a stale status label or slot key.
    if not current(lake, tp, idea_dir):
        return None
    try:
        ret = terminate_execution(tp, idea_dir, phase="training", reaper=launcher._terminate_and_reap)
    except TerminationUnconfirmed:
        return None
    tp.close_log()
    prepared = prepare_interruption(tp, idea_dir.parent, cfg, reason, "SIGTERM", ret)
    return finish(lake, tp, slot, idea_dir, cfg, ret, failure_counts,
                  forced=("interrupted", _interruption_reason_code(reason), detail),
                  interruption=prepared)
