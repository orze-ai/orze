"""Native posthoc wrapper for the shared READY and process-tree proof.

This fixes the execution phase, not a scientific validation or pipeline stage.
Legacy processes never acquire supervision from a raw PID or exit integer.
"""
from __future__ import annotations

from orze.engine import process_supervision as _shared

PROTOCOL = _shared.PROTOCOL


def identity(ep, idea_dir):
    return _shared.identity(ep, idea_dir, phase="posthoc")


def ready_binding(ep, idea_dir):
    return _shared.ready_binding(ep, idea_dir, phase="posthoc")


def bound_binding(ep, row, idea_dir, *, allow_launch_cleanup=False):
    return _shared.bound_binding(ep, row, idea_dir, phase="posthoc",
                                 allow_launch_cleanup=allow_launch_cleanup)


def require_closed(ep, row, idea_dir, ret, *, allow_launch_cleanup=False):
    return _shared.require_closed(ep, row, idea_dir, ret, phase="posthoc",
                                  allow_launch_cleanup=allow_launch_cleanup)


def failure_override(closure, forced):
    return _shared.failure_override(closure, forced, phase="posthoc")


def bind_launch_cleanup(row, closure):
    return _shared.bind_launch_cleanup(row, closure, phase="posthoc")
