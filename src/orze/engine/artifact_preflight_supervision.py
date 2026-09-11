"""Artifact resolver CPU supervision; not artifact or scientific validation."""
from orze.engine import process_supervision as _shared

PROTOCOL = _shared.PROTOCOL


def identity(ep, idea_dir):
    return _shared.identity(ep, idea_dir, phase="artifact_preflight")


def ready_binding(ep, idea_dir):
    return _shared.ready_binding(ep, idea_dir, phase="artifact_preflight")


def bound_binding(ep, row, idea_dir, *, allow_launch_cleanup=False):
    return _shared.bound_binding(ep, row, idea_dir, phase="artifact_preflight",
                                allow_launch_cleanup=allow_launch_cleanup)


def require_closed(ep, row, idea_dir, ret, *, allow_launch_cleanup=False):
    return _shared.require_closed(ep, row, idea_dir, ret, phase="artifact_preflight",
                                 allow_launch_cleanup=allow_launch_cleanup)


def failure_override(closure, forced):
    return _shared.failure_override(closure, forced, phase="artifact_preflight")


def bind_launch_cleanup(row, closure):
    return _shared.bind_launch_cleanup(row, closure, phase="artifact_preflight")
