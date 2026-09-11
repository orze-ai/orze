"""CPU pre-script wrapper; closure is not GPU or scientific accounting."""
from orze.engine import process_supervision as _shared

PROTOCOL = _shared.PROTOCOL


def identity(ep, idea_dir):
    return _shared.identity(ep, idea_dir, phase="pre_script")


def ready_binding(ep, idea_dir):
    return _shared.ready_binding(ep, idea_dir, phase="pre_script")


def bound_binding(ep, row, idea_dir, *, allow_launch_cleanup=False):
    return _shared.bound_binding(ep, row, idea_dir, phase="pre_script",
                                allow_launch_cleanup=allow_launch_cleanup)


def require_closed(ep, row, idea_dir, ret, *, allow_launch_cleanup=False):
    return _shared.require_closed(ep, row, idea_dir, ret, phase="pre_script",
                                 allow_launch_cleanup=allow_launch_cleanup)


def failure_override(closure, forced):
    return _shared.failure_override(closure, forced, phase="pre_script")


def bind_launch_cleanup(row, closure):
    return _shared.bind_launch_cleanup(row, closure, phase="pre_script")
