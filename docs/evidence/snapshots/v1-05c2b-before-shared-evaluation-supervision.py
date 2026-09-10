"""Bind native evaluation publication to a prelaunch-owned process tree.

The Linux supervisor proves local descendant closure. This adapter checks its
exact attempt/scope binding before reading results and again at publication.
Neither an integer exit code nor an old unbound native row supplies that proof.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re

from orze.engine.attempt_effect_lock import AttemptEffectBusy

PROTOCOL = "orze.linux_subreaper.v1"
_HEX = re.compile(r"[0-9a-f]{64}")


def identity(ep, idea_dir):
    from orze.engine.native_evaluation import _ref
    return {"attempt_ref": asdict(_ref(ep)), "scope": str(Path(idea_dir).absolute())}


def _same(first, second):
    try:
        return json.dumps(first, sort_keys=True, separators=(",", ":"), allow_nan=False) == (
            json.dumps(second, sort_keys=True, separators=(",", ":"), allow_nan=False))
    except (ValueError, TypeError, RecursionError):
        return False


def ready_binding(ep, idea_dir):
    """Read bounded immutable READY identity; never discover or wrap an old PID."""
    from orze.engine.supervised_process import SupervisedProcess
    process = getattr(ep, "process", None)
    if not isinstance(process, SupervisedProcess):
        raise AttemptEffectBusy("evaluation_supervised_process_required")
    try:
        binding = process.binding
        if (type(binding) is not dict or set(binding) != {
                "schema", "protocol", "identity", "nonce_sha256", "command_sha256",
                "worker", "supervisor"}
                or type(binding["schema"]) is not int or binding["schema"] != 1
                or binding["protocol"] != PROTOCOL
                or not _same(binding["identity"], identity(ep, idea_dir))):
            raise ValueError("binding")
        for field in ("nonce_sha256", "command_sha256"):
            if type(binding[field]) is not str or _HEX.fullmatch(binding[field]) is None:
                raise ValueError("digest")
        for field in ("worker", "supervisor"):
            member = binding[field]
            if (type(member) is not dict or set(member) != {"pid", "start_ticks"}
                    or type(member["pid"]) is not int or member["pid"] <= 0
                    or type(member["start_ticks"]) is not int or member["start_ticks"] < 0):
                raise ValueError("process_identity")
        if (type(process.pid) is not int or process.pid != binding["worker"]["pid"]
                or process.supervisor_pid != binding["supervisor"]["pid"]
                or process.pid == process.supervisor_pid):
            raise ValueError("handle_identity")
        if len(json.dumps(binding, allow_nan=False).encode()) > 8192:
            raise ValueError("binding_limit")
    except Exception as exc:
        raise AttemptEffectBusy("evaluation_supervision_binding_invalid") from exc
    return binding


def bound_binding(ep, row, idea_dir, *, allow_launch_cleanup=False):
    """An old or replaced process cannot acquire supervision by callback."""
    bound = row["binding"]
    if bound.get("process_supervision_protocol") != PROTOCOL:
        raise AttemptEffectBusy("evaluation_supervision_unbound")
    ready = ready_binding(ep, idea_dir)
    stored = bound.get("supervision")
    if not _same(stored, ready):
        if not (allow_launch_cleanup is True and row["state"] == "LAUNCHING"
                and "supervision" not in bound):
            raise AttemptEffectBusy("evaluation_supervision_binding_changed")
    return ready


def require_closed(ep, row, idea_dir, ret, *, allow_launch_cleanup=False):
    """Nonblocking proof check, safe both outside and inside the short writer."""
    from orze.engine.supervised_process import SupervisionUncertain
    if type(ret) is not int:
        raise AttemptEffectBusy("evaluation_exit_unconfirmed")
    ready = bound_binding(ep, row, idea_dir, allow_launch_cleanup=allow_launch_cleanup)
    try:
        actual = ep.process.poll()
        closure = ep.process.closure_receipt()
    except SupervisionUncertain as exc:
        ep._termination_unconfirmed = True
        raise AttemptEffectBusy("evaluation_supervision_unconfirmed") from exc
    if type(actual) is not int or actual != ret or closure is None:
        raise AttemptEffectBusy("evaluation_process_tree_unclosed")
    if (type(closure) is not dict or set(closure) != {
            "schema", "event", "binding", "worker_returncode", "stop_requested",
            "forced_cleanup", "reaped_children", "wait_proof"}
            or type(closure["schema"]) is not int or closure["schema"] != 1
            or closure["event"] != "TREE_CLOSED"
            or not _same(closure["binding"], ready)
            or type(closure["worker_returncode"]) is not int
            or closure["worker_returncode"] != ret
            or type(closure["stop_requested"]) is not bool
            or type(closure["forced_cleanup"]) is not bool
            or type(closure["reaped_children"]) is not int
            or closure["reaped_children"] < 1
            or closure["wait_proof"] != "ECHILD_WALL"):
        raise AttemptEffectBusy("evaluation_process_tree_receipt_invalid")
    if "supervision" not in row["binding"] and not closure["stop_requested"]:
        raise AttemptEffectBusy("evaluation_launch_cleanup_not_proven")
    return closure


def failure_override(closure, forced):
    """Closed effects are not successful evaluation when stopping was required."""
    if closure["stop_requested"] or closure["forced_cleanup"]:
        if forced is None or forced[0] == "completed":
            return ("failed", "evaluation_process_tree_stopped",
                    "Evaluation required process-tree termination")
    return forced


def bind_launch_cleanup(row, closure):
    """A confirmed stopped provisional handle may close failed initialization."""
    binding = dict(row["binding"])
    binding["supervision"] = closure["binding"]
    binding["process_pid"] = closure["binding"]["worker"]["pid"]
    return binding
