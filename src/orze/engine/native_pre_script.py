"""Claim-bound CPU pre-script actions, closed before training or repair.

One action per exact claim. Confirmed results are historical cached booleans,
not a new process proof. A reset's new claim permits a new action; ambiguity
never authorizes replay, NOT_STARTED, training, or repair. No GPU lease or
compute receipts are used. Inputs are detached, not sandboxed; no adoption.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from orze.core.execution_attempts import (
    AttemptAuthorityError, AttemptRef, _json, create_attempt, current_attempt,
    finish_attempt, mark_running, require_current,
)
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
from orze.engine.execution_authority import (
    canonical_identity_equal as same, execution_transaction,
)
from orze.engine import pre_script_supervision as proof
from orze.engine.supervised_process import SupervisedProcess, SupervisionUncertain
from orze.engine.termination_hold import require_no_unconfirmed_stop, terminate_execution
from orze.engine.training_attempts import _claim, _launch_state, _read, require_catalog


class PreScriptHOLD(AttemptEffectInDoubt):
    """The CPU action cannot safely become True/False or release its claim."""


@dataclass(frozen=True)
class PreScriptResult:
    """Captured CPU outcome and exact source for a later controller action."""
    success: bool
    attempt_ref: AttemptRef

    def __bool__(self):
        return self.success


def _sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _inputs(cmd, timeout, env):
    try:
        if type(timeout) not in (int, float):
            raise ValueError("type")
        budget = float(timeout)
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("range")
        if (type(cmd) is not list or not cmd
                or any(type(arg) is not str or "\0" in arg for arg in cmd)
                or type(env) is not dict
                or any(type(k) is not str or not k or "=" in k or "\0" in k
                       or type(v) is not str or "\0" in v for k, v in env.items())):
            raise ValueError("inputs")
        captured = json.loads(_json({"command": cmd, "environment": env}))
        for key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
                    "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            captured["environment"][key] = ""
        return captured["command"], budget, captured["environment"]
    except (ValueError, TypeError, OverflowError, RecursionError, AttemptAuthorityError) as exc:
        raise PreScriptHOLD("pre_script_inputs_invalid") from exc


def _scope(lake, folder, cfg):
    require_catalog(lake, folder, cfg)
    if lake is not None:
        return
    # Even erased in-memory tokens/routes may not downgrade persisted history.
    from orze.reporting.evidence import report_lifecycle_db_path
    from orze.core.evaluation_retry_state import open_existing_lake
    path = report_lifecycle_db_path(folder.parent, cfg)
    if path.exists() or path.is_symlink():
        opened = None
        try:
            opened = open_existing_lake(path)
            if current_attempt(opened.conn, folder.name, "pre_script") is not None:
                raise PreScriptHOLD("pre_script_catalog_required")
        finally:
            if opened is not None:
                opened.close()


def _claim_binding(lake, folder, gpu):
    value, _ = _read(folder / "claim.json", 8192)
    claim_id = value.get("attempt_id")
    AttemptRef(folder.name, "pre_script", claim_id, 1)
    _, digest = _claim(SimpleNamespace(attempt_id=claim_id, gpu=gpu), folder, lake)
    return {"claim_attempt_id": claim_id, "claim_sha256": digest,
            "launch_lifecycle": _launch_state(lake, folder.name)}


def _owned(lake, folder, ref, binding, gpu, *, states):
    row = require_current(lake.conn, ref, states=states)
    if not same({key: row["binding"].get(key) for key in binding}, binding):
        raise PreScriptHOLD("pre_script_binding_changed")
    if not same(_claim_binding(lake, folder, gpu), {
            key: binding[key] for key in ("claim_attempt_id", "claim_sha256", "launch_lifecycle")}):
        raise PreScriptHOLD("pre_script_claim_changed")
    if row["state"] != "LAUNCHING":
        ready = row["binding"].get("supervision") or {}
        worker = ready.get("worker") if type(ready) is dict else None
        pid = row["binding"].get("process_pid")
        if (type(pid) is not int or type(worker) is not dict
                or type(worker.get("pid")) is not int or pid != worker["pid"]):
            raise PreScriptHOLD("pre_script_process_identity_changed")
    return row


def _cached(row, folder, *, pending=None):
    """Read historical proof; only an owning writer may pass its exact intent."""
    from orze.engine.attempt_effect_receipts import _scan, _decode, _read as read_effect
    ref = AttemptRef(row["task_id"], "pre_script", row["attempt_id"], row["generation"])
    terminal, bound = row["terminal"], row["binding"]
    if row["state"] != "TERMINAL" or type(terminal) is not dict:
        raise PreScriptHOLD("pre_script_result_unconfirmed")
    tree = terminal.get("process_tree")
    ready = bound.get("supervision")
    worker = ready.get("worker") if type(ready) is dict else None
    if (type(bound.get("process_pid")) is not int or type(worker) is not dict
            or type(worker.get("pid")) is not int or bound["process_pid"] != worker["pid"]):
        raise PreScriptHOLD("pre_script_cached_process_identity_changed")
    if (type(tree) is not dict or type(ready) is not dict
            or set(tree) != {"schema", "event", "binding", "worker_returncode",
                "stop_requested", "forced_cleanup", "reaped_children", "wait_proof"}
            or type(tree.get("schema")) is not int or tree["schema"] != 1
            or tree["event"] != "TREE_CLOSED" or tree["wait_proof"] != "ECHILD_WALL"
            or type(tree["reaped_children"]) is not int or tree["reaped_children"] < 1
            or type(tree["stop_requested"]) is not bool or type(tree["forced_cleanup"]) is not bool
            or type(tree["worker_returncode"]) is not int
            or type(terminal.get("return_code")) is not int
            or tree["worker_returncode"] != terminal["return_code"]
            or not same(tree["binding"], ready)
            or ready.get("protocol") != proof.PROTOCOL
            or bound.get("process_supervision_protocol") != proof.PROTOCOL
            or ready.get("command_sha256") != bound.get("command_sha256")
            or not same(ready.get("identity"), {"attempt_ref": asdict(ref), "scope": str(folder)})):
        raise PreScriptHOLD("pre_script_cached_proof_invalid")
    digest = terminal.get("effect_receipt_sha256")
    if _scan(folder, pending=pending).get(ref.attempt_id) != (digest, True):
        raise PreScriptHOLD("pre_script_cached_effect_unconfirmed")
    prepared = _decode(read_effect(folder / "_execution_effects" / ref.attempt_id / "prepared.json"))
    expected = {"operation": "pre_script_terminal", **{
        key: terminal[key] for key in ("outcome", "reason_code", "return_code", "process_tree")}}
    if not same(prepared["plan"], expected):
        raise PreScriptHOLD("pre_script_cached_result_changed")
    success = terminal["outcome"] == "completed"
    if (terminal["outcome"] not in ("completed", "failed", "interrupted")
            or success and (terminal["return_code"] != 0
                or tree["stop_requested"] or tree["forced_cleanup"])):
        raise PreScriptHOLD("pre_script_cached_outcome_invalid")
    return success


def require_launch_ready(lake, idea_dir, cfg):
    """Read-only existing-history gate, safe inside the training writer too."""
    folder = Path(idea_dir).absolute()
    try:
        _scope(lake, folder, cfg)
        require_no_unconfirmed_stop(folder)
        if lake is None:
            return
        row = current_attempt(lake.conn, folder.name, "pre_script")
        if row is None:
            return
        if row["state"] not in ("TERMINAL", "NOT_STARTED"):
            raise PreScriptHOLD("pre_script_previous_action_unclosed")
        claim, _ = _read(folder / "claim.json", 8192)
        if row["binding"].get("claim_attempt_id") == claim.get("attempt_id"):
            ref = AttemptRef(folder.name, "pre_script", row["attempt_id"], row["generation"])
            _owned(lake, folder, ref, row["binding"], claim.get("gpu"), states=("TERMINAL",))
            if not _cached(row, folder):
                raise PreScriptHOLD("pre_script_confirmed_failure")
    except PreScriptHOLD:
        raise
    except Exception as exc:
        raise PreScriptHOLD("pre_script_launch_authority_unconfirmed") from exc


def _admission(lake, folder, cfg):
    from orze.engine import launcher
    _scope(lake, folder, cfg)
    if lake is None:
        raise PreScriptHOLD("pre_script_catalog_required")
    launcher._assert_launch_authorized(folder.name, folder.parent, cfg)
    launcher._assert_controller_runtime_attested(cfg)
    require_no_unconfirmed_stop(folder)


def run_native_pre_script(idea_id, gpu, results_dir, cfg, lake, cmd, timeout, env):
    """Return a confirmed CPU result; possible execution never becomes False."""
    from orze.engine import process as owner
    from orze.engine.claim_authority import _closed
    from orze.engine.execution_catalog import bind_catalog
    cmd, timeout, env = _inputs(cmd, timeout, env)
    folder = Path(results_dir).absolute() / idea_id
    handle, stop_attempted, tree_closed = None, False, False
    try:
        _admission(lake, folder, cfg)
        binding = {"origin": "native_pre_script", "scope": str(folder),
            **_claim_binding(lake, folder, gpu), "command_sha256": _sha(cmd),
            "environment_sha256": _sha(env), "timeout_seconds": timeout,
            "process_supervision_protocol": proof.PROTOCOL}
        action_id = _sha({"phase": "pre_script", "task_id": idea_id,
                          "claim_attempt_id": binding["claim_attempt_id"]})[:32]
        with execution_transaction(lake, folder) as tx:
            _scope(lake, folder, cfg)
            _closed(tx.conn, idea_id)
            previous = current_attempt(tx.conn, idea_id, "pre_script")
            if previous is not None and previous["attempt_id"] == action_id:
                ref = AttemptRef(idea_id, "pre_script", action_id, previous["generation"])
                row = _owned(lake, folder, ref, binding, gpu, states=("TERMINAL",))
                return PreScriptResult(_cached(row, folder), ref)
            bind_catalog(lake, folder, tx.lease)
            ref = create_attempt(tx.conn, idea_id, "pre_script", action_id, binding)
            _owned(lake, folder, ref, binding, gpu, states=("LAUNCHING",))
            tx.watch_attempt(ref)
        handle = SimpleNamespace(idea_id=idea_id, gpu=gpu, attempt_id=action_id,
                                 attempt_ref=ref, process=None)
        _admission(lake, folder, cfg)
        _owned(lake, folder, ref, binding, gpu, states=("LAUNCHING",))
        try:
            handle.process = owner.prepare_supervised(list(cmd),
                identity=proof.identity(handle, folder), env=dict(env),
                stdout=owner.subprocess.DEVNULL, stderr=owner.subprocess.DEVNULL)
        except SupervisionUncertain:
            handle._termination_unconfirmed = True
            raise
        ready = proof.ready_binding(handle, folder)
        if ready["command_sha256"] != binding["command_sha256"]:
            raise PreScriptHOLD("pre_script_command_binding_changed")
        with execution_transaction(lake, folder) as tx:
            _scope(lake, folder, cfg)
            _owned(lake, folder, ref, binding, gpu, states=("LAUNCHING",))
            mark_running(tx.conn, ref, {**binding, "process_pid": handle.process.pid,
                                       "supervision": ready})
            tx.watch_attempt(ref)
        _admission(lake, folder, cfg)
        row = _owned(lake, folder, ref, binding, gpu, states=("RUNNING",))
        proof.bound_binding(handle, row, folder)
        handle.process.start()
        try:
            ret = handle.process.wait(timeout=timeout)
            outcome, reason = ("completed", "pre_script_completed") if ret == 0 else (
                "failed", "pre_script_nonzero")
        except owner.subprocess.TimeoutExpired:
            stop_attempted = True
            ret = terminate_execution(handle, folder, phase="pre_script",
                                      reaper=owner._terminate_and_reap)
            outcome, reason = "interrupted", "pre_script_timeout"
        row = _owned(lake, folder, ref, binding, gpu, states=("RUNNING",))
        closure = proof.require_closed(handle, row, folder, ret)
        tree_closed = True
        outcome, reason = proof.failure_override(closure, (outcome, reason, ""))[:2]
        with execution_transaction(lake, folder) as tx:
            _scope(lake, folder, cfg)
            row = _owned(lake, folder, ref, binding, gpu, states=("RUNNING",))
            if not same(closure, proof.require_closed(handle, row, folder, ret)):
                raise PreScriptHOLD("pre_script_process_tree_changed")
            terminal = {"outcome": outcome, "reason_code": reason,
                        "return_code": ret, "process_tree": closure}
            digest = tx.prepare(ref, {"operation": "pre_script_terminal", **terminal})
            terminal.update(effect_receipt_sha256=digest,
                            launch_lifecycle=binding["launch_lifecycle"])
            if finish_attempt(tx.conn, ref, terminal) != "committed":
                raise PreScriptHOLD("pre_script_terminal_not_new")
            _owned(lake, folder, ref, binding, gpu, states=("TERMINAL",))
            tx.watch_attempt(ref)
        _owned(lake, folder, ref, binding, gpu, states=("TERMINAL",))
        return PreScriptResult(outcome == "completed", ref)
    except BaseException as exc:
        if (handle is not None and handle.process is not None and not tree_closed
                and not stop_attempted and getattr(handle, "_termination_unconfirmed", False) is not True):
            try:
                if not isinstance(handle.process, SupervisedProcess):
                    raise PreScriptHOLD("pre_script_supervised_process_required")
                if handle.process.poll() is None:
                    stop_attempted = True
                    terminate_execution(handle, folder, phase="pre_script", reaper=owner._terminate_and_reap)
            except BaseException as stop_exc:
                raise PreScriptHOLD("pre_script_stop_unconfirmed") from stop_exc
        raise PreScriptHOLD("pre_script_action_unconfirmed") from exc
