"""Source-bound post-script actions with exact owned-tree completion.

The existing source/command action ID remains the replay key. Input snapshots
and supervision apply only to new execution; historical closed actions are
not rerun or upgraded into new scientific/process proof. Script contents and
arbitrary input paths are not sandboxed. There is no restart adoption here.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from dataclasses import asdict
from types import SimpleNamespace

from orze.core.execution_attempts import (
    AttemptAuthorityError, AttemptRef, _json, create_attempt, current_attempt,
    finish_attempt, mark_running, require_current,
)
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
from orze.engine.completion_events import require_completion
from orze.engine.execution_authority import canonical_identity_equal, execution_transaction
from orze.engine import post_script_supervision as proof
from orze.engine.supervised_process import SupervisedProcess, SupervisionUncertain
from orze.engine.termination_hold import require_no_unconfirmed_stop, terminate_execution


def _inputs(cmd, timeout, env):
    if type(timeout) not in (int, float):
        raise AttemptEffectBusy("post_script_timeout_invalid")
    try:
        budget = float(timeout)
    except (OverflowError, ValueError) as exc:
        raise AttemptEffectBusy("post_script_timeout_invalid") from exc
    if not math.isfinite(budget) or budget <= 0:
        raise AttemptEffectBusy("post_script_timeout_invalid")
    if (type(cmd) is not list or not cmd
            or any(type(item) is not str or "\0" in item for item in cmd)
            or type(env) is not dict
            or any(type(key) is not str or not key or "=" in key or "\0" in key
                   or type(value) is not str or "\0" in value for key, value in env.items())):
        raise AttemptEffectBusy("post_script_inputs_invalid")
    try:
        captured = json.loads(_json({"command": cmd, "environment": env}))
    except (ValueError, TypeError, RecursionError) as exc:
        raise AttemptEffectBusy("post_script_inputs_invalid") from exc
    return captured["command"], budget, captured["environment"]


def _sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _admission(source_event, idea_id, gpu, results_dir, cfg, lake):
    from orze.engine import evaluator
    evaluator._assert_launch_authorized(idea_id, results_dir, cfg)
    evaluator._assert_gpu_authorized(gpu, cfg)
    evaluator._assert_controller_runtime_attested(cfg)
    evaluator._assert_campaign_evidence_authorized(cfg, lake)
    require_no_unconfirmed_stop(results_dir / idea_id)
    require_completion(source_event, lake, results_dir)


def _owned(conn, ref, expected, *, states):
    row = require_current(conn, ref, states=states)
    if not canonical_identity_equal(
            {key: row["binding"].get(key) for key in expected}, expected):
        raise AttemptAuthorityError("post_script_binding_changed")
    return row


def run_native_post_script(source_event, idea_id, gpu, results_dir, cfg, lake,
                           cmd, timeout, log_path, env):
    """Run at most once per source/command; unknown execution remains HOLD."""
    from orze.engine import evaluator
    from orze.engine.native_evaluation import _verify_compute
    cmd, timeout, env = _inputs(cmd, timeout, env)
    results_dir = Path(results_dir).absolute()
    folder = results_dir / idea_id
    source_ref = getattr(source_event, "attempt_ref", None)
    if not isinstance(source_ref, AttemptRef) or source_ref.task_id != idea_id:
        raise AttemptEffectBusy("post_script_source_invalid")
    command_sha = _sha(cmd)
    historical = {"origin": "native_post_script", "source_ref": asdict(source_ref),
                  "command_sha256": command_sha, "physical_gpu": gpu}
    binding = {**historical, "timeout_seconds": timeout,
               "environment_sha256": _sha(env),
               "process_supervision_protocol": proof.PROTOCOL}
    action_id = hashlib.sha256(json.dumps({"source_ref": asdict(source_ref),
        "command_sha256": command_sha}, sort_keys=True).encode()).hexdigest()[:32]
    _admission(source_event, idea_id, gpu, results_dir, cfg, lake)
    with execution_transaction(lake, folder) as tx:
        require_completion(source_event, lake, results_dir)
        tx.watch_dependency(source_ref)
        previous = current_attempt(tx.conn, idea_id, "post_script")
        if previous is not None and previous["state"] not in ("TERMINAL", "NOT_STARTED"):
            raise AttemptEffectBusy("post_script_previous_action_unclosed")
        if previous is not None:
            old = tx.conn.execute(
                "SELECT state,task_id,phase,binding_json FROM main.execution_attempts "
                "WHERE attempt_id=? COLLATE BINARY", (action_id,)).fetchone()
            if old is not None:
                prior_binding = json.loads(old[3])
                if (old[0] not in ("TERMINAL", "NOT_STARTED") or old[1] != idea_id
                        or old[2] != "post_script" or not isinstance(prior_binding, dict)
                        or not canonical_identity_equal(
                            {key: prior_binding.get(key) for key in historical}, historical)):
                    raise AttemptEffectBusy("post_script_action_unclosed")
                # Only preserve old at-most-once delivery, not acceptance of
                # today's new environment/budget or a historical tree proof.
                return
        ref = create_attempt(tx.conn, idea_id, "post_script", action_id, binding)
        tx.watch_attempt(ref)

    handle = SimpleNamespace(idea_id=idea_id, gpu=gpu, process=None,
        start_time=time.time(), attempt_id=action_id, attempt_ref=ref)
    stop_attempted, tree_closed = False, False
    try:
        with open(log_path, "w", encoding="utf-8") as log_fh:
            with evaluator.gpu_execution_lease(gpu, require_idle=True) as lease_fds:
                evaluator._verify_gpu_free(gpu, evaluator._launch_min_free_vram(cfg))
                _admission(source_event, idea_id, gpu, results_dir, cfg, lake)
                _owned(lake.conn, ref, binding, states=("LAUNCHING",))
                try:
                    handle.process = evaluator.prepare_supervised(
                        list(cmd), identity=proof.identity(handle, folder),
                        env=dict(env), stdout=log_fh, stderr=evaluator.subprocess.STDOUT,
                        pass_fds=lease_fds)
                except SupervisionUncertain:
                    handle._termination_unconfirmed = True
                    raise
                handle.start_time = time.time()
                supervision = proof.ready_binding(handle, folder)
                if supervision["command_sha256"] != command_sha:
                    raise AttemptEffectBusy("post_script_command_binding_changed")
                with execution_transaction(lake, folder) as tx:
                    _owned(tx.conn, ref, binding, states=("LAUNCHING",))
                    require_completion(source_event, lake, results_dir)
                    tx.watch_dependency(source_ref)
                    receipt = evaluator.record_compute_start(handle, folder, phase="post_script")
                    _verify_compute(folder, receipt, process=handle, phase="post_script",
                                    event="start", outcome="started")
                    mark_running(tx.conn, ref, {**binding, "process_pid": handle.process.pid,
                                               "supervision": supervision})
                    tx.watch_attempt(ref)
            _admission(source_event, idea_id, gpu, results_dir, cfg, lake)
            row = _owned(lake.conn, ref, binding, states=("RUNNING",))
            proof.bound_binding(handle, row, folder)
            handle.process.start()
            try:
                ret = handle.process.wait(timeout=timeout)
                outcome, reason = ("completed", "post_script_completed") if ret == 0 else (
                    "failed", "post_script_nonzero")
            except evaluator.subprocess.TimeoutExpired:
                stop_attempted = True
                ret = terminate_execution(handle, folder, phase="post_script",
                                          reaper=evaluator._terminate_and_reap)
                outcome, reason = "interrupted", "post_script_timeout"
            row = _owned(lake.conn, ref, binding, states=("RUNNING",))
            closure = proof.require_closed(handle, row, folder, ret)
            tree_closed = True
            override = proof.failure_override(closure, (outcome, reason, ""))
            outcome, reason = override[:2]
            with execution_transaction(lake, folder) as tx:
                row = _owned(tx.conn, ref, binding, states=("RUNNING",))
                require_completion(source_event, lake, results_dir)
                tx.watch_dependency(source_ref)
                if not canonical_identity_equal(closure, proof.require_closed(handle, row, folder, ret)):
                    raise AttemptEffectBusy("post_script_process_tree_changed")
                digest = tx.prepare(ref, {"operation": "post_script_terminal",
                    "outcome": outcome, "return_code": ret, "process_tree": closure})
                receipt = evaluator.record_compute_terminal(handle, folder, outcome, reason,
                    phase="post_script", return_code=ret)
                _verify_compute(folder, receipt, process=handle, phase="post_script",
                    event="terminal", outcome=outcome, reason_code=reason, return_code=ret)
                if finish_attempt(tx.conn, ref, {"outcome": outcome, "reason_code": reason,
                        "return_code": ret, "effect_receipt_sha256": digest,
                        "process_tree": closure}) != "committed":
                    raise AttemptAuthorityError("post_script_terminal_not_new")
        return ret
    except BaseException as exc:
        # Never turn a possible process into NOT_STARTED, or retry an unknown
        # STOP. A returned supervisor owns a tree even after its worker exited.
        if (handle.process is not None and not tree_closed and not stop_attempted
                and getattr(handle, "_termination_unconfirmed", False) is not True):
            try:
                if not isinstance(handle.process, SupervisedProcess):
                    raise AttemptEffectBusy("post_script_supervised_process_required")
                if handle.process.poll() is None:
                    stop_attempted = True
                    terminate_execution(handle, folder, phase="post_script",
                                        reaper=evaluator._terminate_and_reap)
            except BaseException as stop_exc:
                raise AttemptEffectInDoubt("post_script_stop_unconfirmed") from stop_exc
        raise AttemptEffectInDoubt("post_script_action_unconfirmed") from exc
