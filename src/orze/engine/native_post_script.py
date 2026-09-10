"""Source-bound post-script action intents; never scientific observations.

Commit a dependency-linked LAUNCHING intent before leaving the short guard.
Popen, GPU acquisition and waiting happen outside SQLite/effect locks. Unknown
launch/closure retains its nonterminal attempt, blocking automatic replay and
retry. Normal process exit does not attest surviving descendants or outputs.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from types import SimpleNamespace

from orze.core.execution_attempts import (
    AttemptAuthorityError, create_attempt, current_attempt, finish_attempt,
    mark_running, require_current,
)
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
from orze.engine.completion_events import require_completion
from orze.engine.execution_authority import canonical_identity_equal, execution_transaction
from orze.engine.termination_hold import terminate_execution


def run_native_post_script(source_event, idea_id, gpu, results_dir, cfg, lake,
                           cmd, timeout, log_path, env):
    """Run one action at most once per source ref/command in this catalog."""
    from orze.engine import evaluator
    from orze.engine.native_evaluation import _verify_compute
    folder = results_dir / idea_id
    source_ref = source_event.attempt_ref
    command_sha = hashlib.sha256(json.dumps(cmd, ensure_ascii=False,
        separators=(",", ":")).encode()).hexdigest()
    binding = {"origin": "native_post_script", "source_ref": asdict(source_ref),
               "command_sha256": command_sha, "physical_gpu": gpu}
    action_id = hashlib.sha256(json.dumps({"source_ref": asdict(source_ref),
        "command_sha256": command_sha}, sort_keys=True).encode()).hexdigest()[:32]
    with execution_transaction(lake, folder) as tx:
        require_completion(source_event, lake, results_dir)
        tx.watch_dependency(source_ref)
        previous = current_attempt(tx.conn, idea_id, "post_script")
        if previous is not None and previous["state"] not in ("TERMINAL", "NOT_STARTED"):
            raise AttemptEffectBusy("post_script_previous_action_unclosed")
        if previous is not None:
            old = tx.conn.execute(
                "SELECT state,task_id,phase,binding_json FROM main.execution_attempts "
                "WHERE attempt_id=? COLLATE BINARY",
                (action_id,)).fetchone()
            if old is not None:
                prior_binding = json.loads(old[3])
                if (old[0] not in ("TERMINAL", "NOT_STARTED") or old[1] != idea_id
                        or old[2] != "post_script" or not isinstance(prior_binding, dict)
                        or not canonical_identity_equal(
                            {key: prior_binding.get(key) for key in binding}, binding)):
                    raise AttemptEffectBusy("post_script_action_unclosed")
                return  # Historical accepted action is not a new delivery.
        ref = create_attempt(tx.conn, idea_id, "post_script", action_id, binding)
        tx.watch_attempt(ref)

    handle = None
    try:
        with evaluator.gpu_execution_lease(gpu, require_idle=True) as lease_fds:
            evaluator._verify_gpu_free(gpu, evaluator._launch_min_free_vram(cfg))
            # Recheck after potentially waiting for the GPU. The durable
            # LAUNCHING row prevents authorized retry/reset in this interval.
            require_completion(source_event, lake, results_dir)
            with open(log_path, "w", encoding="utf-8") as log_fh:
                started = time.time()
                process = evaluator.subprocess.Popen(
                    cmd, env=env, stdout=log_fh, stderr=evaluator.subprocess.STDOUT,
                    preexec_fn=evaluator._new_process_group, pass_fds=lease_fds)
                handle = SimpleNamespace(idea_id=idea_id, gpu=gpu, process=process,
                    start_time=started, attempt_id=action_id, attempt_ref=ref)
                with execution_transaction(lake, folder) as tx:
                    require_current(tx.conn, ref, states=("LAUNCHING",))
                    require_completion(source_event, lake, results_dir)
                    tx.watch_dependency(source_ref)
                    receipt = evaluator.record_compute_start(handle, folder, phase="post_script")
                    _verify_compute(folder, receipt, process=handle,
                                    phase="post_script", event="start",
                                    outcome="started")
                    mark_running(tx.conn, ref, {**binding, "process_pid": process.pid})
                    tx.watch_attempt(ref)
        try:
            ret = process.wait(timeout=timeout)
            if type(ret) is not int:
                raise ValueError("post_script_exit_unconfirmed")
            outcome = "completed" if ret == 0 else "failed"
            reason = "post_script_completed" if ret == 0 else "post_script_nonzero"
        except Exception as exc:
            ret = terminate_execution(handle, folder, phase="post_script",
                                      reaper=evaluator._terminate_and_reap)
            outcome = "interrupted" if isinstance(exc, evaluator.subprocess.TimeoutExpired) else "failed"
            reason = "post_script_timeout" if outcome == "interrupted" else "post_script_error"
        with execution_transaction(lake, folder) as tx:
            row = require_current(tx.conn, ref, states=("RUNNING",))
            if (type(row["binding"].get("process_pid")) is not type(process.pid)
                    or row["binding"].get("process_pid") != process.pid):
                raise AttemptAuthorityError("post_script_process_identity_changed")
            tx.watch_dependency(source_ref)
            digest = tx.prepare(ref, {"operation": "post_script_terminal",
                                       "outcome": outcome, "return_code": ret})
            receipt = evaluator.record_compute_terminal(handle, folder, outcome, reason,
                phase="post_script", return_code=ret)
            _verify_compute(folder, receipt, process=handle,
                            phase="post_script", event="terminal", outcome=outcome,
                            reason_code=reason, return_code=ret)
            if finish_attempt(tx.conn, ref, {"outcome": outcome, "reason_code": reason,
                    "return_code": ret, "effect_receipt_sha256": digest}) != "committed":
                raise AttemptAuthorityError("post_script_terminal_not_new")
        return ret
    except BaseException as exc:
        # No callback can infer that a failed Popen/registration had no side
        # effect. The existing LAUNCHING/RUNNING row remains an operational HOLD.
        if handle is not None:
            try:
                if handle.process.poll() is None:
                    terminate_execution(handle, folder, phase="post_script",
                                        reaper=evaluator._terminate_and_reap)
            except BaseException as stop_exc:
                raise AttemptEffectInDoubt("post_script_stop_unconfirmed") from stop_exc
        raise AttemptEffectInDoubt("post_script_action_unconfirmed") from exc
