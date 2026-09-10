"""C2b new lifecycle mechanisms with actual escaped synthetic CPU writers.

No training model/GPU/provider is used. Core DB, D1 stop receipts, native
completion/requeue and shutdown publication stay real. Only known own pidfds
are used for fault injection and inherited fixture cleanup.
"""
import json
import os
from pathlib import Path
import signal
from unittest.mock import Mock

import pytest

from orze.core.execution_attempts import current_attempt
from orze.core.research_artifacts import artifacts_for_attempt
from orze.engine import launcher, lifecycle, shutdown_publication, training_completion
from orze.engine.attempt_effect_lock import AttemptEffectBusy
from orze.engine.attempt_effect_receipts import require_closed_effects
from orze.engine.termination_hold import require_no_unconfirmed_stop, terminate_execution
from test_native_training_tree_completion import (
    cpu_training, native_case, _launch, _alive, _wait_dead,
)


def _state(c, tp):
    return {
        "attempt": current_attempt(c.lake.conn, c.idea, "training"),
        "global": c.lake.get_fsm_state(c.idea),
        "stage": c.lake.get_stage_state(c.idea, "training"),
        "artifacts": artifacts_for_attempt(c.lake.conn, tp.attempt_ref),
        "files": {str(path.relative_to(c.folder)): path.read_bytes()
                  for path in c.folder.rglob("*") if path.is_file()},
    }


@pytest.mark.parametrize("declared", ["completed", "insufficient_vram"])
def test_owned_stop_with_leader_zero_cannot_succeed_or_auto_requeue(cpu_training, tmp_path, declared):
    c = cpu_training
    tp = _launch(c, tmp_path, detached=True)
    if declared == "insufficient_vram":
        (c.folder / "metrics.json").write_text(json.dumps({
            "status": "FAILED", "error": "insufficient_vram: synthetic admission conflict"}))
    assert _alive(c.daemon_pidfd) and tp.process.poll() is None

    assert terminate_execution(tp, c.folder, phase="training",
                               reaper=launcher._terminate_and_reap) == 0

    assert not _alive(c.daemon_pidfd)
    closure = tp.process.closure_receipt()
    assert closure["worker_returncode"] == 0 and closure["stop_requested"] is True
    active, failures = {0: tp}, {}
    assert launcher.check_active(active, c.results, c.cfg, failures, lake=c.lake) == [(c.idea, 0)]
    row = current_attempt(c.lake.conn, c.idea, "training")
    assert row["state"] == "TERMINAL" and row["terminal"]["outcome"] == "failed"
    assert row["terminal"]["reason_code"] == "training_process_tree_stopped"
    assert row["terminal"]["return_code"] == 0
    assert row["terminal"]["process_tree"] == closure
    assert row["terminal"]["artifact_ids"] == []
    assert artifacts_for_attempt(c.lake.conn, tp.attempt_ref) == []
    assert c.lake.get_fsm_state(c.idea) == "FAILED"
    assert c.lake.get_stage_state(c.idea, "training") == "FAILED"
    assert active == {} and failures == {c.idea: 1}
    assert json.loads((c.folder / "claim.json").read_bytes())["attempt_id"] == tp.attempt_id
    terminal = json.loads((c.folder / "_compute_receipts" / tp.attempt_id / "terminal.json").read_bytes())
    assert terminal["outcome"] == "failed" and terminal["return_code"] == 0
    assert terminal["process_pid"] == tp.process.pid != tp.process.supervisor_pid
    assert not any(was_live for _, was_live in c.publications)
    require_no_unconfirmed_stop(c.folder)
    require_closed_effects(c.folder)


def test_actual_training_shutdown_closes_tree_before_interrupted_publication(cpu_training, tmp_path):
    c = cpu_training
    tp = _launch(c, tmp_path, detached=True)
    assert _alive(c.daemon_pidfd)
    stop = Mock(wraps=lifecycle._stop_for_shutdown)

    assert shutdown_publication.handle_shutdown(
        tp, c.results, "training", stop, lake=c.lake, cfg=c.cfg) is True

    stop.assert_called_once_with(tp, c.results, "training")
    assert not _alive(c.daemon_pidfd)
    closure = tp.process.closure_receipt()
    assert closure["stop_requested"] is True and closure["worker_returncode"] == 0
    row = current_attempt(c.lake.conn, c.idea, "training")
    assert row["state"] == "TERMINAL" and row["terminal"]["outcome"] == "interrupted"
    assert row["terminal"]["return_code"] == 0
    assert row["terminal"]["process_tree"] == closure
    assert c.lake.get_fsm_state(c.idea) == "FAILED"
    assert artifacts_for_attempt(c.lake.conn, tp.attempt_ref) == []
    terminal = json.loads((c.folder / "_compute_receipts" / tp.attempt_id / "terminal.json").read_bytes())
    assert terminal["outcome"] == "interrupted" and terminal["return_code"] == 0
    assert terminal["process_pid"] == tp.process.pid != tp.process.supervisor_pid
    assert not any(was_live for _, was_live in c.publications)
    require_no_unconfirmed_stop(c.folder)
    require_closed_effects(c.folder)


def test_killed_own_supervisor_keeps_training_and_live_writer_in_hold(cpu_training, tmp_path):
    c = cpu_training
    tp = _launch(c, tmp_path, detached=True)
    descriptor = os.dup(tp.process.supervisor_pidfd)
    c.pidfds.append((tp.process.supervisor_pid, descriptor))
    before = _state(c, tp)
    assert _alive(descriptor) and _alive(c.daemon_pidfd)
    signal.pidfd_send_signal(descriptor, signal.SIGKILL)
    _wait_dead(descriptor)
    active, failures = {0: tp}, {}

    assert launcher.check_active(active, c.results, c.cfg, failures, lake=c.lake) == []

    assert active.get(0) is tp and tp._termination_unconfirmed is True
    assert _alive(c.daemon_pidfd)
    assert _state(c, tp) == before
    assert c.publications == [] and failures == {}
    assert launcher.check_active(active, c.results, c.cfg, failures, lake=c.lake) == []
    assert active.get(0) is tp and _state(c, tp) == before


def test_direct_vram_requeue_cannot_bypass_live_tree_fence(cpu_training, tmp_path):
    c = cpu_training
    tp = _launch(c, tmp_path, detached=True)
    before = _state(c, tp)
    assert _alive(c.daemon_pidfd) and tp.process.poll() is None

    with pytest.raises(AttemptEffectBusy, match="training_process_tree_unclosed"):
        training_completion.requeue(c.lake, tp, 0, c.folder, c.cfg, 0,
                                    "trainer_vram_precheck")

    assert _alive(c.daemon_pidfd)
    assert _state(c, tp) == before
    assert c.publications == []
