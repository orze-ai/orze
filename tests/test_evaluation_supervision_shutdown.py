"""C2a new shutdown mechanisms with real supervised CPU evaluator trees.

Native DB/receipts/publication and the lifecycle shutdown stop callback are
real. Only training/GPU setup is simulated by the imported source fixture.
These new protocol requirements are not additional historical behavior reds.
"""

import hashlib
import json
import os
from pathlib import Path
import signal
from unittest.mock import Mock

import pytest

from orze.core.execution_attempts import current_attempt
from orze.core.research_artifacts import artifacts_for_attempt
from orze.core.research_observations import observations_for_attempt
from orze.engine import evaluator, lifecycle, shutdown_publication
from orze.engine.attempt_effect_receipts import require_closed_effects
from orze.engine.termination_hold import TerminationUnconfirmed, require_no_unconfirmed_stop
from test_native_eval_tree_completion import (
    cpu_project, b2_project, artifact_project, native_case, _launch, _alive, _wait_dead,
)


def _state(c, ep):
    return {
        "attempt": current_attempt(c.lake.conn, c.idea, "evaluation"),
        "global": c.lake.get_fsm_state(c.idea),
        "stage": c.lake.get_stage_state(c.idea, "evaluation"),
        "artifacts": artifacts_for_attempt(c.lake.conn, ep.attempt_ref),
        "observations": observations_for_attempt(c.lake.conn, ep.attempt_ref),
        "files": {str(path.relative_to(c.folder)): path.read_bytes()
                  for path in c.folder.rglob("*") if path.is_file()},
    }


def test_actual_shutdown_callback_closes_detached_writer_before_interrupted_terminal(cpu_project, tmp_path):
    c = cpu_project
    ep, _ = _launch(c, tmp_path, detached=True)
    assert _alive(c.daemon_pidfd) and ep.process.poll() is None
    stop = Mock(wraps=lifecycle._stop_for_shutdown)

    assert shutdown_publication.handle_shutdown(
        ep, c.results, "evaluation", stop, lake=c.lake, cfg=c.cfg) is True

    stop.assert_called_once_with(ep, c.results, "evaluation")
    assert not _alive(c.daemon_pidfd)
    assert ep.process.poll() == 0
    closure = ep.process.closure_receipt()
    assert closure["event"] == "TREE_CLOSED"
    assert closure["stop_requested"] is True
    assert type(closure["forced_cleanup"]) is bool  # Escalation to SIGKILL, not any STOP.
    assert closure["wait_proof"] == "ECHILD_WALL" and closure["reaped_children"] >= 2
    row = current_attempt(c.lake.conn, c.idea, "evaluation")
    assert row["state"] == "TERMINAL"
    assert row["terminal"]["outcome"] == "interrupted"
    assert row["terminal"]["reason_code"] == "evaluation_controller_shutdown"
    assert row["terminal"]["return_code"] == 0
    assert row["terminal"]["process_tree"] == closure
    assert c.lake.get_fsm_state(c.idea) == "FAILED"
    assert c.lake.get_stage_state(c.idea, "evaluation") == "FAILED"
    assert c.lake.get_stage_state(c.idea, "training") == "COMPLETE"
    assert artifacts_for_attempt(c.lake.conn, ep.attempt_ref) == []
    assert observations_for_attempt(c.lake.conn, ep.attempt_ref) == []
    terminal = json.loads((c.folder / "_compute_receipts" / ep.attempt_id / "terminal.json").read_bytes())
    assert terminal["outcome"] == "interrupted" and terminal["return_code"] == 0
    assert terminal["process_pid"] == ep.process.pid != ep.process.supervisor_pid
    stop_dir = c.folder / "_execution_stops" / ep.attempt_id
    requested = (stop_dir / "requested.json").read_bytes()
    confirmed = json.loads((stop_dir / "confirmed.json").read_bytes())
    assert confirmed["tree_stopped"] is True and confirmed["return_code"] == 0
    assert confirmed["request_sha256"] == hashlib.sha256(requested).hexdigest()
    require_no_unconfirmed_stop(c.folder)
    require_closed_effects(c.folder)
    assert not any(was_live for _, was_live in c.publications)
    assert (c.folder / "metrics.json").read_bytes() == c.metrics_before


def test_shutdown_rejects_changed_persistent_nonce_before_any_stop_call(cpu_project, tmp_path):
    c = cpu_project
    ep, _ = _launch(c, tmp_path, detached=True)
    row = current_attempt(c.lake.conn, c.idea, "evaluation")
    binding = row["binding"]
    old_nonce = binding["supervision"]["nonce_sha256"]
    binding["supervision"]["nonce_sha256"] = ("a" if old_nonce != "a" * 64 else "b") * 64
    c.lake.conn.execute("UPDATE main.execution_attempts SET binding_json=? WHERE attempt_id=? COLLATE BINARY",
                        (json.dumps(binding, sort_keys=True, separators=(",", ":")), ep.attempt_id))
    c.lake.conn.commit()
    before = _state(c, ep)
    stop = Mock(wraps=lifecycle._stop_for_shutdown)

    assert shutdown_publication.handle_shutdown(
        ep, c.results, "evaluation", stop, lake=c.lake, cfg=c.cfg) is False

    stop.assert_not_called()
    assert _alive(c.daemon_pidfd)
    assert _state(c, ep) == before
    assert c.publications == []


def test_killed_own_supervisor_keeps_live_writer_and_active_native_attempt_in_hold(cpu_project, tmp_path):
    c = cpu_project
    ep, _ = _launch(c, tmp_path, detached=True)
    supervisor_fd = os.dup(ep.process.supervisor_pidfd)
    c.cpu_pidfds.append((ep.process.supervisor_pid, supervisor_fd))
    before = _state(c, ep)
    assert _alive(supervisor_fd) and _alive(c.daemon_pidfd)
    # Only the already captured pidfd of this fixture's own supervisor is used.
    signal.pidfd_send_signal(supervisor_fd, signal.SIGKILL)
    _wait_dead(supervisor_fd)
    active = {0: ep}

    with pytest.raises(TerminationUnconfirmed):
        evaluator.check_active_evals(active, c.results, c.cfg, lake=c.lake)

    assert active.get(0) is ep and ep._termination_unconfirmed is True
    assert _alive(c.daemon_pidfd)
    assert _state(c, ep) == before
    assert c.publications == []
    assert not (c.folder / "_compute_receipts" / ep.attempt_id / "terminal.json").exists()
