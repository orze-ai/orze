"""C2c new consumer contracts, not missing-API or released-behavior reds.

Delivery and shutdown use real tiny CPU posthoc executions, exact supervision,
SQLite and compute/effect receipts. Slot registration is a separately labelled
dispatcher mechanism with explicit launch/STOP/requeue doubles, not OS proof.
"""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orze.core.execution_attempts import AttemptRef, current_attempt
from orze.engine import launcher, lifecycle, phases, posthoc_completion, training_completion
from orze.engine.attempt_effect_receipts import require_closed_effects
from orze.engine.completion_events import CompletionEvent, require_completion
from orze.engine.shutdown_publication import handle_shutdown
from test_native_posthoc_tree_completion import (
    cpu_posthoc, cpu_training, native_case, _launch_posthoc, _alive,
)
from test_training_launch_termination_handoff import scenario, RefusingActive


@pytest.mark.parametrize("pending", [False, True], ids=["new-completion", "pending-completion"])
def test_actual_posthoc_event_delivers_without_a_second_evaluator(cpu_posthoc, tmp_path, monkeypatch, pending):
    c = cpu_posthoc
    evaluator = tmp_path / "must-not-run-evaluator.py"
    evaluator.write_text("raise AssertionError('posthoc must not launch another evaluator')\n")
    c.cfg["eval_script"] = str(evaluator)
    tp = _launch_posthoc(c, tmp_path, detached=False)
    active, failures = {0: tp}, {}
    completed = launcher.check_active(active, c.results, c.cfg, failures, lake=c.lake)
    assert len(completed) == 1 and isinstance(completed[0], CompletionEvent)
    event = completed[0]
    assert event.attempt_ref == tp.attempt_ref and event.attempt_ref.phase == "posthoc"
    accepted = require_completion(event, c.lake, c.results, phase="posthoc")
    assert accepted["terminal"]["outcome"] == "completed"
    assert not active and failures == {}
    assert tp.process.closure_receipt()["worker_returncode"] == 0
    claim_before = (c.folder / "claim.json").read_bytes()
    metrics_before = (c.folder / "metrics.json").read_bytes()
    history_before = c.lake.get_fsm_history(c.idea)
    runner = SimpleNamespace(cfg=c.cfg, lake=c.lake, results_dir=c.results,
                             active={}, active_evals={}, gpu_ids=[0],
                             pending_evals=[event] if pending else [])
    launch_eval = Mock(side_effect=AssertionError("posthoc completion is not a training source"))
    monkeypatch.setattr(phases, "launch_eval", launch_eval)

    delivered, backlog = phases.OrzePhaseMixin._launch_evals(
        runner, [] if pending else [event], [],
        {c.idea: {"title": "CPU posthoc", "config": c.adapter_config}})

    assert len(delivered) == 1 and delivered[0] is event
    assert delivered[0].attempt_ref.phase == "posthoc"
    assert runner.pending_evals == [] and runner.active_evals == {} and backlog == []
    launch_eval.assert_not_called()
    assert current_attempt(c.lake.conn, c.idea, "training") is None
    assert current_attempt(c.lake.conn, c.idea, "evaluation") is None
    assert current_attempt(c.lake.conn, c.idea, "posthoc") == accepted
    assert c.lake.get_fsm_history(c.idea) == history_before
    assert (c.folder / "claim.json").read_bytes() == claim_before
    assert (c.folder / "metrics.json").read_bytes() == metrics_before


def test_actual_shutdown_stops_posthoc_tree_and_records_its_own_phase(cpu_posthoc, tmp_path):
    c = cpu_posthoc
    tp = _launch_posthoc(c, tmp_path, detached=True)
    assert _alive(c.daemon_pidfd) and tp.process.poll() is None
    before_claim = (c.folder / "claim.json").read_bytes()

    assert handle_shutdown(tp, c.results, "posthoc", lifecycle._stop_for_shutdown,
                           lake=c.lake, cfg=c.cfg) is True

    closure = tp.process.closure_receipt()
    assert not _alive(c.daemon_pidfd)
    assert closure["worker_returncode"] == 0 and closure["stop_requested"] is True
    assert closure["wait_proof"] == "ECHILD_WALL"
    actual = current_attempt(c.lake.conn, c.idea, "posthoc")
    assert actual["attempt_id"] == tp.attempt_id and actual["state"] == "TERMINAL"
    assert actual["terminal"]["outcome"] == "interrupted"
    assert actual["terminal"]["process_tree"] == closure
    assert actual["terminal"]["lifecycle_phase"] == "training"
    assert actual["terminal"]["lifecycle"]["global_state"] == "FAILED"
    compute = json.loads((c.folder / "_compute_receipts" / tp.attempt_id / "terminal.json").read_bytes())
    assert compute["phase"] == "posthoc" and compute["outcome"] == "interrupted"
    assert compute["process_pid"] == tp.process.pid and compute["return_code"] == 0
    assert not any(was_live for _, was_live in c.publications)
    assert current_attempt(c.lake.conn, c.idea, "training") is None
    assert current_attempt(c.lake.conn, c.idea, "evaluation") is None
    assert not (c.folder / "metrics.json").exists()
    assert not (c.folder / "interruption.json").exists(), "posthoc shutdown is not a training-resume checkpoint"
    assert (c.folder / "claim.json").read_bytes() == before_claim
    require_closed_effects(c.folder)


def test_slot_refusal_dispatches_posthoc_requeue_not_training_mechanism(scenario, monkeypatch):
    runner, child, popen, fixer = scenario
    runner.active = RefusingActive()
    captured = []

    def prepared_handle(idea_id, gpu, results_dir, cfg, *, lake):
        assert lake is runner.lake
        claim = json.loads((results_dir / idea_id / "claim.json").read_bytes())
        handle = SimpleNamespace(idea_id=idea_id, gpu=gpu, process=child,
            attempt_id=claim["attempt_id"], is_posthoc=True, close_log=Mock(),
            attempt_ref=AttemptRef(idea_id, "posthoc", claim["attempt_id"], 1))
        captured.append(handle)
        return handle

    launch = Mock(side_effect=prepared_handle)
    stop = Mock(return_value=0)
    posthoc_requeue = Mock()
    training_requeue = Mock(side_effect=AssertionError("wrong attempt adapter"))
    monkeypatch.setattr(phases, "launch", launch)
    monkeypatch.setattr(phases, "terminate_execution", stop)
    monkeypatch.setattr(posthoc_completion, "requeue", posthoc_requeue)
    monkeypatch.setattr(training_completion, "requeue", training_requeue)

    phases.OrzePhaseMixin._launch_training(runner, ["idea-handoff"], True,
        {"idea-handoff": {"title": "Explicit dispatcher double", "config": {"seed": 1}}})

    launch.assert_called_once()
    assert len(captured) == 1
    stop.assert_called_once()
    assert stop.call_args.args[0] is captured[0]
    assert stop.call_args.kwargs["phase"] == "posthoc"
    posthoc_requeue.assert_called_once_with(runner.lake, captured[0], 4,
        runner.results_dir / "idea-handoff", runner.cfg, 0, "scheduler_slot_race")
    training_requeue.assert_not_called()
    captured[0].close_log.assert_called_once()
    popen.assert_not_called()
    fixer.assert_not_called()
    assert runner.failure_counts == {}
