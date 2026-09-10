"""Durable eval-only retry survives coordinator/controller restart.

Real admission, Lake, scheduler, launch, validation, and receipts are used.
Only process/GPU observation boundaries are doubled, with training tripwires.
"""

import json
import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orze.engine import accounting, evaluator, launcher, phases
from orze.engine.evaluation_retry import request_evaluation_retry
from orze.idea_lake import IdeaLake


@pytest.fixture
def project(tmp_path, monkeypatch):
    results = tmp_path / "results"
    folder = results / "idea-retry-execution"
    folder.mkdir(parents=True)
    metrics = b'{"status":"COMPLETED","quality":999}'
    checkpoint = b"successful-training-checkpoint\x00\xff"
    (folder / "metrics.json").write_bytes(metrics)
    (folder / "checkpoint.pt").write_bytes(checkpoint)
    (folder / "assessment.json").write_bytes(b'{"status":"FAILED","quality":999}')
    (folder / "eval_output.log").write_bytes(b"old failed evaluation log\n")
    (folder / "_eval_audit.jsonl").write_bytes(b'{"action":"old-audit"}\n')
    lake = IdeaLake(tmp_path / "ideas.db")
    lake.insert(folder.name, "persisted retry", "seed: 7", "", status="queued")
    assert lake.reconcile_training_complete(folder.name, "reconcile_test_training")
    assert lake.record_stage_transition(
        folder.name, "evaluation", "PENDING", "IN_PROGRESS", "evaluation_launched")
    assert lake.record_state_transition(folder.name, "IN_PROGRESS", "FAILED", "evaluation_failed")
    for phase, attempt, outcome, code in (
            ("training", "training-original", "completed", 0),
            ("evaluation", "evaluation-original", "failed", 1)):
        process = SimpleNamespace(
            idea_id=folder.name, gpu=0, process=SimpleNamespace(pid=None),
            start_time=time.time() - 5, attempt_id=attempt)
        accounting.record_compute_start(process, folder, phase=phase)
        accounting.record_compute_terminal(
            process, folder, outcome, "test_original_attempt", phase=phase, return_code=code)
    old_receipts = {str(path.relative_to(folder)): path.read_bytes()
                    for path in (folder / "_compute_receipts").rglob("*.json")}
    cfg = {
        "_project_root": str(tmp_path), "results_dir": str(results),
        "idea_lake_db": str(lake.db_path), "eval_script": "unused-evaluator.py",
        "eval_output": "assessment.json", "eval_checkpoint": "checkpoint.pt",
        "report": {"primary_metric": "quality", "sort": "ascending",
                   "columns": [{"key": "quality", "source": "assessment.json:quality"}]},
    }
    p = SimpleNamespace(
        results=results, folder=folder, lake=lake, cfg=cfg, idea_id=folder.name,
        metrics=metrics, checkpoint=checkpoint, old_receipts=old_receipts, exit_code=0)

    class EvalProcess:
        pid = None
        returncode = None

        def poll(self):
            if self.returncode is None:
                (folder / "assessment.json").write_text(
                    '{"status":"COMPLETED","quality":0}', encoding="utf-8")
                self.returncode = p.exit_code
            return self.returncode

    p.popen = Mock(side_effect=lambda *args, **kwargs: EvalProcess())
    p.gpu_check = Mock()
    p.lease = Mock(side_effect=lambda *args, **kwargs: nullcontext(()))
    p.training = Mock(side_effect=AssertionError("Retry must never launch training"))
    monkeypatch.setattr(evaluator.subprocess, "Popen", p.popen)
    monkeypatch.setattr(evaluator, "_verify_gpu_free", p.gpu_check)
    monkeypatch.setattr(evaluator, "gpu_execution_lease", p.lease)
    monkeypatch.setattr(phases, "get_gpu_memory_used", lambda _gpu: 0)
    monkeypatch.setattr(phases, "_eval_already_running", lambda *_args: False)
    monkeypatch.setattr(phases, "launch", p.training)
    monkeypatch.setattr(launcher, "launch", p.training)
    try:
        yield p
    finally:
        p.lake.close()


@pytest.mark.parametrize("exit_code,expected", [(0, "COMPLETE"), (1, "FAILED")])
def test_restart_recovers_persisted_retry_without_pending_memory_or_ideas_and_only_evaluates(
        project, exit_code, expected):
    p = project
    p.exit_code = exit_code
    accepted = request_evaluation_retry(p.idea_id, p.results, p.cfg, p.lake)
    assert accepted["status"] == "evaluation_retry_pending"
    db_path = p.lake.db_path
    p.lake.close()
    p.lake = IdeaLake(db_path)
    controller = SimpleNamespace(
        cfg=p.cfg, results_dir=p.results, lake=p.lake, gpu_ids=[0],
        active={}, active_evals={}, pending_evals=[])
    assert phases.launch_eval is evaluator.launch_eval

    delivered, _ = phases.OrzePhaseMixin._launch_evals(controller, [], [], {})

    assert delivered == []
    assert len(controller.active_evals) == 1
    ep = controller.active_evals[0]
    assert ep.idea_id == p.idea_id
    assert ep.attempt_id not in {"training-original", "evaluation-original"}
    assert p.lake.get_stage_state(p.idea_id, "training") == "COMPLETE"
    assert p.lake.get_stage_state(p.idea_id, "evaluation") == "IN_PROGRESS"
    # A second scheduler tick while active must not repeat the durable request.
    assert phases.OrzePhaseMixin._launch_evals(controller, [], [], {})[0] == []
    p.popen.assert_called_once()

    finished = evaluator.check_active_evals(
        controller.active_evals, p.results, p.cfg, lake=p.lake)
    assert finished == [(p.idea_id, 0)]
    assert phases.OrzePhaseMixin._launch_evals(controller, [], finished, {})[0] == finished
    assert phases.OrzePhaseMixin._launch_evals(controller, [], [], {})[0] == []

    p.training.assert_not_called()
    p.popen.assert_called_once()
    p.gpu_check.assert_called_once()
    p.lease.assert_called_once_with(0, require_idle=True)
    assert controller.active_evals == {}
    assert controller.pending_evals == []
    assert p.lake.get_fsm_state(p.idea_id) == expected
    assert p.lake.get_stage_state(p.idea_id, "evaluation") == expected
    assert p.lake.get_stage_state(p.idea_id, "training") == "COMPLETE"
    assert (p.folder / "metrics.json").read_bytes() == p.metrics
    assert (p.folder / "checkpoint.pt").read_bytes() == p.checkpoint
    assert (p.folder / "_eval_audit.jsonl").read_bytes().startswith(b'{"action":"old-audit"}\n')
    for relative, payload in p.old_receipts.items():
        assert (p.folder / relative).read_bytes() == payload
    receipt_root = p.folder / "_compute_receipts"
    assert {path.name for path in receipt_root.iterdir()} == {
        "training-original", "evaluation-original", ep.attempt_id}
    start = json.loads((receipt_root / ep.attempt_id / "start.json").read_text())
    terminal = json.loads((receipt_root / ep.attempt_id / "terminal.json").read_text())
    assert start["phase"] == terminal["phase"] == "evaluation"
    assert start["attempt_id"] == terminal["attempt_id"] == ep.attempt_id
    assert terminal["outcome"] == ("completed" if exit_code == 0 else "failed")
    assert terminal["return_code"] == exit_code
