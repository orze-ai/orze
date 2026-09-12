"""V1-01G1: both evaluator completion entry points honor declared evidence.

These are CPU-only lifecycle/accounting integration tests, not real GPU or
evaluator executions. Only the subprocess and host GPU admission/lease
boundaries are replaced. Launch authorization, source loading, validation,
IdeaLake transitions, and compute receipts remain real.
"""

import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orze.engine import evaluator
from orze.idea_lake import IdeaLake
from orze.reporting.evidence import (
    authoritative_completed_idea_ids,
    qualify_authoritative_report_evidence_with_identity,
)


@pytest.fixture
def project(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    idea_id = "idea-eval-source"
    folder = results / idea_id
    folder.mkdir()
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    lake.insert(idea_id, "generic evaluator", "seed: 7", "", status="queued")
    assert lake.record_state_transition(idea_id, "QUEUED", "CLAIMED")
    assert lake.record_state_transition(idea_id, "CLAIMED", "IN_PROGRESS")
    assert lake.record_stage_transition(
        idea_id, "training", "IN_PROGRESS", "COMPLETE",
        "training_completed_evaluation_pending")
    checkpoint = folder / "checkpoint.pt"
    checkpoint.write_bytes(b"already-completed-training\x00checkpoint\xff")
    cfg = {
        "_project_root": str(tmp_path),
        "results_dir": str(results),
        "idea_lake_db": str(lake.db_path),
        "eval_script": "unused-test-evaluator.py",
        "eval_output": "assessment.json",
        "eval_timeout": 60,
        "report": {
            "primary_metric": "quality_delta",
            "sort": "ascending",
            "columns": [{
                "key": "quality_delta",
                "source": "assessment.json:measurements.quality",
            }],
        },
    }
    p = SimpleNamespace(
        results=results, folder=folder, idea_id=idea_id, lake=lake, cfg=cfg,
        raw={"status": "COMPLETED", "quality_delta": 1.0},
        output={"status": "COMPLETED", "measurements": {"quality": 0.0}},
        exit_code=0, checkpoint_bytes=checkpoint.read_bytes(), emitted=False,
    )

    class FinishedProcess:
        pid = 123456
        returncode = None

        def finish(self):
            if not p.emitted:
                # Files appear only after launch, so the existing-output
                # reconciliation shortcut cannot satisfy these tests.
                assert lake.get_stage_state(idea_id, "evaluation") == "IN_PROGRESS"
                if p.output is not None:
                    (folder / "assessment.json").write_text(
                        json.dumps(p.output), encoding="utf-8")
                p.emitted = True
            self.returncode = p.exit_code
            return self.returncode

        def poll(self):
            return self.finish()

        def wait(self, timeout=None):
            assert timeout == cfg["eval_timeout"]
            return self.finish()

    p.popen = Mock(side_effect=lambda *args, **kwargs: FinishedProcess())
    p.gpu_free = Mock()
    p.gpu_lease = Mock(side_effect=lambda *args, **kwargs: nullcontext(()))
    monkeypatch.setattr(evaluator.subprocess, "Popen", p.popen)
    monkeypatch.setattr(evaluator, "_verify_gpu_free", p.gpu_free)
    monkeypatch.setattr(evaluator, "gpu_execution_lease", p.gpu_lease)
    from supervision_fixture import install
    install(monkeypatch)
    try:
        yield p
    finally:
        lake.close()


def _complete(p, mode):
    metrics_path = p.folder / "metrics.json"
    metrics_path.write_text(json.dumps(p.raw), encoding="utf-8")
    p.training_bytes = metrics_path.read_bytes()
    assert not (p.folder / "assessment.json").exists()
    if mode == "sync":
        evaluator.run_eval(p.idea_id, 0, p.results, p.cfg, lake=p.lake)
    else:
        ep = evaluator.launch_eval(p.idea_id, 0, p.results, p.cfg, lake=p.lake)
        assert ep is not None
        assert p.lake.get_stage_state(p.idea_id, "evaluation") == "IN_PROGRESS"
        active = {0: ep}
        assert evaluator.check_active_evals(
            active, p.results, p.cfg, lake=p.lake) == [(p.idea_id, 0)]
        assert active == {}
    p.popen.assert_called_once()
    p.gpu_free.assert_called_once()
    p.gpu_lease.assert_called_once_with(0, require_idle=True)
    assert p.emitted


def _assert_terminal(p, *, success):
    # Evaluator rejection must never rewrite a successful training artifact
    # or destroy its checkpoint, even when the raw metric itself is invalid.
    assert (p.folder / "metrics.json").read_bytes() == p.training_bytes
    assert (p.folder / "checkpoint.pt").read_bytes() == p.checkpoint_bytes
    assert p.lake.get_stage_state(p.idea_id, "training") == "COMPLETE"
    expected = "COMPLETE" if success else "FAILED"
    assert (p.lake.get_fsm_state(p.idea_id),
            p.lake.get_stage_state(p.idea_id, "evaluation")) == (expected, expected)
    receipt_root = p.folder / "_compute_receipts"
    starts = list(receipt_root.glob("*/start.json"))
    terminals = list(receipt_root.glob("*/terminal.json"))
    assert len(starts) == len(terminals) == 1
    start = json.loads(starts[0].read_text(encoding="utf-8"))
    terminal = json.loads(terminals[0].read_text(encoding="utf-8"))
    assert start["attempt_id"] == terminal["attempt_id"]
    assert start["phase"] == terminal["phase"] == "evaluation"
    assert start["event"] == "start"
    assert terminal["event"] == "terminal"
    assert terminal["outcome"] == ("completed" if success else "failed")
    assert terminal["return_code"] == p.exit_code


def _assert_current_qualified_value(p, value):
    completed, _ = authoritative_completed_idea_ids(p.lake.db_path)
    assert qualify_authoritative_report_evidence_with_identity(
        p.idea_id, p.results, p.cfg, completed)[2] == value


@pytest.mark.parametrize("mode,direction,value", [
    ("async", "ascending", 0.0),
    ("sync", "descending", -2.0),
])
def test_declared_non_domain_source_completes_with_min_max_zero_and_negative(
        project, mode, direction, value):
    p = project
    p.cfg["report"]["sort"] = direction
    p.output["measurements"]["quality"] = value

    _complete(p, mode)

    _assert_terminal(p, success=True)
    _assert_current_qualified_value(p, value)


@pytest.mark.parametrize("mode", ["async", "sync"])
def test_valid_declared_source_overrides_raw_primary_nan(project, mode):
    p = project
    p.raw["quality_delta"] = float("nan")
    p.output["measurements"]["quality"] = -1.0

    _complete(p, mode)

    _assert_terminal(p, success=True)
    _assert_current_qualified_value(p, -1.0)


@pytest.mark.parametrize("mode", ["async", "sync"])
@pytest.mark.parametrize("invalid_source", [
    "nan", "range", "coverage", "source_failed", "missing_declared_source",
])
def test_valid_raw_metrics_cannot_authorize_invalid_declared_source(
        project, mode, invalid_source):
    p = project
    if invalid_source == "nan":
        p.output["measurements"]["quality"] = float("nan")
    elif invalid_source == "range":
        p.cfg["metric_validation"] = {"min_value": {"quality_delta": 0.0}}
        p.output["measurements"]["quality"] = -1.0
    elif invalid_source == "coverage":
        p.cfg["report"]["min_datasets"] = 3
        p.cfg["report"]["dataset_keys"] = ["quality_delta", "fold_a", "fold_b"]
        p.cfg["report"]["columns"].extend([
            {"key": "fold_a", "source": "assessment.json:measurements.fold_a"},
            {"key": "fold_b", "source": "assessment.json:measurements.fold_b"},
        ])
        p.raw.update(fold_a=1.0, fold_b=1.0)
        p.output["measurements"]["fold_a"] = 1.0
    elif invalid_source == "source_failed":
        p.output["status"] = "FAILED"
    else:
        p.output = None

    _complete(p, mode)

    _assert_terminal(p, success=False)


@pytest.mark.parametrize("mode,explicit_metrics_objective", [
    ("async", True), ("sync", False),
])
def test_metrics_only_or_no_objective_does_not_require_default_eval_output(
        project, mode, explicit_metrics_objective):
    p = project
    p.cfg.pop("eval_output")
    p.output = None
    if explicit_metrics_objective:
        p.cfg["report"]["columns"][0]["source"] = "metrics.json:quality_delta"
    else:
        p.cfg.pop("report")

    _complete(p, mode)

    _assert_terminal(p, success=True)
    assert not (p.folder / "eval_report.json").exists()
    assert not (p.folder / "assessment.json").exists()


@pytest.mark.parametrize("mode", ["async", "sync"])
def test_nonzero_evaluator_exit_fails_even_with_valid_source(project, mode):
    p = project
    p.exit_code = 1

    _complete(p, mode)

    _assert_terminal(p, success=False)
