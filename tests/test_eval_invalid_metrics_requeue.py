"""BUG #1 regression test.

When a finished eval produces INVALID metrics (NaN/inf/out-of-range, per
core.integrity.validate_metrics), check_active_evals must durably fail the
idea by writing the eval_output marker — SYMMETRICALLY with the sealed-file
violation path — so the backlog scanner in engine/phases.py (metrics.json
present + eval_output absent => re-queue) does NOT re-queue it forever.

Root cause being guarded: the metric-invalid branch only called
write_failure_analysis (writes failure_analysis.json, NOT the eval_output
marker), while the sealed-violation branch ALSO calls
_write_eval_failure_marker (writes the eval_output). So invalid-metrics ideas
had no eval_output and were re-queued every cycle.
"""
import json
from pathlib import Path

import pytest

from orze.engine.evaluator import check_active_evals


class _FakeProc:
    """Minimal stand-in for a finished subprocess (returncode 0)."""

    def __init__(self):
        self.returncode = 0

    def poll(self):
        return 0


class _FakeEvalProcess:
    def __init__(self, idea_id, log_path):
        self.idea_id = idea_id
        self.process = _FakeProc()
        self.start_time = 0.0  # long ago -> elapsed is large but ret==0 path
        self.log_path = log_path
        self.timeout = 10_000_000  # never "timed out" (ret is 0 anyway)

    def close_log(self):
        pass


def _setup_idea(results_dir: Path, idea_id: str, metrics: dict):
    d = results_dir / idea_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(metrics))
    (d / "eval_output.log").write_text("eval ran\n")
    return d


EVAL_OUTPUT = "eval_report.json"
CFG = {
    "eval_output": EVAL_OUTPUT,
    # Force validate_metrics to reject NaN/inf (these are the defaults).
    "metric_validation": {"reject_nan": True, "reject_inf": True},
}


def _backlog_would_requeue(idea_dir: Path) -> bool:
    """Mirror engine/phases.py backlog predicate: re-queue when metrics.json
    exists but the eval_output marker does not."""
    return ((idea_dir / "metrics.json").exists()
            and not (idea_dir / EVAL_OUTPUT).exists())


def test_invalid_metrics_writes_eval_marker_and_is_not_requeued(tmp_path):
    idea_id = "idea-bug1"
    # status COMPLETED so validate_metrics actually inspects the values, and
    # a NaN value so it fails.
    idea_dir = _setup_idea(
        tmp_path, idea_id,
        {"status": "COMPLETED", "wer": float("nan")},
    )

    active = {0: _FakeEvalProcess(idea_id, idea_dir / "eval_output.log")}
    finished = check_active_evals(active, tmp_path, CFG)

    assert finished == [(idea_id, 0)]
    # The durable eval_output marker MUST exist (symmetry with sealed path).
    assert (idea_dir / EVAL_OUTPUT).exists(), (
        "invalid-metrics path did not write the eval_output marker; the "
        "backlog scanner will re-queue this idea forever"
    )
    # And therefore the backlog scanner will NOT re-queue it.
    assert _backlog_would_requeue(idea_dir) is False


def test_sealed_path_unaffected_valid_metrics_no_marker(tmp_path):
    """Control: valid metrics with no sealed_files leaves the eval script
    responsible for the report — check_active_evals must NOT fabricate a
    failure marker for a healthy eval."""
    idea_id = "idea-ok"
    idea_dir = _setup_idea(
        tmp_path, idea_id,
        {"status": "COMPLETED", "wer": 5.3},
    )
    active = {0: _FakeEvalProcess(idea_id, idea_dir / "eval_output.log")}
    check_active_evals(active, tmp_path, CFG)
    # No failure marker should be written for a valid result.
    assert not (idea_dir / EVAL_OUTPUT).exists()
