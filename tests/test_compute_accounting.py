"""Framework-owned compute receipts are immutable and non-sensitive."""

import json
import stat
import sys
import time

import pytest

from orze.engine.accounting import (
    ComputeAccountingError,
    record_compute_start,
    record_compute_terminal,
    record_zero_gpu_outcome,
    summarize_compute_receipts,
)
from orze.engine.evaluator import check_active_evals, launch_eval
from orze.engine.failure import _reset_idea_for_retry
from orze.engine.launcher import check_active, launch
from orze.engine.process import TrainingProcess
from orze.engine.process import run_pre_script
from orze.engine.resume import write_interruption_receipt
from orze.engine.scheduler import claim


class FinishedProcess:
    def __init__(self, return_code=0, pid=12345):
        self.return_code = return_code
        self.pid = pid

    def poll(self):
        return self.return_code

    def wait(self, timeout=None):
        return self.return_code


def _tp(tmp_path, *, attempt_id="a" * 32, return_code=0):
    return TrainingProcess(
        idea_id="idea-accounting",
        gpu=4,
        process=FinishedProcess(return_code),
        start_time=time.time() - 5,
        log_path=tmp_path / "train.log",
        timeout=60,
        attempt_id=attempt_id,
    )


def test_receipts_are_private_idempotent_and_terminal_immutable(tmp_path):
    idea_dir = tmp_path / "idea-accounting"
    tp = _tp(tmp_path)

    start = record_compute_start(tp, idea_dir)
    terminal = record_compute_terminal(
        tp, idea_dir, "completed", "trainer_completed", return_code=0)
    repeated = record_compute_terminal(
        tp, idea_dir, "completed", "trainer_completed", return_code=0)

    assert repeated == terminal
    assert terminal["allocated_gpu_seconds"] >= 5
    assert terminal["physical_gpu"] == 4
    assert set(terminal) == {
        "schema_version", "idea_id", "attempt_id", "phase", "event",
        "outcome", "physical_gpu", "started_at_epoch", "finished_at",
        "allocated_gpu_seconds", "return_code", "reason_code", "process_pid",
    }
    assert start["outcome"] == "started"
    terminal_path = (
        idea_dir / "_compute_receipts" / tp.attempt_id / "terminal.json")
    assert stat.S_IMODE(terminal_path.stat().st_mode) == 0o600
    with pytest.raises(ComputeAccountingError, match="conflicting_receipt"):
        record_compute_terminal(
            tp, idea_dir, "failed", "late_conflict", return_code=1)


@pytest.mark.parametrize("reason", ["contains a space", "secret/token", ""])
def test_receipt_reason_is_a_closed_non_payload_token(tmp_path, reason):
    with pytest.raises(ComputeAccountingError, match="reason_code_invalid"):
        record_compute_terminal(
            _tp(tmp_path), tmp_path / "idea-accounting",
            "failed", reason, return_code=1)


def test_claim_attempt_id_reaches_launch_start_receipt(tmp_path, monkeypatch):
    results = tmp_path / "results"
    train = tmp_path / "train.py"
    train.write_text("# trainer\n", encoding="utf-8")
    base = tmp_path / "base.yaml"
    base.write_text("{}\n", encoding="utf-8")
    ideas = tmp_path / "ideas.md"
    ideas.write_text("", encoding="utf-8")
    assert claim("idea-accounting", results, 4)
    monkeypatch.setattr(
        "orze.engine.launcher._verify_gpu_free", lambda *args: None)
    monkeypatch.setattr(
        "orze.engine.launcher.capture_process_identity",
        lambda pid: {"pid": pid, "pgid": pid, "start_ticks": 99},
    )
    monkeypatch.setattr(
        "orze.engine.launcher.subprocess.Popen",
        lambda *args, **kwargs: FinishedProcess(return_code=None),
    )

    tp = launch("idea-accounting", 4, results, {
        "train_script": str(train),
        "base_config": str(base),
        "ideas_file": str(ideas),
        "python": "python3",
    })

    claim_data = json.loads(
        (results / "idea-accounting" / "claim.json").read_text())
    start = json.loads((
        results / "idea-accounting" / "_compute_receipts"
        / tp.attempt_id / "start.json").read_text())
    assert tp.attempt_id == claim_data["attempt_id"]
    assert start["attempt_id"] == claim_data["attempt_id"]
    assert start["physical_gpu"] == 4
    tp.close_log()


def test_claimed_prelaunch_rejection_records_exactly_zero_gpu_time(tmp_path):
    results = tmp_path / "results"
    assert claim("idea-accounting", results, 4)
    claim_data = json.loads(
        (results / "idea-accounting" / "claim.json").read_text())

    terminal = record_zero_gpu_outcome(
        "idea-accounting",
        results / "idea-accounting",
        4,
        "rejected",
        "artifact_preflight_failed",
    )

    assert terminal["attempt_id"] == claim_data["attempt_id"]
    assert terminal["phase"] == "admission"
    assert terminal["outcome"] == "rejected"
    assert terminal["allocated_gpu_seconds"] == 0.0
    assert terminal["process_pid"] is None


def test_zero_gpu_receipt_cannot_overwrite_allocated_attempt(tmp_path):
    results = tmp_path / "results"
    assert claim("idea-accounting", results, 4)
    claim_data = json.loads(
        (results / "idea-accounting" / "claim.json").read_text())
    tp = _tp(tmp_path, attempt_id=claim_data["attempt_id"])
    record_compute_start(tp, results / "idea-accounting")

    with pytest.raises(
            ComputeAccountingError, match="attempt_already_allocated"):
        record_zero_gpu_outcome(
            "idea-accounting",
            results / "idea-accounting",
            4,
            "requeued",
            "gpu_unavailable_before_launch",
        )


def test_zero_gpu_receipt_rejects_claimed_device_mismatch(tmp_path):
    results = tmp_path / "results"
    assert claim("idea-accounting", results, 4)

    with pytest.raises(
            ComputeAccountingError, match="physical_gpu_claim_mismatch"):
        record_zero_gpu_outcome(
            "idea-accounting",
            results / "idea-accounting",
            5,
            "rejected",
            "artifact_preflight_failed",
        )


def test_immediate_retry_renews_attempt_and_archives_old_claim(tmp_path):
    results = tmp_path / "results"
    idea_dir = results / "idea-accounting"
    assert claim("idea-accounting", results, 4)
    old_claim = json.loads((idea_dir / "claim.json").read_text())
    old_tp = _tp(tmp_path, attempt_id=old_claim["attempt_id"])
    record_compute_start(old_tp, idea_dir)
    record_compute_terminal(
        old_tp, idea_dir, "failed", "trainer_declared_failed",
        return_code=1,
    )

    _reset_idea_for_retry(idea_dir)

    new_claim = json.loads((idea_dir / "claim.json").read_text())
    archived = list(idea_dir.glob("claim.retry.*.json"))
    assert new_claim["attempt_id"] != old_claim["attempt_id"]
    assert len(archived) == 1
    assert json.loads(archived[0].read_text())["attempt_id"] == old_claim[
        "attempt_id"
    ]


def test_compute_summary_uses_receipts_and_reports_invalid_evidence(tmp_path):
    results = tmp_path / "results"
    completed_dir = results / "idea-accounting"
    completed = _tp(tmp_path, attempt_id="b" * 32)
    record_compute_start(completed, completed_dir)
    record_compute_terminal(
        completed, completed_dir, "completed", "trainer_completed",
        return_code=0,
    )
    rejected_dir = results / "idea-rejected"
    assert claim("idea-rejected", results, 5)
    record_zero_gpu_outcome(
        "idea-rejected", rejected_dir, 5, "rejected",
        "method_validator_rejected",
    )
    invalid = (
        results / "idea-invalid" / "_compute_receipts" / ("c" * 32)
        / "terminal.json"
    )
    invalid.parent.mkdir(parents=True)
    invalid.write_text('{"schema_version": 1, "idea_id": "leaked payload"}')

    summary = summarize_compute_receipts(results)

    assert summary["attempts_started"] == 1
    assert summary["attempts_terminal"] == 2
    assert summary["incomplete_started_attempts"] == 0
    assert summary["zero_gpu_terminal_attempts"] == 1
    assert summary["allocated_gpu_seconds_total"] >= 5
    assert summary["invalid_receipts"] == 1
    assert summary["by_phase"]["training"]["outcomes"] == {
        "completed": 1
    }
    assert summary["by_phase"]["admission"]["allocated_gpu_seconds"] == 0.0


def test_pre_script_allocation_has_its_own_receipt(tmp_path):
    results = tmp_path / "results"
    idea_dir = results / "idea-accounting"
    assert claim("idea-accounting", results, 4)
    claim_attempt = json.loads((idea_dir / "claim.json").read_text())[
        "attempt_id"
    ]
    script = tmp_path / "prepare.py"
    script.write_text("raise SystemExit(0)\n", encoding="utf-8")

    assert run_pre_script("idea-accounting", 4, {
        "pre_script": str(script),
        "python": sys.executable,
        "pre_timeout": 5,
    }, results)

    receipts = list((idea_dir / "_compute_receipts").glob("*"))
    assert len(receipts) == 1
    assert receipts[0].name != claim_attempt
    start = json.loads((receipts[0] / "start.json").read_text())
    terminal = json.loads((receipts[0] / "terminal.json").read_text())
    assert start["phase"] == "pre_script"
    assert terminal["outcome"] == "completed"
    assert terminal["reason_code"] == "pre_script_completed"


def test_completed_process_gets_framework_terminal_receipt(tmp_path):
    results = tmp_path / "results"
    idea_dir = results / "idea-accounting"
    idea_dir.mkdir(parents=True)
    (idea_dir / "metrics.json").write_text(
        json.dumps({"status": "COMPLETED", "training_time": 999999}),
        encoding="utf-8",
    )
    tp = _tp(tmp_path)
    record_compute_start(tp, idea_dir)

    assert check_active(
        {4: tp}, results, {"sops": {"failure_feedback": False}}, {}
    ) == [("idea-accounting", 4)]

    terminal = json.loads((
        idea_dir / "_compute_receipts" / tp.attempt_id / "terminal.json"
    ).read_text())
    assert terminal["outcome"] == "completed"
    assert terminal["reason_code"] == "trainer_completed"
    assert terminal["allocated_gpu_seconds"] < 60


def test_interruption_ledger_does_not_copy_free_form_reason(tmp_path):
    results = tmp_path / "results"
    idea_dir = results / "idea-accounting"
    idea_dir.mkdir(parents=True)
    tp = _tp(tmp_path)
    record_compute_start(tp, idea_dir)

    write_interruption_receipt(
        tp, results, {"resume": {"enabled": False}},
        reason="credential secret value", terminating_signal="SIGTERM",
        return_code=-15,
    )

    terminal_text = (
        idea_dir / "_compute_receipts" / tp.attempt_id / "terminal.json"
    ).read_text()
    terminal = json.loads(terminal_text)
    assert terminal["outcome"] == "interrupted"
    assert terminal["reason_code"] == "interruption_other"
    assert "credential" not in terminal_text
    assert "secret value" not in terminal_text


def test_evaluation_has_separate_validated_compute_receipt(
        tmp_path, monkeypatch):
    results = tmp_path / "results"
    idea_dir = results / "idea-accounting"
    idea_dir.mkdir(parents=True)
    (idea_dir / "metrics.json").write_text(json.dumps({
        "status": "COMPLETED",
        "score": 0.5,
    }), encoding="utf-8")
    monkeypatch.setattr(
        "orze.engine.evaluator._verify_gpu_free", lambda *args: None)
    monkeypatch.setattr(
        "orze.engine.evaluator.subprocess.Popen",
        lambda *args, **kwargs: FinishedProcess(0),
    )
    cfg = {"eval_script": "/usr/bin/true"}

    ep = launch_eval("idea-accounting", 4, results, cfg)
    assert ep is not None
    assert check_active_evals({4: ep}, results, cfg) == [
        ("idea-accounting", 4)
    ]

    receipt_dir = idea_dir / "_compute_receipts" / ep.attempt_id
    start = json.loads((receipt_dir / "start.json").read_text())
    terminal = json.loads((receipt_dir / "terminal.json").read_text())
    assert start["phase"] == "evaluation"
    assert terminal["phase"] == "evaluation"
    assert terminal["outcome"] == "completed"
    assert terminal["reason_code"] == "evaluation_validated"
