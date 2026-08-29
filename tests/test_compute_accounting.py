"""Framework-owned compute receipts are immutable and non-sensitive."""

import json
import stat
import subprocess
import sys
import time

import pytest

from orze.engine.accounting import (
    audit_campaign_compute_receipts,
    ComputeAccountingError,
    finalize_failed_launch_accounting,
    record_compute_start,
    record_compute_terminal,
    record_zero_gpu_outcome,
    summarize_compute_receipts,
)
from orze.engine.evaluator import check_active_evals, launch_eval
from orze.engine.failure import _reset_idea_for_retry
from orze.engine.launcher import check_active, launch
from orze.engine.process import TrainingProcess
from orze.engine.process import EvalProcess
from orze.engine.process import run_pre_script
from orze.engine.resume import write_interruption_receipt
from orze.engine.scheduler import claim
from orze.idea_lake import IdeaLake


class FinishedProcess:
    def __init__(self, return_code=0, pid=12345):
        self.return_code = return_code
        self.pid = pid

    def poll(self):
        return self.return_code

    def wait(self, timeout=None):
        return self.return_code


class InterruptibleProcess(FinishedProcess):
    def __init__(self, pid=12346):
        super().__init__(return_code=None, pid=pid)

    def wait(self, timeout=None):
        self.return_code = -15
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
    receipt_dir = completed_dir / "_compute_receipts" / completed.attempt_id
    (receipt_dir / "boundary.json").write_text(
        '{"separately_validated":"model_lineage"}', encoding="utf-8")
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


def test_pre_script_is_cpu_only_and_creates_no_compute_receipt(
        tmp_path, monkeypatch):
    results = tmp_path / "results"
    idea_dir = results / "idea-accounting"
    assert claim("idea-accounting", results, 4)
    claim_attempt = json.loads((idea_dir / "claim.json").read_text())[
        "attempt_id"
    ]
    script = tmp_path / "prepare.py"
    script.write_text("raise SystemExit(0)\n", encoding="utf-8")

    observed = {}
    real_popen = subprocess.Popen

    def capture_popen(*args, **kwargs):
        observed["cmd"] = args[0]
        observed["env"] = kwargs["env"]
        observed["pass_fds"] = kwargs.get("pass_fds")
        return real_popen(*args, **kwargs)

    monkeypatch.setattr("orze.engine.process.subprocess.Popen", capture_popen)
    assert run_pre_script("idea-accounting", 4, {
        "pre_script": str(script),
        "python": sys.executable,
        "pre_timeout": 5,
        "pre_args": ["--gpu", "{gpu}"],
        "train_extra_env": {"CUDA_VISIBLE_DEVICES": "4"},
    }, results)

    receipts = list((idea_dir / "_compute_receipts").glob("*"))
    assert receipts == []
    assert claim_attempt
    assert observed["cmd"][-2:] == ["--gpu", "-1"]
    assert observed["pass_fds"] is None
    for key in (
            "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        assert observed["env"][key] == ""


def test_pre_script_does_not_observe_or_validate_gpu_scope(tmp_path):
    script = tmp_path / "prepare.py"
    script.write_text("raise SystemExit(0)\n", encoding="utf-8")
    assert run_pre_script("idea-scope", 0, {
        "pre_script": str(script),
        "python": sys.executable,
        "gpu_scheduling": {
            "allowed_gpus": [4, 5, 6, 7],
            "reserved_gpus": [0, 1, 2, 3],
        },
        "_managed_gpu_ids": [4, 5, 6, 7],
    }, tmp_path)


def test_final_launch_failure_is_zero_gpu_only_without_allocation(tmp_path):
    results = tmp_path / "results"
    assert claim("idea-launch-failure", results, 4)

    terminal = finalize_failed_launch_accounting(
        "idea-launch-failure", results / "idea-launch-failure", 4,
        "launch_failed",
    )

    assert terminal["phase"] == "admission"
    assert terminal["allocated_gpu_seconds"] == 0.0
    assert terminal["reason_code"] == "launch_failed"


def test_final_launch_failure_preserves_paired_gpu_allocation(tmp_path):
    results = tmp_path / "results"
    assert claim("idea-launch-paired", results, 4)
    claim_data = json.loads(
        (results / "idea-launch-paired" / "claim.json").read_text())
    tp = _tp(tmp_path, attempt_id=claim_data["attempt_id"])
    tp.idea_id = "idea-launch-paired"
    tp.gpu = 4
    idea_dir = results / tp.idea_id
    record_compute_start(tp, idea_dir)
    expected = record_compute_terminal(
        tp, idea_dir, "failed", "training_launch_initialization_failed",
        return_code=-15,
    )

    observed = finalize_failed_launch_accounting(
        tp.idea_id, idea_dir, 4, "launch_failed",
    )

    assert observed == expected
    assert observed["phase"] == "training"
    assert observed["allocated_gpu_seconds"] >= 0.0


@pytest.mark.parametrize("missing", ["start", "terminal"])
def test_final_launch_failure_rejects_half_written_allocation(
        tmp_path, missing):
    results = tmp_path / "results"
    assert claim("idea-launch-incomplete", results, 4)
    claim_data = json.loads(
        (results / "idea-launch-incomplete" / "claim.json").read_text())
    tp = _tp(tmp_path, attempt_id=claim_data["attempt_id"])
    tp.idea_id = "idea-launch-incomplete"
    tp.gpu = 4
    idea_dir = results / tp.idea_id
    record_compute_start(tp, idea_dir)
    record_compute_terminal(
        tp, idea_dir, "failed", "training_launch_initialization_failed",
        return_code=-15,
    )
    (idea_dir / "_compute_receipts" / tp.attempt_id
     / f"{missing}.json").unlink()

    with pytest.raises(
            ComputeAccountingError,
            match="failed_launch_receipt_pair_incomplete"):
        finalize_failed_launch_accounting(
            tp.idea_id, idea_dir, 4, "launch_failed",
        )


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


def test_controller_shutdown_closes_eval_compute_and_requeues_stage(
        tmp_path, monkeypatch):
    from orze.engine import lifecycle

    results = tmp_path / "results"
    idea_id = "idea-eval-shutdown"
    idea_dir = results / idea_id
    idea_dir.mkdir(parents=True)
    lake_path = tmp_path / "lake.db"
    lake = IdeaLake(str(lake_path))
    lake.insert(idea_id, "eval shutdown", "{}", "", status="queued")
    assert lake.record_state_transition(idea_id, "QUEUED", "CLAIMED")
    assert lake.record_state_transition(idea_id, "CLAIMED", "IN_PROGRESS")
    assert lake.record_stage_transition(
        idea_id, "training", "IN_PROGRESS", "COMPLETE", "trained")
    assert lake.record_stage_transition(
        idea_id, "evaluation", "PENDING", "IN_PROGRESS", "evaluating")

    process = InterruptibleProcess()
    ep = EvalProcess(
        idea_id=idea_id,
        gpu=4,
        process=process,
        start_time=time.time() - 5,
        log_path=idea_dir / "eval.log",
        timeout=60,
        attempt_id="d" * 32,
    )
    record_compute_start(ep, idea_dir, phase="evaluation")
    active_evals = {4: ep}
    signals = []
    monkeypatch.setattr(
        lifecycle, "_kill_pg", lambda proc, sig: signals.append(sig))
    monkeypatch.setattr(lifecycle, "save_state", lambda *args: None)
    monkeypatch.setattr(lifecycle, "notify", lambda *args: None)

    lifecycle.graceful_shutdown(
        results, {}, {}, active_evals, {}, 1, {}, lake,
        "host", "instance", kill_all=False,
    )

    terminal = json.loads((
        idea_dir / "_compute_receipts" / ep.attempt_id / "terminal.json"
    ).read_text())
    reopened = IdeaLake(str(lake_path))
    try:
        assert terminal["phase"] == "evaluation"
        assert terminal["outcome"] == "interrupted"
        assert terminal["reason_code"] == "evaluation_controller_shutdown"
        assert reopened.get_stage_state(idea_id, "evaluation") == "PENDING"
        assert reopened.get_fsm_state(idea_id) == "IN_PROGRESS"
        assert active_evals == {}
        assert signals
    finally:
        reopened.close()


def test_atexit_closes_every_tracked_gpu_allocation(tmp_path, monkeypatch):
    from orze.engine import lifecycle

    results = tmp_path / "results"
    training_dir = results / "idea-atexit-training"
    evaluation_dir = results / "idea-atexit-evaluation"
    training_dir.mkdir(parents=True)
    evaluation_dir.mkdir(parents=True)
    training = _tp(tmp_path, attempt_id="a" * 32)
    training.idea_id = training_dir.name
    evaluation = EvalProcess(
        idea_id=evaluation_dir.name,
        gpu=5,
        process=FinishedProcess(return_code=None),
        start_time=time.time() - 2,
        log_path=evaluation_dir / "eval.log",
        timeout=60,
        attempt_id="b" * 32,
    )
    record_compute_start(training, training_dir, phase="training")
    record_compute_start(evaluation, evaluation_dir, phase="evaluation")

    def kill(process, _signal):
        process.return_code = -9

    monkeypatch.setattr(lifecycle, "_kill_pg", kill)
    lifecycle.atexit_cleanup(
        {4: training}, {5: evaluation}, {}, results,
    )

    training_terminal = json.loads((
        training_dir / "_compute_receipts" / training.attempt_id
        / "terminal.json"
    ).read_text())
    evaluation_terminal = json.loads((
        evaluation_dir / "_compute_receipts" / evaluation.attempt_id
        / "terminal.json"
    ).read_text())
    assert training_terminal["outcome"] == "interrupted"
    assert training_terminal["reason_code"] == "training_atexit_cleanup"
    assert evaluation_terminal["outcome"] == "interrupted"
    assert evaluation_terminal["reason_code"] == "evaluation_atexit_cleanup"


def test_campaign_compute_audit_verifies_closed_scoped_exact_ideas(tmp_path):
    results = tmp_path / "results"
    now = time.time()
    trained = _tp(tmp_path, attempt_id="e" * 32)
    trained.start_time = now - 5
    trained_dir = results / trained.idea_id
    record_compute_start(trained, trained_dir)
    record_compute_terminal(
        trained, trained_dir, "completed", "trainer_completed", return_code=0
    )
    (trained_dir / "_compute_receipts" / trained.attempt_id
     / "boundary.json").write_text("{}", encoding="utf-8")
    rejected_id = "idea-campaign-rejected"
    assert claim(rejected_id, results, 5)
    record_zero_gpu_outcome(
        rejected_id, results / rejected_id, 5, "rejected",
        "method_validator_rejected",
    )

    audit = audit_campaign_compute_receipts(
        results,
        idea_ids=[trained.idea_id, rejected_id],
        start_epoch=now - 10,
        end_epoch=now + 10,
        physical_scope=[4, 5, 6, 7],
    )

    assert audit["status"] == "VERIFIED"
    assert audit["invalid_receipts"] == 0
    assert audit["out_of_scope_receipts"] == 0
    assert audit["incomplete_started_attempts"] == 0
    assert audit["zero_gpu_rejection_rate"] == 1.0
    assert audit["rejection_attempts"] == 1
    assert audit["zero_gpu_rejection_attempts"] == 1
    assert audit["duplicate_training_attempts"] == 0
    assert audit["missing_terminal_ideas"] == []


def test_campaign_compute_audit_does_not_invent_unobserved_rejection_rate(
        tmp_path):
    results = tmp_path / "results"
    now = time.time()
    trained = _tp(tmp_path, attempt_id="7" * 32)
    trained.start_time = now - 5
    record_compute_start(trained, results / trained.idea_id)
    record_compute_terminal(
        trained, results / trained.idea_id, "completed", "trainer_completed",
        return_code=0,
    )

    audit = audit_campaign_compute_receipts(
        results,
        idea_ids=[trained.idea_id],
        start_epoch=now - 10,
        end_epoch=now + 10,
        physical_scope=[4, 5, 6, 7],
    )

    assert audit["status"] == "VERIFIED"
    assert audit["rejection_attempts"] == 0
    assert audit["zero_gpu_rejection_attempts"] == 0
    assert audit["zero_gpu_rejection_rate"] is None


def test_campaign_compute_audit_counts_allocated_rejection_in_denominator(
        tmp_path):
    results = tmp_path / "results"
    now = time.time()
    rejected = _tp(tmp_path, attempt_id="8" * 32)
    rejected.start_time = now - 5
    record_compute_start(rejected, results / rejected.idea_id)
    record_compute_terminal(
        rejected, results / rejected.idea_id, "rejected",
        "late_policy_rejection", return_code=1,
    )

    audit = audit_campaign_compute_receipts(
        results,
        idea_ids=[rejected.idea_id],
        start_epoch=now - 10,
        end_epoch=now + 10,
        physical_scope=[4, 5, 6, 7],
    )

    assert audit["status"] == "VERIFIED"
    assert audit["rejection_attempts"] == 1
    assert audit["zero_gpu_rejection_attempts"] == 0
    assert audit["zero_gpu_rejection_rate"] == 0.0


def test_campaign_compute_audit_fails_closed_on_scope_and_missing_terminal(
        tmp_path):
    results = tmp_path / "results"
    now = time.time()
    tp = _tp(tmp_path, attempt_id="f" * 32)
    tp.start_time = now - 5
    record_compute_start(tp, results / tp.idea_id)

    audit = audit_campaign_compute_receipts(
        results,
        idea_ids=[tp.idea_id],
        start_epoch=now - 10,
        end_epoch=now + 10,
        physical_scope=[5, 6, 7],
    )

    assert audit["status"] == "UNVERIFIED"
    assert audit["out_of_scope_receipts"] == 1
    assert audit["incomplete_started_attempts"] == 1
    assert audit["missing_terminal_ideas"] == [tp.idea_id]


def test_campaign_compute_audit_rejects_redirected_receipt(tmp_path):
    results = tmp_path / "results"
    now = time.time()
    tp = _tp(tmp_path, attempt_id="1" * 32)
    tp.start_time = now - 5
    idea_dir = results / tp.idea_id
    record_compute_start(tp, idea_dir)
    record_compute_terminal(
        tp, idea_dir, "completed", "trainer_completed", return_code=0
    )
    terminal = idea_dir / "_compute_receipts" / tp.attempt_id / "terminal.json"
    outside = tmp_path / "outside.json"
    terminal.rename(outside)
    terminal.symlink_to(outside)

    audit = audit_campaign_compute_receipts(
        results,
        idea_ids=[tp.idea_id],
        start_epoch=now - 10,
        end_epoch=now + 10,
        physical_scope=[4, 5, 6, 7],
    )

    assert audit["status"] == "UNVERIFIED"
    assert audit["invalid_receipts"] == 1
    assert audit["incomplete_started_attempts"] == 1


def test_campaign_compute_audit_cannot_hide_unregistered_losing_idea(tmp_path):
    results = tmp_path / "results"
    now = time.time()
    expected = _tp(tmp_path, attempt_id="2" * 32)
    expected.start_time = now - 5
    record_compute_start(expected, results / expected.idea_id)
    record_compute_terminal(
        expected, results / expected.idea_id, "completed",
        "trainer_completed", return_code=0,
    )
    hidden = _tp(tmp_path, attempt_id="3" * 32, return_code=1)
    hidden.idea_id = "idea-hidden-loser"
    hidden.start_time = now - 4
    record_compute_start(hidden, results / hidden.idea_id)
    record_compute_terminal(
        hidden, results / hidden.idea_id, "failed",
        "trainer_declared_failed", return_code=1,
    )

    audit = audit_campaign_compute_receipts(
        results,
        idea_ids=[expected.idea_id],
        start_epoch=now - 10,
        end_epoch=now + 10,
        physical_scope=[4, 5, 6, 7],
    )

    assert audit["status"] == "UNVERIFIED"
    assert audit["unexpected_campaign_idea_ids"] == ["idea-hidden-loser"]
