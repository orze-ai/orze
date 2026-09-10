"""Fail-closed interruption and cooperative checkpoint resume tests."""

import json
import time
from pathlib import Path

import pytest

from orze.core.config import _validate_config
from orze.engine.launcher import check_active, launch
from orze.engine.process import TrainingProcess
from orze.engine.resume import (
    ResumeValidationError,
    admit_resume,
    mark_resume_launched,
    prepare_resume_launch,
    validate_resume_evidence,
    write_interruption_receipt,
)
from orze.engine.scheduler import get_unclaimed
from orze.idea_lake import IdeaLake


class FakeProcess:
    pid = 424242

    def __init__(self, return_code=-15):
        self.return_code = return_code

    def poll(self):
        return self.return_code


@pytest.fixture
def resume_case(tmp_path):
    project = tmp_path
    results = project / "results"
    idea_dir = results / "idea-resume"
    checkpoint = idea_dir / "checkpoints" / "step-7"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.bin").write_bytes(b"checkpoint-v1")
    train_script = project / "train.py"
    train_script.write_text("# trainer v1\n", encoding="utf-8")
    immutable = project / "dataset.lock"
    immutable.write_text("private-revision-123\n", encoding="utf-8")
    (idea_dir / "idea_config.yaml").write_text("lr: 0.001\n", encoding="utf-8")
    (idea_dir / "progress.json").write_text(json.dumps({
        "schema_version": 1,
        "last_completed_step": 7,
        "checkpoint_path": "checkpoints/step-7",
        "resume_eligible": True,
    }), encoding="utf-8")
    cfg = {
        "_project_root": str(project),
        "results_dir": str(results),
        "train_script": str(train_script),
        "resume": {
            "enabled": True,
            "progress_file": "progress.json",
            "args": ["--resume-from", "{checkpoint}"],
            "checkpoint_roots": [],
            "immutable_inputs": ["dataset.lock"],
            "input_roots": [],
            "max_files": 100,
            "max_bytes": 1024 * 1024,
        },
    }
    tp = TrainingProcess(
        idea_id="idea-resume", gpu=4, process=FakeProcess(),
        start_time=time.time() - 5, log_path=idea_dir / "train_output.log",
        timeout=60, train_script=str(train_script),
        config_path=str(idea_dir / "idea_config.yaml"),
    )
    return project, results, idea_dir, checkpoint, cfg, tp


def _write_valid_receipt(case):
    _, results, _, _, cfg, tp = case
    return write_interruption_receipt(
        tp, results, cfg, reason="timeout", terminating_signal="SIGTERM",
        return_code=-15,
    )


def test_interruption_receipt_attests_contract_without_raw_input(resume_case):
    _, results, idea_dir, checkpoint, cfg, _ = resume_case
    receipt = _write_valid_receipt(resume_case)

    assert receipt["resume_eligible"] is True
    assert receipt["last_completed_step"] == 7
    assert receipt["reason"] == "timeout"
    assert receipt["terminating_signal"] == "SIGTERM"
    assert receipt["allocated_gpu_seconds"] >= 5
    assert receipt["checkpoint"]["file_count"] == 1
    assert receipt["immutable_inputs"][0]["sha256"]
    raw = (idea_dir / "interruption.json").read_text(encoding="utf-8")
    assert "private-revision-123" not in raw

    _, verified_checkpoint, _ = validate_resume_evidence(
        "idea-resume", results, cfg, str(checkpoint))
    assert verified_checkpoint == checkpoint


@pytest.mark.parametrize("target", [
    "checkpoint", "idea_config", "train_script", "immutable_input", "progress",
])
def test_any_attested_input_drift_rejects_resume(resume_case, target):
    project, results, idea_dir, checkpoint, cfg, _ = resume_case
    _write_valid_receipt(resume_case)
    paths = {
        "checkpoint": checkpoint / "model.bin",
        "idea_config": idea_dir / "idea_config.yaml",
        "train_script": project / "train.py",
        "immutable_input": project / "dataset.lock",
        "progress": idea_dir / "progress.json",
    }
    paths[target].write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ResumeValidationError, match="hash_mismatch"):
        validate_resume_evidence("idea-resume", results, cfg, str(checkpoint))


def test_unattested_checkpoint_override_is_rejected(resume_case):
    _, results, idea_dir, checkpoint, cfg, _ = resume_case
    _write_valid_receipt(resume_case)
    other = idea_dir / "checkpoints" / "other"
    other.mkdir()
    (other / "model.bin").write_bytes(b"checkpoint-v1")

    with pytest.raises(
            ResumeValidationError,
            match="resume_override_not_attested_checkpoint"):
        validate_resume_evidence("idea-resume", results, cfg, str(other))


def test_checkpoint_symlink_is_never_attested(resume_case):
    _, results, idea_dir, checkpoint, cfg, tp = resume_case
    symlink = idea_dir / "checkpoints" / "linked"
    symlink.symlink_to(checkpoint, target_is_directory=True)
    progress_path = idea_dir / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["checkpoint_path"] = "checkpoints/linked"
    progress_path.write_text(json.dumps(progress), encoding="utf-8")

    receipt = write_interruption_receipt(
        tp, results, cfg, "timeout", "SIGTERM", -15)
    assert receipt["resume_eligible"] is False
    assert receipt["resume_reason"] == "symlink_artifact_forbidden"


def test_admission_archives_state_and_launch_revalidates(resume_case):
    _, results, idea_dir, checkpoint, cfg, _ = resume_case
    receipt = _write_valid_receipt(resume_case)
    (idea_dir / "metrics.json").write_text(
        '{"status":"FAILED"}', encoding="utf-8")
    (idea_dir / "claim.json").write_text(
        '{"trainer_pid":999999999}', encoding="utf-8")

    admit_resume("idea-resume", results, cfg, str(checkpoint))
    assert (idea_dir / "resume_request.json").exists()
    assert list(idea_dir.glob("metrics.interrupted.*.json"))
    assert list(idea_dir.glob("claim.interrupted.*.json"))

    context = prepare_resume_launch("idea-resume", results, cfg)
    assert context["args"] == ["--resume-from", str(checkpoint)]
    (idea_dir / "claim.json").write_text(
        '{"trainer_pid":123}', encoding="utf-8")
    mark_resume_launched(context, idea_dir / "claim.json")
    claim = json.loads((idea_dir / "claim.json").read_text(encoding="utf-8"))
    assert claim["resume_checkpoint"] == str(checkpoint)
    assert claim["resume_receipt_sha256"]
    assert not (idea_dir / "resume_request.json").exists()
    assert (idea_dir / "resume_request.consumed.json").exists()
    assert claim["resume_receipt_sha256"] == context["receipt_sha256"]
    assert receipt["checkpoint"]["sha256"]


def test_launch_rejects_tamper_before_gpu_check(resume_case, monkeypatch):
    project, results, idea_dir, checkpoint, cfg, _ = resume_case
    _write_valid_receipt(resume_case)
    admit_resume("idea-resume", results, cfg, str(checkpoint))
    (idea_dir / "claim.json").write_text(
        '{"claimed_by":"test"}', encoding="utf-8")
    (checkpoint / "model.bin").write_bytes(b"changed-after-admission")
    cfg.update({
        "base_config": str(idea_dir / "idea_config.yaml"),
        "ideas_file": str(project / "ideas.md"),
        "python": "python3",
    })
    (project / "ideas.md").write_text("", encoding="utf-8")
    gpu_checked = []
    monkeypatch.setattr(
        "orze.engine.launcher._verify_gpu_free",
        lambda *args, **kwargs: gpu_checked.append(True),
    )

    with pytest.raises(ResumeValidationError, match="checkpoint_hash_mismatch"):
        launch("idea-resume", 4, results, cfg)
    assert gpu_checked == []


def test_explicit_resume_bypasses_failure_skip_only(resume_case):
    _, results, idea_dir, checkpoint, cfg, _ = resume_case
    _write_valid_receipt(resume_case)
    admit_resume("idea-resume", results, cfg, str(checkpoint))
    ideas = {"idea-resume": {"priority": "high", "config": {}}}

    assert get_unclaimed(ideas, results, skipped={"idea-resume"}) == [
        "idea-resume"]


def test_resume_rejects_path_traversal_idea_id(resume_case):
    _, results, _, checkpoint, cfg, _ = resume_case
    with pytest.raises(ResumeValidationError, match="idea_id_invalid"):
        validate_resume_evidence("../idea-resume", results, cfg, str(checkpoint))


def test_resume_rejects_ambiguous_live_claim(resume_case):
    _, results, idea_dir, checkpoint, cfg, _ = resume_case
    _write_valid_receipt(resume_case)
    (idea_dir / "claim.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ResumeValidationError, match="claim_identity_missing"):
        admit_resume("idea-resume", results, cfg, str(checkpoint))


def test_admission_requeues_audited_idea_lake_state(resume_case):
    project, results, _, checkpoint, cfg, _ = resume_case
    _write_valid_receipt(resume_case)
    db_path = project / "ideas.db"
    lake = IdeaLake(str(db_path))
    lake.insert("idea-resume", "resume", "{}", "", status="failed")
    lake.close()
    cfg["idea_lake_db"] = str(db_path)

    admit_resume("idea-resume", results, cfg, str(checkpoint))
    lake = IdeaLake(str(db_path))
    assert lake.get("idea-resume")["status"] == "queued"
    lake.close()


def test_timeout_writes_non_resumable_receipt_when_policy_disabled(
        tmp_path, monkeypatch):
    results = tmp_path / "results"
    idea_dir = results / "idea-timeout"
    idea_dir.mkdir(parents=True)
    process = FakeProcess(return_code=None)
    tp = TrainingProcess(
        idea_id="idea-timeout", gpu=4, process=process,
        start_time=time.time() - 100, log_path=idea_dir / "train_output.log",
        timeout=1,
    )

    def terminate(proc, *args, **kwargs):
        proc.return_code = -15

    monkeypatch.setattr("orze.engine.launcher._terminate_and_reap", terminate)
    monkeypatch.setattr(
        "orze.engine.launcher.notify", lambda *args, **kwargs: None)
    finished = check_active(
        {4: tp}, results,
        {"resume": {"enabled": False}, "sops": {"failure_feedback": False}},
        {}, fix_counts={},
    )
    receipt = json.loads(
        (idea_dir / "interruption.json").read_text(encoding="utf-8"))
    assert finished == [("idea-timeout", 4)]
    assert receipt["reason"] == "timeout"
    assert receipt["resume_eligible"] is False
    assert receipt["resume_reason"] == "resume_policy_disabled"


def test_graceful_detach_is_not_left_for_atexit_to_kill(tmp_path, monkeypatch):
    import orze.engine.lifecycle as lifecycle

    results = tmp_path / "results"
    results.mkdir()
    tp = TrainingProcess(
        idea_id="idea-running", gpu=4, process=FakeProcess(return_code=None),
        start_time=time.time(), log_path=results / "train.log", timeout=60,
    )
    active = {4: tp}
    killed = []
    monkeypatch.setattr(lifecycle, "save_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(lifecycle, "notify", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        lifecycle, "_kill_pg", lambda proc, sig: killed.append((proc, sig)))

    lifecycle.graceful_shutdown(
        results, {}, active, {}, {}, 1, {}, None, "host", "instance",
        kill_all=False,
    )
    lifecycle.atexit_cleanup(active, {}, {})
    assert active == {}
    assert killed == []


def test_resume_config_requires_pinned_inputs(tmp_path, monkeypatch):
    train = tmp_path / "train.py"
    train.write_text("# idea_config.yaml\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    errors, _ = _validate_config({
        "train_script": "train.py",
        "resume": {
            "enabled": True,
            "progress_file": "progress.json",
            "args": ["--resume-from", "{checkpoint}"],
            "immutable_inputs": [],
        },
    })
    assert any("at least one pinned" in error for error in errors)
