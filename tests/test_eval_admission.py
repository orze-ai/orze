"""Evaluation must consume completed training outcomes only."""

import json

import pytest

from orze.engine.evaluator import (
    is_training_complete_for_downstream,
    launch_eval,
    run_post_scripts,
)
from orze.idea_lake import IdeaLake


@pytest.mark.parametrize("metrics,reason", [
    (None, "training_metrics_missing"),
    ("{broken", "training_metrics_invalid"),
    ("[]", "training_metrics_invalid"),
    ('{"status":"FAILED"}', "training_not_completed"),
    ('{"status":"PARTIAL"}', "training_not_completed"),
])
def test_only_completed_metrics_authorize_downstream(
        tmp_path, metrics, reason):
    idea_dir = tmp_path / "idea-test"
    idea_dir.mkdir()
    if metrics is not None:
        (idea_dir / "metrics.json").write_text(metrics, encoding="utf-8")
    eligible, observed_reason = is_training_complete_for_downstream(idea_dir)
    assert eligible is False
    assert observed_reason == reason


def test_completed_metrics_authorize_downstream(tmp_path):
    idea_dir = tmp_path / "idea-test"
    idea_dir.mkdir()
    (idea_dir / "metrics.json").write_text(
        '{"status":"COMPLETED"}', encoding="utf-8")
    assert is_training_complete_for_downstream(idea_dir) == (
        True, "training_completed")


@pytest.mark.parametrize("status", ["FAILED", "PARTIAL", "RUNNING"])
def test_checkpoint_file_cannot_override_noncompleted_status(
        tmp_path, monkeypatch, status):
    idea_dir = tmp_path / "idea-test"
    idea_dir.mkdir()
    (idea_dir / "metrics.json").write_text(
        json.dumps({"status": status}), encoding="utf-8")
    (idea_dir / "best_model.pt").write_bytes(b"unattested checkpoint")
    gpu_checked = []
    monkeypatch.setattr(
        "orze.engine.evaluator._verify_gpu_free",
        lambda *args: gpu_checked.append(True))

    assert launch_eval("idea-test", 4, tmp_path, {
        "eval_script": "/usr/bin/true",
    }) is None
    assert gpu_checked == []
    audit = json.loads(
        (idea_dir / "_eval_audit.jsonl").read_text(encoding="utf-8"))
    assert audit["reason"] == "training_not_completed"


def test_corrupt_metrics_with_checkpoint_fails_before_gpu_check(
        tmp_path, monkeypatch):
    idea_dir = tmp_path / "idea-test"
    idea_dir.mkdir()
    (idea_dir / "metrics.json").write_text("{broken", encoding="utf-8")
    (idea_dir / "best_model.pt").write_bytes(b"checkpoint")
    gpu_checked = []
    monkeypatch.setattr(
        "orze.engine.evaluator._verify_gpu_free",
        lambda *args: gpu_checked.append(True))

    assert launch_eval("idea-test", 4, tmp_path, {
        "eval_script": "/usr/bin/true",
    }) is None
    assert gpu_checked == []


def test_post_scripts_do_not_run_for_failed_checkpoint(
        tmp_path, monkeypatch):
    idea_dir = tmp_path / "idea-test"
    idea_dir.mkdir()
    (idea_dir / "metrics.json").write_text(
        '{"status":"FAILED"}', encoding="utf-8")
    (idea_dir / "best_model.pt").write_bytes(b"checkpoint")
    calls = []
    monkeypatch.setattr(
        "orze.engine.evaluator.subprocess.run",
        lambda *args, **kwargs: calls.append(True))

    run_post_scripts("idea-test", 4, tmp_path, {
        "post_scripts": [{"script": "/usr/bin/true"}],
    })
    assert calls == []


def test_completed_eval_reaches_masked_popen(tmp_path, monkeypatch):
    idea_dir = tmp_path / "idea-test"
    idea_dir.mkdir()
    (idea_dir / "metrics.json").write_text(
        '{"status":"COMPLETED"}', encoding="utf-8")
    observed = {}

    class RunningProcess:
        pid = 12345

        def poll(self):
            return None

    monkeypatch.setattr(
        "orze.engine.evaluator._verify_gpu_free", lambda *args: None)

    def popen(cmd, **kwargs):
        observed["cmd"] = cmd
        observed["env"] = kwargs["env"]
        return RunningProcess()

    monkeypatch.setattr("orze.engine.evaluator.subprocess.Popen", popen)
    ep = launch_eval("idea-test", 4, tmp_path, {
        "eval_script": "/usr/bin/true",
        "gpu_scheduling": {"allowed_gpus": [4, 5, 6, 7]},
    })
    assert ep is not None
    assert observed["env"]["CUDA_VISIBLE_DEVICES"] == "4"
    ep.close_log()


@pytest.mark.parametrize(
    "report,global_state,evaluation_state",
    [
        ({"status": "COMPLETED"}, "COMPLETE", "COMPLETE"),
        ({"status": "FAILED", "reason": "domain"}, "FAILED", "FAILED"),
    ],
)
def test_existing_eval_output_is_reconciled_without_gpu_launch(
        tmp_path, monkeypatch, report, global_state, evaluation_state):
    idea_id = "idea-existing-eval"
    idea_dir = tmp_path / idea_id
    idea_dir.mkdir()
    (idea_dir / "metrics.json").write_text(
        json.dumps({"status": "COMPLETED"}), encoding="utf-8")
    (idea_dir / "eval_report.json").write_text(
        json.dumps(report), encoding="utf-8")
    lake = IdeaLake(tmp_path / "lake.db")
    lake.insert(idea_id, "existing", "{}", "", status="queued")
    assert lake.reconcile_training_complete(
        idea_id, "reconcile_test_training_completed")
    gpu_checked = []
    monkeypatch.setattr(
        "orze.engine.evaluator._verify_gpu_free",
        lambda *args: gpu_checked.append(True),
    )

    assert launch_eval(idea_id, 4, tmp_path, {
        "eval_script": "/usr/bin/true",
        "eval_output": "eval_report.json",
    }, lake=lake) is None

    assert gpu_checked == []
    assert lake.get_fsm_state(idea_id) == global_state
    assert lake.get_stage_state(idea_id, "training") == "COMPLETE"
    assert lake.get_stage_state(idea_id, "evaluation") == evaluation_state
    lake.close()
