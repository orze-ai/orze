"""Existing artifacts cannot authorize evaluation or bypass its output contract."""

import json
from contextlib import contextmanager
from unittest.mock import Mock

import pytest

from orze.engine import evaluator
from orze.idea_lake import IdeaLake
from orze.reporting.evidence import qualify_local_report_evidence


@pytest.fixture
def project(tmp_path, monkeypatch):
    results = tmp_path / "results"
    idea_id = "idea-existing-contract"
    folder = results / idea_id
    folder.mkdir(parents=True)
    lake = IdeaLake(tmp_path / "ideas.db")
    lake.insert(idea_id, "Existing output", "{}", "", status="queued")
    assert lake.record_state_transition(idea_id, "QUEUED", "CLAIMED")
    assert lake.record_state_transition(idea_id, "CLAIMED", "IN_PROGRESS")
    cfg = {
        "_project_root": str(tmp_path),
        "idea_lake_db": str(lake.db_path),
        "eval_script": "never_run.py",
        "eval_output": "eval_report.json",
        "gpu_scheduling": {"allowed_gpus": [4], "reserved_gpus": [0, 1, 2, 3]},
        "report": {
            "primary_metric": "quality",
            "sort": "descending",
            "columns": [{"key": "quality", "source": "eval_report.json:quality"}],
        },
    }
    gpu_check = Mock()
    popen = Mock(side_effect=AssertionError("unexpected evaluation launch"))
    monkeypatch.setattr(evaluator, "_verify_gpu_free", gpu_check)
    monkeypatch.setattr(evaluator.subprocess, "Popen", popen)

    @contextmanager
    def fake_lease(_gpu, **_kwargs):
        yield ()

    monkeypatch.setattr(evaluator, "gpu_execution_lease", fake_lease)
    try:
        yield results, folder, idea_id, lake, cfg, gpu_check, popen
    finally:
        lake.close()


def _completed_training(project, metrics=None):
    _, folder, idea_id, lake, _, _, _ = project
    (folder / "metrics.json").write_text(
        json.dumps(metrics or {"status": "COMPLETED", "quality": 999}),
        encoding="utf-8",
    )
    assert lake.record_stage_transition(
        idea_id, "training", "IN_PROGRESS", "COMPLETE",
        "training_completed_evaluation_pending",
    )


def _launch(project):
    results, _, idea_id, lake, cfg, _, _ = project
    return evaluator.launch_eval(idea_id, 4, results, cfg, lake=lake)


@pytest.mark.parametrize("training_document,reason", [
    (None, "training_metrics_missing"),
    ("{broken", "training_metrics_invalid"),
    ('{"status":"FAILED"}', "training_not_completed"),
    ('{"status":"PARTIAL"}', "training_not_completed"),
])
def test_existing_report_cannot_override_ineligible_training(
    project, training_document, reason,
):
    _, folder, idea_id, lake, _, gpu_check, popen = project
    if training_document is not None:
        (folder / "metrics.json").write_text(training_document, encoding="utf-8")
    (folder / "eval_report.json").write_text(
        '{"status":"COMPLETED","quality":1}', encoding="utf-8",
    )
    before = lake.get_stage_history(idea_id)

    assert _launch(project) is None

    assert lake.get_fsm_state(idea_id) == "IN_PROGRESS"
    assert lake.get_stage_state(idea_id, "training") == "IN_PROGRESS"
    assert lake.get_stage_state(idea_id, "evaluation") == "PENDING"
    assert lake.get_stage_history(idea_id) == before
    gpu_check.assert_not_called()
    popen.assert_not_called()
    audit = [json.loads(line) for line in
             (folder / "_eval_audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row["reason"] == reason for row in audit)


@pytest.mark.parametrize("evaluation_document", [
    "{broken", "[]", "null",
    '{"status":"FAILED","quality":1}',
    '{"status":"PARTIAL","quality":1}',
])
def test_existing_invalid_output_cannot_complete_evaluation(
    project, evaluation_document,
):
    _, folder, idea_id, lake, _, gpu_check, popen = project
    _completed_training(project)
    training_before = (folder / "metrics.json").read_bytes()
    (folder / "eval_report.json").write_text(evaluation_document, encoding="utf-8")

    assert _launch(project) is None

    assert lake.get_fsm_state(idea_id) != "COMPLETE"
    assert lake.get_stage_state(idea_id, "evaluation") != "COMPLETE"
    assert lake.get_stage_state(idea_id, "training") == "COMPLETE"
    assert (folder / "metrics.json").read_bytes() == training_before
    assert (folder / "eval_report.json").read_text(encoding="utf-8") == evaluation_document
    gpu_check.assert_not_called()
    popen.assert_not_called()


def test_existing_valid_exact_source_zero_completes_without_launch(project):
    _, folder, idea_id, lake, cfg, gpu_check, popen = project
    _completed_training(project)
    cfg["metric_validation"] = {"min_value": {"quality": 0}, "max_value": {"quality": 1}}
    (folder / "eval_report.json").write_text(
        '{"status":"COMPLETED","quality":0}', encoding="utf-8",
    )
    assert qualify_local_report_evidence(folder, cfg)[2] == 0.0

    assert _launch(project) is None

    assert lake.get_fsm_state(idea_id) == "COMPLETE"
    assert lake.get_stage_state(idea_id, "training") == "COMPLETE"
    assert lake.get_stage_state(idea_id, "evaluation") == "COMPLETE"
    gpu_check.assert_not_called()
    popen.assert_not_called()


class _Process:
    pid = 12345
    returncode = None

    def poll(self):
        return self.returncode


def _permit_masked_launch(project):
    _, _, _, _, _, _, popen = project
    process = _Process()
    popen.side_effect = None
    popen.return_value = process
    return process


def test_metrics_only_evaluator_without_objective_needs_no_new_report_file(project):
    results, folder, idea_id, lake, cfg, gpu_check, popen = project
    _completed_training(project, {"status": "COMPLETED", "diagnostic_count": 1})
    cfg.pop("report")
    process = _permit_masked_launch(project)

    ep = _launch(project)
    assert ep is not None
    try:
        process.returncode = 0
        assert evaluator.check_active_evals({4: ep}, results, cfg, lake=lake) == [
            (idea_id, 4),
        ]
    finally:
        ep.close_log()

    assert lake.get_fsm_state(idea_id) == "COMPLETE"
    assert lake.get_stage_state(idea_id, "evaluation") == "COMPLETE"
    assert not (folder / "eval_report.json").exists()
    gpu_check.assert_called_once()
    popen.assert_called_once()
    assert popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "4"


def test_metrics_output_alias_cannot_mistake_training_artifact_for_evaluation(project):
    _, folder, idea_id, lake, cfg, gpu_check, popen = project
    _completed_training(project, {"status": "COMPLETED", "quality": 0})
    cfg["eval_output"] = "metrics.json"
    cfg["report"]["columns"] = [{"key": "quality"}]
    training_before = (folder / "metrics.json").read_bytes()
    _permit_masked_launch(project)

    ep = _launch(project)

    assert ep is not None
    try:
        assert lake.get_fsm_state(idea_id) == "IN_PROGRESS"
        assert lake.get_stage_state(idea_id, "training") == "COMPLETE"
        assert lake.get_stage_state(idea_id, "evaluation") == "IN_PROGRESS"
        assert (folder / "metrics.json").read_bytes() == training_before
        gpu_check.assert_called_once()
        popen.assert_called_once()
        assert popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "4"
    finally:
        ep.close_log()
