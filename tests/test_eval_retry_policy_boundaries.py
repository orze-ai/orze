"""An explicit evaluation retry pins its inputs and preserves framework config."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from orze.engine.evaluation_retry import (
    EvaluationRetryError, request_evaluation_retry, validate_pending_retry,
)
from orze.idea_lake import IdeaLake


@pytest.fixture
def project(tmp_path):
    results = tmp_path / "results"
    folder = results / "idea-policy-retry"
    folder.mkdir(parents=True)
    (folder / "metrics.json").write_text('{"status":"COMPLETED","quality":0}')
    (folder / "checkpoint.pt").write_bytes(b"original-generation-checkpoint")
    (folder / "assessment.json").write_text('{"status":"FAILED"}')
    (folder / "eval_output.log").write_text("original failed evaluation\n")
    lake = IdeaLake(tmp_path / "ideas.db")
    lake.insert(folder.name, "Retry policy", "seed: 7", "", status="queued")
    assert lake.reconcile_training_complete(folder.name, "reconcile_training_completed")
    assert lake.record_stage_transition(folder.name, "evaluation", "PENDING", "IN_PROGRESS", "evaluation_launched")
    assert lake.record_state_transition(folder.name, "IN_PROGRESS", "FAILED", "evaluation_failed")
    cfg = {
        "_project_root": str(tmp_path), "results_dir": str(results),
        "idea_lake_db": lake.db_path, "eval_script": "unused-evaluator.py",
        "eval_output": "assessment.json", "eval_checkpoint": "checkpoint.pt",
        "python": "/usr/bin/python3", "train_extra_env": {"DATASET_VERSION": "original"},
        "report": {"primary_metric": "quality", "sort": "ascending",
                   "columns": [{"key": "quality", "source": "assessment.json:quality"}]},
    }
    try:
        yield SimpleNamespace(results=results, folder=folder, idea_id=folder.name,
                              lake=lake, cfg=cfg)
    finally:
        lake.close()


def _request(p):
    return request_evaluation_retry(p.idea_id, p.results, p.cfg, p.lake)


def _snapshot(p):
    return {
        "files": {str(path.relative_to(p.folder)): path.read_bytes()
                  for path in p.folder.rglob("*") if path.is_file()},
        "database": {table: [tuple(row) for row in p.lake.conn.execute(
            f"SELECT * FROM {table} ORDER BY rowid")]
            for table in ("ideas", "idea_state", "idea_stage_state",
                          "idea_transitions", "idea_stage_transitions")},
    }


@pytest.mark.parametrize("field", ["python", "train_extra_env"])
def test_changed_evaluation_execution_policy_cannot_reuse_pending_admission(project, field):
    p = project
    _request(p)
    if field == "python":
        p.cfg[field] = "/different/python"
    else:
        p.cfg[field] = {"DATASET_VERSION": "different"}
    before = _snapshot(p)

    with pytest.raises(EvaluationRetryError):
        _request(p)
    with pytest.raises(EvaluationRetryError):
        validate_pending_retry(p.idea_id, p.results, p.cfg, p.lake)

    assert _snapshot(p) == before


def test_changed_generated_checkpoint_cannot_pass_pending_retry_launch_gate(project):
    p = project
    _request(p)
    (p.folder / "checkpoint.pt").write_bytes(b"different-generation-checkpoint")
    before = _snapshot(p)

    with pytest.raises(EvaluationRetryError):
        validate_pending_retry(p.idea_id, p.results, p.cfg, p.lake)

    assert _snapshot(p) == before


@pytest.mark.parametrize("filename", ["idea_config.yaml", "resolved_config.yaml"])
def test_declared_eval_output_cannot_move_framework_training_configuration(project, filename):
    p = project
    (p.folder / filename).write_text("seed: 7\nmodel: preserve-original\n")
    p.cfg["eval_output"] = filename
    before = _snapshot(p)

    with pytest.raises(EvaluationRetryError):
        _request(p)

    assert _snapshot(p) == before


def test_partial_preparation_cannot_resume_under_changed_declared_eval_arguments(project, monkeypatch):
    p = project
    original_rename = os.rename
    moved = []

    def interrupt_after_first_move(source, destination):
        if Path(source).parent == p.folder:
            if moved:
                raise OSError("injected partial preparation")
            original_rename(source, destination)
            moved.append(Path(source))
            return
        original_rename(source, destination)

    with monkeypatch.context() as patcher:
        patcher.setattr(os, "rename", interrupt_after_first_move)
        with pytest.raises(EvaluationRetryError):
            _request(p)
    assert len(moved) == 1
    assert p.lake.get_fsm_state(p.idea_id) == "FAILED"
    p.cfg["eval_args"] = ["--different-policy"]
    before = _snapshot(p)

    with pytest.raises(EvaluationRetryError):
        _request(p)

    assert _snapshot(p) == before
