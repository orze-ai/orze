"""A launcher's no-process result must not mint successful evaluation state."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orze.core.benchmark_contract import BenchmarkContractError
from orze.engine import evaluator, phases
from orze.idea_lake import IdeaLake


@pytest.fixture
def project(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    idea_id = "idea-observation"
    folder = results / idea_id
    folder.mkdir()
    (folder / "metrics.json").write_text(
        json.dumps({"status": "COMPLETED", "score": 99}), encoding="utf-8"
    )
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    lake.insert(idea_id, idea_id, "{}", "", status="queued")
    assert lake.reconcile_training_complete(idea_id, "reconcile_test_completed_training")
    assert lake.get_fsm_state(idea_id) == "IN_PROGRESS"
    assert lake.get_stage_state(idea_id, "training") == "COMPLETE"
    assert lake.get_stage_state(idea_id, "evaluation") == "PENDING"
    cfg = {
        "_project_root": str(tmp_path),
        "results_dir": str(results),
        "idea_lake_db": lake.db_path,
        "eval_script": "unused-evaluation-entrypoint.py",
        "eval_output": "assessment.json",
        "gpu_scheduling": {"allowed_gpus": [4]},
        "report": {
            "primary_metric": "score", "sort": "ascending",
            "columns": [{"key": "score", "source": "assessment.json:score"}],
        },
    }
    controller = SimpleNamespace(
        cfg=cfg, results_dir=results, lake=lake, gpu_ids=[4],
        active={}, active_evals={}, pending_evals=[],
    )
    # These branches must terminate before any subprocess or GPU boundary.
    boundaries = []
    for name in ("gpu_execution_lease", "_verify_gpu_free"):
        tripwire = Mock(side_effect=AssertionError("Unexpected GPU access"))
        monkeypatch.setattr(evaluator, name, tripwire)
        boundaries.append(tripwire)
    popen = Mock(side_effect=AssertionError("Unexpected evaluator subprocess"))
    monkeypatch.setattr(evaluator.subprocess, "Popen", popen)
    boundaries.append(popen)
    try:
        yield controller, idea_id, folder
    finally:
        lake.close()
        for boundary in boundaries:
            boundary.assert_not_called()


@pytest.mark.parametrize("origin", ["finished-training", "pending-evaluation"])
def test_configured_preflight_rejection_is_not_success_and_work_is_not_lost(
    project, monkeypatch, origin
):
    controller, idea_id, folder = project
    preflight = Mock(side_effect=BenchmarkContractError("test_preflight_rejected"))
    monkeypatch.setattr(evaluator, "prepare_benchmark_evaluation", preflight)
    assert phases.launch_eval is evaluator.launch_eval
    finished = [(idea_id, 4)] if origin == "finished-training" else []
    if origin == "pending-evaluation":
        controller.pending_evals = [(idea_id, 4)]

    delivered, _ = phases.OrzePhaseMixin._launch_evals(
        controller, finished, [], {},
    )

    assert preflight.called, "Exercise the real launcher's preflight rejection"
    state = controller.lake.get_fsm_state(idea_id)
    stage = controller.lake.get_stage_state(idea_id, "evaluation")
    assert state != "COMPLETE", "A configured evaluator was never completed"
    assert stage != "SKIPPED", "Configured evaluation cannot become no-eval work"
    if state == "FAILED":
        assert stage == "FAILED"
    else:
        assert state == "IN_PROGRESS"
        assert stage in ("PENDING", "NOT_STARTED")
        assert (idea_id, 4) in controller.pending_evals
        assert (idea_id, 4) not in delivered
    assert controller.active_evals == {}
    assert not (folder / "assessment.json").exists() or state == "FAILED"


def test_training_only_project_legitimately_completes_without_an_evaluator(project):
    controller, idea_id, _ = project
    controller.cfg.pop("eval_script")

    delivered, _ = phases.OrzePhaseMixin._launch_evals(
        controller, [(idea_id, 4)], [], {},
    )

    assert delivered == [(idea_id, 4)]
    assert controller.pending_evals == []
    assert controller.active_evals == {}
    assert controller.lake.get_fsm_state(idea_id) == "COMPLETE"
    assert controller.lake.get_stage_state(idea_id, "training") == "COMPLETE"
    assert controller.lake.get_stage_state(idea_id, "evaluation") == "SKIPPED"


def test_existing_valid_evaluation_is_reconciled_and_delivered_as_complete(project):
    controller, idea_id, folder = project
    (folder / "assessment.json").write_text(
        json.dumps({"status": "COMPLETED", "score": 0}), encoding="utf-8"
    )

    delivered, _ = phases.OrzePhaseMixin._launch_evals(
        controller, [(idea_id, 4)], [], {},
    )

    assert delivered == [(idea_id, 4)]
    assert controller.pending_evals == []
    assert controller.active_evals == {}
    assert controller.lake.get_fsm_state(idea_id) == "COMPLETE"
    assert controller.lake.get_stage_state(idea_id, "evaluation") == "COMPLETE"
