"""One scheduler tick attempts each evaluation at most once and keeps failures."""

import json
from unittest.mock import Mock

import pytest

from orze.core.benchmark_contract import BenchmarkContractError
from orze.engine import evaluator, phases
from test_eval_dispatch_outcomes import project  # Real IdeaLake and GPU tripwires.


@pytest.fixture
def preflight_rejection(monkeypatch):
    preflight = Mock(side_effect=BenchmarkContractError("test_preflight_rejected"))
    monkeypatch.setattr(evaluator, "prepare_benchmark_evaluation", preflight)
    # A read-only backlog admission check must not query actual host GPUs.
    monkeypatch.setattr(phases, "get_gpu_memory_used", lambda _gpu: 0)
    monkeypatch.setattr(phases, "_eval_already_running", lambda *_args: False)
    return preflight


@pytest.mark.parametrize("duplicate_pending", [False, True])
def test_finished_work_is_not_retried_by_pending_or_backlog_in_the_same_tick(
    project, preflight_rejection, duplicate_pending,
):
    controller, idea_id, _ = project
    if duplicate_pending:
        controller.pending_evals = [(idea_id, 4), (idea_id, 4)]

    delivered, _ = phases.OrzePhaseMixin._launch_evals(
        controller, [(idea_id, 4)], [], {idea_id: {}},
    )

    assert preflight_rejection.call_count == 1
    assert delivered == []
    assert controller.pending_evals == [(idea_id, 4)]
    assert controller.lake.get_fsm_state(idea_id) == "IN_PROGRESS"


@pytest.mark.parametrize("output", ["assessment.json", "metrics.json"])
def test_backlog_preflight_failure_is_pending_even_when_eval_overwrites_metrics(
    project, preflight_rejection, output,
):
    controller, idea_id, _ = project
    controller.cfg["eval_output"] = output
    controller.cfg["report"]["columns"][0]["source"] = output + ":score"

    delivered, _ = phases.OrzePhaseMixin._launch_evals(
        controller, [], [], {idea_id: {}},
    )

    assert preflight_rejection.call_count == 1
    assert delivered == []
    assert controller.pending_evals == [(idea_id, 4)]
    assert controller.lake.get_fsm_state(idea_id) == "IN_PROGRESS"
    assert controller.lake.get_stage_state(idea_id, "evaluation") == "PENDING"


def test_backlog_existing_valid_output_is_reconciled_and_delivered_once(
    project, preflight_rejection,
):
    controller, idea_id, folder = project
    (folder / "assessment.json").write_text(
        json.dumps({"status": "COMPLETED", "score": 0}), encoding="utf-8",
    )

    delivered, _ = phases.OrzePhaseMixin._launch_evals(
        controller, [], [], {idea_id: {}},
    )

    assert delivered == [(idea_id, 4)]
    assert controller.lake.get_fsm_state(idea_id) == "COMPLETE"
    assert controller.lake.get_stage_state(idea_id, "evaluation") == "COMPLETE"
    preflight_rejection.assert_not_called()
    again, backlog = phases.OrzePhaseMixin._launch_evals(
        controller, [], [], {idea_id: {}},
    )
    assert again == []
    assert backlog == []
