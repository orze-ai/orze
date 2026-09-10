"""Dispatch rejects malformed training documents without inventing success."""

import json
from unittest.mock import Mock

import pytest

from orze.core.benchmark_contract import BenchmarkContractError
from orze.engine import evaluator, phases
from test_eval_dispatch_outcomes import project


@pytest.mark.parametrize("document", [None, [], 42], ids=["null", "array", "scalar"])
def test_nonmapping_training_metrics_are_delivered_unverified_not_completed(
    project, document,
):
    controller, idea_id, folder = project
    (folder / "metrics.json").write_text(json.dumps(document), encoding="utf-8")

    delivered, _ = phases.OrzePhaseMixin._launch_evals(
        controller, [(idea_id, 4)], [], {},
    )

    assert delivered == [(idea_id, 4)]
    assert controller.lake.get_fsm_state(idea_id) == "IN_PROGRESS"
    assert controller.lake.get_stage_state(idea_id, "evaluation") == "PENDING"


def test_legacy_no_lake_backlog_recognizes_normalized_metrics_alias(
    project, monkeypatch,
):
    controller, idea_id, _ = project
    controller.lake = None
    controller.cfg["eval_output"] = "./metrics.json"
    controller.cfg["report"]["columns"][0]["source"] = "./metrics.json:score"
    preflight = Mock(side_effect=BenchmarkContractError("test_preflight_rejected"))
    monkeypatch.setattr(evaluator, "prepare_benchmark_evaluation", preflight)
    monkeypatch.setattr(phases, "get_gpu_memory_used", lambda _gpu: 0)
    monkeypatch.setattr(phases, "_eval_already_running", lambda *_args: False)

    delivered, _ = phases.OrzePhaseMixin._launch_evals(
        controller, [], [], {idea_id: {}},
    )

    preflight.assert_called_once()
    assert delivered == []
    assert controller.pending_evals == [(idea_id, 4)]
