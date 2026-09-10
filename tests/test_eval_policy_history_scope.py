"""Evaluation completion policy revisions cannot borrow prior guard history."""
import pytest

from orze.engine.champion_history import objective_scope


@pytest.mark.parametrize("change", ["activate", "output"])
def test_evaluation_contract_change_isolates_champion_history_scope(change):
    cfg = {"report": {"primary_metric": "quality", "sort": "ascending",
                      "columns": [{"key": "quality"}]},
           "eval_output": "assessment.json"}
    before = objective_scope(cfg)
    if change == "activate":
        cfg["eval_script"] = "unused-evaluator.py"
    else:
        cfg["eval_output"] = "revised-assessment.json"
    assert objective_scope(cfg) != before
