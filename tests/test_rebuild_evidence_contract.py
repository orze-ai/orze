"""Reconstruction must not resurrect evidence rejected by normal steering."""
import json
import os

import pytest

from orze.engine.rebuild_state import rebuild_state_file
from orze.idea_lake import IdeaLake
from orze.reporting.evidence import (
    authoritative_completed_idea_ids,
    qualify_authoritative_report_evidence,
)


def _project(tmp_path, rows, report=None):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    for index, (idea_id, metrics) in enumerate(rows):
        lake.insert(idea_id, idea_id, "x: 1", "raw", status="completed",
                    eval_metrics=metrics if isinstance(metrics, dict) else {"score": 999})
        folder = results / idea_id
        folder.mkdir()
        path = folder / "metrics.json"
        path.write_text(json.dumps(metrics))
        os.utime(path, (100 + index, 100 + index))
    cfg = {"results_dir": str(results), "idea_lake_db": lake.db_path,
           "report": report or {"primary_metric": "score", "sort": "descending"}}
    return results, lake, cfg


@pytest.mark.parametrize("sort,expected", [
    ("ascending", "idea-b"), ("descending", "idea-a"),
])
def test_rebuild_uses_declared_source_not_training_proxy(tmp_path, sort, expected):
    results, lake, cfg = _project(tmp_path, [
        ("idea-a", {"status": "COMPLETED", "score": 0.1}),
        ("idea-b", {"status": "COMPLETED", "score": 0.9}),
    ], {"primary_metric": "score", "sort": sort,
        "columns": [{"key": "score", "source": "evaluation.json:value"}]})
    try:
        for idea_id, value in (("idea-a", 0.8), ("idea-b", 0.2)):
            (results / idea_id / "evaluation.json").write_text(json.dumps({"value": value}))
        assert rebuild_state_file(results, cfg, lake=lake)["best_idea_id"] == expected
    finally:
        lake.close()


@pytest.mark.parametrize("invalid", [
    {"score": 999},
    {"status": "COMPLETED", "score": 999},
    {"status": "COMPLETED", "score": 0.99, "tainted_leakage": True},
    [],
])
def test_rebuild_counts_only_qualified_outcomes(tmp_path, invalid):
    results, lake, cfg = _project(tmp_path, [
        ("idea-valid", {"status": "COMPLETED", "score": 0.5}),
        ("idea-invalid", invalid),
    ])
    cfg["metric_validation"] = {"max_value": {"score": 1}}
    try:
        ids, _ = authoritative_completed_idea_ids(lake.db_path)
        assert qualify_authoritative_report_evidence(
            "idea-invalid", results, cfg, ids)[2] is None
        summary = rebuild_state_file(results, cfg, lake=lake)
        assert summary["best_idea_id"] == "idea-valid"
        assert summary["completions_since_best"] == 0
    finally:
        lake.close()


@pytest.mark.parametrize("reason", ["range", "source", "benchmark", "lifecycle"])
def test_rejected_artifacts_never_fall_back_to_lake_scores(tmp_path, reason):
    results, lake, cfg = _project(tmp_path, [
        ("idea-rejected", {"status": "COMPLETED", "score": 10}),
    ])
    if reason == "range":
        cfg["metric_validation"] = {"max_value": {"score": 1}}
    elif reason == "source":
        cfg["report"]["columns"] = [{"key": "score", "source": "missing.json:score"}]
    elif reason == "benchmark":
        cfg["report"]["benchmark_contract"] = {"receipt": "absent.json"}
    else:
        lake.conn.execute("UPDATE idea_state SET current_state='QUEUED'")
        lake.conn.commit()
    try:
        summary = rebuild_state_file(results, cfg, lake=lake)
        assert summary["best_idea_id"] is None
        assert summary["completions_since_best"] == 0
    finally:
        lake.close()


def test_rebuild_does_not_create_a_missing_lifecycle_database(tmp_path):
    db = tmp_path / "absent.db"
    summary = rebuild_state_file(tmp_path, {
        "idea_lake_db": str(db), "report": {"primary_metric": "score"},
    })
    assert summary["best_idea_id"] is None
    assert not db.exists()
