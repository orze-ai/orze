"""Tests for F4 — rebuild best_idea_id from idea_lake.db."""
import json
from pathlib import Path

import pytest

from orze.idea_lake import IdeaLake
from orze.engine.rebuild_state import (
    rebuild_best_from_lake,
    rebuild_state_file,
)


def _make_lake(tmp_path, rows):
    db = tmp_path / "idea_lake.db"
    lake = IdeaLake(str(db))
    for r in rows:
        lake.insert(
            r["id"], r.get("title", r["id"]),
            "lr: 0.001\n", f"## {r['id']}: t\n",
            eval_metrics=r.get("eval_metrics"),
            status=r.get("status", "completed"),
        )
    return lake


def test_rebuild_picks_max_metric_completed(tmp_path):
    lake = _make_lake(tmp_path, [
        {"id": "idea-001", "eval_metrics": {"test_accuracy": 0.80}},
        {"id": "idea-002", "eval_metrics": {"test_accuracy": 0.85}},
        {"id": "idea-003", "eval_metrics": {"test_accuracy": 0.82}},
    ])
    best, since = rebuild_best_from_lake(lake, "test_accuracy")
    assert best == "idea-002"
    assert since >= 0
    lake.close()


def test_rebuild_ignores_non_completed(tmp_path):
    lake = _make_lake(tmp_path, [
        {"id": "idea-001", "eval_metrics": {"map": 0.9}, "status": "failed"},
        {"id": "idea-002", "eval_metrics": {"map": 0.5}, "status": "completed"},
    ])
    best, _ = rebuild_best_from_lake(lake, "map")
    assert best == "idea-002"
    lake.close()


def test_rebuild_honors_ascending_sort_and_complete_dataset_gate(tmp_path):
    complete = {f"wer_set_{i}": float(i) for i in range(8)}
    lake = _make_lake(tmp_path, [
        {"id": "idea-partial", "eval_metrics": {
            "avg_wer": 1.0,
            **{f"wer_set_{i}": float(i) for i in range(7)},
        }},
        {"id": "idea-complete-worse", "eval_metrics": {
            "avg_wer": 6.0, **complete,
        }},
        {"id": "idea-complete-best", "eval_metrics": {
            "avg_wer": 5.0, **complete,
        }},
    ])

    best, _ = rebuild_best_from_lake(
        lake, "avg_wer", sort_order="ascending", min_datasets=8,
        dataset_keys=list(complete),
    )

    assert best == "idea-complete-best"
    lake.close()


def test_rebuild_none_when_empty(tmp_path):
    db = tmp_path / "empty.db"
    lake = IdeaLake(str(db))
    best, since = rebuild_best_from_lake(lake, "test_accuracy")
    assert best is None
    assert since == 0
    lake.close()


def test_rebuild_state_file_writes(tmp_path):
    _ = _make_lake(tmp_path, [
        {"id": "idea-001", "eval_metrics": {"test_accuracy": 0.81}},
        {"id": "idea-002", "eval_metrics": {"test_accuracy": 0.88}},
    ])
    cfg = {"results_dir": str(tmp_path),
           "report": {"primary_metric": "test_accuracy"},
           "idea_lake_db": str(tmp_path / "idea_lake.db")}
    summary = rebuild_state_file(tmp_path, cfg)
    assert summary["best_idea_id"] == "idea-002"
    assert summary["wrote_state_file"] is True


def test_rebuild_state_file_idempotent_without_overwrite(tmp_path):
    _ = _make_lake(tmp_path, [
        {"id": "idea-001", "eval_metrics": {"test_accuracy": 0.9}},
    ])
    cfg = {"results_dir": str(tmp_path),
           "report": {"primary_metric": "test_accuracy"},
           "idea_lake_db": str(tmp_path / "idea_lake.db")}
    rebuild_state_file(tmp_path, cfg)
    # Second call without overwrite should NOT rewrite.
    s = rebuild_state_file(tmp_path, cfg)
    assert s["wrote_state_file"] is False


def test_rebuild_state_prefers_terminal_artifacts_over_stale_lake_units(tmp_path):
    lake = _make_lake(tmp_path, [
        {"id": "idea-stale", "eval_metrics": {"avg_wer": 0.05}},
        {"id": "idea-best", "eval_metrics": {"avg_wer": 0.06}},
    ])
    lake.close()
    for idea_id, avg_wer in (("idea-stale", 6.0), ("idea-best", 5.0)):
        idea_dir = tmp_path / idea_id
        idea_dir.mkdir()
        (idea_dir / "metrics.json").write_text(json.dumps({
            "status": "COMPLETED",
            "avg_wer": avg_wer,
        }))
    cfg = {
        "results_dir": str(tmp_path),
        "report": {"primary_metric": "avg_wer", "sort": "ascending"},
        "idea_lake_db": str(tmp_path / "idea_lake.db"),
    }

    summary = rebuild_state_file(tmp_path, cfg, overwrite=True)

    assert summary["best_idea_id"] == "idea-best"
