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
