"""Tests for orze.engine.metric_harvester — mid-run metric extraction."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orze.engine.metric_harvester import (
    HARVEST_SOURCE,
    extract_best_metric,
    harvest_running_ideas,
)

VJEPA2_LOG = """\
Loaded idea config keys: ['batch_size', 'lr']
Training: 60 epochs, 9000 spe, lr=1e-05
    per-group mAP=0.7910 (overall_AP=0.7876, groups={0:0.8137, 1:0.7732})
Epoch   1/60 | loss=0.6232 | test_mAP=0.7910 | lr=5.00e-06 | time=2890.4s
  ** New best: 0.7910
    per-group mAP=0.7992 (overall_AP=0.7997)
Epoch   2/60 | loss=0.5225 | test_mAP=0.7992 | lr=1.00e-05 | time=2682.2s
  ** New best: 0.7992
    per-group mAP=0.8314 (overall_AP=0.8306)
Epoch   3/60 | loss=0.4170 | test_mAP=0.8314 | lr=9.99e-06 | time=2947.7s
  ** New best: 0.8314
    per-group mAP=0.8100 (overall_AP=0.8123)
Epoch   4/60 | loss=0.3692 | test_mAP=0.8100 | lr=9.97e-06 | time=2674.4s
"""


def test_extract_map_finds_best_across_epochs():
    result = extract_best_metric(VJEPA2_LOG, "map")
    assert result is not None
    best, last_epoch = result
    assert best == pytest.approx(0.8314)
    assert last_epoch == 4


def test_extract_returns_none_when_no_match():
    assert extract_best_metric("nothing here", "map") is None


def test_extract_honors_maximize_false_for_loss():
    loss_log = "val_loss=1.5\nval_loss=0.8\nval_loss=1.2\nEpoch 3/10\n"
    result = extract_best_metric(loss_log, "loss", maximize=False)
    assert result is not None
    best, _ = result
    assert best == pytest.approx(0.8)


def test_extract_generic_fallback_for_unknown_metric():
    log = "custom_score=0.91\nEpoch 2/5\ncustom_score=0.95\n"
    result = extract_best_metric(log, "custom_score")
    assert result is not None
    best, last_epoch = result
    assert best == pytest.approx(0.95)
    assert last_epoch == 2


def test_extract_uses_extra_patterns_first(tmp_path):
    # Extra pattern should win over default if both match.
    log = "project_metric=0.8\ntest_mAP=0.5\nEpoch 1/2\n"
    result = extract_best_metric(
        log, "map", extra_patterns=[r"project_metric\s*=\s*([0-9.]+)"])
    assert result is not None
    assert result[0] == pytest.approx(0.8)


def test_harvest_writes_metrics_json_for_running_idea(tmp_path):
    results = tmp_path / "results"
    idea_dir = results / "idea-abc123"
    idea_dir.mkdir(parents=True)
    (idea_dir / "train_output.log").write_text(VJEPA2_LOG)

    n = harvest_running_ideas(results, primary_metric="map")

    assert n == 1
    mj = idea_dir / "metrics.json"
    assert mj.exists()
    data = json.loads(mj.read_text())
    assert data["map"] == pytest.approx(0.8314)
    assert data["best_map"] == pytest.approx(0.8314)
    assert data["last_epoch"] == 4
    assert data["_source"] == HARVEST_SOURCE


def test_harvest_respects_genuine_metrics_json(tmp_path):
    results = tmp_path / "results"
    idea_dir = results / "idea-real"
    idea_dir.mkdir(parents=True)
    (idea_dir / "train_output.log").write_text(VJEPA2_LOG)
    # Training script wrote a real metrics.json — harvester must not touch it.
    real = {"map": 0.5, "note": "authoritative"}
    (idea_dir / "metrics.json").write_text(json.dumps(real))

    n = harvest_running_ideas(results, primary_metric="map")

    assert n == 0
    data = json.loads((idea_dir / "metrics.json").read_text())
    assert data == real


def test_harvest_overwrites_its_own_prior_output(tmp_path):
    results = tmp_path / "results"
    idea_dir = results / "idea-grow"
    idea_dir.mkdir(parents=True)
    # First harvest sees only 2 epochs.
    (idea_dir / "train_output.log").write_text(
        "Epoch 1/60 | loss=0.6 | test_mAP=0.79\n"
        "Epoch 2/60 | loss=0.5 | test_mAP=0.80\n")
    assert harvest_running_ideas(results) == 1
    first = json.loads((idea_dir / "metrics.json").read_text())
    assert first["best_map"] == pytest.approx(0.80)
    assert first["last_epoch"] == 2

    # Log grows — harvester must pick up the new best.
    (idea_dir / "train_output.log").write_text(
        "Epoch 1/60 | loss=0.6 | test_mAP=0.79\n"
        "Epoch 2/60 | loss=0.5 | test_mAP=0.80\n"
        "Epoch 3/60 | loss=0.4 | test_mAP=0.83\n")
    assert harvest_running_ideas(results) == 1
    second = json.loads((idea_dir / "metrics.json").read_text())
    assert second["best_map"] == pytest.approx(0.83)
    assert second["last_epoch"] == 3


def test_harvest_skips_ideas_without_log(tmp_path):
    results = tmp_path / "results"
    (results / "idea-empty").mkdir(parents=True)
    assert harvest_running_ideas(results) == 0


def test_harvest_ignores_non_idea_dirs(tmp_path):
    results = tmp_path / "results"
    # Non-matching dir should be skipped.
    (results / "_analysis").mkdir(parents=True)
    (results / "_analysis" / "train_output.log").write_text(VJEPA2_LOG)
    assert harvest_running_ideas(results) == 0
    assert not (results / "_analysis" / "metrics.json").exists()
