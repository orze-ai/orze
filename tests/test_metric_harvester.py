"""Tests for orze.engine.metric_harvester — mid-run metric extraction."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orze.engine.metric_harvester import (
    HARVEST_SOURCE,
    PATTERN_CACHE_FILENAME,
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


def _setup_exotic_idea(tmp_path):
    """Idea whose log format matches no default pattern."""
    project = tmp_path / "project"
    project.mkdir()
    train_script = project / "train_weird.py"
    train_script.write_text("# fake train script\n")

    results = project / "results"
    idea_dir = results / "idea-exotic"
    idea_dir.mkdir(parents=True)
    (idea_dir / "idea_config.yaml").write_text(
        "lr: 1e-5\ntrain_script: train_weird.py\n")
    # Log format the built-in regex can't parse: prose sentences with the
    # metric embedded mid-string, not a `metric=VALUE` kv.
    (idea_dir / "train_output.log").write_text(
        "Finished evaluation pass on fold 1 — NDCG score came in at 0.72.\n"
        "Epoch 1/5\n"
        "Finished evaluation pass on fold 2 — NDCG score came in at 0.81.\n"
        "Epoch 2/5\n"
    )
    return project, results, idea_dir, train_script


def test_harvest_calls_inferrer_when_regex_misses(tmp_path):
    project, results, idea_dir, train_script = _setup_exotic_idea(tmp_path)
    calls = []

    def fake_inferrer(script_path, log_text, metric_name):
        calls.append((script_path.name, metric_name))
        return [r"NDCG score came in at ([0-9.]+)"]

    n = harvest_running_ideas(
        results, primary_metric="ndcg", pattern_inferrer=fake_inferrer,
        train_script=train_script)

    assert n == 1
    assert calls == [("train_weird.py", "ndcg")]
    data = json.loads((idea_dir / "metrics.json").read_text())
    assert data["best_ndcg"] == pytest.approx(0.81)
    assert data["last_epoch"] == 2


def test_harvest_caches_inferred_patterns_per_train_script(tmp_path):
    project, results, idea_dir, train_script = _setup_exotic_idea(tmp_path)
    calls = []

    def fake_inferrer(script_path, log_text, metric_name):
        calls.append(1)
        return [r"NDCG score came in at ([0-9.]+)"]

    # First harvest triggers the inferrer and caches.
    harvest_running_ideas(
        results, primary_metric="ndcg", pattern_inferrer=fake_inferrer,
        train_script=train_script)
    assert len(calls) == 1

    # Cache file exists and contains the learned pattern.
    cache = json.loads((results / PATTERN_CACHE_FILENAME).read_text())
    assert "train_weird.py" in cache
    assert "ndcg" in cache["train_weird.py"]["patterns_by_metric"]

    # Second harvest with a different idea for the same script must
    # NOT call the inferrer again.
    idea2 = results / "idea-exotic2"
    idea2.mkdir()
    (idea2 / "idea_config.yaml").write_text("train_script: train_weird.py\n")
    (idea2 / "train_output.log").write_text(
        "Finished evaluation pass on fold 1 — NDCG score came in at 0.77.\n"
        "Epoch 1/5\n")

    harvest_running_ideas(
        results, primary_metric="ndcg", pattern_inferrer=fake_inferrer,
        train_script=train_script)
    assert len(calls) == 1  # still one — served from cache
    data = json.loads((idea2 / "metrics.json").read_text())
    assert data["best_ndcg"] == pytest.approx(0.77)


def test_harvest_inferrer_not_retried_when_it_returns_empty(tmp_path):
    project, results, idea_dir, train_script = _setup_exotic_idea(tmp_path)
    calls = []

    def empty_inferrer(script_path, log_text, metric_name):
        calls.append(1)
        return []  # Couldn't figure out patterns.

    harvest_running_ideas(
        results, primary_metric="ndcg", pattern_inferrer=empty_inferrer,
        train_script=train_script)
    assert len(calls) == 1

    # Another idea with the same (unmodified) train script — must not retry.
    idea2 = results / "idea-exotic2"
    idea2.mkdir()
    (idea2 / "idea_config.yaml").write_text("train_script: train_weird.py\n")
    (idea2 / "train_output.log").write_text("same exotic format 0.9 something\n")

    harvest_running_ideas(
        results, primary_metric="ndcg", pattern_inferrer=empty_inferrer,
        train_script=train_script)
    assert len(calls) == 1  # still one; cached the empty result


def test_harvest_reinvokes_inferrer_when_train_script_edited(tmp_path):
    import time
    project, results, idea_dir, train_script = _setup_exotic_idea(tmp_path)
    calls = []

    def fake_inferrer(script_path, log_text, metric_name):
        calls.append(1)
        return [r"NDCG score came in at ([0-9.]+)"]

    harvest_running_ideas(
        results, primary_metric="ndcg", pattern_inferrer=fake_inferrer,
        train_script=train_script)
    assert len(calls) == 1

    # Simulate a script edit — bump mtime past the cache tolerance.
    time.sleep(0.7)
    train_script.write_text("# edited\n")

    # New idea, same (but modified) train script.
    idea2 = results / "idea-exotic2"
    idea2.mkdir()
    (idea2 / "idea_config.yaml").write_text("train_script: train_weird.py\n")
    (idea2 / "train_output.log").write_text(
        "Finished evaluation pass on fold 1 — NDCG score came in at 0.88.\n"
        "Epoch 1/5\n")

    harvest_running_ideas(
        results, primary_metric="ndcg", pattern_inferrer=fake_inferrer,
        train_script=train_script)
    assert len(calls) == 2  # cache was invalidated by mtime change


def test_harvest_without_inferrer_leaves_exotic_log_untouched(tmp_path):
    project, results, idea_dir, train_script = _setup_exotic_idea(tmp_path)
    n = harvest_running_ideas(results, primary_metric="ndcg")
    assert n == 0
    assert not (idea_dir / "metrics.json").exists()
    # Cache file must not be created if no inferrer is used.
    assert not (results / PATTERN_CACHE_FILENAME).exists()


def test_harvest_skips_inferrer_on_warmup_only_log(tmp_path):
    """Log with no eval-like signal must not burn an LLM call."""
    project = tmp_path / "project"
    project.mkdir()
    train_script = project / "train_sparse.py"
    train_script.write_text("# stub\n")
    results = project / "results"
    idea_dir = results / "idea-warmup"
    idea_dir.mkdir(parents=True)
    (idea_dir / "idea_config.yaml").write_text(
        "train_script: train_sparse.py\n")
    # Early training — no eval, no metric numbers yet.
    (idea_dir / "train_output.log").write_text(
        "Loading checkpoint...\n"
        "Building model architecture...\n"
        "Starting training loop...\n"
    )

    calls = []

    def spy_inferrer(script_path, log_text, metric_name):
        calls.append(1)
        return ["this_should_never_be_cached\\s*([0-9.]+)"]

    n = harvest_running_ideas(
        results, primary_metric="ndcg", pattern_inferrer=spy_inferrer,
        train_script=train_script)
    assert n == 0
    assert calls == []  # gated by _log_has_training_signal
    assert not (results / PATTERN_CACHE_FILENAME).exists()


def test_cached_empty_entry_expires_and_retries(tmp_path):
    """Empty cache entry older than TTL must allow re-inference."""
    from orze.engine import metric_harvester as mh
    project, results, idea_dir, train_script = _setup_exotic_idea(tmp_path)
    calls = []

    def inferrer_v1(script_path, log_text, metric_name):
        calls.append("v1")
        return []  # empty — simulates "log too early"

    def inferrer_v2(script_path, log_text, metric_name):
        calls.append("v2")
        return [r"NDCG score came in at ([0-9.]+)"]

    # First harvest: inferrer returns empty, cached with timestamp.
    harvest_running_ideas(
        results, primary_metric="ndcg", pattern_inferrer=inferrer_v1,
        train_script=train_script)
    assert calls == ["v1"]

    # Monkey-patch time to jump past the TTL.
    original_time = mh.time.time
    try:
        mh.time.time = lambda: original_time() + mh._EMPTY_TTL_SECONDS + 60
        harvest_running_ideas(
            results, primary_metric="ndcg", pattern_inferrer=inferrer_v2,
            train_script=train_script)
    finally:
        mh.time.time = original_time

    assert calls == ["v1", "v2"]
    data = json.loads((idea_dir / "metrics.json").read_text())
    assert data["best_ndcg"] == pytest.approx(0.81)


def test_cached_empty_entry_honored_within_ttl(tmp_path):
    """Empty cache entry within TTL must NOT re-invoke."""
    project, results, idea_dir, train_script = _setup_exotic_idea(tmp_path)
    calls = []

    def inferrer(script_path, log_text, metric_name):
        calls.append(1)
        return []

    harvest_running_ideas(
        results, primary_metric="ndcg", pattern_inferrer=inferrer,
        train_script=train_script)
    assert calls == [1]

    # Same immediate retry — within TTL.
    harvest_running_ideas(
        results, primary_metric="ndcg", pattern_inferrer=inferrer,
        train_script=train_script)
    assert calls == [1]


def test_store_patterns_records_learned_at_timestamp(tmp_path):
    from orze.engine.metric_harvester import _store_patterns
    ts = tmp_path / "s.py"
    ts.write_text("pass\n")
    cache = {}
    _store_patterns(cache, ts, "map", ["pat"], now=12345.0)
    assert cache["s.py"]["learned_at"]["map"] == 12345.0
