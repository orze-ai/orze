"""BUG #2 regression test.

Config dedup compares an ingest-time hash (over user OVERRIDES, via
engine.orchestrator._config_override_hash -> core.integrity.hash_config on
idea["config"]) against a completion-time hash stored by the reporter's
_notify_finished. The bug: completion stored hash_config(FULL
resolved_config.yaml) — a different, larger key-set — so the stored hash
never equalled the ingest override hash and dedup never fired. Completion
also silently skipped storing when resolved_config.yaml was absent.

This test asserts the two sides now hash the SAME canonical key-set
(overrides) so a completed idea's stored hash matches what ingest checks.
"""
import json
from pathlib import Path

import pytest

from orze.core.integrity import hash_config

# The reporter class name is resolved dynamically (the class that owns
# _notify_finished / _recover_overrides), so this test does not depend on the
# concrete class name.
import orze.reporting.leaderboard as lb


def _reporter_class():
    for name in dir(lb):
        obj = getattr(lb, name)
        if isinstance(obj, type) and hasattr(obj, "_notify_finished"):
            return obj
    raise AssertionError("no reporter class with _notify_finished found")


def _make_reporter(results_dir):
    cls = _reporter_class()
    rep = cls.__new__(cls)            # bypass __init__ (needs heavy ctx)
    rep.results_dir = Path(results_dir)
    rep.lake = None
    return rep


def test_completion_stores_override_hash_matching_ingest(tmp_path):
    overrides = {"learning_rate": 1e-4, "epochs": 3}
    idea_id = "idea-0001"
    d = tmp_path / idea_id
    d.mkdir()
    # metrics.json marks the run COMPLETED.
    (d / "metrics.json").write_text(json.dumps({"status": "COMPLETED"}))
    # resolved_config.yaml is the FULL merged config (base + overrides).
    import yaml
    (d / "resolved_config.yaml").write_text(yaml.safe_dump(
        {"learning_rate": 1e-4, "epochs": 3,
         "seed": 42, "model": "base", "batch": 8, "weight_decay": 0.0}))

    rep = _make_reporter(tmp_path)

    stored = {}
    def save_hash_fn(iid, cfg_for_hash):
        stored[iid] = hash_config(cfg_for_hash)

    ideas = {idea_id: {"title": "t", "config": dict(overrides)}}
    rep._notify_finished(
        idea_id, 0, {"base_config": "nonexistent.yaml"}, "wer",
        row_lookup={}, rank_lookup={}, leaderboard=[], view_lbs={},
        ideas=ideas, save_config_hash_fn=save_hash_fn)

    # The ingest side hashes the OVERRIDES.
    ingest_hash = hash_config(overrides)
    assert idea_id in stored, "completion did not store any dedup hash"
    assert stored[idea_id] == ingest_hash, (
        "completion-stored hash != ingest override hash; dedup will never "
        f"fire (stored={stored[idea_id]} ingest={ingest_hash})"
    )


def test_completion_recovers_overrides_when_idea_config_absent(tmp_path):
    """When the in-memory idea record carries no config (archived/stub row),
    completion must recover overrides from resolved_config.yaml minus the base
    config and store the SAME override hash (no silent skip)."""
    overrides = {"learning_rate": 2e-4}
    idea_id = "idea-0002"
    d = tmp_path / idea_id
    d.mkdir()
    (d / "metrics.json").write_text(json.dumps({"status": "COMPLETED"}))
    import yaml
    base = {"learning_rate": 1e-3, "seed": 42, "model": "base"}
    (d / "resolved_config.yaml").write_text(yaml.safe_dump(
        {**base, "learning_rate": 2e-4}))
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base))

    rep = _make_reporter(tmp_path)
    stored = {}
    def save_hash_fn(iid, cfg_for_hash):
        stored[iid] = hash_config(cfg_for_hash)

    # ideas record has NO 'config' key.
    ideas = {idea_id: {"title": "t"}}
    rep._notify_finished(
        idea_id, 0, {"base_config": str(base_path)}, "wer",
        row_lookup={}, rank_lookup={}, leaderboard=[], view_lbs={},
        ideas=ideas, save_config_hash_fn=save_hash_fn)

    assert idea_id in stored, (
        "completion silently skipped storing a dedup hash when the idea "
        "record carried no config"
    )
    assert stored[idea_id] == hash_config(overrides)
