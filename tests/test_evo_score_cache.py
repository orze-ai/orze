import logging
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from orze.engine.phases import (
    _evo_evidence_signature,
    _evo_score_signature,
    _evidence_relative_paths,
    _log_evo_score_if_changed,
)
from orze.reporting import search_path


def test_evo_score_rebuilds_only_when_lake_inputs_change(tmp_path, monkeypatch):
    lake = tmp_path / "idea_lake.db"
    lake.write_bytes(b"db-v1")
    calls = []

    def fake_build(path, cfg):
        calls.append((path, cfg))
        return {"research_efficiency": {
            "score": 50.0,
            "grade": "C",
            "evidence_qualification": {
                "mode": "verified_local_artifact",
                "primary_metric": "score",
                "fallback_metrics_allowed": False,
                "accepted": 3,
                "rejected": {"local_metrics_missing": 2},
            },
            "presentation": {
                "claim_scope": "internal_research_efficiency",
                "qualification_applied": True,
                "evidence_label": "verified local artifacts",
                "leaderboard_rank_comparable": False,
            },
        }}

    monkeypatch.setattr(search_path, "build_from_lake", fake_build)
    owner = SimpleNamespace()
    cfg = {"report": {"sort": "ascending"}}

    assert _log_evo_score_if_changed(owner, lake, cfg) is True
    assert _log_evo_score_if_changed(owner, lake, cfg) is False
    assert len(calls) == 1

    # WAL-only commits and report changes must both invalidate the cache.
    (tmp_path / "idea_lake.db-wal").write_bytes(b"commit")
    assert _log_evo_score_if_changed(owner, lake, cfg) is True
    cfg["report"]["sort"] = "descending"
    assert _log_evo_score_if_changed(owner, lake, cfg) is True
    assert len(calls) == 3


def _completed_lake(path, idea_id="idea-a"):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE ideas (idea_id TEXT, status TEXT)")
    conn.execute(
        "INSERT INTO ideas (idea_id, status) VALUES (?, 'completed')",
        (idea_id,),
    )
    conn.commit()
    conn.close()


def test_evidence_artifact_changes_invalidate_score_cache(
        tmp_path, monkeypatch):
    lake = tmp_path / "idea_lake.db"
    _completed_lake(lake)
    idea_dir = tmp_path / "results" / "idea-a"
    idea_dir.mkdir(parents=True)
    metrics = idea_dir / "metrics.json"
    external = idea_dir / "external.json"
    metrics.write_text('{"status":"COMPLETED","score":1}', encoding="utf-8")
    external.write_text('{"score":1}', encoding="utf-8")
    cfg = {
        "_env_ORZE_RESULTS_DIR": str(tmp_path / "results"),
        "report": {
            "primary_metric": "score",
            "columns": [
                {"key": "score", "source": "external.json:score"},
            ],
        },
    }
    calls = []

    def fake_build(path, config):
        calls.append(path)
        return {"research_efficiency": {
            "score": 1.0,
            "grade": "F",
            "evidence_qualification": {
                "mode": "verified_local_artifact",
                "primary_metric": "score",
                "fallback_metrics_allowed": False,
                "accepted": 1,
                "rejected": {},
            },
            "presentation": {
                "claim_scope": "internal_research_efficiency",
                "qualification_applied": True,
                "evidence_label": "verified local artifacts",
                "leaderboard_rank_comparable": False,
            },
        }}

    monkeypatch.setattr(search_path, "build_from_lake", fake_build)
    owner = SimpleNamespace()
    assert _log_evo_score_if_changed(owner, lake, cfg) is True
    assert _log_evo_score_if_changed(owner, lake, cfg) is False

    # Same-length content replacements must invalidate even when filesystem
    # timestamps and SQLite itself do not change.
    external.write_text('{"score":2}', encoding="utf-8")
    assert _log_evo_score_if_changed(owner, lake, cfg) is True
    metrics.unlink()
    assert _log_evo_score_if_changed(owner, lake, cfg) is True
    assert len(calls) == 3


def test_qualification_policy_and_benchmark_ledger_invalidate_signature(tmp_path):
    lake = tmp_path / "idea_lake.db"
    _completed_lake(lake)
    idea_dir = tmp_path / "results" / "idea-a"
    idea_dir.mkdir(parents=True)
    (idea_dir / "metrics.json").write_text(
        '{"status":"COMPLETED","score":1}', encoding="utf-8")
    (idea_dir / "receipt.json").write_text('{}', encoding="utf-8")
    (idea_dir / "_benchmark_evaluation.json").write_text(
        '{}', encoding="utf-8")
    orze_dir = tmp_path / ".orze"
    orze_dir.mkdir()
    ledger = orze_dir / "_benchmark_exposures.jsonl"
    ledger.write_text('{}\n', encoding="utf-8")
    cfg = {
        "_project_root": str(tmp_path),
        "_orze_dir": str(orze_dir),
        "_env_ORZE_RESULTS_DIR": str(tmp_path / "results"),
        "report": {
            "primary_metric": "score",
            "benchmark_contract": {"receipt": "receipt.json"},
        },
    }
    first = _evo_score_signature(lake, cfg)
    cfg["metric_validation"] = {"min_value": {"score": 0}}
    assert _evo_score_signature(lake, cfg) != first

    before_ledger = _evo_evidence_signature(lake, cfg)
    ledger.write_text('{"changed":true}\n', encoding="utf-8")
    assert _evo_evidence_signature(lake, cfg) != before_ledger


def test_evidence_path_inventory_excludes_escape_sources():
    paths = _evidence_relative_paths({
        "columns": [
            {"key": "safe", "source": "nested/eval.json:score"},
            {"key": "escape", "source": "../outside.json:score"},
            {"key": "absolute", "source": "/tmp/outside.json:score"},
        ],
        "benchmark_contract": {"receipt": "receipt.json"},
    })
    assert paths == (
        Path("_benchmark_evaluation.json"),
        Path("metrics.json"),
        Path("nested/eval.json"),
        Path("receipt.json"),
    )


def test_evidence_signature_does_not_follow_invalid_idea_id(tmp_path):
    lake = tmp_path / "idea_lake.db"
    _completed_lake(lake, "../outside")
    results = tmp_path / "results"
    results.mkdir()
    outside = tmp_path / "metrics.json"
    outside.write_text('{"status":"COMPLETED","score":1}', encoding="utf-8")
    cfg = {"_env_ORZE_RESULTS_DIR": str(results), "report": {}}
    before = _evo_evidence_signature(lake, cfg)
    outside.write_text('{"status":"COMPLETED","score":2}', encoding="utf-8")
    assert _evo_evidence_signature(lake, cfg) == before


def test_evo_score_log_never_presents_unqualified_number(
        tmp_path, monkeypatch, caplog):
    lake = tmp_path / "idea_lake.db"
    lake.write_bytes(b"db")

    def fake_build(path, cfg):
        return {"research_efficiency": {"score": 99.9, "grade": "A"}}

    monkeypatch.setattr(search_path, "build_from_lake", fake_build)
    with caplog.at_level(logging.WARNING, logger="orze"):
        assert _log_evo_score_if_changed(SimpleNamespace(), lake, {}) is True
    assert "Internal Evo Score unavailable" in caplog.text
    assert "99.9" not in caplog.text


def test_evo_score_log_discloses_internal_evidence_scope(
        tmp_path, monkeypatch, caplog):
    lake = tmp_path / "idea_lake.db"
    lake.write_bytes(b"db")

    def fake_build(path, cfg):
        return {"research_efficiency": {
            "score": 42.0,
            "grade": "D",
            "evidence_qualification": {
                "mode": "verified_local_artifact",
                "primary_metric": "score",
                "fallback_metrics_allowed": False,
                "accepted": 3,
                "rejected": {"local_metrics_missing": 2},
            },
            "presentation": {
                "claim_scope": "internal_research_efficiency",
                "qualification_applied": True,
                "evidence_label": "verified local artifacts",
                "leaderboard_rank_comparable": False,
            },
        }}

    monkeypatch.setattr(search_path, "build_from_lake", fake_build)
    with caplog.at_level(logging.INFO, logger="orze"):
        assert _log_evo_score_if_changed(SimpleNamespace(), lake, {}) is True
    assert "Internal Evo Score: 42.0 grade D" in caplog.text
    assert "evidence=verified local artifacts" in caplog.text
    assert "accepted=3 rejected=2" in caplog.text
    assert "leaderboard_rank_comparable=false" in caplog.text
