import logging
from types import SimpleNamespace

from orze.engine.phases import _log_evo_score_if_changed
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
