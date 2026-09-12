"""Native authority and the deliberately unverified offline report API.

Real IdeaLake/artifacts exercise public update_report. The two metadata
cases separately specify a new output field/label: their baseline failures
are API acceptance failures, not evidence of an old ranking regression.
"""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from orze.core.config import DEFAULT_CONFIG
from orze.idea_lake import IdeaLake
from orze.reporting import leaderboard
from orze.reporting.evidence import (
    authoritative_completed_idea_ids,
    qualify_authoritative_report_evidence_with_identity,
)


@pytest.fixture
def project(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(str(tmp_path / "authority.db"))
    cfg = {
        "_project_root": str(tmp_path),
        "results_dir": str(results),
        "idea_lake_db": str(lake.db_path),
        "report": {
            "primary_metric": "score",
            "sort": "ascending",
            "columns": [{"key": "score"}],
        },
    }
    try:
        yield SimpleNamespace(results=results, lake=lake, cfg=cfg, ideas={})
    finally:
        lake.close()


def _publish(p, idea_id="idea-native", *, status="completed", **values):
    folder = p.results / idea_id
    folder.mkdir()
    metrics = {"status": "COMPLETED", "score": 0, **values}
    (folder / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    p.lake.insert(idea_id, idea_id, "seed: 3", "", status=status,
                  eval_metrics=metrics)
    p.ideas[idea_id] = {"title": idea_id, "config": {"seed": 3}}
    return folder


def _oracle(p, idea_id="idea-native"):
    completed, reason = authoritative_completed_idea_ids(p.lake.db_path)
    assert reason == "authoritative_lifecycle_loaded"
    return qualify_authoritative_report_evidence_with_identity(
        idea_id, p.results, dict(p.cfg, _env_ORZE_RESULTS_DIR=str(p.results)),
        completed,
    )


def _payload(results, filename="_leaderboard.json"):
    return json.loads((results / filename).read_text(encoding="utf-8"))


@pytest.mark.parametrize("harvest_source", ["missing", "contradictory"])
def test_native_empty_columns_do_not_acquire_harvest_metric_sources(
    project, harvest_source,
):
    p = project
    p.cfg["report"]["columns"] = []
    p.cfg["metric_harvest"] = {"columns": [
        {"key": "score", "source": "harvest.json:proxy"},
    ]}
    folder = _publish(p)
    if harvest_source == "contradictory":
        (folder / "harvest.json").write_text('{"proxy": 900}', encoding="utf-8")
    cfg_before = deepcopy(p.cfg)
    assert _oracle(p)[2] == 0

    rows = leaderboard.update_report(p.results, p.ideas, p.cfg, lake=p.lake)

    assert [(row["id"], row["primary_val"]) for row in rows] == [("idea-native", 0)]
    assert _payload(p.results)["top"][0]["metric_value"] == 0
    assert p.cfg == cfg_before


def test_native_declared_default_columns_are_not_replaced_by_harvest(project):
    p = project
    p.cfg["report"]["primary_metric"] = "test_accuracy"
    p.cfg["report"]["columns"] = deepcopy(DEFAULT_CONFIG["report"]["columns"])
    p.cfg["metric_harvest"] = {"columns": [
        {"key": "test_accuracy", "source": "harvest.json:proxy"},
    ]}
    folder = _publish(p, test_accuracy=0, test_loss=2, training_time=3)
    (folder / "harvest.json").write_text('{"proxy": 900}', encoding="utf-8")
    cfg_before = deepcopy(p.cfg)
    assert _oracle(p)[2] == 0

    rows = leaderboard.update_report(p.results, p.ideas, p.cfg, lake=p.lake)

    assert [row["primary_val"] for row in rows] == [0]
    assert rows[0]["values"]["test_loss"] == 2
    assert _payload(p.results)["top"][0]["metric_value"] == 0
    assert p.cfg == cfg_before


def test_native_coverage_is_not_satisfied_by_undeclared_harvest_columns(project):
    p = project
    p.cfg["report"].update(columns=[], dataset_keys=[], min_datasets=2)
    p.cfg["metric_harvest"] = {"columns": [
        {"key": "subset_a"}, {"key": "subset_b"},
    ]}
    _publish(p, subset_a=1, subset_b=2)
    assert _oracle(p)[2] is None
    assert _oracle(p)[3] == "metric_coverage_below_min:0/2"

    rows = leaderboard.update_report(p.results, p.ideas, p.cfg, lake=p.lake)

    assert rows == []
    assert _payload(p.results)["top"] == []
    assert _payload(p.results)["evidence_qualification"]["rejected"] == {
        "metric_coverage_below_min:0/2": 1,
    }


def test_configured_database_supplies_history_without_lake_or_inbox(project):
    p = project
    _publish(p, "idea-historical", score=-2)
    before = p.lake.conn.execute("SELECT * FROM ideas ORDER BY idea_id").fetchall()
    assert _oracle(p, "idea-historical")[2] == -2

    rows = leaderboard.update_report(p.results, {}, p.cfg, lake=None)

    assert [(row["id"], row["primary_val"]) for row in rows] == [
        ("idea-historical", -2),
    ]
    assert _payload(p.results)["top"][0]["idea_id"] == "idea-historical"
    assert p.lake.conn.execute("SELECT * FROM ideas ORDER BY idea_id").fetchall() == before


def _offline_project(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    cfg = {"report": {
        "primary_metric": "score", "sort": "ascending",
        "columns": [{"key": "score"}],
        "views": [{"name": "kept", "filter": {"group": "keep"}}],
    }}
    ideas = {}
    for idea_id, value in (("idea-zero", 0), ("idea-negative", -2)):
        folder = results / idea_id
        folder.mkdir()
        (folder / "metrics.json").write_text(
            json.dumps({"status": "COMPLETED", "score": value}), encoding="utf-8")
        (folder / "resolved_config.yaml").write_text("group: keep\n", encoding="utf-8")
        ideas[idea_id] = {"title": idea_id}
    return results, ideas, cfg


def test_legacy_offline_retains_order_and_unchanged_evidence_fast_cache(
    tmp_path, monkeypatch,
):
    results, ideas, cfg = _offline_project(tmp_path)
    first = leaderboard.update_report(results, ideas, cfg)
    assert [row["primary_val"] for row in first] == [-2, 0]
    outputs = [results / name for name in (
        "report.md", "_results_cache.json", "_leaderboard.json",
        "_leaderboard_kept.json", "_leaderboard_views.json",
    )]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in outputs}

    def unexpected_hash(*args, **kwargs):
        pytest.fail("unchanged offline evidence should retain its metadata fast path")

    monkeypatch.setattr(leaderboard, "_evidence_content_hash", unexpected_hash)
    second = leaderboard.update_report(results, ideas, cfg)

    assert [row["primary_val"] for row in second] == [-2, 0]
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in outputs} == before
    assert not list(tmp_path.rglob("*.db"))


@pytest.mark.parametrize("output", ["json", "markdown"])
def test_legacy_lifecycle_authority_metadata_is_explicit(tmp_path, output):
    """New metadata acceptance, recorded separately from behavior red counts."""
    results, ideas, cfg = _offline_project(tmp_path)
    rows = leaderboard.update_report(results, ideas, cfg)
    assert [row["primary_val"] for row in rows] == [-2, 0]
    if output == "markdown":
        text = (results / "report.md").read_text(encoding="utf-8")
        assert "unverified_local_artifact" in text
    else:
        for name in ("_leaderboard.json", "_leaderboard_kept.json"):
            payload = _payload(results, name)
            assert [row["metric_value"] for row in payload["top"]] == [-2, 0]
            assert payload["lifecycle_authority"] == "unverified_local_artifact"
            assert payload["evidence_qualification"]["lifecycle_authority"] == (
                "unverified_local_artifact")


def test_explicit_empty_database_declaration_never_silently_becomes_offline(tmp_path):
    results, ideas, cfg = _offline_project(tmp_path)
    cfg["idea_lake_db"] = None

    rows = leaderboard.update_report(results, ideas, cfg)

    assert rows == []
    assert _payload(results)["top"] == []
    assert not list(tmp_path.rglob("*.db"))


def test_native_pipeline_counts_do_not_become_rank_qualification_counts(project):
    p = project
    _publish(p, "idea-valid", score=0)
    _publish(p, "idea-vetoed", score=-10, honest=False)
    _publish(p, "idea-conflict", score=-20)
    _publish(p, "idea-failed", status="failed", score=-30)
    p.lake.conn.execute(
        "UPDATE ideas SET status='queued' WHERE idea_id='idea-conflict'")
    p.lake.conn.commit()
    assert p.lake.get_fsm_state("idea-conflict") == "COMPLETE"

    rows = leaderboard.update_report(p.results, {}, p.cfg, lake=p.lake)

    assert [row["id"] for row in rows] == ["idea-valid"]
    payload = _payload(p.results)
    assert payload["evidence_qualification"]["accepted"] == 1
    assert payload["evidence_qualification"]["rejected"] == {
        "local_evidence_declared_non_honest": 1,
    }
    text = (p.results / "report.md").read_text(encoding="utf-8")
    assert "| 4 | 3 | 1 | 0 | 0 |" in text
