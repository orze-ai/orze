"""Configured-DB reports discover the full catalog without a writable Lake."""
import json
from pathlib import Path

import pytest

from orze.idea_lake import IdeaLake
from orze.reporting.leaderboard import update_report


@pytest.fixture
def project(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    database = tmp_path / "catalog.sqlite3"
    lake = IdeaLake(database)
    for idea_id, status in (("idea-complete", "completed"),
                            ("idea-queued", "queued"),
                            ("idea-failed", "failed"),
                            ("idea-archived", "archived")):
        lake.insert(idea_id, idea_id, "seed: 13", "", status=status)
    folder = results / "idea-complete"
    folder.mkdir()
    (folder / "metrics.json").write_text('{"status":"COMPLETED","score":4}')
    cfg = {
        "results_dir": str(results), "idea_lake_db": str(database),
        "report": {"primary_metric": "score", "sort": "ascending",
                   "columns": [{"key": "score", "source": "metrics.json:score"}]},
    }
    try:
        yield results, lake, cfg
    finally:
        lake.close()


def test_database_only_report_counts_and_lists_noncompleted_catalog(project):
    results, lake, cfg = project
    before = Path(lake.db_path).read_bytes()

    ranked = update_report(results, {}, cfg)

    report = (results / "report.md").read_text()
    assert "| 3 | 1 | 1 | 0 | 1 |" in report
    assert "## Queue (1 ideas)" in report
    assert "idea-queued" in report.split("## Queue", 1)[1]
    assert [(row["id"], row["primary_val"]) for row in ranked] == [("idea-complete", 4)]
    assert Path(lake.db_path).read_bytes() == before


def test_native_catalog_does_not_consult_malformed_legacy_archive_index(project):
    results, lake, cfg = project
    legacy = results / "_archived_index.json"
    legacy.write_text('["not a native authority"]')
    before = legacy.read_bytes(), Path(lake.db_path).read_bytes()

    ranked = update_report(results, {}, cfg)

    assert [row["id"] for row in ranked] == ["idea-complete"]
    assert (legacy.read_bytes(), Path(lake.db_path).read_bytes()) == before


def test_invalid_stage_catalog_cannot_silently_supply_native_report_authority(project):
    results, lake, cfg = project
    lake.conn.execute("DROP TABLE idea_stage_state")
    lake.conn.execute("CREATE TABLE idea_stage_state (idea_id TEXT)")
    lake.conn.commit()
    before = Path(lake.db_path).read_bytes()

    ranked = update_report(results, {}, cfg)

    assert ranked == []
    payload = json.loads((results / "_leaderboard.json").read_text())
    assert payload["lifecycle_authority"] == "unavailable_idea_lake"
    assert payload["pipeline_scope"] == "unavailable_lake_catalog"
    assert Path(lake.db_path).read_bytes() == before
