"""Existing stage rows are facts, not optional-history absence.

Public catalog observation and the notification metric-mirror write boundary
use a real temporary SQLite lake. Notification candidate values and identity
come from real source qualification before deliberate stage/schema drift.
No lifecycle/evidence helpers, filesystem or SQLite operations are mocked.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from orze.idea_lake import IdeaLake
from orze.reporting.catalog import load_catalog_snapshot
from orze.reporting.notification_evidence import (
    qualified_notification_rows, refresh_metric_snapshot,
)


IDEA = "idea-stage-boundary"


@pytest.fixture
def project(tmp_path):
    results = tmp_path / "results"
    folder = results / IDEA
    folder.mkdir(parents=True)
    (folder / "metrics.json").write_text(
        json.dumps({"status": "COMPLETED", "score": 999}), encoding="utf-8")
    (folder / "evaluation.json").write_text(
        json.dumps({"status": "COMPLETED", "score": 0, "penalty": -2}),
        encoding="utf-8")
    lake = IdeaLake(tmp_path / "ideas.db")
    lake.insert(
        IDEA, "Preserved archived title", "seed: 13", "Preserved original idea",
        status="queued", priority="high", eval_metrics={"score": 123},
    )
    for old, new in (("QUEUED", "CLAIMED"), ("CLAIMED", "IN_PROGRESS"),
                     ("IN_PROGRESS", "COMPLETE")):
        assert lake.record_state_transition(IDEA, old, new, "real fixture lifecycle")
    cfg = {
        "_project_root": str(tmp_path), "results_dir": str(results),
        "idea_lake_db": str(lake.db_path),
        "report": {
            "primary_metric": "score", "secondary_metric": "penalty",
            "sort": "ascending", "columns": [
                {"key": "score", "source": "evaluation.json:score"},
                {"key": "penalty", "source": "evaluation.json:penalty"},
            ],
        },
    }
    # Capture a genuine previously qualified row, not a forged cache identity.
    rows = qualified_notification_rows(results, cfg, [{"id": IDEA}], lake=lake)
    assert len(rows) == 1 and rows[0]["evidence_identity"]
    assert rows[0]["values"] == {"score": 0, "penalty": -2}
    assert load_catalog_snapshot(lake.db_path).records[IDEA]["lifecycle_state"] == "COMPLETE"
    p = SimpleNamespace(root=tmp_path, lake=lake, cfg=cfg, results=results,
                        row=rows[0])
    try:
        yield p
    finally:
        lake.close()


def _files(p):
    # Includes the DB and any persistent journal/WAL sidecar, as well as all
    # source evidence. The fixture uses the framework's shared-DB policy.
    return {str(path.relative_to(p.root)): path.read_bytes()
            for path in p.root.rglob("*") if path.is_file()}


def _idea_row(p):
    row = p.lake.conn.execute("SELECT * FROM ideas WHERE idea_id=?", (IDEA,)).fetchone()
    return dict(row)


def _nullable_historical_stage_schema(p):
    """Relax only current_state nullability, retaining actual columns/PK/indexes."""
    conn = p.lake.conn
    original_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='idea_stage_state'",
    ).fetchone()[0]
    assert original_sql.count("current_state TEXT NOT NULL") == 1
    columns_before = [tuple(row) for row in conn.execute("PRAGMA table_info(idea_stage_state)")]
    indexes = [row[0] for row in conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' "
        "AND tbl_name='idea_stage_state' AND sql IS NOT NULL")]
    rows = [tuple(row) for row in conn.execute("SELECT * FROM idea_stage_state")]
    conn.execute("DROP TABLE idea_stage_state")
    conn.execute(original_sql.replace("current_state TEXT NOT NULL", "current_state TEXT", 1))
    marks = ",".join("?" for _ in columns_before)
    conn.executemany(f"INSERT INTO idea_stage_state VALUES ({marks})", rows)
    for sql in indexes:
        conn.execute(sql)
    conn.commit()
    columns_after = [tuple(row) for row in conn.execute("PRAGMA table_info(idea_stage_state)")]
    expected = [(*row[:3], 0, *row[4:]) if row[1] == "current_state" else row
                for row in columns_before]
    assert columns_after == expected
    assert [(row[1], row[5]) for row in columns_after if row[5]] == [
        ("idea_id", 1), ("stage", 2)]


def _set_stage(p, stage, state):
    changed = p.lake.conn.execute(
        "UPDATE idea_stage_state SET current_state=? WHERE idea_id=? AND stage=?",
        (state, IDEA, stage),
    )
    assert changed.rowcount == 1
    p.lake.conn.commit()


@pytest.mark.parametrize("stage", ["training", "evaluation"])
def test_catalog_distinguishes_present_null_stage_from_absent_historical_stage(project, stage):
    p = project
    _nullable_historical_stage_schema(p)
    _set_stage(p, stage, None)
    before = _files(p)

    snapshot = load_catalog_snapshot(p.lake.db_path)

    assert _files(p) == before
    assert snapshot.available
    row = snapshot.records[IDEA]
    assert row["lifecycle_state"] == "UNKNOWN"
    assert row["lifecycle_reason"] != "lifecycle_agreed"
    assert snapshot.unknown_count == 1


@pytest.mark.parametrize("stage,state", [
    ("training", "FAILED"), ("evaluation", "FAILED"),
    ("training", None), ("evaluation", None),
])
def test_metric_refresh_rechecks_recorded_stages_after_real_source_qualification(
    project, stage, state,
):
    p = project
    if state is None:
        _nullable_historical_stage_schema(p)
    _set_stage(p, stage, state)
    before_row, before_files = _idea_row(p), _files(p)

    refresh_metric_snapshot(p.lake, p.row)

    assert _idea_row(p) == before_row
    assert _files(p) == before_files


@pytest.mark.parametrize("history", ["missing_table", "missing_rows", "complete", "skipped_eval"])
def test_optional_history_and_valid_terminal_stages_allow_only_metric_mirror_refresh(
    project, history,
):
    p = project
    if history == "missing_table":
        p.lake.conn.execute("DROP TABLE idea_stage_state")
    elif history == "missing_rows":
        p.lake.conn.execute("DELETE FROM idea_stage_state WHERE idea_id=?", (IDEA,))
    elif history == "complete":
        _set_stage(p, "evaluation", "COMPLETE")
    else:
        assert p.lake.conn.execute(
            "SELECT current_state FROM idea_stage_state WHERE idea_id=? AND stage='evaluation'",
            (IDEA,),
        ).fetchone()[0] == "SKIPPED"
    p.lake.conn.commit()
    before_files = _files(p)
    before_row = _idea_row(p)
    nonmetric_before = dict(before_row)
    nonmetric_before.pop("eval_metrics")

    snapshot = load_catalog_snapshot(p.lake.db_path)

    assert snapshot.available and snapshot.records[IDEA]["lifecycle_state"] == "COMPLETE"
    assert snapshot.unknown_count == 0
    assert _files(p) == before_files
    # Historical absence is compatibility, not a requirement to manufacture
    # stage rows during read or update an unrelated part of the archived idea.
    refresh_metric_snapshot(p.lake, p.row)

    after_row = _idea_row(p)
    assert json.loads(after_row.pop("eval_metrics")) == {"score": 0, "penalty": -2}
    assert after_row == nonmetric_before
    if history == "missing_table":
        assert p.lake.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='idea_stage_state'",
        ).fetchone() is None
    elif history == "missing_rows":
        assert p.lake.conn.execute(
            "SELECT COUNT(*) FROM idea_stage_state WHERE idea_id=?", (IDEA,),
        ).fetchone()[0] == 0
    database = str(Path(p.lake.db_path).relative_to(p.root))
    assert {key: value for key, value in _files(p).items() if key != database} == {
        key: value for key, value in before_files.items() if key != database}


@pytest.mark.parametrize("broken", ["malformed_stage_schema", "duplicate_stage_identity"])
def test_structurally_invalid_stage_read_discards_whole_catalog_and_cannot_refresh_stale_row(
    project, broken,
):
    p = project
    # A second valid identity ensures failure cannot silently become a partial
    # catalog. This test targets catalog and the write boundary, not the
    # independently tested authoritative-ID/report consumers.
    p.lake.insert("idea-other-valid", "Other valid item", "seed: 7", "other",
                  status="completed")
    p.lake.conn.execute("DROP TABLE idea_stage_state")
    if broken == "malformed_stage_schema":
        p.lake.conn.execute(
            "CREATE TABLE idea_stage_state (idea_id TEXT NOT NULL, stage TEXT NOT NULL, "
            "updated_at TEXT NOT NULL, PRIMARY KEY (idea_id, stage))")
    else:
        # Deliberate malformed historical schema without the identity PK is
        # needed to represent duplicate rows; no current NOT NULL is violated.
        p.lake.conn.execute(
            "CREATE TABLE idea_stage_state (idea_id TEXT NOT NULL, stage TEXT NOT NULL, "
            "current_state TEXT NOT NULL, updated_at TEXT NOT NULL, "
            "started_at TEXT, terminal_at TEXT)")
        p.lake.conn.executemany(
            "INSERT INTO idea_stage_state VALUES (?, 'evaluation', 'COMPLETE', 'fixture', NULL, NULL)",
            [(IDEA,), (IDEA,)],
        )
    p.lake.conn.commit()
    before_row, before_files = _idea_row(p), _files(p)

    snapshot = load_catalog_snapshot(p.lake.db_path)

    assert not snapshot.available
    assert snapshot.get_metadata_index() == {}
    assert snapshot.get_lifecycle_counts() == {}
    assert _files(p) == before_files

    refresh_metric_snapshot(p.lake, p.row)

    assert _idea_row(p) == before_row
    assert _files(p) == before_files
