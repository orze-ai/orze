"""Stage completion is consistent across SQL/Python and owned transactions.

Real SQLite schema, transactions, competing connections and triggers are used.
The sole callback wrapper in the concurrency control runs the real validator,
then attempts a second connection's write at the validation/update boundary.
"""

import json
import sqlite3

import pytest

from orze.reporting import notification_evidence
from orze.reporting.catalog import load_catalog_snapshot
from orze.reporting.evidence import (
    authoritative_completed_idea_ids, authoritative_idea_lifecycle,
)

from test_stage_observer_write_boundary import project, IDEA, _idea_row


def test_metric_refresh_preserves_callers_uncommitted_transaction(project):
    p = project
    before = _idea_row(p)
    p.lake.conn.execute("UPDATE ideas SET title='caller-owned edit' WHERE idea_id=?", (IDEA,))
    assert p.lake.conn.in_transaction

    notification_evidence.refresh_metric_snapshot(p.lake, p.row)

    assert p.lake.conn.in_transaction, "refresh must not commit or roll back caller work"
    assert _idea_row(p)["title"] == "caller-owned edit"
    assert _idea_row(p)["eval_metrics"] == before["eval_metrics"]
    with sqlite3.connect(p.lake.db_path) as observer:
        assert observer.execute("SELECT title FROM ideas WHERE idea_id=?", (IDEA,)).fetchone()[0] == before["title"]
    p.lake.conn.rollback()
    assert _idea_row(p) == before


def test_failed_metric_update_rolls_back_only_its_owned_transaction(project):
    p = project
    p.lake.conn.executescript(
        "CREATE TABLE fixture_audit (value TEXT);"
        "CREATE TRIGGER reject_fixture_metric_update BEFORE UPDATE OF eval_metrics ON ideas "
        "BEGIN INSERT INTO fixture_audit VALUES ('attempted'); "
        "SELECT RAISE(ABORT, 'fixture storage rejection'); END;"
    )
    p.lake.conn.commit()
    before = _idea_row(p)

    notification_evidence.refresh_metric_snapshot(p.lake, p.row)

    assert not p.lake.conn.in_transaction
    assert _idea_row(p) == before
    assert p.lake.conn.execute("SELECT COUNT(*) FROM fixture_audit").fetchone()[0] == 0


def test_competing_stage_write_cannot_interleave_validation_and_metric_update(project, monkeypatch):
    p = project
    attempts = []
    validate = notification_evidence.validate_lifecycle_schema
    competitor = sqlite3.connect(p.lake.db_path, timeout=0)

    def validated_then_competing_write(connection):
        schema = validate(connection)
        assert connection.in_transaction
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            competitor.execute(
                "UPDATE idea_stage_state SET current_state='FAILED' "
                "WHERE idea_id=? AND stage='evaluation'", (IDEA,),
            )
        competitor.rollback()
        attempts.append("blocked")
        return schema

    try:
        monkeypatch.setattr(notification_evidence, "validate_lifecycle_schema", validated_then_competing_write)
        notification_evidence.refresh_metric_snapshot(p.lake, p.row)
        assert attempts == ["blocked"]
        assert json.loads(_idea_row(p)["eval_metrics"]) == {"score": 0, "penalty": -2}
        assert not p.lake.conn.in_transaction
        # The same real writer proceeds once the owned refresh transaction ends.
        competitor.execute(
            "UPDATE idea_stage_state SET current_state='FAILED' "
            "WHERE idea_id=? AND stage='evaluation'", (IDEA,),
        )
        competitor.commit()
    finally:
        competitor.close()


def test_future_custom_stage_does_not_gain_invented_completion_semantics(project):
    p = project
    p.lake.conn.execute(
        "INSERT INTO idea_stage_state (idea_id,stage,current_state,updated_at) "
        "VALUES (?, 'custom_analysis', 'FAILED', 'fixture')", (IDEA,),
    )
    p.lake.conn.commit()

    ids, reason = authoritative_completed_idea_ids(p.lake.db_path)
    snapshot = load_catalog_snapshot(p.lake.db_path)
    assert reason == "authoritative_lifecycle_loaded" and ids == {IDEA}
    assert snapshot.records[IDEA]["lifecycle_state"] == "COMPLETE"
    notification_evidence.refresh_metric_snapshot(p.lake, p.row)
    assert json.loads(_idea_row(p)["eval_metrics"]) == {"score": 0, "penalty": -2}


def _case_insensitive_state_storage(p, table):
    assert table in ("idea_state", "idea_stage_state")
    schema = p.lake.conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,),
    ).fetchone()[0]
    assert schema.count("current_state TEXT NOT NULL") == 1
    rows = [tuple(row) for row in p.lake.conn.execute(f"SELECT * FROM {table}")]
    indexes = [row[0] for row in p.lake.conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name=? "
        "AND sql IS NOT NULL", (table,),
    )]
    p.lake.conn.execute(f"DROP TABLE {table}")
    p.lake.conn.execute(schema.replace(
        "current_state TEXT NOT NULL", "current_state TEXT COLLATE NOCASE NOT NULL", 1,
    ))
    if rows:
        marks = ",".join("?" for _ in rows[0])
        p.lake.conn.executemany(f"INSERT INTO {table} VALUES ({marks})", rows)
    for index in indexes:
        p.lake.conn.execute(index)
    p.lake.conn.commit()


@pytest.mark.parametrize("table,state,accepted", [
    ("idea_stage_state", "complete", False),
    ("idea_stage_state", "COMPLETE", True),
    ("idea_state", "complete", False),
    ("idea_state", "COMPLETE", True),
])
def test_sql_and_python_agree_on_exact_completion_tokens_even_with_nocase_storage(
    project, table, state, accepted,
):
    p = project
    _case_insensitive_state_storage(p, table)
    where = " AND stage='training'" if table == "idea_stage_state" else ""
    p.lake.conn.execute(
        f"UPDATE {table} SET current_state=? WHERE idea_id=?" + where, (state, IDEA),
    )
    p.lake.conn.commit()
    before = _idea_row(p)

    ids, _ = authoritative_completed_idea_ids(p.lake.db_path)
    bounded, _ = authoritative_idea_lifecycle(p.lake.db_path, [IDEA])
    snapshot = load_catalog_snapshot(p.lake.db_path)
    assert snapshot.available
    assert (snapshot.records[IDEA]["lifecycle_state"] == "COMPLETE") is accepted
    assert bool(bounded) is accepted
    assert (IDEA in ids) is accepted, "SQL collation must not relax Python's exact stage/FSM tokens"

    notification_evidence.refresh_metric_snapshot(p.lake, p.row)
    if accepted:
        assert json.loads(_idea_row(p)["eval_metrics"]) == {"score": 0, "penalty": -2}
    else:
        assert _idea_row(p) == before
