"""Draft admission regression: SQLite collation cannot redefine a task ID."""
import pytest

from orze.idea_lake import IdeaLake


@pytest.fixture
def lake(tmp_path):
    instance = IdeaLake(str(tmp_path / "lake.db"))
    connection = instance.conn
    # Preserve the actual schema, constraints and indexes; only simulate the
    # historical primary-key collation. No production migration is invoked.
    schema = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='ideas'",
    ).fetchone()[0]
    indexes = [row[0] for row in connection.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='ideas' AND sql IS NOT NULL",
    )]
    triggers = [row[0] for row in connection.execute(
        "SELECT sql FROM sqlite_master WHERE type='trigger' AND tbl_name='ideas'",
    )]
    assert "idea_id TEXT PRIMARY KEY" in schema
    connection.execute("DROP TABLE ideas")
    connection.execute(schema.replace("idea_id TEXT PRIMARY KEY", "idea_id TEXT COLLATE NOCASE PRIMARY KEY", 1))
    for statement in [*indexes, *triggers]:
        connection.execute(statement)
    connection.commit()
    try:
        yield instance
    finally:
        instance.close()


def _proposal(lake, idea_id, **extra):
    return lake.insert(
        idea_id, "Proposal", "seed: 13\n", "identical source fields",
        status="queued", priority="medium", if_absent=True, **extra,
    )


def _snapshot(lake):
    return {
        table: [tuple(row) for row in lake.conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
        for table in ("ideas", "idea_state", "idea_transitions", "idea_stage_state", "idea_stage_transitions")
    }


def test_nocase_primary_key_does_not_authorize_cross_id_source_ack(lake):
    lake.insert("idea-Mixed", "Proposal", "seed: 13\n", "identical source fields",
                status="failed", priority="medium")
    before = _snapshot(lake)
    result = _proposal(lake, "idea-mixed")
    assert _snapshot(lake) == before
    assert result["status"] not in {"inserted", "already_present_exact"}, result
    assert lake.conn.execute("SELECT idea_id FROM ideas").fetchone()[0] == "idea-Mixed"


def test_nocase_primary_key_still_allows_byte_exact_same_id_replay(lake):
    assert _proposal(lake, "idea-Mixed")["status"] == "inserted"
    before = _snapshot(lake)
    assert _proposal(lake, "idea-Mixed")["status"] == "already_present_exact"
    assert _snapshot(lake) == before
