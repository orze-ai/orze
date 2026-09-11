"""New caller-owned normal admission requirements, not old API-absence reds."""
import hashlib
import sqlite3

import pytest
import yaml

from orze.core.integrity import hash_config
from orze.core.proposal_admission import (
    ProposalAdmissionError,
    admit_proposal_in_tx,
)
from orze.idea_lake import IdeaLake


@pytest.fixture
def lake(tmp_path):
    instance = IdeaLake(str(tmp_path / "lake.db"))
    try:
        yield instance
    finally:
        if instance.conn.in_transaction:
            instance.conn.rollback()
        instance.close()


def _prepared(lake, idea_id="idea-a", **overrides):
    values = {
        "title": "Ordinary proposal", "config_yaml": "seed: 13\n",
        "raw_markdown": "exact proposal source", "hypothesis": "bounded test",
    }
    values.update(overrides)
    return lake.prepare_proposal(idea_id, **values)


def _snapshot(lake):
    return {
        table: [tuple(row) for row in lake.conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
        for table in ("ideas", "idea_state", "idea_transitions", "idea_stage_state", "idea_stage_transitions")
    }


def test_preparation_is_pure_and_matches_legacy_normalization(lake):
    statements = []
    lake.conn.set_trace_callback(statements.append)
    prepared = _prepared(lake, config_yaml="nested:\n  seed: 13\n", eval_metrics={"score": 0})
    lake.conn.set_trace_callback(None)
    assert statements == []
    assert not lake.conn.in_transaction
    assert prepared["config_hash"] == hash_config(yaml.safe_load(prepared["config"]))
    assert prepared["config_source_sha256"] == hashlib.sha256(prepared["config"].encode()).hexdigest()
    lake.insert("idea-a", "Ordinary proposal", "nested:\n  seed: 13\n", "exact proposal source",
                eval_metrics={"score": 0}, hypothesis="bounded test", status="queued")
    row = dict(lake.conn.execute("SELECT * FROM ideas WHERE idea_id='idea-a'").fetchone())
    assert {key: row[key] for key in prepared if key != "created_at"} == {
        key: value for key, value in prepared.items() if key != "created_at"
    }


def test_new_entry_requires_existing_caller_transaction(lake):
    before = _snapshot(lake)
    with pytest.raises(ProposalAdmissionError, match="proposal_requires_caller_transaction"):
        admit_proposal_in_tx(lake, _prepared(lake))
    assert not lake.conn.in_transaction
    assert _snapshot(lake) == before


def test_insert_is_provisional_and_outer_rollback_owns_all_changes(lake):
    peer = sqlite3.connect(lake.db_path)
    try:
        before = _snapshot(lake)
        original_id = lake.conn.execute("SELECT next_id FROM id_sequence").fetchone()[0]
        lake.conn.execute("PRAGMA busy_timeout=2718")
        lake.conn.execute("BEGIN IMMEDIATE")
        lake.conn.execute("UPDATE id_sequence SET next_id=12345")
        statements = []
        lake.conn.set_trace_callback(statements.append)
        result = admit_proposal_in_tx(lake, _prepared(lake))
        lake.conn.set_trace_callback(None)
        assert result["status"] == "inserted"
        assert lake.conn.in_transaction
        assert not any(statement.lstrip().split()[0].upper() in {
            "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE", "PRAGMA",
        } for statement in statements)
        assert lake.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 2718
        assert peer.execute("SELECT COUNT(*) FROM ideas").fetchone()[0] == 0
        assert peer.execute("SELECT next_id FROM id_sequence").fetchone()[0] == original_id
        lake.conn.rollback()
        assert _snapshot(lake) == before
        assert lake.conn.execute("SELECT next_id FROM id_sequence").fetchone()[0] == original_id
    finally:
        peer.close()


def test_only_caller_commit_publishes_exact_admission(lake):
    peer = sqlite3.connect(lake.db_path)
    try:
        lake.conn.execute("BEGIN IMMEDIATE")
        result = admit_proposal_in_tx(lake, _prepared(lake))
        lake.conn.commit()
        assert peer.execute("SELECT idea_id, status FROM ideas").fetchall() == [("idea-a", "queued")]
        assert peer.execute("SELECT current_state FROM idea_state").fetchall() == [("QUEUED",)]
        assert peer.execute("SELECT id, reason FROM idea_transitions").fetchall() == [
            (result["transition_id"], "proposal_admitted"),
        ]
    finally:
        peer.close()


@pytest.mark.parametrize("overrides,status", [
    ({}, "already_present_exact"),
    ({"title": "Changed title"}, "conflict"),
    ({"idea_id": "idea-b"}, "config_duplicate"),
])
def test_read_only_outcomes_preserve_caller_writer(lake, overrides, status):
    lake.conn.execute("BEGIN IMMEDIATE")
    assert admit_proposal_in_tx(lake, _prepared(lake))["status"] == "inserted"
    lake.conn.commit()
    assert lake.record_state_transition("idea-a", "QUEUED", "CLAIMED", "claim")
    assert lake.record_state_transition("idea-a", "CLAIMED", "IN_PROGRESS", "start")
    assert lake.record_state_transition("idea-a", "IN_PROGRESS", "COMPLETE", "finish")
    before = _snapshot(lake)
    lake.conn.execute("BEGIN IMMEDIATE")
    lake.conn.execute("UPDATE id_sequence SET next_id=12345")
    result = admit_proposal_in_tx(lake, _prepared(lake, **overrides))
    assert result["status"] == status
    assert lake.conn.in_transaction
    assert lake.conn.execute("SELECT next_id FROM id_sequence").fetchone()[0] == 12345
    assert _snapshot(lake) == before
    lake.conn.rollback()


@pytest.mark.parametrize("table,rewrite", [
    ("ideas", False), ("idea_state", False), ("idea_transitions", False),
    ("idea_state", True),
])
def test_second_admission_write_failure_raises_and_caller_rolls_back_whole_batch(lake, table, rewrite):
    if rewrite:
        statement = "UPDATE idea_state SET current_state='FAILED' WHERE idea_id=NEW.idea_id;"
        timing = "AFTER"
    else:
        statement = "SELECT RAISE(IGNORE);"
        timing = "BEFORE"
    lake.conn.executescript(
        f"CREATE TRIGGER admission_fault {timing} INSERT ON {table} "
        f"WHEN NEW.idea_id='idea-b' BEGIN {statement} END;"
    )
    before = _snapshot(lake)
    lake.conn.execute("BEGIN IMMEDIATE")
    lake.conn.execute("UPDATE id_sequence SET next_id=12345")
    assert admit_proposal_in_tx(lake, _prepared(lake))["status"] == "inserted"
    with pytest.raises(ProposalAdmissionError, match="proposal_"):
        admit_proposal_in_tx(lake, _prepared(lake, "idea-b", config_yaml="seed: 14\n"))
    assert lake.conn.in_transaction
    assert lake.conn.execute("SELECT COUNT(*) FROM ideas WHERE idea_id='idea-a'").fetchone()[0] == 1
    lake.conn.rollback()
    assert _snapshot(lake) == before
    assert lake.conn.execute("SELECT next_id FROM id_sequence").fetchone()[0] != 12345


def test_orphan_rejection_does_not_rollback_unrelated_caller_state(lake):
    lake.insert("idea-a", "Old", "seed: 2\n", "old", status="queued")
    lake.conn.execute("DELETE FROM ideas WHERE idea_id='idea-a'")
    lake.conn.commit()
    before = _snapshot(lake)
    lake.conn.execute("BEGIN IMMEDIATE")
    lake.conn.execute("UPDATE id_sequence SET next_id=12345")
    with pytest.raises(ProposalAdmissionError, match="proposal_orphan_lifecycle"):
        admit_proposal_in_tx(lake, _prepared(lake))
    assert lake.conn.in_transaction
    assert lake.conn.execute("SELECT next_id FROM id_sequence").fetchone()[0] == 12345
    assert _snapshot(lake) == before


@pytest.mark.parametrize("field", ["config_hash", "config_source_sha256"])
def test_prepared_cache_cannot_authorize_dedup_bypass(lake, field):
    lake.insert("idea-existing", "Existing", "seed: 13\n", "old", status="queued")
    prepared = _prepared(lake)
    prepared[field] = "0" * 64
    before = _snapshot(lake)
    lake.conn.execute("BEGIN IMMEDIATE")
    with pytest.raises(ProposalAdmissionError, match="proposal_config_identity_mismatch"):
        admit_proposal_in_tx(lake, prepared)
    assert lake.conn.in_transaction
    assert _snapshot(lake) == before


def test_ordinary_cpu_proposals_keep_normal_resource_aware_dedup(lake):
    lake.conn.execute("BEGIN IMMEDIATE")
    assert admit_proposal_in_tx(lake, _prepared(lake, "idea-training"))["status"] == "inserted"
    assert admit_proposal_in_tx(lake, _prepared(lake, "idea-cpu", kind="native_cpu_action"))["status"] == "inserted"
    result = admit_proposal_in_tx(lake, _prepared(lake, "idea-cpu-copy", kind="native_cpu_action"))
    assert result == {"status": "config_duplicate", "reason": "proposal_config_duplicate",
                      "idea_id": "idea-cpu-copy", "existing_id": "idea-cpu"}
    assert lake.conn.execute("SELECT COUNT(*) FROM ideas").fetchone()[0] == 2
    assert lake.conn.execute("SELECT sop_type FROM idea_state WHERE idea_id='idea-cpu'").fetchone()[0] == "action"


@pytest.mark.parametrize("table", ["ideas", "IdEaS"])
def test_temporary_catalog_cannot_redirect_normal_admission(lake, table):
    lake.conn.execute(f"CREATE TEMP TABLE {table} AS SELECT * FROM main.ideas")
    lake.conn.execute("BEGIN IMMEDIATE")
    with pytest.raises(ProposalAdmissionError, match="proposal_temporary_catalog"):
        admit_proposal_in_tx(lake, _prepared(lake))
    assert lake.conn.in_transaction
    assert lake.conn.execute("SELECT COUNT(*) FROM main.ideas").fetchone()[0] == 0
    assert lake.conn.execute("SELECT COUNT(*) FROM temp.ideas").fetchone()[0] == 0


def test_later_caller_fault_cannot_turn_provisional_insert_into_durable_receipt(lake):
    peer = sqlite3.connect(lake.db_path)
    try:
        lake.conn.execute("BEGIN IMMEDIATE")
        provisional = admit_proposal_in_tx(lake, _prepared(lake))
        assert provisional["status"] == "inserted"
        with pytest.raises(sqlite3.IntegrityError):
            lake.conn.execute("INSERT INTO ideas (idea_id) VALUES ('idea-a')")
        lake.conn.rollback()
        assert peer.execute("SELECT COUNT(*) FROM ideas").fetchone()[0] == 0
        assert peer.execute("SELECT COUNT(*) FROM idea_state").fetchone()[0] == 0
        assert peer.execute("SELECT COUNT(*) FROM idea_transitions").fetchone()[0] == 0
    finally:
        peer.close()
