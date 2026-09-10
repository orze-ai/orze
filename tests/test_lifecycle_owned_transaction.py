"""Lifecycle writes compose in a caller transaction and require exact receipts.

Public-path trigger cases can replay against the old implementation. New
transaction-core cases are explicitly skipped when the new API is absent;
absence is not counted as a behavioral failure.
"""
import pytest

from orze.idea_lake import IdeaLake


@pytest.fixture
def lake(tmp_path):
    instance = IdeaLake(tmp_path / "ideas.db")
    instance.insert("idea-transaction", "Fixture", "{}", "", status="queued")
    try:
        yield instance
    finally:
        instance.close()


def _snapshot(lake):
    return {table: [tuple(row) for row in lake.conn.execute(
        f"SELECT * FROM {table} ORDER BY rowid")]
        for table in ("ideas", "idea_state", "idea_transitions",
                      "idea_stage_state", "idea_stage_transitions")}


@pytest.mark.parametrize("fault", [
    "state_insert_ignored", "state_update_ignored", "global_audit_ignored",
    "legacy_update_ignored", "state_after_update", "global_audit_after_insert",
    "stage_insert_ignored", "stage_after_update", "stage_audit_ignored",
    "stage_audit_after_insert", "legacy_after_pipeline",
])
def test_public_transition_rejects_unacknowledged_or_changed_write(lake, fault):
    idea = "idea-transaction"
    stage_case = fault.startswith("stage_")
    if stage_case or fault == "legacy_after_pipeline":
        assert lake.record_state_transition(idea, "QUEUED", "CLAIMED")
    if fault == "state_insert_ignored":
        lake.conn.execute("DELETE FROM idea_state WHERE idea_id=?", (idea,))
    if fault == "stage_after_update":
        assert lake.record_stage_transition(idea, "training", "NOT_STARTED", "PENDING", "setup")
    declarations = {
        "state_insert_ignored": "BEFORE INSERT ON idea_state BEGIN SELECT RAISE(IGNORE); END",
        "state_update_ignored": "BEFORE UPDATE ON idea_state BEGIN SELECT RAISE(IGNORE); END",
        "global_audit_ignored": "BEFORE INSERT ON idea_transitions BEGIN SELECT RAISE(IGNORE); END",
        "legacy_update_ignored": "BEFORE UPDATE OF status ON ideas BEGIN SELECT RAISE(IGNORE); END",
        "state_after_update": "AFTER UPDATE ON idea_state BEGIN UPDATE idea_state SET claimed_at='changed-clock' WHERE idea_id=NEW.idea_id; END",
        "global_audit_after_insert": "AFTER INSERT ON idea_transitions BEGIN UPDATE idea_transitions SET reason='changed-reason' WHERE id=NEW.id; END",
        "stage_insert_ignored": "BEFORE INSERT ON idea_stage_state BEGIN SELECT RAISE(IGNORE); END",
        "stage_after_update": "AFTER UPDATE ON idea_stage_state BEGIN UPDATE idea_stage_state SET current_state='FAILED' WHERE idea_id=NEW.idea_id AND stage=NEW.stage; END",
        "stage_audit_ignored": "BEFORE INSERT ON idea_stage_transitions BEGIN SELECT RAISE(IGNORE); END",
        "stage_audit_after_insert": "AFTER INSERT ON idea_stage_transitions BEGIN UPDATE idea_stage_transitions SET reason='changed-reason' WHERE id=NEW.id; END",
        "legacy_after_pipeline": "AFTER UPDATE OF status ON ideas BEGIN UPDATE idea_stage_state SET current_state='FAILED' WHERE idea_id=NEW.idea_id AND stage='training'; END",
    }
    lake.conn.execute("CREATE TRIGGER fault " + declarations[fault])
    lake.conn.commit()
    before = _snapshot(lake)
    if stage_case:
        initial = "PENDING" if fault == "stage_after_update" else "NOT_STARTED"
        target = "IN_PROGRESS" if initial == "PENDING" else "PENDING"
        accepted = lake.record_stage_transition(idea, "training", initial, target, "expected")
    elif fault == "legacy_after_pipeline":
        accepted = lake.record_state_transition(idea, "CLAIMED", "IN_PROGRESS", "expected")
    else:
        accepted = lake.record_state_transition(idea, "QUEUED", "CLAIMED", "expected")
    assert accepted is False, "an ignored/changed write is not a committed lifecycle edge"
    assert not lake.conn.in_transaction
    assert _snapshot(lake) == before


def _nocase_table(lake, table):
    sql = lake.conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()[0]
    lake.conn.execute(f"ALTER TABLE {table} RENAME TO previous_{table}")
    for column in ("idea_id", "stage", "current_state"):
        sql = sql.replace(f"{column} TEXT", f"{column} TEXT COLLATE NOCASE")
    lake.conn.execute(sql)
    lake.conn.execute(f"INSERT INTO {table} SELECT * FROM previous_{table}")
    lake.conn.execute(f"DROP TABLE previous_{table}")
    lake.conn.commit()


@pytest.mark.parametrize("entry", ["global_identity", "stage_identity", "stage_name", "state_value"])
def test_existing_transition_requires_binary_identity_and_state(lake, entry):
    idea = "idea-transaction"
    if entry != "global_identity":
        assert lake.record_state_transition(idea, "QUEUED", "CLAIMED")
    if entry in ("stage_identity", "stage_name"):
        assert lake.record_stage_transition(idea, "training", "NOT_STARTED", "PENDING", "setup")
        _nocase_table(lake, "idea_stage_state")
        if entry == "stage_name":
            lake.conn.execute("UPDATE idea_stage_state SET stage='TRAINING'")
    else:
        _nocase_table(lake, "idea_state")
        if entry == "state_value":
            lake.conn.execute("UPDATE idea_state SET current_state='claimed'")
    lake.conn.commit()
    before = _snapshot(lake)
    if entry == "global_identity":
        accepted = lake.record_state_transition(idea.upper(), "QUEUED", "CLAIMED", "must not alias")
    elif entry in ("stage_identity", "stage_name"):
        lake.conn.execute("BEGIN IMMEDIATE")
        try:
            accepted = lake._record_stage_transition_in_tx(
                idea.upper() if entry == "stage_identity" else idea,
                "training", "PENDING", "IN_PROGRESS", "must not alias", "host", 17, "at")
        finally:
            lake.conn.rollback()
    else:
        accepted = lake.record_state_transition(idea, "CLAIMED", "IN_PROGRESS", "must not alias")
    assert accepted is False
    assert _snapshot(lake) == before


def _core(lake):
    method = getattr(lake, "_record_state_transition_in_tx", None)
    if method is None:
        pytest.skip("new transaction-core API absent; not an old behavior red")
    return method


def test_new_core_requires_caller_transaction_without_writing(lake):
    method = _core(lake)
    before = _snapshot(lake)
    assert method("idea-transaction", "QUEUED", "CLAIMED") is False
    assert not lake.conn.in_transaction
    assert _snapshot(lake) == before


@pytest.mark.parametrize("finish", ["commit", "rollback"])
def test_new_core_never_manages_its_callers_transaction(lake, finish):
    method = _core(lake)
    before = _snapshot(lake)
    lake.conn.execute("BEGIN IMMEDIATE")
    statements = []
    lake.conn.set_trace_callback(statements.append)
    try:
        assert method("idea-transaction", "QUEUED", "CLAIMED", "owned", "host", 17, "training", "explicit-at")
        assert method("idea-transaction", "CLAIMED", "IN_PROGRESS", "owned", "host", 17, "training", "explicit-at")
        assert lake.conn.in_transaction
        assert not any(statement.strip().upper().startswith(("BEGIN", "COMMIT", "ROLLBACK"))
                       for statement in statements)
    finally:
        lake.conn.set_trace_callback(None)
    getattr(lake.conn, finish)()
    if finish == "rollback":
        assert _snapshot(lake) == before
    else:
        assert lake.get_fsm_state("idea-transaction") == "IN_PROGRESS"
        assert lake.get_stage_state("idea-transaction", "training") == "IN_PROGRESS"
        assert lake.get_fsm_history("idea-transaction")[-1]["ts"] == "explicit-at"


def test_new_core_rejection_leaves_prior_caller_work_for_caller_rollback(lake):
    method = _core(lake)
    before = _snapshot(lake)
    lake.conn.execute("BEGIN IMMEDIATE")
    lake.conn.execute("UPDATE ideas SET title='caller-owned'")
    assert method("idea-transaction", "IN_PROGRESS", "COMPLETE") is False
    assert lake.conn.in_transaction
    assert lake.get("idea-transaction")["title"] == "caller-owned"
    lake.conn.rollback()
    assert _snapshot(lake) == before


def test_new_stage_core_requires_caller_transaction(lake):
    _core(lake)
    assert lake.record_state_transition("idea-transaction", "QUEUED", "CLAIMED")
    before = _snapshot(lake)
    assert lake._record_stage_transition_in_tx(
        "idea-transaction", "training", "NOT_STARTED", "PENDING", "owned", "host", 17, "at") is False
    assert not lake.conn.in_transaction
    assert _snapshot(lake) == before


def test_new_core_failed_write_never_rolls_back_prior_caller_work(lake):
    method = _core(lake)
    lake.conn.execute("CREATE TRIGGER ignore_legacy BEFORE UPDATE OF status ON ideas BEGIN SELECT RAISE(IGNORE); END")
    lake.conn.commit()
    before = _snapshot(lake)
    lake.conn.execute("BEGIN IMMEDIATE")
    lake.conn.execute("UPDATE ideas SET title='caller-owned'")
    assert method("idea-transaction", "QUEUED", "CLAIMED") is False
    assert lake.conn.in_transaction
    assert lake.get("idea-transaction")["title"] == "caller-owned"
    lake.conn.rollback()
    assert _snapshot(lake) == before


def test_public_transaction_and_stage_helpers_keep_valid_controls(lake):
    assert lake.record_state_transition("idea-transaction", "QUEUED", "CLAIMED")
    assert not lake.conn.in_transaction
    assert lake.record_state_transition("idea-transaction", "CLAIMED", "IN_PROGRESS")
    assert lake.record_stage_transition("idea-transaction", "training", "IN_PROGRESS", "COMPLETE", "done")
    assert lake.record_stage_transition("idea-transaction", "evaluation", "PENDING", "IN_PROGRESS", "start")
    assert lake.record_state_transition("idea-transaction", "IN_PROGRESS", "COMPLETE")
    assert not lake.conn.in_transaction
    assert lake.get("idea-transaction")["status"] == "completed"
