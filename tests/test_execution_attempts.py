"""New generic attempt mechanism acceptance; not old-API-absence red tests."""
import dataclasses
import json
import sqlite3

import pytest

from orze.core import execution_attempts as attempts
from orze.core.execution_attempts import (
    AttemptAuthorityError, AttemptRef, StaleAttempt, create_attempt,
    current_attempt, ensure_schema, finish_attempt, hold_attempt,
    mark_running, require_current,
)


@pytest.fixture
def conn(tmp_path):
    connection = sqlite3.connect(tmp_path / "attempts.db")
    connection.execute("BEGIN IMMEDIATE")
    ensure_schema(connection)
    connection.commit()
    yield connection
    connection.close()


def _create(conn, *, task="task-1", phase="arbitrary-tool", identity="attempt-A"):
    return create_attempt(conn, task, phase, identity, {"adapter": "test", "verified": True})


def test_intent_running_and_terminal_are_caller_transaction_facts(conn):
    conn.execute("BEGIN IMMEDIATE")
    ref = _create(conn)
    assert ref == AttemptRef("task-1", "arbitrary-tool", "attempt-A", 1)
    assert current_attempt(conn, ref.task_id, ref.phase)["state"] == "LAUNCHING"
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.generation = 2
    mark_running(conn, ref, {"created": True, "process_diagnostic": 0})
    assert finish_attempt(conn, ref, {"adapter_outcome": "done", "value": -1}) == "committed"
    assert conn.in_transaction
    conn.commit()
    row = current_attempt(conn, ref.task_id, ref.phase)
    assert row["state"] == "TERMINAL"
    assert row["binding"] == {"created": True, "process_diagnostic": 0}
    assert row["terminal"] == {"adapter_outcome": "done", "value": -1}
    row["binding"]["created"] = False
    assert current_attempt(conn, ref.task_id, ref.phase)["binding"]["created"] is True


@pytest.mark.parametrize("operation", ["schema", "create", "running", "finish", "hold"])
def test_writes_never_own_or_start_a_transaction(conn, operation):
    ref = AttemptRef("task-1", "tool", "attempt-A", 1)
    calls = {
        "schema": lambda: ensure_schema(conn),
        "create": lambda: _create(conn),
        "running": lambda: mark_running(conn, ref),
        "finish": lambda: finish_attempt(conn, ref, {}),
        "hold": lambda: hold_attempt(conn, ref, "uncertain"),
    }
    with pytest.raises(AttemptAuthorityError, match="caller_transaction"):
        calls[operation]()
    assert not conn.in_transaction
    assert conn.execute("SELECT COUNT(*) FROM execution_attempts").fetchone()[0] == 0


def test_missing_table_read_is_readonly_and_does_not_create():
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("PRAGMA query_only=ON")
        assert current_attempt(connection, "task-1", "cpu") is None
        assert connection.execute("SELECT name FROM sqlite_master").fetchall() == []
        assert not connection.in_transaction
    finally:
        connection.close()


def test_schema_creation_and_caller_data_rollback_together():
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE TABLE marker(value TEXT)")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("INSERT INTO marker VALUES ('before')")
        ensure_schema(connection)
        _create(connection)
        assert connection.in_transaction
        connection.rollback()
        assert current_attempt(connection, "task-1", "arbitrary-tool") is None
        assert connection.execute("SELECT * FROM marker").fetchall() == []
    finally:
        connection.close()


def test_two_connections_cannot_replace_uncommitted_or_running_owner(conn):
    path = conn.execute("PRAGMA database_list").fetchone()[2]
    peer = sqlite3.connect(path, timeout=0.001)
    try:
        conn.execute("BEGIN IMMEDIATE")
        ref = _create(conn)
        assert current_attempt(peer, ref.task_id, ref.phase) is None
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            peer.execute("BEGIN IMMEDIATE")
        assert conn.in_transaction and not peer.in_transaction
        conn.commit()
        peer.execute("BEGIN IMMEDIATE")
        with pytest.raises(AttemptAuthorityError, match="previous_not_closed"):
            _create(peer, identity="attempt-B")
        assert peer.in_transaction
        peer.rollback()
        assert current_attempt(peer, ref.task_id, ref.phase)["attempt_id"] == ref.attempt_id
    finally:
        peer.close()


def test_lifecycle_and_attempt_finish_can_be_rolled_back_atomically(conn):
    conn.execute("CREATE TABLE lifecycle(state TEXT)")
    conn.execute("INSERT INTO lifecycle VALUES ('IN_PROGRESS')")
    ref = _create(conn)
    mark_running(conn, ref)
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("UPDATE lifecycle SET state='COMPLETE'")
    assert finish_attempt(conn, ref, {"observed": "exit"}) == "committed"
    conn.rollback()
    assert conn.execute("SELECT state FROM lifecycle").fetchone()[0] == "IN_PROGRESS"
    assert current_attempt(conn, ref.task_id, ref.phase)["state"] == "RUNNING"
    conn.execute("BEGIN IMMEDIATE")
    assert finish_attempt(conn, ref, {"observed": "exit"}) == "committed"
    conn.execute("UPDATE lifecycle SET state='COMPLETE'")
    conn.commit()


def test_terminal_duplicate_is_exact_canonical_and_has_no_write(conn):
    conn.execute("BEGIN IMMEDIATE")
    ref = _create(conn)
    mark_running(conn, ref)
    assert finish_attempt(conn, ref, {"a": 1, "b": [True, None]}) == "committed"
    before = conn.total_changes
    assert finish_attempt(conn, ref, {"b": [True, None], "a": 1}) == "duplicate"
    assert conn.total_changes == before
    with pytest.raises(AttemptAuthorityError, match="terminal_conflict"):
        finish_attempt(conn, ref, {"a": 2})
    assert conn.in_transaction


def test_new_generation_makes_old_reference_stale_without_touching_current(conn):
    conn.execute("BEGIN IMMEDIATE")
    old = _create(conn)
    mark_running(conn, old)
    finish_attempt(conn, old, {"result": "failure"})
    new = _create(conn, identity="attempt-B")
    assert new.generation == 2
    before = conn.total_changes
    assert finish_attempt(conn, old, {"result": "failure"}) == "stale"
    with pytest.raises(StaleAttempt):
        require_current(conn, old)
    with pytest.raises(StaleAttempt):
        mark_running(conn, old)
    assert conn.total_changes == before
    assert current_attempt(conn, new.task_id, new.phase)["state"] == "LAUNCHING"


def test_attempt_id_is_globally_unique_but_binary_case_distinct(conn):
    conn.execute("BEGIN IMMEDIATE")
    first = _create(conn, identity="Attempt-A")
    with pytest.raises(AttemptAuthorityError, match="id_already_used"):
        _create(conn, task="task-2", identity="Attempt-A")
    second = _create(conn, task="task-2", identity="attempt-a")
    assert first.attempt_id != second.attempt_id
    with pytest.raises(AttemptAuthorityError, match="id_already_used"):
        _create(conn, identity="Attempt-A")


def test_running_binding_can_change_only_once(conn):
    conn.execute("BEGIN IMMEDIATE")
    ref = _create(conn)
    mark_running(conn, ref, {"actual": {"created": True}})
    with pytest.raises(AttemptAuthorityError):
        mark_running(conn, ref, {"actual": "changed"})
    assert require_current(conn, ref)["binding"] == {"actual": {"created": True}}


def test_not_started_never_infers_execution_and_allows_next_intent(conn):
    conn.execute("BEGIN IMMEDIATE")
    ref = _create(conn)
    with pytest.raises(AttemptAuthorityError, match="finish_not_authorized"):
        finish_attempt(conn, ref, {})
    assert finish_attempt(conn, ref, {"proof": "exec_not_called"}, not_started=True) == "committed"
    new = _create(conn, identity="attempt-B")
    mark_running(conn, new)
    with pytest.raises(AttemptAuthorityError, match="finish_not_authorized"):
        finish_attempt(conn, new, {}, not_started=True)


def test_in_doubt_never_becomes_an_automatic_retry_or_terminal(conn):
    conn.execute("BEGIN IMMEDIATE")
    ref = _create(conn)
    hold_attempt(conn, ref, "exec may have happened")
    before = conn.total_changes
    hold_attempt(conn, ref, "exec may have happened")
    assert conn.total_changes == before
    for action in [lambda: mark_running(conn, ref),
                   lambda: finish_attempt(conn, ref, {}),
                   lambda: finish_attempt(conn, ref, {}, not_started=True),
                   lambda: _create(conn, identity="attempt-B")]:
        with pytest.raises(AttemptAuthorityError):
            action()
    assert current_attempt(conn, ref.task_id, ref.phase)["state"] == "IN_DOUBT"


@pytest.mark.parametrize("operation", ["insert", "running", "finish", "hold"])
def test_ignored_dml_never_reports_success_and_leaves_rollback_to_caller(conn, operation):
    conn.execute("BEGIN IMMEDIATE")
    if operation == "insert":
        conn.execute("CREATE TRIGGER ignore_effect BEFORE INSERT ON execution_attempts "
                     "BEGIN SELECT RAISE(IGNORE); END")
        action = lambda: _create(conn)
    else:
        ref = _create(conn)
        if operation == "finish":
            mark_running(conn, ref)
        conn.execute("CREATE TRIGGER ignore_effect BEFORE UPDATE ON execution_attempts "
                     "BEGIN SELECT RAISE(IGNORE); END")
        action = {"running": lambda: mark_running(conn, ref),
                  "finish": lambda: finish_attempt(conn, ref, {}),
                  "hold": lambda: hold_attempt(conn, ref, "unknown")}[operation]
    with pytest.raises(AttemptAuthorityError):
        action()
    assert conn.in_transaction
    conn.rollback()
    assert current_attempt(conn, "task-1", "arbitrary-tool") is None


def test_trigger_changed_readback_is_not_accepted(conn):
    conn.execute("BEGIN IMMEDIATE")
    ref = _create(conn)
    conn.execute("CREATE TRIGGER corrupt_binding AFTER UPDATE ON execution_attempts "
                 "BEGIN UPDATE execution_attempts SET binding_json='{}' "
                 "WHERE attempt_id=NEW.attempt_id; END")
    with pytest.raises(AttemptAuthorityError, match="update_not_confirmed"):
        mark_running(conn, ref, {"created": True})
    assert conn.in_transaction
    conn.rollback()


@pytest.mark.parametrize("change", ["nocase_pk", "nocase_stream", "missing_unique", "extra_unique", "case_name"])
def test_existing_ambiguous_or_nocase_schema_is_rejected_without_migration(change):
    conn = sqlite3.connect(":memory:")
    schema = attempts._SQL
    if change == "nocase_pk":
        schema = schema.replace("COLLATE BINARY PRIMARY KEY", "COLLATE NOCASE PRIMARY KEY")
    elif change == "nocase_stream":
        schema = schema.replace("task_id TEXT NOT NULL COLLATE BINARY", "task_id TEXT NOT NULL COLLATE NOCASE")
    elif change == "missing_unique":
        schema = schema.replace(",\n    UNIQUE (task_id, phase, generation)", "")
    elif change == "case_name":
        schema = schema.replace("execution_attempts", "Execution_Attempts")
    conn.execute(schema)
    if change == "extra_unique":
        conn.execute("CREATE UNIQUE INDEX unexpected_identity ON execution_attempts(task_id COLLATE NOCASE)")
    before = conn.execute("SELECT type,name,sql FROM sqlite_master ORDER BY name").fetchall()
    try:
        with pytest.raises(AttemptAuthorityError):
            current_attempt(conn, "task-1", "tool")
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(AttemptAuthorityError):
            ensure_schema(conn)
        assert conn.in_transaction
        assert conn.execute("SELECT type,name,sql FROM sqlite_master ORDER BY name").fetchall() == before
    finally:
        conn.close()


@pytest.mark.parametrize("raw", ['{"same":1,"same":2}', '{bad', ' {"ok":1}', '"' + 'x' * 65536 + '"'])
def test_malformed_noncanonical_or_oversized_stored_json_fails_closed(conn, raw):
    conn.execute("BEGIN IMMEDIATE")
    ref = _create(conn)
    conn.execute("UPDATE execution_attempts SET binding_json=?", (raw,))
    with pytest.raises(AttemptAuthorityError):
        current_attempt(conn, ref.task_id, ref.phase)


@pytest.mark.parametrize("kind", ["key", "nan", "recursive", "depth", "nodes", "bytes", "custom"])
def test_json_limits_reject_before_insert_without_rolling_back_caller(conn, kind):
    values = {"key": {1: "not a string"}, "nan": {"x": float("nan")},
              "nodes": {"x": [0] * 2048}, "bytes": {"x": "界" * 22000},
              "custom": {"x": object()}, "recursive": {}, "depth": {}}
    values["recursive"]["self"] = values["recursive"]
    cursor = values["depth"]
    for _ in range(34):
        cursor["child"] = {}
        cursor = cursor["child"]
    conn.execute("BEGIN IMMEDIATE")
    with pytest.raises(AttemptAuthorityError):
        create_attempt(conn, "task-1", "custom", "attempt-A", values[kind])
    assert conn.in_transaction
    assert conn.execute("SELECT COUNT(*) FROM execution_attempts").fetchone()[0] == 0


@pytest.mark.parametrize("generation", [True, 0, -1, 1.0, 2**63])
def test_generation_is_a_positive_sqlite_integer_not_bool(generation):
    with pytest.raises(AttemptAuthorityError, match="generation_invalid"):
        AttemptRef("task-1", "custom", "attempt-A", generation)


def test_query_only_current_snapshot_and_main_schema_ignore_temp_shadow(conn):
    conn.execute("BEGIN IMMEDIATE")
    ref = _create(conn)
    conn.commit()
    conn.execute("CREATE TEMP TABLE execution_attempts(fake TEXT)")
    conn.execute("PRAGMA query_only=ON")
    before = conn.total_changes
    assert current_attempt(conn, ref.task_id, ref.phase)["attempt_id"] == ref.attempt_id
    assert conn.total_changes == before
    assert not conn.in_transaction


@pytest.mark.parametrize("stored,replayed", [({"value": True}, {"value": 1}),
                                           ({"value": 1}, {"value": 1.0})])
def test_terminal_duplicate_requires_exact_json_types(conn, stored, replayed):
    conn.execute("BEGIN IMMEDIATE")
    ref = _create(conn)
    mark_running(conn, ref)
    assert finish_attempt(conn, ref, stored) == "committed"
    with pytest.raises(AttemptAuthorityError, match="terminal_conflict"):
        finish_attempt(conn, ref, replayed)
    assert current_attempt(conn, ref.task_id, ref.phase)["terminal"] == stored


@pytest.mark.parametrize("operation", ["insert", "running"])
def test_readback_rejects_trigger_converting_json_bool_to_number(conn, operation):
    conn.execute("BEGIN IMMEDIATE")
    if operation == "running":
        ref = _create(conn)
    event = "INSERT" if operation == "insert" else "UPDATE"
    conn.execute(f"CREATE TRIGGER mutate_json AFTER {event} ON execution_attempts "
                 "BEGIN UPDATE execution_attempts SET binding_json="
                 "'{\"adapter\":\"test\",\"verified\":1}' "
                 "WHERE attempt_id=NEW.attempt_id; END")
    with pytest.raises(AttemptAuthorityError, match="not_confirmed"):
        if operation == "insert":
            _create(conn)
        else:
            mark_running(conn, ref)
    assert conn.in_transaction
    conn.rollback()
