import sqlite3

import pytest

from orze.core.sqlite_policy import (
    SQLitePolicyError,
    apply_shared_database_policy,
    inspect_shared_database_policy,
)
from orze.engine.trigger_ledger import _open_short_lived
from orze.engine import trigger_ledger
from orze.idea_lake import IdeaLake


def test_fresh_database_uses_verified_rollback_policy(tmp_path):
    connection = sqlite3.connect(tmp_path / "fresh.db")
    policy = apply_shared_database_policy(connection)
    inspected = inspect_shared_database_policy(connection)
    assert policy == {
        "policy_version": 1,
        "journal_mode": "delete",
        "synchronous": "full",
        "locking_mode": "normal",
    }
    assert inspected["compliant"] is True
    connection.close()


def test_quiescent_wal_database_is_converted_on_disposable_copy(tmp_path):
    db_path = tmp_path / "wal.db"
    connection = sqlite3.connect(db_path)
    assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    connection.execute("CREATE TABLE evidence (value TEXT)")
    connection.execute("INSERT INTO evidence VALUES ('preserved')")
    connection.commit()
    connection.close()

    connection = sqlite3.connect(db_path)
    assert inspect_shared_database_policy(connection)["journal_mode"] == "wal"
    applied = apply_shared_database_policy(connection)
    assert applied["journal_mode"] == "delete"
    assert connection.execute("SELECT value FROM evidence").fetchone()[0] == "preserved"
    assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    connection.close()


def test_active_wal_writer_blocks_transition_without_modifying_data(tmp_path):
    db_path = tmp_path / "busy.db"
    writer = sqlite3.connect(db_path)
    assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    writer.execute("CREATE TABLE evidence (value TEXT)")
    writer.commit()
    writer.execute("BEGIN IMMEDIATE")
    writer.execute("INSERT INTO evidence VALUES ('uncommitted')")

    contender = sqlite3.connect(db_path, timeout=0.01)
    contender.execute("PRAGMA busy_timeout=10")
    with pytest.raises(
        SQLitePolicyError, match="sqlite_journal_mode_transition_failed"
    ):
        apply_shared_database_policy(contender)
    assert inspect_shared_database_policy(contender)["journal_mode"] == "wal"
    contender.close()
    writer.rollback()
    assert writer.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 0
    writer.close()


def test_policy_refuses_to_run_inside_transaction(tmp_path):
    connection = sqlite3.connect(tmp_path / "transaction.db")
    connection.execute("CREATE TABLE evidence (value TEXT)")
    connection.execute("BEGIN")
    with pytest.raises(
        SQLitePolicyError, match="sqlite_policy_applied_inside_transaction"
    ):
        apply_shared_database_policy(connection)
    connection.rollback()
    connection.close()


def test_unaccepted_delete_request_fails_closed_without_raw_error():
    class Cursor:
        def fetchone(self):
            return ("wal",)

    class Connection:
        in_transaction = False

        def execute(self, _pragma):
            return Cursor()

    with pytest.raises(
        SQLitePolicyError, match="^sqlite_delete_journal_unavailable$"
    ) as caught:
        apply_shared_database_policy(Connection())
    assert "wal" not in str(caught.value)


def test_sqlite_operational_detail_is_not_exposed():
    class Connection:
        in_transaction = False

        def execute(self, _pragma):
            raise sqlite3.OperationalError("token=must-not-leak")

    with pytest.raises(
        SQLitePolicyError, match="^sqlite_journal_mode_transition_failed$"
    ) as caught:
        apply_shared_database_policy(Connection())
    assert caught.value.__cause__ is None
    assert "must-not-leak" not in str(caught.value)


def test_idea_lake_and_trigger_ledger_share_policy(tmp_path):
    db_path = tmp_path / "lake.db"
    lake = IdeaLake(db_path)
    assert inspect_shared_database_policy(lake.conn)["compliant"] is True
    lake.close()

    connection = _open_short_lived(db_path)
    assert inspect_shared_database_policy(connection)["compliant"] is True
    connection.close()


def test_idea_lake_closes_connection_when_policy_rejects(monkeypatch):
    class Connection:
        row_factory = None
        closed = False

        def close(self):
            self.closed = True

    connection = Connection()
    monkeypatch.setattr("orze.idea_lake.sqlite3.connect", lambda *a, **k: connection)
    monkeypatch.setattr(
        "orze.idea_lake.apply_shared_database_policy",
        lambda _: (_ for _ in ()).throw(
            SQLitePolicyError("sqlite_delete_journal_unavailable")
        ),
    )
    with pytest.raises(SQLitePolicyError):
        IdeaLake("rejected.db")
    assert connection.closed is True


def test_trigger_ledger_closes_connection_when_policy_rejects(monkeypatch):
    class Connection:
        closed = False

        def close(self):
            self.closed = True

    connection = Connection()
    monkeypatch.setattr(
        trigger_ledger.sqlite3, "connect", lambda *a, **k: connection
    )
    monkeypatch.setattr(
        trigger_ledger,
        "apply_shared_database_policy",
        lambda _: (_ for _ in ()).throw(
            SQLitePolicyError("sqlite_delete_journal_unavailable")
        ),
    )
    with pytest.raises(SQLitePolicyError):
        _open_short_lived("rejected.db")
    assert connection.closed is True
