"""New delivery mechanisms, not historical behavior-red evidence.

All launches are ledger calls; these tests never create a subprocess/provider.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from orze.engine import trigger_delivery as delivery
from orze.engine.trigger_delivery_storage import TriggerDeliveryError
from orze.engine.trigger_ledger import claim_trigger, init_schema, _fingerprint


SCOPE = "project-a"
ROLE = "thinker"
HASH = "a" * 64


def query(db, sql, params=()):
    with sqlite3.connect(db) as conn:
        return conn.execute(sql, params).fetchall()


def execute(db, sql, params=()):
    with sqlite3.connect(db) as conn:
        conn.execute(sql, params)


def observe(db, trigger, now=10, scope=SCOPE, role=ROLE):
    return delivery.observe_trigger(db, scope, role, trigger, now=now)


def lease(db, pending, owner="owner-1", now=11, ttl=60, **changes):
    arguments = dict(scope=pending["scope"], role_name=pending["role_name"],
                     expected_sha256=pending["payload_sha256"], owner=owner, now=now, ttl=ttl)
    arguments.update(changes)
    return delivery.lease_trigger(db, pending["delivery_id"], **arguments)


def launch(db, token, attempt="attempt-1", now=12):
    return delivery.begin_launch(db, token, attempt_id=attempt, nonce_sha256=HASH,
                                 command_sha256="b" * 64, now=now)


@pytest.fixture
def received(tmp_path):
    db, trigger = tmp_path / "lake.db", tmp_path / "_trigger_thinker"
    trigger.write_text("captured request", encoding="utf-8")
    response = observe(db, trigger)
    assert response["blocked"] is False
    assert response["pending"]
    return db, trigger, response["pending"]


def test_missing_file_and_database_do_not_create_anything(tmp_path):
    before = list(tmp_path.iterdir())
    assert observe(tmp_path / "absent.db", tmp_path / "absent") == {
        "pending": None, "blocked": False, "reason": "no_pending"}
    assert list(tmp_path.iterdir()) == before


def test_only_valid_file_intake_initializes_missing_database_parent(tmp_path):
    db, trigger = tmp_path / "new-project" / ".orze" / "lake.db", tmp_path / "trigger"
    assert not observe(db, trigger)["blocked"]
    assert not db.parent.exists()
    trigger.write_bytes(b"\xff")
    assert observe(db, trigger)["blocked"]
    assert not db.parent.exists()
    trigger.write_text("first request")
    assert observe(db, trigger)["pending"]["payload"] == "first request"
    assert db.is_file()


@pytest.mark.parametrize("payload", ["", "0", "零和负值都是内容\n-2"])
def test_intake_preserves_file_and_exact_bounded_payload(tmp_path, payload):
    db, trigger = tmp_path / "lake.db", tmp_path / "trigger"
    trigger.write_text(payload, encoding="utf-8")
    pending = observe(db, trigger)["pending"]
    assert pending["payload"] == payload
    assert pending["payload_sha256"] == hashlib.sha256(payload.encode()).hexdigest()
    assert pending["state"] == "PENDING" and pending["generation"] == 0
    assert isinstance(pending["delivery_id"], str)
    assert trigger.read_text() == payload
    assert query(db, "SELECT COUNT(*) FROM trigger_consumptions") == [(1,)]
    assert query(db, "SELECT COUNT(*) FROM trigger_deliveries") == [(1,)]


@pytest.mark.parametrize("kind", ["oversize", "encoding", "directory", "symlink"])
def test_invalid_present_file_blocks_without_creating_database(tmp_path, kind):
    trigger, db = tmp_path / "trigger", tmp_path / "absent.db"
    if kind == "oversize":
        trigger.write_bytes(b"x" * (65536 + 1))
    elif kind == "encoding":
        trigger.write_bytes(b"\xff")
    elif kind == "directory":
        trigger.mkdir()
    else:
        target = tmp_path / "target"
        target.write_text("request")
        trigger.symlink_to(target)
    result = observe(db, trigger)
    assert result["blocked"] is True and result["pending"] is None
    assert not db.exists()


def test_file_replacement_during_read_is_not_received_or_removed(tmp_path, monkeypatch):
    from orze.engine import trigger_ingress
    trigger, replacement, db = tmp_path / "trigger", tmp_path / "next", tmp_path / "lake.db"
    trigger.write_text("A")
    replacement.write_text("B")
    original = trigger_ingress.os.read
    swapped = False

    def read_and_replace(fd, limit):
        nonlocal swapped
        part = original(fd, limit)
        if not swapped:
            swapped = True
            replacement.replace(trigger)
        return part

    monkeypatch.setattr(trigger_ingress.os, "read", read_and_replace)
    result = observe(db, trigger)
    assert result["blocked"] and result["reason"] == "trigger_file_changed"
    assert trigger.read_text() == "B"
    assert not db.exists()


def test_known_or_missing_file_uses_only_readonly_connections(received, monkeypatch):
    db, trigger, pending = received
    before = (db.read_bytes(), db.stat().st_mtime_ns)
    original, connections = delivery.connect, []

    def checked(*args, **kwargs):
        assert not kwargs.get("write") and not kwargs.get("create")
        conn = original(*args, **kwargs)
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
        connections.append(conn)
        return conn

    monkeypatch.setattr(delivery, "connect", checked)
    assert observe(db, trigger)["pending"] == pending
    trigger.unlink()
    assert observe(db, trigger)["pending"] == pending
    assert before == (db.read_bytes(), db.stat().st_mtime_ns)
    assert not db.with_name(db.name + "-journal").exists()
    for conn in connections:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


def test_original_known_file_does_not_refire_after_terminal(received):
    db, trigger, pending = received
    started = launch(db, lease(db, pending))
    assert delivery.record_started(db, started, process_pid=123, now=13)
    assert delivery.record_terminal(db, started, outcome="ok", exit_code=0,
                                    cleanup_verified=True, now=14)
    assert observe(db, trigger, now=100)["pending"] is None
    assert trigger.read_text() == "captured request"
    trigger.write_text("new request, same role")
    replacement = observe(db, trigger, now=101)["pending"]
    assert replacement and replacement["delivery_id"] != pending["delivery_id"]


def test_legacy_unmapped_receipt_is_only_matching_file_tombstone(tmp_path):
    db, trigger = tmp_path / "lake.db", tmp_path / "trigger"
    trigger.write_text("old uncertain request")
    conn = sqlite3.connect(db)
    init_schema(conn)
    conn.execute("INSERT INTO trigger_consumptions(role_name,file_path,fingerprint,consumed_at) "
                 "VALUES (?,?,?,'old')", (ROLE, str(trigger), _fingerprint(trigger)))
    conn.commit()
    conn.close()
    before = db.read_bytes()
    assert observe(db, trigger) == {"pending": None, "blocked": True, "reason": "legacy_consumed"}
    assert db.read_bytes() == before and trigger.exists()
    assert not query(db, "SELECT name FROM sqlite_master WHERE name='trigger_deliveries'")
    trigger.unlink()
    assert not observe(db, trigger)["blocked"]
    trigger.write_text("genuinely new request")
    assert observe(db, trigger)["pending"]["payload"] == "genuinely new request"


def test_native_mapping_arbitrates_unchanged_old_claim_api(received):
    db, trigger, pending = received
    assert claim_trigger(db, ROLE, trigger) is None
    assert not trigger.exists()  # Old API retains its historical unlink behavior.
    assert observe(db, trigger)["pending"]["delivery_id"] == pending["delivery_id"]


def test_two_connections_intake_only_one_native_mapping(tmp_path):
    db, trigger = tmp_path / "lake.db", tmp_path / "trigger"
    trigger.write_text("concurrent request")
    barrier = threading.Barrier(2)

    def worker(_):
        barrier.wait(timeout=5)
        return observe(db, trigger)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker, range(2)))
    assert all(not result["blocked"] for result in results), results
    assert len({result["pending"]["delivery_id"] for result in results}) == 1
    assert query(db, "SELECT COUNT(*) FROM trigger_consumptions") == [(1,)]
    assert query(db, "SELECT COUNT(*) FROM trigger_deliveries") == [(1,)]


def test_two_actual_connections_can_lease_only_one_owner(received):
    db, _, pending = received
    barrier = threading.Barrier(2)

    def worker(index):
        barrier.wait(timeout=5)
        return lease(db, pending, owner="owner-" + str(index))

    with ThreadPoolExecutor(max_workers=2) as pool:
        tokens = list(pool.map(worker, range(2)))
    winners = [token for token in tokens if token]
    assert len(winners) == 1
    assert winners[0]["generation"] == 1
    assert query(db, "SELECT COUNT(*) FROM trigger_delivery_transitions WHERE to_state='LEASED'") == [(1,)]


def test_expired_lease_changes_generation_and_fences_all_old_owner_writes(received):
    db, trigger, pending = received
    first = lease(db, pending, now=11, ttl=2)
    assert observe(db, trigger, now=12)["blocked"]
    assert observe(db, trigger, now=13)["pending"]["delivery_id"] == pending["delivery_id"]
    second = lease(db, pending, owner="owner-2", now=13)
    assert second["generation"] == first["generation"] + 1
    assert not delivery.defer_trigger(db, first, "stale", now=14)
    assert launch(db, first, now=14) is None
    active = launch(db, second, now=14)
    stale = dict(first, attempt_id=active["attempt_id"])
    assert not delivery.record_started(db, stale, process_pid=123, now=15)
    assert not delivery.record_not_started(db, stale, "stale", now=15)
    assert not delivery.record_in_doubt(db, stale, "stale", now=15)
    assert delivery.record_started(db, active, process_pid=123, now=15)
    assert not delivery.record_terminal(db, stale, outcome="ok", exit_code=0,
                                        cleanup_verified=True, now=16)


@pytest.mark.parametrize("changes", [{"scope": "another"}, {"role_name": "other"},
                                     {"expected_sha256": "0" * 64}])
def test_lease_never_substitutes_another_message_or_contract(received, changes):
    db, _, pending = received
    assert lease(db, pending, **changes) is None
    assert query(db, "SELECT state,generation FROM trigger_deliveries") == [("PENDING", 0)]


def test_lease_deadline_is_checked_again_at_launch_boundary(received):
    db, _, pending = received
    token = lease(db, pending, now=10, ttl=1)
    assert launch(db, token, now=11) is None
    assert query(db, "SELECT COUNT(*) FROM trigger_delivery_attempts") == [(0,)]


@pytest.mark.parametrize("state", ["LAUNCHING", "STARTED", "IN_DOUBT"])
def test_uncertain_or_running_delivery_blocks_entire_role_not_other_scope(received, state):
    db, trigger, pending = received
    active = launch(db, lease(db, pending))
    if state == "STARTED":
        assert delivery.record_started(db, active, process_pid=123)
    elif state == "IN_DOUBT":
        assert delivery.record_in_doubt(db, active, "lost_start_confirmation")
    trigger.write_text("second queued request")
    response = observe(db, trigger, now=10_000)
    assert response == {"pending": None, "blocked": True, "reason": state.lower()}
    other = query(db, "SELECT delivery_id FROM trigger_deliveries WHERE state='PENDING'")[0][0]
    assert delivery.lease_trigger(db, other, scope=SCOPE, role_name=ROLE,
                                  expected_sha256=hashlib.sha256(trigger.read_bytes()).hexdigest(),
                                  owner="other", now=10_000) is None
    trigger.unlink()
    assert not observe(db, trigger, scope="other-project")["blocked"]


def test_unknown_state_fails_closed_without_rewriting_it(received):
    db, trigger, _ = received
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA ignore_check_constraints=ON")
        conn.execute("UPDATE trigger_deliveries SET state='FUTURE_STATE'")
    response = observe(db, trigger)
    assert response["blocked"] and response["pending"] is None
    assert query(db, "SELECT state FROM trigger_deliveries") == [("FUTURE_STATE",)]


def test_not_started_retries_preserve_immutable_attempt_history(received):
    db, trigger, pending = received
    first = launch(db, lease(db, pending), attempt="attempt-first")
    initial = query(db, "SELECT * FROM trigger_delivery_attempts")
    assert delivery.record_not_started(db, first, "exec_not_found", now=13)
    pending_again = observe(db, trigger, now=14)["pending"]
    second_lease = lease(db, pending_again, owner="new-owner", now=15)
    second = launch(db, second_lease, attempt="attempt-second", now=16)
    assert second["generation"] > first["generation"]
    assert query(db, "SELECT * FROM trigger_delivery_attempts WHERE attempt_id='attempt-first'") == initial
    assert len(query(db, "SELECT * FROM trigger_delivery_attempts")) == 2
    assert not delivery.record_not_started(db, first, "stale", now=17)
    assert delivery.record_started(db, second, process_pid=321, now=17)
    assert not delivery.record_not_started(db, second, "already_executed", now=18)


def test_started_and_terminal_are_idempotent_only_for_exact_fields(received):
    db, _, pending = received
    active = launch(db, lease(db, pending))
    assert delivery.record_started(db, active, process_pid=123, now=13)
    assert delivery.record_started(db, active, process_pid=123, now=14)
    assert not delivery.record_started(db, active, process_pid=456, now=14)
    assert delivery.record_terminal(db, active, outcome="ok", exit_code=0, cleanup_verified=True, now=15)
    assert delivery.record_terminal(db, active, outcome="ok", exit_code=0, cleanup_verified=True, now=16)
    assert not delivery.record_terminal(db, active, outcome="error", exit_code=0, cleanup_verified=True)
    assert not delivery.record_terminal(db, active, outcome="ok", exit_code=1, cleanup_verified=True)
    assert not delivery.record_terminal(db, active, outcome="ok", exit_code=0, cleanup_verified=False)
    assert query(db, "SELECT COUNT(*) FROM trigger_delivery_transitions WHERE to_state='STARTED'") == [(1,)]
    assert query(db, "SELECT COUNT(*) FROM trigger_delivery_transitions WHERE to_state='TERMINAL'") == [(1,)]


@pytest.mark.parametrize("field", ["nonce_sha256", "command_sha256"])
def test_launch_carries_and_revalidates_immutable_attempt_hashes(received, field):
    db, _, pending = received
    active = launch(db, lease(db, pending))
    assert active["nonce_sha256"] == HASH
    assert active["command_sha256"] == "b" * 64
    wrong = dict(active, **{field: "c" * 64})
    assert not delivery.record_started(db, wrong, process_pid=123)
    assert not delivery.record_not_started(db, wrong, "not_ours")
    assert not delivery.record_in_doubt(db, wrong, "not_ours")
    assert delivery.record_started(db, active, process_pid=123)
    assert not delivery.record_terminal(db, wrong, outcome="ok", exit_code=0, cleanup_verified=True)
    assert delivery.record_terminal(db, active, outcome="ok", exit_code=0, cleanup_verified=True)


@pytest.mark.parametrize("table,state", [("trigger_deliveries", "LEASED"),
                                        ("trigger_delivery_transitions", "LEASED")])
def test_lease_ignored_write_does_not_report_success(received, table, state):
    db, _, pending = received
    operation = "UPDATE" if table == "trigger_deliveries" else "INSERT"
    field = "state" if operation == "UPDATE" else "to_state"
    execute(db, f"CREATE TRIGGER ignore_lease BEFORE {operation} ON {table} "
            f"WHEN NEW.{field}='{state}' BEGIN SELECT RAISE(IGNORE); END")
    with pytest.raises(TriggerDeliveryError):
        lease(db, pending)
    assert query(db, "SELECT state,generation,owner FROM trigger_deliveries") == [("PENDING", 0, None)]
    assert query(db, "SELECT COUNT(*) FROM trigger_delivery_transitions") == [(1,)]


def test_unverified_cleanup_is_in_doubt_not_terminal_or_pending(received):
    db, trigger, pending = received
    active = launch(db, lease(db, pending))
    assert delivery.record_started(db, active, process_pid=123)
    assert delivery.record_terminal(db, active, outcome="error", exit_code=None, cleanup_verified=False)
    assert delivery.record_terminal(db, active, outcome="error", exit_code=None, cleanup_verified=False)
    assert observe(db, trigger, now=100_000)["reason"] == "in_doubt"
    assert query(db, "SELECT state FROM trigger_deliveries") == [("IN_DOUBT",)]


@pytest.mark.parametrize("target", ["attempt", "state", "event"])
def test_begin_launch_ignored_write_rolls_back_every_component(received, target):
    db, _, pending = received
    token = lease(db, pending)
    if target == "attempt":
        clause = "BEFORE INSERT ON trigger_delivery_attempts"
    elif target == "state":
        clause = "BEFORE UPDATE ON trigger_deliveries WHEN NEW.state='LAUNCHING'"
    else:
        clause = "BEFORE INSERT ON trigger_delivery_transitions WHEN NEW.to_state='LAUNCHING'"
    execute(db, "CREATE TRIGGER ignore_write " + clause + " BEGIN SELECT RAISE(IGNORE); END")
    before = query(db, "SELECT * FROM trigger_deliveries")
    with pytest.raises(TriggerDeliveryError):
        launch(db, token)
    assert query(db, "SELECT * FROM trigger_deliveries") == before
    assert query(db, "SELECT COUNT(*) FROM trigger_delivery_attempts") == [(0,)]
    assert query(db, "SELECT COUNT(*) FROM trigger_delivery_transitions") == [(2,)]


@pytest.mark.parametrize("target", ["legacy", "delivery", "event"])
def test_intake_ignored_write_never_leaves_half_reservation(received, target):
    db, trigger, _ = received
    table = {"legacy": "trigger_consumptions", "delivery": "trigger_deliveries",
             "event": "trigger_delivery_transitions"}[target]
    execute(db, "CREATE TRIGGER ignore_intake BEFORE INSERT ON " + table +
            " BEGIN SELECT RAISE(IGNORE); END")
    trigger.write_text("new request rejected atomically")
    response = observe(db, trigger)
    assert response["blocked"] and response["pending"] is None
    for name in ("trigger_consumptions", "trigger_deliveries", "trigger_delivery_transitions"):
        assert query(db, "SELECT COUNT(*) FROM " + name) == [(1,)]
    assert trigger.read_text() == "new request rejected atomically"


def test_started_ignored_update_leaves_launching_and_can_explicitly_hold(received):
    db, trigger, pending = received
    active = launch(db, lease(db, pending))
    execute(db, "CREATE TRIGGER ignore_started BEFORE UPDATE ON trigger_deliveries "
            "WHEN NEW.state='STARTED' BEGIN SELECT RAISE(IGNORE); END")
    with pytest.raises(TriggerDeliveryError):
        delivery.record_started(db, active, process_pid=123)
    assert query(db, "SELECT state,process_pid FROM trigger_deliveries") == [("LAUNCHING", None)]
    assert delivery.record_in_doubt(db, active, "start_receipt_write_failed")
    assert observe(db, trigger)["blocked"]


def test_mutations_never_create_a_missing_database(tmp_path):
    db = tmp_path / "missing.db"
    with pytest.raises(TriggerDeliveryError, match="missing"):
        delivery.lease_trigger(db, "delivery", scope=SCOPE, role_name=ROLE,
                               expected_sha256=HASH, owner="owner")
    assert not db.exists()


def test_incompatible_or_redirected_database_is_readonly_fail_closed(tmp_path):
    db, trigger = tmp_path / "bad.db", tmp_path / "trigger"
    execute(db, "CREATE TABLE trigger_deliveries(delivery_id TEXT)")
    before = db.read_bytes()
    assert observe(db, trigger)["blocked"]
    assert db.read_bytes() == before
    link = tmp_path / "redirect.db"
    link.symlink_to(db)
    assert observe(link, trigger)["reason"] == "trigger_database_redirected"
    assert db.read_bytes() == before


def test_wal_database_is_not_silently_changed_by_observer(tmp_path):
    db, trigger = tmp_path / "wal.db", tmp_path / "trigger"
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE untouched(value TEXT)")
    trigger.write_text("request")
    assert observe(db, trigger)["reason"] == "trigger_database_policy_invalid"
    assert query(db, "PRAGMA journal_mode") == [("wal",)]
    assert not query(db, "SELECT name FROM sqlite_master WHERE name='trigger_deliveries'")
