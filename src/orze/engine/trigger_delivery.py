"""Durable trigger delivery: lease before launch, uncertainty never auto-replays.

CALLING SPEC:
    observe_trigger(db_path, scope, role_name, trigger_file, *, now=None)
        -> {pending: dict|None, blocked: bool, reason: str}
    lease_trigger(db_path, delivery_id, *, scope, role_name, expected_sha256,
                  owner, ttl=60, now=None) -> dict|None
    defer_trigger(db_path, lease, reason, *, now=None) -> bool
    begin_launch(db_path, lease, *, attempt_id, nonce_sha256, command_sha256,
                 now=None) -> dict|None
    record_started(db_path, launch, *, process_pid, now=None) -> bool
    record_not_started(db_path, launch, reason, *, now=None) -> bool
    record_in_doubt(db_path, launch, reason, *, now=None) -> bool
    record_terminal(db_path, launch, *, outcome, exit_code, cleanup_verified,
                    now=None) -> bool

PID here is diagnostic, never signal authority. The caller must attest its
RoleProcess first. record_not_started is reserved for positively known pre-exec
failure; a missing process or receipt is NOT proof. Process terminal outcomes do
not establish scientific success. Native ingress never unlinks a live file.
"""
from __future__ import annotations

import hashlib
import math
import os
import socket
import sqlite3
import uuid
from pathlib import Path

from orze.engine.trigger_delivery_storage import (
    STATES, TriggerDeliveryError, bounded_text, connect, digest, ensure_schema,
    event, schema_status, timestamp, write_row,
)
from orze.engine.trigger_ingress import MAX_PAYLOAD_BYTES, read_trigger


def _get(conn, delivery_id):
    row = conn.execute("SELECT * FROM trigger_deliveries WHERE delivery_id=?", (delivery_id,)).fetchone()
    if row is None:
        return None
    row = dict(row)
    payload = row["payload"]
    if (not isinstance(payload, str) or len(payload.encode("utf-8")) > MAX_PAYLOAD_BYTES
            or hashlib.sha256(payload.encode("utf-8")).hexdigest() != row["payload_sha256"]
            or row["state"] not in STATES or type(row["generation"]) is not int
            or row["generation"] < 0):
        raise TriggerDeliveryError("trigger_delivery_invalid")
    if row["state"] == "LEASED" and (
        not isinstance(row["owner"], str) or not row["owner"]
        or type(row["lease_until"]) not in (int, float)
        or not math.isfinite(row["lease_until"])
    ):
        raise TriggerDeliveryError("trigger_delivery_invalid")
    if row["state"] in ("LAUNCHING", "STARTED", "IN_DOUBT", "TERMINAL"):
        attempt = conn.execute(
            "SELECT * FROM trigger_delivery_attempts WHERE attempt_id=?", (row["attempt_id"],)
        ).fetchone()
        if (attempt is None or any(attempt[key] != row[key] for key in
                                  ("delivery_id", "generation", "owner"))):
            raise TriggerDeliveryError("trigger_attempt_invalid")
    return row


def _blocked(conn, scope, role_name, now, *, excluding=None):
    row = conn.execute(
        "SELECT delivery_id,state FROM trigger_deliveries WHERE scope=? AND role_name=? "
        "AND delivery_id IS NOT ? AND (state NOT IN ('PENDING','TERMINAL','LEASED') "
        "OR state IS NULL OR (state='LEASED' AND (lease_until IS NULL OR lease_until>?))) "
        "ORDER BY created_at,delivery_id LIMIT 1", (scope, role_name, excluding, now),
    ).fetchone()
    if row is None:
        return None
    _get(conn, row["delivery_id"])
    return row["state"].lower()


def _summary(conn, scope, role_name, now):
    reason = _blocked(conn, scope, role_name, now)
    if reason:
        return {"pending": None, "blocked": True, "reason": reason}
    row = conn.execute(
        "SELECT delivery_id FROM trigger_deliveries WHERE scope=? AND role_name=? "
        "AND (state='PENDING' OR (state='LEASED' AND lease_until<=?)) "
        "ORDER BY created_at,delivery_id LIMIT 1", (scope, role_name, now),
    ).fetchone()
    pending = _get(conn, row[0]) if row else None
    return {"pending": pending, "blocked": False, "reason": "pending" if pending else "no_pending"}


def _existing_snapshot(conn, snapshot, scope, role_name, has_schema):
    table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='trigger_consumptions' AND type='table'"
    ).fetchone()
    if table is None:
        return None
    legacy = conn.execute(
        "SELECT id FROM trigger_consumptions WHERE role_name=? AND fingerprint=?",
        (role_name, snapshot["fingerprint"]),
    ).fetchone()
    if legacy is None:
        return None
    mapped = conn.execute(
        "SELECT delivery_id FROM trigger_deliveries WHERE legacy_consumption_id=?", (legacy[0],)
    ).fetchone() if has_schema else None
    if mapped is None:
        return "legacy_consumed"
    row = _get(conn, mapped[0])
    if any(row[key] != value for key, value in (
        ("scope", scope), ("role_name", role_name),
        ("file_path", snapshot["file_path"]), ("payload_sha256", snapshot["payload_sha256"]),
    )):
        return "trigger_source_collision"
    return "known"


def _intake(db_path, scope, role_name, snapshot, now):
    conn = connect(db_path, write=True, create=True)
    try:
        conn.execute("BEGIN IMMEDIATE")
        ensure_schema(conn)
        existing = _existing_snapshot(conn, snapshot, scope, role_name, True)
        if existing:
            conn.rollback()
            return existing
        legacy_values = (role_name, snapshot["file_path"], snapshot["fingerprint"],
                         snapshot["payload"], str(now), socket.gethostname(), os.getpid())
        legacy = conn.execute(
            "INSERT INTO trigger_consumptions(role_name,file_path,fingerprint,payload,"
            "consumed_at,consumed_by_host,consumed_by_pid) VALUES (?,?,?,?,?,?,?)", legacy_values,
        )
        saved = conn.execute(
            "SELECT role_name,file_path,fingerprint,payload,consumed_at,consumed_by_host,"
            "consumed_by_pid FROM trigger_consumptions WHERE id=?", (legacy.lastrowid,),
        ).fetchone()
        if legacy.rowcount != 1 or saved is None or tuple(saved) != legacy_values:
            raise TriggerDeliveryError("trigger_intake_write_rejected")
        delivery_id = uuid.uuid4().hex
        values = (delivery_id, legacy.lastrowid, scope, role_name, snapshot["source_key"],
                  snapshot["file_path"], snapshot["fingerprint"], snapshot["payload"],
                  snapshot["payload_sha256"], now, now)
        cursor = conn.execute(
            "INSERT INTO trigger_deliveries(delivery_id,legacy_consumption_id,scope,role_name,"
            "source_key,file_path,fingerprint,payload,payload_sha256,state,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,'PENDING',?,?)", values,
        )
        row = _get(conn, delivery_id)
        expected = dict(zip(("delivery_id", "legacy_consumption_id", "scope", "role_name",
                            "source_key", "file_path", "fingerprint", "payload", "payload_sha256",
                            "created_at", "updated_at"), values))
        expected.update(state="PENDING", generation=0, owner=None, lease_until=None,
                        attempt_id=None, process_pid=None)
        if cursor.rowcount != 1 or row != expected:
            raise TriggerDeliveryError("trigger_intake_write_rejected")
        event(conn, row, None, "trigger_received", now)
        conn.commit()
        return "known"
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def observe_trigger(db_path, scope, role_name, trigger_file, *, now=None):
    """Intake a stable file if new, then inspect the durable per-role queue."""
    bounded_text(scope, "scope")
    bounded_text(role_name, "role", 256)
    now = timestamp(now)
    try:
        snapshot = read_trigger(trigger_file)
        path = Path(db_path)
        if not path.exists() and not path.is_symlink() and snapshot is None:
            return {"pending": None, "blocked": False, "reason": "no_pending"}
        known = None
        if path.exists() or path.is_symlink():
            conn = connect(db_path)
            try:
                conn.execute("BEGIN")
                has_schema = schema_status(conn)
                if snapshot is not None:
                    known = _existing_snapshot(conn, snapshot, scope, role_name, has_schema)
                if known not in (None, "known"):
                    return {"pending": None, "blocked": True, "reason": known}
                if snapshot is None or known == "known":
                    return _summary(conn, scope, role_name, now) if has_schema else {
                        "pending": None, "blocked": False, "reason": "no_pending"}
            finally:
                conn.close()
        if snapshot is not None:
            known = _intake(db_path, scope, role_name, snapshot, now)
            if known != "known":
                return {"pending": None, "blocked": True, "reason": known}
        conn = connect(db_path)
        try:
            conn.execute("BEGIN")
            if not schema_status(conn):
                raise TriggerDeliveryError("trigger_database_schema_invalid")
            return _summary(conn, scope, role_name, now)
        finally:
            conn.close()
    except (OSError, sqlite3.Error, ValueError, RuntimeError) as exc:
        reason = str(exc) if isinstance(exc, TriggerDeliveryError) else "trigger_observation_unavailable"
        return {"pending": None, "blocked": True, "reason": reason}


def _transaction(db_path, operation):
    conn = connect(db_path, write=True)
    try:
        conn.execute("BEGIN IMMEDIATE")
        if not schema_status(conn):
            raise TriggerDeliveryError("trigger_database_schema_invalid")
        result = operation(conn)
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _matches(row, token, *, launch=False, conn=None):
    keys = ("delivery_id", "scope", "role_name", "payload_sha256", "owner", "generation")
    if launch:
        keys += ("attempt_id",)
    matched = (row is not None and isinstance(token, dict)
            and isinstance(token.get("owner"), str) and bool(token["owner"])
            and type(token.get("generation")) is int and token["generation"] > 0
            and all(key in token and row[key] == token[key] for key in keys))
    if not matched or not launch:
        return matched
    attempt = conn.execute(
        "SELECT nonce_sha256,command_sha256 FROM trigger_delivery_attempts WHERE attempt_id=?",
        (row["attempt_id"],),
    ).fetchone()
    return (attempt is not None and all(key in token and token[key] == attempt[key]
                                       for key in ("nonce_sha256", "command_sha256")))


def lease_trigger(db_path, delivery_id, *, scope, role_name, expected_sha256,
                  owner, ttl=60, now=None):
    bounded_text(delivery_id, "delivery_id", 256)
    bounded_text(scope, "scope")
    bounded_text(role_name, "role", 256)
    bounded_text(owner, "owner", 256)
    digest(expected_sha256)
    now = timestamp(now)
    if type(ttl) not in (int, float) or not math.isfinite(ttl) or ttl <= 0 or not math.isfinite(now + ttl):
        raise TriggerDeliveryError("trigger_ttl_invalid")

    def action(conn):
        row = _get(conn, delivery_id)
        if (row is None or row["scope"] != scope or row["role_name"] != role_name
                or row["payload_sha256"] != expected_sha256
                or _blocked(conn, scope, role_name, now, excluding=delivery_id)):
            return None
        if not (row["state"] == "PENDING" or
                (row["state"] == "LEASED" and row["lease_until"] <= now)):
            return None
        new = write_row(conn, row, dict(state="LEASED", generation=row["generation"] + 1,
                                       owner=owner, lease_until=now + ttl, attempt_id=None,
                                       process_pid=None, updated_at=now))
        event(conn, new, row["state"], "trigger_leased", now)
        return new
    return _transaction(db_path, action)


def _transition(db_path, token, from_states, to_state, reason, now, *, launch=True,
                extra=None, idempotent=None, outcome=None, exit_code=None, cleanup_verified=None):
    bounded_text(reason, "reason", 1024)
    now = timestamp(now)
    if not isinstance(token, dict) or not isinstance(token.get("delivery_id"), str):
        return False

    def action(conn):
        row = _get(conn, token["delivery_id"])
        if not _matches(row, token, launch=launch, conn=conn):
            return False
        if row["state"] not in from_states:
            return bool(idempotent and idempotent(conn, row))
        changes = dict(state=to_state, updated_at=now)
        changes.update(extra or {})
        new = write_row(conn, row, changes)
        event(conn, new, row["state"], reason, now, outcome=outcome,
              exit_code=exit_code, cleanup_verified=cleanup_verified)
        return True
    return _transaction(db_path, action)


def defer_trigger(db_path, lease, reason, *, now=None):
    return _transition(db_path, lease, ("LEASED",), "PENDING", reason, now,
                       launch=False, extra={"lease_until": None})


def begin_launch(db_path, lease, *, attempt_id, nonce_sha256, command_sha256, now=None):
    bounded_text(attempt_id, "attempt_id", 256)
    digest(nonce_sha256)
    digest(command_sha256)
    now = timestamp(now)
    if not isinstance(lease, dict) or not isinstance(lease.get("delivery_id"), str):
        return None

    def action(conn):
        row = _get(conn, lease["delivery_id"])
        if (not _matches(row, lease) or row["state"] != "LEASED" or row["lease_until"] <= now
                or _blocked(conn, row["scope"], row["role_name"], now, excluding=row["delivery_id"])):
            return None
        if conn.execute("SELECT 1 FROM trigger_delivery_attempts WHERE attempt_id=?", (attempt_id,)).fetchone():
            return None
        values = (attempt_id, row["delivery_id"], row["generation"], row["owner"],
                  nonce_sha256, command_sha256, now)
        cursor = conn.execute(
            "INSERT INTO trigger_delivery_attempts(attempt_id,delivery_id,generation,owner,"
            "nonce_sha256,command_sha256,created_at) VALUES (?,?,?,?,?,?,?)", values,
        )
        saved = conn.execute("SELECT * FROM trigger_delivery_attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
        if cursor.rowcount != 1 or saved is None or tuple(saved) != values:
            raise TriggerDeliveryError("trigger_attempt_write_rejected")
        new = write_row(conn, row, dict(state="LAUNCHING", attempt_id=attempt_id,
                                       lease_until=None, updated_at=now))
        event(conn, new, "LEASED", "trigger_launching", now)
        return dict(new, nonce_sha256=nonce_sha256, command_sha256=command_sha256)
    return _transaction(db_path, action)


def record_started(db_path, launch, *, process_pid, now=None):
    if type(process_pid) is not int or process_pid <= 0:
        raise TriggerDeliveryError("trigger_process_pid_invalid")
    return _transition(db_path, launch, ("LAUNCHING",), "STARTED", "trigger_started", now,
                       extra={"process_pid": process_pid},
                       idempotent=lambda conn, row: row["state"] == "STARTED" and row["process_pid"] == process_pid)


def record_not_started(db_path, launch, reason, *, now=None):
    return _transition(db_path, launch, ("LAUNCHING",), "PENDING", reason, now,
                       extra={"lease_until": None, "process_pid": None})


def record_in_doubt(db_path, launch, reason, *, now=None):
    return _transition(db_path, launch, ("LAUNCHING", "STARTED"), "IN_DOUBT", reason, now)


def record_terminal(db_path, launch, *, outcome, exit_code, cleanup_verified, now=None):
    bounded_text(outcome, "outcome", 256)
    if type(cleanup_verified) is not bool or (exit_code is not None and type(exit_code) is not int):
        raise TriggerDeliveryError("trigger_terminal_invalid")
    state = "TERMINAL" if cleanup_verified else "IN_DOUBT"
    reason = "trigger_terminal" if cleanup_verified else "trigger_cleanup_unverified"

    def same_terminal(conn, row):
        if row["state"] != state:
            return False
        latest = conn.execute(
            "SELECT to_state,reason,outcome,exit_code,cleanup_verified FROM trigger_delivery_transitions "
            "WHERE delivery_id=? ORDER BY id DESC LIMIT 1", (row["delivery_id"],),
        ).fetchone()
        return latest is not None and tuple(latest) == (state, reason, outcome, exit_code, int(cleanup_verified))

    return _transition(db_path, launch, ("STARTED",), state, reason, now,
                       idempotent=same_terminal, outcome=outcome, exit_code=exit_code,
                       cleanup_verified=int(cleanup_verified))
