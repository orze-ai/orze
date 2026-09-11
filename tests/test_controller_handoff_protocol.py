"""New local wire/storage boundaries, not substitutes for real CLI handoff.

Socket tests use real UNIX SEQPACKET + SCM_CREDENTIALS from this process.
Storage tests use explicit in-memory SQLite and grant no controller authority.
"""
import os
import socket
import sqlite3
import time

import pytest

from orze.engine import controller_handoff as handoff
from orze.engine.controller_control import ControllerHOLD


def test_real_channel_binds_each_packet_to_kernel_sender():
    left, right = handoff._new_channel()
    try:
        packet = {"schema": 1, "event": "wire_test"}
        handoff._send(left, packet)
        assert handoff._receive(right, os.getpid(), time.monotonic() + 1) == packet
        handoff._send(left, packet)
        with pytest.raises(ControllerHOLD, match="peer_unconfirmed"):
            handoff._receive(right, os.getpid() + 100000000, time.monotonic() + 1)
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize("raw", [b'{} ', b'{"x":1,"x":1}', b'[]', b'\xff', b'x' * 65537])
def test_noncanonical_or_oversized_actual_packet_is_refused(raw):
    left, right = handoff._new_channel()
    try:
        left.send(raw)
        with pytest.raises(ControllerHOLD):
            handoff._receive(right, os.getpid(), time.monotonic() + 1)
    finally:
        left.close()
        right.close()


def test_inherited_stream_claim_does_not_close_foreign_fd(monkeypatch):
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        monkeypatch.setenv(handoff._FD_ENV, str(right.fileno()))
        with pytest.raises(ControllerHOLD, match="channel_invalid"):
            handoff._inherited_channel()
        left.sendall(b"owned")
        assert right.recv(5) == b"owned"
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize("value", [True, False, 0, -1, 3601, float("inf"), float("nan"), "5"])
def test_timeout_requires_bounded_finite_number(value):
    with pytest.raises(ControllerHOLD, match="timeout_invalid"):
        handoff._timeout(value)


@pytest.mark.parametrize("claim", [None, True, {"state": "ISSUED"}, object()])
def test_admission_is_not_a_public_label_or_copied_mapping(monkeypatch, claim):
    monkeypatch.setattr(handoff, "_ADMISSION", claim)
    with pytest.raises(ControllerHOLD, match="admission_invalid"):
        handoff._validated_admission(claim, None, None)


def test_storage_keeps_single_request_and_single_source_and_rejects_trigger():
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("BEGIN IMMEDIATE")
        handoff._schema(conn, create=True)
        insert = "INSERT INTO controller_handoffs VALUES (?,?,?,?,?,?,'SPAWNING',NULL,NULL,NULL)"
        conn.execute(insert, ("grant", "request", "scope", "source", "target", "{}"))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(insert, ("grant-2", "request", "scope", "source-2", "target-2", "{}"))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(insert, ("grant-3", "request-3", "scope", "source", "target-3", "{}"))
        assert conn.execute("SELECT count(*) FROM controller_handoffs").fetchone()[0] == 1
        conn.execute("CREATE TRIGGER unqualified AFTER UPDATE ON controller_handoffs BEGIN SELECT 1; END")
        with pytest.raises(ControllerHOLD, match="trigger_unsupported"):
            handoff._schema(conn)
    finally:
        conn.close()


def test_completed_result_is_only_information():
    result = handoff.CompletedControllerHandoff("r", "g", "old", "new", 1, "digest")
    with pytest.raises(TypeError, match="not launch authority"):
        bool(result)
