"""New delivery/process contract through actual Core completion and SQLite.

Only Linux identity/reaping boundaries are simulated. Receipt files, lifecycle
state, transition transactions and check_active_roles are real.
"""
import hashlib
import json
import sqlite3
import time
from types import SimpleNamespace
import uuid

import pytest

from orze.engine import process, roles, trigger_delivery as delivery
from orze.idea_lake import IdeaLake


@pytest.fixture
def project(tmp_path, monkeypatch):
    lake = IdeaLake(tmp_path / "authority.db")
    source = tmp_path / "_trigger_worker"
    source.write_text("PRIVATE REQUEST NOT FOR PROCESS RECEIPT", encoding="utf-8")
    pending = delivery.observe_trigger(lake.db_path, "test-scope", "worker", source)["pending"]
    lease = delivery.lease_trigger(
        lake.db_path, pending["delivery_id"], scope="test-scope", role_name="worker",
        expected_sha256=pending["payload_sha256"], owner="test-controller")
    nonce = "f" * 64
    launch = delivery.begin_launch(
        lake.db_path, lease, attempt_id=uuid.uuid4().hex,
        nonce_sha256=hashlib.sha256(nonce.encode()).hexdigest(), command_sha256="a" * 64)
    assert delivery.record_started(lake.db_path, launch, process_pid=12345)
    fake = SimpleNamespace(pid=None, poll=lambda: 0)
    lock = tmp_path / "locks" / "worker"
    lock.mkdir(parents=True)
    log = tmp_path / "worker.log"
    rp = process.RoleProcess(
        role_name="worker", process=fake, start_time=time.time(), log_path=log,
        timeout=60, lock_dir=lock, cycle_num=1, _log_fh=log.open("w"),
        writes_ideas_file=False, process_nonce=nonce,
        trigger_delivery_db=str(lake.db_path), trigger_launch=launch)
    # Supply simulated OS identity after construction; no /proc traversal.
    fake.pid = 12345
    rp._root_start_ticks, rp._pgid = 23, 12345
    monkeypatch.setattr(process, "capture_process_identity", lambda pid: {
        "pid": pid, "pgid": pid, "start_ticks": 23})
    monkeypatch.setattr(process, "refresh_role_process_descendants", lambda rp: 0)
    monkeypatch.setattr(roles, "refresh_role_process_descendants", lambda rp: 0)
    monkeypatch.setattr(process, "process_is_running", lambda *args: False)
    monkeypatch.setattr(process, "_process_identities_with_nonce_hash", lambda value: [])
    monkeypatch.setattr(process, "_terminate_and_reap", lambda *args, **kwargs: True)
    monkeypatch.setattr(roles, "_terminate_and_reap", lambda *args, **kwargs: True)
    assert process.persist_role_process_receipt(rp)
    p = SimpleNamespace(root=tmp_path, lake=lake, source=source, launch=launch,
                        rp=rp, active={"worker": rp}, receipt=lock / "role-process.json")
    try:
        yield p
    finally:
        rp.close_log()
        lake.close()


def _state(p):
    with sqlite3.connect(p.lake.db_path) as connection:
        return connection.execute("SELECT state FROM trigger_deliveries WHERE delivery_id=?",
                                  (p.launch["delivery_id"],)).fetchone()[0]


@pytest.mark.parametrize("exit_code,expected", [(0, roles.OUTCOME_OK),
                                               (1, roles.OUTCOME_ERROR),
                                               (42, roles.OUTCOME_RATE_LIMITED)])
def test_actual_completion_commits_attempt_before_releasing_receipt(project, exit_code, expected):
    p = project
    p.rp.process.poll = lambda: exit_code
    result = roles.check_active_roles(p.active, ideas_file=str(p.root / "ideas.md"))
    assert result == [("worker", expected)]
    assert _state(p) == "TERMINAL"
    assert not p.receipt.exists()
    assert not p.active
    assert p.source.exists()  # Native receipt never unlinks the ingress source.
    observed = delivery.observe_trigger(p.lake.db_path, "test-scope", "worker", p.source)
    assert not observed["blocked"] and observed["pending"] is None


def test_terminal_write_ignore_keeps_uncertain_receipt_and_blocks_replay(project):
    p = project
    before = p.receipt.read_bytes()
    with sqlite3.connect(p.lake.db_path) as connection:
        connection.execute("CREATE TRIGGER refuse_terminal BEFORE UPDATE ON trigger_deliveries "
                           "WHEN NEW.state='TERMINAL' BEGIN SELECT RAISE(IGNORE); END")
    result = roles.check_active_roles(p.active, ideas_file=str(p.root / "ideas.md"))
    assert result == [("worker", roles.OUTCOME_ERROR)]
    assert _state(p) in ("STARTED", "IN_DOUBT")
    assert p.receipt.read_bytes() == before
    observed = delivery.observe_trigger(p.lake.db_path, "test-scope", "worker", p.source)
    assert observed["blocked"] and observed["pending"] is None


def test_unverified_cleanup_is_not_terminal_or_permission_to_retry(project, monkeypatch):
    p = project
    monkeypatch.setattr(roles, "_terminate_and_reap", lambda *args, **kwargs: False)
    result = roles.check_active_roles(p.active, ideas_file=str(p.root / "ideas.md"))
    assert result == [("worker", roles.OUTCOME_ERROR)]
    assert _state(p) == "IN_DOUBT"
    assert p.receipt.exists()
    assert delivery.observe_trigger(p.lake.db_path, "test-scope", "worker", p.source)["blocked"]


def test_shutdown_reaping_does_not_invent_safe_retry_or_delete_delivery_receipt(project):
    p = project
    assert process.terminate_role_process(p.rp, "simulated shutdown")
    p.active.clear()
    assert _state(p) == "STARTED"
    assert p.receipt.exists()
    assert delivery.observe_trigger(p.lake.db_path, "test-scope", "worker", p.source)["blocked"]


def test_timeout_records_operational_terminal_without_scientific_success(project):
    p = project
    p.rp.process.poll = lambda: None
    p.rp.start_time = time.time() - 120
    result = roles.check_active_roles(p.active, ideas_file=str(p.root / "ideas.md"))
    assert result == [("worker", roles.OUTCOME_TIMEOUT)]
    assert _state(p) == "TERMINAL"
    assert not p.receipt.exists()


def test_process_receipt_contains_only_captured_ids_and_hashes(project):
    p = project
    p.launch["attempt_id"] = "later-mutable-role-state"
    assert process.persist_role_process_receipt(p.rp)
    receipt_text = p.receipt.read_text(encoding="utf-8")
    receipt = json.loads(receipt_text)
    reference = receipt["trigger_delivery"]
    assert set(reference) == {"delivery_id", "scope", "role_name", "generation",
                              "attempt_id", "nonce_sha256", "command_sha256"}
    assert reference["attempt_id"] != p.launch["attempt_id"]
    assert "PRIVATE REQUEST" not in receipt_text
    assert "payload" not in receipt_text
    assert "trigger_delivery_db" not in receipt_text


def test_foreign_process_receipt_is_not_removed_by_old_completion(project):
    p = project
    foreign = json.loads(p.receipt.read_text(encoding="utf-8"))
    foreign["trigger_delivery"]["attempt_id"] = "another-attempt"
    foreign["nonce_sha256"] = "b" * 64
    p.receipt.write_text(json.dumps(foreign), encoding="utf-8")
    result = roles.check_active_roles(p.active, ideas_file=str(p.root / "ideas.md"))
    assert result == [("worker", roles.OUTCOME_ERROR)]
    assert _state(p) == "TERMINAL"  # Own result committed, foreign file retained.
    assert json.loads(p.receipt.read_text(encoding="utf-8")) == foreign


def test_malformed_nonlegacy_binding_cannot_discard_process_receipt(project):
    p = project
    before = p.receipt.read_bytes()
    p.rp.trigger_launch = {}
    assert not process.persist_role_process_receipt(p.rp)
    result = roles.check_active_roles(p.active, ideas_file=str(p.root / "ideas.md"))
    assert result == [("worker", roles.OUTCOME_ERROR)]
    assert p.receipt.read_bytes() == before
    assert _state(p) == "STARTED"
