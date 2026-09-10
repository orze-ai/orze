"""Bounded native role-result failures through real Core completion/trigger DB.

The producer envelope and consumer are real. Only Linux process identity/reap
boundaries (from the existing fixture) and selected publication IO are faked.
New protocol absence is not an old-behavior red. No provider or GPU executes.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from orze.core import research_result
from orze.engine import process, roles
from test_trigger_role_completion import project as trigger_project, _state


@pytest.fixture
def project(trigger_project):
    p = trigger_project
    results = p.root / "results"
    results.mkdir()
    ideas = p.root / "ideas.md"
    ideas.write_text("# Ideas\n", encoding="utf-8")
    result_dir = p.root / ".orze" / "state" / "agent_results"
    result_dir.mkdir(parents=True)
    result_path = result_dir / (p.launch["attempt_id"] + ".json")
    ref = research_result.make_native_result_ref(
        attempt_id=p.launch["attempt_id"], role_name="worker",
        project_root=p.root, results_dir=results, ideas_file=ideas,
        result_path=result_path, process_nonce=p.rp.process_nonce,
    )
    p.rp.native_result_ref = ref
    p.rp.native_result_details = None
    p.ref, p.result_path, p.ideas = ref, result_path, ideas
    return p


def _result():
    return {
        "status": "accepted", "reason": "proposals_accepted",
        "accepted_ids": ["idea-result-boundary"], "accepted_count": 1,
        "rejected_count": 0, "rejection_reasons": {},
    }


def _publish(p):
    research_result.publish_native_result(
        p.ref, _result(), process_nonce=p.rp.process_nonce)
    assert research_result.read_native_result(
        p.ref, process_nonce=p.rp.process_nonce) == _result()


def _assert_nonproductive_completion(p):
    # The OS leader exits zero, but missing/unbound native result evidence is
    # not productive. Actual settlement must store the same operational error.
    assert p.rp.process.poll() == 0
    finished = roles.check_active_roles(p.active, ideas_file=str(p.ideas))
    assert finished == [("worker", roles.OUTCOME_ERROR)]
    assert not p.active
    assert _state(p) == "TERMINAL"
    row = p.lake.conn.execute(
        "SELECT outcome FROM trigger_delivery_transitions WHERE attempt_id=? "
        "AND to_state='TERMINAL' ORDER BY id DESC LIMIT 1",
        (p.launch["attempt_id"],),
    ).fetchone()
    assert row is not None and row[0] == "error"
    assert p.source.exists()


@pytest.mark.parametrize("field", ["attempt_id", "nonce_sha256", "project_root"])
def test_foreign_native_result_identity_never_becomes_productive(project, field):
    p = project
    _publish(p)
    value = json.loads(p.result_path.read_bytes())
    if field == "attempt_id":
        value["identity"][field] = "another-attempt"
    elif field == "nonce_sha256":
        value["identity"][field] = hashlib.sha256(b"another-process").hexdigest()
    else:
        value["identity"][field] = str(p.root / "another-project")
    p.result_path.write_text(json.dumps(value), encoding="utf-8")
    _assert_nonproductive_completion(p)


@pytest.mark.parametrize("damage", ["malformed", "oversize"])
def test_unreadable_native_result_never_becomes_productive(project, damage):
    p = project
    if damage == "malformed":
        p.result_path.write_bytes(b'{"schema":')
    else:
        # Deliberately beyond the agreed 64-KiB result contract, without a
        # large manifest or provider response fixture.
        p.result_path.write_bytes(b" " * (64 * 1024 + 1))
    _assert_nonproductive_completion(p)


def test_missing_native_result_rejects_but_explicit_legacy_role_remains_compatible(project):
    p = project
    assert not p.result_path.exists()
    _assert_nonproductive_completion(p)

    # A separate legacy handle declares no native-result contract. It keeps
    # existing exit-zero behavior, not authority borrowed from the bad native
    # artifact or its settled trigger delivery.
    legacy_lock = p.root / "locks" / "legacy"
    legacy_lock.mkdir()
    log = p.root / "legacy.log"
    legacy = process.RoleProcess(
        role_name="legacy", process=SimpleNamespace(pid=None, poll=lambda: 0),
        start_time=p.rp.start_time, log_path=log, timeout=60,
        lock_dir=legacy_lock, cycle_num=1, _log_fh=log.open("w"),
        writes_ideas_file=False,
    )
    active = {"legacy": legacy}
    try:
        assert roles.check_active_roles(active, ideas_file=str(p.ideas)) == [
            ("legacy", roles.OUTCOME_OK)]
        assert not active
    finally:
        legacy.close_log()


def _temporary_result_fd(fd, final_path):
    try:
        info = os.fstat(fd)
        path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        return stat.S_ISREG(info.st_mode) and path.parent == final_path.parent and path != final_path
    except OSError:
        return False


def test_result_write_failure_cannot_leave_a_productive_publication(project, monkeypatch):
    p = project
    real_write = os.write
    faults = []

    def failed_temporary_write(fd, data):
        if _temporary_result_fd(fd, p.result_path):
            faults.append(fd)
            raise OSError("test-only native result write unavailable")
        return real_write(fd, data)

    with monkeypatch.context() as fault:
        fault.setattr(research_result.os, "write", failed_temporary_write)
        with pytest.raises((OSError, research_result.NativeResultError)):
            research_result.publish_native_result(
                p.ref, _result(), process_nonce=p.rp.process_nonce)
    assert faults, "the actual result temporary-file write must be reached"
    assert not p.result_path.exists()
    _assert_nonproductive_completion(p)


def test_result_close_uncertainty_cannot_leave_a_productive_publication(project, monkeypatch):
    p = project
    real_close = os.close
    faults = []

    def unconfirmed_temporary_close(fd):
        selected = _temporary_result_fd(fd, p.result_path)
        real_close(fd)  # Avoid leaking the real fixture descriptor.
        if selected:
            faults.append(fd)
            raise OSError("test-only native result close confirmation unavailable")

    with monkeypatch.context() as fault:
        fault.setattr(research_result.os, "close", unconfirmed_temporary_close)
        with pytest.raises((OSError, research_result.NativeResultError)):
            research_result.publish_native_result(
                p.ref, _result(), process_nonce=p.rp.process_nonce)
    assert len(faults) == 1, "the actual temporary result close must be reached once"
    assert not p.result_path.exists()
    _assert_nonproductive_completion(p)
