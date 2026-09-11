"""New role protocol mechanisms; real tiny CPU workers, no provider or GPU."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from orze.core.fs import _fs_lock
from orze.engine import process, role_supervision as supervision, trigger_delivery
from orze.engine.supervised_process import prepare_supervised, SupervisionUnavailable, SupervisionUncertain
from orze.engine.supervisor_worker import canonical
from orze.idea_lake import IdeaLake


@pytest.fixture
def project(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    lock = tmp_path / ".orze" / "locks" / "engineer"
    assert _fs_lock(lock)
    nonce = hashlib.sha256(str(tmp_path).encode()).hexdigest()
    marker = tmp_path / "executed"
    command = [sys.executable, "-c", "from pathlib import Path; Path(" + repr(str(marker)) + ").write_text('done')"]
    metadata = {"role_name": "engineer", "attempt_id": "role-attempt-1",
                "scope": str(results), "lock_dir": str(lock),
                "nonce_sha256": hashlib.sha256(nonce.encode()).hexdigest(),
                "command_sha256": hashlib.sha256(canonical(command)).hexdigest(),
                "trigger_delivery": None, "trigger_delivery_db": None}
    p = SimpleNamespace(root=tmp_path, results=results, lock=lock, nonce=nonce,
                        marker=marker, command=command, metadata=metadata, owners=[])

    def begin():
        owner = supervision.begin_role_launch(metadata)
        p.owners.append(owner)
        return owner

    p.begin = begin
    p.env = {**os.environ, "ORZE_ROLE_PROCESS_NONCE": nonce, "CUDA_VISIBLE_DEVICES": ""}
    yield p
    # Only the fixture's actual returned handles are drained. No host scan or
    # bare-PID cleanup; production sticky owners are not used as closure proof.
    for owner in p.owners:
        child = owner.process
        if child is not None:
            try:
                if child.poll() is None:
                    child.stop(timeout=3)
            except Exception:
                child._supervisor.wait(timeout=3)
        owner.close_log()
        with supervision._GUARD:
            supervision._OWNERS.pop(id(owner), None)
            for key, (_, bound) in list(supervision._BOUND.items()):
                if bound is owner:
                    supervision._BOUND.pop(key)


def _prepare(p, owner, **kwargs):
    return owner.prepare(p.command, env=p.env, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, **kwargs)


def _role(p, owner):
    return process.RoleProcess(role_name="engineer", process=owner.process,
        start_time=time.time(), log_path=p.root / "role.log", timeout=30,
        lock_dir=p.lock, cycle_num=1, writes_ideas_file=False,
        process_nonce=p.nonce, trigger_launch=None, supervision_owner=owner)


def test_intent_precedes_actual_prepare_and_ready_keeps_worker_blocked(project, monkeypatch):
    p = project
    owner = p.begin()
    receipt = p.lock / "role-process.json"
    before = json.loads(receipt.read_bytes())
    assert before["stage"] == "INTENT"
    assert "trigger_delivery_db" not in receipt.read_text()
    assert p.nonce not in receipt.read_text()
    observed = []

    def prepare(*args, **kwargs):
        observed.append(json.loads(receipt.read_bytes())["stage"])
        return prepare_supervised(*args, **kwargs)

    child = _prepare(p, owner, prepare=prepare)
    assert observed == ["INTENT"]
    assert child.pid != child.supervisor_pid
    assert not p.marker.exists()
    assert json.loads(receipt.read_bytes())["stage"] == "READY"
    monkeypatch.setattr(process, "_process_has_nonce_hash", lambda *args: pytest.fail("pre-GO nonce scan"))
    rp = _role(p, owner)
    assert supervision.supervised_role_owner(rp) is owner
    owner.start()
    assert child.wait(timeout=3) == 0
    closure = owner.require_closed(0)
    assert closure["wait_proof"] == "ECHILD_WALL"
    assert closure["binding"]["worker"]["pid"] == child.pid
    assert closure["stop_requested"] is False
    assert p.marker.read_text() == "done"
    assert owner.release(outcome="ok", exit_code=0) is True
    assert not p.lock.exists()


def test_intent_write_failure_has_pending_owner_and_no_prepare(project, monkeypatch):
    p = project
    writes = []

    def fail_publish(path, raw):
        writes.append(path)
        raise OSError("controlled INTENT publication failure")

    monkeypatch.setattr(supervision, "_publish", fail_publish)
    monkeypatch.setattr(supervision, "prepare_supervised", lambda *a, **k: pytest.fail("prepare before INTENT"))
    with pytest.raises(supervision.RoleSupervisionHOLD) as caught:
        p.begin()
    owner = caught.value.owner
    p.owners.append(owner)
    assert owner is not None and owner.process is None
    assert writes == [p.lock / "role-process.json"]
    assert p.lock.exists()
    with pytest.raises(supervision.RoleSupervisionHOLD):
        owner.never_executed_proof()


def test_metadata_is_detached_and_command_change_never_prepares(project, monkeypatch):
    p = project
    owner = p.begin()
    p.metadata["role_name"] = "changed-after-intent"
    assert owner.role_name == "engineer"
    monkeypatch.setattr(supervision, "prepare_supervised", lambda *a, **k: pytest.fail("changed command prepared"))
    with pytest.raises(supervision.RoleSupervisionHOLD):
        owner.prepare([sys.executable, "-c", "pass"], env=p.env)
    assert owner.process is None
    assert p.lock.exists()


def test_explicit_unavailable_preserves_known_no_execution_release(project):
    p = project
    owner = p.begin()

    def unavailable(*args, **kwargs):
        raise SupervisionUnavailable("controlled unsupported platform")

    with pytest.raises(SupervisionUnavailable):
        _prepare(p, owner, prepare=unavailable)
    assert owner.abort() is None
    assert owner.never_executed_proof()["proof"] == "prepare_not_entered"
    assert owner.release() is True
    assert not p.marker.exists() and not p.lock.exists()


def test_ready_abort_can_prove_no_go_without_faking_no_fork(project):
    p = project
    owner = p.begin()
    child = _prepare(p, owner)
    closure = owner.abort(timeout=3)
    assert closure["stop_requested"] is True
    assert type(closure["worker_returncode"]) is int
    assert child.pid > 0
    assert owner.never_executed_proof()["proof"] == "no_go_tree_closed"
    assert not p.marker.exists()
    assert owner.release() is True


def test_prepare_uncertainty_retains_actual_pending_owner_without_second_stop(project):
    p = project
    owner = p.begin()

    def uncertain(*args, **kwargs):
        child = prepare_supervised(*args, **kwargs)
        raise SupervisionUncertain("controlled READY handoff loss", process=child)

    with pytest.raises(supervision.RoleSupervisionHOLD) as caught:
        _prepare(p, owner, prepare=uncertain)
    assert caught.value.owner is owner and owner.process is not None
    with pytest.raises(supervision.RoleSupervisionHOLD):
        owner.abort(timeout=0.1)
    with pytest.raises(supervision.RoleSupervisionHOLD):
        owner.never_executed_proof()
    assert owner.process._stop_sent is False
    assert p.lock.exists() and not p.marker.exists()


@pytest.mark.parametrize("field", ["supervision_owner", "process"])
def test_erased_public_role_binding_never_downgrades_to_legacy(project, field):
    p = project
    owner = p.begin()
    _prepare(p, owner)
    rp = _role(p, owner)
    setattr(rp, field, None)
    with pytest.raises(supervision.RoleSupervisionHOLD):
        supervision.supervised_role_owner(rp)
    assert p.lock.exists()


def test_bool_returncode_is_not_a_closure_receipt(project):
    p = project
    owner = p.begin()
    child = _prepare(p, owner)
    owner.start()
    assert child.wait(timeout=3) == 0
    with pytest.raises(supervision.RoleSupervisionHOLD):
        owner.require_closed(False)
    assert p.lock.exists()


def test_replaced_role_directory_is_never_released(project):
    p = project
    owner = p.begin()
    child = _prepare(p, owner)
    owner.start()
    assert child.wait(timeout=3) == 0
    owner.require_closed(0)
    retired = p.lock.with_name("retired-engineer")
    p.lock.rename(retired)
    p.lock.mkdir()
    foreign = p.lock / "foreign"
    foreign.write_text("must survive")
    with pytest.raises(supervision.RoleSupervisionHOLD):
        owner.release(outcome="ok", exit_code=0)
    assert foreign.read_text() == "must survive"
    assert (retired / "role-process.json").exists()


def test_unknown_v2_receipt_cannot_use_legacy_nonce_recovery(project, monkeypatch):
    p = project
    p.begin()
    monkeypatch.setattr(process, "_process_identities_with_nonce_hash",
                        lambda *args: pytest.fail("v2 entered legacy recovery scan"))
    report = process.reconcile_orphaned_role_receipts(p.root / ".orze")
    assert report["recovered"] == []
    assert report["errors"] == ["role_supervision_recovery_required:engineer"]
    with pytest.raises(supervision.RoleSupervisionHOLD):
        supervision.supervised_role_owner(SimpleNamespace(process=None, lock_dir=p.lock))
    assert p.lock.exists()


def test_trigger_requires_matching_terminal_before_exact_release(project):
    p = project
    lake = IdeaLake(p.root / "authority.db")
    try:
        source = p.root / "_trigger_engineer"
        source.write_text("PRIVATE trigger text")
        pending = trigger_delivery.observe_trigger(lake.db_path, "scope", "engineer", source)["pending"]
        lease = trigger_delivery.lease_trigger(lake.db_path, pending["delivery_id"],
            scope="scope", role_name="engineer", expected_sha256=pending["payload_sha256"], owner="controller")
        launch = trigger_delivery.begin_launch(lake.db_path, lease, attempt_id=p.metadata["attempt_id"],
            nonce_sha256=p.metadata["nonce_sha256"], command_sha256=p.metadata["command_sha256"])
        p.metadata["trigger_delivery"] = {key: launch[key] for key in supervision._REF}
        p.metadata["trigger_delivery_db"] = str(lake.db_path)
        owner = p.begin()
        child = _prepare(p, owner)
        assert trigger_delivery.record_started(lake.db_path, launch, process_pid=child.pid)
        owner.start()
        assert child.wait(timeout=3) == 0
        owner.require_closed(0)
        assert "PRIVATE trigger text" not in (p.lock / "role-process.json").read_text()
        assert trigger_delivery.record_terminal(lake.db_path, launch, outcome="ok", exit_code=0, cleanup_verified=True)
        assert owner.release(outcome="ok", exit_code=0) is True
        assert source.exists() and not p.lock.exists()
    finally:
        lake.close()
