"""New V105B1 mechanism: actual native launch/completion and real SQLite.

The existing fixture replaces only child/process/GPU ownership boundaries;
this is NOT a real CPU executor or a GPU training/research-benefit claim.
No observation or evaluation-retry semantics are introduced by these tests.
"""
import copy
import errno
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from orze.core.execution_attempts import current_attempt
from orze.core.research_artifacts import artifacts_for_attempt
from orze.engine import artifact_publication, launcher
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
from test_native_training_caller_boundaries import case


def _declare(c, outputs=None):
    c.cfg["artifact_contract"] = {"version": 1, "outputs": (
        {"weights": {"path": "model.bin", "max_bytes": 128}}
        if outputs is None else outputs)}
    c.cfg["_project_root"] = str(c.results.parent)
    c.cfg["_orze_dir"] = str(c.results.parent / "control")


def _launch(c):
    return launcher.launch(c.idea, 0, c.results, c.cfg, lake=c.lake)


def _complete(c, tp, ret=0):
    c.child.returncode = ret
    (c.folder / "metrics.json").write_text(json.dumps({
        "status": "COMPLETED" if ret == 0 else "FAILED", "score": 0}))
    c.active = {0: tp}
    return launcher.check_active(c.active, c.results, c.cfg, {}, lake=c.lake)


def _assert_unaccepted(c, tp):
    assert artifacts_for_attempt(c.lake.conn, tp.attempt_ref) == []
    assert current_attempt(c.lake.conn, c.idea, "training")["state"] == "RUNNING"
    assert c.lake.get_fsm_state(c.idea) == "IN_PROGRESS"
    assert c.active[0] is tp
    assert not (c.folder / "_compute_receipts" / tp.attempt_id / "terminal.json").exists()


def test_pre_popen_binding_survives_started_and_new_inode_survives_old_worker_fd(case):
    c = case
    _declare(c)
    observed = []
    c.before_popen = lambda: observed.append(copy.deepcopy(
        current_attempt(c.lake.conn, c.idea, "training")["binding"]["artifact_publication"]))
    tp = _launch(c)
    source = c.folder / "model.bin"
    original = b"synthetic-weights\x00\xff"
    source.write_bytes(original)
    with source.open("r+b") as late_worker:
        assert _complete(c, tp) == [(c.idea, 0)]
        row = current_attempt(c.lake.conn, c.idea, "training")
        assert observed == [row["binding"]["artifact_publication"]]
        records = artifacts_for_attempt(c.lake.conn, tp.attempt_ref)
        assert len(records) == 1
        record = records[0]
        snapshot = Path(record["path"])
        assert row["terminal"]["artifact_ids"] == [record["artifact_id"]]
        assert snapshot == c.results.parent / "control" / "artifacts" / record["artifact_id"] / "content"
        assert snapshot.stat().st_ino != source.stat().st_ino
        assert snapshot.stat().st_nlink == 1
        assert record["content_sha256"] == hashlib.sha256(original).hexdigest()
        late_worker.seek(0)
        late_worker.write(b"X" * len(original))
        late_worker.flush()
        assert snapshot.read_bytes() == original
    assert c.active == {}


@pytest.mark.parametrize("fault", ["missing", "oversize", "symlink", "source_changed"])
def test_bad_declared_source_never_registers_or_completes(case, monkeypatch, fault):
    c = case
    _declare(c)
    tp = _launch(c)
    source = c.folder / "model.bin"
    if fault == "oversize":
        source.write_bytes(b"X" * 129)
    elif fault == "symlink":
        outside = c.results.parent / "outside.bin"
        outside.write_bytes(b"untouched")
        source.symlink_to(outside)
    elif fault == "source_changed":
        source.write_bytes(b"original")
        inode = source.stat().st_ino
        read = artifact_publication.os.read
        seen = []

        def changed(fd, maximum):
            value = read(fd, maximum)
            if os.fstat(fd).st_ino == inode and not seen:
                seen.append(True)
                assert not c.lake.conn.in_transaction
                assert not (c.folder / "_attempt_effect.lock").exists()
                source.write_bytes(b"replaced")
            return value

        monkeypatch.setattr(artifact_publication.os, "read", changed)
    with pytest.raises(AttemptEffectBusy):
        _complete(c, tp)
    _assert_unaccepted(c, tp)
    if fault == "source_changed":
        assert seen == [True]
    if fault == "symlink":
        assert outside.read_bytes() == b"untouched"


@pytest.mark.parametrize("change", ["path", "root", "disable", "previously_unbound"])
def test_current_cfg_cannot_gain_or_redirect_launch_bound_contract(case, change):
    c = case
    if change != "previously_unbound":
        _declare(c)
    tp = _launch(c)
    (c.folder / "model.bin").write_bytes(b"original")
    if change == "path":
        c.cfg["artifact_contract"]["outputs"]["weights"]["path"] = "other.bin"
    elif change == "root":
        c.cfg["_orze_dir"] = str(c.results.parent / "redirected")
    elif change == "disable":
        c.cfg.pop("artifact_contract")
    else:
        _declare(c)
    with pytest.raises(AttemptEffectBusy, match="contract_unbound_or_changed"):
        _complete(c, tp)
    _assert_unaccepted(c, tp)
    assert not (c.results.parent / "control" / "artifacts").exists()
    assert not (c.results.parent / "redirected").exists()


def test_partial_snapshot_write_failure_preserves_source_and_no_accepted_records(case, monkeypatch):
    c = case
    _declare(c)
    tp = _launch(c)
    source = c.folder / "model.bin"
    source.write_bytes(b"source remains intact")
    write = artifact_publication.os.write
    calls = []

    def fail_write(fd, payload):
        target = os.readlink(f"/proc/self/fd/{fd}")
        if target.endswith("/content"):
            calls.append(True)
            if len(calls) == 1:
                return write(fd, payload[:2])
            raise OSError(errno.ENOSPC, "synthetic snapshot exhaustion")
        return write(fd, payload)

    monkeypatch.setattr(artifact_publication.os, "write", fail_write)
    with pytest.raises(AttemptEffectBusy):
        _complete(c, tp)
    _assert_unaccepted(c, tp)
    assert calls == [True, True]
    assert source.read_bytes() == b"source remains intact"
    staged = list((c.results.parent / "control" / "artifacts").glob("*/content"))
    assert len(staged) == 1 and staged[0].read_bytes() == b"so"
    with pytest.raises(AttemptEffectBusy, match="staging_requires_resolution"):
        _complete(c, tp)
    assert calls == [True, True], "a later tick must not repeat the expensive copy"
    assert list((c.results.parent / "control" / "artifacts").glob("*/content")) == staged


def test_sql_terminal_rejection_rolls_back_all_artifact_rows_and_retains_hold(case):
    c = case
    _declare(c)
    tp = _launch(c)
    (c.folder / "model.bin").write_bytes(b"unaccepted staged output")
    c.lake.conn.execute("CREATE TRIGGER reject_artifact_terminal BEFORE UPDATE ON execution_attempts "
                        "WHEN NEW.state='TERMINAL' BEGIN SELECT RAISE(IGNORE); END")
    c.lake.conn.commit()
    with pytest.raises(AttemptEffectInDoubt):
        _complete(c, tp)
    assert artifacts_for_attempt(c.lake.conn, tp.attempt_ref) == []
    assert current_attempt(c.lake.conn, c.idea, "training")["state"] == "RUNNING"
    assert c.lake.get_fsm_state(c.idea) == "IN_PROGRESS"
    assert c.active[0] is tp
    assert (c.folder / "_attempt_effect.lock").is_dir()
    assert list((c.results.parent / "control" / "artifacts").glob("*/content"))


@pytest.mark.parametrize("mode", ["disabled", "empty", "failed"])
def test_legacy_off_and_explicit_zero_artifact_outcomes_remain_distinct(case, mode):
    c = case
    if mode != "disabled":
        _declare(c, {} if mode == "empty" else None)
    tp = _launch(c)
    assert _complete(c, tp, ret=1 if mode == "failed" else 0) == [(c.idea, 0)]
    row = current_attempt(c.lake.conn, c.idea, "training")
    assert artifacts_for_attempt(c.lake.conn, tp.attempt_ref) == []
    if mode == "disabled":
        assert "artifact_ids" not in row["terminal"]
    else:
        assert row["terminal"]["artifact_ids"] == []
    assert not (c.results.parent / "control" / "artifacts").exists()


def test_declared_artifacts_cannot_launch_without_native_catalog(case):
    c = case
    _declare(c)
    with pytest.raises(launcher.LaunchIntegrityError, match="native_catalog_required"):
        launcher.launch(c.idea, 0, c.results, c.cfg)
    assert c.popen_calls == []


def test_pre_native_import_cannot_gain_artifact_provenance_after_execution(case):
    from orze.engine.accounting import record_compute_start
    from orze.engine.training_completion import finish
    c = case
    assert c.lake.record_state_transition(c.idea, "CLAIMED", "IN_PROGRESS")
    stored = json.loads((c.folder / "claim.json").read_text())
    tp = SimpleNamespace(idea_id=c.idea, gpu=0, process=c.child, start_time=time.time(),
                         attempt_id=stored["attempt_id"], attempt_ref=None)
    record_compute_start(tp, c.folder)
    (c.folder / "metrics.json").write_text('{"status":"COMPLETED"}')
    (c.folder / "model.bin").write_bytes(b"historical unbound output")
    _declare(c)
    with pytest.raises(AttemptEffectBusy, match="contract_unbound_or_changed"):
        finish(c.lake, tp, 0, c.folder, c.cfg, 0, {})
    assert current_attempt(c.lake.conn, c.idea, "training") is None
    assert c.lake.get_fsm_state(c.idea) == "IN_PROGRESS"
    assert not (c.results.parent / "control" / "artifacts").exists()
