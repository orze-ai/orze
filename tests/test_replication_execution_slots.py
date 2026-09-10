"""B3 new mechanisms through real native training and SQLite authority.

Only GPU/process boundaries use the existing test fixture. No real training,
provider, statistical replication benefit, or old missing-API bug is claimed.
"""
import copy
import errno
import json
import os
from pathlib import Path

import pytest

from orze.core.execution_attempts import current_attempt
from orze.core.replication_requests import ReplicationError
from orze.core.research_artifacts import artifacts_for_attempt
from orze.engine import execution_identity, launcher
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
from orze.engine.failure import _reset_idea_for_retry
from orze.engine.replication import request_replication
from orze.engine.scheduler import claim
from test_native_artifact_publication import _complete, _declare, _launch
from test_native_training_caller_boundaries import case


@pytest.fixture
def source(case, monkeypatch):
    c = case
    _declare(c)
    c.cfg.update(results_dir=str(c.results), idea_lake_db=str(c.lake.db_path))
    (c.folder / "idea_config.yaml").write_text(c.lake.get(c.idea)["config"])
    tp = _launch(c)
    (c.folder / "model.bin").write_bytes(b"same-seed-output")
    events = _complete(c, tp)
    assert len(events) == 1
    c.source_id, c.source_tp = c.idea, tp
    c.source_config = c.lake.get(c.idea)["config"]
    c.source_records = artifacts_for_attempt(c.lake.conn, tp.attempt_ref)
    c.flat = execution_identity._registry_root(c.results, c.cfg) / (tp.execution_identity + ".json")
    c.flat_bytes = c.flat.read_bytes()
    c.popen_calls.clear()

    def reap(proc, idea_id, **kwargs):
        assert proc is c.child
        c.stops.append(idea_id)
        proc.returncode = -15
        return True
    monkeypatch.setattr(launcher, "_terminate_and_reap", reap)
    return c


def _request(c, request_id="request-1"):
    return request_replication(c.source_id, c.results, c.cfg, c.lake,
                               request_id=request_id)


def _select(c, request):
    c.idea = request["task_id"]
    c.folder = c.results / c.idea
    assert claim(c.idea, c.results, 0, lake=c.lake)
    (c.folder / "idea_config.yaml").write_text(c.lake.get(c.idea)["config"])
    c.child.returncode = None
    return c.folder


def _slot(c, request):
    return c.flat.parent / (c.source_tp.execution_identity + ".replicas") / (request["request_id"] + ".json")


def test_two_explicit_requests_keep_semantics_and_original_owner_but_publish_new_occurrences(source):
    c = source
    requests = [_request(c, "repeat.1"), _request(c, "repeat:2")]
    seen = []
    records = []
    for request in requests:
        _select(c, request)
        assert c.lake.get(c.idea)["config"] == c.source_config
        c.before_popen = lambda: seen.append(copy.deepcopy(
            current_attempt(c.lake.conn, c.idea, "training")["binding"]["replication"]))
        tp = _launch(c)
        assert tp.execution_identity == c.source_tp.execution_identity
        row = current_attempt(c.lake.conn, c.idea, "training")
        assert row["binding"]["replication"]["request_id"] == request["request_id"]
        assert seen[-1] == row["binding"]["replication"]
        assert row["binding"]["artifact_publication"]["spec_fingerprint"] == c.source_records[0]["spec_fingerprint"]
        assert _slot(c, request).is_file()
        (c.folder / "model.bin").write_bytes(b"same-seed-output")
        assert len(_complete(c, tp)) == 1
        records += artifacts_for_attempt(c.lake.conn, tp.attempt_ref)
    assert len({r["artifact_id"] for r in records + c.source_records}) == 3
    assert len({r["content_sha256"] for r in records + c.source_records}) == 1
    assert c.flat.read_bytes() == c.flat_bytes
    assert c.popen_calls == [True, True]


def test_model_authored_replica_labels_still_use_ordinary_dedup(source):
    c = source
    c.idea = "idea-unrequested-repeat"
    c.lake.insert(c.idea, "Unrequested", c.source_config + "\nreplication_role: rerun\nreplication_index: 5\n", "", status="queued")
    _select(c, {"task_id": c.idea})
    with pytest.raises(launcher.DuplicateLaunchError, match="already_completed"):
        _launch(c)
    assert c.popen_calls == [] and c.flat.read_bytes() == c.flat_bytes
    assert not list(c.flat.parent.glob("*.replicas"))


@pytest.mark.parametrize("change", ["script", "config", "claim"])
def test_captured_request_does_not_authorize_drifted_execution_or_claim(source, change):
    c = source
    request = _request(c)
    _select(c, request)
    if change == "script":
        Path(c.cfg["train_script"]).write_text("# changed execution\n")
    elif change == "config":
        (c.folder / "idea_config.yaml").write_text("seed: 999\n")
    else:
        p = c.folder / "claim.json"
        value = json.loads(p.read_text())
        value["lifecycle_db"] = str(c.results / "wrong.db")
        p.write_text(json.dumps(value))
    with pytest.raises((ReplicationError, AttemptEffectBusy)):
        _launch(c)
    assert c.popen_calls == [] and current_attempt(c.lake.conn, c.idea, "training") is None
    assert not _slot(c, request).exists()
    assert c.flat.read_bytes() == c.flat_bytes


def test_native_begin_rechecks_actual_request_after_slot_reservation(source, monkeypatch):
    import orze.core.model_lineage as lineage
    c = source
    request = _request(c)
    _select(c, request)
    original = lineage.prepare_model_lineage_launch
    def revoke(*args, **kwargs):
        c.lake.conn.execute("DELETE FROM replication_requests WHERE request_id=?", (request["request_id"],))
        c.lake.conn.commit()
        return original(*args, **kwargs)
    monkeypatch.setattr(lineage, "prepare_model_lineage_launch", revoke)
    with pytest.raises(AttemptEffectBusy, match="authorization_changed"):
        _launch(c)
    assert c.popen_calls == [] and current_attempt(c.lake.conn, c.idea, "training") is None
    assert not _slot(c, request).exists()
    assert c.flat.read_bytes() == c.flat_bytes


def test_partial_slot_write_is_held_without_launch_or_blind_replacement(source, monkeypatch):
    c = source
    request = _request(c)
    _select(c, request)
    path = _slot(c, request)
    original = os.write
    calls = []
    def partial_then_fail(fd, payload):
        if os.readlink(f"/proc/self/fd/{fd}") == str(path):
            calls.append(True)
            if len(calls) == 1:
                return original(fd, payload[:7])
            raise OSError(errno.EIO, "synthetic partial slot failure")
        return original(fd, payload)
    monkeypatch.setattr(execution_identity.os, "write", partial_then_fail)
    with pytest.raises(AttemptEffectInDoubt):
        _launch(c)
    partial = path.read_bytes()
    assert len(partial) == 7 and path.with_suffix(".lock").is_dir()
    monkeypatch.setattr(execution_identity.os, "write", original)
    with pytest.raises(AttemptEffectBusy):
        _launch(c)
    assert path.read_bytes() == partial and calls == [True, True]
    assert c.popen_calls == [] and current_attempt(c.lake.conn, c.idea, "training") is None
    assert c.flat.read_bytes() == c.flat_bytes


def test_known_no_popen_cleanup_uses_captured_slot_not_mutated_cfg(source, monkeypatch):
    c = source
    request = _request(c)
    _select(c, request)
    path = _slot(c, request)
    other = c.results.parent / "other-control"
    foreign = other / "state" / "execution_identities" / path.parent.name / path.name
    foreign.parent.mkdir(parents=True)
    foreign.write_bytes(b"other occurrence must remain")
    def no_gpu(*args, **kwargs):
        c.cfg["_orze_dir"] = str(other)
        raise launcher.GpuUnavailableError("synthetic unavailable GPU")
    monkeypatch.setattr(launcher, "_verify_gpu_free", no_gpu)
    with pytest.raises(launcher.GpuUnavailableError):
        _launch(c)
    assert not path.exists() and foreign.read_bytes() == b"other occurrence must remain"
    assert current_attempt(c.lake.conn, c.idea, "training")["state"] == "NOT_STARTED"
    assert c.popen_calls == [] and c.flat.read_bytes() == c.flat_bytes


def test_current_running_occurrence_cannot_be_replayed(source):
    c = source
    request = _request(c)
    _select(c, request)
    tp = _launch(c)
    try:
        before = _slot(c, request).read_bytes()
        with pytest.raises((ReplicationError, AttemptEffectBusy)):
            _launch(c)
        assert c.popen_calls == [True] and _slot(c, request).read_bytes() == before
        assert current_attempt(c.lake.conn, c.idea, "training")["attempt_id"] == tp.attempt_id
    finally:
        tp.close_log()


def test_confirmed_failed_task_retry_retains_request_and_replaces_only_its_closed_slot(source):
    c = source
    request = _request(c)
    _select(c, request)
    first = _launch(c)
    assert len(_complete(c, first, ret=1)) == 1
    assert current_attempt(c.lake.conn, c.idea, "training")["state"] == "TERMINAL"
    _reset_idea_for_retry(c.folder, release_claim=True, lake=c.lake)
    assert c.lake.record_state_transition(c.idea, "FAILED", "QUEUED", reason="explicit bounded test retry")
    _select(c, request)
    second = _launch(c)
    try:
        assert second.attempt_id != first.attempt_id
        assert second.execution_identity == first.execution_identity
        assert json.loads(_slot(c, request).read_text())["attempt_id"] == second.attempt_id
        assert current_attempt(c.lake.conn, c.idea, "training")["binding"]["replication"]["request_id"] == request["request_id"]
        assert c.flat.read_bytes() == c.flat_bytes
    finally:
        second.close_log()


def test_unclaimed_replica_cannot_drop_lake_and_contract_to_downgrade(source):
    c = source
    request = _request(c)
    c.idea = request["task_id"]
    c.folder = c.results / c.idea
    c.folder.mkdir()
    (c.folder / "idea_config.yaml").write_text(c.source_config)
    cfg = copy.deepcopy(c.cfg)
    cfg.pop("artifact_contract")
    before = Path(c.lake.db_path).read_bytes()
    with pytest.raises(AttemptEffectBusy, match="replication_catalog_required"):
        launcher.launch(c.idea, 0, c.results, cfg, lake=None)
    assert c.popen_calls == [] and Path(c.lake.db_path).read_bytes() == before
    assert current_attempt(c.lake.conn, c.idea, "training") is None
    assert not (c.folder / "_execution_catalog.json").exists()
