"""Independent opt-in native training artifact contract acceptance.

These are new mechanisms, not historical missing-API regressions. Actual
launch, terminal publication, SQLite and copies are exercised; process/GPU
boundaries come from the existing offline native-training fixture. Different
seed tasks deliberately avoid claiming a same-config replication capability.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path

import pytest

from orze.core.evaluation_retry_state import open_existing_lake
from orze.core.execution_attempts import current_attempt
from orze.core.research_artifacts import artifacts_for_attempt, get_artifact
from orze.engine import artifact_publication, launcher, training_completion
from orze.engine.scheduler import claim
from orze.engine.termination_hold import TerminationUnconfirmed
from test_native_training_caller_boundaries import case as native_case, Child


BODY = b"immutable snapshot content\x00\xff"


@pytest.fixture
def project(native_case, monkeypatch):
    c = native_case
    c.cfg.update({
        "_project_root": str(c.results.parent), "results_dir": str(c.results),
        "idea_lake_db": str(c.lake.db_path), "_orze_dir": str(c.results.parent / ".orze"),
        "artifact_contract": {"version": 1, "outputs": {
            "checkpoint": {"path": "best_model.pt", "max_bytes": 4096},
            "note": {"path": "summary.txt", "max_bytes": 4096},
        }},
    })
    c.handles = []

    def popen(*args, **kwargs):
        child = Child(pid=741300 + len(c.handles))
        child.wait = lambda timeout=None: child.returncode
        c.popen_calls.append(True)
        c.child = child
        return child

    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    try:
        yield c
    finally:
        for handle in c.handles:
            handle.close_log()


def _launch(c, *, idea_id=None, seed=13):
    idea_id = idea_id or c.idea
    folder = c.results / idea_id
    if idea_id != c.idea:
        c.lake.insert(idea_id, "Different seeded task", f"seed: {seed}", "", status="queued")
        assert claim(idea_id, c.results, 0, lake=c.lake)
    (folder / "idea_config.yaml").write_text(f"seed: {seed}\n", encoding="utf-8")
    tp = launcher.launch(idea_id, 0, c.results, c.cfg, lake=c.lake)
    c.handles.append(tp)
    assert current_attempt(c.lake.conn, idea_id, "training")["state"] == "RUNNING"
    return tp, folder


def _output(tp, folder):
    (folder / "metrics.json").write_text('{"status":"COMPLETED","score":0}', encoding="utf-8")
    (folder / "best_model.pt").write_bytes(BODY)
    (folder / "summary.txt").write_text("same summary\n", encoding="utf-8")
    (folder / "not-declared.bin").write_bytes(b"do not infer this as an output")
    tp.process.returncode = 0


def _poll(c, tp):
    active, failures = {0: tp}, {}
    try:
        events = launcher.check_active(active, c.results, c.cfg, failures, lake=c.lake)
    except TerminationUnconfirmed:
        events = []
    return events, active, failures


def _accepted(c, tp):
    row = current_attempt(c.lake.conn, tp.idea_id, "training")
    assert row["state"] == "TERMINAL" and row["terminal"]["outcome"] == "completed"
    records = artifacts_for_attempt(c.lake.conn, tp.attempt_ref)
    assert {record["logical_name"] for record in records} == {"checkpoint", "note"}
    assert set(row["terminal"]["artifact_ids"]) == {record["artifact_id"] for record in records}
    binding = row["binding"]["artifact_publication"]
    for record in records:
        assert record["producer"] == asdict(tp.attempt_ref)
        assert record["scope"] == str(c.results)
        assert record["spec_fingerprint"] == binding["spec_fingerprint"]
        assert Path(record["path"]) == Path(binding["root"]) / record["artifact_id"] / "content"
        assert get_artifact(c.lake.conn, record["artifact_id"]) == record
    return records


def test_open_worker_descriptor_cannot_change_accepted_independent_inode(project):
    c = project
    tp, folder = _launch(c)
    _output(tp, folder)
    source = folder / "best_model.pt"
    worker_fd = os.open(source, os.O_RDWR)
    try:
        source_stat = os.fstat(worker_fd)
        events, active, failures = _poll(c, tp)
        assert events == [(tp.idea_id, 0)] and not active and not failures
        records = _accepted(c, tp)
        checkpoint = next(record for record in records if record["logical_name"] == "checkpoint")
        artifact = Path(checkpoint["path"])
        frozen = artifact.read_bytes()
        assert frozen == BODY
        assert (artifact.stat().st_dev, artifact.stat().st_ino) != (source_stat.st_dev, source_stat.st_ino)
        assert artifact.stat().st_nlink == 1 and artifact.stat().st_mode & 0o222 == 0
        assert checkpoint["content_sha256"] == hashlib.sha256(frozen).hexdigest()
        assert checkpoint["size_bytes"] == len(frozen)
        os.lseek(worker_fd, 0, os.SEEK_SET)
        os.write(worker_fd, b"LATE WORKER MODIFICATION")
        os.ftruncate(worker_fd, len(b"LATE WORKER MODIFICATION"))
        os.fsync(worker_fd)
        assert source.read_bytes() != frozen
        assert artifact.read_bytes() == frozen
        assert artifacts_for_attempt(c.lake.conn, tp.attempt_ref) == records
    finally:
        os.close(worker_fd)


def test_same_bytes_from_distinct_seeded_native_tasks_have_distinct_occurrence_ids(project):
    c = project
    first, folder = _launch(c)
    _output(first, folder)
    assert _poll(c, first)[0] == [(first.idea_id, 0)]
    first_records = _accepted(c, first)
    second, folder = _launch(c, idea_id="idea-other-seed", seed=17)
    _output(second, folder)
    assert _poll(c, second)[0] == [(second.idea_id, 0)]
    second_records = _accepted(c, second)
    a = {record["logical_name"]: record for record in first_records}
    b = {record["logical_name"]: record for record in second_records}
    for name in a:
        assert a[name]["content_sha256"] == b[name]["content_sha256"]
        assert a[name]["artifact_id"] != b[name]["artifact_id"]
        assert a[name]["artifact_id"] != a[name]["content_sha256"]
        assert a[name]["path"] != b[name]["path"]
    assert artifacts_for_attempt(c.lake.conn, first.attempt_ref) == first_records


def test_peer_terminal_during_unlocked_copy_prevents_second_artifact_acceptance(project, monkeypatch):
    c = project
    tp, folder = _launch(c)
    _output(tp, folder)
    peer = open_existing_lake(Path(c.lake.db_path))
    original = artifact_publication.prepare_artifacts
    raced, accepted = [], []
    captured = []

    def prepare_then_peer(ref, idea_dir, binding):
        # Reentrant same-process callback gets the already captured immutable
        # preparation, not permission to recover disk staging or copy twice.
        if captured:
            return captured[0]
        prepared = original(ref, idea_dir, binding)
        captured.append(prepared)
        if not raced:
            raced.append(True)
            assert not c.lake.conn.in_transaction and not peer.conn.in_transaction
            assert not (folder / "_attempt_effect.lock").exists()
            event = training_completion.finish(peer, tp, 0, folder, c.cfg, 0, {})
            assert event is not None and event.attempt_ref == tp.attempt_ref
            accepted.append(event)
        return prepared

    monkeypatch.setattr(artifact_publication, "prepare_artifacts", prepare_then_peer)
    try:
        events, _, failures = _poll(c, tp)
        assert raced == [True] and len(accepted) == 1
        assert events == [] and failures == {}
        records = _accepted(c, tp)
        assert c.lake.conn.execute("SELECT COUNT(*) FROM research_artifacts").fetchone()[0] == len(records) == 2
        assert c.lake.conn.execute(
            "SELECT COUNT(*) FROM idea_transitions WHERE idea_id=? AND to_state='COMPLETE'",
            (tp.idea_id,),
        ).fetchone()[0] == 1
    finally:
        peer.close()


def test_actual_copy_and_hash_do_not_hold_the_effect_guard_or_sql_transaction(project, monkeypatch):
    c = project
    tp, folder = _launch(c)
    _output(tp, folder)
    sources = {folder / "best_model.pt", folder / "summary.txt"}
    original_read = os.read
    reads = []

    def bounded_read(fd, count):
        target = Path(os.readlink(f"/proc/self/fd/{fd}"))
        if target in sources or (target.name == "content" and "artifacts" in target.parts):
            reads.append(target)
            assert not c.lake.conn.in_transaction, "artifact bytes were read inside the writer transaction"
            assert not (folder / "_attempt_effect.lock").exists(), "large artifact IO holds the effect guard"
        return original_read(fd, count)

    monkeypatch.setattr(os, "read", bounded_read)
    assert _poll(c, tp)[0] == [(tp.idea_id, 0)]
    assert sources.issubset(set(reads))
    _accepted(c, tp)


@pytest.mark.parametrize("mutation", ["metadata", "replacement_inode"])
def test_prepared_snapshot_changes_cannot_register_a_terminal(project, monkeypatch, mutation):
    c = project
    tp, folder = _launch(c)
    _output(tp, folder)
    original = artifact_publication.prepare_artifacts
    changed = []

    def changed_after_copy(ref, idea_dir, binding):
        prepared = original(ref, idea_dir, binding)
        records = json.loads(prepared.records_json)
        if mutation == "metadata":
            records[0]["content_sha256"] = "0" * 64
            prepared = replace(prepared, records_json=json.dumps(
                records, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        else:
            destination = Path(records[0]["path"])
            replacement = destination.with_name("replacement")
            replacement.write_bytes(destination.read_bytes())
            replacement.chmod(0o400)
            os.replace(replacement, destination)
        changed.append(True)
        return prepared

    monkeypatch.setattr(artifact_publication, "prepare_artifacts", changed_after_copy)
    events, active, failures = _poll(c, tp)
    assert changed == [True]
    assert events == [] and active.get(0) is tp and failures == {}
    assert current_attempt(c.lake.conn, tp.idea_id, "training")["state"] == "RUNNING"
    assert c.lake.get_fsm_state(tp.idea_id) == "IN_PROGRESS"
    assert artifacts_for_attempt(c.lake.conn, tp.attempt_ref) == []


def test_terminal_sql_rejection_rolls_back_all_artifact_rows_and_retains_hold(project):
    c = project
    tp, folder = _launch(c)
    _output(tp, folder)
    c.lake.conn.execute(
        "CREATE TRIGGER ignore_training_terminal BEFORE UPDATE ON execution_attempts "
        "WHEN NEW.state='TERMINAL' BEGIN SELECT RAISE(IGNORE); END")
    c.lake.conn.commit()
    events, active, failures = _poll(c, tp)
    assert events == [] and active.get(0) is tp and failures == {}
    assert current_attempt(c.lake.conn, tp.idea_id, "training")["state"] == "RUNNING"
    assert c.lake.get_fsm_state(tp.idea_id) == "IN_PROGRESS"
    assert artifacts_for_attempt(c.lake.conn, tp.attempt_ref) == []
    assert (folder / "_attempt_effect.lock").is_dir()
    assert (folder / "_execution_effects" / tp.attempt_id / "prepared.json").exists()
    assert not (folder / "_execution_effects" / tp.attempt_id / "committed.json").exists()


def test_disabled_native_path_does_not_infer_artifacts_or_create_the_registry(project):
    c = project
    c.cfg["artifact_contract"] = None
    tp, folder = _launch(c)
    _output(tp, folder)
    before_schema = list(c.lake.conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"))
    assert _poll(c, tp)[0] == [(tp.idea_id, 0)]
    row = current_attempt(c.lake.conn, tp.idea_id, "training")
    assert row["state"] == "TERMINAL"
    assert row["binding"].get("artifact_publication") is None
    assert "artifact_ids" not in row["terminal"]
    assert artifacts_for_attempt(c.lake.conn, tp.attempt_ref) == []
    assert list(c.lake.conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")) == before_schema
    assert not (Path(c.cfg["_orze_dir"]) / "artifacts").exists()
    assert (folder / "best_model.pt").read_bytes() == BODY
