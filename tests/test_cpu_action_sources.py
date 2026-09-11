"""New multi-source mechanisms using actual native CPU publications.

Real Lake/claim/budget/worker/supervisor/effect and temp files. Faults target
the indicated IO/SQL boundary; no GPU/provider, adopted source, or fake proof.
"""
import copy
import fcntl
import json
import os
from pathlib import Path
import sys

import pytest
import yaml

from orze.core import cpu_action_budget as budget
from orze.core.execution_attempts import create_attempt
from orze.core.research_artifacts import artifacts_for_attempt
from orze.engine import cpu_action_sources as sources
from orze.engine import native_cpu_action as native
from orze.engine.scheduler import claim
from orze.engine.supervised_process import prepare_supervised
from orze.idea_lake import IdeaLake
from test_native_cpu_action import _finish


@pytest.fixture
def published(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(tmp_path / "lake.db")
    scope = budget.initialize(lake, results, {
        "version": 1, "resource": "cpu", "slots": 2, "wall_budget_seconds": 20})
    cfg = {"_project_root": str(tmp_path), "_orze_dir": str(tmp_path / ".orze")}
    handles, records = [], []
    try:
        for index, data in enumerate(("one", "two")):
            name = "source-" + str(index)
            action = {"version": 1, "adapter": "command", "purpose": "publish CPU source " + data,
                "inputs": {}, "command": [sys.executable, "-c",
                    "from pathlib import Path; Path('result').write_bytes(" + repr(data.encode()) + ")"],
                "timeout_seconds": 2, "outputs": {"result": {"path": "result", "max_bytes": 1024}}}
            assert lake.insert(name, name, yaml.safe_dump({"kind": "native_cpu_action", "action": action}),
                "", status="queued", kind="native_cpu_action", if_absent=True)["status"] == "inserted"
            permit = budget.reserve(lake, scope, name, 2)
            assert claim(name, results, None, lake, resource="cpu")
            handle = native.launch(name, results, cfg, lake=lake, action=action,
                                   permit=permit, admission=lambda: None)
            handles.append(handle)
            assert _finish(handle, results, cfg, lake, permit)["outcome"] == "completed"
            records.extend(artifacts_for_attempt(lake.conn, handle.attempt_ref))
        yield lake, results, records, handles
    finally:
        for handle in handles:
            if handle.process.poll() is None:
                handle.process.stop(timeout=0.2)
            assert type(handle.process.poll()) is int
        lake.close()


def capture(published):
    lake, results, records, _ = published
    return sources.capture_sources(lake, results, [record["artifact_id"] for record in records])


def test_distinct_spec_sources_are_detached_and_sealed_to_actual_worker(published, tmp_path):
    lake, results, records, _ = published
    prepared = capture(published)
    assert records[0]["spec_fingerprint"] != records[1]["spec_fingerprint"]
    assert sources.records(prepared) == records
    changed = sources.snapshot(prepared)
    changed["inputs"][0]["artifact"]["content_sha256"] = "0" * 64
    assert sources.snapshot(prepared)["inputs"][0]["artifact"] == records[0]
    readback = tmp_path / "readback.json"
    code = """import os,json,fcntl
from pathlib import Path
fds=json.loads(os.environ['ORZE_ACTION_SOURCE_FDS'])
out={}
for key,fd in fds.items():
 assert fcntl.fcntl(fd,fcntl.F_GETFL)&os.O_ACCMODE==os.O_RDONLY
 assert fcntl.fcntl(fd,fcntl.F_GET_SEALS)&15==15
 out[key]=os.read(fd,1024).decode()
 try: os.write(fd,b'bad')
 except OSError: pass
 else: raise AssertionError('writable source')
Path(os.environ['READBACK']).write_text(json.dumps(out))
"""
    process = None
    borrowed = ()
    try:
        with sources.sealed_sources(prepared) as (environment, borrowed):
            process = prepare_supervised([sys.executable, "-c", code],
                identity={"test": "source-fds"}, cwd=str(tmp_path),
                env={**os.environ, **environment, "READBACK": str(readback)}, worker_only_fds=borrowed)
            process.start()
            assert process.wait(timeout=3) == 0
        assert json.loads(readback.read_text()) == {records[0]["artifact_id"]: "one", records[1]["artifact_id"]: "two"}
        for fd in borrowed:
            with pytest.raises(OSError):
                os.fstat(fd)
    finally:
        if process is not None and process.poll() is None:
            process.stop(timeout=0.2)
    sources.require_sources(lake, results, prepared)


def test_empty_sources_and_empty_file_transport_have_explicit_bounds(published):
    lake, results, _, _ = published
    prepared = sources.capture_sources(lake, results, [])
    assert sources.records(prepared) == []
    with sources.sealed_sources(prepared) as (env, fds):
        assert env == {"ORZE_ACTION_SOURCE_FDS": "{}"} and fds == ()
    with sources._sealed(b"") as fd:
        assert os.read(fd, 1) == b"" and os.fstat(fd).st_size == 0


def test_writer_verification_never_reads_large_content_and_requires_exact_owner(published, monkeypatch):
    lake, results, _, _ = published
    prepared = capture(published)
    def forbidden(*args, **kwargs):
        pytest.fail("metadata-only watch read source content")
    monkeypatch.setattr(sources, "_content", forbidden)
    lake.conn.execute("BEGIN IMMEDIATE")
    try:
        sources.require_sources(lake, results, prepared, metadata_only=True)
        with pytest.raises(sources.SourceHOLD, match="inside_transaction"):
            sources.require_sources(lake, results, prepared)
        with pytest.raises(sources.SourceHOLD):
            sources.capture_sources(lake, results, [])
    finally:
        lake.conn.rollback()
    with pytest.raises(sources.SourceHOLD):
        sources.require_sources(lake, results, copy.copy(prepared), metadata_only=True)
    with pytest.raises(sources.SourceHOLD):
        sources.require_sources(lake, results, sources.snapshot(prepared), metadata_only=True)
    other = IdeaLake(lake.db_path)
    try:
        with pytest.raises(sources.SourceHOLD):
            sources.require_sources(other, results, prepared, metadata_only=True)
    finally:
        other.close()


@pytest.mark.parametrize("fault", ["spec", "membership", "current"])
def test_actual_sql_source_changes_are_rejected_even_inside_same_writer(published, fault):
    lake, results, records, handles = published
    prepared = capture(published)
    lake.conn.execute("BEGIN IMMEDIATE")
    try:
        if fault == "spec":
            lake.conn.execute("UPDATE research_artifacts SET record_json=json_set(record_json,"
                "'$.spec_fingerprint',?) WHERE artifact_id=?", ("0" * 64, records[0]["artifact_id"]))
        elif fault == "membership":
            lake.conn.execute("UPDATE execution_attempts SET terminal_json=json_set(terminal_json,"
                "'$.artifact_ids',json('[]')) WHERE attempt_id=?", (handles[0].attempt_id,))
        else:
            # Real core API rotates this task/phase; old artifact metadata stays.
            create_attempt(lake.conn, "source-0", "action", "replacement", {})
        with pytest.raises(sources.SourceHOLD):
            sources.require_sources(lake, results, prepared, metadata_only=True)
    finally:
        lake.conn.rollback()
    sources.require_sources(lake, results, prepared)


@pytest.mark.parametrize("fault", ["missing_commit", "same_bytes_new_inode", "hardlink", "symlink"])
def test_confirmed_receipt_and_original_file_identity_cannot_be_replaced(published, fault):
    lake, results, records, handles = published
    prepared = capture(published)
    original = Path(records[0]["path"])
    if fault == "missing_commit":
        original = results / "source-0" / "_execution_effects" / handles[0].attempt_id / "committed.json"
        original.rename(original.with_name("saved-committed"))
    elif fault == "hardlink":
        os.link(original, original.with_name("second-name"))
    else:
        content = original.read_bytes()
        original.rename(original.with_name("old-content"))
        if fault == "symlink":
            original.symlink_to(original.with_name("old-content"))
        else:
            original.write_bytes(content)
    with pytest.raises(sources.SourceHOLD):
        sources.require_sources(lake, results, prepared, metadata_only=True)
    if fault != "same_bytes_new_inode":
        with pytest.raises(sources.SourceHOLD):
            capture(published)


def test_capture_limits_and_short_read_do_not_return_prepared_owner(published, monkeypatch):
    lake, results, records, _ = published
    ids = [record["artifact_id"] for record in records]
    with pytest.raises(sources.SourceHOLD):
        sources.capture_sources(lake, results, ids + ids)
    with pytest.raises(sources.SourceHOLD):
        sources.capture_sources(lake, results, [str(i) for i in range(33)])
    with monkeypatch.context() as patch:
        patch.setattr(sources, "MAX_SOURCE_BYTES", 5)
        with pytest.raises(sources.SourceHOLD, match="aggregate_byte_limit"):
            capture(published)
    with monkeypatch.context() as patch:
        patch.setattr(sources, "MAX_SNAPSHOT_BYTES", 100)
        with pytest.raises(sources.SourceHOLD, match="snapshot_byte_limit"):
            capture(published)
    real_read = os.read
    def read(fd, count):
        info = os.fstat(fd)
        source = Path(records[0]["path"]).stat()
        if (info.st_dev, info.st_ino) == (source.st_dev, source.st_ino):
            return b""
        return real_read(fd, count)
    monkeypatch.setattr(sources.os, "read", read)
    with pytest.raises(sources.SourceHOLD):
        capture(published)


def test_seal_failure_closes_only_created_descriptors_without_yield(published, monkeypatch):
    prepared = capture(published)
    created = []
    actual_create, actual_fcntl = os.memfd_create, fcntl.fcntl
    def create(*args, **kwargs):
        fd = actual_create(*args, **kwargs)
        created.append(fd)
        return fd
    def fail(fd, cmd, *args):
        if fd in created and cmd == fcntl.F_ADD_SEALS:
            raise OSError("controlled seal fault")
        return actual_fcntl(fd, cmd, *args)
    monkeypatch.setattr(os, "memfd_create", create)
    monkeypatch.setattr(fcntl, "fcntl", fail)
    yielded = False
    with pytest.raises(sources.SourceHOLD):
        with sources.sealed_sources(prepared):
            yielded = True
    assert created and not yielded
    for fd in created:
        with pytest.raises(OSError):
            os.fstat(fd)
