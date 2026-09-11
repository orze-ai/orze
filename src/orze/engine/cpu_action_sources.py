"""Current, confirmed artifact inputs for an explicitly admitted CPU action.

The six public helpers capture/recheck sources, return detached binding/record
metadata, and lend sealed worker-only FDs. They never claim a task, write SQL,
acquire a source's effect lock, adopt history, or grant source mutation rights.
Capture/full verification performs bounded content IO outside writer TXs;
metadata_only verification uses the supplied Lake's actual current connection
and cheap captured path/receipt identities, suitable for a final SQL watch.

Limits are 32 inputs, 16 MiB aggregate content, and 32 KiB binding metadata.
Empty files and empty input sets are valid. PreparedSources holds immutable
bytes and the original Lake/connection/PID until its caller releases it; JSON
metadata cannot reconstruct a capability. There is no global owner registry.
Source files remain cooperative same-UID storage, not a hostile-code sandbox.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat

from orze.core.execution_attempts import AttemptRef, require_current
from orze.core.research_artifacts import artifacts_for_attempt, get_artifact, _token
from orze.core.artifact_contract import validate_artifact_publication_binding
from orze.engine import artifact_publication as files
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
from orze.engine.attempt_effect_receipts import _scan, _read, _decode, _ref_fields
from orze.engine.claim_authority import _lake_path, read_claim
from orze.engine.execution_catalog import declared_catalog
from orze.engine.sealed_payload import _features
from orze.engine.termination_hold import require_no_unconfirmed_stop

MAX_SOURCES = 32
MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_SNAPSHOT_BYTES = 32768
_CHUNK = 65536
_KEY = object()


class SourceHOLD(AttemptEffectInDoubt):
    """The captured inputs cannot authorize another GO or publication."""


def _fail(reason):
    raise SourceHOLD("cpu_action_sources_" + reason)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


class PreparedSources:
    __slots__ = ("_owner",)

    def __init__(self, key, owner):
        if key is not _KEY:
            _fail("capture_required")
        self._owner = owner
        owner.handle = self


@dataclass
class _Owner:
    lake: object
    conn: object
    pid: int
    scope: str
    database: str
    route_ids: tuple
    items: tuple
    binding: bytes
    handle: object = None


@dataclass(frozen=True)
class _Item:
    record: bytes
    row: bytes
    identities: tuple
    content: bytes


def _owner(prepared):
    if type(prepared) is not PreparedSources:
        _fail("capture_required")
    owner = prepared._owner
    if type(owner) is not _Owner or owner.handle is not prepared or owner.pid != os.getpid():
        _fail("owner_changed")
    return owner


def _absolute(path):
    path = Path(path).absolute()
    if ".." in path.parts or len(str(path).encode()) > 4096:
        _fail("path_invalid")
    return path


def _route_identity(path, *, directory):
    path = _absolute(path)
    result = []
    for parent in reversed(path.parents):
        info = parent.lstat()
        if not stat.S_ISDIR(info.st_mode):
            _fail("route_redirected")
        result.append((str(parent), True, files._identity(info, True)))
    info = path.lstat()
    if directory:
        if not stat.S_ISDIR(info.st_mode):
            _fail("scope_invalid")
    elif not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        _fail("database_invalid")
    # SQLite's content/size/timestamps change on legitimate peer writes.
    result.append((str(path), True, files._identity(info, True)))
    return tuple(result)


def _route(lake, results_dir):
    if lake is None or _lake_path(lake) is None:
        _fail("lake_required")
    scope, database = _absolute(results_dir), _absolute(_lake_path(lake))
    identities = (_route_identity(scope, directory=True)
                  + _route_identity(database, directory=False))
    return str(scope), str(database), identities


def _metadata(lake, scope, database, artifact_id):
    record = get_artifact(lake.conn, artifact_id)
    if record is None or record["scope"] != scope:
        _fail("artifact_scope_invalid")
    ref = AttemptRef(**record["producer"])
    folder = Path(scope) / ref.task_id
    require_no_unconfirmed_stop(folder)
    if declared_catalog(folder) != database:
        _fail("catalog_changed")
    claim = read_claim(folder / "claim.json")
    if claim is not None and claim.get("lifecycle_db") != database:
        _fail("claim_catalog_changed")
    row = require_current(lake.conn, ref, states=("TERMINAL",))
    terminal = row["terminal"]
    if type(terminal) is not dict or terminal.get("outcome") != "completed":
        _fail("source_not_completed")
    binding = validate_artifact_publication_binding(row["binding"].get("artifact_publication"))
    if (binding["scope"] != scope or binding["spec_fingerprint"] != record["spec_fingerprint"]
            or record["path"] != str(Path(binding["root"]) / artifact_id / "content")):
        _fail("artifact_binding_changed")
    declared = binding["contract"]["outputs"].get(record["logical_name"])
    if declared is None or record["size_bytes"] > declared["max_bytes"]:
        _fail("artifact_declaration_changed")
    all_records = artifacts_for_attempt(lake.conn, ref)
    ids = terminal.get("artifact_ids")
    expected = [item["artifact_id"] for item in all_records]
    if (type(ids) is not list or any(type(value) is not str for value in ids)
            or len(ids) != len(set(ids)) or set(ids) != set(expected)
            or artifact_id not in ids):
        _fail("terminal_artifacts_changed")
    return record, row, ref, folder


def _effect(ref, folder, row):
    digest = row["terminal"].get("effect_receipt_sha256")
    if type(digest) is not str or _scan(folder).get(ref.attempt_id) != (digest, True):
        _fail("effect_unconfirmed")
    parent = folder / "_execution_effects" / ref.attempt_id
    paths = (parent / "prepared.json", parent / "committed.json")
    identities = tuple(value for path in paths for value in files._path_identities(path))
    raw = _read(paths[0])
    prepared = _decode(raw)
    fields = _ref_fields(ref, folder)
    if (_sha(raw) != digest or _json({key: prepared.get(key) for key in fields}) != _json(fields)):
        _fail("effect_reference_changed")
    committed = _decode(_read(paths[1]))
    if _json(committed) != _json({**fields, "event": "effect_committed", "prepared_sha256": digest}):
        _fail("effect_confirmation_changed")
    files._verify_identities(identities)
    return identities


def _content(record, identities):
    path = Path(record["path"])
    parent = files._open_directory(path.parent)
    fd = None
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        before = os.fstat(fd)
        if (files._identity(before) != identities[-1][2]
                or before.st_size != record["size_bytes"] or before.st_size > MAX_SOURCE_BYTES):
            _fail("content_identity_changed")
        chunks, size = [], 0
        while size <= before.st_size:
            chunk = os.read(fd, min(_CHUNK, before.st_size - size + 1))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        raw = b"".join(chunks)
        if (size != before.st_size or _sha(raw) != record["content_sha256"]
                or files._identity(os.fstat(fd)) != files._identity(before)):
            _fail("content_changed")
        files._verify_identities(identities)
        return raw
    finally:
        try:
            if fd is not None:
                os.close(fd)
        finally:
            os.close(parent)


def capture_sources(lake, results_dir, artifact_ids):
    """Capture only current confirmed native artifacts; no SQL or disk writes."""
    try:
        if lake.conn.in_transaction:
            _fail("capture_inside_transaction")
        if type(artifact_ids) is not list or len(artifact_ids) > MAX_SOURCES:
            _fail("input_count_invalid")
        for identity in artifact_ids:
            _token(identity)
        if len(set(artifact_ids)) != len(artifact_ids):
            _fail("duplicate_input")
        scope, database, route_ids = _route(lake, results_dir)
        entries, items, total = [], [], 0
        for identity in artifact_ids:
            record, row, ref, folder = _metadata(lake, scope, database, identity)
            total += record["size_bytes"]
            if total > MAX_SOURCE_BYTES:
                _fail("aggregate_byte_limit")
            entry = {"artifact": record, "source_sha256": _sha(_json(row)),
                     "effect_sha256": row["terminal"].get("effect_receipt_sha256")}
            entries.append(entry)
            binding = {"schema": 1, "scope": scope, "database": database,
                       "database_identity": list(route_ids[-1][2][:2]),
                       "scope_identity": list(_route_identity(Path(scope), directory=True)[-1][2][:2]),
                       "inputs": entries}
            if len(_json(binding)) > MAX_SNAPSHOT_BYTES:
                _fail("snapshot_byte_limit")
            receipt_ids = _effect(ref, folder, row)
            content_ids = files._path_identities(Path(record["path"]))
            raw = _content(record, content_ids)
            items.append(_Item(_json(record), _json(row), receipt_ids + content_ids, raw))
        binding = {"schema": 1, "scope": scope, "database": database,
                   "database_identity": list(route_ids[-1][2][:2]),
                   "scope_identity": list(_route_identity(Path(scope), directory=True)[-1][2][:2]),
                   "inputs": entries}
        encoded = _json(binding)
        if len(encoded) > MAX_SNAPSHOT_BYTES:
            _fail("snapshot_byte_limit")
        owner = _Owner(lake, lake.conn, os.getpid(), scope, database, route_ids, tuple(items), encoded)
        prepared = PreparedSources(_KEY, owner)
        require_sources(lake, results_dir, prepared, metadata_only=True)
        return prepared
    except SourceHOLD:
        raise
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        raise SourceHOLD("cpu_action_sources_capture_unconfirmed") from exc


def snapshot(prepared):
    """Detached bounded binding metadata; not authority to reconstruct inputs."""
    return json.loads(_owner(prepared).binding)


def records(prepared):
    """Detached original metadata, retaining each input's own spec/producer."""
    return [json.loads(item.record) for item in _owner(prepared).items]


def require_sources(lake, results_dir, prepared, *, metadata_only=False):
    try:
        owner = _owner(prepared)
        if (type(metadata_only) is not bool or lake is not owner.lake or lake.conn is not owner.conn
                or str(_absolute(results_dir)) != owner.scope):
            _fail("owner_scope_changed")
        if not metadata_only and lake.conn.in_transaction:
            _fail("content_read_inside_transaction")
        if _route(lake, results_dir) != (owner.scope, owner.database, owner.route_ids):
            _fail("route_changed")
        files._verify_identities(owner.route_ids)
        for item in owner.items:
            original = json.loads(item.record)
            record, row, ref, folder = _metadata(lake, owner.scope, owner.database, original["artifact_id"])
            if _json(record) != item.record or _json(row) != item.row:
                _fail("source_metadata_changed")
            files._verify_identities(item.identities)
            if not metadata_only:
                _effect(ref, folder, row)
                if _content(record, files._path_identities(Path(record["path"]))) != item.content:
                    _fail("source_bytes_changed")
                files._verify_identities(item.identities)
            # A lock-free content read must not hide concurrent SQL rotation.
            again, current, _, _ = _metadata(lake, owner.scope, owner.database, original["artifact_id"])
            if _json(again) != item.record or _json(current) != item.row:
                _fail("source_changed_during_read")
        files._verify_identities(owner.route_ids)
    except SourceHOLD:
        raise
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        raise SourceHOLD("cpu_action_sources_verification_unconfirmed") from exc


@contextmanager
def _sealed(raw):
    import fcntl
    create, flags, seals, add, get = _features()
    writable = readonly = None
    try:
        writable = create("orze-action-source", flags)
        offset = 0
        while offset < len(raw):
            count = os.write(writable, memoryview(raw)[offset:])
            if type(count) is not int or not 0 < count <= len(raw) - offset:
                _fail("sealed_write_unconfirmed")
            offset += count
        fcntl.fcntl(writable, add, seals)
        # This is an exact self-owned descriptor, not process discovery. Reopen
        # read-only after sealing; dup would retain the writable access mode.
        readonly = os.open("/proc/self/fd/" + str(writable), os.O_RDONLY | os.O_CLOEXEC)
        before = os.fstat(writable)
        actual = os.fstat(readonly)
        if ((before.st_dev, before.st_ino, before.st_size) != (actual.st_dev, actual.st_ino, actual.st_size)
                or actual.st_size != len(raw) or not stat.S_ISREG(actual.st_mode)
                or fcntl.fcntl(readonly, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY
                or fcntl.fcntl(readonly, get) & seals != seals):
            _fail("sealed_identity_changed")
        digest, size = hashlib.sha256(), 0
        while size < len(raw):
            chunk = os.pread(readonly, min(_CHUNK, len(raw) - size), size)
            if not chunk:
                _fail("sealed_read_incomplete")
            digest.update(chunk)
            size += len(chunk)
        if digest.hexdigest() != _sha(raw) or os.pread(readonly, 1, size) != b"":
            _fail("sealed_readback_failed")
        if os.lseek(readonly, 0, os.SEEK_SET) != 0:
            _fail("sealed_offset_invalid")
        fd, writable = writable, None
        os.close(fd)
        yield readonly
    finally:
        try:
            if readonly is not None:
                fd, readonly = readonly, None
                os.close(fd)
        finally:
            if writable is not None:
                fd, writable = writable, None
                os.close(fd)


@contextmanager
def sealed_sources(prepared):
    """Lend offset-zero, immutable read-only descriptors to the worker only."""
    owner = _owner(prepared)
    require_sources(owner.lake, owner.scope, prepared)
    try:
        with ExitStack() as stack:
            descriptors = {}
            for item in owner.items:
                descriptors[json.loads(item.record)["artifact_id"]] = stack.enter_context(_sealed(item.content))
            yield {"ORZE_ACTION_SOURCE_FDS": _json(descriptors).decode()}, tuple(descriptors.values())
    except SourceHOLD:
        raise
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        raise SourceHOLD("cpu_action_sources_sealed_transport_unconfirmed") from exc
