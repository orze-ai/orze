"""Bounded, explicitly published research declarations in an existing database.

Each logical project has one current document and at most one pending update.
Preparation commits the pending document; publication replaces current and
clears pending in one transaction. Readers refuse pending or unavailable
state. An operation is identified by its ID AND expected predecessor revision;
content returning to older bytes does not revive an old request.

Existing authored records cannot be removed or rewritten. Their bindings may
refresh for the same source identities; new records may be appended. Capacity
rejects the whole document. This is not semantic compaction or evidence
qualification: every readable document remains pending_verification.

Connections require an existing DELETE/NORMAL database; writers use EXTRA
synchronization locally. No WAL conversion or execution-state writes occur.
The protocol depends on SQLite's locking/sync guarantees and cooperating
namespace owners. Path checks detect observed changes, not arbitrary rename
away-and-back races. These APIs do not establish network-filesystem support,
power-loss durability of a deployment, or authority to take over execution.
"""
import hashlib
from pathlib import Path
import re
import sqlite3
import stat

from orze.research.memory_format import MAX_BYTES, MemoryUnavailable, decode_memory, project_scope

MAX_REVISION = 9223372036854775807
_CURRENT = 'research_memory_current_v1'
_PENDING = 'research_memory_pending_v1'
_TABLE_SQL = {
    name: 'CREATE TABLE ' + name + ''' (
    scope TEXT PRIMARY KEY NOT NULL,
    revision INTEGER NOT NULL CHECK(revision BETWEEN 1 AND 9223372036854775807),
    operation_id TEXT NOT NULL CHECK(length(operation_id)=32),
    predecessor INTEGER,
    document_sha256 TEXT NOT NULL CHECK(length(document_sha256)=64),
    document_json TEXT NOT NULL CHECK(length(CAST(document_json AS BLOB)) BETWEEN 1 AND 65536)
)''' for name in (_CURRENT, _PENDING)}


def _route(path):
    path = Path(path).absolute()
    if '..' in path.parts:
        raise MemoryUnavailable('memory_database_route_invalid')
    identities = []
    for parent in reversed(path.parents):
        info = parent.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise MemoryUnavailable('memory_database_route_invalid')
        identities.append((str(parent), info.st_dev, info.st_ino, info.st_mode))
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise MemoryUnavailable('memory_database_unsafe')
    return (*identities, (str(path), info.st_dev, info.st_ino, info.st_mode))


def _open(database, *, write):
    path = Path(database).absolute()
    before = _route(path)
    connection = sqlite3.connect(path.as_uri() + ('?mode=rw' if write else '?mode=ro'),
                                 uri=True, isolation_level=None, timeout=1)
    try:
        if connection.execute('PRAGMA journal_mode').fetchone()[0] != 'delete':
            raise MemoryUnavailable('memory_delete_journal_required')
        if connection.execute('PRAGMA locking_mode').fetchone()[0] != 'normal':
            raise MemoryUnavailable('memory_normal_locking_required')
        if write:
            connection.execute('PRAGMA synchronous=EXTRA')
            if connection.execute('PRAGMA synchronous').fetchone()[0] != 3:
                raise MemoryUnavailable('memory_extra_sync_unavailable')
        else:
            connection.execute('PRAGMA query_only=ON')
        def fence():
            if _route(path) != before:
                raise MemoryUnavailable('memory_database_namespace_changed')
        fence()
        return connection, fence
    except BaseException:
        connection.close()
        raise


def _schema(connection, *, create=False):
    existing = {name: connection.execute('SELECT sql FROM sqlite_master WHERE name=?', (name,)).fetchone()
                for name in _TABLE_SQL}
    if all(value is None for value in existing.values()):
        if not create:
            return False
        for sql in _TABLE_SQL.values():
            connection.execute(sql)
    elif any(value is None for value in existing.values()):
        raise MemoryUnavailable('memory_store_schema_incomplete')
    for name, sql in _TABLE_SQL.items():
        if connection.execute('SELECT sql FROM sqlite_master WHERE name=?', (name,)).fetchone()[0] != sql:
            raise MemoryUnavailable('memory_store_schema_invalid')
        if connection.execute("SELECT 1 FROM sqlite_master WHERE type='trigger' AND tbl_name=? LIMIT 1", (name,)).fetchone():
            raise MemoryUnavailable('memory_store_trigger_invalid')
    return True


def _revision(value):
    if value is not None and (type(value) is not int or not 1 <= value <= MAX_REVISION):
        raise MemoryUnavailable('memory_revision_invalid')


def _operation_id(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{32}', value):
        raise MemoryUnavailable('memory_operation_id_invalid')


def _digest(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise MemoryUnavailable('memory_document_digest_invalid')


def _row(connection, table, scope):
    value = connection.execute(
        'SELECT revision,operation_id,predecessor,document_sha256,'
        "CASE WHEN typeof(document_json)='text' AND length(CAST(document_json AS BLOB))<=? "
        'THEN document_json ELSE NULL END FROM ' + table + ' WHERE scope=?',
        (MAX_BYTES, scope)).fetchone()
    if value is None:
        return None
    revision, operation_id, predecessor, digest, text = value
    _revision(revision)
    _revision(predecessor)
    if revision is None or revision != (predecessor or 0) + 1:
        raise MemoryUnavailable('memory_revision_invalid')
    _operation_id(operation_id)
    _digest(digest)
    if type(text) is not str or hashlib.sha256(text.encode()).hexdigest() != digest:
        raise MemoryUnavailable('memory_store_row_invalid')
    document = decode_memory(text.encode(), scope)
    return {'revision': revision, 'operation_id': operation_id, 'predecessor': predecessor,
            'document_sha256': digest, 'document_json': text, 'document': document}


def _preserve_records(previous, proposed):
    old = {entry['id']: entry for entry in previous['entries']}
    new = {entry['id']: entry for entry in proposed['entries']}
    if not old.keys() <= new.keys():
        raise MemoryUnavailable('memory_record_removal_refused')
    for key, entry in old.items():
        candidate = new[key]
        if {k: v for k, v in entry.items() if k != 'sources'} != {k: v for k, v in candidate.items() if k != 'sources'}:
            raise MemoryUnavailable('memory_record_rewrite_refused')
        def identities(record):
            return sorted((source['kind'], source['binding_schema'], source['idea_id']) for source in record['sources'])
        if identities(entry) != identities(candidate):
            raise MemoryUnavailable('memory_source_retarget_refused')


def _unavailable(write, reason, *, unknown=False):
    return {'status' if write else 'availability': 'unknown' if unknown else 'unavailable', 'reason': reason}


def _session(database, *, write, action):
    connection, result = None, None
    state = {'committing': False}
    try:
        connection, fence = _open(database, write=write)
        result = action(connection, fence, state)
    except MemoryUnavailable as exc:
        result = _unavailable(write, str(exc), unknown=write and state['committing'])
    except (OSError, sqlite3.Error, ValueError, TypeError, AttributeError):
        result = _unavailable(write, 'memory_commit_unknown' if state['committing'] else 'memory_database_unavailable',
                              unknown=write and state['committing'])
    finally:
        cleanup_failed = False
        if connection is not None:
            try:
                if connection.in_transaction:
                    connection.rollback()
            except (OSError, sqlite3.Error):
                cleanup_failed = True
            try:
                connection.close()
            except (OSError, sqlite3.Error):
                cleanup_failed = True
        if cleanup_failed:
            result = _unavailable(write, 'memory_cleanup_unavailable', unknown=write)
    return result


def _commit(connection, fence, state):
    fence()
    state['committing'] = True
    connection.commit()
    fence()


def read_stored_memory(results_dir, *, database):
    """Read one transaction without recovery writes; declarations stay unverified."""
    try:
        scope = project_scope(results_dir)
    except (MemoryUnavailable, OSError, ValueError, TypeError):
        return _unavailable(False, 'memory_scope_invalid')
    def action(connection, fence, state):
        connection.execute('BEGIN')
        if not _schema(connection):
            fence()
            return _unavailable(False, 'memory_absent')
        pending = _row(connection, _PENDING, scope)
        current = _row(connection, _CURRENT, scope)
        fence()
        if pending:
            return {**_unavailable(False, 'memory_update_pending'),
                    'revision': pending['revision'], 'operation_id': pending['operation_id']}
        if current is None:
            return _unavailable(False, 'memory_absent')
        return {key: current[key] for key in ('revision', 'operation_id', 'document_sha256', 'document')} | {
            'availability': 'pending_verification'}
    return _session(database, write=False, action=action)


def inspect_pending_memory(results_dir, *, database):
    """Read an unverified prepared intent, preserving its exact JSON text.

    This does not publish, repair, qualify sources or grant execution rights.
    A recovery caller must explicitly authorize publication and pass the same
    predecessor, operation ID and digest to publish_memory_update.
    """
    try:
        scope = project_scope(results_dir)
    except (MemoryUnavailable, OSError, ValueError, TypeError):
        return _unavailable(False, 'memory_scope_invalid')
    def action(connection, fence, state):
        connection.execute('BEGIN')
        if not _schema(connection):
            fence()
            return _unavailable(False, 'memory_pending_absent')
        pending = _row(connection, _PENDING, scope)
        current = _row(connection, _CURRENT, scope)
        if pending is not None:
            if pending['predecessor'] != (current['revision'] if current else None):
                raise MemoryUnavailable('memory_pending_predecessor_mismatch')
            if current is not None:
                _preserve_records(current['document'], pending['document'])
        fence()
        if pending is None:
            return _unavailable(False, 'memory_pending_absent')
        return {key: pending[key] for key in ('revision', 'operation_id', 'predecessor',
                'document_sha256', 'document_json')} | {'availability': 'prepared_update'}
    return _session(database, write=False, action=action)


def prepare_memory_update(results_dir, raw, *, database, operation_id, expected_revision):
    """Persist a bounded intent; replay only the same predecessor/ID/bytes.

None denotes an absent current revision. A prepared result is not publication.
Keep the returned document_sha256 for publish_memory_update. Unknown outcomes
must be reconciled by repeating this exact request; pending updates never age
into permission for a different operation.
"""
    try:
        _operation_id(operation_id)
        _revision(expected_revision)
        if expected_revision == MAX_REVISION:
            raise MemoryUnavailable('memory_revision_exhausted')
        scope = project_scope(results_dir)
        document = decode_memory(raw, scope)
        text = raw.decode('utf-8')
        digest = hashlib.sha256(raw).hexdigest()
    except MemoryUnavailable as exc:
        return _unavailable(True, str(exc))
    except (OSError, ValueError, TypeError):
        return _unavailable(True, 'memory_request_invalid')
    def action(connection, fence, state):
        connection.execute('BEGIN IMMEDIATE')
        _schema(connection, create=True)
        pending = _row(connection, _PENDING, scope)
        current = _row(connection, _CURRENT, scope)
        fence()
        if pending:
            if (current['revision'] if current else None) != pending['predecessor']:
                return {'status': 'conflict'}
            if (pending['operation_id'], pending['predecessor'], pending['document_sha256']) == (operation_id, expected_revision, digest):
                return {'status': 'prepared', 'revision': pending['revision'], 'document_sha256': digest, 'replayed': True}
            return {'status': 'busy', 'revision': pending['revision']}
        if current and (current['operation_id'], current['predecessor']) == (operation_id, expected_revision):
            if current['document_sha256'] == digest:
                return {'status': 'already_committed', 'revision': current['revision'], 'document_sha256': digest}
            return {'status': 'operation_collision'}
        if (current['revision'] if current else None) != expected_revision:
            return {'status': 'conflict'}
        if current:
            _preserve_records(current['document'], document)
        desired = (expected_revision or 0) + 1
        connection.execute('INSERT INTO ' + _PENDING + ' VALUES (?,?,?,?,?,?)',
                           (scope, desired, operation_id, expected_revision, digest, text))
        _commit(connection, fence, state)
        return {'status': 'prepared', 'revision': desired, 'document_sha256': digest, 'replayed': False}
    return _session(database, write=True, action=action)


def publish_memory_update(results_dir, *, database, operation_id, expected_revision, document_sha256):
    """Publish the exact prepared intent, or recognize its committed replay.

No abort/takeover/overwrite fallback is provided. This can recover SQLite's
rollback journal as an explicit writer; a read-only call cannot do so.
"""
    try:
        _operation_id(operation_id)
        _revision(expected_revision)
        _digest(document_sha256)
        scope = project_scope(results_dir)
    except MemoryUnavailable as exc:
        return _unavailable(True, str(exc))
    except (OSError, ValueError, TypeError):
        return _unavailable(True, 'memory_request_invalid')
    expected = (operation_id, expected_revision, document_sha256)
    def identity(record):
        return record['operation_id'], record['predecessor'], record['document_sha256']
    def action(connection, fence, state):
        connection.execute('BEGIN IMMEDIATE')
        if not _schema(connection):
            fence()
            return {'status': 'pending_absent'}
        pending = _row(connection, _PENDING, scope)
        current = _row(connection, _CURRENT, scope)
        fence()
        if pending is None:
            if current and identity(current) == expected:
                return {'status': 'already_committed', 'revision': current['revision'], 'document_sha256': document_sha256}
            return {'status': 'pending_absent'}
        if identity(pending) != expected:
            return {'status': 'operation_mismatch'}
        if (current['revision'] if current else None) != expected_revision:
            return {'status': 'conflict'}
        if current:
            _preserve_records(current['document'], pending['document'])
        connection.execute('INSERT INTO ' + _CURRENT + ' VALUES (?,?,?,?,?,?) ON CONFLICT(scope) DO UPDATE SET '
                           'revision=excluded.revision, operation_id=excluded.operation_id, predecessor=excluded.predecessor, '
                           'document_sha256=excluded.document_sha256, document_json=excluded.document_json',
                           (scope, pending['revision'], operation_id, expected_revision, document_sha256, pending['document_json']))
        connection.execute('DELETE FROM ' + _PENDING + ' WHERE scope=?', (scope,))
        _commit(connection, fence, state)
        return {'status': 'committed', 'revision': pending['revision'], 'document_sha256': document_sha256}
    return _session(database, write=True, action=action)
