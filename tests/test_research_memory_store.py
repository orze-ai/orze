"""Publication/restart/failure boundaries for unverified memory declarations."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

import pytest

from orze.idea_lake import IdeaLake


CURRENT = 'research_memory_current_v1'
PENDING = 'research_memory_pending_v1'
SOURCE = str(Path(__file__).resolve().parents[1] / 'src')


def api():
    return importlib.import_module('orze.research.memory_store')


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def document(root, count=1, large=False):
    return {'schema': 1, 'project_scope': hashlib.sha256(str(root.absolute()).encode()).hexdigest(), 'entries': [
        {'id': 'record-' + str(index), 'question': 'Can this hypothesis survive the counterexample?',
         'hypothesis': 'The behavior may generalize.', 'rationale': 'r' * (800 if large else 10),
         'claim': 'c' * (800 if large else 10), 'claimed_state': 'refuted' if index == 0 else 'unknown',
         'origin': 'authored', 'sources': [{'kind': 'report', 'binding_schema': 1, 'idea_id': 'idea-counterexample',
                                          'binding_sha256': 'a'*64}] if index == 0 else []} for index in range(count)]}


def fixture(root, *, count=1, large=False):
    results = root / 'results'
    results.mkdir()
    database = root / 'ideas.sqlite'
    lake = IdeaLake(database)
    lake.close()
    return results, database, encoded(document(results, count, large))


def prepare(results, database, raw, op='1'*32, previous=None):
    return api().prepare_memory_update(results, raw, database=database, operation_id=op, expected_revision=previous)


def publish(results, database, raw, op='1'*32, previous=None):
    return api().publish_memory_update(results, database=database, operation_id=op,
                                       expected_revision=previous, document_sha256=hashlib.sha256(raw).hexdigest())


def read(results, database):
    return api().read_stored_memory(results, database=database)


def initialize(results, database, raw):
    assert prepare(results, database, raw)['status'] == 'prepared'
    assert publish(results, database, raw)['status'] == 'committed'


def sql(database, query, parameters=()):
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(query, parameters).fetchall()
        connection.commit()
        return rows
    finally:
        connection.close()


def native(database):
    connection = sqlite3.connect(database)
    try:
        names = [r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                 if r[0] not in (CURRENT, PENDING)]
        return {name: (connection.execute('SELECT sql FROM sqlite_master WHERE name=?', (name,)).fetchone()[0],
                       sorted(connection.execute('SELECT * FROM "' + name + '"').fetchall(), key=repr)) for name in names}
    finally:
        connection.close()


def file_bytes(database):
    return {p.name: p.read_bytes() for p in database.parent.glob(database.name + '*') if p.is_file()}


def child(script, *arguments, cwd):
    return subprocess.run([sys.executable, '-c', script, *map(str, arguments)], cwd=cwd,
                          capture_output=True, text=True, timeout=30,
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=SOURCE))


def test_restart_publishes_only_pending_verification_and_never_execution_rights(tmp_path):
    results, database, raw = fixture(tmp_path)
    before = native(database)
    assert read(results, database) == {'availability': 'unavailable', 'reason': 'memory_absent'}
    assert prepare(results, database, raw)['revision'] == 1
    assert read(results, database)['reason'] == 'memory_update_pending'
    assert publish(results, database, raw)['status'] == 'committed'
    script = '''
import json, sys
from orze.research.memory_store import read_stored_memory
print(json.dumps(read_stored_memory(sys.argv[1], database=sys.argv[2])))
'''
    original_bytes = file_bytes(database)
    reads = []
    for _ in range(2):
        process = child(script, results, database, cwd=tmp_path)
        assert process.returncode == 0, process.stderr
        value = json.loads(process.stdout)
        assert value['availability'] == 'pending_verification'
        assert value['document'] == json.loads(raw)
        assert value['revision'] == 1
        reads.append(value)
    assert reads[0] == reads[1]
    assert original_bytes == file_bytes(database)
    assert before == native(database)
    assert sql(database, 'SELECT count(*) FROM ' + CURRENT) == [(1,)]
    assert sql(database, 'SELECT count(*) FROM ' + PENDING) == [(0,)]


def test_retry_payload_collisions_and_aba_use_revision_and_hash(tmp_path):
    results, database, raw_a = fixture(tmp_path)
    changed = json.loads(raw_a)
    changed['entries'][0]['sources'][0]['binding_sha256'] = 'b'*64
    raw_b = encoded(changed)
    assert prepare(results, database, raw_a)['status'] == 'prepared'
    assert prepare(results, database, raw_a)['replayed'] is True
    assert prepare(results, database, raw_b)['status'] == 'busy'
    assert publish(results, database, raw_b)['status'] == 'operation_mismatch'
    assert publish(results, database, raw_a, op='2'*32)['status'] == 'operation_mismatch'
    assert publish(results, database, raw_a)['status'] == 'committed'
    assert prepare(results, database, raw_a)['status'] == 'already_committed'
    assert publish(results, database, raw_a)['status'] == 'already_committed'
    assert prepare(results, database, raw_b)['status'] == 'operation_collision'
    for old, op, raw in [(1, '2'*32, raw_b), (2, '1'*32, raw_a)]:
        assert prepare(results, database, raw, op, old)['revision'] == old + 1
        assert publish(results, database, raw, op, old)['revision'] == old + 1
    assert read(results, database)['revision'] == 3
    assert read(results, database)['document_sha256'] == hashlib.sha256(raw_a).hexdigest()
    assert prepare(results, database, raw_b, '2'*32, 1)['status'] == 'conflict'
    assert publish(results, database, raw_a, '1'*32, None)['status'] == 'pending_absent'


@pytest.mark.parametrize('field,replacement,reason', [
    ('id', 'different-id', 'memory_record_removal_refused'),
    ('question', 'A different question', 'memory_record_rewrite_refused'),
    ('hypothesis', 'A different hypothesis', 'memory_record_rewrite_refused'),
    ('rationale', 'A different rationale', 'memory_record_rewrite_refused'),
    ('claim', 'A different claim', 'memory_record_rewrite_refused'),
    ('claimed_state', 'confirmed', 'memory_record_rewrite_refused'),
    ('origin', 'derived', 'memory_record_rewrite_refused'),
    ('sources', [{'kind': 'report', 'binding_schema': 1, 'idea_id': 'idea-other', 'binding_sha256': 'b'*64}],
     'memory_source_retarget_refused'),
])
def test_existing_counterexample_cannot_be_rewritten(tmp_path, field, replacement, reason):
    results, database, raw = fixture(tmp_path)
    initialize(results, database, raw)
    before = file_bytes(database)
    value = json.loads(raw)
    value['entries'][0][field] = replacement
    assert prepare(results, database, encoded(value), '2'*32, 1) == {'status': 'unavailable', 'reason': reason}
    assert before == file_bytes(database)
    assert read(results, database)['document'] == json.loads(raw)


def test_removal_capacity_and_append_reorder_preserve_old_records(tmp_path):
    results, database, raw = fixture(tmp_path)
    initialize(results, database, raw)
    empty = encoded({'schema': 1, 'project_scope': document(results)['project_scope'], 'entries': []})
    assert prepare(results, database, empty, '2'*32, 1)['reason'] == 'memory_record_removal_refused'
    full = document(results, 32)
    full['entries'].reverse()
    raw_full = encoded(full)
    assert prepare(results, database, raw_full, '2'*32, 1)['status'] == 'prepared'
    assert publish(results, database, raw_full, '2'*32, 1)['status'] == 'committed'
    before = file_bytes(database)
    assert prepare(results, database, encoded(document(results, 33)), '3'*32, 2)['reason'] == 'memory_entry_count_invalid'
    assert prepare(results, database, raw_full + b' '*65536, '3'*32, 2)['reason'] == 'memory_size_invalid'
    assert before == file_bytes(database)
    value = read(results, database)
    assert len(value['document']['entries']) == 32
    assert json.loads(raw)['entries'][0] in value['document']['entries']


@pytest.mark.parametrize('action', ['prepare', 'publish'])
def test_pending_predecessor_conflict_never_takes_over(tmp_path, action):
    results, database, raw = fixture(tmp_path)
    initialize(results, database, raw)
    assert prepare(results, database, raw, '2'*32, 1)['status'] == 'prepared'
    sql(database, 'UPDATE ' + CURRENT + ' SET revision=3,predecessor=2')
    before = file_bytes(database)
    value = (prepare if action == 'prepare' else publish)(results, database, raw, '2'*32, 1)
    assert value == {'status': 'conflict'}
    assert before == file_bytes(database)
    assert read(results, database)['reason'] == 'memory_update_pending'


def test_publication_rechecks_retention_and_document_hash(tmp_path):
    results, database, raw = fixture(tmp_path)
    initialize(results, database, raw)
    assert prepare(results, database, raw, '2'*32, 1)['status'] == 'prepared'
    empty = encoded({'schema': 1, 'project_scope': document(results)['project_scope'], 'entries': []})
    sql(database, 'UPDATE ' + PENDING + ' SET document_json=?,document_sha256=?',
        (empty.decode(), hashlib.sha256(empty).hexdigest()))
    before = file_bytes(database)
    assert publish(results, database, raw, '2'*32, 1)['status'] == 'operation_mismatch'
    assert publish(results, database, empty, '2'*32, 1)['reason'] == 'memory_record_removal_refused'
    assert before == file_bytes(database)


def test_two_process_preparations_have_exactly_one_winner(tmp_path):
    results, database, raw = fixture(tmp_path)
    path = tmp_path / 'document.json'
    path.write_bytes(raw)
    barrier = tmp_path / 'go'
    script = '''
import json, os, sys, time
from pathlib import Path
from orze.research.memory_store import prepare_memory_update
results, database, raw, operation, ready, go = sys.argv[1:]
Path(ready).write_text('ready')
deadline=time.monotonic()+20
while not Path(go).exists():
    assert time.monotonic()<deadline
    time.sleep(.01)
print(json.dumps(prepare_memory_update(results,Path(raw).read_bytes(),database=database,operation_id=operation,expected_revision=None)))
'''
    processes = []
    try:
        for index in (1, 2):
            process = subprocess.Popen([sys.executable, '-c', script, str(results), str(database), str(path), str(index)*32,
                                        str(tmp_path / ('ready-' + str(index))), str(barrier)], cwd=tmp_path,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=SOURCE))
            processes.append(process)
        deadline = time.monotonic() + 20
        while not all((tmp_path / ('ready-' + str(index))).exists() for index in (1, 2)):
            assert time.monotonic() < deadline
            time.sleep(.01)
        barrier.write_text('go')
        values = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=30)
            assert process.returncode == 0, stderr
            values.append(json.loads(stdout))
        assert sorted(value['status'] for value in values) == ['busy', 'prepared']
        winner = str(1 + next(index for index, value in enumerate(values) if value['status'] == 'prepared')) * 32
        assert publish(results, database, raw, winner)['status'] == 'committed'
    finally:
        # Let disposable children finish; do not terminate test runners.
        if not barrier.exists():
            barrier.write_text('go')
        for process in processes:
            if process.poll() is None:
                process.communicate(timeout=30)


@pytest.mark.parametrize('point', ['prepare_inserted', 'prepare_committed', 'publish_upserted', 'publish_pending_deleted', 'publish_committed'])
def test_process_crash_and_explicit_recovery(tmp_path, point):
    results, database, raw = fixture(tmp_path, count=32, large=True)
    before = native(database)
    if point.startswith('publish_'):
        assert prepare(results, database, raw)['status'] == 'prepared'
    path = tmp_path / 'document.json'
    path.write_bytes(raw)
    script = '''
import hashlib, os, sys
from pathlib import Path
from orze.research import memory_store as m
results, database, raw_path, point = sys.argv[1:]
raw=Path(raw_path).read_bytes()
real_open=m._open
class Proxy:
    def __init__(self, connection): self.connection=connection
    def __getattr__(self, name): return getattr(self.connection,name)
    def execute(self, sql, *args):
        value=self.connection.execute(sql,*args)
        prefixes={'prepare_inserted':'INSERT INTO research_memory_pending_v1',
                  'publish_upserted':'INSERT INTO research_memory_current_v1',
                  'publish_pending_deleted':'DELETE FROM research_memory_pending_v1'}
        if point in prefixes and sql.startswith(prefixes[point]): os._exit(73)
        return value
    def commit(self):
        self.connection.commit()
        if point in ('prepare_committed','publish_committed'): os._exit(73)
def wrapped(database, *, write):
    connection,fence=real_open(database,write=write)
    connection.execute('PRAGMA cache_size=1') # fixture-only spill to a hot rollback journal
    return Proxy(connection),fence
m._open=wrapped
if point.startswith('prepare_'):
    m.prepare_memory_update(results,raw,database=database,operation_id='1'*32,expected_revision=None)
else:
    m.publish_memory_update(results,database=database,operation_id='1'*32,expected_revision=None,document_sha256=hashlib.sha256(raw).hexdigest())
raise AssertionError('fault point did not run')
'''
    process = child(script, results, database, path, point, cwd=tmp_path)
    assert process.returncode == 73, process.stderr
    crash_bytes = file_bytes(database)
    snapshot = tmp_path / 'crash-snapshot'
    snapshot.mkdir()
    for name, data in crash_bytes.items():
        (snapshot / name).write_bytes(data)
    value = read(results, database)
    if point == 'publish_committed':
        assert value['availability'] == 'pending_verification'
    else:
        assert value['availability'] == 'unavailable'
    if point in ('prepare_inserted', 'publish_upserted', 'publish_pending_deleted'):
        assert value['reason'] == 'memory_database_unavailable'
        assert crash_bytes[database.name + '-journal'].startswith(bytes.fromhex('d9d505f920a163d7'))
    assert crash_bytes == file_bytes(database)
    assert prepare(results, database, raw)['status'] == ('already_committed' if point == 'publish_committed' else 'prepared')
    assert publish(results, database, raw)['status'] == ('already_committed' if point == 'publish_committed' else 'committed')
    assert read(results, database)['document_sha256'] == hashlib.sha256(raw).hexdigest()
    assert native(database) == before
    assert sql(database, 'SELECT count(*) FROM ' + CURRENT) == [(1,)]
    assert sql(database, 'SELECT count(*) FROM ' + PENDING) == [(0,)]


class Proxy:
    def __init__(self, connection, mode):
        self.connection, self.mode = connection, mode
    def __getattr__(self, name):
        return getattr(self.connection, name)
    def commit(self):
        if self.mode == 'commit-before':
            raise sqlite3.OperationalError('token=must-not-leak-before')
        self.connection.commit()
        if self.mode == 'commit-after':
            raise sqlite3.OperationalError('token=must-not-leak-after')
    def rollback(self):
        if self.mode == 'rollback':
            raise sqlite3.OperationalError('token=must-not-leak-rollback')
        return self.connection.rollback()
    def close(self):
        self.connection.close()
        if self.mode == 'close':
            raise sqlite3.OperationalError('token=must-not-leak-close')


@pytest.mark.parametrize('action', ['prepare', 'publish'])
@pytest.mark.parametrize('mode', ['commit-before', 'commit-after'])
def test_commit_error_is_unknown_and_exact_retry_reconciles(tmp_path, monkeypatch, action, mode):
    results, database, raw = fixture(tmp_path)
    initialize(results, database, raw)
    if action == 'publish':
        assert prepare(results, database, raw, '2'*32, 1)['status'] == 'prepared'
    original = api()._open
    def wrapped(database, *, write):
        connection, fence = original(database, write=write)
        return Proxy(connection, mode), fence
    with monkeypatch.context() as patch:
        patch.setattr(api(), '_open', wrapped)
        value = (prepare if action == 'prepare' else publish)(results, database, raw, '2'*32, 1)
    assert value == {'status': 'unknown', 'reason': 'memory_commit_unknown'}
    current = read(results, database)
    pending = (action == 'prepare' and mode == 'commit-after') or (action == 'publish' and mode == 'commit-before')
    if pending:
        assert current['reason'] == 'memory_update_pending'
    else:
        assert current['revision'] == (2 if action == 'publish' else 1)
    already = action == 'publish' and mode == 'commit-after'
    assert prepare(results, database, raw, '2'*32, 1)['status'] == ('already_committed' if already else 'prepared')
    assert publish(results, database, raw, '2'*32, 1)['status'] == ('already_committed' if already else 'committed')


@pytest.mark.parametrize('mode', ['rollback', 'close'])
@pytest.mark.parametrize('action', ['read', 'prepare', 'publish'])
def test_cleanup_failure_has_structured_unavailable_or_unknown_result(tmp_path, monkeypatch, mode, action):
    results, database, raw = fixture(tmp_path)
    initialize(results, database, raw)
    before = file_bytes(database)
    original = api()._open
    def wrapped(database, *, write):
        connection, fence = original(database, write=write)
        return Proxy(connection, mode), fence
    with monkeypatch.context() as patch:
        patch.setattr(api(), '_open', wrapped)
        value = read(results, database) if action == 'read' else (prepare if action == 'prepare' else publish)(results, database, raw)
    assert value == ({'availability': 'unavailable', 'reason': 'memory_cleanup_unavailable'} if action == 'read'
                     else {'status': 'unknown', 'reason': 'memory_cleanup_unavailable'})
    assert before == file_bytes(database)


@pytest.mark.parametrize('phase', ['read', 'prepare', 'after-commit'])
def test_observed_database_replacement_is_not_acknowledged(tmp_path, monkeypatch, phase):
    results, database, raw = fixture(tmp_path)
    initialize(results, database, raw)
    if phase == 'after-commit':
        assert prepare(results, database, raw, '2'*32, 1)['status'] == 'prepared'
    original_open, original_row = api()._open, api()._row
    replaced = False
    def replace():
        nonlocal replaced
        if not replaced:
            database.rename(database.with_name('original.sqlite'))
            shutil.copyfile(database.with_name('original.sqlite'), database)
            replaced = True
    def row(*args):
        value = original_row(*args)
        if args[1] == CURRENT:
            replace()
        return value
    class CommitProxy(Proxy):
        def commit(self):
            self.connection.commit()
            replace()
    def wrapped(database, *, write):
        connection, fence = original_open(database, write=write)
        return CommitProxy(connection, ''), fence
    with monkeypatch.context() as patch:
        if phase == 'after-commit':
            patch.setattr(api(), '_open', wrapped)
            value = publish(results, database, raw, '2'*32, 1)
        else:
            patch.setattr(api(), '_row', row)
            value = read(results, database) if phase == 'read' else prepare(results, database, raw, '2'*32, 1)
    assert replaced
    key = 'availability' if phase == 'read' else 'status'
    assert value == {key: 'unknown' if phase == 'after-commit' else 'unavailable', 'reason': 'memory_database_namespace_changed'}
    assert read(results, database)['revision'] == (2 if phase == 'after-commit' else 1)


@pytest.mark.parametrize('query,parameters,reason', [
    ('DROP TABLE ' + PENDING, (), 'memory_store_schema_incomplete'),
    ('ALTER TABLE ' + CURRENT + ' ADD COLUMN surprise TEXT', (), 'memory_store_schema_invalid'),
    ('CREATE TRIGGER memory_side_effect AFTER INSERT ON ' + PENDING + ' BEGIN UPDATE id_sequence SET next_id=999; END', (), 'memory_store_trigger_invalid'),
    ('UPDATE ' + CURRENT + ' SET document_sha256=?', ('b'*64,), 'memory_store_row_invalid'),
    ('UPDATE ' + CURRENT + ' SET predecessor=1', (), 'memory_revision_invalid'),
    ('UPDATE ' + CURRENT + ' SET document_json=?,document_sha256=?', ('{', hashlib.sha256(b'{').hexdigest()), 'memory_json_invalid'),
    ('UPDATE ' + CURRENT + ' SET document_json=?', (b'blob',), 'memory_store_row_invalid'),
])
def test_corrupt_schema_or_rows_reject_without_repair(tmp_path, query, parameters, reason):
    results, database, raw = fixture(tmp_path)
    initialize(results, database, raw)
    before_native = native(database)
    sql(database, query, parameters)
    before = file_bytes(database)
    assert read(results, database) == {'availability': 'unavailable', 'reason': reason}
    assert prepare(results, database, raw, '2'*32, 1) == {'status': 'unavailable', 'reason': reason}
    assert file_bytes(database) == before
    assert native(database) == before_native


@pytest.mark.parametrize('value', ['', 'A'*32, 'a'*31, 'g'*32, 42, True, None])
def test_invalid_operation_id_does_not_open_database(tmp_path, monkeypatch, value):
    results, database, raw = fixture(tmp_path)
    def unexpected(*args, **kwargs):
        pytest.fail('invalid input opened database')
    monkeypatch.setattr(api(), '_open', unexpected)
    assert prepare(results, database, raw, value)['reason'] == 'memory_operation_id_invalid'
    assert publish(results, database, raw, value)['reason'] == 'memory_operation_id_invalid'


@pytest.mark.parametrize('value', [0, -1, True, 1.0, '1', 9223372036854775808])
def test_invalid_revision_does_not_open_database(tmp_path, monkeypatch, value):
    results, database, raw = fixture(tmp_path)
    def unexpected(*args, **kwargs):
        pytest.fail('invalid input opened database')
    monkeypatch.setattr(api(), '_open', unexpected)
    assert prepare(results, database, raw, previous=value)['reason'] == 'memory_revision_invalid'
    assert publish(results, database, raw, previous=value)['reason'] == 'memory_revision_invalid'


def test_revision_exhaustion_never_wraps(tmp_path):
    results, database, raw = fixture(tmp_path)
    initialize(results, database, raw)
    sql(database, 'UPDATE ' + CURRENT + ' SET revision=?,predecessor=?', (api().MAX_REVISION, api().MAX_REVISION - 1))
    before = file_bytes(database)
    assert prepare(results, database, raw, '2'*32, api().MAX_REVISION)['reason'] == 'memory_revision_exhausted'
    assert before == file_bytes(database)


def test_wal_is_refused_and_never_converted(tmp_path):
    results, database, raw = fixture(tmp_path)
    initialize(results, database, raw)
    sql(database, 'PRAGMA journal_mode=WAL')
    assert read(results, database)['reason'] == 'memory_delete_journal_required'
    assert prepare(results, database, raw, '2'*32, 1)['reason'] == 'memory_delete_journal_required'
    assert sql(database, 'PRAGMA journal_mode') == [('wal',)]


def test_extra_is_local_to_writer_and_reader_is_query_only(tmp_path):
    results, database, raw = fixture(tmp_path)
    reader, _ = api()._open(database, write=False)
    writer, _ = api()._open(database, write=True)
    try:
        assert reader.execute('PRAGMA synchronous').fetchone()[0] == 2
        assert reader.execute('PRAGMA query_only').fetchone()[0] == 1
        assert writer.execute('PRAGMA synchronous').fetchone()[0] == 3
    finally:
        reader.close()
        writer.close()


def test_busy_writer_does_not_publish_and_uncommitted_work_does_not_hide_current(tmp_path):
    results, database, raw = fixture(tmp_path)
    initialize(results, database, raw)
    blocker = sqlite3.connect(database, isolation_level=None)
    blocker.execute('BEGIN IMMEDIATE')
    try:
        assert prepare(results, database, raw, '2'*32, 1) == {'status': 'unavailable', 'reason': 'memory_database_unavailable'}
        assert read(results, database)['revision'] == 1
    finally:
        blocker.rollback()
        blocker.close()


@pytest.mark.parametrize('kind', ['symlink', 'hardlink', 'parent-symlink', 'missing'])
def test_database_namespace_boundary(tmp_path, kind):
    results, database, raw = fixture(tmp_path)
    if kind == 'parent-symlink':
        link = tmp_path / 'parent-link'
        link.symlink_to(tmp_path, target_is_directory=True)
        target = link / database.name
    else:
        target = tmp_path / 'other.sqlite'
        if kind == 'symlink':
            target.symlink_to(database)
        elif kind == 'hardlink':
            os.link(database, target)
    before = file_bytes(database)
    assert read(results, target)['availability'] == 'unavailable'
    assert prepare(results, target, raw)['status'] == 'unavailable'
    assert before == file_bytes(database)
    if kind == 'missing':
        assert not target.exists()


def test_separate_project_scopes_do_not_share_pending_or_current(tmp_path):
    results, database, raw = fixture(tmp_path)
    initialize(results, database, raw)
    other = tmp_path / 'other-results'
    other.mkdir()
    other_raw = encoded(document(other))
    assert read(other, database)['reason'] == 'memory_absent'
    assert prepare(other, database, raw)['reason'] == 'memory_scope_mismatch'
    assert prepare(other, database, other_raw)['status'] == 'prepared'
    assert read(results, database)['revision'] == 1
    assert read(other, database)['reason'] == 'memory_update_pending'
    assert publish(other, database, other_raw)['status'] == 'committed'
    assert read(other, database)['document'] == json.loads(other_raw)
    assert read(results, database)['document'] == json.loads(raw)
    assert sql(database, 'SELECT count(*) FROM ' + CURRENT) == [(2,)]
