"""Pending requests are inspectable without publishing or reserializing them."""
import hashlib
import json
import os
import shutil

import pytest

from test_research_memory_store import (
    CURRENT, PENDING, Proxy, api, child, encoded, fixture, file_bytes, initialize,
    native, prepare, publish, read, sql,
)


def test_pending_inspector_preserves_exact_prepared_bytes_without_writes(tmp_path):
    results, database, raw = fixture(tmp_path)
    raw = b' \n' + json.dumps(json.loads(raw), indent=2).encode() + b'\n '
    assert prepare(results, database, raw)['status'] == 'prepared'
    before, execution = file_bytes(database), native(database)
    assert read(results, database)['reason'] == 'memory_update_pending'
    value = api().inspect_pending_memory(results, database=database)
    assert value == {'availability': 'prepared_update', 'revision': 1,
        'operation_id': '1'*32, 'predecessor': None,
        'document_sha256': hashlib.sha256(raw).hexdigest(), 'document_json': raw.decode()}
    assert value['document_sha256'] != hashlib.sha256(
        json.dumps(json.loads(value['document_json'])).encode()).hexdigest()
    assert file_bytes(database) == before and native(database) == execution
    assert read(results, database)['reason'] == 'memory_update_pending'


@pytest.mark.parametrize('state', ['empty', 'current', 'another_project'])
def test_inspector_does_not_create_or_repair_absent_pending(tmp_path, state):
    results, database, raw = fixture(tmp_path)
    if state == 'current':
        initialize(results, database, raw)
    elif state == 'another_project':
        assert prepare(results, database, raw)['status'] == 'prepared'
        results = tmp_path / 'another-project'
    before = file_bytes(database)
    assert api().inspect_pending_memory(results, database=database) == {
        'availability': 'unavailable', 'reason': 'memory_pending_absent'}
    assert file_bytes(database) == before


def test_cold_process_can_publish_original_predecessor_and_digest(tmp_path):
    results, database, raw = fixture(tmp_path)
    initialize(results, database, raw)
    raw = b'\n  ' + raw + b' \n'
    assert prepare(results, database, raw, '2'*32, 1)['status'] == 'prepared'
    script = '''
import json,sys
from orze.research.memory_store import inspect_pending_memory,publish_memory_update
value=inspect_pending_memory(sys.argv[1],database=sys.argv[2])
print(json.dumps(value))
print(json.dumps(publish_memory_update(sys.argv[1],database=sys.argv[2],
    operation_id=value['operation_id'],expected_revision=value['predecessor'],
    document_sha256=value['document_sha256'])))
'''
    before = native(database)
    process = child(script, results, database, cwd=tmp_path)
    assert process.returncode == 0, process.stderr
    intent, result = map(json.loads, process.stdout.splitlines())
    assert intent['document_json'].encode() == raw and intent['predecessor'] == 1
    assert result['status'] == 'committed' and result['revision'] == 2
    assert read(results, database)['document_sha256'] == hashlib.sha256(raw).hexdigest()
    assert native(database) == before


@pytest.mark.parametrize('query,args,reason', [
    ('UPDATE '+PENDING+' SET predecessor=2,revision=3', (), 'memory_pending_predecessor_mismatch'),
    ('DELETE FROM '+CURRENT, (), 'memory_pending_predecessor_mismatch'),
    ('UPDATE '+CURRENT+' SET document_sha256=?', ('b'*64,), 'memory_store_row_invalid'),
    ('UPDATE '+PENDING+' SET document_sha256=?', ('b'*64,), 'memory_store_row_invalid'),
    ('UPDATE '+PENDING+' SET document_json=?', (b'blob',), 'memory_store_row_invalid'),
    ('DROP TABLE '+CURRENT, (), 'memory_store_schema_incomplete'),
    ('ALTER TABLE '+PENDING+' ADD COLUMN extra TEXT', (), 'memory_store_schema_invalid'),
])
def test_invalid_pending_or_predecessor_is_unavailable_without_repair(tmp_path, query, args, reason):
    results, database, raw = fixture(tmp_path)
    initialize(results, database, raw)
    assert prepare(results, database, raw, '2'*32, 1)['status'] == 'prepared'
    sql(database, query, args)
    before = file_bytes(database)
    assert api().inspect_pending_memory(results, database=database) == {
        'availability': 'unavailable', 'reason': reason}
    assert file_bytes(database) == before


@pytest.mark.parametrize('change,reason', [('remove', 'memory_record_removal_refused'),
    ('rewrite', 'memory_record_rewrite_refused'), ('retarget', 'memory_source_retarget_refused')])
def test_inspector_rechecks_retention_even_for_manually_corrupted_pending(tmp_path, change, reason):
    results, database, raw = fixture(tmp_path)
    initialize(results, database, raw)
    assert prepare(results, database, raw, '2'*32, 1)['status'] == 'prepared'
    value = json.loads(raw)
    if change == 'remove':
        value['entries'] = []
    elif change == 'rewrite':
        value['entries'][0]['claim'] = 'Hide the earlier counterexample.'
    else:
        value['entries'][0]['sources'][0]['idea_id'] = 'idea-different'
    changed = encoded(value)
    sql(database, 'UPDATE '+PENDING+' SET document_json=?,document_sha256=?',
        (changed.decode(), hashlib.sha256(changed).hexdigest()))
    before = file_bytes(database)
    assert api().inspect_pending_memory(results, database=database)['reason'] == reason
    assert file_bytes(database) == before


@pytest.mark.parametrize('mode', ['rollback', 'close'])
def test_inspector_cleanup_failure_remains_unavailable(tmp_path, monkeypatch, mode):
    results, database, raw = fixture(tmp_path)
    assert prepare(results, database, raw)['status'] == 'prepared'
    original = api()._open
    def wrapped(database, *, write):
        assert write is False
        connection, fence = original(database, write=write)
        return Proxy(connection, mode), fence
    monkeypatch.setattr(api(), '_open', wrapped)
    before = file_bytes(database)
    assert api().inspect_pending_memory(results, database=database) == {
        'availability': 'unavailable', 'reason': 'memory_cleanup_unavailable'}
    assert file_bytes(database) == before


def test_inspector_rejects_observed_namespace_replacement(tmp_path, monkeypatch):
    results, database, raw = fixture(tmp_path)
    assert prepare(results, database, raw)['status'] == 'prepared'
    original = api()._row
    def replaced(*args):
        value = original(*args)
        if args[1] == CURRENT:
            parked = database.with_name('parked.sqlite')
            database.rename(parked)
            shutil.copyfile(parked, database)
        return value
    monkeypatch.setattr(api(), '_row', replaced)
    assert api().inspect_pending_memory(results, database=database)['reason'] == 'memory_database_namespace_changed'


@pytest.mark.parametrize('kind', ['symlink', 'hardlink', 'parent_symlink', 'missing'])
def test_inspector_namespace_boundary(tmp_path, kind):
    results, database, raw = fixture(tmp_path)
    assert prepare(results, database, raw)['status'] == 'prepared'
    target = tmp_path / 'other.sqlite'
    if kind == 'symlink':
        target.symlink_to(database)
    elif kind == 'hardlink':
        os.link(database, target)
    elif kind == 'parent_symlink':
        target.symlink_to(tmp_path, target_is_directory=True)
        target = target / database.name
    before = file_bytes(database)
    assert api().inspect_pending_memory(results, database=target)['availability'] == 'unavailable'
    assert file_bytes(database) == before


def test_inspector_never_converts_wal_database(tmp_path):
    results, database, raw = fixture(tmp_path)
    assert prepare(results, database, raw)['status'] == 'prepared'
    sql(database, 'PRAGMA journal_mode=WAL')
    assert api().inspect_pending_memory(results, database=database)['reason'] == 'memory_delete_journal_required'
    assert sql(database, 'PRAGMA journal_mode') == [('wal',)]
