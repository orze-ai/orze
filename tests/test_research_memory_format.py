"""Persistent declarations remain bounded, fresh and pending source verification."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def api():
    return importlib.import_module('orze.research.memory_format')


def document(root):
    return {'schema': 1, 'project_scope': hashlib.sha256(str(root.absolute()).encode()).hexdigest(),
            'entries': [{'id': 'counterexample', 'question': 'Does the conclusion survive this counterexample?',
                         'hypothesis': 'The observed condition may invalidate the hypothesis.',
                         'rationale': 'Preserve this counterexample across the next process restart.',
                         'claim': 'An author reported that the hypothesis was refuted.',
                         'claimed_state': 'refuted', 'origin': 'authored',
                         'sources': [{'kind': 'report', 'binding_schema': 1, 'idea_id': 'idea-counterexample',
                                      'binding_sha256': 'a' * 64}]}]}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()


def store(root, value):
    path = root / 'knowledge/research_memory.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded(value))
    return path


def test_restart_retains_claims_without_granting_verified_state(tmp_path):
    value = document(tmp_path)
    original = copy.deepcopy(value['entries'][0])
    value['entries'] = [{**copy.deepcopy(original), 'id': state, 'claimed_state': state}
                        for state in ('confirmed', 'refuted', 'unknown')]
    path = store(tmp_path, value)
    before = path.read_bytes()
    script = '''
import importlib, json, sys
try:
    memory = importlib.import_module('orze.research.memory_format')
except ModuleNotFoundError as exc:
    if exc.name != 'orze.research.memory_format':
        raise
    print(json.dumps({'availability':'format_not_available'}))
else:
    print(json.dumps(memory.read_memory(sys.argv[1])))
'''
    observations = []
    for _ in range(2):
        process = subprocess.run([sys.executable, '-c', script, str(tmp_path)], capture_output=True,
                                 text=True, check=True, cwd=tmp_path,
                                 env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                                          PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src')))
        observed = json.loads(process.stdout)
        assert observed['availability'] == 'pending_verification', observed
        assert observed['document'] == value
        observations.append(observed)
    assert observations[0] == observations[1]
    assert path.read_bytes() == before
    assert not (tmp_path / 'idea-counterexample').exists()
    assert not (tmp_path / 'lake.db').exists()


@pytest.mark.parametrize('change,reason', [
    (lambda v: v.update(schema=2), 'memory_schema_invalid'),
    (lambda v: v.update(schema=True), 'memory_schema_invalid'),
    (lambda v: v.update(execute=True), 'memory_fields_invalid'),
    (lambda v: v.update(project_scope='b' * 64), 'memory_scope_mismatch'),
    (lambda v: v['entries'].append(copy.deepcopy(v['entries'][0])), 'memory_record_id_invalid'),
    (lambda v: v['entries'][0].update(sources=[]), 'memory_claim_sources_missing'),
    (lambda v: v['entries'][0].update(claimed_state=[]), 'memory_claimed_state_invalid'),
    (lambda v: v['entries'][0].update(origin='verified'), 'memory_origin_invalid'),
    (lambda v: v['entries'][0].update(permit='claimed'), 'memory_fields_invalid'),
    (lambda v: v['entries'][0].update(id='../elsewhere'), 'memory_record_id_invalid'),
    (lambda v: v['entries'][0].update(question='界' * 171), 'memory_text_invalid'),
    (lambda v: v['entries'][0].update(claim='bad\x00text'), 'memory_text_invalid'),
    (lambda v: v['entries'][0].update(rationale=' '), 'memory_text_invalid'),
    (lambda v: v['entries'][0]['sources'][0].update(idea_id='../idea-secret'), 'memory_source_id_invalid'),
    (lambda v: v['entries'][0]['sources'][0].update(binding_sha256='bad'), 'memory_source_digest_invalid'),
    (lambda v: v['entries'][0]['sources'][0].update(binding_schema=True), 'memory_binding_schema_invalid'),
    (lambda v: v['entries'][0]['sources'][0].update(binding_schema=2), 'memory_binding_schema_invalid'),
    (lambda v: v['entries'][0]['sources'][0].update(kind='execution'), 'memory_source_kind_invalid'),
    (lambda v: v['entries'][0]['sources'].append(copy.deepcopy(v['entries'][0]['sources'][0])), 'memory_source_duplicate'),
])
def test_bad_record_rejects_whole_document(tmp_path, change, reason):
    value = document(tmp_path)
    change(value)
    path = store(tmp_path, value)
    before = path.read_bytes()
    assert api().read_memory(tmp_path) == {'availability': 'unavailable', 'reason': reason}
    assert path.read_bytes() == before


@pytest.mark.parametrize('capacity', ['records', 'per_record_sources', 'total_sources', 'bytes'])
def test_capacity_boundary_accepts_complete_input_then_rejects_overflow(tmp_path, capacity):
    value = document(tmp_path)
    if capacity == 'records':
        value['entries'] = [{**copy.deepcopy(value['entries'][0]), 'id': 'record-' + str(i)} for i in range(32)]
        extra = copy.deepcopy(value['entries'][0])
        extra['id'] = 'record-overflow'
        reason = 'memory_entry_count_invalid'
    elif capacity in ('per_record_sources', 'total_sources'):
        source = value['entries'][0]['sources'][0]
        value['entries'][0]['sources'] = [{**source, 'idea_id': 'idea-source-' + str(i)} for i in range(8)]
        if capacity == 'total_sources':
            value['entries'] = [{**copy.deepcopy(value['entries'][0]), 'id': 'record-' + str(i)} for i in range(8)]
        reason = 'memory_source_count_invalid'
    else:
        reason = 'memory_size_invalid'
    scope = api().project_scope(tmp_path)
    raw = encoded(value)
    if capacity == 'bytes':
        raw += b' ' * (65536 - len(raw))
    assert api().decode_memory(raw, scope) == value
    if capacity == 'records':
        value['entries'].append(extra)
    elif capacity == 'per_record_sources':
        value['entries'][0]['sources'].append({**source, 'idea_id': 'idea-overflow'})
    elif capacity == 'total_sources':
        extra = copy.deepcopy(value['entries'][0])
        extra.update(id='overflow', sources=extra['sources'][:1])
        value['entries'].append(extra)
    raw = raw + b' ' if capacity == 'bytes' else encoded(value)
    with pytest.raises(api().MemoryUnavailable, match=reason):
        api().decode_memory(raw, scope)


@pytest.mark.parametrize('raw,reason', [
    (b'\xff', 'memory_json_invalid'),
    (b'{"schema":1,"schema":1}', 'memory_duplicate_json_key'),
    (b'{"schema":NaN}', 'memory_nonfinite_json_number'),
    (b'[' * 2000 + b']' * 2000, 'memory_json_invalid'),
])
def test_invalid_json_does_not_produce_partial_memory(tmp_path, raw, reason):
    with pytest.raises(api().MemoryUnavailable, match=reason):
        api().decode_memory(raw, api().project_scope(tmp_path))


@pytest.mark.parametrize('mode', ['absent', 'symlink', 'hardlink', 'fifo', 'directory', 'parent_symlink'])
def test_redirected_or_special_publication_is_unavailable(tmp_path, mode):
    root = tmp_path / 'project'
    path = root / 'knowledge/research_memory.json'
    path.parent.mkdir(parents=True)
    other = tmp_path / 'elsewhere'
    other.mkdir()
    target = other / 'research_memory.json'
    target.write_bytes(encoded(document(root)))
    if mode == 'symlink':
        path.symlink_to(target)
    elif mode == 'hardlink':
        os.link(target, path)
    elif mode == 'fifo':
        os.mkfifo(path)
    elif mode == 'directory':
        path.mkdir()
    elif mode == 'parent_symlink':
        path.parent.rmdir()
        path.parent.symlink_to(other, target_is_directory=True)
    assert api().read_memory(root)['availability'] == 'unavailable'
    assert target.read_bytes() == encoded(document(root))


@pytest.mark.parametrize('kind', ['file', 'parent', 'in_place'])
def test_generation_change_during_read_rejects_whole_document(tmp_path, monkeypatch, kind):
    value = document(tmp_path)
    path = store(tmp_path, value)
    original = os.read
    changed = False

    def read(fd, size):
        nonlocal changed
        raw = original(fd, size)
        if not changed:
            changed = True
            if kind == 'file':
                other = path.with_name('replacement.json')
                other.write_bytes(encoded(value))
                other.replace(path)
            elif kind == 'parent':
                path.parent.rename(tmp_path / 'previous-knowledge')
                store(tmp_path, value)
            else:
                path.write_bytes(encoded(value) + b' ')
        return raw

    monkeypatch.setattr(os, 'read', read)
    assert api().read_memory(tmp_path) == {'availability': 'unavailable', 'reason': 'memory_changed_during_read'}


def test_new_read_observes_current_bytes_and_other_project_cannot_inherit_scope(tmp_path):
    root = tmp_path / 'project'
    value = document(root)
    path = store(root, value)
    first = api().read_memory(root)
    value['entries'][0]['claimed_state'] = 'unknown'
    store(root, value)
    second = api().read_memory(root)
    assert first['document_sha256'] != second['document_sha256']
    assert first['availability'] == second['availability'] == 'pending_verification'
    assert second['document']['entries'][0]['claimed_state'] == 'unknown'
    other = tmp_path / 'other-project'
    store(other, json.loads(path.read_bytes()))
    assert api().read_memory(other) == {'availability': 'unavailable', 'reason': 'memory_scope_mismatch'}


def test_oversized_publication_rejected_before_payload_read(tmp_path, monkeypatch):
    path = store(tmp_path, document(tmp_path))
    path.write_bytes(b' ' * 65537)
    monkeypatch.setattr(os, 'read', lambda *_: pytest.fail('must not read oversized payload'))
    assert api().read_memory(tmp_path) == {'availability': 'unavailable', 'reason': 'memory_size_invalid'}
