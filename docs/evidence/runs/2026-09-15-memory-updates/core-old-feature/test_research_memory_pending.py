"""Pending requests are inspectable without publishing or reserializing them."""
import hashlib
import json

from test_research_memory_store import api, fixture, file_bytes, native, prepare, read


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
