"""Real SQLite historical pages; synthetic records are not native proof."""
import json
import sqlite3
from types import SimpleNamespace

import pytest

from orze.core import cpu_proposal_requests as store
from orze.engine import cpu_proposals as proposals
from orze.engine.cpu_policy_evidence import EvidencePager, PolicyEvidenceHOLD
from orze.idea_lake import IdeaLake
from test_cpu_proposal_requests import record


@pytest.fixture
def catalog(tmp_path):
    results = tmp_path / 'results'
    results.mkdir()
    lake = IdeaLake(tmp_path / 'requests.db')
    try:
        yield lake, results, tmp_path
    finally:
        lake.close()


def populate(catalog, count=3):
    lake, results, root = catalog
    statuses = ('inserted', 'already_present_exact', 'config_duplicate', 'conflict')
    lake.conn.execute('BEGIN IMMEDIATE')
    for index in reversed(range(count)):
        store.insert_request(lake.conn, record(root, key=f'r-{index:03}', status=statuses[index % 4]))
    lake.conn.commit()


def pager(catalog, limit=32):
    lake, results, _ = catalog
    guard = EvidencePager(lake, results)
    guard.read()
    return proposals.ProposalPager(lake, results, limit=limit, guard=guard), guard


def test_65_historical_outcomes_traverse_without_growing_window_or_writes(catalog):
    populate(catalog, 65)
    lake, results, _ = catalog
    before = list(lake.conn.iterdump())
    legacy = proposals.recorded_proposals(lake, results)
    assert len(legacy['results']) == 32 and legacy['more_available'] is True
    scan, _ = pager(catalog)
    page = scan.read()
    seen, sizes = [], []
    while True:
        values, meta = page['recorded_proposals'], page['proposal_page']
        seen.extend(values['results'])
        sizes.append(len(values['results']))
        assert meta['seen'] == len(seen)
        assert meta['missing_request_ids'] == [] and meta['mode'] == 'scan'
        if meta['next_cursor'] is None:
            assert meta['traversal_end'] is True and values['more_available'] is True
            break
        page = scan.read(cursor=meta['next_cursor'])
    assert sizes == [32, 32, 1]
    assert [r['request_id'] for r in seen] == [f'r-{i:03}' for i in range(65)]
    assert {r['status'] for r in seen} == {'inserted', 'already_present_exact', 'config_duplicate', 'conflict'}
    assert list(lake.conn.iterdump()) == before
    assert proposals.recorded_proposals(lake, results) == legacy


def test_empty_scan_and_missing_selection_do_not_create_schema(catalog):
    lake, _, _ = catalog
    before = list(lake.conn.iterdump())
    scan, _ = pager(catalog)
    first = scan.read()
    assert first['recorded_proposals'] == {'results': [], 'more_available': False}
    selected = scan.read(request_ids=['absent'])
    assert selected['recorded_proposals'] == {'results': [], 'more_available': True}
    assert selected['proposal_page']['missing_request_ids'] == ['absent']
    assert list(lake.conn.iterdump()) == before


def test_selection_preserves_scan_position_and_cursors_are_channel_specific(catalog):
    populate(catalog)
    scan, evidence = pager(catalog, 1)
    first = scan.read()
    token = first['proposal_page']['next_cursor']
    selected = scan.read(request_ids=['r-002', 'absent', 'r-000'])
    assert [r['request_id'] for r in selected['recorded_proposals']['results']] == ['r-002', 'r-000']
    assert selected['proposal_page']['missing_request_ids'] == ['absent']
    assert selected['proposal_page']['seen'] == 1 and selected['proposal_page']['next_cursor'] == token
    with pytest.raises(PolicyEvidenceHOLD):
        evidence.read(cursor=token)
    other, _ = pager(catalog, 1)
    with pytest.raises(proposals.ProposalHOLD):
        other.read(cursor=token)
    second = scan.read(cursor=token)
    assert second['recorded_proposals']['results'][0]['request_id'] == 'r-001'
    with pytest.raises(proposals.ProposalHOLD):
        scan.read(cursor=token)


@pytest.mark.parametrize('mutation', ['peer_commit', 'local_rollback'])
def test_shared_revision_rejects_sql_change_even_if_records_are_equal(catalog, mutation):
    populate(catalog)
    lake, _, _ = catalog
    scan, guard = pager(catalog, 1)
    first = scan.read()
    before = list(lake.conn.iterdump())
    if mutation == 'peer_commit':
        with sqlite3.connect(lake.db_path) as peer:
            peer.execute('UPDATE cpu_proposal_requests SET record_json=record_json')
    else:
        lake.conn.execute('BEGIN IMMEDIATE')
        lake.conn.execute('UPDATE cpu_proposal_requests SET record_json=record_json')
        lake.conn.rollback()
    with pytest.raises(proposals.ProposalHOLD):
        scan.read(cursor=first['proposal_page']['next_cursor'])
    with pytest.raises(PolicyEvidenceHOLD):
        guard.verify()
    assert list(lake.conn.iterdump()) == before


@pytest.mark.parametrize('limit', [None, True, 0, 33, 1.0])
def test_reader_limits_are_exact(catalog, limit):
    with pytest.raises(ValueError):
        pager(catalog, limit)


@pytest.mark.parametrize('ids', [[], ['r-000', 'r-000'], [True], ['bad/name']])
def test_bad_selected_keys_do_not_change_scan(catalog, ids):
    populate(catalog)
    scan, _ = pager(catalog, 1)
    first = scan.read()
    with pytest.raises(proposals.ProposalHOLD):
        scan.read(request_ids=ids)
    assert scan.read(cursor=first['proposal_page']['next_cursor'])['proposal_page']['page'] == 2


def test_lookahead_is_fully_decoded_not_hidden_as_more_available(catalog):
    populate(catalog)
    lake, _, _ = catalog
    lake.conn.execute("UPDATE cpu_proposal_requests SET record_json='{}' WHERE request_id='r-001'")
    lake.conn.commit()
    scan, _ = pager(catalog, 1)
    before = list(lake.conn.iterdump())
    with pytest.raises(proposals.ProposalHOLD):
        scan.read()
    assert list(lake.conn.iterdump()) == before


def test_actual_database_binding_cannot_be_replaced_by_historical_record(catalog):
    lake, _, root = catalog
    value = record(root)
    value['database'] = str(root / 'other.db')
    value['source_snapshot']['database'] = value['database']
    lake.conn.execute('BEGIN IMMEDIATE')
    store.insert_request(lake.conn, store.seal_record(value))
    lake.conn.commit()
    scan, _ = pager(catalog)
    with pytest.raises(proposals.ProposalHOLD, match='database'):
        scan.read()


def test_history_does_not_revalidate_old_sources_or_authorize_new_ones(catalog):
    lake, _, root = catalog
    lake.conn.execute('BEGIN IMMEDIATE')
    store.insert_request(lake.conn, record(root, inputs=('old-input',)))
    lake.conn.commit()
    scan, _ = pager(catalog)
    page = scan.read()
    assert page['recorded_proposals']['results'][0]['status'] == 'inserted'
    assert 'source_snapshot' not in json.dumps(page)
    assert lake.conn.execute("SELECT name FROM sqlite_master WHERE name='execution_attempts'").fetchall() == []


def test_forged_guard_does_not_authorize_a_page(catalog):
    lake, results, _ = catalog
    with pytest.raises(proposals.ProposalHOLD):
        proposals.ProposalPager(lake, results, guard=SimpleNamespace(verify=lambda: None))
