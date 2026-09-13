"""Paging recorded metadata is read access, never source/execution authority."""
from dataclasses import asdict

import pytest

from orze.core.execution_attempts import create_attempt, mark_running, finish_attempt
from orze.engine import cpu_policy_evidence as evidence
from orze.engine.execution_authority import execution_transaction
from orze.engine.execution_catalog import bind_catalog
from orze.idea_lake import IdeaLake


def record(lake, results, task, attempt=None, outcome="completed"):
    """Real SQLite/effect fixture, explicitly not a native worker or closure."""
    folder = results / task
    folder.mkdir(exist_ok=True)
    with execution_transaction(lake, folder) as tx:
        bind_catalog(lake, folder, tx.lease)
        ref = create_attempt(tx.conn, task, "action", attempt or task + "-attempt", {})
        mark_running(tx.conn, ref)
        digest = tx.prepare(ref, {"operation": "paging_metadata_fixture", "outcome": outcome})
        finish_attempt(tx.conn, ref, {"outcome": outcome, "artifact_ids": [],
                                     "observation_ids": [], "effect_receipt_sha256": digest})
    return ref


@pytest.fixture
def lake(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    obj = IdeaLake(tmp_path / "lake.db")
    try:
        yield obj, results
    finally:
        obj.close()


def test_pages_visit_65_current_results_without_window_growth(lake):
    obj, results = lake
    refs = [record(obj, results, f"task-{i:03}") for i in range(65)]
    old = evidence.recorded_evidence(obj, results)
    assert len(old["results"]) == 32 and old["more_available"]
    before = obj.conn.total_changes
    pager = evidence.EvidencePager(obj, results)
    page = pager.read()
    seen, sizes = [], []
    while True:
        view, meta = page["recorded_evidence"], page["evidence_page"]
        assert not view["unavailable"]
        assert meta["mode"] == "scan" and meta["unavailable_seen"] == 0
        seen.extend(item["ref"] for item in view["results"])
        sizes.append(len(view["results"]))
        assert meta["seen"] == len(seen)
        if meta["next_cursor"] is None:
            assert meta["traversal_end"] is True
            # A final page is still not an all-results snapshot.
            assert view["more_available"] is True
            break
        assert meta["traversal_end"] is False
        page = pager.read(cursor=meta["next_cursor"])
    assert sizes == [32, 32, 1]
    assert seen == [asdict(ref) for ref in refs]
    assert obj.conn.total_changes == before
    selected = pager.read(refs=[asdict(refs[0]), asdict(refs[-1])])
    assert selected["evidence_page"]["mode"] == "selection"
    assert selected["evidence_page"]["seen"] == 65
    assert [r["ref"] for r in selected["recorded_evidence"]["results"]] == [asdict(refs[0]), asdict(refs[-1])]
    pager.verify()


def test_empty_pager_does_not_create_execution_schema(lake):
    obj, results = lake
    before = obj.conn.total_changes
    page = evidence.EvidencePager(obj, results).read()
    assert page["recorded_evidence"] == {"results": [], "unavailable": [], "more_available": False}
    assert page["evidence_page"]["traversal_end"] is True
    assert page["evidence_page"]["next_cursor"] is None
    assert obj.conn.total_changes == before
    assert obj.conn.execute("SELECT name FROM sqlite_master WHERE name='execution_attempts'").fetchall() == []


@pytest.mark.parametrize("limit", [True, 0, 33, 1.0, None])
def test_page_limits_are_exact_not_coerced(lake, limit):
    obj, results = lake
    with pytest.raises(ValueError):
        evidence.EvidencePager(obj, results, limit=limit)


def test_cursor_is_single_use_bound_to_its_live_reader(lake):
    obj, results = lake
    for i in range(3):
        record(obj, results, f"task-{i}")
    pager = evidence.EvidencePager(obj, results, limit=1)
    first = pager.read()
    cursor = first["evidence_page"]["next_cursor"]
    assert type(cursor) is str and cursor
    with pytest.raises(evidence.PolicyEvidenceHOLD):
        pager.read(cursor=cursor + "changed")
    with pytest.raises(evidence.PolicyEvidenceHOLD):
        evidence.EvidencePager(obj, results, limit=1).read(cursor=cursor)
    second = pager.read(cursor=cursor)
    assert second["evidence_page"]["page"] == 2
    with pytest.raises(evidence.PolicyEvidenceHOLD):
        pager.read(cursor=cursor)
    with pytest.raises(evidence.PolicyEvidenceHOLD):
        pager.read()


def test_select_rechecks_current_generation_not_historical_ownership(lake):
    obj, results = lake
    old = record(obj, results, "task-a", "old")
    current = record(obj, results, "task-a", "new", outcome="failed")
    pager = evidence.EvidencePager(obj, results)
    assert pager.read()["recorded_evidence"]["results"][0]["ref"] == asdict(current)
    stale = pager.read(refs=[asdict(old)])
    assert stale["recorded_evidence"]["results"] == []
    assert stale["recorded_evidence"]["unavailable"][0]["ref"] == asdict(old)
    selected = pager.read(refs=[asdict(current)])
    assert selected["recorded_evidence"]["results"][0]["outcome"] == "failed"


def test_own_connection_change_invalidates_a_scan(lake):
    obj, results = lake
    record(obj, results, "task-a")
    record(obj, results, "task-b")
    pager = evidence.EvidencePager(obj, results, limit=1)
    first = pager.read()
    record(obj, results, "task-0")
    with pytest.raises(evidence.PolicyEvidenceHOLD):
        pager.read(cursor=first["evidence_page"]["next_cursor"])
    with pytest.raises(evidence.PolicyEvidenceHOLD):
        pager.verify()
