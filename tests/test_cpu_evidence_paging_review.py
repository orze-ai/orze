"""Independent new paging requirements, using real metadata/effect storage.

No worker, supervisor, native closure, provider or GPU is synthesized or run.
The temporary attempts below deliberately establish only the recorded-metadata
reader's contract, not execution/source authority or scientific convergence.
Controlled SQL/file changes are denial probes against an already opened view.
"""
from dataclasses import asdict
import sqlite3

import pytest

from orze.core.execution_attempts import create_attempt, finish_attempt, mark_running
from orze.engine import cpu_policy_evidence as evidence
from orze.engine.execution_authority import execution_transaction
from orze.engine.execution_catalog import bind_catalog
from orze.idea_lake import IdeaLake
from test_cpu_policy_evidence import _recorded_observations


@pytest.fixture
def project(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(tmp_path / "lake.db")
    try:
        yield lake, results
    finally:
        lake.close()


def _running(lake, results, task, attempt=None):
    folder = results / task
    folder.mkdir(exist_ok=True)
    with execution_transaction(lake, folder) as tx:
        bind_catalog(lake, folder, tx.lease)
        ref = create_attempt(tx.conn, task, "action", attempt or task + "-attempt", {})
        mark_running(tx.conn, ref)
    return ref


def _finish(lake, results, ref):
    with execution_transaction(lake, results / ref.task_id) as tx:
        digest = tx.prepare(ref, {"operation": "paging_metadata_fixture"})
        finish_attempt(tx.conn, ref, {"outcome": "completed", "artifact_ids": [],
            "observation_ids": [], "effect_receipt_sha256": digest})
    return ref


def _completed(lake, results, task, attempt=None):
    return _finish(lake, results, _running(lake, results, task, attempt))


def _page(value):
    assert set(value) == {"recorded_evidence", "evidence_page"}
    assert set(value["recorded_evidence"]) == {"results", "unavailable", "more_available"}
    meta = value["evidence_page"]
    assert set(meta) == {"schema", "scan_id", "mode", "page", "seen",
                         "traversal_end", "unavailable_seen", "next_cursor"}
    assert type(meta["schema"]) is int and meta["schema"] == 1
    assert type(meta["scan_id"]) is str and meta["scan_id"]
    assert meta["mode"] in ("scan", "selection")
    assert all(type(meta[k]) is int and meta[k] >= 0
               for k in ("page", "seen", "unavailable_seen"))
    assert type(meta["traversal_end"]) is bool
    assert meta["next_cursor"] is None or (
        type(meta["next_cursor"]) is str and meta["next_cursor"])
    return meta


def _refs(value):
    return [row["ref"] for row in value["recorded_evidence"]["results"]]


def _unavailable_selection(pager, ref):
    # Both typed rejection and a lossless explicit-unavailable result deny the
    # requested occurrence. No generic exception or silently empty page passes.
    try:
        selected = pager.read(refs=[asdict(ref)])
    except evidence.PolicyEvidenceHOLD:
        return
    _page(selected)
    assert _refs(selected) == []
    assert [row["ref"] for row in selected["recorded_evidence"]["unavailable"]] == [asdict(ref)]
    assert selected["recorded_evidence"]["unavailable"][0]["reason"]


def test_empty_pager_and_verify_do_not_create_schema_or_write(project):
    lake, results = project
    before = lake.conn.execute("SELECT type,name,sql FROM main.sqlite_master ORDER BY name").fetchall()
    changes = lake.conn.total_changes
    pager = evidence.EvidencePager(lake, results)
    value = pager.read()
    meta = _page(value)
    pager.verify()
    assert value["recorded_evidence"] == {"results": [], "unavailable": [], "more_available": False}
    assert meta["mode"] == "scan" and meta["seen"] == 0
    assert meta["traversal_end"] is True and meta["next_cursor"] is None
    assert lake.conn.total_changes == changes
    assert lake.conn.execute("SELECT type,name,sql FROM main.sqlite_master ORDER BY name").fetchall() == before


def test_thirty_third_current_result_is_accessible_without_duplicate_or_write(project):
    lake, results = project
    refs = [_completed(lake, results, "task-%02d" % i) for i in range(33)]
    changes = lake.conn.total_changes
    pager = evidence.EvidencePager(lake, results)
    first = pager.read()
    one = _page(first)
    assert _refs(first) == [asdict(ref) for ref in refs[:32]]
    assert one["seen"] == 32 and one["traversal_end"] is False
    second = pager.read(cursor=one["next_cursor"])
    two = _page(second)
    assert _refs(second) == [asdict(refs[32])]
    assert two["scan_id"] == one["scan_id"] and two["page"] == one["page"] + 1
    assert two["seen"] == 33 and two["unavailable_seen"] == 0
    assert two["traversal_end"] is True and two["next_cursor"] is None
    pager.verify()
    assert lake.conn.total_changes == changes


def test_selection_is_detached_and_does_not_advance_or_consume_scan_cursor(project):
    lake, results = project
    refs = [_completed(lake, results, task) for task in ("b-source", "z-source")]
    pager = evidence.EvidencePager(lake, results, limit=1)
    first = pager.read()
    cursor = first["evidence_page"]["next_cursor"]
    selected = pager.read(refs=[asdict(refs[1])])
    meta = _page(selected)
    assert meta["mode"] == "selection" and meta["seen"] == first["evidence_page"]["seen"]
    assert _refs(selected) == [asdict(refs[1])]
    selected["recorded_evidence"]["results"][0]["ref"]["generation"] = 999
    first["evidence_page"]["seen"] = 999
    assert _refs(pager.read(refs=[asdict(refs[1])])) == [asdict(refs[1])]
    following = pager.read(cursor=cursor)
    assert _refs(following) == [asdict(refs[1])]
    assert following["evidence_page"]["seen"] == 2


def test_new_current_attempt_does_not_make_old_terminal_selectable(project):
    lake, results = project
    old = _completed(lake, results, "source")
    _running(lake, results, "source", "source-new-running")
    pager = evidence.EvidencePager(lake, results)
    first = pager.read()
    assert _refs(first) == []
    assert first["evidence_page"]["traversal_end"] is True
    _unavailable_selection(pager, old)


@pytest.mark.parametrize("change", ["sorts_before_cursor", "running_becomes_terminal", "new_generation"])
def test_peer_changes_invalidate_read_view_instead_of_skipping_evidence(project, change):
    lake, results = project
    _completed(lake, results, "b-source")
    _completed(lake, results, "z-source")
    pending = _running(lake, results, "a-pending") if change == "running_becomes_terminal" else None
    peer = IdeaLake(lake.db_path)
    try:
        pager = evidence.EvidencePager(lake, results, limit=1)
        cursor = pager.read()["evidence_page"]["next_cursor"]
        if change == "sorts_before_cursor":
            _completed(peer, results, "a-new")
        elif change == "running_becomes_terminal":
            _finish(peer, results, pending)
        else:
            _running(peer, results, "b-source", "b-replacement")
        changes = lake.conn.total_changes
        with pytest.raises(evidence.PolicyEvidenceHOLD):
            pager.verify()
        with pytest.raises(evidence.PolicyEvidenceHOLD):
            pager.read(cursor=cursor)
        assert lake.conn.total_changes == changes
    finally:
        peer.close()


@pytest.mark.parametrize("change", ["committed_write", "rolled_back_write", "schema_change"])
def test_same_connection_mutation_is_detected_without_relying_on_data_version(project, change):
    lake, results = project
    _completed(lake, results, "source")
    pager = evidence.EvidencePager(lake, results)
    pager.read()
    if change == "schema_change":
        lake.conn.execute("CREATE TABLE paging_review_extra(value TEXT)")
    else:
        lake.conn.execute("UPDATE main.execution_attempts SET binding_json=binding_json")
    if change == "rolled_back_write":
        lake.conn.rollback()
    else:
        lake.conn.commit()
    with pytest.raises(evidence.PolicyEvidenceHOLD):
        pager.verify()


@pytest.mark.parametrize("fault", ["modified", "another_pager", "consumed", "another_scope"])
def test_cursor_is_one_use_and_owned_by_exact_pager_not_a_reconstructible_claim(project, fault):
    lake, results = project
    _completed(lake, results, "b-source")
    _completed(lake, results, "z-source")
    other_scope = results.parent / "other-results"
    other_scope.mkdir()
    pager = evidence.EvidencePager(lake, results, limit=1)
    cursor = pager.read()["evidence_page"]["next_cursor"]
    if fault == "modified":
        cursor += "changed"
    elif fault == "another_pager":
        pager = evidence.EvidencePager(lake, results, limit=1)
        pager.read()
    elif fault == "consumed":
        pager.read(cursor=cursor)
    with pytest.raises(evidence.PolicyEvidenceHOLD):
        if fault == "another_scope":
            pager = evidence.EvidencePager(lake, other_scope, limit=1)
        pager.read(cursor=cursor)


def test_read_without_cursor_cannot_silently_restart_existing_scan(project):
    lake, results = project
    _completed(lake, results, "source")
    pager = evidence.EvidencePager(lake, results)
    pager.read()
    with pytest.raises(evidence.PolicyEvidenceHOLD):
        pager.read()


def test_same_lake_object_with_replaced_connection_is_not_original_read_view(project):
    lake, results = project
    _completed(lake, results, "source")
    pager = evidence.EvidencePager(lake, results)
    pager.read()
    original = lake.conn
    replacement = sqlite3.connect(str(lake.db_path))
    replacement.row_factory = sqlite3.Row
    try:
        lake.conn = replacement
        with pytest.raises(evidence.PolicyEvidenceHOLD):
            pager.verify()
    finally:
        lake.conn = original
        replacement.close()


def test_same_path_same_database_bytes_with_different_inode_rejects_pager(project):
    lake, results = project
    _completed(lake, results, "source")
    lake.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    path = lake.db_path
    # Preserve the original open database and restore its exact path before
    # closing the Lake. The replacement is temporary, same-byte denial input.
    from pathlib import Path
    path = Path(path)
    raw, original_inode = path.read_bytes(), path.stat().st_ino
    pager = evidence.EvidencePager(lake, results)
    pager.read()
    retained = path.with_name("retained-original.db")
    path.rename(retained)
    try:
        path.write_bytes(raw)
        assert path.read_bytes() == raw and path.stat().st_ino != original_inode
        with pytest.raises(evidence.PolicyEvidenceHOLD):
            pager.verify()
    finally:
        if path.exists():
            path.rename(path.with_name("denied-replacement.db"))
        retained.rename(path)


@pytest.mark.parametrize("large", [False, True])
def test_observations_remain_lossless_or_explicitly_unavailable_at_traversal_end(tmp_path, large):
    lake, results, ref, claims = _recorded_observations(tmp_path, large=large)
    try:
        pager = evidence.EvidencePager(lake, results, limit=1)
        value = pager.read()
        meta = _page(value)
        assert meta["traversal_end"] is True and meta["next_cursor"] is None
        if large:
            assert _refs(value) == []
            assert value["recorded_evidence"]["unavailable"] == [
                {"ref": asdict(ref), "reason": "evidence_result_limit"}]
            assert meta["unavailable_seen"] == 1
        else:
            rows = value["recorded_evidence"]["results"][0]["observation_records"]
            assert rows == claims
            assert [r["validation"]["status"] for r in rows] == ["valid", "invalid", "unknown"]
            assert [r["values"]["value"] for r in rows] == [-1, -2, -3]
            assert meta["unavailable_seen"] == 0
    finally:
        lake.close()


def test_recorded_page_is_not_permanent_source_or_file_confirmation(project):
    lake, results = project
    ref = _completed(lake, results, "source")
    pager = evidence.EvidencePager(lake, results)
    first = pager.read()
    assert _refs(first) == [asdict(ref)]
    confirmation = results / ref.task_id / "_execution_effects" / ref.attempt_id / "committed.json"
    confirmation.rename(confirmation.with_name("retained-confirmation.json"))
    _unavailable_selection(pager, ref)
