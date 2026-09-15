"""Qualified label pages preserve legacy labels and reject partial authority."""
import sqlite3

import pytest

from orze.reporting.completed_scan import CompletedIdeaScan, CompletedScanUnavailable
from orze.reporting.evidence import authoritative_completed_idea_families


def lake(tmp_path, count=259):
    path = tmp_path / "families.db"
    keys = frozenset(f"idea-{n:04d}" for n in range(count))
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE ideas (idea_id TEXT PRIMARY KEY, status TEXT, approach_family TEXT)")
        conn.execute("CREATE TABLE idea_state (idea_id TEXT PRIMARY KEY, current_state TEXT)")
        conn.execute("CREATE TABLE idea_stage_state (idea_id TEXT, stage TEXT, current_state TEXT, PRIMARY KEY(idea_id,stage))")
        for index, key in enumerate(keys):
            family = (None, " Data ", "a-b_3", "bad\nfamily", "x" * 65)[index % 5]
            conn.execute("INSERT INTO ideas VALUES (?, 'completed', ?)", (key, family))
            conn.execute("INSERT INTO idea_state VALUES (?, 'COMPLETE')", (key,))
    return path, keys


@pytest.mark.parametrize("size", [1, 2, 128, 256])
def test_ordered_pages_preserve_complete_legacy_labels_and_release_readers(tmp_path, size):
    path, ids = lake(tmp_path)
    expected, reason = authoritative_completed_idea_families(path)
    assert reason == "authoritative_lifecycle_loaded"
    seen = []
    with CompletedIdeaScan(path, page_size=size) as scan:
        for page in scan.family_pages(ids):
            assert 0 < len(page) <= size
            assert not scan._conn.in_transaction
            with sqlite3.connect(path, timeout=0) as writer:
                writer.execute("BEGIN EXCLUSIVE")
                writer.rollback()
            seen.extend(page.items())
            page.clear()  # The consumer owns a distinct page.
    assert seen == [(key, expected[key]) for key in ids]
    assert scan._conn is None


def test_legacy_ids_are_not_restricted_to_the_new_proposal_id_grammar(tmp_path):
    path, _ = lake(tmp_path, 0)
    ids = frozenset({"ok", "汉", "contains spaces", "x" * 200})
    with sqlite3.connect(path) as conn:
        conn.executemany("INSERT INTO ideas VALUES (?, 'completed', 'data')", [(key,) for key in ids])
        conn.executemany("INSERT INTO idea_state VALUES (?, 'COMPLETE')", [(key,) for key in ids])
    with CompletedIdeaScan(path) as scan:
        assert list(scan.family_pages(ids)) == [{key: "data" for key in ids}]


@pytest.mark.parametrize("ids", [[], (), set(), "idea-x", None, frozenset({""}), frozenset({"a/b"}), frozenset({b"binary"})])
def test_invalid_or_mutable_requests_never_publish_labels(tmp_path, ids):
    path, _ = lake(tmp_path, 1)
    with pytest.raises(CompletedScanUnavailable, match="idea_ids_invalid"):
        with CompletedIdeaScan(path) as scan:
            list(scan.family_pages(ids))


@pytest.mark.parametrize("damage", ["missing", "status", "state", "stage", "null_stage"])
def test_missing_or_noncomplete_late_page_discards_whole_scan(tmp_path, damage):
    path, ids = lake(tmp_path)
    key = list(ids)[-1]
    with sqlite3.connect(path) as conn:
        if damage == "missing":
            conn.execute("DELETE FROM ideas WHERE idea_id=?", (key,))
        elif damage == "status":
            conn.execute("UPDATE ideas SET status='failed' WHERE idea_id=?", (key,))
        elif damage == "state":
            conn.execute("UPDATE idea_state SET current_state='FAILED' WHERE idea_id=?", (key,))
        else:
            conn.execute("INSERT INTO idea_stage_state VALUES (?, 'training', ?)", (key, None if damage == "null_stage" else "FAILED"))
    with pytest.raises(CompletedScanUnavailable, match="family_evidence_incomplete"):
        with CompletedIdeaScan(path) as scan:
            pages = scan.family_pages(ids)
            assert len(next(pages)) == 128
            list(pages)
    assert scan._conn is None


@pytest.mark.parametrize("mode", ["changed", "restored", "rollback", "after_exhaustion"])
def test_labels_cannot_survive_a_later_committed_change(tmp_path, mode):
    path, ids = lake(tmp_path)
    def consume():
        with CompletedIdeaScan(path) as scan:
            pages = scan.family_pages(ids)
            assert len(next(pages)) == 128
            if mode == "after_exhaustion":
                list(pages)
            with sqlite3.connect(path, timeout=0) as writer:
                writer.execute("UPDATE ideas SET approach_family='changed'")
                if mode == "rollback":
                    writer.rollback()
                elif mode == "restored":
                    writer.commit()
                    writer.execute("UPDATE ideas SET approach_family='restored'")
            if mode != "after_exhaustion":
                list(pages)
    if mode == "rollback":
        consume()
    else:
        with pytest.raises(CompletedScanUnavailable, match="scan_changed"):
            consume()


def test_empty_request_early_exit_and_both_enumeration_modes(tmp_path):
    path, ids = lake(tmp_path)
    with CompletedIdeaScan(path) as scan:
        assert list(scan.family_pages(frozenset())) == []
        with pytest.raises(CompletedScanUnavailable, match="scan_reused"):
            list(scan.pages())
    with CompletedIdeaScan(path) as scan:
        list(scan.pages())
        with pytest.raises(CompletedScanUnavailable, match="scan_reused"):
            list(scan.family_pages(ids))
    with pytest.raises(CompletedScanUnavailable, match="scan_incomplete"):
        with CompletedIdeaScan(path) as scan:
            next(scan.family_pages(ids))


@pytest.mark.parametrize("damage", ["family_column", "duplicate_other_identity"])
def test_family_query_and_global_schema_fail_closed(tmp_path, damage):
    path, ids = lake(tmp_path)
    with sqlite3.connect(path) as conn:
        if damage == "family_column":
            conn.execute("ALTER TABLE ideas DROP COLUMN approach_family")
        else:
            conn.execute("ALTER TABLE ideas RENAME TO original_ideas")
            conn.execute("CREATE TABLE ideas AS SELECT * FROM original_ideas")
            conn.executemany("INSERT INTO ideas VALUES ('unused', 'failed', 'data')", [(), ()])
    with pytest.raises(CompletedScanUnavailable, match="database_invalid"):
        with CompletedIdeaScan(path) as scan:
            list(scan.family_pages(ids))
