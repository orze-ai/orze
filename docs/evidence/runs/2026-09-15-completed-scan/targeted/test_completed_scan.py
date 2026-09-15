"""A global aggregate must enumerate all agreed lifecycle candidates coherently."""
import os
from pathlib import Path
import shutil
import sqlite3

import pytest

from orze.reporting.completed_scan import CompletedIdeaScan, CompletedScanUnavailable
from orze.reporting.evidence import authoritative_completed_idea_ids


def lake(tmp_path, *, primary=True):
    path = tmp_path / "lake.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE ideas (idea_id TEXT " + ("PRIMARY KEY" if primary else "") + ", status TEXT)")
        conn.execute("CREATE TABLE idea_state (idea_id TEXT PRIMARY KEY, current_state TEXT)")
        conn.execute("CREATE TABLE idea_stage_state (idea_id TEXT, stage TEXT, current_state TEXT, PRIMARY KEY(idea_id,stage))")
        for key in ("idea-z", "idea-a", "idea-中", "idea-02", "idea-01"):
            conn.execute("INSERT INTO ideas VALUES (?,'completed')", (key,))
            conn.execute("INSERT INTO idea_state VALUES (?,'COMPLETE')", (key,))
    return path


@pytest.mark.parametrize("page_size", [1, 2, 3, 128, 256])
def test_full_enumeration_matches_legacy_without_holding_reader(tmp_path, page_size):
    path = lake(tmp_path)
    expected, reason = authoritative_completed_idea_ids(path)
    assert reason == "authoritative_lifecycle_loaded"
    with CompletedIdeaScan(path, page_size=page_size) as scan:
        pages = []
        for page in scan.pages():
            assert 0 < len(page) <= page_size
            assert not scan._conn.in_transaction
            # A writer can take the exclusive lock while the consumer qualifies.
            with sqlite3.connect(path, timeout=0) as writer:
                writer.execute("BEGIN EXCLUSIVE")
                writer.rollback()
            pages.append(page)
    assert sum(map(len, pages)) == len(expected)
    assert set().union(*pages) == expected
    assert scan._conn is None


@pytest.mark.parametrize("mutation", ["status", "state", "stage", "drop_stage", "unrelated", "same_value", "replace", "symlink", "hardlink", "delete"])
def test_changes_after_a_page_reject_whole_scan(tmp_path, mutation):
    path = lake(tmp_path)
    with pytest.raises(CompletedScanUnavailable):
        with CompletedIdeaScan(path, page_size=2) as scan:
            pages = scan.pages()
            assert len(next(pages)) == 2
            if mutation in {"status", "state", "stage", "drop_stage", "unrelated", "same_value"}:
                sql = {
                    "status": "UPDATE ideas SET status='failed' WHERE idea_id='idea-z'",
                    "state": "UPDATE idea_state SET current_state='FAILED' WHERE idea_id='idea-z'",
                    "stage": "INSERT INTO idea_stage_state VALUES ('idea-z','training','FAILED')",
                    "drop_stage": "DROP TABLE idea_stage_state",
                    "unrelated": "CREATE TABLE unrelated (x)",
                    "same_value": "UPDATE ideas SET status=status",
                }[mutation]
                with sqlite3.connect(path, timeout=0) as writer:
                    writer.execute(sql)
            elif mutation == "hardlink":
                os.link(path, tmp_path / "linked.db")
            else:
                replacement = tmp_path / "copy.db"
                shutil.copyfile(path, replacement)
                path.unlink()
                if mutation == "replace":
                    replacement.rename(path)
                elif mutation == "symlink":
                    path.symlink_to(replacement)
            list(pages)
    assert scan._conn is None


def test_exit_rechecks_after_last_qualification(tmp_path):
    path = lake(tmp_path)
    with pytest.raises(CompletedScanUnavailable, match="scan_changed"):
        with CompletedIdeaScan(path) as scan:
            assert len(list(scan.pages())) == 1
            with sqlite3.connect(path) as writer:
                writer.execute("UPDATE ideas SET status='failed'")


def test_rollback_does_not_invalidate_and_early_exit_is_unavailable(tmp_path):
    path = lake(tmp_path)
    with CompletedIdeaScan(path, page_size=2) as scan:
        pages = scan.pages()
        first = next(pages)
        with sqlite3.connect(path) as writer:
            writer.execute("UPDATE ideas SET status='failed'")
            writer.rollback()
        assert len(first) + sum(map(len, pages)) == 5
    with pytest.raises(CompletedScanUnavailable, match="scan_incomplete"):
        with CompletedIdeaScan(path, page_size=2) as scan:
            next(scan.pages())
    assert scan._conn is None


@pytest.mark.parametrize("primary", [False, True])
def test_global_schema_validation_rejects_non_candidate_identity_corruption(tmp_path, primary):
    path = lake(tmp_path, primary=primary)
    with sqlite3.connect(path) as writer:
        if primary:
            writer.execute("INSERT INTO ideas VALUES (NULL,'failed')")
        else:
            writer.executemany("INSERT INTO ideas VALUES (?,'failed')", [("bad",), ("bad",)])
    with pytest.raises(CompletedScanUnavailable, match="database_invalid"):
        CompletedIdeaScan(path)


def test_candidate_rules_and_historical_missing_stages_match_legacy(tmp_path):
    path = lake(tmp_path)
    with sqlite3.connect(path) as writer:
        writer.execute("INSERT INTO idea_stage_state VALUES ('idea-z','training','FAILED')")
        writer.execute("INSERT INTO idea_stage_state VALUES ('idea-a','evaluation','SKIPPED')")
        writer.execute("UPDATE idea_state SET current_state='FAILED' WHERE idea_id='idea-中'")
        for key in (".", "..", "", "a/b", b"binary", "ok"):
            writer.execute("INSERT INTO ideas VALUES (?,'completed')", (key,))
            writer.execute("INSERT INTO idea_state VALUES (?,'COMPLETE')", (key,))
    expected, _ = authoritative_completed_idea_ids(path)
    with CompletedIdeaScan(path, page_size=1) as scan:
        assert set().union(*scan.pages()) == expected == {"idea-a", "idea-01", "idea-02", "ok"}
    with sqlite3.connect(path) as writer:
        writer.execute("DROP TABLE idea_stage_state")
    expected, _ = authoritative_completed_idea_ids(path)
    with CompletedIdeaScan(path, page_size=2) as scan:
        assert set().union(*scan.pages()) == expected


@pytest.mark.parametrize("size", [0, -1, True, 257, 1.0, "2"])
def test_invalid_page_size_never_opens_database(tmp_path, size):
    with pytest.raises(CompletedScanUnavailable, match="page_size_invalid"):
        CompletedIdeaScan(tmp_path / "absent.db", page_size=size)
    assert not (tmp_path / "absent.db").exists()


def test_scan_reuse_and_closed_reader_rejected(tmp_path):
    path = lake(tmp_path)
    with CompletedIdeaScan(path) as scan:
        list(scan.pages())
        with pytest.raises(CompletedScanUnavailable, match="scan_reused"):
            list(scan.pages())
    with pytest.raises(CompletedScanUnavailable, match="scan_changed"):
        scan.verify()


def test_missing_database_does_not_create_it(tmp_path):
    with pytest.raises(CompletedScanUnavailable, match="database_unavailable"):
        CompletedIdeaScan(tmp_path / "absent.db")
    assert list(tmp_path.iterdir()) == []
