"""Public completion authority must not ignore explicitly conflicting stages.

Real IdeaLake and artifact fixtures; contradictory persisted rows model older
writers/imports, not a request for observers to repair lifecycle. Missing stage
history remains compatible. No provider or process boundary is exercised.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from orze.idea_lake import IdeaLake
from orze.reporting.evidence import (
    authoritative_completed_idea_families,
    authoritative_completed_idea_ids,
    authoritative_idea_lifecycle,
)
from orze.reporting.leaderboard import update_report


@pytest.fixture
def project(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(tmp_path / "authority.db")
    p = SimpleNamespace(lake=lake, results=results, ideas={}, cfg={
        "_project_root": str(tmp_path), "results_dir": str(results),
        "idea_lake_db": str(lake.db_path),
        "report": {"primary_metric": "quality", "sort": "descending",
                   "columns": [{"key": "quality", "source": "assessment.json:quality"}]},
    })
    try:
        yield p
    finally:
        lake.close()


def _insert(p, idea_id, family="architecture", score=None):
    p.lake.insert(idea_id, idea_id, "seed: 13", "", status="completed",
                  approach_family=family)
    p.ideas[idea_id] = {"title": idea_id, "config": {"seed": 13}}
    if score is not None:
        folder = p.results / idea_id
        folder.mkdir()
        (folder / "metrics.json").write_text(
            json.dumps({"status": "COMPLETED", "quality": 999 - score}), encoding="utf-8")
        (folder / "assessment.json").write_text(json.dumps({"quality": score}), encoding="utf-8")


def _stage(p, idea_id, stage, state):
    # Seed the exact recorded state, including contradictions a read-only
    # authority must reject. Normal global transition APIs synchronize stages.
    p.lake.conn.execute(
        "INSERT INTO idea_stage_state (idea_id,stage,current_state,updated_at) VALUES (?,?,?,?)",
        (idea_id, stage, state, "2026-09-10T00:00:00"),
    )
    p.lake.conn.commit()


def _read(p, ids):
    db = Path(p.lake.db_path)
    before = (db.read_bytes(), db.stat().st_mtime_ns)
    results = (
        authoritative_completed_idea_ids(db),
        authoritative_completed_idea_families(db),
        authoritative_idea_lifecycle(db, ids),
    )
    assert (db.read_bytes(), db.stat().st_mtime_ns) == before
    assert not db.with_name(db.name + "-journal").exists()
    return results


@pytest.mark.parametrize("stage,state", [
    ("training", "PENDING"), ("training", "IN_PROGRESS"),
    ("training", "FAILED"), ("training", "NOT_STARTED"),
    ("evaluation", "PENDING"), ("evaluation", "IN_PROGRESS"),
    ("evaluation", "FAILED"), ("evaluation", "UNKNOWN_FUTURE_STATE"),
])
def test_completed_ids_and_families_exclude_recorded_stage_conflicts(project, stage, state):
    p = project
    _insert(p, "idea-good", "architecture")
    _insert(p, "idea-conflict", "optimization")
    _stage(p, "idea-conflict", stage, state)

    completed, families, _ = _read(p, ["idea-good"])

    assert completed == ({"idea-good"}, "authoritative_lifecycle_loaded")
    assert families == ({"idea-good": "architecture"}, "authoritative_lifecycle_loaded")
    assert p.lake.get_fsm_state("idea-conflict") == "COMPLETE"
    assert p.lake.get("idea-conflict")["status"] == "completed"
    assert p.lake.get_stage_state("idea-conflict", stage) == state


def test_bounded_lifecycle_request_refuses_any_completed_stage_conflict(project):
    p = project
    _insert(p, "idea-good")
    _insert(p, "idea-conflict")
    _stage(p, "idea-conflict", "training", "COMPLETE")
    _stage(p, "idea-conflict", "evaluation", "FAILED")

    _, _, (lifecycle, reason) = _read(p, ["idea-good", "idea-conflict"])

    assert lifecycle == {}, "Do not silently turn the bad member into an agreed COMPLETE"
    assert reason != "authoritative_lifecycle_loaded"


@pytest.mark.parametrize("history", ["missing_table", "missing_rows", "training_only", "evaluation_only"])
def test_truly_unrecorded_optional_stage_history_remains_compatible(project, history):
    p = project
    _insert(p, "idea-historical")
    if history == "missing_table":
        p.lake.conn.execute("DROP TABLE idea_stage_state")
        p.lake.conn.commit()
    elif history == "training_only":
        _stage(p, "idea-historical", "training", "COMPLETE")
    elif history == "evaluation_only":
        _stage(p, "idea-historical", "evaluation", "COMPLETE")

    completed, families, lifecycle = _read(p, ["idea-historical"])

    assert completed == ({"idea-historical"}, "authoritative_lifecycle_loaded")
    assert families == ({"idea-historical": "architecture"}, "authoritative_lifecycle_loaded")
    assert lifecycle == ({"idea-historical": {"state": "COMPLETE", "family": "architecture"}},
                         "authoritative_lifecycle_loaded")
    if history == "missing_table":
        assert p.lake.conn.execute(
            "SELECT name FROM sqlite_master WHERE name='idea_stage_state'").fetchone() is None
    else:
        expected_rows = 0 if history == "missing_rows" else 1
        assert p.lake.conn.execute("SELECT COUNT(*) FROM idea_stage_state").fetchone()[0] == expected_rows


@pytest.mark.parametrize("evaluation", ["COMPLETE", "SKIPPED"])
def test_explicit_complete_or_training_only_skipped_evaluation_is_compatible(project, evaluation):
    p = project
    _insert(p, "idea-valid")
    _stage(p, "idea-valid", "training", "COMPLETE")
    _stage(p, "idea-valid", "evaluation", evaluation)

    completed, families, lifecycle = _read(p, ["idea-valid"])

    assert completed == ({"idea-valid"}, "authoritative_lifecycle_loaded")
    assert families[0] == {"idea-valid": "architecture"}
    assert lifecycle[0]["idea-valid"]["state"] == "COMPLETE"


@pytest.mark.parametrize("stage", ["training", "evaluation"])
def test_existing_stage_row_with_null_value_is_not_missing_history(project, stage):
    p = project
    _insert(p, "idea-good")
    _insert(p, "idea-null")
    p.lake.conn.execute("DROP TABLE idea_stage_state")
    # Some old/imported schemas do not enforce NOT NULL. Presence, not merely
    # the joined current_state value, distinguishes unrecorded from malformed.
    p.lake.conn.execute("CREATE TABLE idea_stage_state (idea_id TEXT NOT NULL, stage TEXT NOT NULL, "
                        "current_state TEXT, PRIMARY KEY(idea_id,stage))")
    p.lake.conn.execute("INSERT INTO idea_stage_state VALUES ('idea-null',?,NULL)", (stage,))
    p.lake.conn.commit()

    completed, families, lifecycle = _read(p, ["idea-good", "idea-null"])

    assert completed == ({"idea-good"}, "authoritative_lifecycle_loaded")
    assert families == ({"idea-good": "architecture"}, "authoritative_lifecycle_loaded")
    assert lifecycle[0] == {} and lifecycle[1] != "authoritative_lifecycle_loaded"


@pytest.mark.parametrize("conflict", ["legacy_mirror", "global_fsm"])
def test_original_global_and_mirror_agreement_remains_required(project, conflict):
    p = project
    _insert(p, "idea-good")
    _insert(p, "idea-conflict")
    _stage(p, "idea-conflict", "training", "COMPLETE")
    _stage(p, "idea-conflict", "evaluation", "COMPLETE")
    if conflict == "legacy_mirror":
        p.lake.conn.execute("UPDATE ideas SET status='queued' WHERE idea_id='idea-conflict'")
    else:
        p.lake.conn.execute("UPDATE idea_state SET current_state='FAILED' WHERE idea_id='idea-conflict'")
    p.lake.conn.commit()

    completed, families, lifecycle = _read(p, ["idea-good", "idea-conflict"])

    assert completed == ({"idea-good"}, "authoritative_lifecycle_loaded")
    assert families == ({"idea-good": "architecture"}, "authoritative_lifecycle_loaded")
    assert lifecycle[0] == {} and lifecycle[1] != "authoritative_lifecycle_loaded"


@pytest.mark.parametrize("damage", ["missing_stage_column", "duplicate_stage_join", "duplicate_global_join"])
def test_structural_or_duplicate_join_identity_refuses_entire_authority_read(project, damage):
    p = project
    _insert(p, "idea-good")
    _insert(p, "idea-damaged")
    if damage == "duplicate_global_join":
        p.lake.conn.execute("DROP TABLE idea_state")
        p.lake.conn.execute("CREATE TABLE idea_state (idea_id TEXT,current_state TEXT)")
        p.lake.conn.executemany("INSERT INTO idea_state VALUES (?, 'COMPLETE')",
                               [("idea-good",), ("idea-damaged",), ("idea-damaged",)])
    else:
        p.lake.conn.execute("DROP TABLE idea_stage_state")
        if damage == "missing_stage_column":
            p.lake.conn.execute("CREATE TABLE idea_stage_state (idea_id TEXT,stage TEXT)")
        else:
            p.lake.conn.execute("CREATE TABLE idea_stage_state (idea_id TEXT,stage TEXT,current_state TEXT)")
            p.lake.conn.executemany("INSERT INTO idea_stage_state VALUES (?, 'evaluation', 'COMPLETE')",
                                   [("idea-damaged",), ("idea-damaged",)])
    p.lake.conn.commit()

    completed, families, lifecycle = _read(p, ["idea-good", "idea-damaged"])

    assert completed[0] == set() and completed[1] != "authoritative_lifecycle_loaded"
    assert families[0] == {} and families[1] != "authoritative_lifecycle_loaded"
    assert lifecycle[0] == {} and lifecycle[1] != "authoritative_lifecycle_loaded"


def test_native_report_retracts_warm_cached_result_when_recorded_stage_conflicts(project):
    p = project
    _insert(p, "idea-zero", score=0)
    _insert(p, "idea-stale", score=100)
    initial = update_report(p.results, p.ideas, p.cfg, lake=p.lake)
    assert [row["id"] for row in initial] == ["idea-stale", "idea-zero"]
    assert [row["primary_val"] for row in initial] == [100, 0]
    pipeline_counts = p.lake.get_lifecycle_counts()
    assert pipeline_counts == {"COMPLETED": 2}
    _stage(p, "idea-stale", "evaluation", "IN_PROGRESS")

    rows = update_report(p.results, p.ideas, p.cfg, lake=p.lake)

    assert [row["id"] for row in rows] == ["idea-zero"]
    assert rows[0]["primary_val"] == 0
    payload = json.loads((p.results / "_leaderboard.json").read_text(encoding="utf-8"))
    assert [row["idea_id"] for row in payload["top"]] == ["idea-zero"]
    assert payload["evidence_qualification"]["accepted"] == 1
    assert p.lake.get_lifecycle_counts() == pipeline_counts  # Preserve audited-FSM Pipeline basis.
