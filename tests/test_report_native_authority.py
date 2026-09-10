"""Native report rows require the same authority as steering consumers.

These tests exercise real IdeaLake, source artifacts and public update_report.
No admin, manual-result or external-adapter compatibility policy is asserted.
"""

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from orze.idea_lake import IdeaLake
from orze.reporting.evidence import (
    authoritative_completed_idea_ids,
    qualify_authoritative_report_evidence_with_identity,
)
from orze.reporting.leaderboard import update_report


@pytest.fixture
def project(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    cfg = {
        "_project_root": str(tmp_path),
        "results_dir": str(results),
        "idea_lake_db": str(lake.db_path),
        "report": {
            "primary_metric": "quality",
            "sort": "ascending",
            "columns": [{
                "key": "quality", "source": "assessment.json:measurement.quality",
            }],
        },
    }
    try:
        yield SimpleNamespace(results=results, lake=lake, cfg=cfg, ideas={})
    finally:
        lake.close()


def _publish(p, idea_id="idea-candidate", score=0, *, archived=True):
    folder = p.results / idea_id
    folder.mkdir()
    # Exact declared values sort in the opposite order from the raw proxy.
    raw = {"status": "COMPLETED", "quality": -score}
    (folder / "metrics.json").write_text(json.dumps(raw), encoding="utf-8")
    (folder / "assessment.json").write_text(
        json.dumps({"measurement": {"quality": score}}), encoding="utf-8",
    )
    p.ideas[idea_id] = {"title": idea_id, "config": {"seed": 3}}
    if archived:
        p.lake.insert(idea_id, idea_id, "seed: 3", "", status="completed",
                      eval_metrics=raw)
    return folder


def _oracle(p, idea_id="idea-candidate"):
    completed, lifecycle_reason = authoritative_completed_idea_ids(
        Path(p.cfg["idea_lake_db"]),
    )
    _, _, value, reason, identity = (
        qualify_authoritative_report_evidence_with_identity(
            idea_id, p.results,
            {**p.cfg, "_env_ORZE_RESULTS_DIR": str(p.results)}, completed,
        )
    )
    return value, reason, identity, lifecycle_reason


def _assert_not_ranked(p, rows):
    assert rows == []
    payload = json.loads(
        (p.results / "_leaderboard.json").read_text(encoding="utf-8"),
    )
    assert payload["top"] == []
    # Keep the public existing schema: accepted is an integer, not a new list.
    assert payload["evidence_qualification"]["accepted"] == 0


@pytest.mark.parametrize("direction,expected", [
    ("ascending", ["idea-negative", "idea-zero", "idea-positive"]),
    ("descending", ["idea-positive", "idea-zero", "idea-negative"]),
])
def test_native_qualified_zero_negative_and_exact_source_order_are_preserved(
    project, direction, expected,
):
    p = project
    p.cfg["report"]["sort"] = direction
    scores = {"idea-negative": -2, "idea-zero": 0, "idea-positive": 2}
    for idea_id, score in scores.items():
        _publish(p, idea_id, score)
        value, _, identity, reason = _oracle(p, idea_id)
        assert reason == "authoritative_lifecycle_loaded"
        assert value == score and identity

    rows = update_report(p.results, p.ideas, p.cfg, lake=p.lake)

    assert [row["id"] for row in rows] == expected
    assert [row["primary_val"] for row in rows] == [scores[key] for key in expected]
    payload = json.loads((p.results / "_leaderboard.json").read_text(encoding="utf-8"))
    assert [row["idea_id"] for row in payload["top"]] == expected
    assert payload["evidence_qualification"]["accepted"] == 3


@pytest.mark.parametrize("warm_cache", [False, True], ids=["cold", "warm"])
@pytest.mark.parametrize("rejection", ["taint", "watch", "fsm_mirror_conflict"])
def test_native_report_cannot_rank_evidence_rejected_by_shared_authority(
    project, rejection, warm_cache,
):
    p = project
    folder = _publish(p)
    if rejection == "watch":
        p.cfg["managed_run"] = {"require_clean_training_access_log": True}
        (folder / "_access_log.tsv").write_text("", encoding="utf-8")
    assert _oracle(p)[0] == 0
    if warm_cache:
        assert len(update_report(p.results, p.ideas, p.cfg, lake=p.lake)) == 1

    if rejection == "taint":
        metrics_path = folder / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["tainted_leakage"] = True
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    elif rejection == "watch":
        (folder / "_access_log.tsv").write_text(
            "WATCH\t/private/eval\t/private/eval/sample.arrow\n", encoding="utf-8",
        )
    else:
        # Native ranking must require both authorities to agree, not merely
        # prefer the COMPLETE FSM over a contradictory legacy status mirror.
        p.lake.conn.execute(
            "UPDATE ideas SET status='queued' WHERE idea_id='idea-candidate'",
        )
        p.lake.conn.commit()
    value, _, _, lifecycle_reason = _oracle(p)
    assert lifecycle_reason == "authoritative_lifecycle_loaded"
    assert value is None

    rows = update_report(p.results, p.ideas, p.cfg, lake=p.lake)

    _assert_not_ranked(p, rows)


@pytest.mark.parametrize("pass_lake", [False, True], ids=["configured-db", "native-lake"])
def test_native_unknown_id_cannot_authorize_itself_from_completed_artifacts(
    project, pass_lake,
):
    p = project
    _publish(p, archived=False)
    assert _oracle(p)[0] is None
    assert p.lake.get("idea-candidate") is None
    assert p.lake.get_fsm_state("idea-candidate") == "UNKNOWN"
    assert p.lake.conn.execute(
        "SELECT * FROM idea_state WHERE idea_id='idea-candidate'",
    ).fetchone() is None

    rows = update_report(p.results, p.ideas, p.cfg, lake=p.lake if pass_lake else None)

    _assert_not_ranked(p, rows)
    assert p.lake.get("idea-candidate") is None
    assert p.lake.get_fsm_state("idea-candidate") == "UNKNOWN"
    assert p.lake.conn.execute(
        "SELECT * FROM idea_state WHERE idea_id='idea-candidate'",
    ).fetchone() is None


@pytest.mark.parametrize("warm_cache", [False, True], ids=["cold", "warm"])
@pytest.mark.parametrize("database", ["missing", "incompatible"])
def test_explicit_unavailable_native_database_cannot_fall_back_to_local_artifacts(
    project, database, warm_cache,
):
    p = project
    _publish(p)
    if warm_cache:
        assert len(update_report(p.results, p.ideas, p.cfg, lake=None)) == 1
    unavailable = p.results.parent / f"{database}.sqlite3"
    if database == "incompatible":
        with sqlite3.connect(unavailable) as connection:
            connection.execute("CREATE TABLE unrelated (value TEXT)")
    p.cfg["idea_lake_db"] = str(unavailable)
    value, _, _, lifecycle_reason = _oracle(p)
    assert value is None
    assert lifecycle_reason != "authoritative_lifecycle_loaded"
    before = {
        path.name: path.read_bytes()
        for path in unavailable.parent.glob(unavailable.name + "*")
    }

    rows = update_report(p.results, p.ideas, p.cfg, lake=None)

    _assert_not_ranked(p, rows)
    assert {
        path.name: path.read_bytes()
        for path in unavailable.parent.glob(unavailable.name + "*")
    } == before  # No creation, sidecars, bootstrap, migration or database writes.
