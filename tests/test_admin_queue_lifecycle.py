"""Admin queue task state comes from native lifecycle, not training metrics.

Every case exercises write_admin_cache and the real GET /api/queue route using
an in-process client. No network listener, GPU, process launcher, provider, or
qualification mock is used. Legacy offline display is retained but is not
native lifecycle authority. Its new authority label is a separate field
contract, not an existing-behavior red.
"""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from orze.admin import server
from orze.core.config import orze_path
from orze.engine.evaluation_retry import request_evaluation_retry
from orze.idea_lake import IdeaLake
from orze.reporting.leaderboard import write_admin_cache


@pytest.fixture
def project(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(str(tmp_path / "authority.db"))
    cfg = {
        "_project_root": str(tmp_path),
        "_orze_dir": str(tmp_path / ".orze"),
        "_env_ORZE_RESULTS_DIR": str(results),
        "results_dir": str(results),
        "idea_lake_db": str(lake.db_path),
        "eval_script": "never-execute-evaluator.py",
        "eval_output": "assessment.json",
        "eval_checkpoint": "checkpoint.pt",
        "report": {
            "primary_metric": "quality", "sort": "ascending",
            "columns": [{"key": "quality", "source": "assessment.json:quality"}],
        },
    }
    monkeypatch.setattr(server, "_cfg", cfg)
    monkeypatch.setattr(server, "_cache", {})
    with TestClient(server.app) as client:
        p = SimpleNamespace(root=tmp_path, results=results, lake=lake,
                            cfg=cfg, ideas={}, client=client)
        try:
            yield p
        finally:
            if p.lake is not None:
                p.lake.close()


def _insert(p, idea_id="idea-task", *, in_lake=True):
    folder = p.results / idea_id
    folder.mkdir()
    # These successful *training* bytes cannot prove downstream task closure.
    (folder / "metrics.json").write_text(
        '{"status":"COMPLETED","quality":999}', encoding="utf-8")
    (folder / "checkpoint.pt").write_bytes(b"successful-training-checkpoint\x00\xff")
    raw = ("**Category**: optimization\n**Parent**: none\n"
           "**Hypothesis**: lifecycle contract\n")
    p.ideas[idea_id] = {
        "title": "task label", "priority": "high", "config": {"seed": 13},
        "raw": raw,
    }
    if in_lake:
        p.lake.insert(idea_id, "task label", "seed: 13", raw,
                      status="queued", priority="high", category="optimization",
                      parent="none", hypothesis="lifecycle contract")
    return folder


def _evaluation_state(p, idea_id, state):
    assert p.lake.reconcile_training_complete(idea_id, "reconcile_admin_training")
    if state != "PENDING":
        assert p.lake.record_stage_transition(
            idea_id, "evaluation", "PENDING", "IN_PROGRESS", "evaluation_launched")
    if state in ("FAILED", "COMPLETE"):
        assert p.lake.record_state_transition(
            idea_id, "IN_PROGRESS", state,
            "evaluation_failed" if state == "FAILED" else "evaluation_completed")
    assert p.lake.get_stage_state(idea_id, "training") == "COMPLETE"
    assert p.lake.get_stage_state(idea_id, "evaluation") == state


def _produce_and_read(p):
    write_admin_cache(p.results, p.ideas, p.cfg)
    cache = json.loads(orze_path(p.cfg, "state", "admin_cache.json").read_text(
        encoding="utf-8"))
    # Exercise the actual file reader on this observation, not a previous TTL
    # result. Cache replay/freshness beyond the writer contract is not claimed.
    server._cache.clear()
    response = p.client.get("/api/queue")
    assert response.status_code == 200
    api = response.json()
    assert {"queue", "total", "total_all", "page", "per_page",
            "total_pages", "counts"}.issubset(api)
    assert isinstance(api["queue"], list)
    assert api["counts"] == cache["queue"]["counts"]
    assert api["total_all"] == cache["queue"]["total_all"]
    assert {row["idea_id"]: row["status"] for row in api["queue"]} == {
        row["idea_id"]: row["status"] for row in cache["queue"]["items"]}
    return cache["queue"], api


def _assert_item_schema(item):
    assert {"idea_id", "title", "priority", "status", "config", "sweep_parent",
            "category", "parent", "hypothesis"}.issubset(item)
    assert isinstance(item["config"], dict)
    assert item["title"] == "task label"
    assert item["priority"] == "high"
    assert item["config"] == {"seed": 13}


@pytest.mark.parametrize("evaluation,expected", [
    ("PENDING", "running"), ("IN_PROGRESS", "running"),
    ("FAILED", "failed"), ("COMPLETE", "completed"),
])
def test_completed_training_does_not_override_evaluation_task_state(
    project, evaluation, expected,
):
    p = project
    _insert(p)
    _evaluation_state(p, "idea-task", evaluation)

    queue, api = _produce_and_read(p)

    assert queue["counts"] == {expected: 1}
    assert api["total"] == api["total_all"] == 1
    assert api["queue"][0]["status"] == expected
    _assert_item_schema(api["queue"][0])


def test_empty_inbox_still_shows_durable_evaluation_retry_after_lake_close(project):
    p = project
    folder = _insert(p, "idea-retry")
    _evaluation_state(p, "idea-retry", "FAILED")
    (folder / "assessment.json").write_text(
        '{"status":"FAILED","error":"old evaluation failed"}', encoding="utf-8")
    original_metrics = (folder / "metrics.json").read_bytes()
    original_checkpoint = (folder / "checkpoint.pt").read_bytes()
    admitted = request_evaluation_retry("idea-retry", p.results, p.cfg, p.lake)
    assert admitted["status"] == "evaluation_retry_pending"
    assert p.lake.get_fsm_state("idea-retry") == "IN_PROGRESS"
    assert p.lake.get_stage_state("idea-retry", "evaluation") == "PENDING"
    p.lake.close()
    p.lake = None
    p.ideas.clear()

    queue, api = _produce_and_read(p)

    assert [item["idea_id"] for item in api["queue"]] == ["idea-retry"]
    assert api["queue"][0]["status"] == "running"
    assert queue["counts"] == {"running": 1}
    assert api["total"] == api["total_all"] == 1
    _assert_item_schema(api["queue"][0])
    assert (folder / "metrics.json").read_bytes() == original_metrics
    assert (folder / "checkpoint.pt").read_bytes() == original_checkpoint


@pytest.mark.parametrize("corruption", ["legacy_conflict", "fsm_missing"])
def test_unknown_lifecycle_isolated_without_hiding_other_completed_task(project, corruption):
    p = project
    for idea_id in ("idea-unknown", "idea-good"):
        _insert(p, idea_id)
        _evaluation_state(p, idea_id, "COMPLETE")
    if corruption == "legacy_conflict":
        p.lake.conn.execute("UPDATE ideas SET status='queued' WHERE idea_id='idea-unknown'")
    else:
        p.lake.conn.execute("DELETE FROM idea_state WHERE idea_id='idea-unknown'")
    p.lake.conn.commit()

    queue, api = _produce_and_read(p)

    assert {item["idea_id"]: item["status"] for item in api["queue"]} == {
        "idea-unknown": "unknown", "idea-good": "completed",
    }
    assert queue["counts"] == {"unknown": 1, "completed": 1}
    filtered = p.client.get("/api/queue", params={"status_filter": "unknown"})
    assert filtered.status_code == 200
    assert [item["idea_id"] for item in filtered.json()["queue"]] == ["idea-unknown"]


def test_missing_declared_database_does_not_create_authority_or_claim_completed(project):
    p = project
    _insert(p)
    missing = p.root / "missing-authority" / "never-create.db"
    p.cfg["idea_lake_db"] = str(missing)

    queue, api = _produce_and_read(p)

    assert api["queue"][0]["status"] == "unknown"
    assert queue["counts"] == {"unknown": 1}
    assert not missing.parent.exists()  # No database, journal, or bootstrap.


def test_artifact_without_native_catalog_row_is_unknown(project):
    p = project
    _insert(p, in_lake=False)

    queue, api = _produce_and_read(p)

    assert api["queue"][0]["status"] == "unknown"
    assert queue["counts"] == {"unknown": 1}


def test_other_native_states_keep_existing_ui_status_and_item_schema(project):
    p = project
    for idea_id in ("idea-queued", "idea-claimed", "idea-skipped", "idea-archived"):
        _insert(p, idea_id)
    assert p.lake.record_state_transition("idea-claimed", "QUEUED", "CLAIMED", "claim")
    for idea_id in ("idea-skipped", "idea-archived"):
        assert p.lake.record_state_transition(idea_id, "QUEUED", "SKIPPED", "not_admitted")
    assert p.lake.record_state_transition("idea-archived", "SKIPPED", "ARCHIVED", "retired")

    queue, api = _produce_and_read(p)

    assert {item["idea_id"]: item["status"] for item in api["queue"]} == {
        "idea-queued": "pending", "idea-claimed": "running",
        "idea-skipped": "skipped", "idea-archived": "archived",
    }
    assert queue["counts"] == {"pending": 1, "running": 1, "skipped": 1, "archived": 1}
    assert api["total"] == api["total_all"] == 4
    for item in api["queue"]:
        _assert_item_schema(item)


def test_offline_artifact_display_survives_with_explicit_unverified_authority(project):
    p = project
    _insert(p, in_lake=False)
    p.cfg.pop("idea_lake_db")

    queue, api = _produce_and_read(p)

    assert api["queue"][0]["status"] == "completed"
    assert queue["counts"] == {"completed": 1}
    _assert_item_schema(api["queue"][0])
    assert queue.get("lifecycle_authority") == "unverified_local_artifact", (
        "NEW_FIELD_CONTRACT: offline compatibility must be explicitly non-authoritative")
    assert api.get("lifecycle_authority") == "unverified_local_artifact"
