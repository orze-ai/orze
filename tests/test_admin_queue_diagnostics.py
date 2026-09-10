"""Ancillary diagnostics/enrichment cannot break actual admin queue delivery.

Real temporary files, native lifecycle, write_admin_cache and GET /api/queue
are used. The client runs in process without a network listener or providers.
Malformed metrics belong to an unrelated diagnostic-only directory, ensuring
that the failure boundary is the Alerts scan, not queue candidate selection.
"""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from orze.admin import server
from orze.core.config import orze_path
from orze.idea_lake import IdeaLake
from orze.reporting.leaderboard import write_admin_cache


@pytest.fixture
def project(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(str(tmp_path / "authority.db"))
    cfg = {
        "_project_root": str(tmp_path), "_orze_dir": str(tmp_path / ".orze"),
        "_env_ORZE_RESULTS_DIR": str(results), "results_dir": str(results),
        "idea_lake_db": str(lake.db_path),
    }
    monkeypatch.setattr(server, "_cfg", cfg)
    monkeypatch.setattr(server, "_cache", {})
    monkeypatch.setattr(server, "_BACKBONE_REGISTRY", {})
    with TestClient(server.app) as client:
        p = SimpleNamespace(results=results, lake=lake, cfg=cfg, ideas={}, client=client)
        try:
            yield p
        finally:
            lake.close()


def _task(p, *, complete=False, config=None):
    idea_id = "idea-live"
    folder = p.results / idea_id
    folder.mkdir()
    p.lake.insert(idea_id, "live task", "seed: 23", "", status="queued")
    assert p.lake.record_state_transition(idea_id, "QUEUED", "CLAIMED", "claim")
    assert p.lake.record_state_transition(idea_id, "CLAIMED", "IN_PROGRESS", "launch")
    if complete:
        assert p.lake.record_state_transition(
            idea_id, "IN_PROGRESS", "COMPLETE", "training_completed")
    # Preserve a valid legacy control so failures come from diagnostics/HF,
    # not from the lifecycle-versus-training correction tested elsewhere.
    (folder / "metrics.json").write_text(json.dumps({
        "status": "COMPLETED" if complete else "RUNNING",
    }), encoding="utf-8")
    p.ideas[idea_id] = {
        "title": "live task", "config": config if config is not None else {"seed": 23},
    }


def _read(p):
    write_admin_cache(p.results, p.ideas, p.cfg)
    cache = json.loads(orze_path(p.cfg, "state", "admin_cache.json").read_text(
        encoding="utf-8"))
    server._cache.clear()
    response = p.client.get("/api/queue")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == payload["total_all"] == 1
    assert [row["idea_id"] for row in payload["queue"]] == ["idea-live"]
    assert payload["counts"] == cache["queue"]["counts"]
    return cache, payload


@pytest.mark.parametrize("raw", [
    pytest.param(b"[]", id="json-list"),
    pytest.param(b"null", id="json-null"),
    pytest.param(b'"not-a-mapping"', id="json-string"),
    pytest.param(b"\xff\xfe", id="invalid-utf8"),
])
def test_unrelated_malformed_metrics_do_not_block_native_queue_publication(project, raw):
    p = project
    _task(p)
    bad = p.results / "idea-malformed-diagnostic"
    bad.mkdir()
    (bad / "metrics.json").write_bytes(raw)
    good = p.results / "idea-real-failure-diagnostic"
    good.mkdir()
    (good / "metrics.json").write_text(
        '{"status":"FAILED","error":"known diagnostic failure"}', encoding="utf-8")

    cache, payload = _read(p)

    assert payload["queue"][0]["status"] == "running"
    assert payload["counts"] == {"running": 1}
    assert p.lake.get_fsm_state("idea-live") == "IN_PROGRESS"
    failures = [row["idea_id"] for row in cache["alerts"]["alerts"]
                if row["type"] == "failure"]
    assert failures == [good.name]
    assert (bad / "metrics.json").read_bytes() == raw


@pytest.mark.parametrize("config", [
    pytest.param({"backbone": None}, id="null-backbone"),
    pytest.param({"backbone": "generic-encoder"}, id="string-backbone"),
    pytest.param({"backbone": []}, id="list-backbone"),
    pytest.param({"backbone": {"name": []}}, id="non-string-name"),
])
def test_generic_backbone_metadata_does_not_crash_actual_queue_api(project, config):
    p = project
    _task(p, complete=True, config=config)

    _, payload = _read(p)

    assert payload["queue"][0]["status"] == "completed"
    assert payload["counts"] == {"completed": 1}
    assert payload["queue"][0]["config"] == config
    assert payload["queue"][0]["huggingface"] is None
