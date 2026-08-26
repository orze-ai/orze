"""Status snapshots must distinguish historical data from live activity."""

import asyncio
import json

from orze.admin import mcp, server
from orze.reporting import state


def test_annotate_status_freshness_current_and_stale():
    snapshot = {
        "snapshot_epoch": 1_000.0,
        "snapshot_ttl_seconds": 300,
        "snapshot_valid_until_epoch": 1_300.0,
        "active": [{"idea_id": "idea-1"}],
    }

    current = state.annotate_status_freshness(snapshot, now=1_299.0)
    stale = state.annotate_status_freshness(snapshot, now=1_301.0)

    assert current["snapshot_state"] == "CURRENT"
    assert current["snapshot_stale"] is False
    assert current["snapshot_age_seconds"] == 299.0
    assert stale["snapshot_state"] == "STALE"
    assert stale["snapshot_stale"] is True
    assert snapshot.get("snapshot_stale") is None


def test_annotate_legacy_timestamp_and_fail_closed():
    legacy = state.annotate_status_freshness(
        {"timestamp": "1970-01-01T00:16:40+00:00"}, now=1_100.0,
    )
    unknown = state.annotate_status_freshness(
        {"timestamp": "not-a-time", "active": [{"idea_id": "old"}]},
        now=1_100.0,
    )

    assert legacy["snapshot_state"] == "CURRENT"
    assert legacy["snapshot_age_seconds"] == 100.0
    assert unknown["snapshot_state"] == "UNKNOWN"
    assert unknown["snapshot_stale"] is True
    assert unknown["snapshot_freshness_reason"] == "missing_or_invalid_timestamp"


def test_write_status_declares_validity_window(tmp_path, monkeypatch):
    monkeypatch.setattr(state.time, "time", lambda: 1_000.0)
    monkeypatch.setattr(state, "_read_all_heartbeats", lambda _path: [])

    state.write_status_json(
        tmp_path, iteration=7, active={}, free_gpus=[4, 5], queue_depth=3,
        completed_count=2, failed_count=1, skipped_count=0, top_results=[],
        cfg={"poll": 30, "roles": {}},
    )

    written = json.loads((tmp_path / "status.json").read_text())
    assert written["snapshot_epoch"] == 1_000.0
    assert written["snapshot_ttl_seconds"] == 300
    assert written["snapshot_valid_until_epoch"] == 1_300.0


def test_admin_status_dynamically_marks_stale(tmp_path, monkeypatch):
    (tmp_path / "status.json").write_text(json.dumps({
        "snapshot_epoch": 1_000.0,
        "snapshot_ttl_seconds": 300,
        "active": [{"idea_id": "idea-old"}],
    }))
    monkeypatch.setattr(server, "_cfg", {"results_dir": str(tmp_path)})
    monkeypatch.setattr(state.time, "time", lambda: 1_301.0)

    response = asyncio.run(server.get_status())

    assert response["snapshot_stale"] is True
    assert response["snapshot_state"] == "STALE"


def test_admin_runs_suppresses_stale_active_runs(tmp_path, monkeypatch):
    (tmp_path / "status.json").write_text(json.dumps({
        "snapshot_epoch": 1_000.0,
        "snapshot_ttl_seconds": 300,
        "active": [{"idea_id": "idea-old"}],
        "top_results": [{"idea_id": "idea-best"}],
    }))
    monkeypatch.setattr(server, "_cfg", {
        "results_dir": str(tmp_path),
        "ideas_file": str(tmp_path / "missing-ideas.md"),
    })
    monkeypatch.setattr(server, "_cache", {})
    monkeypatch.setattr(server, "parse_ideas", lambda _path: {})
    monkeypatch.setattr(state.time, "time", lambda: 1_301.0)

    response = asyncio.run(server.get_runs())

    assert response["status_snapshot_stale"] is True
    assert response["active"] == []
    assert response["top_results"] == [{"idea_id": "idea-best"}]


def test_mcp_status_suppresses_stale_active_runs(tmp_path, monkeypatch):
    (tmp_path / "status.json").write_text(json.dumps({
        "snapshot_epoch": 1_000.0,
        "snapshot_ttl_seconds": 300,
        "active": [{"idea_id": "idea-old"}],
        "top_results": [{"idea_id": "idea-best"}],
    }))
    monkeypatch.setattr(state.time, "time", lambda: 1_301.0)

    result = json.loads(mcp._tool_status({}, {"results_dir": str(tmp_path)}))

    assert result["snapshot_stale"] is True
    assert result["active"] == []
    assert result["top_results"] == [{"idea_id": "idea-best"}]
