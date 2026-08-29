"""Active role progress and deadlines must be durable and privacy-safe."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from orze.reporting import state


def _role(**overrides):
    values = {
        "start_time": 1_000.0,
        "timeout": 2_400.0,
        "cycle_num": 7,
        "stall_minutes_override": 5.0,
        "stall_warmup_seconds": 60.0,
        "_stall_since": 1_120.0,
        "_last_progress_at": 1_100.0,
        "_last_observed_at": 1_120.0,
        "_last_progress_kinds": ("artifact", "untrusted"),
        "_last_log_size": 0,
        "progress_paths": (
            Path("/secret/output-one"), Path("/secret/output-two"),
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_active_role_status_exposes_progress_and_exact_deadlines_only():
    rows = state.build_active_role_status(
        {"professor": _role()}, global_stall_minutes=20, now=1_200.0,
    )

    assert rows == {
        "professor": {
            "state": "RUNNING",
            "cycle": 7,
            "started_at_epoch": 1_000.0,
            "elapsed_seconds": 200.0,
            "last_observed_at_epoch": 1_120.0,
            "last_progress_at_epoch": 1_100.0,
            "last_progress_age_seconds": 100.0,
            "last_progress_kinds": ["artifact"],
            "observable_progress_sources": [
                "log", "process_tree_cpu", "declared_artifact_metadata",
            ],
            "declared_progress_path_count": 2,
            "wall_timeout_seconds": 2_400.0,
            "wall_deadline_epoch": 3_400.0,
            "wall_remaining_seconds": 2_200.0,
            "stall_timeout_seconds": 300.0,
            "stall_timer_state": "counting",
            "stall_warmup_deadline_epoch": None,
            "stall_deadline_epoch": 1_420.0,
            "stall_remaining_seconds": 220.0,
            "next_termination_deadline_epoch": 1_420.0,
        },
    }
    rendered = json.dumps(rows)
    assert "/secret" not in rendered
    assert "untrusted" not in rendered


def test_active_role_status_distinguishes_warmup_armed_and_disabled():
    rows = state.build_active_role_status({
        "warmup": _role(
            start_time=1_180.0, _stall_since=0.0,
            _last_progress_at=1_180.0, _last_observed_at=1_180.0,
        ),
        "armed": _role(_stall_since=0.0, _last_log_size=10),
        "disabled": _role(stall_minutes_override=0),
    }, global_stall_minutes=20, now=1_200.0)

    assert rows["warmup"]["stall_timer_state"] == "warmup"
    assert rows["warmup"]["stall_warmup_deadline_epoch"] == 1_240.0
    assert rows["warmup"]["stall_deadline_epoch"] is None
    assert rows["armed"]["stall_timer_state"] == "armed"
    assert rows["armed"]["stall_deadline_epoch"] is None
    assert rows["disabled"]["stall_timer_state"] == "disabled"
    assert rows["disabled"]["stall_timeout_seconds"] == 0.0


def test_status_json_publishes_active_role_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(state.time, "time", lambda: 1_200.0)
    monkeypatch.setattr(state, "_read_all_heartbeats", lambda _path: [])
    active = {"professor": _role()}
    cfg = {
        "poll": 30,
        "role_stall_minutes": 20,
        "roles": {
            "professor": {},
            "disabled": {"enabled": False},
        },
    }

    state.write_status_json(
        tmp_path, iteration=1, active={}, free_gpus=[], queue_depth=0,
        completed_count=0, failed_count=0, skipped_count=0, top_results=[],
        cfg=cfg, role_states={}, active_roles=active,
    )

    written = json.loads((tmp_path / "status.json").read_text())
    assert written["roles"]["professor"]["active"] is True
    assert written["roles"]["professor"]["active_run"][
        "last_progress_kinds"] == ["artifact"]
    assert written["roles"]["disabled"] == {
        "enabled": False,
        "active": False,
        "active_run": None,
        "cycles": 0,
        "last_run_min_ago": None,
    }
