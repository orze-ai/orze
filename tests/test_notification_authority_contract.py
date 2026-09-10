"""V1-01E: public notification processing cannot bypass result authority.

All candidates come from real report generation, with real IdeaLake lifecycle
rows and source-backed artifacts. Only the notification transport is mocked.
Champion-guard policy and duplicate-delivery semantics are outside this suite.
"""

import json
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orze.engine.rebuild_state import restore_reporter_from_evidence
from orze.idea_lake import IdeaLake
from orze.reporting import notifications
from orze.reporting.evidence import (
    authoritative_completed_idea_ids,
    qualify_authoritative_report_evidence_with_identity,
)
from orze.reporting.leaderboard import NotificationProcessor, update_report
from orze.reporting.state import load_state, save_state


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    send = Mock(return_value=(True, ""))
    monkeypatch.setattr(notifications, "_notify_send", send)
    cfg = {
        "_project_root": str(tmp_path),
        "_env_ORZE_RESULTS_DIR": str(results),
        "results_dir": str(results),
        "idea_lake_db": str(lake.db_path),
        "eval_output": "evaluation.json",
        "report": {
            "primary_metric": "score",
            "sort": "ascending",
            "columns": [{"key": "score", "source": "evaluation.json:score"}],
        },
        "notifications": {
            "enabled": True,
            "on": ["completed", "failed", "new_best", "plateau"],
            "channels": [{"type": "webhook", "url": "https://unused.invalid"}],
        },
        "champion_guard": {"enabled": False},
        "plateau_threshold": 0,
    }
    c = SimpleNamespace(results=results, lake=lake, cfg=cfg, ideas={},
                        send=send, save_hash=Mock())
    try:
        yield c
    finally:
        c.lake.close()


def _publish(c, idea_id, score, *, status="completed", raw=999.0, mtime=100):
    folder = c.results / idea_id
    folder.mkdir()
    metrics = {"status": "COMPLETED", "score": raw}
    metrics_path = folder / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    os.utime(metrics_path, (mtime, mtime))
    if score is not None:
        (folder / "evaluation.json").write_text(
            json.dumps({"score": score}), encoding="utf-8")
    c.ideas[idea_id] = {"title": idea_id, "config": {"seed": 7}}
    if status is not None:
        c.lake.insert(idea_id, idea_id, "seed: 7", "original metadata",
                      status=status, eval_metrics=metrics)
    return folder


def _rows(c):
    # The native lake supplies report candidates, as after inbox archival.
    return update_report(c.results, {}, c.cfg, lake=c.lake)


def _reporter(c, best_id="idea-baseline", since=0):
    reporter = NotificationProcessor(c.results, c.cfg, lake=c.lake)
    reporter.load_state({"best_idea_id": best_id,
                         "completions_since_best": since,
                         "plateau_notified": False})
    return reporter


def _process(c, reporter, finished, *, rows=None):
    if rows is None:
        rows = _rows(c)
    reporter.process(
        [(idea_id, None) for idea_id in finished], rows, c.ideas,
        c.lake.get_lifecycle_counts(), 0,
        save_config_hash_fn=c.save_hash,
        build_machine_status_fn=lambda: [])


def _steering(reporter):
    state = reporter.get_state()
    return state["best_idea_id"], state["completions_since_best"]


def _deliveries(c):
    return [call.args[1] for call in c.send.call_args_list]


def _assert_not_advertised(c, idea_id):
    for payload in _deliveries(c):
        data = payload["data"]
        assert idea_id not in [row["id"] for row in data.get("leaderboard", [])]
        if data.get("idea_id") == idea_id:
            assert payload["event"] != "new_best"
            assert data.get("metric_value") is None
            assert not isinstance(data.get("rank"), int)


def _qualified_value(c, idea_id):
    completed, reason = authoritative_completed_idea_ids(c.lake.db_path)
    assert reason == "authoritative_lifecycle_loaded"
    return qualify_authoritative_report_evidence_with_identity(
        idea_id, c.results, c.cfg, completed)[2]


def _lifecycle_snapshot(c, idea_id):
    return {
        "idea": c.lake.get(idea_id),
        "state": c.lake.get_fsm_state(idea_id),
        "transitions": c.lake.get_fsm_history(idea_id),
    }


@pytest.mark.parametrize("enabled", [False, True])
def test_notification_switch_does_not_freeze_qualified_bookkeeping(campaign, enabled):
    c = campaign
    _publish(c, "idea-baseline", 1, mtime=100)
    _publish(c, "idea-later", 2, mtime=200)
    c.cfg["notifications"]["enabled"] = enabled
    reporter = _reporter(c)

    _process(c, reporter, ["idea-later"])

    assert _steering(reporter) == ("idea-baseline", 1)
    c.save_hash.assert_called_once_with("idea-later", {"seed": 7})
    if enabled:
        assert [item["event"] for item in _deliveries(c)] == ["completed"]
    else:
        c.send.assert_not_called()


def test_disabled_updates_survive_toggle_and_real_state_restart(campaign):
    c = campaign
    _publish(c, "idea-baseline", 3, mtime=100)
    reporter = _reporter(c)
    c.cfg["notifications"]["enabled"] = False
    _publish(c, "idea-new-best", 1, mtime=200)
    _process(c, reporter, ["idea-new-best"])
    before_restart = _steering(reporter)
    before_restart_deliveries = len(_deliveries(c))

    save_state(c.results, reporter.get_state())
    c.lake.close()
    c.lake = IdeaLake(c.cfg["idea_lake_db"])
    restarted = NotificationProcessor(c.results, c.cfg, lake=c.lake)
    restarted.load_state(load_state(c.results))
    restore_reporter_from_evidence(restarted, c.results, c.cfg, lake=c.lake)
    restored = _steering(restarted)

    _publish(c, "idea-disabled-later", 2, mtime=300)
    _process(c, restarted, ["idea-disabled-later"])
    after_disabled_tick = _steering(restarted)
    disabled_deliveries = len(_deliveries(c))
    c.cfg["notifications"]["enabled"] = True
    _publish(c, "idea-enabled-later", 4, mtime=400)
    _process(c, restarted, ["idea-enabled-later"])

    assert [before_restart, restored, after_disabled_tick, _steering(restarted)] == [
        ("idea-new-best", 0), ("idea-new-best", 0),
        ("idea-new-best", 1), ("idea-new-best", 2),
    ]
    assert (before_restart_deliveries, disabled_deliveries) == (0, 0)
    assert [(item["event"], item["data"].get("idea_id"))
            for item in _deliveries(c)] == [("completed", "idea-enabled-later")]


def test_public_process_delivers_exact_qualified_zero_not_raw_score(campaign):
    c = campaign
    _publish(c, "idea-baseline", 1, mtime=100)
    _publish(c, "idea-zero", 0, raw=999, mtime=200)
    reporter = _reporter(c)
    assert _qualified_value(c, "idea-zero") == 0

    _process(c, reporter, ["idea-zero"])

    completed = [item["data"] for item in _deliveries(c)
                 if item["event"] == "completed"]
    assert len(completed) == 1
    assert completed[0]["metric_value"] == "0.0000"
    assert completed[0]["rank"] == 1
    assert _steering(reporter) == ("idea-zero", 0)


@pytest.mark.parametrize("invalid_source", ["missing", "rejected"])
def test_unqualified_completion_cannot_fall_back_to_raw_score_or_budget(
        campaign, invalid_source):
    c = campaign
    _publish(c, "idea-baseline", 5)
    _publish(c, "idea-unqualified", None if invalid_source == "missing" else -1,
             raw=999, mtime=200)
    c.cfg["metric_validation"] = {"min_value": {"score": 0}}
    rows = _rows(c)
    assert [row["id"] for row in rows] == ["idea-baseline"]
    assert _qualified_value(c, "idea-unqualified") is None
    reporter = _reporter(c)

    _process(c, reporter, ["idea-unqualified"], rows=rows)

    _assert_not_advertised(c, "idea-unqualified")
    assert _steering(reporter) == ("idea-baseline", 0)


@pytest.mark.parametrize("initial_status", [None, "queued", "running", "failed"],
                         ids=["unknown", "queued", "in-progress", "failed"])
def test_notification_cannot_create_or_change_lifecycle_from_completed_artifact(
        campaign, initial_status):
    c = campaign
    _publish(c, "idea-baseline", 5)
    _publish(c, "idea-untrusted", 0, status=initial_status, mtime=200)
    before = _lifecycle_snapshot(c, "idea-untrusted")
    rows = _rows(c)
    assert [row["id"] for row in rows] == ["idea-baseline"]
    assert _qualified_value(c, "idea-untrusted") is None
    reporter = _reporter(c)

    _process(c, reporter, ["idea-untrusted"], rows=rows)
    after = _lifecycle_snapshot(c, "idea-untrusted")
    second_report = [row["id"] for row in _rows(c)]
    c.lake.close()
    c.lake = IdeaLake(c.cfg["idea_lake_db"])

    assert after == before
    assert _lifecycle_snapshot(c, "idea-untrusted") == before
    assert second_report == ["idea-baseline"]
    assert _qualified_value(c, "idea-untrusted") is None
    _assert_not_advertised(c, "idea-untrusted")
    assert _steering(reporter) == ("idea-baseline", 0)


@pytest.mark.parametrize("change", ["source_missing", "current_config", "tainted"])
def test_passed_report_rows_are_hints_not_current_qualification(campaign, change):
    c = campaign
    _publish(c, "idea-baseline", 5)
    candidate = _publish(c, "idea-stale", 1, mtime=200)
    rows = _rows(c)
    assert [row["id"] for row in rows] == ["idea-stale", "idea-baseline"]
    assert _qualified_value(c, "idea-stale") == 1
    reporter = _reporter(c)
    if change == "source_missing":
        (candidate / "evaluation.json").unlink()
    elif change == "current_config":
        c.cfg["metric_validation"] = {"min_value": {"score": 2}}
    else:
        metrics_path = candidate / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["tainted_leakage"] = True
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    assert _qualified_value(c, "idea-stale") is None

    _process(c, reporter, ["idea-stale"], rows=rows)

    _assert_not_advertised(c, "idea-stale")
    assert _steering(reporter) == ("idea-baseline", 0)
