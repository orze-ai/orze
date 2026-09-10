"""A malformed diagnostic must not consume unrelated completion bookkeeping."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orze.idea_lake import IdeaLake
from orze.reporting import notifications
from orze.reporting.leaderboard import NotificationProcessor, update_report


@pytest.fixture
def project(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    send = Mock(return_value=(True, ""))
    monkeypatch.setattr(notifications, "_notify_send", send)
    cfg = {
        "_project_root": str(tmp_path),
        "results_dir": str(results),
        "idea_lake_db": str(lake.db_path),
        "report": {
            "primary_metric": "score", "sort": "ascending",
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
    p = SimpleNamespace(results=results, lake=lake, cfg=cfg, ideas={},
                        send=send, save_hash=Mock())
    try:
        yield p
    finally:
        lake.close()


def _complete(p, idea_id, score):
    folder = p.results / idea_id
    folder.mkdir()
    metrics = {"status": "COMPLETED", "score": 999}
    (folder / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (folder / "evaluation.json").write_text(
        json.dumps({"score": score}), encoding="utf-8")
    p.lake.insert(idea_id, idea_id, "seed: 3", "", status="completed",
                  eval_metrics=metrics)
    p.ideas[idea_id] = {"title": idea_id, "config": {"seed": 3}}


def _reporter(p):
    reporter = NotificationProcessor(p.results, p.cfg, lake=p.lake)
    reporter.load_state({"best_idea_id": "idea-baseline",
                         "completions_since_best": 0,
                         "plateau_notified": False})
    return reporter


def _process(p, reporter, finished, rows=None):
    if rows is None:
        rows = update_report(p.results, p.ideas, p.cfg, lake=p.lake)
    reporter.process(
        [(idea_id, None) for idea_id in finished], rows, p.ideas, {}, 0,
        save_config_hash_fn=p.save_hash, build_machine_status_fn=lambda: [])


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("diagnostic", [
    {"status": "FAILED", "error": None},
    {"status": "FAILED", "error": 123},
    {"status": "FAILED", "error": "code 1", "training_time": None},
    ["not", "a", "metrics", "object"],
], ids=["null-error", "numeric-error", "null-duration", "list-metrics"])
def test_bad_diagnostic_does_not_prevent_unrelated_qualified_bookkeeping(
        project, enabled, diagnostic):
    p = project
    _complete(p, "idea-baseline", 1)
    _complete(p, "idea-qualified", 2)
    rows = update_report(p.results, p.ideas, p.cfg, lake=p.lake)
    assert [row["id"] for row in rows] == ["idea-baseline", "idea-qualified"]
    # The diagnostic arrives after the report snapshot. Notification processing
    # must tolerate its raw shape without trusting it or dropping the good row.
    failed = p.results / "idea-broken-diagnostic"
    failed.mkdir()
    (failed / "metrics.json").write_text(json.dumps(diagnostic), encoding="utf-8")
    p.lake.insert("idea-broken-diagnostic", "diagnostic", "{}", "",
                  status="failed")
    p.ideas["idea-broken-diagnostic"] = {"title": "diagnostic"}
    p.cfg["notifications"]["enabled"] = enabled
    reporter = _reporter(p)

    _process(p, reporter, ["idea-broken-diagnostic", "idea-qualified"], rows)

    state = reporter.get_state()
    assert state["best_idea_id"] == "idea-baseline"
    assert state["completions_since_best"] == 1
    p.save_hash.assert_called_once_with("idea-qualified", {"seed": 3})
    if enabled:
        completed = [call.args[1]["data"]["idea_id"]
                     for call in p.send.call_args_list
                     if call.args[1]["event"] == "completed"]
        assert completed == ["idea-qualified"]
    else:
        p.send.assert_not_called()


def test_disabled_threshold_crossing_does_not_consume_plateau_notification(project):
    p = project
    _complete(p, "idea-baseline", 1)
    _complete(p, "idea-disabled-later", 2)
    p.cfg["plateau_threshold"] = 1
    p.cfg["notifications"]["enabled"] = False
    reporter = _reporter(p)

    _process(p, reporter, ["idea-disabled-later"])

    state = reporter.get_state()
    assert state["completions_since_best"] == 1
    assert state["plateau_notified"] is False
    p.send.assert_not_called()
    # This is a different newly completed observation, not a delivery replay.
    p.cfg["notifications"]["enabled"] = True
    _complete(p, "idea-enabled-later", 3)
    _process(p, reporter, ["idea-enabled-later"])

    state = reporter.get_state()
    assert state["completions_since_best"] == 2
    assert state["plateau_notified"] is True
    plateau = [call.args[1] for call in p.send.call_args_list
               if call.args[1]["event"] == "plateau"]
    assert len(plateau) == 1
    assert plateau[0]["data"]["since_best"] == 2


def test_missing_lifecycle_database_is_not_created_from_report_artifacts(project):
    p = project
    _complete(p, "idea-baseline", 1)
    rows = update_report(p.results, p.ideas, p.cfg, lake=p.lake)
    assert [row["id"] for row in rows] == ["idea-baseline"]
    missing = p.results.parent / "absent-authority.db"
    p.cfg["idea_lake_db"] = str(missing)
    reporter = NotificationProcessor(p.results, p.cfg, lake=None)
    reporter.load_state({"best_idea_id": "idea-baseline",
                         "completions_since_best": 7,
                         "plateau_notified": True})

    _process(p, reporter, ["idea-baseline"], rows)

    assert reporter.get_state() == {
        "best_idea_id": None, "completions_since_best": 0,
        "plateau_notified": False,
    }
    assert not missing.exists()
    assert list(missing.parent.glob(missing.name + "*")) == []
    p.save_hash.assert_not_called()
    p.send.assert_not_called()
