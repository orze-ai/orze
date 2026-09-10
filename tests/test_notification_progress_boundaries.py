"""Independent H1 boundaries through real report/evidence/process consumers.

Only the delivery transport is mocked. Every candidate has a real IdeaLake
lifecycle and declared observation source. These are local comparisons of two
current qualified rows, not tests of statistical progress, same-ID revisions,
replayed batches, historical measurements, or restart counter persistence.
"""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orze.idea_lake import IdeaLake
from orze.reporting import notifications
from orze.reporting.leaderboard import NotificationProcessor, update_report
from orze.reporting.notification_evidence import qualified_notification_rows


@pytest.fixture
def project(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(str(tmp_path / "authority.db"))
    send = Mock(return_value=(True, ""))
    monkeypatch.setattr(notifications, "_notify_send", send)
    cfg = {
        "_project_root": str(tmp_path),
        "_orze_dir": str(tmp_path / ".orze"),
        "results_dir": str(results),
        "idea_lake_db": str(lake.db_path),
        "eval_output": "observation.json",
        "report": {
            "primary_metric": "quality",
            "sort": "descending",
            "columns": [
                {"key": "quality", "source": "observation.json:quality"},
                {"key": "secondary", "source": "observation.json:secondary"},
            ],
        },
        "notifications": {
            "enabled": True,
            "on": ["completed", "new_best", "plateau", "report", "audit"],
            "channels": [{"type": "webhook", "url": "https://unused.invalid"}],
        },
        "champion_guard": {"enabled": False},
        "plateau_threshold": 0,
    }
    p = SimpleNamespace(root=tmp_path, results=results, lake=lake, cfg=cfg,
                        ideas={}, send=send, save_hash=Mock())
    try:
        yield p
    finally:
        lake.close()


def _publish(p, idea_id, quality, secondary):
    folder = p.results / idea_id
    folder.mkdir()
    raw = {"status": "COMPLETED", "quality": 999, "secondary": 888}
    (folder / "metrics.json").write_text(json.dumps(raw), encoding="utf-8")
    (folder / "observation.json").write_text(
        json.dumps({"quality": quality, "secondary": secondary}), encoding="utf-8")
    p.lake.insert(idea_id, idea_id, "seed: 13", "", status="completed",
                  eval_metrics=raw)
    p.ideas[idea_id] = {"title": idea_id, "config": {"seed": 13}}
    return folder


def _report_and_processor(p):
    rows = update_report(p.results, p.ideas, p.cfg, lake=p.lake)
    # Verify that the intended rows reach the authoritative consumer, rather
    # than obtaining a false negative because the observation was rejected.
    qualified = qualified_notification_rows(p.results, p.cfg, rows, p.lake)
    assert {row["id"] for row in qualified} == set(p.ideas)
    assert all(row["evidence_identity"] for row in qualified)
    reporter = NotificationProcessor(p.results, p.cfg, lake=p.lake)
    reporter.load_state({"best_idea_id": "idea-z-old",
                         "completions_since_best": 7,
                         "plateau_notified": True})
    return rows, reporter


def _process(p, reporter, rows, finished):
    reporter.process([(idea_id, None) for idea_id in finished], rows,
                     p.ideas, {}, 0, save_config_hash_fn=p.save_hash,
                     build_machine_status_fn=lambda: [])


def _events(p, event):
    return [call.args[1]["data"] for call in p.send.call_args_list
            if call.args[1]["event"] == event]


def _assert_selection_only(p, reporter):
    assert reporter.get_state() == {
        "best_idea_id": "idea-a-new",
        "completions_since_best": 8,
        "plateau_notified": True,
    }
    assert _events(p, "new_best") == []
    assert len(_events(p, "completed")) == 1


@pytest.mark.parametrize("old_secondary", [
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="positive-infinity"),
    pytest.param(float("-inf"), id="negative-infinity"),
    pytest.param(True, id="true-is-not-one"),
    pytest.param(False, id="false-is-not-zero"),
])
def test_invalid_optional_secondary_to_finite_cannot_prove_progress(
    project, old_secondary,
):
    p = project
    p.cfg["report"]["secondary_metric"] = "secondary"
    # Explicitly permit nonfinite *optional diagnostics*. Default validation
    # still rejects NaN/Inf observations; primary remains finite-required.
    p.cfg["metric_validation"] = {"reject_nan": False, "reject_inf": False}
    _publish(p, "idea-z-old", 0, old_secondary)
    _publish(p, "idea-a-new", 0, 0)
    rows, reporter = _report_and_processor(p)

    _process(p, reporter, rows, ["idea-a-new"])

    _assert_selection_only(p, reporter)


@pytest.mark.parametrize("direction,old,new", [
    ("ascending", 100, -100), ("descending", -100, 100),
])
def test_displayed_but_undeclared_secondary_does_not_prove_progress(
    project, direction, old, new,
):
    p = project
    p.cfg["report"]["sort"] = direction
    assert "secondary_metric" not in p.cfg["report"]
    _publish(p, "idea-z-old", 0, old)
    _publish(p, "idea-a-new", 0, new)
    rows, reporter = _report_and_processor(p)

    _process(p, reporter, rows, ["idea-a-new"])

    _assert_selection_only(p, reporter)


@pytest.mark.parametrize("direction,secondary", [
    ("ascending", 0), ("descending", -2),
])
def test_both_objectives_equal_only_stable_id_changes(project, direction, secondary):
    p = project
    p.cfg["report"].update(sort=direction, secondary_metric="secondary")
    _publish(p, "idea-z-old", 0, secondary)
    _publish(p, "idea-a-new", 0, secondary)
    rows, reporter = _report_and_processor(p)

    _process(p, reporter, rows, ["idea-a-new"])

    _assert_selection_only(p, reporter)


@pytest.mark.parametrize("enabled", [True, False])
def test_no_finished_missing_authority_revokes_without_creating_database(
    project, enabled,
):
    p = project
    _publish(p, "idea-z-old", 1, 0)
    rows, reporter = _report_and_processor(p)
    missing_db = p.root / "missing-authority" / "never-created.db"
    p.cfg["idea_lake_db"] = str(missing_db)
    reporter.lake = None  # No open Lake may override the new explicit DB path.
    p.cfg["notifications"].update(enabled=enabled, report_interval=1)
    p.cfg["plateau_threshold"] = 1

    _process(p, reporter, rows, [])

    assert reporter.get_state() == {
        "best_idea_id": None, "completions_since_best": 0,
        "plateau_notified": False,
    }
    assert not missing_db.parent.exists()  # Includes DB, WAL and journal files.
    p.save_hash.assert_not_called()
    p.send.assert_not_called()


@pytest.mark.parametrize("enabled", [True, False])
def test_no_finished_tainted_current_source_revokes_cached_champion(project, enabled):
    p = project
    old = _publish(p, "idea-z-old", 1, 0)
    rows, reporter = _report_and_processor(p)
    p.cfg["notifications"].update(enabled=enabled, report_interval=1)
    p.cfg["plateau_threshold"] = 1
    metrics = json.loads((old / "metrics.json").read_text(encoding="utf-8"))
    metrics["tainted_leakage"] = True
    (old / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    _process(p, reporter, rows, [])

    assert reporter.get_state() == {
        "best_idea_id": None, "completions_since_best": 0,
        "plateau_notified": False,
    }
    p.save_hash.assert_not_called()
    p.send.assert_not_called()


@pytest.mark.parametrize("direction,old", [("ascending", 1), ("descending", -1)])
def test_strict_primary_improvement_does_not_require_valid_optional_secondary(
    project, direction, old,
):
    p = project
    p.cfg["report"].update(sort=direction, secondary_metric="secondary")
    p.cfg["metric_validation"] = {"reject_nan": False, "reject_inf": False}
    _publish(p, "idea-z-old", old, float("nan"))
    _publish(p, "idea-a-new", 0, False)
    rows, reporter = _report_and_processor(p)

    _process(p, reporter, rows, ["idea-a-new"])

    assert reporter.get_state() == {
        "best_idea_id": "idea-a-new", "completions_since_best": 0,
        "plateau_notified": False,
    }
    assert len(_events(p, "new_best")) == 1
    assert _events(p, "new_best")[0]["metric_value"] == "0.0000"
