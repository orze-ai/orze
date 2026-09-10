"""Current champion selection is not automatically objective improvement.

V1-01H1 compares two currently qualified rows, not historical measurements or
scientific progress. Same-ID revisions, repeated finished batches and durable
observation identity deliberately remain outside this bounded contract.
"""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orze.engine.champion_guard import check_promotion
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
        "eval_output": "evaluation.json",
        "report": {
            "primary_metric": "quality",
            "sort": "descending",
            "columns": [
                {"key": "quality", "source": "evaluation.json:quality"},
                {"key": "secondary", "source": "evaluation.json:secondary"},
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
    p = SimpleNamespace(results=results, lake=lake, cfg=cfg, ideas={}, send=send)
    try:
        yield p
    finally:
        lake.close()


def _publish(p, idea_id, quality, *, secondary=None):
    folder = p.results / idea_id
    folder.mkdir()
    # Raw fields deliberately disagree with the declared observation source.
    raw = {"status": "COMPLETED", "quality": 999, "secondary": 888}
    (folder / "metrics.json").write_text(json.dumps(raw), encoding="utf-8")
    source = {"quality": quality}
    if secondary is not None:
        source["secondary"] = secondary
    (folder / "evaluation.json").write_text(json.dumps(source), encoding="utf-8")
    p.lake.insert(idea_id, idea_id, "seed: 7", "", status="completed",
                  eval_metrics=raw)
    p.ideas[idea_id] = {"title": idea_id, "config": {"seed": 7}}
    return folder


def _reporter(p, best_id="idea-z-old"):
    reporter = NotificationProcessor(p.results, p.cfg, lake=p.lake)
    reporter.load_state({"best_idea_id": best_id,
                         "completions_since_best": 7,
                         "plateau_notified": True})
    return reporter


def _process(p, reporter, finished, *, rows=None):
    if rows is None:
        rows = update_report(p.results, p.ideas, p.cfg, lake=p.lake)
    reporter.process(
        [(idea_id, None) for idea_id in finished], rows, p.ideas, {}, 0,
        save_config_hash_fn=Mock(), build_machine_status_fn=lambda: [],
    )


def _events(p, event):
    return [call.args[1]["data"] for call in p.send.call_args_list
            if call.args[1]["event"] == event]


def _assert_progress(reporter, best_id, since, notified):
    state = reporter.get_state()
    assert state["best_idea_id"] == best_id
    assert state["completions_since_best"] == since
    assert state["plateau_notified"] is notified


@pytest.mark.parametrize("direction,old", [("ascending", 1), ("descending", -1)])
def test_strict_primary_improvement_uses_declared_direction_and_exact_zero(
    project, direction, old,
):
    p = project
    p.cfg["report"]["sort"] = direction
    _publish(p, "idea-z-old", old)
    _publish(p, "idea-a-new", 0)
    reporter = _reporter(p)

    _process(p, reporter, ["idea-a-new"])

    _assert_progress(reporter, "idea-a-new", 0, False)
    assert len(_events(p, "new_best")) == 1
    assert _events(p, "new_best")[0]["metric_value"] == "0.0000"


@pytest.mark.parametrize("direction", ["ascending", "descending"])
def test_equal_objective_stable_id_reselection_is_not_improvement(project, direction):
    p = project
    p.cfg["report"]["sort"] = direction
    _publish(p, "idea-z-old", 0)
    _publish(p, "idea-a-new", 0)
    reporter = _reporter(p)

    _process(p, reporter, ["idea-a-new"])

    _assert_progress(reporter, "idea-a-new", 8, True)
    assert _events(p, "new_best") == []
    assert len(_events(p, "completed")) == 1


@pytest.mark.parametrize("direction,old", [("ascending", 1), ("descending", -1)])
def test_equal_primary_with_two_finite_secondary_values_can_improve(
    project, direction, old,
):
    p = project
    p.cfg["report"].update(sort=direction, secondary_metric="secondary")
    _publish(p, "idea-z-old", 0, secondary=old)
    _publish(p, "idea-a-new", 0, secondary=0)
    reporter = _reporter(p)

    _process(p, reporter, ["idea-a-new"])

    _assert_progress(reporter, "idea-a-new", 0, False)
    assert len(_events(p, "new_best")) == 1


@pytest.mark.parametrize("direction", ["ascending", "descending"])
def test_missing_secondary_to_measured_is_selection_not_improvement(project, direction):
    p = project
    p.cfg["report"].update(sort=direction, secondary_metric="secondary")
    _publish(p, "idea-z-old", 0)
    _publish(p, "idea-a-new", 0, secondary=0)
    reporter = _reporter(p)

    _process(p, reporter, ["idea-a-new"])

    _assert_progress(reporter, "idea-a-new", 8, True)
    assert _events(p, "new_best") == []


@pytest.mark.parametrize("revocation", ["source_missing", "fsm_conflict"])
def test_ineligible_previous_champion_is_replaced_without_claiming_improvement(
    project, revocation,
):
    p = project
    old = _publish(p, "idea-z-old", 10)
    _publish(p, "idea-a-fallback", 1)
    reporter = _reporter(p)
    # The report was generated before authority/artifacts changed. The public
    # processing boundary must requalify, not trust this cached champion row.
    rows = update_report(p.results, p.ideas, p.cfg, lake=p.lake)
    if revocation == "source_missing":
        (old / "evaluation.json").unlink()
    else:
        p.lake.conn.execute(
            "UPDATE idea_state SET current_state='QUEUED' WHERE idea_id=?",
            ("idea-z-old",),
        )
        p.lake.conn.commit()

    _process(p, reporter, ["idea-a-fallback"], rows=rows)

    _assert_progress(reporter, "idea-a-fallback", 8, True)
    assert _events(p, "new_best") == []


@pytest.mark.parametrize("change", ["revoke_all", "fallback", "strict_improvement"])
def test_empty_finished_batch_still_reconciles_current_qualified_champion(
    project, change,
):
    p = project
    old = _publish(p, "idea-z-old", 1)
    reporter = _reporter(p)
    if change != "revoke_all":
        _publish(p, "idea-a-new", 2)
    rows = update_report(p.results, p.ideas, p.cfg, lake=p.lake)
    if change != "strict_improvement":
        (old / "evaluation.json").unlink()
    # No completion batch must not turn on periodic delivery as a side effect.
    p.cfg["notifications"]["report_interval"] = 1
    p.cfg["plateau_threshold"] = 1

    _process(p, reporter, [], rows=rows)

    if change == "revoke_all":
        _assert_progress(reporter, None, 0, False)
    elif change == "fallback":
        _assert_progress(reporter, "idea-a-new", 7, True)
    else:
        _assert_progress(reporter, "idea-a-new", 0, False)
    assert len(_events(p, "new_best")) == int(change == "strict_improvement")
    assert _events(p, "completed") == []
    assert _events(p, "plateau") == []
    assert _events(p, "report") == []


@pytest.mark.parametrize("comparison", ["strict_improvement", "tie", "revoked_old"])
def test_operational_hold_only_applies_to_a_comparable_improvement(project, comparison):
    p = project
    p.cfg["champion_guard"] = {
        "enabled": True, "action": "hold", "min_history": 4,
        "history_size": 50, "z_threshold": 4.0,
    }
    # Seed actual qualified history through the public guard, not a fake hold.
    for index, value in enumerate((-2, -1, 0, 1)):
        idea_id = f"idea-history-{index}"
        _publish(p, idea_id, value)
        allowed, info = check_promotion(p.results, idea_id, value, p.cfg, lake=p.lake)
        assert allowed, info
    old = _publish(p, "idea-z-old", 100 if comparison == "tie" else 1)
    _publish(p, "idea-a-new", 100)
    reporter = _reporter(p)
    if comparison == "revoked_old":
        (old / "evaluation.json").unlink()

    _process(p, reporter, ["idea-a-new"])

    expected_id = "idea-z-old" if comparison == "strict_improvement" else "idea-a-new"
    _assert_progress(reporter, expected_id, 8, True)
    assert _events(p, "new_best") == []
    assert len(_events(p, "audit")) == int(comparison == "strict_improvement")
