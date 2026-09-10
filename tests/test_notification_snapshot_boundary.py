"""Notifications may refresh a qualified metric cache, never lifecycle authority."""

import json
from unittest.mock import Mock

import pytest

from orze.idea_lake import IdeaLake
from orze.reporting import leaderboard


@pytest.fixture
def project(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    cfg = {
        "_project_root": str(tmp_path),
        "idea_lake_db": str(lake.db_path),
        "notifications": {"enabled": True},
        "report": {
            "primary_metric": "score",
            "secondary_metric": "penalty",
            "sort": "ascending",
            "columns": [
                {"key": "score", "source": "evaluation.json:score"},
                {"key": "penalty", "source": "evaluation.json:penalty"},
            ],
        },
    }
    notify = Mock()
    monkeypatch.setattr(leaderboard, "notify", notify)
    try:
        yield results, lake, cfg, {}, notify
    finally:
        lake.close()


def _publish(project, idea_id="idea-qualified", *, lifecycle="completed",
             tainted=False, source_present=True, group="kept", score=0.0):
    results, lake, _, ideas, _ = project
    folder = results / idea_id
    folder.mkdir()
    metrics = {"status": "COMPLETED", "score": 999, "penalty": 888}
    if tainted:
        metrics["tainted_leakage"] = True
    (folder / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    source = {"score": score, "penalty": -2} if source_present else {}
    (folder / "evaluation.json").write_text(json.dumps(source), encoding="utf-8")
    (folder / "resolved_config.yaml").write_text(
        f"group: {group}\n", encoding="utf-8"
    )
    # The current in-memory stub is deliberately less rich than the archive.
    ideas[idea_id] = {"title": "Runtime display title"}
    if lifecycle is not None:
        lake.insert(
            idea_id, "Preserve archived title", "seed: 42\nbatch_size: 8\n",
            "Preserve complete original markdown", status="queued",
            priority="high", category="configuration", parent="idea-parent",
            hypothesis="Preserve the original hypothesis", training_time=123.5,
            created_at="2001-02-03T04:05:06", approach_family="ablation",
            kind="posthoc_eval", eval_metrics={"score": 123, "penalty": 456},
        )
        if lifecycle == "completed":
            for old, new in (
                ("QUEUED", "CLAIMED"), ("CLAIMED", "IN_PROGRESS"),
                ("IN_PROGRESS", "COMPLETE"),
            ):
                assert lake.record_state_transition(
                    idea_id, old, new, reason="real fixture lifecycle"
                )
    return idea_id


def _snapshot(lake, idea_id):
    idea = lake.conn.execute(
        "SELECT * FROM ideas WHERE idea_id=?", (idea_id,)
    ).fetchone()
    state = lake.conn.execute(
        "SELECT * FROM idea_state WHERE idea_id=?", (idea_id,)
    ).fetchone()
    transitions = lake.conn.execute(
        "SELECT * FROM idea_transitions WHERE idea_id=? ORDER BY id", (idea_id,)
    ).fetchall()
    return {
        "idea": dict(idea) if idea is not None else None,
        "state": dict(state) if state is not None else None,
        "transitions": [dict(row) for row in transitions],
    }


def _report_rows(project):
    results, lake, cfg, ideas, _ = project
    return leaderboard.update_report(results, ideas, cfg, lake=lake)


def _process(project, finished_ids, rows):
    results, lake, cfg, ideas, _ = project
    reporter = leaderboard.NotificationProcessor(results, cfg, lake=lake)
    reporter.process(
        [(idea_id, None) for idea_id in finished_ids], rows, ideas, {}, 0,
        save_config_hash_fn=Mock(), build_machine_status_fn=lambda: [],
    )
    return reporter


@pytest.mark.parametrize("notifications_enabled", [True, False])
def test_qualified_metric_snapshot_preserves_all_nonmetric_archive_and_state_fields(
    project, notifications_enabled
):
    _, lake, cfg, _, _ = project
    cfg["notifications"]["enabled"] = notifications_enabled
    idea_id = _publish(project)
    rows = _report_rows(project)
    before = _snapshot(lake, idea_id)

    _process(project, [idea_id], rows)

    after = _snapshot(lake, idea_id)
    mirror = json.loads(after["idea"].pop("eval_metrics"))
    before["idea"].pop("eval_metrics")
    assert after == before
    assert mirror["score"] == 0.0
    assert mirror["penalty"] == -2


@pytest.mark.parametrize("rejection", ["queued", "tainted", "missing_source"])
def test_unqualified_completion_cannot_update_the_archived_metric_snapshot(
    project, rejection
):
    _, lake, _, _, _ = project
    idea_id = _publish(
        project, lifecycle="queued" if rejection == "queued" else "completed",
        tainted=rejection == "tainted", source_present=rejection != "missing_source",
    )
    rows = _report_rows(project)
    before = _snapshot(lake, idea_id)

    _process(project, [idea_id], rows)

    assert _snapshot(lake, idea_id) == before


def test_unknown_completion_cannot_insert_its_own_lifecycle_authority(project):
    _, lake, _, _, _ = project
    idea_id = _publish(project, lifecycle=None)
    rows = _report_rows(project)
    before = _snapshot(lake, idea_id)
    assert before == {"idea": None, "state": None, "transitions": []}

    _process(project, [idea_id], rows)

    assert _snapshot(lake, idea_id) == before


def test_snapshot_update_rechecks_lifecycle_after_qualification(project, monkeypatch):
    _, lake, _, _, _ = project
    idea_id = _publish(project)
    rows = _report_rows(project)
    original_notify_completed = leaderboard.NotificationProcessor._notify_completed
    requeued = []

    def requeue_after_qualified_notification(self, *args, **kwargs):
        original_notify_completed(self, *args, **kwargs)
        # Both implementations call this hook after resolving the completion
        # and before archiving/updating its metric mirror. A concurrent owner
        # now requeues the task; the observer must not overwrite that decision.
        lake.conn.execute(
            "UPDATE ideas SET status='queued' WHERE idea_id=?", (idea_id,)
        )
        lake.conn.execute(
            "UPDATE idea_state SET current_state='QUEUED' WHERE idea_id=?",
            (idea_id,),
        )
        lake.conn.commit()
        requeued.append(_snapshot(lake, idea_id))

    monkeypatch.setattr(
        leaderboard.NotificationProcessor, "_notify_completed",
        requeue_after_qualified_notification,
    )

    _process(project, [idea_id], rows)

    assert len(requeued) == 1
    assert _snapshot(lake, idea_id) == requeued[0]
    assert json.loads(requeued[0]["idea"]["eval_metrics"])["score"] == 123


def test_notification_views_are_rebuilt_from_current_qualified_source_values(project):
    results, _, cfg, _, notify = project
    cfg["report"]["views"] = [
        {"name": "eligible", "title": "Eligible", "filter": {"group": "kept"}}
    ]
    good = _publish(project)
    _publish(project, "idea-tainted", tainted=True, score=-1)
    _publish(project, "idea-other-view", group="other", score=2)
    rows = _report_rows(project)
    # Poison the derived cache after the real report has generated its rows.
    (results / "_leaderboard_eligible.json").write_text(json.dumps({
        "title": "Stale view",
        "top": [
            {"idea_id": "idea-unknown", "metric_value": -9999},
            {"idea_id": "idea-tainted", "metric_value": -999},
            {"idea_id": good, "metric_value": 999},
            {"idea_id": "idea-other-view", "metric_value": -888},
        ],
    }), encoding="utf-8")

    _process(project, [good], rows)

    completed = [
        call.args[1] for call in notify.call_args_list if call.args[0] == "completed"
    ]
    assert len(completed) == 1
    entries = completed[0]["view_leaderboards"]["eligible"]["entries"]
    assert [(entry["id"], entry["value"]) for entry in entries] == [(good, 0.0)]
