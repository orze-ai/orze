"""The real controller constructor must restore only qualified evidence.

These tests exercise Orze.__init__, state loading, and the shared evidence
reader together. They never call run(), allocate a GPU, or launch a process.
"""
import json
import os
from unittest.mock import Mock

import pytest

from orze.engine.orchestrator import Orze
from orze.idea_lake import IdeaLake
from orze.reporting.state import save_state


@pytest.fixture
def start_controller(monkeypatch):
    # Config schema is covered separately; retain the actual constructor,
    # persistence, lake, reporter, and recovery implementation under test.
    monkeypatch.setattr(
        "orze.engine.orchestrator._validate_config", lambda _cfg: ([], []))
    monkeypatch.setattr("orze.engine.orchestrator.signal.signal", Mock())
    monkeypatch.setattr("orze.engine.orchestrator.atexit.register", Mock())
    monkeypatch.setattr("orze.engine.orchestrator.notify", Mock())
    popen = Mock(side_effect=AssertionError("startup test must not launch work"))
    run = Mock(side_effect=AssertionError("startup test must not query devices"))
    monkeypatch.setattr("orze.engine.orchestrator.subprocess.Popen", popen)
    monkeypatch.setattr("orze.engine.orchestrator.subprocess.run", run)
    controllers = []

    def start(cfg):
        controller = Orze([], cfg)
        controllers.append(controller)
        return controller

    yield start

    for controller in controllers:
        if controller.lake is not None:
            controller.lake.close()
        assert not controller.active
        assert controller.active_evals == {}
        assert controller.active_roles == {}
        assert controller._gpu_leases is None
    popen.assert_not_called()
    run.assert_not_called()


def _project(tmp_path, rows, report=None, state=None):
    results = tmp_path / "results"
    results.mkdir()
    database = tmp_path / "ideas.db"
    lake = IdeaLake(database)
    try:
        for index, (idea_id, status, metrics, evaluation) in enumerate(rows):
            lake.insert(
                idea_id, idea_id, "seed: 1", "", status=status,
                eval_metrics={"score": 999})
            folder = results / idea_id
            folder.mkdir()
            if metrics is not None:
                path = folder / "metrics.json"
                path.write_text(json.dumps(metrics), encoding="utf-8")
                os.utime(path, (100 + index, 100 + index))
            if evaluation is not None:
                (folder / "evaluation.json").write_text(
                    json.dumps(evaluation), encoding="utf-8")
    finally:
        lake.close()
    save_state(results, state or {
        "best_idea_id": "idea-stale",
        "completions_since_best": 77,
    })
    return {
        "results_dir": str(results),
        "idea_lake_db": str(database),
        "ideas_file": str(tmp_path / "ideas.md"),
        "gpu_scheduling": {"mode": "exclusive"},
        "notifications": {"enabled": False},
        "report": report or {"primary_metric": "score", "sort": "descending"},
    }


@pytest.mark.parametrize("metrics", [
    None,
    {"score": 999},
    {"status": "COMPLETED", "score": 999},
])
def test_startup_revokes_stale_champion_with_no_qualified_evidence(
        tmp_path, start_controller, metrics):
    cfg = _project(tmp_path, [
        ("idea-stale", "completed", metrics, None),
    ])
    cfg["metric_validation"] = {"max_value": {"score": 1}}

    controller = start_controller(cfg)

    restored = controller._build_state_dict()
    assert restored["best_idea_id"] is None
    assert restored["completions_since_best"] == 0


def test_startup_uses_source_and_counts_only_qualified_newer_results(
        tmp_path, start_controller):
    completed = {"status": "COMPLETED"}
    cfg = _project(tmp_path, [
        ("idea-champion", "completed", {**completed, "score": 90}, {"score": 2}),
        ("idea-later", "completed", {**completed, "score": 80}, {"score": 3}),
        ("idea-invalid", "completed", {**completed, "score": 0}, {"score": -1}),
        ("idea-unfinished", "queued", {**completed, "score": 0}, {"score": 1}),
        ("idea-source-missing", "completed", {**completed, "score": 0}, None),
    ], {
        "primary_metric": "score",
        "sort": "ascending",
        "columns": [{"key": "score", "source": "evaluation.json:score"}],
    })
    cfg["metric_validation"] = {"min_value": {"score": 0}}

    controller = start_controller(cfg)

    restored = controller._build_state_dict()
    assert restored["best_idea_id"] == "idea-champion"
    assert restored["completions_since_best"] == 1


def test_startup_recovery_exception_cannot_retain_stale_steering_state(
        tmp_path, monkeypatch, start_controller):
    cfg = _project(tmp_path, [], state={
        "best_idea_id": "idea-stale",
        "completions_since_best": 77,
        "plateau_notified": True,
    })
    monkeypatch.setattr(
        "orze.engine.rebuild_state.rebuild_best_from_evidence",
        Mock(side_effect=OSError("injected evidence read failure")),
    )

    controller = start_controller(cfg)

    restored = controller._build_state_dict()
    assert restored["best_idea_id"] is None
    assert restored["completions_since_best"] == 0
    assert restored["plateau_notified"] is False


def test_startup_without_lifecycle_authority_cannot_retain_stale_champion(
        tmp_path, monkeypatch, start_controller):
    cfg = _project(tmp_path, [])
    cfg["idea_lake_db"] = str(tmp_path / "unavailable.db")
    monkeypatch.setattr(
        "orze.idea_lake.IdeaLake",
        Mock(side_effect=OSError("injected database open failure")),
    )

    controller = start_controller(cfg)

    restored = controller._build_state_dict()
    assert controller.lake is None
    assert restored["best_idea_id"] is None
    assert restored["completions_since_best"] == 0
    assert not (tmp_path / "unavailable.db").exists()


@pytest.mark.parametrize(
    "previous_best,previous_since,has_evidence,expected_best,expected_notified",
    [
        ("idea-stale", 77, False, None, False),
        ("idea-stale", 77, True, "idea-champion", False),
        ("idea-champion", 77, True, "idea-champion", False),
        ("idea-champion", 1, True, "idea-champion", True),
    ],
    ids=["revoked", "replaced", "counter-decreased", "unchanged"],
)
def test_startup_scopes_plateau_notification_to_recovered_champion(
        tmp_path, start_controller, previous_best, previous_since,
        has_evidence, expected_best, expected_notified):
    rows = [
        ("idea-champion", "completed", {"status": "COMPLETED", "score": 0.8}, None),
        ("idea-later", "completed", {"status": "COMPLETED", "score": 0.5}, None),
    ] if has_evidence else []
    cfg = _project(tmp_path, rows, state={
        "best_idea_id": previous_best,
        "completions_since_best": previous_since,
        "plateau_notified": True,
    })

    controller = start_controller(cfg)

    restored = controller._build_state_dict()
    assert restored["best_idea_id"] == expected_best
    assert restored["completions_since_best"] == (1 if has_evidence else 0)
    assert restored["plateau_notified"] is expected_notified
