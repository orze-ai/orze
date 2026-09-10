"""V1-01F: promotion requires current authoritative evidence, not metric guesses.

Guard anomaly policy is deliberately not exercised here. The existing
``verified`` return field means qualified current artifact value in this suite,
not an independent replication or a statistical improvement claim.
"""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orze.engine import champion_guard
from orze.idea_lake import IdeaLake
from orze.reporting import notifications
from orze.reporting.evidence import (
    authoritative_completed_idea_ids,
    qualify_authoritative_report_evidence_with_identity,
)
from orze.reporting.leaderboard import NotificationProcessor, update_report


@pytest.fixture
def project(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    cfg = {
        "_project_root": str(tmp_path),
        "results_dir": str(results),
        "idea_lake_db": str(lake.db_path),
        "report": {
            "primary_metric": "analysis.delta",
            "sort": "ascending",
            "columns": [{
                "key": "analysis.delta",
                "source": "assessment.json:measurement.delta",
            }],
        },
    }
    try:
        yield SimpleNamespace(results=results, lake=lake, cfg=cfg, ideas={})
    finally:
        lake.close()


def _publish(p, value, *, idea_id="idea-candidate", status="completed"):
    folder = p.results / idea_id
    folder.mkdir()
    metrics = {
        "status": "COMPLETED", "analysis.delta": 999,
        "pgmAP_ALL": 777, "map": 555, "score": 999,
    }
    (folder / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (folder / "assessment.json").write_text(
        json.dumps({"measurement": {"delta": value}}), encoding="utf-8")
    p.lake.insert(idea_id, idea_id, "seed: 7", "", status=status,
                  eval_metrics=metrics)
    p.ideas[idea_id] = {"title": idea_id, "config": {"seed": 7}}
    return folder


def _qualified_value(p, idea_id="idea-candidate"):
    completed, _ = authoritative_completed_idea_ids(Path(p.cfg["idea_lake_db"]))
    return qualify_authoritative_report_evidence_with_identity(
        idea_id, p.results, p.cfg, completed)[2]


def _check(p, claim, **kwargs):
    return champion_guard.check_promotion(
        p.results, "idea-candidate", claim, p.cfg, **kwargs)


@pytest.mark.parametrize("direction,value,explicit_disabled", [
    ("ascending", 0.0, False),
    ("descending", -2.0, False),
    ("ascending", -2.0, True),
    ("descending", 0.0, True),
])
def test_exact_source_non_domain_objective_promotes_without_implicit_anomaly_gate(
        project, direction, value, explicit_disabled):
    p = project
    p.cfg["report"]["sort"] = direction
    _publish(p, value)
    if explicit_disabled:
        p.cfg["champion_guard"] = {"enabled": False}
    else:
        # No anomaly policy is declared. An old unscoped score distribution
        # must not turn a fresh qualified observation into a scientific veto.
        (p.results / "_champion_history.json").write_text(
            json.dumps({"metrics": [0.0, 0.1] * 10}), encoding="utf-8")
    assert _qualified_value(p) == value

    allowed, info = _check(p, value)

    assert allowed is True
    assert info["blocked"] is False
    assert info["verified"] == value


@pytest.mark.parametrize("claim,source", [
    (True, 1.0),
    (False, 0.0),
    (float("nan"), 1.0),
    (float("inf"), 1.0),
    (float("-inf"), 1.0),
    ("1.0", 1.0),
    (1.0000000000000002, 1.0),
], ids=["true-is-not-one", "false-is-not-zero", "nan", "positive-inf",
        "negative-inf", "numeric-string", "one-ulp-mismatch"])
def test_claim_must_be_nonboolean_finite_number_exactly_equal_to_source(
        project, claim, source):
    p = project
    _publish(p, source)
    p.cfg["champion_guard"] = {"enabled": False}
    assert _qualified_value(p) == source

    allowed, info = _check(p, claim)

    assert allowed is False
    assert info["blocked"] is True


@pytest.mark.parametrize("rejection", [
    "missing_database", "missing_source", "queued", "lifecycle_mismatch", "taint",
])
def test_disabling_optional_guard_never_disables_mandatory_evidence_authority(
        project, rejection):
    p = project
    folder = _publish(p, 1.0, status="queued" if rejection == "queued" else "completed")
    p.cfg["champion_guard"] = {"enabled": False}
    if rejection == "missing_database":
        p.cfg["idea_lake_db"] = str(p.results.parent / "absent-authority.db")
    elif rejection == "missing_source":
        (folder / "assessment.json").unlink()
    elif rejection == "lifecycle_mismatch":
        p.lake.conn.execute(
            "UPDATE ideas SET status='queued' WHERE idea_id='idea-candidate'")
        p.lake.conn.commit()
    elif rejection == "taint":
        metrics_path = folder / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["tainted_leakage"] = True
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    assert _qualified_value(p) is None

    allowed, info = _check(p, 1.0)

    assert allowed is False
    assert info["blocked"] is True
    if rejection == "missing_database":
        missing = Path(p.cfg["idea_lake_db"])
        assert list(missing.parent.glob(missing.name + "*")) == []


def test_repeated_check_reads_current_source_not_previous_claim_or_raw_cache(project):
    p = project
    folder = _publish(p, 1.0)
    p.cfg["champion_guard"] = {"enabled": False}
    allowed, _ = _check(p, 1.0)
    assert allowed is True
    (folder / "assessment.json").write_text(
        json.dumps({"measurement": {"delta": 2.0}}), encoding="utf-8")
    assert _qualified_value(p) == 2.0

    allowed, info = _check(p, 1.0)

    assert allowed is False
    assert info["blocked"] is True


@pytest.mark.parametrize("enabled", [False, True])
def test_legacy_reproducer_is_rejected_without_running_shell_or_parsing_stdout(
        project, monkeypatch, enabled):
    p = project
    _publish(p, 1.0)
    p.cfg["champion_guard"] = {"enabled": enabled}
    tripwire = Mock(side_effect=AssertionError("promotion must not execute a reproducer"))
    monkeypatch.setattr(subprocess, "run", tripwire)

    allowed, info = _check(p, 1.0, idea_cfg={"reproducer": "untrusted-reproducer 123"})

    tripwire.assert_not_called()
    assert allowed is False
    assert info["blocked"] is True


def test_actual_notification_caller_does_not_promote_when_guard_raises(
        project, monkeypatch):
    p = project
    _publish(p, 5.0, idea_id="idea-baseline")
    _publish(p, 1.0)
    p.cfg["champion_guard"] = {"enabled": False}
    p.cfg["plateau_threshold"] = 0
    p.cfg["notifications"] = {
        "enabled": True, "on": ["completed", "new_best"],
        "channels": [{"type": "webhook", "url": "https://unused.invalid"}],
    }
    rows = update_report(p.results, p.ideas, p.cfg, lake=p.lake)
    assert [row["id"] for row in rows] == ["idea-candidate", "idea-baseline"]
    guard = Mock(side_effect=OSError("injected authority recheck failure"))
    monkeypatch.setattr(champion_guard, "check_promotion", guard)
    send = Mock(return_value=(True, ""))
    monkeypatch.setattr(notifications, "_notify_send", send)
    reporter = NotificationProcessor(p.results, p.cfg, lake=p.lake)
    reporter.load_state({"best_idea_id": "idea-baseline",
                         "completions_since_best": 0,
                         "plateau_notified": False})

    reporter.process(
        [("idea-candidate", None)], rows, p.ideas, {}, 0,
        save_config_hash_fn=Mock(), build_machine_status_fn=lambda: [])

    guard.assert_called_once()
    assert reporter.get_state()["best_idea_id"] == "idea-baseline"
    assert "new_best" not in [call.args[1]["event"] for call in send.call_args_list]
