"""Completion notifications must preserve the qualified objective, including 0."""

import json
from unittest.mock import Mock

import pytest

from orze.reporting import leaderboard
from orze.reporting.evidence import qualify_local_report_evidence


@pytest.mark.parametrize("primary", ["score", "evaluation.score"])
def test_completion_notification_preserves_qualified_zero_over_raw_metric(
    tmp_path, monkeypatch, primary
):
    """Exercise the artifact-reading notification path with a source-backed row."""
    idea_id = "idea-zero"
    idea_dir = tmp_path / idea_id
    idea_dir.mkdir()
    (idea_dir / "metrics.json").write_text(
        json.dumps({"status": "COMPLETED", primary: 999.0}),
        encoding="utf-8",
    )
    (idea_dir / "evaluation.json").write_text(
        json.dumps({"evaluation": {"score": 0.0}}), encoding="utf-8"
    )
    cfg = {
        "report": {
            "primary_metric": primary,
            "sort": "ascending",
            "columns": [
                {"key": primary, "source": "evaluation.json:evaluation.score"}
            ],
        }
    }
    _, values, value, reason = qualify_local_report_evidence(idea_dir, cfg)
    assert reason == "local_evidence_verified"
    assert value == 0.0
    row = {"id": idea_id, "primary_val": value, "values": values}
    notify = Mock()
    monkeypatch.setattr(leaderboard, "notify", notify)
    save_hash = Mock()
    reporter = leaderboard.NotificationProcessor(tmp_path, cfg)

    reporter._notify_finished(
        idea_id,
        None,
        cfg,
        primary,
        row_lookup={idea_id: row},
        rank_lookup={idea_id: 1},
        leaderboard=[row],
        view_lbs={},
        ideas={idea_id: {"title": "A valid zero", "config": {}}},
        save_config_hash_fn=save_hash,
    )

    notify.assert_called_once()
    event, payload, notification_cfg = notify.call_args.args
    assert event == "completed"
    assert payload["metric_name"] == primary
    assert payload["metric_value"] == "0.0000"
    assert payload["rank"] == 1
    assert notification_cfg is cfg
    save_hash.assert_called_once_with(idea_id, {})
