"""An explicit secondary metric need not also be a display column."""
import json

import pytest

from orze.reporting.evidence import qualify_local_report_evidence
from orze.reporting.leaderboard import update_report


@pytest.mark.parametrize("metrics,secondary,expected", [
    ({"penalty": 0}, "penalty", 0),
    ({"measurement": {"penalty": -2}}, "measurement.penalty", -2),
    ({"measurement.penalty": 0, "measurement": {"penalty": 9}},
     "measurement.penalty", 0),
])
def test_non_display_secondary_resolves_exact_metric(metrics, secondary, expected, tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({
        "status": "COMPLETED", "score": 5, **metrics,
    }))
    cfg = {"report": {"primary_metric": "score", "secondary_metric": secondary,
                      "sort": "ascending", "columns": [{"key": "score"}]}}
    _, values, primary, reason = qualify_local_report_evidence(tmp_path, cfg)
    assert primary == 5 and reason == "local_evidence_verified"
    assert values.get(secondary) == expected


def test_changing_non_display_secondary_invalidates_cached_values(tmp_path):
    ideas = {}
    for idea_id, penalty, bonus in (("idea-a", 1, 3), ("idea-b", 2, 0)):
        folder = tmp_path / idea_id
        folder.mkdir()
        (folder / "metrics.json").write_text(json.dumps({
            "status": "COMPLETED", "score": 5, "penalty": penalty, "bonus": bonus,
        }))
        ideas[idea_id] = {"title": idea_id}
    report = {"primary_metric": "score", "sort": "ascending",
              "secondary_metric": "penalty", "columns": [{"key": "score"}]}
    first = update_report(tmp_path, ideas, {"report": report})
    assert first[0]["id"] == "idea-a"
    assert first[0]["values"].get("penalty") == 1
    report["secondary_metric"] = "bonus"
    second = update_report(tmp_path, ideas, {"report": report})
    assert second[0]["id"] == "idea-b"
    assert second[0]["values"].get("bonus") == 0
