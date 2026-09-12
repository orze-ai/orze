"""Independent S3 consumer review using actual local files and warm caches."""

from copy import deepcopy
import json

import pytest

from orze.core.config import DEFAULT_CONFIG
from orze.reporting.evidence import qualify_local_report_evidence
from orze.reporting.leaderboard import update_report


INVALID = "dataset_coverage_declaration_invalid"


def _project(tmp_path, metrics, report):
    folder = tmp_path / "idea-result"
    folder.mkdir()
    (folder / "metrics.json").write_text(
        json.dumps({"status": "COMPLETED", "score": 0, **metrics}),
        encoding="utf-8",
    )
    return folder, {"idea-result": {"title": "real offline result"}}, {"report": report}


def _payload(tmp_path):
    return json.loads((tmp_path / "_leaderboard.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("invalid_kind", ["explicit-null", "tuple-instead-of-list"])
def test_warm_offline_cache_cannot_authorize_an_invalid_declaration(tmp_path, invalid_kind):
    report = {
        "primary_metric": "score", "sort": "ascending", "min_datasets": 0,
        "columns": [{"key": "score"}, {"key": "part"}],
    }
    if invalid_kind == "tuple-instead-of-list":
        report["dataset_keys"] = ["part"]
        report["min_datasets"] = 1
    folder, ideas, cfg = _project(tmp_path, {"part": 1}, report)
    assert qualify_local_report_evidence(folder, cfg)[2:] == (0.0, "local_evidence_verified")
    assert [row["id"] for row in update_report(tmp_path, ideas, cfg)] == ["idea-result"]
    cache = json.loads((tmp_path / "_results_cache.json").read_text(encoding="utf-8"))
    assert cache["idea-result"]["row"]["evidence_qualified"] is True
    assert _payload(tmp_path)["top"][0]["idea_id"] == "idea-result"

    changed = deepcopy(cfg)
    changed["report"]["dataset_keys"] = (
        None if invalid_kind == "explicit-null" else ("part",)
    )
    assert qualify_local_report_evidence(folder, changed)[2:] == (None, INVALID)
    assert update_report(tmp_path, ideas, changed) == []
    assert _payload(tmp_path)["top"] == []


def test_harvest_columns_cannot_supply_missing_explicit_coverage_columns(tmp_path):
    report = {
        "primary_metric": "score", "sort": "ascending", "min_datasets": 1,
        "columns": [], "dataset_keys": ["part"],
    }
    folder, ideas, cfg = _project(tmp_path, {"part": 1}, report)
    cfg["metric_harvest"] = {"columns": [{"key": "part"}]}
    assert qualify_local_report_evidence(folder, cfg)[2:] == (None, INVALID)
    assert update_report(tmp_path, ideas, cfg) == []
    assert _payload(tmp_path)["top"] == []


def test_harvest_source_cannot_replace_the_declared_default_column_source(tmp_path):
    report = {
        "primary_metric": "score", "sort": "ascending", "min_datasets": 1,
        "columns": deepcopy(DEFAULT_CONFIG["report"]["columns"]),
        "dataset_keys": ["test_accuracy"],
    }
    folder, ideas, cfg = _project(tmp_path, {"test_accuracy": None}, report)
    (folder / "harvest.json").write_text(json.dumps({"proxy": 9}), encoding="utf-8")
    cfg["metric_harvest"] = {
        "columns": [{"key": "test_accuracy", "source": "harvest.json:proxy"}],
    }
    assert qualify_local_report_evidence(folder, cfg)[2:] == (
        None, "metric_coverage_below_min:0/1",
    )
    assert update_report(tmp_path, ideas, cfg) == []
    assert _payload(tmp_path)["top"] == []


def test_direct_qualifier_rejects_noniterable_columns_with_stable_reason(tmp_path):
    folder, _, cfg = _project(
        tmp_path, {"part": 1},
        {"primary_metric": "score", "columns": 1,
         "dataset_keys": ["part"], "min_datasets": 1},
    )
    assert qualify_local_report_evidence(folder, cfg)[2:] == (None, INVALID)
