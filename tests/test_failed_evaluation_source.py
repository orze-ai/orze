"""Declared evaluation failures cannot contribute attractive source metrics.

This contract concerns cfg.eval_output, not arbitrary diagnostic JSON documents.
Absent status remains compatible with evaluators that emit measurements only.
"""

import json
import re

import pytest

from orze.engine.rebuild_state import rebuild_state_file
from orze.idea_lake import IdeaLake
from orze.reporting.evidence import qualify_local_report_evidence
from orze.reporting.leaderboard import update_report
from orze.research.context_builder import build_digest


_MISSING = object()


@pytest.fixture
def project(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    cfg = {
        "_project_root": str(tmp_path),
        "results_dir": str(results),
        "idea_lake_db": lake.db_path,
        "eval_script": "unused-evaluation-entrypoint.py",
        "eval_output": "assessment.json",
        "report": {
            "primary_metric": "score", "sort": "ascending",
            "columns": [
                {"key": "score", "source": "assessment.json:measurement.score"},
            ],
        },
        "metric_validation": {"max_value": {"score": 10}},
    }
    try:
        yield results, lake, cfg
    finally:
        lake.close()


def _publish(project, idea_id, score, status=_MISSING):
    results, lake, _ = project
    folder = results / idea_id
    folder.mkdir()
    (folder / "metrics.json").write_text(
        json.dumps({"status": "COMPLETED", "score": 99}), encoding="utf-8"
    )
    report = {"measurement": {"score": score}}
    if status is not _MISSING:
        report["status"] = status
    (folder / "assessment.json").write_text(json.dumps(report), encoding="utf-8")
    lake.insert(
        idea_id, idea_id, "{}", "", status="completed",
        eval_metrics={"status": "COMPLETED", "score": -999},
    )
    return folder


def _top_ids(digest):
    section = re.search(
        r"^## Top-[^\n]*\n(.*?)(?=^## |\Z)",
        digest, re.MULTILINE | re.DOTALL,
    )
    assert section, digest
    return re.findall(r"\bidea-[a-z0-9-]+\b", section.group(1))


@pytest.mark.parametrize("status", ["FAILED", "PARTIAL", "ERROR"])
@pytest.mark.parametrize("consumer", ["qualifier", "report", "rebuild", "digest"])
def test_explicit_failed_evaluation_source_cannot_rank_or_steer(
    project, status, consumer
):
    results, lake, cfg = project
    _publish(project, "idea-baseline", 1, "COMPLETED")
    folder = _publish(project, "idea-rejected", 0, status)

    if consumer == "qualifier":
        assert qualify_local_report_evidence(folder, cfg)[2] is None
    elif consumer == "report":
        rows = update_report(results, {}, cfg, lake=lake)
        assert [row["id"] for row in rows] == ["idea-baseline"]
    elif consumer == "rebuild":
        assert rebuild_state_file(results, cfg, lake=lake)["best_idea_id"] == "idea-baseline"
    else:
        digest = build_digest(results, cfg)
        assert _top_ids(digest) == ["idea-baseline"]
        assert "idea-rejected" not in digest


@pytest.mark.parametrize("status", [_MISSING, "COMPLETED"], ids=["absent", "completed"])
def test_measurement_only_or_completed_eval_output_preserves_source_zero(project, status):
    results, lake, cfg = project
    folder = _publish(project, "idea-zero", 0, status)

    assert qualify_local_report_evidence(folder, cfg)[2] == 0.0
    rows = update_report(results, {}, cfg, lake=lake)
    assert [(row["id"], row["primary_val"]) for row in rows] == [("idea-zero", 0.0)]
    assert rebuild_state_file(results, cfg, lake=lake)["best_idea_id"] == "idea-zero"
    assert _top_ids(build_digest(results, cfg)) == ["idea-zero"]


def test_unrelated_diagnostic_source_does_not_inherit_evaluation_status_contract(project):
    results, lake, cfg = project
    folder = _publish(project, "idea-zero", 0, "COMPLETED")
    (folder / "diagnostic.json").write_text(
        json.dumps({"status": "FAILED", "measurement": {"score": 0}}),
        encoding="utf-8",
    )
    cfg["report"]["columns"][0]["source"] = "diagnostic.json:measurement.score"

    assert qualify_local_report_evidence(folder, cfg)[2] == 0.0
    assert [row["id"] for row in update_report(results, {}, cfg, lake=lake)] == ["idea-zero"]
