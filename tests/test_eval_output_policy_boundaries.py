"""Configured evaluation output is policy evidence even for metrics-only scores."""

import json
import os

import pytest

from orze.idea_lake import IdeaLake
from orze.reporting.evidence import (
    authoritative_completed_idea_ids,
    qualify_authoritative_report_evidence_with_identity,
    qualify_local_report_evidence,
)
from orze.reporting.leaderboard import update_report


@pytest.fixture
def project(tmp_path):
    results = tmp_path / "results"
    folder = results / "idea-output-policy"
    folder.mkdir(parents=True)
    (folder / "metrics.json").write_text(
        '{"status":"COMPLETED","quality":0}', encoding="utf-8",
    )
    (folder / "assessment.json").write_text(
        '{"status":"COMPLETED","evaluation_revision":1}', encoding="utf-8",
    )
    lake = IdeaLake(tmp_path / "ideas.db")
    lake.insert("idea-output-policy", "Output policy", "{}", "", status="completed")
    cfg = {
        "_project_root": str(tmp_path),
        "results_dir": str(results),
        "idea_lake_db": str(lake.db_path),
        "eval_script": "not-executed.py",
        "eval_output": "assessment.json",
        "report": {
            "primary_metric": "quality", "sort": "ascending",
            # The numerical observation is genuinely in metrics.json. The
            # evaluator's output is still its declared completion contract.
            "columns": [{"key": "quality"}],
        },
    }
    try:
        yield results, folder, lake, cfg
    finally:
        lake.close()


def _rank(project):
    results, _, lake, cfg = project
    return update_report(results, {}, cfg, lake=lake)


def _identity(project):
    results, _, lake, cfg = project
    completed, reason = authoritative_completed_idea_ids(lake.db_path)
    assert reason == "authoritative_lifecycle_loaded"
    _, _, value, reason, identity = (
        qualify_authoritative_report_evidence_with_identity(
            "idea-output-policy", results, cfg, completed,
        )
    )
    assert value == 0.0, reason
    assert identity is not None
    return identity


def test_failed_configured_output_is_not_ignored_when_primary_comes_from_metrics(project):
    _, folder, _, cfg = project
    (folder / "assessment.json").write_text('{"status":"FAILED"}', encoding="utf-8")

    assert qualify_local_report_evidence(folder, cfg)[2] is None


def test_output_status_change_invalidates_a_warm_metrics_only_report_cache(project):
    _, folder, _, _ = project
    assert [(row["id"], row["primary_val"]) for row in _rank(project)] == [
        ("idea-output-policy", 0.0),
    ]
    output = folder / "assessment.json"
    before = output.stat()
    output.write_text('{"status":"FAILED"}', encoding="utf-8")
    os.utime(output, ns=(before.st_atime_ns, before.st_mtime_ns))

    assert _rank(project) == []


def test_switching_configured_output_invalidates_cache_without_artifact_rewrite(project):
    results, folder, _, cfg = project
    (folder / "failed-assessment.json").write_text(
        '{"status":"FAILED"}', encoding="utf-8",
    )
    assert [row["id"] for row in _rank(project)] == ["idea-output-policy"]
    cache = results / "_results_cache.json"
    old_hash = json.loads(cache.read_text())["idea-output-policy"]["col_hash"]

    cfg["eval_output"] = "failed-assessment.json"

    assert _rank(project) == []
    new_hash = json.loads(cache.read_text())["idea-output-policy"]["col_hash"]
    assert new_hash != old_hash


def test_unused_output_without_an_evaluator_has_no_implicit_status_authority(project):
    _, folder, _, cfg = project
    cfg.pop("eval_script")
    (folder / "assessment.json").write_text('{"status":"FAILED"}', encoding="utf-8")

    assert qualify_local_report_evidence(folder, cfg)[2] == 0.0
    assert [row["id"] for row in _rank(project)] == ["idea-output-policy"]


def test_activating_evaluator_invalidates_warm_cache_for_existing_failed_output(project):
    results, folder, _, cfg = project
    cfg.pop("eval_script")
    (folder / "assessment.json").write_text('{"status":"FAILED"}', encoding="utf-8")
    assert [row["id"] for row in _rank(project)] == ["idea-output-policy"]
    cache = results / "_results_cache.json"
    old_hash = json.loads(cache.read_text())["idea-output-policy"]["col_hash"]

    cfg["eval_script"] = "not-executed.py"

    assert _rank(project) == []
    new_hash = json.loads(cache.read_text())["idea-output-policy"]["col_hash"]
    assert new_hash != old_hash


def test_configured_output_contributes_to_authoritative_evidence_identity(project):
    _, folder, _, _ = project
    before = _identity(project)
    (folder / "assessment.json").write_text(
        '{"status":"COMPLETED","evaluation_revision":2}', encoding="utf-8",
    )

    assert _identity(project) != before


@pytest.mark.parametrize("redirect", ["symlink", "parent_path"])
def test_configured_output_cannot_escape_idea_evidence_even_with_valid_raw_metric(
    project, redirect,
):
    results, folder, _, cfg = project
    outside = results / "outside.json"
    outside.write_text('{"status":"COMPLETED"}', encoding="utf-8")
    if redirect == "symlink":
        (folder / "redirected-output.json").symlink_to(outside)
        cfg["eval_output"] = "redirected-output.json"
    else:
        cfg["eval_output"] = "../outside.json"
    before = outside.read_bytes()

    assert qualify_local_report_evidence(folder, cfg)[2] is None
    assert outside.read_bytes() == before
