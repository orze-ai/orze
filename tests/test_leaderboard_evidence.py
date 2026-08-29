import json
import os

import pytest

import orze.reporting.leaderboard as leaderboard_module
from orze.reporting.leaderboard import update_report
from orze.reporting.evidence import (
    evidence_content_sha256,
    qualify_authoritative_report_evidence_with_identity,
    report_evidence_paths,
)


def _cfg(*, source=False):
    column = {"key": "score", "label": "Score"}
    if source:
        column["source"] = "evaluation.json:score"
    return {
        "report": {
            "title": "Qualified local results",
            "primary_metric": "score",
            "sort": "descending",
            "columns": [column],
        },
    }


def _write_result(results, idea_id, metrics, *, source=None):
    idea_dir = results / idea_id
    idea_dir.mkdir(parents=True)
    (idea_dir / "metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8")
    if source is not None:
        (idea_dir / "evaluation.json").write_text(
            json.dumps(source), encoding="utf-8")
    return idea_dir


def _ranked_report(results):
    report = (results / "report.md").read_text(encoding="utf-8")
    ranked = report.split("## Results", 1)[1]
    for marker in (
        "## Unranked:", "## Sweep Details", "## Failed", "## Queue",
        "## Score Ceiling Warning", "## Config Diversity",
    ):
        ranked = ranked.split(marker, 1)[0]
    return ranked


def test_decision_identity_covers_benchmark_and_exposure_evidence(tmp_path):
    results = tmp_path / "results"
    _write_result(
        results,
        "idea-benchmark",
        {"status": "COMPLETED", "score": 1.0},
    )
    cfg = _cfg()
    cfg["_project_root"] = str(tmp_path)
    cfg["_orze_dir"] = str(tmp_path / ".orze")
    cfg["report"]["benchmark_contract"] = {
        "receipt": "benchmark.json",
    }
    (tmp_path / ".orze").mkdir()
    paths = report_evidence_paths("idea-benchmark", results, cfg)
    names = {path.name for path in paths}

    assert {
        "metrics.json",
        "benchmark.json",
        "_benchmark_evaluation.json",
        "_benchmark_exposures.jsonl",
    }.issubset(names)
    prior = evidence_content_sha256(paths)
    for index, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"evidence-{index}\n", encoding="utf-8")
        current = evidence_content_sha256(paths)
        assert current != prior
        prior = current


@pytest.mark.parametrize("redirect", ["idea_symlink", "metrics_hardlink"])
def test_decision_identity_never_reads_redirected_or_hardlinked_evidence(
        tmp_path, redirect):
    results = tmp_path / "results"
    results.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_metrics = outside / "metrics.json"
    outside_metrics.write_text(
        json.dumps({"status": "COMPLETED", "score": 999.0}),
        encoding="utf-8",
    )
    idea_dir = results / "idea-redirected"
    if redirect == "idea_symlink":
        idea_dir.symlink_to(outside, target_is_directory=True)
    else:
        idea_dir.mkdir()
        os.link(outside_metrics, idea_dir / "metrics.json")

    _, _, value, reason, digest = (
        qualify_authoritative_report_evidence_with_identity(
            "idea-redirected",
            results,
            _cfg(),
            {"idea-redirected"},
        )
    )

    assert value is None
    assert reason == "report_evidence_identity_unavailable"
    assert digest is None


def test_local_cache_invalidates_on_same_size_backdated_source_rewrite(
        tmp_path):
    results = tmp_path / "results"
    idea_dir = _write_result(
        results,
        "idea-current",
        {"status": "COMPLETED"},
        source={"score": 1.0},
    )
    ideas = {"idea-current": {"title": "Current"}}

    first = update_report(results, ideas, _cfg(source=True))
    cache_before = json.loads(
        (results / "_results_cache.json").read_text(encoding="utf-8"))
    source_path = idea_dir / "evaluation.json"
    original = source_path.stat()
    source_path.write_text('{"score": 9.0}', encoding="utf-8")
    os.utime(source_path, ns=(original.st_atime_ns, original.st_mtime_ns))

    second = update_report(results, ideas, _cfg(source=True))
    cache_after = json.loads(
        (results / "_results_cache.json").read_text(encoding="utf-8"))

    assert first[0]["primary_val"] == 1.0
    assert second[0]["primary_val"] == 9.0
    assert "9.0" in _ranked_report(results)
    assert cache_before["idea-current"]["evidence_hash"] != (
        cache_after["idea-current"]["evidence_hash"])


def test_unchanged_local_evidence_uses_metadata_fast_path(
        tmp_path, monkeypatch):
    results = tmp_path / "results"
    _write_result(
        results,
        "idea-unchanged",
        {"status": "COMPLETED", "score": 1.0},
    )
    ideas = {"idea-unchanged": {"title": "Unchanged"}}
    update_report(results, ideas, _cfg())
    content_hash_calls = 0
    original = leaderboard_module._evidence_content_hash

    def counted(paths):
        nonlocal content_hash_calls
        content_hash_calls += 1
        return original(paths)

    monkeypatch.setattr(
        leaderboard_module, "_evidence_content_hash", counted)

    update_report(results, ideas, _cfg())

    assert content_hash_calls == 0


@pytest.mark.parametrize("value", [True, float("nan"), float("inf")])
def test_boolean_and_nonfinite_primary_values_are_unranked(tmp_path, value):
    results = tmp_path / "results"
    _write_result(
        results,
        "idea-invalid",
        {"status": "COMPLETED", "score": value},
    )

    completed = update_report(
        results, {"idea-invalid": {"title": "Invalid"}}, _cfg())
    report = (results / "report.md").read_text(encoding="utf-8")
    cache = json.loads(
        (results / "_leaderboard.json").read_text(encoding="utf-8"))

    assert completed == []
    assert "idea-invalid" not in _ranked_report(results)
    assert "primary_metric_missing_or_nonfinite" in report or (
        "local_metric_validation_failed" in report)
    assert cache["top"] == []
    assert cache["evidence_qualification"]["accepted"] == 0
    assert sum(cache["evidence_qualification"]["rejected"].values()) == 1


def test_failed_metrics_cannot_rank_via_stray_eval_output(tmp_path):
    results = tmp_path / "results"
    idea_dir = _write_result(
        results,
        "idea-failed",
        {"status": "FAILED", "score": 100.0, "error": "failed"},
    )
    (idea_dir / "eval_report.json").write_text(
        json.dumps({"score": 100.0}), encoding="utf-8")

    completed = update_report(
        results, {"idea-failed": {"title": "Failed"}}, _cfg())
    leaderboard = json.loads(
        (results / "_leaderboard.json").read_text(encoding="utf-8"))

    assert completed == []
    assert leaderboard["top"] == []
    assert "idea-failed" not in _ranked_report(results)


@pytest.mark.parametrize("source_mode", ["traversal", "symlink"])
def test_redirected_local_source_is_never_ranked(tmp_path, source_mode):
    results = tmp_path / "results"
    idea_dir = _write_result(
        results,
        "idea-redirected",
        {"status": "COMPLETED"},
    )
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"score": 999.0}), encoding="utf-8")
    cfg = _cfg(source=True)
    if source_mode == "traversal":
        cfg["report"]["columns"][0]["source"] = "../outside.json:score"
    else:
        (idea_dir / "evaluation.json").symlink_to(outside)

    completed = update_report(
        results, {"idea-redirected": {"title": "Redirected"}}, cfg)
    report = (results / "report.md").read_text(encoding="utf-8")

    assert completed == []
    assert "idea-redirected" not in _ranked_report(results)
    assert "local_metric_source_path_invalid" in report


def test_metric_validation_policy_is_applied_to_local_ordering(tmp_path):
    results = tmp_path / "results"
    _write_result(
        results,
        "idea-out-of-policy",
        {"status": "COMPLETED", "score": -1.0},
    )
    cfg = _cfg()
    cfg["metric_validation"] = {"min_value": {"score": 0.0}}

    completed = update_report(
        results, {"idea-out-of-policy": {"title": "Invalid"}}, cfg)

    assert completed == []
    assert "local_metric_validation_failed" in (
        results / "report.md").read_text(encoding="utf-8")


def test_metric_validation_applies_to_resolved_source_value(tmp_path):
    results = tmp_path / "results"
    _write_result(
        results,
        "idea-source-out-of-policy",
        {"status": "COMPLETED", "score": 10.0},
        source={"score": -1.0},
    )
    cfg = _cfg(source=True)
    cfg["metric_validation"] = {"min_value": {"score": 0.0}}

    completed = update_report(
        results,
        {"idea-source-out-of-policy": {"title": "Invalid source"}},
        cfg,
    )

    assert completed == []
    assert "local_metric_validation_failed" in (
        results / "report.md").read_text(encoding="utf-8")


def test_valid_zero_is_policy_driven_not_hardcoded_as_vacuous(tmp_path):
    results = tmp_path / "results"
    _write_result(
        results,
        "idea-valid-zero",
        {"status": "COMPLETED", "score": 0.0, "training_time": 1},
    )

    completed = update_report(
        results, {"idea-valid-zero": {"title": "Valid zero"}}, _cfg())

    assert [row["id"] for row in completed] == ["idea-valid-zero"]


def test_corrupt_cached_row_is_rebuilt_from_current_evidence(tmp_path):
    results = tmp_path / "results"
    _write_result(
        results,
        "idea-current",
        {"status": "COMPLETED", "score": 1.0},
    )
    ideas = {"idea-current": {"title": "Current"}}
    update_report(results, ideas, _cfg())
    cache_path = results / "_results_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    cache["idea-current"]["row"]["primary_val"] = 999.0
    cache_path.write_text(json.dumps(cache), encoding="utf-8")

    completed = update_report(results, ideas, _cfg())

    assert completed[0]["primary_val"] == 1.0
    repaired = json.loads(cache_path.read_text(encoding="utf-8"))
    assert repaired["idea-current"]["row"]["primary_val"] == 1.0


def test_unqualified_completed_rows_remain_in_pipeline_accounting(tmp_path):
    results = tmp_path / "results"
    _write_result(
        results,
        "idea-unqualified",
        {"status": "COMPLETED", "score": True},
    )

    update_report(
        results, {"idea-unqualified": {"title": "Unqualified"}}, _cfg())
    report = (results / "report.md").read_text(encoding="utf-8")

    assert "| 1 | 1 | 0 | 0 | 0 |" in report
    assert "Accepted completed rows: 0" in report
    assert "Rejected completed rows: 1" in report
