from orze.reporting.evidence import (
    count_dataset_metrics,
    dataset_metric_keys,
    minimum_dataset_coverage,
    load_local_report_evidence,
    efficiency_presentation_is_safe,
    qualification_is_presentable,
)


def test_asr_coverage_excludes_aggregate_and_timing_columns():
    report = {
        "primary_metric": "avg_wer",
        "min_datasets": 2,
        "columns": [
            {"key": "avg_wer"},
            {"key": "wer_a"},
            {"key": "wer_b"},
            {"key": "training_time"},
        ],
    }
    assert dataset_metric_keys(report) == ["wer_a", "wer_b"]
    assert minimum_dataset_coverage(
        report,
        metrics={"avg_wer": 1.0, "training_time": 5.0, "wer_a": 2.0},
    ) == (False, 1, 2)


def test_coverage_rejects_boolean_nan_and_infinity():
    report = {
        "min_datasets": 3,
        "columns": [{"key": key} for key in ("a", "b", "c", "d")],
    }
    assert count_dataset_metrics(
        report,
        values={"a": True, "b": float("nan"), "c": float("inf"), "d": 1.0},
    ) == 1


def test_configured_source_value_cannot_fall_back_to_metrics():
    report = {
        "min_datasets": 1,
        "columns": [{"key": "wer_a", "source": "external.json:wer"}],
    }
    assert minimum_dataset_coverage(
        report,
        values={"wer_a": None},
        metrics={"wer_a": 1.0},
    ) == (False, 0, 1)


def test_legacy_wer_fallback_only_applies_without_declared_columns():
    assert minimum_dataset_coverage(
        {"min_datasets": 1}, metrics={"wer_legacy": 1.0},
    ) == (True, 1, 1)
    assert minimum_dataset_coverage(
        {"min_datasets": 1, "columns": [{"key": "score"}]},
        metrics={"score": None, "wer_legacy": 1.0},
    ) == (False, 0, 1)


def test_invalid_minimum_fails_closed():
    report = {"min_datasets": "not-an-integer", "columns": [{"key": "a"}]}
    assert minimum_dataset_coverage(
        report, metrics={"a": 1.0}) == (False, 0, -1)
    assert minimum_dataset_coverage(
        {"min_datasets": True}, metrics={"a": 1.0}) == (False, 0, -1)
    assert minimum_dataset_coverage(
        {"min_datasets": 1.5}, metrics={"a": 1.0}) == (False, 0, -1)


def test_only_complete_qualification_summaries_are_presentable():
    valid = {
        "mode": "verified_local_artifact",
        "primary_metric": "score",
        "fallback_metrics_allowed": False,
        "accepted": 2,
        "rejected": {"local_metrics_missing": 1},
    }
    assert qualification_is_presentable(valid) is True
    for key, value in (
        ("mode", "unknown"),
        ("fallback_metrics_allowed", True),
        ("accepted", True),
        ("accepted", -1),
        ("rejected", {"reason": float("nan")}),
    ):
        invalid = dict(valid)
        invalid[key] = value
        assert qualification_is_presentable(invalid) is False


def test_only_internal_nonrank_presentation_is_safe():
    valid = {
        "claim_scope": "internal_research_efficiency",
        "qualification_applied": True,
        "evidence_label": "verified local artifacts",
        "leaderboard_rank_comparable": False,
    }
    assert efficiency_presentation_is_safe(valid) is True
    for key, value in (
        ("claim_scope", "official_rank"),
        ("qualification_applied", False),
        ("evidence_label", ""),
        ("leaderboard_rank_comparable", True),
    ):
        invalid = dict(valid)
        invalid[key] = value
        assert efficiency_presentation_is_safe(invalid) is False


def test_local_evidence_requires_completed_nonredirected_artifacts(tmp_path):
    idea_dir = tmp_path / "idea-a"
    idea_dir.mkdir()
    report = {
        "primary_metric": "avg",
        "columns": [
            {"key": "avg"},
            {"key": "external", "source": "external.json:value"},
        ],
    }
    (idea_dir / "metrics.json").write_text(
        '{"status":"COMPLETED","avg":2.0}', encoding="utf-8")
    (idea_dir / "external.json").write_text(
        '{"value":3.0}', encoding="utf-8")
    metrics, values, reason = load_local_report_evidence(idea_dir, report)
    assert reason == "local_evidence_loaded"
    assert metrics["status"] == "COMPLETED"
    assert values == {"avg": 2.0, "external": 3.0}

    (idea_dir / "external.json").unlink()
    outside = tmp_path / "outside.json"
    outside.write_text('{"value":0.0}', encoding="utf-8")
    (idea_dir / "external.json").symlink_to(outside)
    assert load_local_report_evidence(idea_dir, report)[2] == (
        "local_metric_source_path_invalid")


def test_local_evidence_rejects_redirected_idea_directory(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "metrics.json").write_text(
        '{"status":"COMPLETED","score":1.0}', encoding="utf-8")
    redirected = tmp_path / "idea-redirected"
    redirected.symlink_to(outside, target_is_directory=True)

    assert load_local_report_evidence(
        redirected,
        {"primary_metric": "score", "columns": [{"key": "score"}]},
    )[2] == "local_idea_dir_symlink"
