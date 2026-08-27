import os
import sqlite3

import pytest

from orze.reporting.evidence import (
    authoritative_completed_idea_ids,
    count_dataset_metrics,
    dataset_metric_keys,
    minimum_dataset_coverage,
    load_local_report_evidence,
    efficiency_presentation_is_safe,
    qualification_is_presentable,
    qualify_authoritative_report_evidence,
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


def _lifecycle_db(path, rows):
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE ideas (idea_id TEXT, status TEXT)")
    connection.execute(
        "CREATE TABLE idea_state (idea_id TEXT, current_state TEXT)")
    for idea_id, status, state in rows:
        connection.execute("INSERT INTO ideas VALUES (?, ?)", (idea_id, status))
        connection.execute(
            "INSERT INTO idea_state VALUES (?, ?)", (idea_id, state))
    connection.commit()
    connection.close()


def test_authoritative_completion_requires_both_lifecycle_views(tmp_path):
    db_path = tmp_path / "idea_lake.db"
    _lifecycle_db(db_path, [
        ("idea-good", "completed", "COMPLETE"),
        ("idea-state-failed", "completed", "FAILED"),
        ("idea-status-failed", "failed", "COMPLETE"),
        ("../invalid", "completed", "COMPLETE"),
    ])

    completed, reason = authoritative_completed_idea_ids(db_path)

    assert reason == "authoritative_lifecycle_loaded"
    assert completed == {"idea-good"}


def test_authoritative_lifecycle_rejects_wal_without_mutating_it(tmp_path):
    db_path = tmp_path / "wal.db"
    _lifecycle_db(db_path, [("idea-good", "completed", "COMPLETE")])
    connection = sqlite3.connect(db_path)
    assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    before = db_path.stat()

    assert authoritative_completed_idea_ids(db_path) == (
        set(), "authoritative_lifecycle_database_policy_invalid")
    assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    after = db_path.stat()
    assert (after.st_size, after.st_mtime_ns) == (
        before.st_size, before.st_mtime_ns)
    connection.close()


def test_authoritative_lifecycle_inspection_never_creates_database(tmp_path):
    missing = tmp_path / "missing.db"
    completed, reason = authoritative_completed_idea_ids(missing)
    assert completed == set()
    assert reason == "authoritative_lifecycle_database_unavailable"
    assert not missing.exists()


@pytest.mark.parametrize("kind", ["corrupt", "missing_state_schema"])
def test_authoritative_lifecycle_invalid_database_fails_closed(tmp_path, kind):
    db_path = tmp_path / "invalid.db"
    if kind == "corrupt":
        db_path.write_bytes(b"not sqlite and no result content")
    else:
        connection = sqlite3.connect(db_path)
        connection.execute("CREATE TABLE ideas (idea_id TEXT, status TEXT)")
        connection.commit()
        connection.close()

    assert authoritative_completed_idea_ids(db_path) == (
        set(), "authoritative_lifecycle_database_invalid")


@pytest.mark.parametrize("redirect", ["symlink", "parent_symlink", "hardlink"])
def test_authoritative_lifecycle_rejects_redirected_database(tmp_path, redirect):
    real = tmp_path / "real.db"
    _lifecycle_db(real, [("idea-good", "completed", "COMPLETE")])
    candidate = tmp_path / "candidate.db"
    if redirect == "symlink":
        candidate.symlink_to(real)
        expected = "authoritative_lifecycle_database_redirected"
    elif redirect == "parent_symlink":
        real_parent = tmp_path / "real-parent"
        real_parent.mkdir()
        real.rename(real_parent / "idea_lake.db")
        linked_parent = tmp_path / "linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        candidate = linked_parent / "idea_lake.db"
        expected = "authoritative_lifecycle_database_redirected"
    else:
        os.link(real, candidate)
        expected = "authoritative_lifecycle_database_redirected"

    assert authoritative_completed_idea_ids(candidate) == (set(), expected)


def test_authoritative_qualification_rejects_artifact_only_completion(tmp_path):
    results = tmp_path / "results"
    idea = results / "idea-artifact-only"
    idea.mkdir(parents=True)
    (idea / "metrics.json").write_text(
        '{"status":"COMPLETED","score":1.0}', encoding="utf-8")
    cfg = {"report": {
        "primary_metric": "score",
        "columns": [{"key": "score"}],
    }}

    assert qualify_authoritative_report_evidence(
        "idea-artifact-only", results, cfg, set()
    )[3] == "authoritative_lifecycle_not_complete"


def test_authoritative_qualification_rejects_tainted_evidence(tmp_path):
    results = tmp_path / "results"
    idea = results / "idea-tainted"
    idea.mkdir(parents=True)
    (idea / "metrics.json").write_text(
        '{"status":"COMPLETED","score":1.0,"tainted_leakage":true}',
        encoding="utf-8",
    )
    cfg = {"report": {
        "primary_metric": "score",
        "columns": [{"key": "score"}],
    }}

    assert qualify_authoritative_report_evidence(
        "idea-tainted", results, cfg, {"idea-tainted"}
    )[3] == "local_evidence_tainted_leakage"


@pytest.mark.parametrize("valid", [False, True])
def test_benchmark_steering_requires_current_receipt(
        tmp_path, monkeypatch, valid):
    results = tmp_path / "results"
    idea = results / "idea-contract"
    idea.mkdir(parents=True)
    (idea / "metrics.json").write_text(
        '{"status":"COMPLETED","score":1.0}', encoding="utf-8")
    cfg = {"report": {
        "primary_metric": "score",
        "columns": [{"key": "score"}],
        "benchmark_contract": {"enabled": True},
    }}
    monkeypatch.setattr(
        "orze.core.benchmark_contract.validate_benchmark_receipt",
        lambda *args, **kwargs: (
            (True, "benchmark_contract_verified") if valid
            else (False, "benchmark_receipt_missing")
        ),
    )

    result = qualify_authoritative_report_evidence(
        "idea-contract", results, cfg, {"idea-contract"}
    )
    assert result[2:] == (
        (1.0, "benchmark_evidence_verified") if valid
        else (None, "benchmark_receipt_missing")
    )
