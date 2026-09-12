"""S1 archive/architecture controls with explicitly versioned S3 live rules.

The original live name/duplicate/exception characterization remains at Git
69ff7babb57371632fb3048db781b24a3545548a. S3 intentionally replaces those live
rules with explicit declarations; archive behavior and architecture stay fixed.
An archive value grants no authoritative scientific qualification.
"""

import pytest

from orze.engine import rebuild_state
from orze.reporting import evidence


@pytest.mark.parametrize(
    "report,expected",
    [
        pytest.param({}, [], id="empty-columns"),
        pytest.param(
            {"primary_metric": "score", "dataset_keys": ["z", "score", "a"],
             "columns": [{"key": "a"}, {"key": "score"}, {"key": "z"}]},
            ["z", "score", "a"], id="s3-explicit-order-no-duplicate-counting",
        ),
        pytest.param(
            {"primary_metric": "wer_avg", "dataset_keys": ["wer_b", "wer_a"], "columns": [
                {"key": "score"}, {"key": "wer_b"}, {"key": "wer_avg"},
                {"key": "wer_a"},
            ]},
            ["wer_b", "wer_a"], id="s3-explicit-components-exclude-primary",
        ),
        pytest.param(
            {"primary_metric": "wer_avg", "dataset_keys": ["wer_avg", "score"], "columns": [
                {"key": "wer_avg"}, {"key": "score"},
            ]},
            ["wer_avg", "score"], id="s3-primary-counts-only-when-declared",
        ),
    ],
)
def test_dataset_selection_s3_declared_behavior(report, expected):
    assert rebuild_state._report_dataset_keys(report) == expected
    assert evidence.dataset_metric_keys(report) == expected


@pytest.mark.parametrize(
    "report,error",
    [
        pytest.param({"columns": [{"key": 1}], "dataset_keys": ["part"]},
                     ValueError, id="s3-no-matching-string-column"),
        pytest.param({"columns": 1, "dataset_keys": ["part"]},
                     ValueError, id="s3-noniterable-columns"),
    ],
)
def test_dataset_selection_s3_stable_declaration_errors(report, error):
    with pytest.raises(error, match="^dataset_coverage_declaration_invalid$"):
        rebuild_state._report_dataset_keys(report)
    with pytest.raises(error, match="^dataset_coverage_declaration_invalid$"):
        evidence.dataset_metric_keys(report)


@pytest.mark.parametrize(
    "metrics,expected",
    [
        pytest.param({"score": 0}, 0.0, id="zero-is-valid"),
        pytest.param({"score": -2}, -2.0, id="negative-is-valid"),
        pytest.param({"score": True}, None, id="boolean-is-not-a-score"),
        pytest.param({"score": float("nan")}, None, id="nan-is-not-a-score"),
        pytest.param({"score": float("inf")}, None, id="positive-infinity"),
        pytest.param({"score": float("-inf")}, None, id="negative-infinity"),
        pytest.param({"score": "3"}, None, id="numeric-string-is-not-a-score"),
        pytest.param({}, None, id="missing-primary"),
        pytest.param([], None, id="non-dict-metrics"),
    ],
)
def test_archive_primary_value_existing_behavior(metrics, expected):
    actual = rebuild_state._eligible_metric(metrics, "score", 0, [])
    assert actual == expected
    assert actual is None or type(actual) is float


@pytest.mark.parametrize(
    "metrics,keys,minimum,archive_value,report_count",
    [
        pytest.param(
            {"score": 0, "wer_actual": 2}, ["missing"], 1, 0.0, 0,
            id="archive-fallback-even-with-declared-missing-columns",
        ),
        pytest.param(
            {"score": -2, "part": 0, "wer_a": 1, "wer_b": 2},
            ["part", "missing"], 2, None, 1,
            id="partial-declared-coverage-does-not-add-fallback",
        ),
        pytest.param(
            {"score": 1, "part": 0}, ["part", "part"], 2, 1.0, 1,
            id="s3-archive-duplicates-remain-live-declaration-is-unique",
        ),
        pytest.param(
            {"score": 3, "boolean": True, "nonfinite": float("nan"), "zero": 0, "negative": -1},
            ["boolean", "nonfinite", "zero", "negative"], 2, 3.0, 2,
            id="coverage-rejects-bool-nonfinite-but-counts-zero-negative",
        ),
    ],
)
def test_archive_and_report_coverage_keep_their_distinct_rules(
    metrics, keys, minimum, archive_value, report_count,
):
    assert rebuild_state._eligible_metric(metrics, "score", minimum, keys) == archive_value
    # Keep the original archive call and expected value above. Only the live
    # S3 declaration is unique; duplicate archive inputs do not become evidence.
    live_keys = list(dict.fromkeys(keys))
    report = {"primary_metric": "score", "dataset_keys": live_keys,
              "columns": [{"key": key} for key in live_keys]}
    assert evidence.count_dataset_metrics(report, metrics=metrics) == report_count


@pytest.mark.parametrize(
    "metrics,minimum,keys,error",
    [
        pytest.param({"score": 1, 7: 2}, 1, [], AttributeError, id="fallback-nonstring-key"),
        pytest.param({"score": 10 ** 1000}, 0, [], OverflowError, id="primary-float-overflow"),
        pytest.param({"score": 1}, "1", [], TypeError, id="noncomparable-minimum"),
    ],
)
def test_archive_preserves_existing_exceptions(metrics, minimum, keys, error):
    with pytest.raises(error):
        rebuild_state._eligible_metric(metrics, "score", minimum, keys)


def test_architecture_dataset_selector_is_the_public_function():
    assert rebuild_state._report_dataset_keys is evidence.dataset_metric_keys


def test_architecture_archive_filter_is_owned_by_reporting_evidence():
    assert rebuild_state._eligible_metric.__module__ == "orze.reporting.evidence"
