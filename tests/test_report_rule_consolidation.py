"""S1 characterization, plus two explicitly new architecture requirements.

Behavior cases describe existing rules, including their different fallbacks
and exceptions. The architecture cases are expected to fail before the
consolidation; those failures are not historical product defects. No execution
or authoritative scientific qualification is granted by an archive value.
"""

import pytest

from orze.engine import rebuild_state
from orze.reporting import evidence


@pytest.mark.parametrize(
    "report,expected",
    [
        pytest.param({}, [], id="empty-columns"),
        pytest.param(
            {"primary_metric": "score", "columns": [
                {"key": "z"}, {"key": "score"}, {"key": "z"},
                None, "ignored", {}, {"key": ""}, {"key": "a"},
            ]},
            ["z", "score", "z", "a"], id="ordinary-order-duplicates-and-filtering",
        ),
        pytest.param(
            {"primary_metric": "wer_avg", "columns": [
                {"key": "score"}, {"key": "wer_b"}, {"key": "wer_avg"},
                {"key": "wer_a"}, {"key": "wer_b"},
            ]},
            ["wer_b", "wer_a", "wer_b"], id="wer-excludes-primary-preserves-order",
        ),
        pytest.param(
            {"primary_metric": "wer_avg", "columns": [
                {"key": "wer_avg"}, {"key": "score"},
            ]},
            ["wer_avg", "score"], id="only-primary-wer-falls-back-to-columns",
        ),
    ],
)
def test_dataset_selection_existing_behavior(report, expected):
    assert rebuild_state._report_dataset_keys(report) == expected
    assert evidence.dataset_metric_keys(report) == expected


@pytest.mark.parametrize(
    "report,error",
    [
        pytest.param({"columns": [{"key": 1}]}, AttributeError, id="truthy-nonstring-key"),
        pytest.param({"columns": 1}, TypeError, id="noniterable-columns"),
    ],
)
def test_dataset_selection_preserves_existing_exceptions(report, error):
    with pytest.raises(error):
        rebuild_state._report_dataset_keys(report)
    with pytest.raises(error):
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
            {"score": 1, "part": 0}, ["part", "part"], 2, 1.0, 2,
            id="duplicate-columns-count-twice",
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
    report = {"primary_metric": "score", "columns": [{"key": key} for key in keys]}
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
