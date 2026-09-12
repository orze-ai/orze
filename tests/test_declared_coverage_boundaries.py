"""S3 declared coverage requirements and unchanged source/numeric controls.

These use existing public helpers: baseline failures describe behavior, not a
missing API. Files written by the source controls belong only to pytest tmp_path.
"""

import copy
import json

import pytest

from orze.reporting import evidence


INVALID = "dataset_coverage_declaration_invalid"
UNDECLARED = "dataset_coverage_not_declared"


def _report(keys, *, minimum=0):
    return {
        "primary_metric": "objective",
        "columns": [{"key": key} for key in keys],
        "dataset_keys": list(keys),
        "min_datasets": minimum,
    }


def test_explicit_order_and_primary_membership_are_not_inferred_from_names():
    report = _report(["objective", "wer_part", "plain_part", "elapsed"])
    report["dataset_keys"] = ["plain_part", "objective", "wer_part"]
    before = copy.deepcopy(report)
    selected = evidence.dataset_metric_keys(report)
    assert selected == ["plain_part", "objective", "wer_part"]
    selected.append("caller-only")
    assert report == before
    assert evidence.minimum_dataset_coverage(
        {**report, "min_datasets": 3},
        values={"plain_part": -2, "objective": 0, "wer_part": 1, "elapsed": 9},
    ) == (True, 3, 3)


@pytest.mark.parametrize(
    "declaration",
    [
        pytest.param(None, id="null"),
        pytest.param(True, id="boolean"),
        pytest.param("part", id="string"),
        pytest.param(("part",), id="tuple"),
        pytest.param({"part"}, id="set"),
        pytest.param({"part": 1}, id="mapping"),
        pytest.param([True], id="boolean-member"),
        pytest.param([1], id="integer-member"),
        pytest.param([""], id="empty-member"),
        pytest.param([" \t"], id="whitespace-member"),
        pytest.param(["part", "part"], id="duplicate-member"),
        pytest.param(["\ud800"], id="not-utf8-encodable"),
    ],
)
def test_invalid_explicit_declaration_raises_stable_reason(declaration):
    report = _report(["part"])
    report["dataset_keys"] = declaration
    with pytest.raises(ValueError, match=f"^{INVALID}$"):
        evidence.dataset_metric_keys(report)
    with pytest.raises(ValueError, match=f"^{INVALID}$"):
        evidence.minimum_dataset_coverage(report, values={"part": 0})


@pytest.mark.parametrize("size", [256, 257])
def test_declaration_count_bound(size):
    keys = [f"part_{index}" for index in range(size)]
    report = _report(keys, minimum=size)
    if size == 256:
        assert evidence.dataset_metric_keys(report) == keys
        assert evidence.minimum_dataset_coverage(
            report, values=dict.fromkeys(keys, 0),
        ) == (True, 256, 256)
    else:
        with pytest.raises(ValueError, match=f"^{INVALID}$"):
            evidence.dataset_metric_keys(report)


@pytest.mark.parametrize("extra", ["", "a"], ids=["1024-bytes", "1025-bytes"])
def test_key_limit_counts_utf8_bytes_not_characters(extra):
    key = "é" * 512 + extra
    report = _report([key], minimum=1)
    assert len(key.encode("utf-8")) == 1024 + len(extra)
    if not extra:
        assert evidence.dataset_metric_keys(report) == [key]
        assert evidence.minimum_dataset_coverage(report, values={key: -1}) == (True, 1, 1)
    else:
        with pytest.raises(ValueError, match=f"^{INVALID}$"):
            evidence.dataset_metric_keys(report)


@pytest.mark.parametrize(
    "columns",
    [
        pytest.param([], id="no-explicit-column"),
        pytest.param([{"key": "part"}, {"key": "part"}], id="duplicate-column"),
        pytest.param(
            [{"key": "part", "source": "a.json:x"},
             {"key": "part", "source": "b.json:y"}],
            id="two-source-columns",
        ),
        pytest.param("part", id="not-a-column-list"),
    ],
)
def test_each_declared_key_has_exactly_one_explicit_column(columns):
    report = _report(["part"], minimum=1)
    report["columns"] = columns
    with pytest.raises(ValueError, match=f"^{INVALID}$"):
        evidence.dataset_metric_keys(report)


def test_benchmark_required_metrics_are_an_explicit_fallback():
    report = _report(["objective", "wer_unrelated", "a", "b"], minimum=2)
    del report["dataset_keys"]
    report["benchmark_contract"] = {"required_metrics": ["b", "a"]}
    assert evidence.dataset_metric_keys(report) == ["b", "a"]
    assert evidence.minimum_dataset_coverage(
        report, values={"a": 0, "b": -2, "wer_unrelated": 5, "objective": 9},
    ) == (True, 2, 2)


def test_explicit_superset_preserves_its_order_without_implicit_union():
    report = _report(["a", "b", "c"])
    report["dataset_keys"] = ["c", "b", "a"]
    report["benchmark_contract"] = {"required_metrics": ["a", "b"]}
    assert evidence.dataset_metric_keys(report) == ["c", "b", "a"]


@pytest.mark.parametrize(
    "declaration",
    [
        pytest.param(["a"], id="missing-required-component"),
        pytest.param([], id="explicit-empty-does-not-fallback"),
        pytest.param(None, id="explicit-null-does-not-fallback"),
    ],
)
def test_explicit_declaration_cannot_erase_benchmark_requirements(declaration):
    report = _report(["a", "b"])
    report["dataset_keys"] = declaration
    report["benchmark_contract"] = {"required_metrics": ["a", "b"]}
    with pytest.raises(ValueError, match=f"^{INVALID}$"):
        evidence.dataset_metric_keys(report)


@pytest.mark.parametrize(
    "required",
    [
        pytest.param(["a", "a"], id="duplicate-required"),
        pytest.param("a", id="required-not-list"),
        pytest.param(["missing"], id="required-column-missing"),
    ],
)
def test_benchmark_fallback_is_validated_before_use(required):
    report = _report(["a"])
    del report["dataset_keys"]
    report["benchmark_contract"] = {"required_metrics": required}
    with pytest.raises(ValueError, match=f"^{INVALID}$"):
        evidence.dataset_metric_keys(report)


@pytest.mark.parametrize(
    "source_value,create_source,expected_count",
    [
        pytest.param(None, False, 0, id="missing-file"),
        pytest.param(None, True, 0, id="explicit-null"),
        pytest.param(0, True, 1, id="zero"),
        pytest.param(-2, True, 1, id="negative"),
    ],
)
def test_exact_declared_source_survives_loading_without_raw_proxy_fallback(
    tmp_path, source_value, create_source, expected_count,
):
    report = _report(["part"], minimum=1)
    report["columns"][0]["source"] = "evaluation.json:nested.actual"
    (tmp_path / "metrics.json").write_text(
        json.dumps({"status": "COMPLETED", "objective": 0, "part": 999}),
        encoding="utf-8",
    )
    if create_source:
        (tmp_path / "evaluation.json").write_text(
            json.dumps({"nested": {"actual": source_value}}), encoding="utf-8",
        )
    metrics, values, reason = evidence.load_local_report_evidence(tmp_path, report)
    assert reason == "local_evidence_loaded"
    assert values["part"] == source_value
    assert metrics["part"] == 999
    assert evidence.minimum_dataset_coverage(
        report, values=values, metrics=metrics,
    ) == (expected_count == 1, expected_count, 1)


def test_coverage_counts_only_finite_nonboolean_declared_values():
    report = _report(["zero", "negative", "boolean", "nan", "infinity", "missing"], minimum=2)
    assert evidence.minimum_dataset_coverage(
        report,
        values={"zero": 0, "negative": -3, "boolean": True,
                "nan": float("nan"), "infinity": float("inf")},
    ) == (True, 2, 2)


@pytest.mark.parametrize("minimum", [None, 0], ids=["absent-minimum", "zero-minimum"])
def test_undeclared_zero_coverage_keeps_legacy_no_gate_without_name_inference(minimum):
    report = {"primary_metric": "objective", "columns": [{"key": "wer_part"}]}
    if minimum is not None:
        report["min_datasets"] = minimum
    assert evidence.dataset_metric_keys(report) == []
    assert evidence.minimum_dataset_coverage(
        report, values={"wer_part": 1}, metrics={"wer_other": 2},
    ) == (True, 0, 0)


def test_positive_minimum_without_any_declaration_is_not_all_display_columns():
    report = {
        "primary_metric": "objective", "min_datasets": 2,
        "columns": [{"key": "objective"}, {"key": "elapsed"}, {"key": "wer_part"}],
    }
    assert evidence.dataset_metric_keys(report) == []
    with pytest.raises(ValueError, match=f"^{UNDECLARED}$"):
        evidence.minimum_dataset_coverage(
            report, values={"objective": 0, "elapsed": 1, "wer_part": 2},
        )


@pytest.mark.parametrize("minimum", [0, 1])
def test_explicit_empty_declaration_never_counts_undeclared_wer(minimum):
    report = _report([], minimum=minimum)
    assert evidence.dataset_metric_keys(report) == []
    assert evidence.minimum_dataset_coverage(
        report, metrics={"wer_a": 1, "wer_b": 2},
    ) == (minimum == 0, 0, minimum)


@pytest.mark.parametrize("minimum", [True, "invalid", 1.5])
def test_existing_invalid_minimum_result_is_preserved(minimum):
    report = _report(["part"], minimum=minimum)
    assert evidence.minimum_dataset_coverage(report, values={"part": 0}) == (False, 0, -1)
