"""Reject malformed objective declarations before running experiments."""

import pytest

from orze.core.config import _validate_config


@pytest.mark.parametrize("direction", ["ascendng", None, "", True, False])
def test_config_rejects_invalid_objective_direction(direction):
    errors, _ = _validate_config({"report": {"sort": direction}})

    assert any(error.startswith("report.sort:") for error in errors)


@pytest.mark.parametrize("metric", [True, 9, [], {}])
def test_config_rejects_nonstring_secondary_objective(metric):
    errors, _ = _validate_config({"report": {"secondary_metric": metric}})

    assert any(error.startswith("report.secondary_metric:") for error in errors)


@pytest.mark.parametrize("metric", [None, "", "   ", True, 9, [], {}])
def test_config_rejects_invalid_explicit_primary_objective(metric):
    errors, _ = _validate_config({"report": {"primary_metric": metric}})

    assert any(error.startswith("report.primary_metric:") for error in errors)


@pytest.mark.parametrize(
    "report",
    [
        {},
        {"sort": "ascending"},
        {"sort": "descending"},
        {"primary_metric": "score"},
        {"primary_metric": "evaluation.score", "secondary_metric": "latency"},
        {"secondary_metric": None},
        {"secondary_metric": ""},
    ],
)
def test_config_accepts_defaults_and_explicit_scalar_objectives(report):
    errors, _ = _validate_config({"report": report})

    assert not any(
        error.startswith((
            "report.sort:", "report.primary_metric:", "report.secondary_metric:"
        ))
        for error in errors
    )
