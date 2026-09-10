"""Deterministic ordering of already-qualified scalar observations.

CALLING SPEC:
    objective_sort_key(primary_value, values, report_cfg, idea_id="") -> tuple
        Use as an ascending sort key (never reverse=True). The explicit
        secondary objective follows the primary direction, then stable ID.
    objective_improves(primary_value, values, previous_value, previous_values,
                       report_cfg) -> bool
        Strict local comparison of two already-qualified current results.
        Stable ID and missing optional measurements cannot prove improvement.

This is local ordering, not a scientific equivalence test. Eligibility and
comparability remain the evidence/adapter contract's responsibility. Missing
optional measurements are not zero; display metadata is never an objective.
"""
from __future__ import annotations

import math
from typing import Mapping


def objective_sort_key(primary_value, values: Mapping, report_cfg: Mapping,
                       idea_id: str = "") -> tuple:
    direction = report_cfg.get("sort", "descending")
    if direction not in ("ascending", "descending"):
        raise ValueError("report.sort must be ascending or descending")
    sign = 1 if direction == "ascending" else -1

    def component(value):
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value))):
            return (0, sign * float(value))
        return (1, 0.0)

    secondary = report_cfg.get("secondary_metric")
    second = values.get(secondary) if isinstance(secondary, str) and secondary else None
    return component(primary_value), component(second), str(idea_id)


def objective_improves(primary_value, values: Mapping, previous_value,
                       previous_values: Mapping, report_cfg: Mapping) -> bool:
    """Whether the current declared objective is strictly better than another.

    False includes both no improvement and insufficient comparison evidence;
    it must not be interpreted as statistical equivalence. This does not turn
    result revisions into independent observations or a historical record.
    """
    direction = report_cfg.get("sort", "descending")
    if direction not in ("ascending", "descending"):
        raise ValueError("report.sort must be ascending or descending")

    def finite(value):
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value)))

    if not finite(primary_value) or not finite(previous_value):
        return False
    if primary_value != previous_value:
        return (primary_value < previous_value if direction == "ascending"
                else primary_value > previous_value)
    secondary = report_cfg.get("secondary_metric")
    if not isinstance(secondary, str) or not secondary:
        return False
    current, previous = values.get(secondary), previous_values.get(secondary)
    if not finite(current) or not finite(previous):
        return False
    return (current < previous if direction == "ascending"
            else current > previous)
