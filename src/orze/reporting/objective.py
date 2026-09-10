"""Deterministic ordering of already-qualified scalar observations.

CALLING SPEC:
    objective_sort_key(primary_value, values, report_cfg, idea_id="") -> tuple
        Use as an ascending sort key (never reverse=True). The explicit
        secondary objective follows the primary direction, then stable ID.

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
