"""Shared medal-tier ranking. Use this everywhere a "best result" is selected
across heterogeneous competitions, so a wrong-direction raw metric (e.g. RMSE
treated as descending) cannot hijack the choice. Single source of truth.

CALLING SPEC:
    MEDAL_RANK: dict[str, int]   - higher = better tier
    medal_rank(record) -> int    - extracts medal from record, returns rank
"""
from __future__ import annotations
from typing import Any, Dict

MEDAL_RANK: Dict[str, int] = {
    "gold": 4, "silver": 3, "bronze": 2,
    "above_median": 1, "below_median": 0, "none": -1,
}


def medal_rank(record: Any) -> int:
    """Return medal rank for a record (dict). Looks in standard locations:
    record["medal"], record["values"]["medal"], record["eval_metrics"]["medal"].
    Returns -1 when no medal info is present."""
    if not isinstance(record, dict):
        return -1
    m = record.get("medal")
    if m is None:
        v = record.get("values") or record.get("eval_metrics") or {}
        if isinstance(v, dict):
            m = v.get("medal")
    return MEDAL_RANK.get(m, -1) if m else -1
