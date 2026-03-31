"""Runtime guardrails — detect anomalies and contract violations during operation.

CALLING SPEC:
    from orze.engine.guardrails import (
        check_base_config_drift,    # warn if base_config changed between runs
        check_identical_results,    # flag N experiments with same metric value
        validate_avg_metric,        # verify primary metric = mean(per-metric values)
    )

    drift_msg = check_base_config_drift(results_dir, base_config_path)
        -> Optional[str]  (warning message or None)

    anomaly_msg = check_identical_results(recent_metrics, primary_metric, threshold=3)
        -> Optional[str]

    inconsistency_msg = validate_avg_metric(metrics, primary_metric, per_prefix)
        -> Optional[str]

All functions are pure. No side effects beyond return values.
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("orze")

_CONFIG_HASH_FILE = ".base_config_hash"


def check_base_config_drift(results_dir: Path, base_config_path: str) -> Optional[str]:
    """Warn if base_config changed since last run. Updates stored hash."""
    hash_file = results_dir / _CONFIG_HASH_FILE
    try:
        current_hash = hashlib.sha256(
            Path(base_config_path).read_bytes()).hexdigest()[:16]
    except OSError:
        return None

    old_hash = None
    if hash_file.exists():
        try:
            old_hash = hash_file.read_text().strip()
        except OSError:
            pass

    try:
        hash_file.write_text(current_hash)
    except OSError:
        pass

    if old_hash and old_hash != current_hash:
        return (
            f"base_config changed since last run ({old_hash[:8]}→{current_hash[:8]}). "
            f"Old results may not be comparable. Consider `orze reset --all`."
        )
    return None


def check_identical_results(
    recent_metrics: List[Dict],
    primary_metric: str,
    threshold: int = 3,
) -> Optional[str]:
    """Flag when threshold+ experiments produce the same primary metric."""
    if len(recent_metrics) < threshold:
        return None

    value_groups: Dict[str, list] = {}
    for item in recent_metrics:
        val = item.get("metrics", {}).get(primary_metric)
        if val is None:
            continue
        key = f"{val:.4f}"
        value_groups.setdefault(key, []).append(item.get("idea_id", "?"))

    for val, ids in value_groups.items():
        if len(ids) >= threshold:
            return (
                f"ANOMALY: {len(ids)} experiments have identical "
                f"{primary_metric}={val}: {ids[:5]}. "
                f"Config overrides may not be reaching the train script."
            )
    return None


def validate_avg_metric(
    metrics: dict,
    primary_metric: str,
    per_prefix: str = "",
    tolerance: float = 0.5,
) -> Optional[str]:
    """Validate that primary metric ≈ mean of per-sub-metric values.

    per_prefix: if set, look for keys starting with this prefix.
                If empty, auto-detect from primary_metric (e.g. "avg_wer" → "wer_").
    """
    if primary_metric not in metrics:
        return None

    # Auto-detect prefix: "avg_foo" → "foo_", "mean_bar" → "bar_"
    if not per_prefix:
        for strip in ("avg_", "mean_", "average_"):
            if primary_metric.startswith(strip):
                per_prefix = primary_metric[len(strip):] + "_"
                break
    if not per_prefix:
        return None

    per_vals = {k: v for k, v in metrics.items()
                if k.startswith(per_prefix) and isinstance(v, (int, float))}
    if len(per_vals) < 2:
        return None

    expected = sum(per_vals.values()) / len(per_vals)
    actual = metrics[primary_metric]

    if abs(expected - actual) > tolerance:
        return (
            f"{primary_metric}={actual:.2f} but mean of "
            f"{len(per_vals)} sub-metrics = {expected:.2f}"
        )
    return None
