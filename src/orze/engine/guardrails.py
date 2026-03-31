"""Runtime guardrails — detect anomalies and contract violations during operation.

CALLING SPEC:
    from orze.engine.guardrails import (
        check_base_config_drift,    # #5: model change detection
        check_identical_results,    # #10: anomaly detection
        warn_unknown_config_keys,   # #11: silent key drops
        validate_avg_metric,        # #12: metric consistency
    )

All functions are pure (no side effects beyond logging). Call them
from the orchestrator phases as needed.
"""

import hashlib
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger("orze")

# ---------------------------------------------------------------------------
# #5: Model/base_config change detection
# ---------------------------------------------------------------------------

_CONFIG_HASH_FILE = ".base_config_hash"


def check_base_config_drift(results_dir: Path, base_config_path: str) -> Optional[str]:
    """Compare current base_config hash against stored hash from last run.

    Returns None if no drift, or a warning message if config changed.
    Updates the stored hash after checking.
    """
    hash_file = results_dir / _CONFIG_HASH_FILE

    try:
        config_text = Path(base_config_path).read_text()
        current_hash = hashlib.sha256(config_text.encode()).hexdigest()[:16]
    except OSError:
        return None

    old_hash = None
    if hash_file.exists():
        try:
            old_hash = hash_file.read_text().strip()
        except OSError:
            pass

    # Update stored hash
    try:
        hash_file.write_text(current_hash)
    except OSError:
        pass

    if old_hash and old_hash != current_hash:
        return (
            f"base_config has changed since last run (hash {old_hash[:8]}→{current_hash[:8]}). "
            f"Old completed results may not be comparable. Consider `orze reset --all` "
            f"to clear stale ideas from the previous config."
        )
    return None


# ---------------------------------------------------------------------------
# #10: Anomaly detection — identical results from different configs
# ---------------------------------------------------------------------------

def check_identical_results(
    recent_metrics: List[Dict],
    primary_metric: str = "avg_wer",
    threshold: int = 3,
) -> Optional[str]:
    """Detect if multiple recent experiments produced identical primary metric values.

    Args:
        recent_metrics: list of {idea_id, metrics_dict} for recently completed ideas
        primary_metric: the metric key to compare
        threshold: how many identical results before warning

    Returns warning message or None.
    """
    if len(recent_metrics) < threshold:
        return None

    # Group by primary metric value (rounded to avoid float noise)
    value_groups: Dict[str, list] = {}
    for item in recent_metrics:
        val = item.get("metrics", {}).get(primary_metric)
        if val is None:
            continue
        key = f"{val:.4f}"
        value_groups.setdefault(key, []).append(item.get("idea_id", "?"))

    for val, idea_ids in value_groups.items():
        if len(idea_ids) >= threshold:
            return (
                f"ANOMALY: {len(idea_ids)} experiments produced identical "
                f"{primary_metric}={val}: {idea_ids[:5]}. "
                f"This likely means idea configs are not reaching the train script. "
                f"Check idea_config.yaml generation and config merge logic."
            )
    return None


# ---------------------------------------------------------------------------
# #11: Unknown config key detection
# ---------------------------------------------------------------------------

_KNOWN_BASE_KEYS: Set[str] = set()  # populated from base_config on first call


def warn_unknown_config_keys(
    idea_config: dict,
    base_config: dict,
    idea_id: str = "",
) -> List[str]:
    """Warn about idea config keys that don't exist in base_config.

    These keys might be silently ignored by the train script.
    Returns list of warning strings.
    """
    if not idea_config or not base_config:
        return []

    # Build set of known keys (base_config keys + common override keys)
    known = set(base_config.keys()) | {
        "lora_path", "lora_scale", "strategy", "dataset",
        "finetune", "learning_rate", "num_train_steps",
        "lora_rank", "lora_alpha", "train_data",
        "text_normalizer",
    }

    warnings = []
    for key in idea_config:
        if key.startswith("_"):  # internal keys
            continue
        if key not in known:
            warnings.append(
                f"Idea {idea_id}: config key '{key}' not in base_config "
                f"and not a known override — may be silently ignored"
            )
    return warnings


# ---------------------------------------------------------------------------
# #12: Metric consistency validation
# ---------------------------------------------------------------------------

def validate_avg_metric(
    metrics: dict,
    primary_metric: str = "avg_wer",
    per_dataset_prefix: str = "wer_",
    tolerance: float = 0.5,
) -> Optional[str]:
    """Validate that primary metric equals the mean of per-dataset metrics.

    Returns warning message or None.
    """
    if primary_metric not in metrics:
        return None

    per_ds = {k: v for k, v in metrics.items()
              if k.startswith(per_dataset_prefix) and isinstance(v, (int, float))}

    if len(per_ds) < 2:
        return None

    expected = sum(per_ds.values()) / len(per_ds)
    actual = metrics[primary_metric]

    if abs(expected - actual) > tolerance:
        return (
            f"Metric inconsistency: {primary_metric}={actual:.2f} but "
            f"mean of {len(per_ds)} per-dataset values = {expected:.2f} "
            f"(diff={abs(expected-actual):.2f})"
        )
    return None
