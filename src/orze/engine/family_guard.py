"""Approach family taxonomy, distribution tracking, and repetition guard.

CALLING SPEC:
    APPROACH_FAMILIES: frozenset of valid approach family strings

    infer_approach_family(config: dict, category: str) -> str
        Heuristic: map config keys and category to an approach family.
        Returns one of APPROACH_FAMILIES.

    get_family_distribution(lake, status: str = "completed") -> dict[str, int]
        Query idea_lake for approach_family counts among ideas with given status.
        lake: IdeaLake instance. Returns {family: count}.

    get_recent_winning_families(results_dir: Path, primary_metric: str,
                                 n: int = 20, sort: str = "descending") -> list[str]
        Return ordered list of approach_families from the last N winners
        (by mtime, best metric first).

    check_family_concentration(recent_families: list[str],
                                max_consecutive: int = 5) -> str | None
        Return warning message if the same family appears max_consecutive
        times at the head of recent_families. None if OK.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("orze")

APPROACH_FAMILIES = frozenset({
    "architecture", "training_config", "data", "infrastructure",
    "optimization", "regularization", "ensemble", "other",
})

# Config key patterns → family heuristic
_FAMILY_HINTS = {
    "architecture": {"model", "backbone", "encoder", "decoder", "layers",
                     "hidden_size", "num_layers", "attention", "heads",
                     "architecture", "network", "block"},
    "training_config": {"lr", "learning_rate", "optimizer", "scheduler",
                        "epochs", "batch_size", "warmup", "weight_decay",
                        "momentum", "gradient"},
    "data": {"augment", "data", "dataset", "transform", "crop", "resize",
             "normalize", "split", "sampling", "mixup", "cutmix"},
    "regularization": {"dropout", "regularization", "label_smoothing",
                       "noise", "clip", "gradient_clip"},
    "infrastructure": {"mixed_precision", "fp16", "bf16", "distributed",
                       "compile", "checkpoint", "accumulation"},
    "optimization": {"loss", "criterion", "metric", "objective",
                     "temperature", "alpha", "beta", "gamma"},
    "ensemble": {"ensemble", "voting", "stacking", "blending",
                 "num_models", "bagging"},
}


def infer_approach_family(config: dict, category: str = "") -> str:
    """Infer approach family from config keys and category string."""
    # Flatten config keys to check
    flat_keys = set()
    for k, v in config.items():
        flat_keys.add(k.lower())
        if isinstance(v, dict):
            for k2 in v:
                flat_keys.add(k2.lower())

    # Score each family by matching keys
    scores: Dict[str, int] = {}
    for family, hints in _FAMILY_HINTS.items():
        score = len(flat_keys & hints)
        if score > 0:
            scores[family] = score

    # Also check category string
    cat_lower = category.lower() if category else ""
    if cat_lower:
        for family in APPROACH_FAMILIES:
            if family in cat_lower or cat_lower in family:
                scores[family] = scores.get(family, 0) + 2

    if scores:
        return max(scores, key=scores.get)
    return "other"


def get_family_distribution(lake, status: str = "completed") -> Dict[str, int]:
    """Query idea_lake for approach_family counts."""
    dist: Dict[str, int] = {}
    try:
        rows = lake.conn.execute(
            "SELECT approach_family, COUNT(*) as cnt FROM ideas "
            "WHERE status = ? GROUP BY approach_family",
            (status,),
        ).fetchall()
        for r in rows:
            family = r[0] or "other"
            dist[family] = r[1]
    except Exception as e:
        logger.warning("Could not query family distribution: %s", e)
    return dist


def get_recent_winning_families(results_dir: Path, primary_metric: str,
                                n: int = 20,
                                sort: str = "descending") -> List[str]:
    """Get approach families from the N most recent completed ideas, ordered by metric.

    Reads metrics.json and failure_analysis.json (for family) from result dirs.
    Falls back to 'other' if family not found.
    """
    entries = []
    if not results_dir.exists():
        return []

    try:
        with os.scandir(results_dir) as it:
            for entry in it:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                if not entry.name.startswith("idea-"):
                    continue
                mf = Path(entry.path) / "metrics.json"
                if not mf.exists():
                    continue
                try:
                    m = json.loads(mf.read_text(encoding="utf-8"))
                    if m.get("status") != "COMPLETED":
                        continue
                    val = m.get(primary_metric)
                    if val is None or not isinstance(val, (int, float)):
                        continue
                    # Try to get approach_family from claim or resolved config
                    family = "other"
                    claim_path = Path(entry.path) / "claim.json"
                    if claim_path.exists():
                        try:
                            claim = json.loads(claim_path.read_text(encoding="utf-8"))
                            family = claim.get("approach_family", "other")
                        except Exception:
                            pass
                    entries.append((mf.stat().st_mtime, float(val), family))
                except Exception:
                    continue
    except OSError:
        return []

    # Sort by mtime descending (most recent first), take N
    entries.sort(key=lambda x: -x[0])
    recent = entries[:n]

    # Sort by metric to get "winners" ordering
    reverse = sort != "ascending"
    recent.sort(key=lambda x: x[1], reverse=reverse)

    return [family for _, _, family in recent]


def check_family_concentration(recent_families: List[str],
                               max_consecutive: int = 5) -> Optional[str]:
    """Check if the same approach family dominates recent winners.

    Returns warning message if the same family appears at the start
    of recent_families max_consecutive times. None if OK.
    """
    if not recent_families or max_consecutive <= 0:
        return None

    if len(recent_families) < max_consecutive:
        return None

    head = recent_families[:max_consecutive]
    if len(set(head)) == 1:
        family = head[0]
        return (f"Family '{family}' has won the last {max_consecutive} "
                f"experiments. Consider exploring other families: "
                f"{', '.join(sorted(APPROACH_FAMILIES - {family}))}")

    return None
