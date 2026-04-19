"""Example post-hoc adapter — minimal classifier honest-eval driver.

Demonstrates the adapter contract in ~60 LOC. Loads a single ``.npz``
file containing three keys (``logits``, ``labels``, ``split``), tunes
nothing (no hyperparameters in the toy version), and reports
public-split accuracy as the metric.

Real adapters do per-frame averaging, calibration, ensemble weight
search, etc. — see ``nexar_collision.py`` for a worked example.

Config keys consumed:
    posthoc.example_classifier.npz_path   # path to predictions NPZ
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from orze.engine.posthoc_runner import register_adapter

logger = logging.getLogger("posthoc_adapters.example_classifier")


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    """Load ``logits``, ``labels``, ``split`` from a predictions NPZ."""
    with np.load(path, allow_pickle=False) as z:
        return {
            "logits": np.asarray(z["logits"]),
            "labels": np.asarray(z["labels"]),
            "split": np.asarray(z["split"]),
        }


def tune_posthoc(data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Pick post-hoc hyperparameters on the public split only.

    The toy adapter has no tunable knobs; real adapters search a grid.
    """
    return {}  # no hyperparameters to choose


def honest_report(data: Dict[str, np.ndarray],
                  params: Dict[str, Any]) -> Dict[str, float]:
    """Report accuracy on public, private, and all splits."""
    preds = data["logits"].argmax(axis=-1)
    labels = data["labels"]
    split = data["split"]
    out: Dict[str, float] = {}
    for name in ("public", "private", "all"):
        mask = (split == name) if name != "all" else np.ones_like(split, dtype=bool)
        if not mask.any():
            continue
        out[f"acc_{name}"] = float((preds[mask] == labels[mask]).mean())
    return out


@register_adapter("example_classifier")
def run(idea_id: str, cfg: Dict[str, Any], idea_dir: Path) -> Dict[str, Any]:
    """Honest-eval round trip — load NPZ, tune, report."""
    section = cfg.get("posthoc", {}).get("example_classifier", {})
    npz_path = Path(section.get("npz_path") or (idea_dir / "predictions.npz"))
    if not npz_path.exists():
        raise FileNotFoundError(f"example_classifier: missing {npz_path}")
    data = load_npz(npz_path)
    params = tune_posthoc(data)
    metrics = honest_report(data, params)
    metrics["_adapter"] = "example_classifier"
    return metrics
