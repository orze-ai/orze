"""Post-hoc runner (F12) — inference-only idea launcher.

Handles ideas with ``kind != 'train'``: given (ckpt | bundle of npz, ops,
dataset_split) run an inference-only job and write

    <idea_dir>/metrics.json
    <idea_dir>/clip_preds_<idea_id>.npz

Also registers the outputs in the ArtifactCatalog so downstream search
roles can find them.

The core entry point ``run_posthoc(idea_id, cfg, idea_dir)`` delegates the
"actually run inference" step to a pluggable adapter (picked by project
name). This keeps orze generic while letting each consumer (nexar, smac,
…) plug in its own eval commands.

Adapters are registered by the @register_adapter decorator; the ``null``
adapter is always present for tests.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("posthoc_runner")


_ADAPTERS: Dict[str, Callable] = {}


def register_adapter(name: str):
    def deco(fn):
        _ADAPTERS[name] = fn
        return fn
    return deco


def get_adapter(name: str) -> Callable:
    if name not in _ADAPTERS:
        raise KeyError(f"no posthoc adapter {name!r}. "
                       f"Known: {sorted(_ADAPTERS)}")
    return _ADAPTERS[name]


def list_adapters() -> list:
    return sorted(_ADAPTERS)


# ---------------------------------------------------------------------- #
# Default null adapter — returns canned metrics so tests can run end-to-  #
# end without a real model.                                               #
# ---------------------------------------------------------------------- #


@register_adapter("null")
def _null_adapter(idea_id: str, cfg: Dict[str, Any],
                  idea_dir: Path) -> Dict[str, Any]:
    """Canned-response adapter used in tests.

    cfg may include a 'canned_metrics' dict which is returned directly.
    """
    canned = cfg.get("canned_metrics") or {"pgmAP_ALL": 0.0}
    return dict(canned)


# ---------------------------------------------------------------------- #
# Main entry                                                              #
# ---------------------------------------------------------------------- #


def run_posthoc(
    idea_id: str,
    cfg: Dict[str, Any],
    idea_dir: Path,
    *,
    artifact_catalog_db: Optional[Path] = None,
    adapter_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a post-hoc idea and write outputs.

    ``cfg`` schema::

        kind: posthoc_eval | tta_sweep | agg_search | bundle_combine
        adapter: <name of registered adapter>   # default 'null'
        ckpt: <path>                            # for posthoc_eval / tta_sweep
        bundle: [<npz>, <npz>, ...]             # for bundle_combine
        ops:
          aggregation: last | late_k2 | ...
          calibrator: identity | cv_mix | ...
          tta_views: [hflip, fs3, fs6, ...]     # for tta_sweep
        dataset_split: test | val

    Returns the metrics dict produced by the adapter.
    """
    idea_dir = Path(idea_dir)
    idea_dir.mkdir(parents=True, exist_ok=True)
    adapter_name = adapter_name or cfg.get("adapter") or "null"
    adapter = get_adapter(adapter_name)
    t0 = time.time()

    try:
        metrics = adapter(idea_id, cfg, idea_dir)
        metrics = dict(metrics or {})
        metrics.setdefault("_source", f"posthoc_runner:{adapter_name}")
        metrics.setdefault("elapsed_s", round(time.time() - t0, 3))
        status = "completed"
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("posthoc adapter %s crashed: %s", adapter_name, e)
        metrics = {"status": "FAILED", "error": str(e),
                   "_source": f"posthoc_runner:{adapter_name}"}
        status = "failed"

    metrics_path = idea_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Register outputs with the artifact catalog if one exists.
    if artifact_catalog_db is not None:
        try:
            _register_outputs(artifact_catalog_db, idea_id, cfg, idea_dir,
                              metrics, status)
        except Exception as e:  # pragma: no cover
            logger.debug("catalog register skipped: %s", e)

    return metrics


def _register_outputs(db: Path, idea_id: str, cfg: Dict[str, Any],
                      idea_dir: Path, metrics: Dict[str, Any],
                      status: str) -> None:
    from orze.artifact_catalog import ArtifactCatalog
    cat = ArtifactCatalog(db)
    ckpt_sha = cfg.get("ckpt_sha")
    if not ckpt_sha and cfg.get("ckpt"):
        try:
            from orze.artifact_catalog import hash_ckpt
            ckpt_sha = hash_ckpt(cfg["ckpt"])
        except OSError:
            ckpt_sha = None
    metric_val = (metrics.get("pgmAP_ALL")
                  or metrics.get("map")
                  or metrics.get("score_mean"))
    # Any npz output landing next to the idea counts as a prediction artifact.
    for npz in idea_dir.glob("*.npz"):
        kind = "tta_preds" if cfg.get("kind") == "tta_sweep" else "preds_npz"
        cat.upsert(
            npz, kind,
            ckpt_sha=ckpt_sha, idea_id=idea_id,
            inference_config=cfg.get("ops"),
            metric_val=metric_val if status == "completed" else None,
        )
    cat.close()


def subprocess_adapter(
    cmd: list,
    *,
    env: Optional[Dict[str, str]] = None,
    timeout: int = 3600,
    cwd: Optional[Path] = None,
) -> Dict[str, Any]:
    """Convenience wrapper for adapters that shell out to eval scripts.

    The subprocess must write its metrics.json to stdout (as JSON) OR to
    ``<cwd>/metrics.json``. Returns a dict either way.
    """
    full_env = os.environ.copy()
    if env:
        full_env.update({k: str(v) for k, v in env.items()})
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=timeout, env=full_env, cwd=str(cwd) if cwd else None,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"posthoc subprocess failed ({result.returncode}): "
            f"{result.stderr[-400:]}"
        )
    # Try stdout as JSON first, fall back to metrics.json in cwd.
    try:
        return json.loads(result.stdout)
    except (ValueError, TypeError):
        if cwd:
            mj = Path(cwd) / "metrics.json"
            if mj.exists():
                return json.loads(mj.read_text(encoding="utf-8"))
        raise RuntimeError(
            "posthoc subprocess did not emit valid JSON metrics")


# ---------------------------------------------------------------------- #
# Ensure built-in adapters are importable side-effect-free.              #
# ---------------------------------------------------------------------- #


def _load_builtin_adapters():
    try:
        from orze.engine.posthoc_adapters import nexar_collision  # noqa
    except Exception as e:  # pragma: no cover
        logger.debug("nexar_collision adapter not loaded: %s", e)


_load_builtin_adapters()
