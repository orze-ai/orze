"""Nexar-collision post-hoc adapter.

Wraps the existing nexar_collision eval_* scripts (eval_tta.py,
eval_champion_*.py, …). When orze dispatches a posthoc_eval / tta_sweep /
bundle_combine idea, this adapter knows how to translate the high-level
ops dict into the right CLI invocation.

The adapter is thin on purpose — it does not modify the training scripts
in the consumer project; it only invokes them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from orze.engine.posthoc_runner import register_adapter, subprocess_adapter

logger = logging.getLogger("posthoc_adapters.nexar_collision")


@register_adapter("nexar_collision")
def run(idea_id: str, cfg: Dict[str, Any], idea_dir: Path) -> Dict[str, Any]:
    """Dispatch based on cfg['kind'] and cfg['ops'].

    Expected config (YAML-loaded):

        project_root: /mnt/fsx_az_f/vla/nexar_collision
        python: /mnt/fsx_az_f/vla/.venv/bin/python
        kind: tta_sweep | bundle_combine | posthoc_eval | agg_search
        ckpt: results/vjepa2_alertonly_v4/best_model.pt
        bundle: [results/clip_preds_agg_sweep_dense_end.npz, ...]
        ops:
          aggregation: last | late_k2 | cv_mix(last,top6_mean) | ...
          calibrator: identity | cv_mix | isotonic | ...
          tta_views: [hflip, fs3, fs6]
          frame_stride: 4
          sampling: dense_end
        dataset_split: test
        dry_run: false   # for orchestration tests
    """
    kind = cfg.get("kind", "posthoc_eval")
    project_root = Path(cfg["project_root"])
    python = cfg.get("python", "python3")
    dry_run = bool(cfg.get("dry_run"))

    # Map kind → nexar eval script. The user has these scripts in
    # nexar_collision/scripts/; we call them as subprocesses and parse
    # their metrics.json output.
    if kind == "tta_sweep":
        script = cfg.get("script",
                         project_root / "scripts" / "eval_tta.py")
    elif kind == "bundle_combine":
        script = cfg.get("script",
                         project_root / "scripts" / "eval_champion_0905_final.py")
    elif kind == "agg_search":
        script = cfg.get("script",
                         project_root / "eval_agg_sweep.py")
    else:
        script = cfg.get("script",
                         project_root / "eval_e2e.py")

    if dry_run:
        # Return canned metrics directly (used by E2E plumbing tests).
        return cfg.get("canned_metrics", {"pgmAP_ALL": 0.0, "dry_run": True})

    cmd = [str(python), str(script)]
    # Pass ckpt / bundle / ops via CLI flags the eval scripts already accept.
    if cfg.get("ckpt"):
        cmd.extend(["--ckpt", str(cfg["ckpt"])])
    for npz in cfg.get("bundle", []) or []:
        cmd.extend(["--npz", str(npz)])
    ops = cfg.get("ops") or {}
    if ops.get("aggregation"):
        cmd.extend(["--aggregation", str(ops["aggregation"])])
    if ops.get("calibrator"):
        cmd.extend(["--calibrator", str(ops["calibrator"])])
    for view in ops.get("tta_views", []) or []:
        cmd.extend(["--tta", str(view)])
    if cfg.get("dataset_split"):
        cmd.extend(["--split", str(cfg["dataset_split"])])
    cmd.extend(["--out-dir", str(idea_dir)])

    timeout = int(cfg.get("timeout", 3600))
    try:
        metrics = subprocess_adapter(
            cmd, timeout=timeout, cwd=project_root,
            env={"CUDA_VISIBLE_DEVICES": str(cfg.get("gpu", 0))},
        )
    except Exception as e:
        logger.warning("nexar_collision adapter subprocess failed: %s", e)
        # Fall back to reading metrics.json if the script already wrote one.
        mj = idea_dir / "metrics.json"
        if mj.exists():
            metrics = json.loads(mj.read_text(encoding="utf-8"))
        else:
            raise
    metrics.setdefault("_adapter", "nexar_collision")
    metrics.setdefault("_kind", kind)
    return metrics
