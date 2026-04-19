"""Retroactive ingest of a manual champion (F16).

The Nexar Collision project hit pgmAP 0.9057 via a 4-view TTA bundle of
the vjepa2_alertonly_v4 checkpoint before orze natively supported post-
hoc search. This script (and the ``orze ingest-champion`` CLI) records
that champion as a first-class idea of kind='bundle_combine' in the
project's idea_lake so it shows up in the leaderboard and the artifact
catalog like any other orze-produced result.

Usage::

    python -m orze.agents.ingest_champion \\
        --results-dir /mnt/fsx_az_f/vla/nexar_collision/results \\
        --idea-id    idea-champion-0905 \\
        --config     results/_champion_config.json

``--config`` defaults to ``<results_dir>/_champion_config.json``, whose
``primary_champion`` block is read verbatim.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("ingest_champion")


def _safe_resolve(path: str, project_root: Path) -> str:
    """Turn relative champion paths ('results/foo.npz') into absolute ones."""
    p = Path(path)
    if not p.is_absolute():
        p = project_root / p
    return str(p)


def ingest(
    results_dir: Path,
    *,
    idea_id: str = "idea-champion-0905",
    champion_cfg: Optional[Dict[str, Any]] = None,
    config_path: Optional[Path] = None,
    state_path: Optional[Path] = None,
    project_root: Optional[Path] = None,
    artifact_db: Optional[Path] = None,
    lake_db: Optional[Path] = None,
) -> Dict[str, Any]:
    """Insert the champion idea + update orze state best_idea_id.

    Returns a dict summarising what was done (idempotent).
    """
    from orze.artifact_catalog import ArtifactCatalog, hash_ckpt
    from orze.idea_lake import IdeaLake

    results_dir = Path(results_dir)
    project_root = Path(project_root or results_dir.parent)
    config_path = Path(config_path or results_dir / "_champion_config.json")
    state_path = Path(state_path or results_dir / "_orze_state.json")
    lake_db = Path(lake_db or results_dir / "idea_lake.db")
    artifact_db = Path(artifact_db or results_dir / "idea_lake_artifacts.db")

    if champion_cfg is None:
        if not config_path.exists():
            raise FileNotFoundError(f"no champion config at {config_path}")
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        champion_cfg = raw.get("primary_champion") or raw

    ckpt_rel = champion_cfg.get("checkpoint")
    npzs = [_safe_resolve(p, project_root)
            for p in champion_cfg.get("predictions_files", [])]
    ckpt_abs = _safe_resolve(ckpt_rel, project_root) if ckpt_rel else None

    # Compute ckpt_sha if the file exists locally; fall back to the
    # symbolic "ckpt:<path>" token so the idea is still insertable in
    # environments where the weights aren't present.
    ckpt_sha = None
    if ckpt_abs and Path(ckpt_abs).exists():
        try:
            ckpt_sha = hash_ckpt(ckpt_abs)
        except OSError:
            ckpt_sha = None
    if not ckpt_sha and ckpt_rel:
        ckpt_sha = f"path:{ckpt_rel}"

    # --- idea config stored in the lake ---
    idea_cfg = {
        "kind": "bundle_combine",
        "ckpt_sha": ckpt_sha,
        "ckpt_path": ckpt_abs,
        "bundle": npzs,
        "winning_recipe": {
            "aggregation": "cv_mix(last1_max, top6_mean)",
            "per_clip_combiner": "mean across 4 TTA views "
                                  "(intersection on (vid, frame_center))",
            "calibrator": "per_group rank-normalized, 5-fold stratified OOF",
        },
        "reproducer": champion_cfg.get("reproducer"),
        "source": "manual (pre-orze-v3.6.0)",
        "notes": champion_cfg.get("notes", ""),
    }
    eval_metrics = {
        "pgmAP_ALL": champion_cfg.get("pgmAP_ALL"),
        "pgmAP_Public": champion_cfg.get("pgmAP_Public"),
        "pgmAP_Private": champion_cfg.get("pgmAP_Private"),
        "group": champion_cfg.get("per_group") or {},
    }

    # --- upsert idea (kind='bundle_combine', status='completed') ---
    lake_db.parent.mkdir(parents=True, exist_ok=True)
    lake = IdeaLake(lake_db)
    lake.insert(
        idea_id,
        title="4-view TTA bundle → pgmAP 0.9057 (manual baseline champion)",
        config_yaml=json.dumps(idea_cfg, indent=2),
        raw_markdown=json.dumps(idea_cfg, indent=2),
        eval_metrics=eval_metrics,
        status="completed",
        kind="bundle_combine",
    )

    # --- register artifacts in the catalog ---
    artifact_db.parent.mkdir(parents=True, exist_ok=True)
    cat = ArtifactCatalog(artifact_db)
    if ckpt_abs and Path(ckpt_abs).exists():
        cat.upsert(Path(ckpt_abs), "ckpt", ckpt_sha=ckpt_sha,
                   idea_id=idea_id,
                   metric_test=eval_metrics["pgmAP_ALL"])
    for npz in npzs:
        p = Path(npz)
        if p.exists():
            kind = "tta_preds" if "tta" in p.name else "preds_npz"
            cat.upsert(p, kind, ckpt_sha=ckpt_sha, idea_id=idea_id,
                       metric_test=eval_metrics["pgmAP_ALL"])
    cat.close()

    # --- update orze state ---
    state = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            state = {}
    previous_best = state.get("best_idea_id")
    state["best_idea_id"] = idea_id
    state["previous_best_idea_id"] = previous_best
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    return {
        "idea_id": idea_id,
        "ckpt_sha": ckpt_sha,
        "npz_registered": sum(1 for p in npzs if Path(p).exists()),
        "previous_best": previous_best,
        "pgmAP_ALL": eval_metrics["pgmAP_ALL"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(
        "Retroactive champion ingest (F16). Inserts the 0.9057 bundle into "
        "idea_lake as a kind='bundle_combine' idea and promotes it in "
        "_orze_state.json."
    ))
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--idea-id", default="idea-champion-0905")
    ap.add_argument("--config", default=None,
                    help="Path to _champion_config.json")
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args(argv)

    info = ingest(
        Path(args.results_dir),
        idea_id=args.idea_id,
        config_path=Path(args.config) if args.config else None,
        project_root=Path(args.project_root) if args.project_root else None,
    )
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
