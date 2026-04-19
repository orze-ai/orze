"""Tight-loop ``search`` role (F15).

Composes F11 (agg_search) → F10 (bundle_combiner) → F12
(posthoc_runner) into a single cycle: given the ArtifactCatalog, the
role enumerates checkpoints that still have un-searched
{aggregation × calibrator} combinations or un-combined TTA bundles,
enqueues the corresponding ideas (kind=agg_search or kind=bundle_combine)
in idea_lake, and lets the scheduler pick them up.

Default role config (opt-in, see orze.yaml.example)::

    search:
      mode: script
      script: -m orze.agents.search_role
      args: ["--results-dir", "{results_dir}"]
      cooldown: 30
      timeout: 600

Runs are idempotent — already-enqueued (idea_id) keys are skipped so a
re-run with no new artifacts produces no new ideas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("search_role")


def _idea_id(prefix: str, payload: str) -> str:
    h = hashlib.sha1(payload.encode()).hexdigest()[:10]
    return f"idea-{prefix}-{h}"


def enumerate_work(
    results_dir: Path,
    *,
    artifact_db: Optional[Path] = None,
    lake_db: Optional[Path] = None,
    max_new_ideas: int = 8,
    plateau_cycles: int = 3,
) -> Dict[str, Any]:
    """Return a dict summarizing work proposed this cycle.

    Schema::

        {
            "proposed": [{"idea_id": ..., "kind": ..., "title": ...}],
            "skipped_existing": int,
            "plateau_hit": bool,
        }
    """
    from orze.artifact_catalog import ArtifactCatalog
    from orze.idea_lake import IdeaLake

    results_dir = Path(results_dir)
    artifact_db = Path(artifact_db or results_dir / "idea_lake_artifacts.db")
    lake_db = Path(lake_db or results_dir / "idea_lake.db")
    cat = ArtifactCatalog(artifact_db)
    lake = IdeaLake(lake_db)

    existing = {r[0] for r in lake.conn.execute(
        "SELECT idea_id FROM ideas")}
    existing_agg = {r[0] for r in lake.conn.execute(
        "SELECT idea_id FROM ideas WHERE kind='agg_search'")}
    existing_bc = {r[0] for r in lake.conn.execute(
        "SELECT idea_id FROM ideas WHERE kind='bundle_combine'")}

    proposed: List[Dict[str, Any]] = []
    skipped = 0

    # Group preds/tta artifacts by ckpt_sha — each group is a candidate bundle.
    ckpts: Dict[str, Dict[str, Any]] = {}
    for row in cat.conn.execute(
        "SELECT path, ckpt_sha, kind FROM artifacts "
        "WHERE kind IN ('preds_npz', 'tta_preds') AND ckpt_sha IS NOT NULL"
    ):
        path, sha, kind = row
        ckpts.setdefault(sha, {"paths": [], "has_ckpt": False})
        ckpts[sha]["paths"].append(path)
    for row in cat.conn.execute(
        "SELECT ckpt_sha FROM artifacts WHERE kind='ckpt' AND ckpt_sha IS NOT NULL"
    ):
        sha = row[0]
        ckpts.setdefault(sha, {"paths": [], "has_ckpt": False})
        ckpts[sha]["has_ckpt"] = True
    cat.close()

    # For each group, propose agg_search (on a single base npz) and, if
    # there are ≥2 npzs of the same sha, propose bundle_combine.
    for sha, group in ckpts.items():
        if len(proposed) >= max_new_ideas:
            break
        if not group["paths"]:
            continue
        # agg_search: target the lexicographically first npz of the sha
        base = sorted(group["paths"])[0]
        aid = _idea_id("agg", f"{sha}|{base}")
        if aid not in existing:
            proposed.append({
                "idea_id": aid,
                "kind": "agg_search",
                "title": f"agg_search over {Path(base).name} (sha={sha[:8]})",
                "config": {"base_npz": base, "ckpt_sha": sha},
            })
        else:
            skipped += 1
        if len(proposed) >= max_new_ideas:
            break

        if len(group["paths"]) >= 2:
            bid = _idea_id("bc", f"{sha}|" + "|".join(sorted(group["paths"])))
            if bid not in existing:
                proposed.append({
                    "idea_id": bid,
                    "kind": "bundle_combine",
                    "title": (f"bundle_combine of {len(group['paths'])} views "
                              f"(sha={sha[:8]})"),
                    "config": {
                        "ckpt_sha": sha,
                        "bundle": sorted(group["paths"]),
                    },
                })
            else:
                skipped += 1

    # Enqueue proposed ideas as 'pending' so the launcher picks them up.
    for idea in proposed:
        cfg_yaml = json.dumps(idea["config"], indent=2)
        try:
            lake.insert(
                idea["idea_id"], idea["title"], cfg_yaml,
                raw_markdown=cfg_yaml,
                status="pending",
                kind=idea["kind"],
            )
        except Exception as e:  # pragma: no cover
            logger.warning("insert %s failed: %s", idea["idea_id"], e)

    # Plateau detection: if the best metric hasn't moved across the last
    # N search-role cycles, stop proposing to avoid thrashing.
    plateau_hit = _check_plateau(results_dir, lake, plateau_cycles)

    return {
        "proposed": proposed,
        "skipped_existing": skipped,
        "plateau_hit": plateau_hit,
    }


def _check_plateau(results_dir: Path, lake, plateau_cycles: int) -> bool:
    """Has the leaderboard best not moved in ``plateau_cycles`` search cycles?"""
    state_path = results_dir / "_search_role_state.json"
    try:
        current = None
        for r in lake.conn.execute(
            "SELECT eval_metrics FROM ideas "
            "WHERE status='completed' AND eval_metrics IS NOT NULL"
        ):
            try:
                m = json.loads(r[0])
                v = (m.get("pgmAP_ALL") or m.get("map")
                     or m.get("score_mean") or m.get("score"))
                if v is not None:
                    current = max(current or -1e9, float(v))
            except (ValueError, TypeError):
                continue
        prev = {"best": None, "stale": 0}
        if state_path.exists():
            prev = json.loads(state_path.read_text(encoding="utf-8"))
        if current is None:
            stale = prev.get("stale", 0)
        elif prev.get("best") is None or current > prev["best"] + 1e-9:
            stale = 0
        else:
            stale = prev.get("stale", 0) + 1
        state_path.write_text(json.dumps({
            "best": current,
            "stale": stale,
        }), encoding="utf-8")
        return stale >= plateau_cycles
    except Exception:  # pragma: no cover
        return False


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Tight-loop search role (F15)")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--artifact-db", default=None)
    ap.add_argument("--lake-db", default=None)
    ap.add_argument("--max-new-ideas", type=int, default=8)
    ap.add_argument("--plateau-cycles", type=int, default=3)
    args = ap.parse_args(argv)

    out = enumerate_work(
        Path(args.results_dir),
        artifact_db=Path(args.artifact_db) if args.artifact_db else None,
        lake_db=Path(args.lake_db) if args.lake_db else None,
        max_new_ideas=args.max_new_ideas,
        plateau_cycles=args.plateau_cycles,
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
