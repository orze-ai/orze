"""Round-2 D2: polymorphic ``orze ingest-champion`` implementation.

Previously this module returned ``pgmAP_ALL`` and ``ckpt_sha`` — values
hardcoded for the project that originally motivated the feature.
Round-2 makes it project-agnostic by reading ``report.primary_metric``
and ``report.columns`` from ``orze.yaml`` and pulling the matching
values from ``results/<idea_id>/metrics.json`` (and any source-routed
columns the report config declares).

The legacy pgmAP-specific path is preserved behind a ``--legacy-pgmap``
flag for back-compat with existing pipelines.

Note: the previous on-disk file at this path was a dangling symlink
to a host-local "orze_overflow" directory; this file is the canonical
replacement and writes directly into idea_lake.db using the same
schema as ``orze ideas inject`` (round-2 D1).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orze.ingest_champion")


def _load_metrics(results_dir: Path, idea_id: str) -> dict:
    """Read metrics.json from the idea dir; return {} on any error."""
    p = results_dir / idea_id / "metrics.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _load_source_routed_value(results_dir: Path, idea_id: str,
                              source_path: str):
    """Read a single value from a source-routed column. ``source_path``
    is interpreted relative to the idea dir."""
    p = (results_dir / idea_id / source_path).resolve()
    if not p.exists():
        return None
    try:
        if p.suffix == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            return data
        return p.read_text(encoding="utf-8").strip()
    except (OSError, json.JSONDecodeError):
        return None


def _gather_eval_metrics_from_report(cfg: dict, results_dir: Path,
                                     idea_id: str) -> dict:
    report = cfg.get("report") or {}
    metrics = _load_metrics(results_dir, idea_id)
    primary = report.get("primary_metric")
    out: dict = {}
    if primary and primary in metrics:
        out[primary] = metrics[primary]
    for col in (report.get("columns") or []):
        if not isinstance(col, dict):
            continue
        key = col.get("key")
        if not key:
            continue
        if key in metrics:
            out[key] = metrics[key]
            continue
        src = col.get("source")
        if src:
            v = _load_source_routed_value(results_dir, idea_id, src)
            if v is not None:
                # When source returns a dict, prefer the same key
                # within it; otherwise embed the raw value.
                if isinstance(v, dict) and key in v:
                    out[key] = v[key]
                else:
                    out[key] = v
    # Also include any *_report.json aggregates lurking in the dir.
    idea_dir = results_dir / idea_id
    if idea_dir.is_dir():
        for rp in sorted(idea_dir.glob("*_report.json")):
            try:
                rd = json.loads(rp.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            rm = rd.get("metrics") if isinstance(rd, dict) else None
            if not isinstance(rm, dict):
                continue
            prefix = rp.stem.replace("_report", "")
            for k, v in rm.items():
                out.setdefault(f"{prefix}_{k}", v)
    return out


def _legacy_pgmap_metrics(results_dir: Path, idea_id: str) -> dict:
    """Original pre-round-2 behavior: return pgmAP_ALL + ckpt_sha."""
    metrics = _load_metrics(results_dir, idea_id)
    out: dict = {}
    for k in ("pgmAP_ALL", "ckpt_sha", "epoch", "training_time"):
        if k in metrics:
            out[k] = metrics[k]
    return out


def ingest(results_dir: Path, idea_id: str = "idea-champion-0905",
           config_path: Optional[Path] = None,
           project_root: Optional[Path] = None,
           legacy_pgmap: bool = False,
           cfg: Optional[dict] = None) -> dict:
    """Insert a manual champion row into idea_lake.db.

    ``cfg`` is the project's loaded orze.yaml dict. When omitted, we
    load it from ``project_root / 'orze.yaml'`` (or the cwd).
    """
    from orze.idea_lake import IdeaLake
    from orze.core.config import load_project_config

    project_root = project_root or results_dir.parent
    if cfg is None:
        cfg_yaml = project_root / "orze.yaml"
        cfg = load_project_config(str(cfg_yaml) if cfg_yaml.exists() else None)

    db_path = cfg.get("idea_lake_db") or str(project_root / ".orze" / "idea_lake.db")
    lake = IdeaLake(str(db_path))

    if legacy_pgmap:
        eval_metrics = _legacy_pgmap_metrics(results_dir, idea_id)
        approach_family = "legacy"
    else:
        eval_metrics = _gather_eval_metrics_from_report(
            cfg, results_dir, idea_id)
        approach_family = "champion"

    # Optional config_path: legacy pgmAP pipelines wrote a
    # _champion_config.json next to the idea dir. The round-2 spec
    # (F2) prefers idea_config.yaml + full_scale_metrics.json, but we
    # accept either for back-compat.
    config_yaml = ""
    candidates = []
    if config_path:
        candidates.append(Path(config_path))
    candidates.extend([
        results_dir / idea_id / "idea_config.yaml",
        results_dir / idea_id / "_champion_config.json",
        results_dir / "_champion_config.json",
    ])
    for cand in candidates:
        if cand and cand.exists():
            try:
                if cand.suffix == ".json":
                    config_yaml = cand.read_text(encoding="utf-8")
                else:
                    config_yaml = cand.read_text(encoding="utf-8")
                break
            except OSError:
                continue

    title = (eval_metrics.get("idea_title")
             or f"manual champion {idea_id}")
    lake.insert(
        idea_id=idea_id,
        title=title,
        config_yaml=config_yaml,
        raw_markdown=f"## {idea_id}: {title}\n",
        eval_metrics=eval_metrics or None,
        status="completed",
        priority="high",
        category="champion",
        approach_family=approach_family,
    )
    info = {
        "idea_id": idea_id,
        "db": str(db_path),
        "metric_keys": sorted(eval_metrics.keys()),
        "legacy_pgmap": bool(legacy_pgmap),
    }
    logger.info("ingest-champion: wrote %s with %d metric keys",
                idea_id, len(eval_metrics))
    return info
