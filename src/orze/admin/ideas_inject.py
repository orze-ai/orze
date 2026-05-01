"""Round-2 D1: ``orze ideas inject`` implementation.

Replaces the ad-hoc ``scripts/inject_idea.py`` workaround projects had
to write to manually plant a row into ``idea_lake.db``. Uses
``IdeaLake.insert`` directly so the schema stays canonical.

Exposed as a function so unit tests can call it without invoking the
argparse layer.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orze")


def inject_idea(cfg: dict, *, idea_id: str, title: str,
                priority: str = "medium",
                category: str = "architecture",
                parent: Optional[str] = None,
                hypothesis: Optional[str] = None,
                config_yaml_path: Optional[str] = None,
                status: str = "queued",
                metrics_json: Optional[str] = None,
                approach_family: str = "other",
                force: bool = False) -> int:
    from orze.idea_lake import IdeaLake

    db_path = cfg.get("idea_lake_db") or "idea_lake.db"
    lake = IdeaLake(str(db_path))

    if lake.has(idea_id) and not force:
        print(f"orze ideas inject: idea {idea_id!r} already present in "
              f"{db_path}. Use --force to overwrite.")
        return 1

    config_yaml = ""
    if config_yaml_path:
        cp = Path(config_yaml_path)
        if not cp.exists():
            print(f"orze ideas inject: config file {config_yaml_path!r} "
                  f"does not exist")
            return 2
        config_yaml = cp.read_text(encoding="utf-8")

    eval_metrics = None
    if metrics_json:
        mp = Path(metrics_json)
        if not mp.exists():
            # For status==completed, missing metrics is a hard error;
            # for queued / archived, treat as a warning and inject blank.
            if status == "completed":
                print(f"orze ideas inject: --status completed requires "
                      f"a readable --metrics-json (got {metrics_json!r})")
                return 2
            logger.warning("metrics_json %s not found — inserting without "
                           "eval_metrics", metrics_json)
        else:
            try:
                eval_metrics = json.loads(mp.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                print(f"orze ideas inject: failed to parse "
                      f"{metrics_json!r}: {e}")
                return 2
            # Optional schema check: warn (not fail) if the project's
            # primary_metric is missing from the supplied metrics.
            primary = ((cfg.get("report") or {}).get("primary_metric"))
            if (primary and isinstance(eval_metrics, dict)
                    and primary not in eval_metrics):
                logger.warning(
                    "metrics_json %s does not contain report.primary_metric "
                    "%r — leaderboard may show this idea as INCOMPLETE",
                    metrics_json, primary)

    raw_markdown = (
        f"## {idea_id}: {title}\n\n"
        f"**Priority**: {priority}\n"
        f"**Category**: {category}\n"
    )
    if parent:
        raw_markdown += f"**Parent**: {parent}\n"
    if hypothesis:
        raw_markdown += f"**Hypothesis**: {hypothesis}\n"

    lake.insert(
        idea_id=idea_id,
        title=title,
        config_yaml=config_yaml,
        raw_markdown=raw_markdown,
        eval_metrics=eval_metrics,
        status=status,
        priority=priority,
        category=category,
        parent=parent,
        hypothesis=hypothesis,
        approach_family=approach_family,
    )
    print(f"orze ideas inject: wrote {idea_id} (status={status}) to {db_path}")
    return 0
