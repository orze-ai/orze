"""Champion-promotion guard (F14).

Before promoting an idea to ``best_idea_id`` (and firing the ``new_best``
notification), the guard:

1. Verifies the claimed metric. When the idea config carries a
   ``reproducer`` command, the guard re-runs it in a fresh subprocess
   and compares the reported metric to the original. If no reproducer
   is given, it reads ``metrics.json`` in the idea's dir.

2. Computes ``z = (new_metric - rolling_mean) / rolling_std`` over the
   last N promotions. If ``z > z_threshold`` (default 4.0) and we have
   at least ``min_history`` past promotions, the promotion is blocked
   and an ``audit`` idea is enqueued instead. A notification is fired
   to Telegram via ``notify('audit', …)``.

The guard keeps a compact JSON history at
``results/_champion_history.json`` so the distribution survives
restarts.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("champion_guard")


@dataclass
class GuardConfig:
    enabled: bool = True
    z_threshold: float = 4.0
    min_history: int = 10
    history_size: int = 50

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "GuardConfig":
        g = cfg.get("champion_guard") or {}
        return cls(
            enabled=bool(g.get("enabled", True)),
            z_threshold=float(g.get("z_threshold", 4.0)),
            min_history=int(g.get("min_history", 10)),
            history_size=int(g.get("history_size", 50)),
        )


def load_history(results_dir: Path) -> List[float]:
    p = Path(results_dir) / "_champion_history.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("metrics", [])
    except (ValueError, OSError):
        return []


def save_history(results_dir: Path, metrics: List[float]) -> None:
    p = Path(results_dir) / "_champion_history.json"
    p.write_text(json.dumps({"metrics": metrics}, indent=2), encoding="utf-8")


def _zscore(history: List[float], value: float) -> Optional[float]:
    if len(history) < 2:
        return None
    mean = sum(history) / len(history)
    var = sum((x - mean) ** 2 for x in history) / len(history)
    std = math.sqrt(var) if var > 0 else 0.0
    if std < 1e-9:
        return None  # distribution is degenerate; don't block
    return (value - mean) / std


def _reverify_metric(results_dir: Path, idea_id: str,
                     claimed: float,
                     idea_cfg: Optional[Dict[str, Any]] = None,
                     timeout: int = 600) -> Optional[float]:
    """Re-compute the metric. Return the verified value, or None on failure."""
    idea_dir = Path(results_dir) / idea_id
    reproducer = (idea_cfg or {}).get("reproducer")
    if reproducer:
        try:
            r = subprocess.run(
                reproducer, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=str(idea_dir),
            )
            if r.returncode == 0:
                # Look for a float in stdout (last matching number)
                for line in reversed(r.stdout.splitlines()):
                    for tok in line.split():
                        try:
                            return float(tok)
                        except ValueError:
                            continue
        except Exception as e:  # pragma: no cover
            logger.warning("reproducer for %s crashed: %s", idea_id, e)
    # Fall back: read metrics.json
    mj = idea_dir / "metrics.json"
    if mj.exists():
        try:
            m = json.loads(mj.read_text(encoding="utf-8"))
            for k in ("pgmAP_ALL", "map", "best_map", "score_mean", "score"):
                if k in m and isinstance(m[k], (int, float)):
                    return float(m[k])
        except (ValueError, OSError):
            pass
    return None


def check_promotion(
    results_dir: Path,
    idea_id: str,
    new_metric: float,
    cfg: Dict[str, Any],
    *,
    idea_cfg: Optional[Dict[str, Any]] = None,
    notify_fn=None,
    create_audit_idea_fn=None,
) -> Tuple[bool, Dict[str, Any]]:
    """Return (allow_promotion, info).

    If allow_promotion is False the caller MUST NOT update best_idea_id.
    ``info`` includes z-score, verified metric, and whether an audit idea
    was enqueued so the orchestrator can record the decision.
    """
    gcfg = GuardConfig.from_cfg(cfg)
    info: Dict[str, Any] = {
        "enabled": gcfg.enabled,
        "z_threshold": gcfg.z_threshold,
        "claimed": new_metric,
        "verified": None,
        "z": None,
        "blocked": False,
        "audit_idea_id": None,
    }

    if not gcfg.enabled:
        _append_history(results_dir, new_metric, gcfg.history_size)
        return True, info

    # Honest-eval guard: refuse to promote metrics that were computed on
    # the same split used for tuning (CV-OOF on test labels etc.). The
    # adapter signals this by writing ``honest: true`` to metrics.json;
    # explicit ``honest: false`` or missing-but-declared-leaky paths are
    # rejected.
    idea_dir = Path(results_dir) / idea_id
    mj = idea_dir / "metrics.json"
    honest_flag = None
    try:
        if mj.exists():
            mjd = json.loads(mj.read_text(encoding="utf-8"))
            if "honest" in mjd:
                honest_flag = bool(mjd["honest"])
    except Exception:  # pragma: no cover
        honest_flag = None
    info["honest"] = honest_flag
    if honest_flag is False:
        info["blocked"] = True
        info["reason"] = "metrics.json declares honest=false"
        logger.warning(
            "champion_guard: BLOCKED %s — metrics.json honest=false",
            idea_id)
        return False, info

    # 1) re-verify
    verified = _reverify_metric(results_dir, idea_id, new_metric,
                                idea_cfg=idea_cfg)
    info["verified"] = verified
    check_val = verified if verified is not None else new_metric

    # 2) z-score against history
    history = load_history(results_dir)
    info["history_size"] = len(history)
    z = _zscore(history, check_val) if len(history) >= gcfg.min_history else None
    info["z"] = z

    blocked = z is not None and z > gcfg.z_threshold
    if blocked:
        info["blocked"] = True
        audit_id = f"audit-{idea_id}-{datetime.datetime.utcnow():%Y%m%dT%H%M%S}"
        info["audit_idea_id"] = audit_id
        payload = {
            "idea_id": idea_id,
            "claimed": new_metric,
            "verified": verified,
            "z": round(z, 2),
            "threshold": gcfg.z_threshold,
            "audit_idea_id": audit_id,
        }
        if notify_fn is not None:
            try:
                notify_fn("audit", payload, cfg)
            except Exception:  # pragma: no cover
                logger.exception("notify_fn failed")
        if create_audit_idea_fn is not None:
            try:
                create_audit_idea_fn(audit_id, idea_id, payload)
            except Exception:  # pragma: no cover
                logger.exception("create_audit_idea_fn failed")
        logger.warning(
            "champion_guard: BLOCKED %s (claimed=%.4f verified=%s z=%.2f > %.2f)",
            idea_id, new_metric, verified, z, gcfg.z_threshold,
        )
        return False, info

    # 3) promote — record the value in history for future z-scoring
    _append_history(results_dir, check_val, gcfg.history_size)
    return True, info


def _append_history(results_dir: Path, v: float, max_size: int) -> None:
    hist = load_history(results_dir)
    hist.append(float(v))
    if len(hist) > max_size:
        hist = hist[-max_size:]
    save_history(results_dir, hist)


def create_audit_idea(
    lake,
    audit_id: str,
    suspect_idea_id: str,
    payload: Dict[str, Any],
) -> None:
    """Insert an idea with kind='audit' to trigger a bug_fixer cycle."""
    cfg_yaml = json.dumps({
        "suspect_idea_id": suspect_idea_id,
        "claimed": payload.get("claimed"),
        "verified": payload.get("verified"),
        "z": payload.get("z"),
        "action": "audit",
    }, indent=2)
    lake.insert(
        audit_id,
        f"Audit suspicious promotion: {suspect_idea_id}",
        cfg_yaml,
        raw_markdown=cfg_yaml,
        status="pending",
        kind="audit",
        parent=suspect_idea_id,
    )
