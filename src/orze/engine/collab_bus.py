"""Cross-role collaboration bus for structured message passing.

CALLING SPEC:
    post(results_dir, from_role, to_role, msg_type, payload) -> str
        Writes a timestamped JSON message to results/_collab/.
        Returns the message ID (filename stem).

    read(results_dir, role, msg_type=None, since=None, limit=10) -> list[dict]
        Reads messages addressed to `role`, optionally filtered by type
        and timestamp. Returns newest-first, up to `limit`.

    read_latest(results_dir, role, msg_type) -> dict | None
        Convenience: returns the single most recent message of the given type
        addressed to `role`, or None.

    format_for_prompt(messages, max_chars=4000) -> str
        Renders a list of collaboration messages as text suitable for
        injection into an LLM prompt. Truncates to max_chars.

Message types (conventions, not enforced):
    "diversity_report"     — from ensemble_strategist to research
    "diagnosis"            — from data_analyst to research/engineer
    "next_model_request"   — from ensemble_strategist to research
    "failure_analysis"     — from thinker to engineer
    "plateau_signal"       — from retrospection to thinker/meta_research
    "strategy_pivot"       — from meta_research to research
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("orze.collab")

COLLAB_DIR = "_collab"


def _collab_path(results_dir: Path) -> Path:
    p = Path(results_dir) / COLLAB_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def post(
    results_dir: Path,
    from_role: str,
    to_role: str,
    msg_type: str,
    payload: Any,
) -> str:
    """Post a structured message from one role to another."""
    ts = datetime.datetime.utcnow()
    msg_id = f"{from_role}_to_{to_role}_{msg_type}_{ts.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    msg = {
        "id": msg_id,
        "from": from_role,
        "to": to_role,
        "type": msg_type,
        "timestamp": ts.isoformat() + "Z",
        "payload": payload,
    }

    out_path = _collab_path(results_dir) / f"{msg_id}.json"
    out_path.write_text(json.dumps(msg, indent=2, default=str), encoding="utf-8")
    logger.info("collab: %s -> %s [%s]", from_role, to_role, msg_type)
    return msg_id


def read(
    results_dir: Path,
    role: str,
    msg_type: Optional[str] = None,
    since: Optional[datetime.datetime] = None,
    limit: int = 10,
) -> List[Dict]:
    """Read messages addressed to a role."""
    collab = _collab_path(results_dir)
    messages = []

    for f in sorted(collab.glob("*.json"), reverse=True):
        try:
            msg = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        if msg.get("to") != role and msg.get("to") != "*":
            continue
        if msg_type and msg.get("type") != msg_type:
            continue
        if since:
            msg_ts = datetime.datetime.fromisoformat(msg["timestamp"].rstrip("Z"))
            if msg_ts < since:
                continue

        messages.append(msg)
        if len(messages) >= limit:
            break

    return messages


def read_latest(
    results_dir: Path,
    role: str,
    msg_type: str,
) -> Optional[Dict]:
    """Read the most recent message of a given type for a role."""
    msgs = read(results_dir, role, msg_type=msg_type, limit=1)
    return msgs[0] if msgs else None


def format_for_prompt(messages: List[Dict], max_chars: int = 4000) -> str:
    """Render collaboration messages as text for LLM prompt injection."""
    if not messages:
        return ""

    lines = ["## Cross-Role Intelligence\n"]
    total = 0

    for msg in messages:
        header = f"### From {msg['from']} ({msg['type']}, {msg['timestamp'][:16]})\n"
        payload = msg.get("payload", "")
        if isinstance(payload, dict):
            body = json.dumps(payload, indent=2, default=str)
        else:
            body = str(payload)

        entry = header + body + "\n"
        if total + len(entry) > max_chars:
            lines.append(f"\n... ({len(messages) - len(lines) + 1} more messages truncated)")
            break
        lines.append(entry)
        total += len(entry)

    return "\n".join(lines)


def cleanup_old(results_dir: Path, max_age_days: int = 7) -> int:
    """Remove collaboration messages older than max_age_days."""
    collab = _collab_path(results_dir)
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=max_age_days)
    removed = 0

    for f in collab.glob("*.json"):
        try:
            msg = json.loads(f.read_text(encoding="utf-8"))
            msg_ts = datetime.datetime.fromisoformat(msg["timestamp"].rstrip("Z"))
            if msg_ts < cutoff:
                f.unlink()
                removed += 1
        except (json.JSONDecodeError, OSError, KeyError):
            continue

    return removed
