"""Explicit evaluation-only readmission; never requeue successful training.

CALLING SPEC:
    request_evaluation_retry(idea_id, results_dir, cfg, lake) -> dict
        Returns evaluation_retry_pending with idea_id/retry_id. Rejections
        raise EvaluationRetryError. Uses a strict lifecycle transaction and
        recoverable allowlisted file preparation; no GPU/provider/processes.
    pending_evaluation_retries(lake, limit=128) -> list[str]
        Reads admitted evaluation-only work from durable lifecycle history.
"""
from __future__ import annotations

import json
from pathlib import Path

from orze.engine.evaluation_retry_files import (
    EvaluationRetryError, prepare_retry_files, retry_file_policy, safe_file,
    verify_retry_prepared,
)


def _require_closed_evaluations(idea_dir: Path) -> None:
    root = idea_dir / "_compute_receipts"
    if root.is_symlink():
        raise EvaluationRetryError("evaluation_retry_compute_evidence_redirected")
    if not root.exists():
        return  # Legacy failures have no process receipts; FSM remains required.
    if not root.is_dir():
        raise EvaluationRetryError("evaluation_retry_compute_evidence_invalid")
    for attempt in root.iterdir():
        if not attempt.is_dir() or attempt.is_symlink():
            raise EvaluationRetryError("evaluation_retry_compute_evidence_invalid")
        start = safe_file(idea_dir, str((attempt / "start.json").relative_to(idea_dir)))
        if not start.exists():
            continue
        try:
            payload = json.loads(start.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError()
            if payload.get("phase") != "evaluation":
                continue
            terminal = safe_file(
                idea_dir, str((attempt / "terminal.json").relative_to(idea_dir)))
            closed = json.loads(terminal.read_text(encoding="utf-8"))
            if (not isinstance(closed, dict) or closed.get("phase") != "evaluation"
                    or closed.get("event") != "terminal"
                    or closed.get("attempt_id") != payload.get("attempt_id")
                    or closed.get("idea_id") != idea_dir.name
                    or closed.get("outcome") not in {"failed", "interrupted", "completed"}
                    or not isinstance(closed.get("return_code"), int)
                    or isinstance(closed["return_code"], bool)):
                raise ValueError()
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise EvaluationRetryError("evaluation_retry_termination_unconfirmed") from exc


def request_evaluation_retry(idea_id: str, results_dir: Path,
                             cfg: dict, lake) -> dict:
    from orze.core.benchmark_contract import benchmark_exposure_summary
    from orze.core.ideas import IDEA_ID_PATTERN
    from orze.engine.evaluator import is_training_complete_for_downstream
    from orze.reporting.evidence import report_lifecycle_db_path
    import re

    if (not isinstance(idea_id, str) or len(idea_id) > 128
            or re.fullmatch(IDEA_ID_PATTERN, idea_id) is None):
        raise EvaluationRetryError("evaluation_retry_idea_id_invalid")
    if not cfg.get("eval_script"):
        raise EvaluationRetryError("evaluation_retry_evaluator_not_configured")
    results_dir = Path(results_dir).absolute()
    if Path(lake.db_path).absolute() != report_lifecycle_db_path(results_dir, cfg).absolute():
        raise EvaluationRetryError("evaluation_retry_project_database_mismatch")
    idea_dir = results_dir / idea_id
    # Every path is checked before even an archive directory is created.
    safe_file(idea_dir, "metrics.json")
    retry_file_policy(idea_dir, cfg)

    prior = lake.conn.execute(
        "SELECT MAX(id) FROM idea_transitions WHERE idea_id = ? AND to_state = 'FAILED'",
        (idea_id,),
    ).fetchone()[0]
    result = {"status": "evaluation_retry_pending", "idea_id": idea_id,
              "retry_id": str(prior) if prior is not None else None}
    fresh = False

    def prepare(failure_id):
        nonlocal fresh
        eligible, reason = is_training_complete_for_downstream(idea_dir, cfg)
        if not eligible:
            raise EvaluationRetryError(reason)
        _require_closed_evaluations(idea_dir)
        exposure = benchmark_exposure_summary(results_dir, cfg)
        if exposure.get("enabled"):
            if not exposure.get("valid"):
                raise EvaluationRetryError("evaluation_retry_benchmark_history_invalid")
            if exposure.get("remaining", 0) <= 0:
                raise EvaluationRetryError("evaluation_retry_benchmark_budget_exhausted")
        result["retry_id"] = prepare_retry_files(idea_dir, cfg, failure_id)
        fresh = True

    try:
        admitted = lake.retry_evaluation(idea_id, prepare_artifacts=prepare)
    except EvaluationRetryError:
        raise
    except (OSError, ValueError) as exc:
        raise EvaluationRetryError("evaluation_retry_preparation_failed") from exc
    if not admitted or result["retry_id"] is None:
        raise EvaluationRetryError("evaluation_retry_lifecycle_rejected")
    if not fresh:
        verify_retry_prepared(idea_dir, cfg, int(result["retry_id"]))
    return result


def validate_pending_retry(idea_id: str, results_dir: Path, cfg: dict, lake) -> None:
    """Gate only explicit retry launch; ordinary first evaluations are unchanged."""
    if lake is None:
        return
    latest = lake.conn.execute(
        "SELECT id, from_state, to_state FROM idea_transitions "
        "WHERE idea_id=? ORDER BY id DESC LIMIT 1", (idea_id,),
    ).fetchone()
    if latest is None or (latest[1], latest[2]) != ("FAILED", "IN_PROGRESS"):
        return
    if lake.get_stage_state(idea_id, "evaluation") != "PENDING":
        return
    failure_id = lake.conn.execute(
        "SELECT MAX(id) FROM idea_transitions WHERE idea_id=? "
        "AND to_state='FAILED' AND id<?", (idea_id, latest[0]),
    ).fetchone()[0]
    if failure_id is None:
        raise EvaluationRetryError("evaluation_retry_failure_identity_invalid")
    verify_retry_prepared(Path(results_dir) / idea_id, cfg, int(failure_id))


def pending_evaluation_retries(lake, limit: int = 128) -> list[str]:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1024:
        raise ValueError("evaluation_retry_query_limit_invalid")
    rows = lake.conn.execute(
        "SELECT i.idea_id FROM ideas i JOIN idea_state s ON s.idea_id=i.idea_id "
        "JOIN idea_stage_state t ON t.idea_id=i.idea_id AND t.stage='training' "
        "JOIN idea_stage_state e ON e.idea_id=i.idea_id AND e.stage='evaluation' "
        "JOIN idea_transitions h ON h.idea_id=i.idea_id AND h.id=(SELECT MAX(id) "
        "FROM idea_transitions WHERE idea_id=i.idea_id) "
        "WHERE lower(i.status)='running' AND s.current_state='IN_PROGRESS' "
        "AND t.current_state='COMPLETE' AND e.current_state='PENDING' "
        "AND h.from_state='FAILED' AND h.to_state='IN_PROGRESS' "
        "ORDER BY h.id LIMIT ?", (limit,),
    ).fetchall()
    return [row[0] for row in rows]
