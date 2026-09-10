"""Read the framework-designated evaluation document, without lifecycle claims.

CALLING SPEC:
    evaluation_output_path(idea_dir, cfg) -> safe Path | None
    evaluation_output_is_metric_source(cfg) -> bool
    validate_evaluation_output(idea_dir, cfg) -> (valid, stable_reason)
        Missing output is allowed: declared sources/benchmark contracts decide
        which measurements must exist. A present output must be a JSON object
        and its optional framework status must be COMPLETED.
    Read-only; never interprets status fields in unrelated domain documents.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def evaluation_output_path(idea_dir: Path, cfg: Mapping) -> Path | None:
    from orze.reporting.evidence import _evidence_path_unsafe, _safe_source_path

    path = _safe_source_path(
        idea_dir, cfg.get("eval_output") or "eval_report.json")
    if path is None or _evidence_path_unsafe(path):
        return None
    return path


def evaluation_output_is_metric_source(cfg: Mapping) -> bool:
    output = Path(str(cfg.get("eval_output") or "eval_report.json"))
    for column in (cfg.get("report") or {}).get("columns") or []:
        if not isinstance(column, Mapping):
            continue
        source = str(column.get("source") or "")
        if ":" in source and Path(source.split(":", 1)[0]) == output:
            return True
    return False


def evaluation_output_has_contract(cfg: Mapping) -> bool:
    """An active evaluator or explicit metric source gives output semantics."""
    return bool(cfg.get("eval_script")) or evaluation_output_is_metric_source(cfg)


def validate_evaluation_output(idea_dir: Path,
                               cfg: Mapping) -> tuple[bool, str]:
    path = evaluation_output_path(idea_dir, cfg)
    if path is None:
        return False, "evaluation_output_path_invalid"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return True, "evaluation_output_not_required_by_presence"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False, "evaluation_output_invalid"
    if not isinstance(document, dict):
        return False, "evaluation_output_invalid"
    if "status" in document and document["status"] != "COMPLETED":
        return False, "evaluation_output_not_completed"
    return True, "evaluation_output_verified"
