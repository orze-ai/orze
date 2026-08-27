"""Shared qualification rules for locally comparable metric evidence."""
from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Mapping


def _finite_number(value) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def qualification_is_presentable(qualification) -> bool:
    """Return whether a qualification summary is safe beside a numeric score."""
    if not isinstance(qualification, Mapping):
        return False
    if qualification.get("mode") not in {
            "verified_local_artifact", "benchmark_contract"}:
        return False
    if qualification.get("fallback_metrics_allowed") is not False:
        return False
    primary = qualification.get("primary_metric")
    if not isinstance(primary, str) or not primary.strip():
        return False
    accepted = qualification.get("accepted")
    if (isinstance(accepted, bool) or not isinstance(accepted, int)
            or accepted < 0):
        return False
    rejected = qualification.get("rejected")
    if not isinstance(rejected, Mapping):
        return False
    if not all(
        isinstance(reason, str)
        and not isinstance(count, bool)
        and isinstance(count, int)
        and count >= 0
        for reason, count in rejected.items()
    ):
        return False
    if qualification.get("mode") == "benchmark_contract":
        return all(
            isinstance(qualification.get(key), str)
            and bool(qualification[key].strip())
            for key in (
                "benchmark_id", "benchmark_view", "evidence_scope",
                "selection_mode",
            )
        )
    return True


def efficiency_presentation_is_safe(presentation) -> bool:
    """Return whether a presentation block prevents leaderboard ambiguity."""
    return (
        isinstance(presentation, Mapping)
        and presentation.get("claim_scope") == "internal_research_efficiency"
        and presentation.get("qualification_applied") is True
        and isinstance(presentation.get("evidence_label"), str)
        and bool(presentation["evidence_label"].strip())
        and presentation.get("leaderboard_rank_comparable") is False
    )


def dataset_metric_keys(report_cfg: Mapping) -> list[str]:
    """Return the configured columns that constitute dataset coverage.

    Preserve the established report behavior: ASR-style ``wer_*`` columns are
    the explicit per-dataset set and exclude the primary aggregate; other tasks
    use their declared report columns.
    """
    primary = report_cfg.get("primary_metric")
    keys = [
        column["key"]
        for column in (report_cfg.get("columns") or [])
        if isinstance(column, dict) and column.get("key")
    ]
    wer_keys = [
        key for key in keys
        if key.startswith("wer_") and key != primary
    ]
    return wer_keys or keys


def count_dataset_metrics(
    report_cfg: Mapping,
    *,
    values: Mapping | None = None,
    metrics: Mapping | None = None,
) -> int:
    """Count finite, non-boolean configured dataset measurements."""
    values = values if isinstance(values, Mapping) else {}
    metrics = metrics if isinstance(metrics, Mapping) else {}
    dataset_keys = dataset_metric_keys(report_cfg)
    count = sum(
        1 for key in dataset_keys
        if _finite_number(values[key] if key in values else metrics.get(key))
    )
    if not dataset_keys:
        # Backward-compatible evidence for projects that configured no useful
        # columns but historically emitted flat per-dataset WER keys.
        count = sum(
            1 for key, value in metrics.items()
            if str(key).startswith("wer_") and _finite_number(value)
        )
    return count


def minimum_dataset_coverage(
    report_cfg: Mapping,
    *,
    values: Mapping | None = None,
    metrics: Mapping | None = None,
) -> tuple[bool, int, int]:
    """Return ``(qualified, observed, required)`` for report coverage."""
    raw_required = report_cfg.get("min_datasets", 0) or 0
    if isinstance(raw_required, bool):
        return False, 0, -1
    try:
        required = int(raw_required)
    except (TypeError, ValueError):
        return False, 0, -1
    if isinstance(raw_required, float) and not raw_required.is_integer():
        return False, 0, -1
    if required < 0:
        return False, 0, required
    observed = count_dataset_metrics(
        report_cfg, values=values, metrics=metrics)
    return observed >= required, observed, required


def _deep_value(document, key: str):
    if isinstance(document, Mapping) and key in document:
        return document[key]
    value = document
    for part in str(key).split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _safe_source_path(idea_dir: Path, filename: str) -> Path | None:
    relative = Path(str(filename))
    if (relative.is_absolute() or relative == Path(".")
            or ".." in relative.parts):
        return None
    path = Path(idea_dir) / relative
    current = Path(idea_dir)
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return None
    return path


def load_local_report_evidence(
    idea_dir: Path,
    report_cfg: Mapping,
) -> tuple[dict, dict, str]:
    """Read current local result artifacts using the report's exact columns.

    IdeaLake is an index, not immutable result evidence.  This loader requires a
    current completed metrics artifact and rejects redirected source paths.  It
    returns stable reason codes and never includes artifact content in errors.
    """
    idea_dir = Path(idea_dir)
    metrics_path = idea_dir / "metrics.json"
    if metrics_path.is_symlink():
        return {}, {}, "local_metrics_symlink"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, {}, "local_metrics_missing"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}, {}, "local_metrics_invalid"
    if not isinstance(metrics, dict):
        return {}, {}, "local_metrics_invalid"
    if metrics.get("status") != "COMPLETED":
        return metrics, {}, "local_metrics_not_completed"

    values = {}
    for column in report_cfg.get("columns") or []:
        if not isinstance(column, dict) or not column.get("key"):
            continue
        key = str(column["key"])
        source = column.get("source", "")
        if source and ":" in str(source):
            filename, dotpath = str(source).split(":", 1)
            source_path = _safe_source_path(idea_dir, filename)
            if source_path is None:
                return metrics, values, "local_metric_source_path_invalid"
            try:
                document = json.loads(source_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                values[key] = None
                continue
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                return metrics, values, "local_metric_source_invalid"
            values[key] = _deep_value(document, dotpath)
        else:
            values[key] = _deep_value(metrics, key)
    primary = report_cfg.get("primary_metric")
    if isinstance(primary, str) and primary and primary not in values:
        values[primary] = _deep_value(metrics, primary)
    return metrics, values, "local_evidence_loaded"
