"""Shared qualification rules for locally comparable metric evidence."""
from __future__ import annotations

import math
import json
import re
import sqlite3
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
    if idea_dir.is_symlink():
        return {}, {}, "local_idea_dir_symlink"
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


def qualify_local_report_evidence(
    idea_dir: Path,
    cfg: Mapping,
) -> tuple[dict, dict, float | None, str]:
    """Qualify one local result against the complete report policy.

    Returns ``(metrics, values, primary_value, reason)``. The reason is a
    stable token suitable for aggregate reporting; validation messages and
    artifact contents never cross this boundary.
    """
    report = cfg.get("report") or {}
    metrics, values, reason = load_local_report_evidence(idea_dir, report)
    if reason != "local_evidence_loaded":
        return metrics, values, None, reason
    try:
        from orze.core.integrity import validate_metrics
        resolved_metrics = dict(metrics)
        resolved_metrics.update(values)
        resolved_metrics["status"] = "COMPLETED"
        valid, _ = validate_metrics(resolved_metrics, dict(cfg))
    except Exception:
        valid = False
    if not valid:
        return metrics, values, None, "local_metric_validation_failed"
    primary = report.get("primary_metric", "score")
    value = values.get(primary) if isinstance(primary, str) else None
    if not _finite_number(value):
        return metrics, values, None, "primary_metric_missing_or_nonfinite"
    coverage_ok, observed, required = minimum_dataset_coverage(
        report, values=values, metrics=metrics)
    if not coverage_ok:
        return (
            metrics,
            values,
            None,
            f"metric_coverage_below_min:{observed}/{required}",
        )
    lineage = cfg.get("model_lineage", {})
    if isinstance(lineage, Mapping) and lineage.get("enabled") is True:
        try:
            from orze.core.model_lineage import (
                validate_model_lineage_for_evaluation,
            )
            validate_model_lineage_for_evaluation(Path(idea_dir), cfg)
        except Exception:
            return metrics, values, None, "local_model_lineage_invalid"
    return metrics, values, float(value), "local_evidence_verified"


_SAFE_FAMILY_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


def _authoritative_completed_rows(
    db_path: Path,
    *,
    include_family: bool,
) -> tuple[list[tuple], str]:
    """Read agreed lifecycle-complete rows under the shared DB policy."""
    path = Path(db_path)
    try:
        absolute = path.absolute()
        current = Path(absolute.anchor)
        redirected = False
        for part in absolute.parts[1:]:
            current = current / part
            if current.is_symlink():
                redirected = True
                break
        if redirected:
            return [], "authoritative_lifecycle_database_redirected"
        if not path.is_file():
            return [], "authoritative_lifecycle_database_unavailable"
        if path.stat().st_nlink != 1:
            return [], "authoritative_lifecycle_database_redirected"
        connection = sqlite3.connect(
            absolute.as_uri() + "?mode=ro",
            uri=True,
            timeout=5,
        )
        try:
            connection.execute("PRAGMA query_only=ON")
            from orze.core.sqlite_policy import (
                SQLitePolicyError,
                inspect_shared_database_policy,
            )
            try:
                policy = inspect_shared_database_policy(connection)
            except SQLitePolicyError:
                return [], "authoritative_lifecycle_database_invalid"
            if not policy["compliant"]:
                return [], "authoritative_lifecycle_database_policy_invalid"
            select = (
                "i.idea_id, i.approach_family"
                if include_family else "i.idea_id"
            )
            rows = connection.execute(
                f"SELECT {select} FROM ideas AS i "
                "JOIN idea_state AS s ON s.idea_id = i.idea_id "
                "WHERE lower(i.status) = 'completed' "
                "AND s.current_state = 'COMPLETE'"
            ).fetchall()
        finally:
            connection.close()
    except (OSError, sqlite3.Error):
        return [], "authoritative_lifecycle_database_invalid"
    return rows, "authoritative_lifecycle_loaded"


def authoritative_completed_idea_ids(
    db_path: Path,
) -> tuple[set[str], str]:
    """Load lifecycle-complete idea IDs from the authoritative lake read-only.

    A metrics artifact is result evidence, not lifecycle authority. Consumers
    that steer research must require both the audited FSM state and the legacy
    status mirror to agree on completion. Missing, redirected, hard-linked, or
    incompatible databases fail closed and never get created by inspection.
    """
    rows, reason = _authoritative_completed_rows(
        db_path, include_family=False)
    if reason != "authoritative_lifecycle_loaded":
        return set(), reason

    completed = {
        str(row[0])
        for row in rows
        if row and isinstance(row[0], str)
        and row[0] not in ("", ".", "..")
        and Path(row[0]).parts == (row[0],)
    }
    return completed, reason


def authoritative_completed_idea_families(
    db_path: Path,
) -> tuple[dict[str, str], str]:
    """Load closed, content-safe family labels for lifecycle-complete IDs."""
    rows, reason = _authoritative_completed_rows(
        db_path, include_family=True)
    if reason != "authoritative_lifecycle_loaded":
        return {}, reason
    families = {}
    for row in rows:
        if (not row or not isinstance(row[0], str)
                or row[0] in ("", ".", "..")
                or Path(row[0]).parts != (row[0],)):
            continue
        raw_family = row[1] if len(row) > 1 else None
        family = str(raw_family or "other").strip().lower()
        if _SAFE_FAMILY_RE.fullmatch(family) is None:
            family = "other"
        families[row[0]] = family
    return families, reason


def qualify_authoritative_report_evidence(
    idea_id: str,
    results_dir: Path,
    cfg: Mapping,
    completed_idea_ids: set[str],
) -> tuple[dict, dict, float | None, str]:
    """Qualify one steering result against lifecycle and evidence contracts.

    The caller loads ``completed_idea_ids`` once per scan with
    :func:`authoritative_completed_idea_ids`. Benchmark-enabled projects also
    require the current sealed benchmark receipt; local evidence alone cannot
    spend a benchmark-driven plateau budget.
    """
    if (not isinstance(idea_id, str) or idea_id in ("", ".", "..")
            or Path(idea_id).parts != (idea_id,)):
        return {}, {}, None, "idea_id_invalid"
    if idea_id not in completed_idea_ids:
        return {}, {}, None, "authoritative_lifecycle_not_complete"

    idea_dir = Path(results_dir) / idea_id
    metrics, values, value, reason = qualify_local_report_evidence(
        idea_dir, cfg)
    if reason != "local_evidence_verified":
        return metrics, values, None, reason
    if metrics.get("tainted_leakage"):
        return metrics, values, None, "local_evidence_tainted_leakage"

    report = cfg.get("report") or {}
    if isinstance(report, Mapping) and report.get("benchmark_contract"):
        try:
            from orze.core.benchmark_contract import validate_benchmark_receipt
            valid, benchmark_reason = validate_benchmark_receipt(
                idea_dir, cfg, values=values)
        except Exception:
            return metrics, values, None, "benchmark_validation_failed"
        if not valid:
            return metrics, values, None, str(
                benchmark_reason or "benchmark_receipt_invalid")
        return metrics, values, value, "benchmark_evidence_verified"

    return metrics, values, value, "authoritative_local_evidence_verified"


def local_report_evidence_paths(idea_dir: Path,
                                report_cfg: Mapping) -> list[Path]:
    """Return only safe paths that can affect local report qualification."""
    idea_dir = Path(idea_dir)
    paths = [idea_dir / "metrics.json"]
    for column in report_cfg.get("columns") or []:
        if not isinstance(column, dict):
            continue
        source = column.get("source", "")
        if not source or ":" not in str(source):
            continue
        filename = str(source).split(":", 1)[0]
        path = _safe_source_path(idea_dir, filename)
        if path is not None and path not in paths:
            paths.append(path)
    return paths
