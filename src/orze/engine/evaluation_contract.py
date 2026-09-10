"""Shared non-lifecycle admission of completed evaluation artifacts.

CALLING SPEC:
    validate_evaluation_result(idea_dir, cfg) -> (valid, stable_reason)
        Used for existing-output reconciliation and both process completion
        entry points. Reads exact declared sources under one stable content
        identity; makes no claim that a task is already lifecycle-complete.
        No writes, GPU queries, or fallback/guessed optimization objective.
"""
from pathlib import Path

from orze.reporting.evaluation_output import (
    evaluation_output_path, validate_evaluation_output,
)
from orze.reporting.evidence import (
    evidence_content_sha256, qualify_result_artifacts, report_evidence_paths,
)


def validate_evaluation_result(idea_dir: Path, cfg: dict) -> tuple[bool, str]:
    idea_dir = Path(idea_dir)
    try:
        output = evaluation_output_path(idea_dir, cfg)
        if output is None:
            return False, "evaluation_output_path_invalid"
        sealed_files = cfg.get("sealed_files") or []
        if sealed_files:
            from orze.engine.sealed import load_sealed_manifest, verify_sealed_files
            if verify_sealed_files(
                    sealed_files, load_sealed_manifest(idea_dir.parent)):
                return False, "evaluation_sealed_files_changed"
        evidence_cfg = dict(cfg)
        evidence_cfg.setdefault("report", {})
        if evidence_cfg["report"] is None:
            evidence_cfg["report"] = {}
        paths = report_evidence_paths(
            idea_dir.name, idea_dir.parent, evidence_cfg) + [output]
        before = evidence_content_sha256(paths)
        valid, reason = validate_evaluation_output(idea_dir, evidence_cfg)
        if valid:
            _, _, _, reason = qualify_result_artifacts(
                idea_dir, evidence_cfg, require_primary=False)
            valid = reason in {
                "local_artifacts_verified", "benchmark_evidence_verified"}
        after = evidence_content_sha256(paths)
        if before != after:
            return False, "evaluation_evidence_changed_during_read"
        return valid, reason
    except Exception:
        return False, "evaluation_evidence_validation_failed"
