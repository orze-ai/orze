def qualify_local_report_evidence(
    idea_dir: Path,
    cfg: Mapping,
    *,
    require_primary: bool = True,
) -> tuple[dict, dict, float | None, str]:
    """Qualify one local result against the complete report policy.

    Returns ``(metrics, values, primary_value, reason)``. The reason is a
    stable token suitable for aggregate reporting; validation messages and
    artifact contents never cross this boundary.
    """
    if cfg.get("observation_contract") is not None:
        # This legacy adapter cannot turn task-level files into immutable
        # observations, even when the task's operational stage is COMPLETE.
        return {}, {}, None, "observation_adapter_required"
    report = cfg.get("report") or {}
    metrics, values, reason = load_local_report_evidence(idea_dir, report)
    if reason != "local_evidence_loaded":
        return metrics, values, None, reason
    from orze.reporting.evaluation_output import (
        evaluation_output_has_contract, validate_evaluation_output,
    )
    if evaluation_output_has_contract(cfg):
        valid, reason = validate_evaluation_output(idea_dir, cfg)
        if not valid:
            return metrics, values, None, reason
    # This optional adapter declaration can veto ranking, never prove validity.
    # Missing is supported; true still has to pass every other evidence check.
    if "honest" in metrics:
        if not isinstance(metrics["honest"], bool):
            return metrics, values, None, "local_honesty_declaration_invalid"
        if metrics["honest"] is False:
            return metrics, values, None, "local_evidence_declared_non_honest"
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
    primary = report.get("primary_metric", "score" if require_primary else None)
    value = values.get(primary) if isinstance(primary, str) else None
    if (require_primary or "primary_metric" in report) and not _finite_number(value):
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
    return (metrics, values, float(value) if _finite_number(value) else None,
            "local_evidence_verified")
