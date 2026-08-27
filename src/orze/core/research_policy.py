"""Fail-closed admission policy for autonomous research proposals."""

from __future__ import annotations

import math
import re
from typing import Mapping, Optional


UNRESTRICTED_MODEL_FORM = "unrestricted"
SINGLE_MODEL_FORM = "single_model_single_pass"
MODEL_FORMS = frozenset({UNRESTRICTED_MODEL_FORM, SINGLE_MODEL_FORM})
AUTONOMOUS_APPROACH_FAMILIES = frozenset({
    "architecture",
    "data",
    "ensemble",
    "infrastructure",
    "optimization",
    "other",
    "regularization",
    "training_config",
})

# These fields unambiguously request multiple inference-time model artifacts.
# Training-time LoRA, EMA, SWA, distillation, and checkpoint merging are not on
# this list: they may still produce one self-contained inference artifact.
_SINGLE_MODEL_FORBIDDEN_KEYS = frozenset({
    "decode_model_paths",
    "ensemble",
    "ensemble_components",
    "ensemble_members",
    "ensemble_models",
    "ensemble_paths",
    "ensemble_weights",
    "inference_model_paths",
    "per_dataset_models",
    "rover_models",
    "voters",
    "voting_models",
})
_MODEL_FORM_FIELDS = frozenset({
    "approach_family",
    "decoder_method",
    "inference_method",
    "kind",
    "method",
    "strategy",
})
_COMPOSITE_VALUE_TOKENS = frozenset({
    "consensus",
    "ensemble",
    "rover",
    "voter",
    "voting",
})
_DECISION_COMPARATORS = frozenset({"lt", "lte", "gt", "gte"})
_FAILURE_ACTIONS = frozenset({"redirect_family", "stop_branch"})
_SAFE_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")


def _safe_name(value) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_")
    return normalized if _SAFE_NAME_RE.fullmatch(normalized) else None


def validate_research_policy_config(cfg: Mapping) -> list[str]:
    """Return stable errors for the optional ``research_policy`` contract."""
    policy = cfg.get("research_policy")
    if policy is None:
        return []
    prefix = "research_policy"
    if not isinstance(policy, dict):
        return [f"{prefix}: must be a mapping"]
    errors = []
    model_form = policy.get("model_form", UNRESTRICTED_MODEL_FORM)
    if model_form not in MODEL_FORMS:
        errors.append(
            f"{prefix}.model_form: must be one of "
            f"{sorted(MODEL_FORMS)!r}"
        )
    for field in ("forbidden_approach_families", "forbidden_config_keys"):
        values = policy.get(field)
        if values is None:
            continue
        normalized = (
            [_safe_name(value) for value in values]
            if isinstance(values, list) else []
        )
        if (not isinstance(values, list)
                or any(value is None for value in normalized)
                or len(normalized) != len(set(normalized))):
            errors.append(
                f"{prefix}.{field}: must be a list of unique safe names"
            )
    required = policy.get("require_batch_decision_contract", False)
    if not isinstance(required, bool):
        errors.append(
            f"{prefix}.require_batch_decision_contract: must be true or false"
        )
    maximum = policy.get("max_decision_batch", 8)
    if (isinstance(maximum, bool) or not isinstance(maximum, int)
            or not 1 <= maximum <= 64):
        errors.append(
            f"{prefix}.max_decision_batch: must be an integer from 1 to 64"
        )
    minimum_effect = policy.get("min_decision_effect", 0.0)
    if (isinstance(minimum_effect, bool)
            or not isinstance(minimum_effect, (int, float))
            or not math.isfinite(float(minimum_effect))
            or float(minimum_effect) < 0.0):
        errors.append(
            f"{prefix}.min_decision_effect: must be a finite non-negative number"
        )
    return errors


def single_model_required(cfg: Mapping) -> bool:
    """Return whether either policy boundary requires one inference model."""
    policy = cfg.get("research_policy")
    explicit = (
        isinstance(policy, dict)
        and policy.get("model_form") == SINGLE_MODEL_FORM
    )
    report = cfg.get("report")
    contract = report.get("benchmark_contract") if isinstance(report, dict) else None
    benchmark = (
        isinstance(contract, dict)
        and contract.get("model_form") == SINGLE_MODEL_FORM
    )
    return explicit or benchmark


def batch_decision_contract_required(cfg: Mapping) -> bool:
    """Return whether autonomous batches need a prospective decision rule."""
    policy = cfg.get("research_policy")
    return (
        isinstance(policy, dict)
        and policy.get("require_batch_decision_contract") is True
    )


def _bounded_statement(value) -> bool:
    return (
        isinstance(value, str)
        and 20 <= len(value.strip()) <= 500
        and not any(ord(character) < 32 for character in value)
    )


def validate_batch_decision_contract(
    contract,
    cfg: Mapping,
    *,
    idea_count: int,
    qualified_best=None,
) -> Optional[str]:
    """Validate a prospective, batch-bound experiment decision contract."""
    if not batch_decision_contract_required(cfg):
        return None
    if not isinstance(contract, dict):
        return "batch_decision_contract_missing"
    if set(contract) != {
        "uncertainty",
        "metric",
        "baseline",
        "comparator",
        "threshold",
        "on_failure",
        "max_experiments",
    }:
        return "batch_decision_contract_fields_invalid"
    if not _bounded_statement(contract.get("uncertainty")):
        return "batch_decision_contract_uncertainty_invalid"

    report = cfg.get("report")
    if not isinstance(report, dict):
        return "batch_decision_contract_report_invalid"
    metric = report.get("primary_metric")
    if (not isinstance(metric, str) or not metric
            or contract.get("metric") != metric):
        return "batch_decision_contract_metric_mismatch"
    comparator = contract.get("comparator")
    if comparator not in _DECISION_COMPARATORS:
        return "batch_decision_contract_comparator_invalid"
    sort_order = report.get("sort", "descending")
    expected = {"lt", "lte"} if sort_order == "ascending" else {"gt", "gte"}
    if sort_order not in ("ascending", "descending"):
        return "batch_decision_contract_sort_invalid"
    if comparator not in expected:
        return "batch_decision_contract_direction_mismatch"
    baseline = contract.get("baseline")
    if qualified_best is None:
        if baseline is not None:
            return "batch_decision_contract_baseline_mismatch"
    else:
        if (isinstance(qualified_best, bool)
                or not isinstance(qualified_best, (int, float))
                or not math.isfinite(float(qualified_best))):
            return "batch_decision_contract_authoritative_baseline_invalid"
        if (isinstance(baseline, bool)
                or not isinstance(baseline, (int, float))
                or not math.isfinite(float(baseline))
                or not math.isclose(
                    float(baseline), float(qualified_best),
                    rel_tol=0.0, abs_tol=1.0e-12)):
            return "batch_decision_contract_baseline_mismatch"
    threshold = contract.get("threshold")
    if (isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))):
        return "batch_decision_contract_threshold_invalid"
    if qualified_best is not None:
        minimum_effect = (cfg.get("research_policy") or {}).get(
            "min_decision_effect", 0.0)
        if (isinstance(minimum_effect, bool)
                or not isinstance(minimum_effect, (int, float))
                or not math.isfinite(float(minimum_effect))
                or float(minimum_effect) < 0.0):
            return "batch_decision_contract_minimum_effect_invalid"
        if sort_order == "ascending":
            effect = float(qualified_best) - float(threshold)
        else:
            effect = float(threshold) - float(qualified_best)
        if (effect <= 0.0
                or effect + 1.0e-12 < float(minimum_effect)):
            return "batch_decision_contract_effect_too_small"
    if contract.get("on_failure") not in _FAILURE_ACTIONS:
        return "batch_decision_contract_failure_action_invalid"

    maximum = (cfg.get("research_policy") or {}).get(
        "max_decision_batch", 8)
    experiments = contract.get("max_experiments")
    if (isinstance(experiments, bool) or not isinstance(experiments, int)
            or isinstance(maximum, bool) or not isinstance(maximum, int)
            or not 1 <= experiments <= maximum):
        return "batch_decision_contract_budget_invalid"
    if (isinstance(idea_count, bool) or not isinstance(idea_count, int)
            or idea_count < 1 or experiments != idea_count):
        return "batch_decision_contract_count_mismatch"
    return None


def _is_populated(value) -> bool:
    return value not in (None, False, "", [], {})


def _find_forbidden_key(value, forbidden: frozenset[str], *, depth=0):
    """Return a content-safe structural path or ``None``."""
    if depth > 32:
        return "config_depth_limit"
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = _safe_name(raw_key)
            if key is not None and (
                    key in forbidden or key.startswith("ensemble_")):
                if _is_populated(child):
                    return f"config.{key}"
            if key in _MODEL_FORM_FIELDS:
                marker = _safe_name(child)
                if (marker is not None
                        and _COMPOSITE_VALUE_TOKENS.intersection(
                            marker.split("_"))):
                    return f"config.{key}"
            found = _find_forbidden_key(child, forbidden, depth=depth + 1)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_forbidden_key(child, forbidden, depth=depth + 1)
            if found:
                return found
    return None


def validate_idea_against_research_policy(
    idea_cfg: Mapping,
    cfg: Mapping,
    *,
    approach_family: Optional[str] = None,
) -> Optional[str]:
    """Reject a proposal that violates the declared model-form contract."""
    policy = cfg.get("research_policy")
    if policy is None:
        policy = {}
    if not isinstance(policy, dict):
        return "research_policy_invalid"
    model_form = policy.get("model_form", UNRESTRICTED_MODEL_FORM)
    if model_form not in MODEL_FORMS:
        return "research_policy_model_form_invalid"
    if not isinstance(idea_cfg, Mapping):
        return "research_policy_idea_config_invalid"
    if batch_decision_contract_required(cfg):
        # Implicit Cartesian expansion would turn one receipt-bound proposal
        # into unbound ``-ht-N`` executions and silently exceed its declared
        # experiment count. Contract-governed batches must spell out each arm.
        from orze.core.ideas import config_has_implicit_sweep
        if config_has_implicit_sweep(idea_cfg):
            return "batch_decision_contract_implicit_sweep_forbidden"
    if not single_model_required(cfg):
        return None

    forbidden_families = policy.get("forbidden_approach_families", [])
    if not isinstance(forbidden_families, list):
        return "research_policy_forbidden_families_invalid"
    closed_families = {
        "ensemble",
        *(_safe_name(value) for value in forbidden_families),
    }
    if None in closed_families:
        return "research_policy_forbidden_families_invalid"
    family = _safe_name(approach_family)
    if approach_family is not None and family is None:
        return "research_policy_approach_family_invalid"
    if family in closed_families:
        return f"research_policy_approach_family_forbidden:{family}"

    configured_keys = policy.get("forbidden_config_keys", [])
    if not isinstance(configured_keys, list):
        return "research_policy_forbidden_config_keys_invalid"
    closed_keys = {_safe_name(value) for value in configured_keys}
    if None in closed_keys:
        return "research_policy_forbidden_config_keys_invalid"
    forbidden = frozenset(_SINGLE_MODEL_FORBIDDEN_KEYS | closed_keys)
    path = _find_forbidden_key(idea_cfg, forbidden)
    if path:
        return f"research_policy_composite_forbidden:{path}"
    return None
