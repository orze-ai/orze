"""Fail-closed admission policy for autonomous research proposals."""

from __future__ import annotations

import re
from typing import Mapping, Optional


UNRESTRICTED_MODEL_FORM = "unrestricted"
SINGLE_MODEL_FORM = "single_model_single_pass"
MODEL_FORMS = frozenset({UNRESTRICTED_MODEL_FORM, SINGLE_MODEL_FORM})

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
    if not single_model_required(cfg):
        return None
    if not isinstance(idea_cfg, Mapping):
        return "research_policy_idea_config_invalid"

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
