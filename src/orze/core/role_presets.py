"""Explicit permission for optional role defaults, independent of credentials.

CALLING SPEC:
    configured_role_presets(cfg) -> tuple[str, ...]
        Missing means none. Invalid or duplicate names raise ValueError.
    role_preset_enabled(cfg, name) -> bool
        The runtime predicate fails closed for any invalid preset declaration.
    apply_environment_research(cfg) -> list[str]
        Add missing environment-backed roles only after explicit opt-in.

Core recognizes strategy_team for the shared contract but does not implement
its Pro roles, SOPs or trigger writes. Explicit role entries always win whole.
"""
import os


ROLE_PRESETS = frozenset({"environment_research", "strategy_team"})
_ENVIRONMENT_BACKENDS = (
    ("GEMINI_API_KEY", "gemini", "gemini-2.5-flash"),
    ("OPENAI_API_KEY", "openai", "gpt-4o"),
    ("ANTHROPIC_API_KEY", "anthropic", None),
)


def configured_role_presets(cfg):
    if not isinstance(cfg, dict):
        raise ValueError("role_presets: configuration must be a mapping")
    values = cfg.get("role_presets", [])
    if not isinstance(values, list):
        raise ValueError("role_presets: expected a list of preset names")
    if any(not isinstance(value, str) or value not in ROLE_PRESETS for value in values):
        raise ValueError("role_presets: expected only environment_research or strategy_team")
    if len(set(values)) != len(values):
        raise ValueError("role_presets: duplicate preset names are not allowed")
    return tuple(values)


def role_preset_enabled(cfg, name):
    try:
        return name in ROLE_PRESETS and name in configured_role_presets(cfg)
    except (TypeError, ValueError):
        return False


def apply_environment_research(cfg):
    if "environment_research" not in configured_role_presets(cfg):
        return []
    roles = cfg.setdefault("roles", {})
    if not isinstance(roles, dict):
        raise ValueError("roles: expected a mapping for environment_research preset")
    added = []
    for variable, backend, model in _ENVIRONMENT_BACKENDS:
        name = f"research_{backend}"
        if name in roles or not os.environ.get(variable):
            continue
        role = {"mode": "research", "backend": backend}
        if model:
            role["model"] = model
        roles[name] = role
        added.append(name)
    return added
