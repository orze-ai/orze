"""Shared declaration for complete research-prompt UTF-8 byte limits.

This is not a tokenizer, provider billing estimate, or HTTP payload limit.
Consumers must measure the final prompt again before dispatch.
"""
from __future__ import annotations

DEFAULT_PROMPT_BYTES = 131072
MAX_PROMPT_BYTES = 2097152


def prompt_byte_limit(project_cfg: dict | None) -> int:
    """Return a bounded limit; explicit malformed declarations never disable it."""
    if project_cfg is None:
        return DEFAULT_PROMPT_BYTES
    if not isinstance(project_cfg, dict):
        raise ValueError("research_prompt: project configuration must be a mapping")
    declaration = project_cfg.get("research_prompt", {})
    if not isinstance(declaration, dict):
        raise ValueError("research_prompt: must be a mapping")
    if any(key != "max_bytes" for key in declaration):
        raise ValueError("research_prompt: only max_bytes is supported")
    value = declaration.get("max_bytes", DEFAULT_PROMPT_BYTES)
    if (isinstance(value, bool) or not isinstance(value, int)
            or not 1 <= value <= MAX_PROMPT_BYTES):
        raise ValueError(
            "research_prompt.max_bytes: must be an integer from 1 through 2097152")
    return value
