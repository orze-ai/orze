"""Backwards-compatible re-export.

`orze-claude` is the first and currently only activated entry point in
the generic LLM-CLI shim system. All logic lives in
:mod:`orze.shims.llm` — this module just preserves the import path
`orze.shims.claude:main` that early releases wired into pyproject.

New code should configure new backends in ``orze.shims.llm.BACKENDS``
and add ``orze-<name> = "orze.shims.llm:main"`` entries in
``pyproject.toml`` — no new shim module needed.
"""
from orze.shims.llm import main  # noqa: F401

__all__ = ["main"]
