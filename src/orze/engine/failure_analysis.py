"""Shim — contents merged into engine/failure.py in v4.0."""
from orze.engine.failure import (  # noqa: F401
    FAILURE_CATEGORIES,
    classify_failure,
    build_failure_analysis,
    write_failure_analysis,
    load_recent_failures,
)
