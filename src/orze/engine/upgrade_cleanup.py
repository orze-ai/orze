"""Shim — contents merged into engine/upgrade.py in v4.0."""
from orze.engine.upgrade import (  # noqa: F401
    STAMP_FILENAME,
    _GARBAGE_FILES,
    _GARBAGE_GLOBS,
    _current_versions,
    _read_stamp,
    _write_stamp,
    _same_versions,
    _is_live_trigger,
    _delete_garbage,
    check_and_clean,
)
