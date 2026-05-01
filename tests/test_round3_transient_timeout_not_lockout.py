"""Round-3: a single transient timeout must not classify a role as
LOCKED_OUT.

The TIMEOUT branch in role_runner sets ``cooldown_override =
base_cooldown * 2`` on the *first* timeout. For any role configured
with ``cooldown >= 1801s`` this immediately exceeds the 3600s
LOCKED_OUT threshold despite the role being one transient hang away
from a clean cycle. Require a real fault signal — non-zero error
counter or repeated timeouts — before declaring lockout.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orze.reporting.state import derive_role_health  # noqa: E402


def test_single_timeout_with_2x_cooldown_is_not_locked_out(tmp_path):
    """Single timeout on a long-cooldown role: 2× bump trips 3600s
    threshold but the role has zero errors and only one timeout — must
    classify as HEALTHY (eligible to retry after the cooldown), not
    LOCKED_OUT."""
    orze_dir = tmp_path / ".orze"
    (orze_dir / "logs" / "code_evolution").mkdir(parents=True)
    role_state = {
        "last_run_time": 0,
        "consecutive_failures": 0,
        "consecutive_errors": 0,
        "consecutive_timeouts": 1,
        "cooldown_override": 7200,  # base_cooldown=3600 × 2
    }
    role_cfg = {"mode": "claude", "cooldown": 3600}
    h = derive_role_health("code_evolution", role_state, orze_dir,
                           role_cfg=role_cfg, role_states={})
    assert h["status"] != "LOCKED_OUT", h


def test_repeated_timeouts_are_locked_out(tmp_path):
    """Two timeouts in a row + high cooldown: real fault, LOCKED_OUT."""
    orze_dir = tmp_path / ".orze"
    (orze_dir / "logs" / "code_evolution").mkdir(parents=True)
    role_state = {
        "last_run_time": 0,
        "consecutive_failures": 0,
        "consecutive_timeouts": 2,
        "cooldown_override": 7200,
    }
    role_cfg = {"mode": "claude", "cooldown": 3600}
    h = derive_role_health("code_evolution", role_state, orze_dir,
                           role_cfg=role_cfg, role_states={})
    assert h["status"] == "LOCKED_OUT", h


def test_high_errors_are_locked_out(tmp_path):
    """5+ consecutive errors: LOCKED_OUT regardless of cooldown."""
    orze_dir = tmp_path / ".orze"
    (orze_dir / "logs" / "engineer").mkdir(parents=True)
    role_state = {
        "last_run_time": 0,
        "consecutive_failures": 5,
        "cooldown_override": 0,
    }
    role_cfg = {"mode": "claude"}
    h = derive_role_health("engineer", role_state, orze_dir,
                           role_cfg=role_cfg, role_states={})
    assert h["status"] == "LOCKED_OUT", h


def test_cooldown_with_one_error_is_locked_out(tmp_path):
    """High cooldown + at least one error: real fault, LOCKED_OUT."""
    orze_dir = tmp_path / ".orze"
    (orze_dir / "logs" / "engineer").mkdir(parents=True)
    role_state = {
        "last_run_time": 0,
        "consecutive_failures": 1,
        "cooldown_override": 7200,
    }
    role_cfg = {"mode": "claude"}
    h = derive_role_health("engineer", role_state, orze_dir,
                           role_cfg=role_cfg, role_states={})
    assert h["status"] == "LOCKED_OUT", h


if __name__ == "__main__":
    import tempfile
    for fn in (test_single_timeout_with_2x_cooldown_is_not_locked_out,
               test_repeated_timeouts_are_locked_out,
               test_high_errors_are_locked_out,
               test_cooldown_with_one_error_is_locked_out):
        with tempfile.TemporaryDirectory() as td:
            fn(Path(td))
            print(f"{fn.__name__} OK")
