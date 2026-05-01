"""Round-2 C1: triggered roles render as IDLE when their gate is closed."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orze.reporting.state import derive_role_health, build_role_health_block  # noqa: E402


def test_triggered_role_idle_when_no_trigger_file(tmp_path):
    orze_dir = tmp_path / ".orze"
    (orze_dir / "logs" / "engineer").mkdir(parents=True)
    (orze_dir / "triggers").mkdir(parents=True)
    role_state = {"last_run_time": 0, "consecutive_failures": 0,
                  "cooldown_override": 0}
    role_cfg = {"mode": "claude", "triggered_by": "professor"}
    h = derive_role_health("engineer", role_state, orze_dir,
                           role_cfg=role_cfg, role_states={})
    assert h["status"] == "IDLE", h


def test_triggered_role_eligible_when_trigger_file_present(tmp_path):
    orze_dir = tmp_path / ".orze"
    (orze_dir / "logs" / "engineer").mkdir(parents=True)
    (orze_dir / "triggers").mkdir(parents=True)
    (orze_dir / "triggers" / "_trigger_engineer").write_text("go")
    role_state = {"last_run_time": 0}
    role_cfg = {"mode": "claude", "triggered_by": "professor"}
    h = derive_role_health("engineer", role_state, orze_dir,
                           role_cfg=role_cfg, role_states={})
    assert h["status"] == "HEALTHY", h


def test_worst_host_e2(tmp_path):
    orze_dir = tmp_path / ".orze"
    (orze_dir / "logs" / "professor").mkdir(parents=True)
    cfg = {"roles": {"professor": {"mode": "claude"}}}
    role_states = {"professor": {"cooldown_override": 7200,
                                 "consecutive_failures": 6}}
    per_host = {
        "host-A": {"professor": {"cooldown_override": 100,
                                  "consecutive_failures": 1}},
        "host-B": {"professor": {"cooldown_override": 7200,
                                  "consecutive_failures": 6}},
    }
    out = build_role_health_block(cfg, role_states, orze_dir,
                                  per_host_role_states=per_host)
    assert out["professor"]["status"] == "LOCKED_OUT"
    assert out["professor"]["worst_host"] == "host-B"


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_triggered_role_idle_when_no_trigger_file(Path(td))
        print("idle_when_no_trigger OK")
    with tempfile.TemporaryDirectory() as td:
        test_triggered_role_eligible_when_trigger_file_present(Path(td))
        print("healthy_when_trigger_present OK")
    with tempfile.TemporaryDirectory() as td:
        test_worst_host_e2(Path(td))
        print("worst_host OK")
