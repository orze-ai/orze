"""Force-packed work must use the ordinary audited admission pipeline."""

import json
from types import SimpleNamespace

from orze.engine.phases import OrzePhaseMixin


class BusyExclusiveSlots:
    mode = "exclusive"
    slots_per_gpu = 1

    def __init__(self, events):
        self.events = events

    def free_gpu_ids(self, exclude=None):
        return []

    def free_gpu_ids_force_pack(self, floor, exclude=None):
        self.events.append(("force_candidate", floor))
        return [4]

    def force_assign(self, tp, gpu):
        self.events.append(("force_assign", tp.idea_id, gpu))


def _runner(tmp_path, events):
    results = tmp_path / "results"
    return SimpleNamespace(
        cfg={
            "results_dir": str(results),
            "sealed_files": [],
            "gpu_mem_threshold": 2000,
            "sweep": {},
            "gc": {},
            "artifact_preflight": {
                "enabled": True,
                "retry_interval": 60,
            },
        },
        results_dir=results,
        active_evals={},
        active={},
        gpu_ids=[4],
        lake=None,
        failure_counts={},
        fix_counts={},
        slot_mgr=BusyExclusiveSlots(events),
    )


def _ideas():
    return {
        "idea-critical": {
            "title": "critical fixture",
            "priority": "critical",
            "config": {
                "min_free_vram_mib_for_eval": 12000,
                "learning_rate": 0.001,
            },
        },
    }


def test_force_pack_artifact_failure_cannot_reach_gpu_launch(
        tmp_path, monkeypatch):
    events = []
    runner = _runner(tmp_path, events)
    monkeypatch.setattr(
        "orze.engine.phases.run_artifact_preflight",
        lambda *args: events.append(("preflight", args[0])) or False,
    )
    monkeypatch.setattr(
        "orze.engine.phases.run_pre_script",
        lambda *args: events.append(("pre_script", args[0])) or True,
    )
    monkeypatch.setattr(
        "orze.engine.phases.launch",
        lambda *args, **kwargs: events.append(("launch", args[0])),
    )

    OrzePhaseMixin._launch_training(
        runner, ["idea-critical"], True, _ideas())

    assert ("force_candidate", 12000) in events
    assert ("preflight", "idea-critical") in events
    assert not any(event[0] == "pre_script" for event in events)
    assert not any(event[0] == "launch" for event in events)
    receipt = json.loads(next((
        runner.results_dir / "idea-critical" / "_compute_receipts"
    ).glob("*/terminal.json")).read_text())
    assert receipt["outcome"] == "rejected"
    assert receipt["allocated_gpu_seconds"] == 0.0
    assert receipt["reason_code"] == "artifact_preflight_failed"


def test_force_pack_success_runs_preflight_and_setup_before_assignment(
        tmp_path, monkeypatch):
    events = []
    runner = _runner(tmp_path, events)
    monkeypatch.setattr(
        "orze.engine.phases.run_artifact_preflight",
        lambda *args: events.append(("preflight", args[0])) or True,
    )
    monkeypatch.setattr(
        "orze.engine.phases.run_pre_script",
        lambda *args: events.append(("pre_script", args[0])) or True,
    )

    def fake_launch(idea_id, gpu, *args, **kwargs):
        events.append(("launch", idea_id, gpu))
        return SimpleNamespace(idea_id=idea_id, gpu=gpu)

    monkeypatch.setattr("orze.engine.phases.launch", fake_launch)

    OrzePhaseMixin._launch_training(
        runner, ["idea-critical"], True, _ideas())

    ordered = [event[0] for event in events]
    assert ordered.index("preflight") < ordered.index("pre_script")
    assert ordered.index("pre_script") < ordered.index("launch")
    assert ordered.index("launch") < ordered.index("force_assign")
