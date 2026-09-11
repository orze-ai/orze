"""New cross-resource admission mechanisms: actual DB, explicit GPU tripwires."""
from types import SimpleNamespace

import pytest

from orze.engine import launcher, phases
from orze.idea_lake import IdeaLake


@pytest.fixture
def cpu_task(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(tmp_path / "lake.db")
    lake.insert("idea-cpu", "CPU", "kind: native_cpu_action\naction:\n  command: [a,b]\n", "",
                status="queued", kind="native_cpu_action", if_absent=True)
    yield lake, results
    lake.close()


def test_direct_gpu_launcher_refuses_persistent_cpu_kind_before_any_gpu_gate(cpu_task, monkeypatch):
    lake, results = cpu_task
    calls = []
    monkeypatch.setattr(launcher, "_assert_gpu_authorized", lambda *a: calls.append("gpu"))
    monkeypatch.setattr(launcher, "_assert_launch_authorized", lambda *a: calls.append("launch"))
    with pytest.raises(launcher.LaunchIntegrityError, match="requires_cpu_executor"):
        launcher.launch("idea-cpu", 0, results, {}, lake=lake)
    assert calls == []
    assert lake.get_fsm_state("idea-cpu") == "QUEUED"
    assert not (results / "idea-cpu").exists()


def test_gpu_queue_never_expands_cpu_command_array_as_sweep(cpu_task, monkeypatch):
    lake, results = cpu_task
    calls = []
    runner = SimpleNamespace(lake=lake, results_dir=results)
    monkeypatch.setattr(phases, "expand_sweeps", lambda *a, **k: calls.append("sweep"))
    with pytest.raises(launcher.LaunchIntegrityError, match="requires_cpu_executor"):
        phases.OrzePhaseMixin._sync_ideas(runner, {})
    assert calls == []
    assert lake.get_fsm_state("idea-cpu") == "QUEUED"


def test_gpu_phase_rejects_cpu_before_slots_or_claim(cpu_task, monkeypatch):
    lake, results = cpu_task
    calls = []
    runner = SimpleNamespace(lake=lake, results_dir=results, cfg={})
    monkeypatch.setattr(phases, "_controller_admission_ready", lambda *a: calls.append("slots"))
    monkeypatch.setattr(phases, "claim", lambda *a, **k: calls.append("claim"))
    with pytest.raises(launcher.LaunchIntegrityError, match="requires_cpu_executor"):
        phases.OrzePhaseMixin._launch_training(runner, ["idea-cpu"], True, {"idea-cpu": {}})
    assert calls == []
    assert not (results / "idea-cpu").exists()
