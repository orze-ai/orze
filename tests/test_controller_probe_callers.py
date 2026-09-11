"""Explicit OS-free HOLD propagation through the eleven adapted probe calls."""
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orze.core.gpu_lease import assert_gpu_scope_idle
from orze.engine import controller_probe as probe, gpu_slots, launcher
from orze.hardware import gpu


def test_all_local_probe_call_sites_propagate_hold_without_empty_inventory(tmp_path, monkeypatch):
    error = probe.ControllerProbeHOLD("injected_owned_probe_hold")
    run = Mock(side_effect=error)
    for module in (probe, gpu_slots, launcher):
        monkeypatch.setattr(module, "run_probe", run)
    monkeypatch.setattr(launcher, "_tree_cpu_jiffies", lambda pid: 0)
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/test-only/" + name)
    task = SimpleNamespace(process=SimpleNamespace(pid=123), gpu=0, log_path=tmp_path / "absent")
    calls = [
        lambda: assert_gpu_scope_idle([0]),
        lambda: gpu.get_gpu_memory_used(0),
        lambda: gpu._eval_already_running("test-task"),
        lambda: gpu.detect_all_gpus([0]),
        lambda: gpu._query_gpu_details([0]),
        lambda: gpu_slots._query_all_gpu_usage([0]),
        lambda: gpu_slots.poll_fleet(["localhost"]),
        lambda: launcher._probe_kernel_boundary(deny_network=True),
        lambda: launcher._detect_zombie(task),
        lambda: launcher._gpu_util_for_pid(123, 0),
    ]
    for invoke in calls:
        run.reset_mock()
        with pytest.raises(probe.ControllerProbeHOLD) as caught:
            invoke()
        assert caught.value is error
        run.assert_called_once()
    # The second utilization query has a successful first query, not a
    # shortcut that would leave its own catch boundary unexercised.
    run.reset_mock()
    run.side_effect = [subprocess.CompletedProcess([], 0, "123, GPU-test\n"), error]
    with pytest.raises(probe.ControllerProbeHOLD) as caught:
        launcher._gpu_util_for_pid(123, 0)
    assert caught.value is error
    assert run.call_count == 2
