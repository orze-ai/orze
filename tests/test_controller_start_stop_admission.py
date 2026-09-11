"""New request-only controller admission requirements, not old-code reds.

All process/GPU/package boundaries are explicit spies. Tiny temporary marker
files and actual CLI/lifecycle/Orze.run control flow are real; no controller,
provider, GPU, package installer or host process scan is executed.
"""
import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest

import orze.cli as cli
import orze.cli_setup as cli_setup
import orze.lifecycle as lifecycle
from orze.core.control_outcome import (
    ControllerStopHOLD, STOP_SENTINELS, StopOutcome,
)
from orze.engine.orchestrator import Orze


@pytest.fixture
def project(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    cfg = {"results_dir": str(results), "_config_path": str(tmp_path / "orze.yaml"),
           "gpu_scheduling": {"allowed_gpus": [0]}}
    events = []
    monkeypatch.setattr(lifecycle, "_read_pid", lambda *a: None)
    monkeypatch.setattr(lifecycle, "_pgrep", lambda *a: [])
    monkeypatch.setattr(lifecycle.time, "sleep", lambda *a: None)

    def no_launch(*args, **kwargs):
        events.append("popen")
        return SimpleNamespace(pid=850001, poll=lambda: None)

    monkeypatch.setattr(lifecycle.subprocess, "Popen", no_launch)
    monkeypatch.setattr(cli, "load_project_config", lambda *a: dict(cfg))
    monkeypatch.setattr("orze.extensions._find_pro_key", lambda: "fixture-only")
    monkeypatch.setattr("orze.extensions.get_extension", lambda *a: None)
    monkeypatch.setattr(cli, "detect_all_gpus",
                        lambda: pytest.fail("no GPU inventory is permitted"))
    return SimpleNamespace(results=results, cfg=cfg, events=events)


@pytest.mark.parametrize("name", STOP_SENTINELS)
def test_lifecycle_start_preserves_all_stop_markers_before_launch(project, name):
    p = project
    marker = p.results / name
    marker.write_bytes(b"unconfirmed prior state")
    with pytest.raises(ControllerStopHOLD):
        lifecycle.do_start(p.cfg)
    assert marker.read_bytes() == b"unconfirmed prior state"
    assert p.events == []
    assert not (p.results / "orze.log").exists()


@pytest.mark.parametrize("name", STOP_SENTINELS)
def test_direct_cli_refuses_stop_markers_before_controller_construction(project, monkeypatch, name):
    p = project
    marker = p.results / name
    marker.write_bytes(b"unconfirmed prior state")
    class MustNotConstruct:
        def __init__(self, *a, **k):
            pytest.fail("controller construction must not precede stop admission")
    monkeypatch.setattr("orze.engine.orchestrator.Orze", MustNotConstruct)
    monkeypatch.setattr("sys.argv", ["orze", "--once", "--no-admin"])
    assert cli.main() == 75
    assert marker.read_bytes() == b"unconfirmed prior state"
    assert p.events == []


@pytest.mark.parametrize("name", STOP_SENTINELS)
def test_direct_orze_run_preserves_pid_and_never_acquires_resources(project, name):
    p = project
    (p.results / name).write_bytes(b"pending")
    pid = p.results / ".orze.pid"
    pid.write_bytes(b"prior-unverified-controller")
    runner = SimpleNamespace(results_dir=p.results,
        _write_pid_file=lambda: pytest.fail("must not overwrite prior controller PID"))
    with pytest.raises(ControllerStopHOLD):
        Orze.run(runner)
    assert pid.read_bytes() == b"prior-unverified-controller"
    assert p.events == []


@pytest.mark.parametrize("outcome", [None, True, StopOutcome("confirmed", "unverified_label")])
def test_restart_never_consumes_labels_or_legacy_truth_as_proof(project, monkeypatch, outcome):
    p = project
    calls = []
    monkeypatch.setattr(lifecycle, "do_stop", lambda *a, **k: outcome)
    monkeypatch.setattr(lifecycle, "do_start", lambda *a, **k: calls.append("start"))
    result = lifecycle.do_restart(p.cfg)
    assert result.status == "hold"
    assert calls == []


@pytest.mark.parametrize("command", ["stop", "--stop"])
def test_stop_cli_returns_tempfail_for_delivered_request_not_success(project, monkeypatch, command):
    p = project
    monkeypatch.setattr("sys.argv", ["orze", command])
    assert cli.main() == 75
    assert (p.results / ".orze_disabled").exists()
    assert (p.results / ".orze_stop_all").read_text() == "kill"
    assert p.events == []


@pytest.mark.parametrize("command", ["--upgrade", "--reinstall"])
def test_package_control_cli_stops_before_any_installer_or_restart(project, monkeypatch, command):
    p = project
    monkeypatch.setattr("sys.argv", ["orze", command])
    monkeypatch.setattr(cli_setup.subprocess, "run",
                        lambda *a, **k: pytest.fail("no package/process command"))
    assert cli.main() == 75
    assert (p.results / ".orze_disabled").exists()
    assert (p.results / ".orze_stop_all").read_text() == "kill"
    assert p.events == []


def test_marker_free_lifecycle_start_reaches_only_the_controlled_launch_boundary(project):
    p = project
    assert lifecycle.do_start(p.cfg) == 850001
    assert p.events == ["popen"]
    assert all(not (p.results / name).exists() for name in STOP_SENTINELS)


def test_new_stop_marker_at_log_open_prevents_popen(project, monkeypatch):
    p = project
    marker = p.results / ".orze_stop_all"
    real_open = builtins.open
    def open_log(path, *args, **kwargs):
        handle = real_open(path, *args, **kwargs)
        if Path(path) == p.results / "orze.log":
            marker.write_text("kill")
        return handle
    monkeypatch.setattr(builtins, "open", open_log)
    with pytest.raises(ControllerStopHOLD):
        lifecycle.do_start(p.cfg)
    assert marker.read_text() == "kill"
    assert p.events == []
