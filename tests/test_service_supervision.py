"""The installed service and watchdog must have exactly one restart owner."""

from types import SimpleNamespace

import pytest

from orze.service import install
from orze.service import watchdog


def _completed(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_systemd_unit_delegates_restarts_to_sentinel_aware_watchdog(
        tmp_path, monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return _completed()

    monkeypatch.setattr(install, "_SYSTEMD_DIR", tmp_path)
    monkeypatch.setattr(install.subprocess, "run", fake_run)
    install._install_systemd({
        "python": "/opt/orze/bin/python",
        "workdir": "/srv/project",
        "log_file": "/srv/project/results/orze.log",
        "config_file": "/srv/project/orze.yaml",
    })

    service = (tmp_path / "orze.service").read_text(encoding="utf-8")
    timer = (tmp_path / "orze-watchdog.timer").read_text(encoding="utf-8")
    assert "Restart=no" in service
    assert "Restart=always" not in service
    assert ("ExecStartPre=/opt/orze/bin/python -m "
            "orze.service.runtime_contract --startup-check") in service
    assert ("UnsetEnvironment=PYTHONPATH PYTHONHOME PYTHONSTARTUP "
            "PYTHONINSPECT LD_PRELOAD LD_LIBRARY_PATH BASH_ENV ENV") in service
    assert "OnUnitActiveSec=300" in timer
    assert any(
        args == ["systemctl", "--user", "enable", "--now",
                 "orze-watchdog.timer"]
        for args, _ in calls
    )


def test_systemd_install_audits_effective_unit_before_enable(
        tmp_path, monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return _completed()

    monkeypatch.setattr(install, "_SYSTEMD_DIR", tmp_path)
    monkeypatch.setattr(install.subprocess, "run", fake_run)
    monkeypatch.setattr(
        "orze.service.runtime_contract.audit_runtime_contract",
        lambda cfg: {
            "startup_allowed": False,
            "errors": ["systemd_pythonpath_override"],
            "active_latches": [],
        },
    )

    with pytest.raises(RuntimeError, match="systemd_pythonpath_override"):
        install._install_systemd({
            "python": "/opt/orze/bin/python",
            "workdir": "/srv/project",
            "log_file": "/srv/project/results/orze.log",
            "config_file": "/srv/project/orze.yaml",
            "runtime_contract_version": 1,
        })

    assert ["systemctl", "--user", "daemon-reload"] in calls
    assert not any("enable" in call for call in calls)


def test_partial_systemd_enable_is_rolled_back(tmp_path, monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args == ["systemctl", "--user", "enable", "--now",
                    "orze-watchdog.timer"]:
            return _completed(returncode=1)
        return _completed()

    monkeypatch.setattr(install, "_SYSTEMD_DIR", tmp_path)
    monkeypatch.setattr(install.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="service command failed"):
        install._install_systemd({
            "python": "/opt/orze/bin/python",
            "workdir": "/srv/project",
            "log_file": "/srv/project/results/orze.log",
            "config_file": "/srv/project/orze.yaml",
        })

    assert ["systemctl", "--user", "disable", "--now",
            "orze.service"] in calls
    assert ["systemctl", "--user", "disable", "--now",
            "orze-watchdog.timer"] in calls


def test_systemd_watchdog_restarts_the_tracked_main_unit(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:4] == ["systemctl", "--user", "show", "orze.service"]:
            return _completed(stdout="4321\n")
        return _completed()

    def detached_launch_is_a_bug(*args, **kwargs):
        raise AssertionError("systemd watchdog must not spawn a detached daemon")

    monkeypatch.setattr(watchdog.subprocess, "run", fake_run)
    monkeypatch.setattr(watchdog.subprocess, "Popen", detached_launch_is_a_bug)
    monkeypatch.setattr(
        "orze.service.runtime_contract.audit_runtime_contract",
        lambda cfg: {"startup_allowed": True, "errors": [],
                     "active_latches": []},
    )

    pid = watchdog._launch_orze({"method": "systemd"})

    assert pid == 4321
    assert calls == [
        ["systemctl", "--user", "reset-failed", "orze.service"],
        ["systemctl", "--user", "start", "orze.service"],
        ["systemctl", "--user", "show", "orze.service",
         "--property=MainPID", "--value"],
    ]


def test_systemd_watchdog_reports_start_failure(monkeypatch):
    def fake_run(args, **kwargs):
        if args[:4] == ["systemctl", "--user", "start", "orze.service"]:
            return _completed(returncode=1, stderr="preflight rejected startup\n")
        return _completed()

    monkeypatch.setattr(watchdog.subprocess, "run", fake_run)
    monkeypatch.setattr(
        "orze.service.runtime_contract.audit_runtime_contract",
        lambda cfg: {"startup_allowed": True, "errors": [],
                     "active_latches": []},
    )

    with pytest.raises(
            watchdog.WatchdogLaunchError, match="systemd_start_failed"):
        watchdog._launch_orze({"method": "systemd"})


def test_disable_latch_prevents_systemd_restart(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    (results / ".orze_disabled").write_text("operator stop\n", encoding="utf-8")
    log_file = tmp_path / "watchdog.log"

    def launch_is_a_bug(_svc_cfg):
        raise AssertionError("disabled service must not be restarted")

    monkeypatch.setattr(watchdog, "_launch_orze", launch_is_a_bug)

    watchdog.check_and_restart({
        "method": "systemd",
        "results_dir": str(results),
        "stall_threshold": 60,
        "log_file": str(log_file),
    })

    text = log_file.read_text(encoding="utf-8")
    assert "Skipping restart: disabled (.orze_disabled exists)" in text


def test_crontab_watchdog_keeps_detached_launch(monkeypatch, tmp_path):
    log_file = tmp_path / "orze.log"
    launched = []

    def fake_popen(args, **kwargs):
        launched.append((args, kwargs))
        return SimpleNamespace(pid=9876)

    monkeypatch.setattr(watchdog.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        "orze.service.runtime_contract.audit_runtime_contract",
        lambda cfg: {"startup_allowed": True, "errors": [],
                     "active_latches": []},
    )

    pid = watchdog._launch_orze({
        "method": "crontab",
        "python": "/opt/orze/bin/python",
        "config_file": "/srv/project/orze.yaml",
        "workdir": "/srv/project",
        "log_file": str(log_file),
    })

    assert pid == 9876
    assert launched[0][0] == [
        "/opt/orze/bin/python", "-m", "orze.cli", "-c",
        "/srv/project/orze.yaml",
    ]
    assert launched[0][1]["start_new_session"] is True
