"""Fail-closed training data and network boundary tests."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import orze.data_boundaries.wrap as boundary_wrap
from orze.data_boundaries import audit_training_access_log
from orze.core.config import _validate_config
from orze.engine.launcher import (
    LaunchIntegrityError,
    _apply_data_boundary_env,
    _build_isolated_cmd,
    _probe_kernel_boundary,
    _validated_data_boundary_policy,
    launch,
)


@pytest.fixture
def launch_case(tmp_path):
    results = tmp_path / "results"
    idea_dir = results / "idea-boundary"
    idea_dir.mkdir(parents=True)
    train = tmp_path / "train.py"
    train.write_text("# trainer\n", encoding="utf-8")
    base = tmp_path / "base.yaml"
    base.write_text("{}\n", encoding="utf-8")
    ideas = tmp_path / "ideas.md"
    ideas.write_text("", encoding="utf-8")
    return results, idea_dir, {
        "train_script": str(train),
        "base_config": str(base),
        "ideas_file": str(ideas),
        "python": "python3",
    }


@pytest.mark.parametrize(
    "boundaries, expected",
    [
        ("unsafe", "data_boundaries: must be a mapping"),
        ({"forbidden_in_training": "test"},
         "data_boundaries.forbidden_in_training"),
        ({"watch_paths": ["relative"]},
         "data_boundaries.watch_paths: every path must be absolute"),
        ({"watch_paths": ["/tmp/ambiguous:path"]},
         "data_boundaries.watch_paths: paths cannot contain"),
        ({"training_network": "allow"},
         "data_boundaries.training_network"),
    ],
)
def test_data_boundary_config_validation(boundaries, expected):
    errors, _ = _validate_config({"data_boundaries": boundaries})
    assert any(expected in error for error in errors)


def test_training_access_log_audit_is_content_safe_and_fail_closed(tmp_path):
    idea_dir = tmp_path / "idea-boundary"
    idea_dir.mkdir()
    assert audit_training_access_log(idea_dir) == {
        "status": "CLEAN",
        "log_present": False,
        "entries": 0,
        "watch_entries": 0,
        "forbidden_entries": 0,
    }

    access_log = idea_dir / "_access_log.tsv"
    access_log.write_text(
        "WATCH\t/private/eval\t/private/eval/one.arrow\n"
        "FORBIDDEN\t/private/test\t/private/test/two.arrow\n",
        encoding="utf-8",
    )
    assert audit_training_access_log(idea_dir) == {
        "status": "TAINTED",
        "log_present": True,
        "entries": 2,
        "watch_entries": 1,
        "forbidden_entries": 1,
    }

    access_log.write_text("WATCH\t/private/eval\t/outside/one.arrow\n")
    assert audit_training_access_log(idea_dir) == {
        "status": "UNVERIFIED",
        "reason": "access_log_invalid",
    }


@pytest.mark.parametrize("kind", ["missing", "redirected"])
def test_hard_boundary_rejects_unstable_target_before_launch(tmp_path, kind):
    target = tmp_path / "held-out"
    if kind == "redirected":
        real = tmp_path / "real-held-out"
        real.mkdir()
        target.symlink_to(real, target_is_directory=True)

    with pytest.raises(
            LaunchIntegrityError,
            match=("unavailable" if kind == "missing" else "redirected")):
        _validated_data_boundary_policy({
            "forbidden_in_training": [str(target)],
        })


def test_boundary_environment_cannot_spoof_kernel_activation(tmp_path):
    held_out = tmp_path / "held-out"
    held_out.mkdir()
    env = {
        "ORZE_KERNEL_BOUNDARY_ACTIVE": "1",
        "ORZE_BOUNDARY_ATTEST_FD": "99",
        "ORZE_BOUNDARY_ATTEST_NONCE": "spoofed",
        "ORZE_FORBIDDEN_PATHS": "/wrong",
        "ORZE_TRAINING_NETWORK": "inherit",
    }

    _apply_data_boundary_env(
        env,
        {
            "forbidden_in_training": [str(held_out)],
            "training_network": "deny",
        },
        tmp_path / "idea",
    )

    assert env["ORZE_FORBIDDEN_PATHS"] == str(held_out)
    assert env["ORZE_REQUIRE_KERNEL_BOUNDARY"] == "1"
    assert env["ORZE_TRAINING_NETWORK"] == "deny"
    assert "ORZE_KERNEL_BOUNDARY_ACTIVE" not in env
    assert "ORZE_BOUNDARY_ATTEST_FD" not in env
    assert "ORZE_BOUNDARY_ATTEST_NONCE" not in env


def test_isolated_command_has_no_fail_open_mount_and_denies_network(
        tmp_path, monkeypatch):
    held_out = tmp_path / "held-out"
    held_out.mkdir()
    monkeypatch.setattr(
        "orze.engine.launcher.shutil.which", lambda name: f"/usr/bin/{name}")

    command = _build_isolated_cmd(
        ["python3", "train.py"], [str(held_out)], deny_network=True)
    script = command[-1]

    assert command[:5] == [
        "/usr/bin/unshare", "-U", "--map-root-user", "-m", "-n",
    ]
    assert "mount --make-rprivate /" in script
    assert "mount -t tmpfs" in script
    assert "|| true" not in script
    assert script.index("ORZE_KERNEL_BOUNDARY_ACTIVE=1") < script.index(
        "exec python3 train.py")


def test_kernel_probe_is_zero_gpu_and_checks_requested_namespaces(
        monkeypatch):
    observed = {}
    monkeypatch.setattr(
        "orze.engine.launcher.shutil.which", lambda name: f"/usr/bin/{name}")

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("orze.engine.launcher.subprocess.run", fake_run)

    _probe_kernel_boundary(deny_network=True)

    assert "-n" in observed["command"]
    assert any("mount -t tmpfs" in part for part in observed["command"])
    for key in (
        "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
    ):
        assert observed["kwargs"]["env"][key] == ""
    assert observed["kwargs"]["stdout"] is not None
    assert observed["kwargs"]["stderr"] is not None


def test_kernel_probe_failure_blocks_before_gpu_telemetry(
        launch_case, tmp_path, monkeypatch):
    results, _, cfg = launch_case
    held_out = tmp_path / "held-out"
    held_out.mkdir()
    cfg["data_boundaries"] = {
        "forbidden_in_training": [str(held_out)],
        "training_network": "deny",
    }
    gpu_checks = []
    monkeypatch.setattr(
        "orze.engine.launcher.verify_artifact_preflight_receipt",
        lambda *args: True,
    )
    monkeypatch.setattr(
        "orze.engine.launcher._probe_kernel_boundary",
        lambda **kwargs: (_ for _ in ()).throw(
            LaunchIntegrityError("data_boundary_kernel_probe_failed")),
    )
    monkeypatch.setattr(
        "orze.engine.launcher._verify_gpu_free",
        lambda *args: gpu_checks.append(True),
    )

    with pytest.raises(LaunchIntegrityError, match="kernel_probe_failed"):
        launch("idea-boundary", 4, results, cfg)

    assert gpu_checks == []


def test_real_kernel_boundary_when_host_supports_it(tmp_path):
    held_out = tmp_path / "held-out"
    (held_out / "must-not-be-visible").mkdir(parents=True)
    try:
        _probe_kernel_boundary(deny_network=True)
    except LaunchIntegrityError:
        pytest.skip("host does not provide unprivileged mount/network namespaces")
    check = (
        "import os,socket,sys; "
        "ok=(os.listdir(sys.argv[1])==[] and "
        "os.environ.get('ORZE_KERNEL_BOUNDARY_ACTIVE')=='1' and "
        "all(name=='lo' for _,name in socket.if_nameindex()) and "
        "os.environ.get('CUDA_VISIBLE_DEVICES','')==''); "
        "sys.exit(0 if ok else 1)"
    )
    command = _build_isolated_cmd(
        [sys.executable, "-c", check, str(held_out)],
        [str(held_out)],
        deny_network=True,
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["NVIDIA_VISIBLE_DEVICES"] = ""

    completed = subprocess.run(
        command,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0


def test_watch_only_mode_remains_explicitly_audit_only(
        launch_case, tmp_path, monkeypatch):
    results, _, cfg = launch_case
    watched = tmp_path / "watched"
    watched.mkdir()
    cfg["data_boundaries"] = {
        "watch_paths": [str(watched)],
        "training_network": "inherit",
    }
    observed = {}

    class RunningProcess:
        pid = os.getpid()

        def poll(self):
            return None

    monkeypatch.setattr(
        "orze.engine.launcher.verify_artifact_preflight_receipt",
        lambda *args: True,
    )
    monkeypatch.setattr(
        "orze.engine.launcher._probe_kernel_boundary",
        lambda **kwargs: pytest.fail("audit-only mode must not claim kernel isolation"),
    )
    monkeypatch.setattr("orze.engine.launcher._verify_gpu_free", lambda *args: None)

    def fake_popen(command, **kwargs):
        observed["command"] = command
        observed["env"] = kwargs["env"]
        return RunningProcess()

    monkeypatch.setattr("orze.engine.launcher.subprocess.Popen", fake_popen)

    process = launch("idea-boundary", 4, results, cfg)

    assert process.gpu == 4
    assert observed["command"][1:3] == ["-m", "orze.data_boundaries.wrap"]
    assert observed["env"]["ORZE_WATCH_PATHS"] == str(watched)
    assert "ORZE_REQUIRE_KERNEL_BOUNDARY" not in observed["env"]
    process.close_log()


def test_wrapper_refuses_required_but_inactive_kernel_boundary(monkeypatch):
    executed = []
    monkeypatch.setenv("ORZE_REQUIRE_KERNEL_BOUNDARY", "1")
    monkeypatch.delenv("ORZE_KERNEL_BOUNDARY_ACTIVE", raising=False)
    monkeypatch.setattr(boundary_wrap.runpy, "run_path", executed.append)
    monkeypatch.setattr(
        boundary_wrap.sys, "argv", ["wrap", "train.py"])

    with pytest.raises(SystemExit) as exc:
        boundary_wrap.main()

    assert exc.value.code == 126
    assert executed == []
