from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from orze.core.config import _validate_config
from orze.engine.phases import OrzePhaseMixin
from orze.engine.process import process_is_running, run_artifact_preflight


def _config(tmp_path: Path, script: Path, **overrides) -> dict:
    spec = {
        "enabled": True,
        "script": str(script),
        "args": [],
        "timeout": 5,
        "network": "inherit",
        "retry_interval": 300,
    }
    spec.update(overrides)
    return {
        "_project_root": str(tmp_path),
        "results_dir": str(tmp_path / "results"),
        "base_config": str(tmp_path / "base.yaml"),
        "python": sys.executable,
        "train_extra_env": {},
        "artifact_preflight": spec,
    }


def test_preflight_hides_accelerators_and_writes_hashed_receipt(tmp_path):
    capture = tmp_path / "capture.json"
    script = tmp_path / "resolve.py"
    script.write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "keys = ['CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', "
        "'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', "
        "'ORZE_ARTIFACT_NETWORK_POLICY', 'ORZE_IDEA_CONFIG']\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps({k: os.environ[k] "
        "for k in keys}))\n",
        encoding="utf-8",
    )
    (tmp_path / "base.yaml").write_text("model: fixture\n", encoding="utf-8")
    cfg = _config(tmp_path, script)
    cfg["train_extra_env"] = {"CAPTURE": str(capture)}

    assert run_artifact_preflight("idea-1", tmp_path / "results", cfg)

    observed = json.loads(capture.read_text(encoding="utf-8"))
    assert observed["CUDA_VISIBLE_DEVICES"] == ""
    assert observed["NVIDIA_VISIBLE_DEVICES"] == "none"
    assert observed["HIP_VISIBLE_DEVICES"] == ""
    assert observed["ROCR_VISIBLE_DEVICES"] == ""
    assert observed["ORZE_ARTIFACT_NETWORK_POLICY"] == "inherit"
    receipt = json.loads(
        (tmp_path / "results" / "idea-1" / "artifact_preflight.json")
        .read_text(encoding="utf-8")
    )
    assert receipt["status"] == "passed"
    assert receipt["gpu_visibility"] == "hidden"
    assert len(receipt["script_sha256"]) == 64
    assert len(receipt["config_sha256"]) == 64
    assert receipt["finished_at"] >= receipt["started_at"]


def test_required_network_rejects_offline_environment_without_running(
        tmp_path, monkeypatch):
    marker = tmp_path / "ran"
    script = tmp_path / "resolve.py"
    script.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cfg = _config(tmp_path, script, network="required")

    assert not run_artifact_preflight("idea-2", tmp_path / "results", cfg)
    assert not marker.exists()
    receipt = json.loads(
        (tmp_path / "results" / "idea-2" / "artifact_preflight.json")
        .read_text(encoding="utf-8")
    )
    assert receipt["status"] == "configuration_error"
    assert receipt["conflicting_env_keys"] == ["HF_HUB_OFFLINE"]


def test_failure_receipt_does_not_persist_resolver_output(tmp_path):
    secret = "private-evaluation-transcript"
    script = tmp_path / "resolve.py"
    script.write_text(
        "import os, sys\nprint(os.environ['PREFLIGHT_SECRET'], file=sys.stderr)\n"
        "raise SystemExit(7)\n",
        encoding="utf-8",
    )
    cfg = _config(tmp_path, script)
    cfg["train_extra_env"] = {"PREFLIGHT_SECRET": secret}

    assert not run_artifact_preflight("idea-3", tmp_path / "results", cfg)
    receipt_text = (
        tmp_path / "results" / "idea-3" / "artifact_preflight.json"
    ).read_text(encoding="utf-8")
    receipt = json.loads(receipt_text)
    assert receipt["status"] == "failed"
    assert receipt["exit_code"] == 7
    assert secret not in receipt_text
    assert "stderr_sha256" in receipt


def test_timeout_kills_descendant_that_ignores_sigterm(tmp_path):
    child_pid_file = tmp_path / "child.pid"
    script = tmp_path / "resolve.py"
    script.write_text(
        "import os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(60)'])\n"
        "Path(os.environ['CHILD_PID_FILE']).write_text(str(child.pid))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    cfg = _config(tmp_path, script, timeout=0.25)
    cfg["train_extra_env"] = {"CHILD_PID_FILE": str(child_pid_file)}

    assert not run_artifact_preflight("idea-4", tmp_path / "results", cfg)
    deadline = time.time() + 3
    while not child_pid_file.exists() and time.time() < deadline:
        time.sleep(0.01)
    child_pid = int(child_pid_file.read_text(encoding="utf-8"))
    assert not process_is_running(child_pid)
    receipt = json.loads(
        (tmp_path / "results" / "idea-4" / "artifact_preflight.json")
        .read_text(encoding="utf-8")
    )
    assert receipt["status"] == "timed_out"


def test_config_validation_catches_preflight_contract_errors(tmp_path):
    cfg = _config(tmp_path, tmp_path / "missing.py", network="required")
    cfg["train_extra_env"] = {"HF_DATASETS_OFFLINE": "true"}
    cfg["artifact_preflight"]["args"] = "--not-a-list"
    cfg["artifact_preflight"]["timeout"] = True

    errors, _ = _validate_config(cfg)

    assert any("script not found" in error for error in errors)
    assert "artifact_preflight.args: must be a list" in errors
    assert any("timeout" in error for error in errors)
    assert any("HF_DATASETS_OFFLINE" in error for error in errors)


def test_dispatch_failure_backs_off_before_launch(tmp_path, monkeypatch):
    results = tmp_path / "results"
    script = tmp_path / "resolve.py"
    script.write_text("raise SystemExit(1)\n", encoding="utf-8")
    cfg = _config(tmp_path, script, retry_interval=60)
    cfg.update({
        "results_dir": str(results),
        "sealed_files": [],
        "gpu_mem_threshold": 2000,
        "sweep": {},
        "gc": {},
    })
    launched = []
    monkeypatch.setattr("orze.engine.phases.get_gpu_memory_used", lambda _gpu: 0)
    monkeypatch.setattr("orze.engine.phases.run_pre_script", lambda *_args: True)
    monkeypatch.setattr(
        "orze.engine.phases.launch",
        lambda *_args, **_kwargs: launched.append(True),
    )
    runner = SimpleNamespace(
        cfg=cfg,
        results_dir=results,
        active_evals={},
        active={},
        gpu_ids=[4],
        lake=None,
        failure_counts={},
        fix_counts={},
    )
    ideas = {
        "idea-fail": {"title": "fixture", "priority": "high", "config": {}},
        "idea-wait": {"title": "fixture 2", "priority": "high", "config": {}},
    }

    OrzePhaseMixin._launch_training(runner, ["idea-fail"], True, ideas)

    assert launched == []
    assert runner.failure_counts == {"idea-fail": 1}
    assert runner._artifact_preflight_blocked_until > time.time()
    metrics = json.loads(
        (results / "idea-fail" / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["status"] == "FAILED"
    waiting = ["idea-wait"]
    OrzePhaseMixin._launch_training(runner, waiting, True, ideas)
    assert waiting == ["idea-wait"]
    assert not (results / "idea-wait").exists()
    assert launched == []
