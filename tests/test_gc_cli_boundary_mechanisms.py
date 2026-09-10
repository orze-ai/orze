"""GC CLI refusal mechanisms, with no real destructive consumer execution."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from orze.agents import orze_gc


def _setup(tmp_path, monkeypatch, config):
    path = tmp_path / "orze.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["orze-gc", "-c", str(path)])
    calls = []
    monkeypatch.setattr(orze_gc, "run_gc", lambda **kw: calls.append(kw) or {})
    return path, calls


def test_unrepresentable_disk_threshold_is_a_clean_preconsumer_rejection(tmp_path, monkeypatch):
    _, calls = _setup(tmp_path, monkeypatch, {"gc": {"min_free_gb": 10 ** 400}})
    with pytest.raises(SystemExit) as error:
        orze_gc.main()
    assert error.value.code == 2
    assert calls == []


def test_selected_configuration_read_is_bounded(tmp_path, monkeypatch):
    config, calls = _setup(tmp_path, monkeypatch, {})
    config.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(SystemExit) as error:
        orze_gc.main()
    assert error.value.code == 2
    assert calls == []


def test_consumer_blocked_status_reaches_cli_exit_contract(tmp_path, monkeypatch, capsys):
    _setup(tmp_path, monkeypatch, {})
    result = {"blocked": True, "reason": "synthetic_scope_refusal"}
    monkeypatch.setattr(orze_gc, "run_gc", lambda **kw: result)
    assert orze_gc.main() == 2
    assert json.loads(capsys.readouterr().out) == result


def test_actual_module_bad_config_exits_without_echoing_configuration_content(tmp_path):
    config = tmp_path / "bad.yaml"
    marker = "SYNTHETIC_NOT_A_REAL_CREDENTIAL"
    config.write_text("[" + marker, encoding="utf-8")
    source = Path(__file__).resolve().parents[1] / "src"
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1",
               CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        [sys.executable, "-m", "orze.agents.orze_gc", "-c", str(config), "--gc-results"],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=15,
    )
    assert result.returncode == 2
    assert "gc_configuration_unavailable_or_invalid" in result.stderr
    assert marker not in result.stderr + result.stdout
    assert list(tmp_path.iterdir()) == [config]
