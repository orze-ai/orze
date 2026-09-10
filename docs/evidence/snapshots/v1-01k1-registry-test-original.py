"""Actual packaged runner/plugin/procedure registry integration, without execution."""

from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from orze.fsm import engine, runner


def test_packaged_runner_steps_real_verifier_in_the_plugin_registry(tmp_path, monkeypatch):
    pro = pytest.importorskip("orze_pro")
    pro_root = Path(pro.__file__).parent
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine, "_guards", {})
    monkeypatch.setattr(engine, "_actions", {})
    monkeypatch.setattr(engine, "_activity_log_path", None)

    def forbidden(*args, **kwargs):
        pytest.fail("empty verifier tick must not execute a subprocess/provider")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    with patch("orze_pro._gate.require_license"):
        runner._load_plugins([pro_root / "fsm" / "plugins"])
        fsms = runner._load_procedures([pro_root / "procedures"], results)
    verifier = [fsm for fsm in fsms if fsm.name == "idea_verifier"]
    assert len(verifier) == 1
    # No cfg, no database and no queued ideas. The real procedure must be
    # evaluable and inert, not fail because plugins registered in another module.
    assert verifier[0].step() is None
    assert verifier[0].status()["state"] == "watching"
    assert not (tmp_path / "idea_lake.db").exists()
    assert not list(results.glob("idea-*"))
