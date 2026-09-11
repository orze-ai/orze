"""Bounded optional-hook routing: actual pip dispatch spy, SOP body tripwire.

No pip/CLI/provider is run. The SOP case intentionally uses the preceding
admission harness, not a real registered controller or native closure proof.
"""
import subprocess
from types import SimpleNamespace

import pytest

from orze.engine import orchestrator, phases
from orze import extensions
from test_controller_profile import supported
from test_force_pack_admission import _runner, _ideas


class StopBoundary(BaseException):
    pass


def test_profile_pro_discovery_never_dispatches_raw_package_install(tmp_path, monkeypatch):
    runner = orchestrator.Orze.__new__(orchestrator.Orze)
    runner.cfg = supported(tmp_path)
    runner.results_dir = tmp_path
    runner._controller_session = object()
    original = extensions.importlib.import_module
    def missing_pro(name, *args, **kwargs):
        if name == 'orze_pro':
            raise ImportError('explicit absent optional package')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(extensions.importlib, 'import_module', missing_pro)
    monkeypatch.setattr(extensions, '_auto_install_attempted', False)
    monkeypatch.setattr(extensions, '_find_pro_key', lambda: 'ORZE-PRO-TEST-BOUNDARY')
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd[:5])
        return subprocess.CompletedProcess(cmd, 1)
    monkeypatch.setattr(subprocess, 'run', run)
    def health(*args):
        raise StopBoundary()
    monkeypatch.setattr('orze.engine.health.HealthMonitor', health)
    with pytest.raises(StopBoundary):
        runner._run_leased()
    assert calls == []


def test_profile_training_skips_legacy_sop_executable_discovery(tmp_path, monkeypatch):
    calls = []
    runner = _runner(tmp_path, [])
    runner._controller_session = object()
    runner.cfg['train_script'] = str(tmp_path / 'training.py')
    monkeypatch.setattr(phases, '_controller_admission_ready', lambda owner: True)
    def validate(*args):
        calls.append('legacy SOP executable discovery')
        return True, ''
    monkeypatch.setattr(extensions, 'get_extension',
        lambda name: SimpleNamespace(validate_idea=validate) if name == 'sops' else None)
    def preflight(*args, **kwargs):
        raise StopBoundary()
    monkeypatch.setattr(phases, 'run_artifact_preflight', preflight)
    with pytest.raises(StopBoundary):
        phases.OrzePhaseMixin._launch_training(runner, ['idea-critical'], True, _ideas())
    assert calls == []
