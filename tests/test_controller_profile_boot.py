"""Actual Orze boot routing with explicit pre-execution boundary faults."""
from pathlib import Path

import pytest

from orze.engine.orchestrator import Orze
from orze.core.control_outcome import ControllerStopHOLD
from test_controller_profile import supported


def test_direct_profile_run_checks_old_stop_marker_before_registration(tmp_path, monkeypatch):
    cfg = supported(tmp_path)
    folder = Path(cfg['results_dir']); folder.mkdir()
    marker = folder / '.orze_disabled'; marker.write_text('retained stop')
    runner = Orze.__new__(Orze)
    runner.cfg, runner.results_dir = cfg, folder
    calls = []
    def entered(*args):
        calls.append('session')
        raise RuntimeError('registration boundary reached')
    monkeypatch.setattr('orze.engine.controller_session.ControllerSession', entered)
    try:
        runner.run()
    except (ControllerStopHOLD, RuntimeError):
        pass
    else:
        pytest.fail('stop marker was not refused')
    assert calls == []
    assert marker.read_text() == 'retained stop'


def test_profile_init_never_copies_a_legacy_catalog_before_registration(tmp_path, monkeypatch):
    cfg = supported(tmp_path)
    old = Path(cfg['ideas_file']).parent / 'idea_lake.db'
    old.parent.mkdir(parents=True)
    old.write_bytes(b'legacy catalog bytes')
    destination = Path(cfg['idea_lake_db'])
    calls = []
    monkeypatch.setattr('orze.engine.orchestrator._validate_config', lambda cfg: ([], []))
    def entered(path):
        calls.append(path)
        raise OSError('catalog construction boundary')
    monkeypatch.setattr('orze.idea_lake.IdeaLake', entered)
    try:
        Orze([2, 4], cfg)
    except BaseException:
        pass
    assert not destination.exists()
    assert old.read_bytes() == b'legacy catalog bytes'
    assert calls == []


def test_profile_lake_failure_propagates_before_reporter_degradation(tmp_path, monkeypatch):
    cfg = supported(tmp_path)
    error = OSError('injected catalog refusal')
    calls = []
    monkeypatch.setattr('orze.engine.orchestrator._validate_config', lambda cfg: ([], []))
    def failed(path):
        raise error
    monkeypatch.setattr('orze.idea_lake.IdeaLake', failed)
    monkeypatch.setattr('orze.engine.orchestrator.NotificationProcessor',
                        lambda *a, **k: calls.append('reporter'))
    observed = None
    try:
        Orze([2, 4], cfg)
    except BaseException as exc:
        observed = exc
    assert observed is error
    assert calls == []
