"""Loaded/constructed profile declarations cannot downgrade to raw startup."""
import atexit
import signal

import pytest

from orze.engine import orchestrator
from orze.core.controller_profile import profile_fingerprint
from orze.engine.controller_control import ControllerHOLD
from test_controller_profile import supported


class BoundaryReached(BaseException):
    pass


@pytest.mark.parametrize('stage', ['loaded', 'constructed'])
def test_erased_enabled_profile_never_enters_unregistered_resources(tmp_path, monkeypatch, stage):
    cfg = supported(tmp_path)
    cfg['_controller_profile_fingerprint'] = profile_fingerprint(cfg, [2, 4])
    monkeypatch.setattr(orchestrator, '_validate_config', lambda cfg: ([], []))
    calls = []
    if stage == 'loaded':
        cfg['controller_control'] = None
        def lake(path):
            calls.append('lake')
            raise BoundaryReached()
        monkeypatch.setattr('orze.idea_lake.IdeaLake', lake)
        try:
            orchestrator.Orze([2, 4], cfg)
        except (ControllerHOLD, BoundaryReached):
            pass
        else:
            pytest.fail('erased loaded declaration was accepted')
    else:
        before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        runner = None
        try:
            runner = orchestrator.Orze([2, 4], cfg)
            cfg['controller_control'] = None
            def lease(gpus):
                calls.append('lease')
                raise BoundaryReached()
            monkeypatch.setattr(orchestrator, 'acquire_gpu_leases', lease)
            try:
                runner.run()
            except (ControllerHOLD, BoundaryReached):
                pass
            else:
                pytest.fail('erased constructed declaration was accepted')
        finally:
            if runner is not None:
                atexit.unregister(runner._atexit_cleanup)
                runner.lake.close()
            for sig, handler in before.items():
                signal.signal(sig, handler)
    assert calls == []
