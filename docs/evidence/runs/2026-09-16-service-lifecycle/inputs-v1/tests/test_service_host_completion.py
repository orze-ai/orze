"""Natural CPU completion under the actual persistent service owner."""
import json
import select

import pytest
import yaml

from test_cpu_controller_stop_product import cpu_controller, rows
from test_service_host_product import _hosted, idea


@pytest.fixture
def completed_host(cpu_controller):
    root, config, cfg, env, launch, wait = cpu_controller
    cfg['action_policy']['idle'] = 'stop'
    config.write_text(yaml.safe_dump(cfg))
    yield from _hosted(cpu_controller)


@pytest.mark.parametrize('action', [False, True])
def test_natural_completion_remains_observable_and_can_close_host(completed_host, action):
    root, config, cfg, env, service, start, client, wait, prior, captures, births, before_birth = completed_host
    if action:
        idea(root, 'idea-0001', f'from pathlib import Path;Path({str(root / "done")!r}).write_text("done")')
    process = start()
    wait(lambda: captures and all(select.select([fd], [], [], 0)[0] for fd in captures))
    assert process.poll() is None and len(births) == 1
    code, output = client('status')
    assert code == 0, output
    status = json.loads(output)
    assert status['state'] == 'stopped' and status['live_children'] == 0
    assert status['closure']['observed_process'] == births[0]['process']
    assert json.loads(rows(root, 'SELECT request_json FROM controller_sessions')[0][0])['reason'] == 'normal_exit'
    assert rows(root, 'SELECT state FROM cpu_action_reservations') == ([('SETTLED',)] if action else [])
    if action:
        assert (root / 'done').read_text() == 'done'
    assert client('watchdog', watchdog=True)[0] == 0
    assert len(births) == 1 and process.poll() is None
    assert client('stop')[0] == 0
    assert process.wait(timeout=5) == 0
    assert json.loads((root / 'service.json.host.lock/closed.json').read_text()) == status['closure']


def test_initial_completion_before_first_host_observation_is_supported(completed_host):
    root, config, cfg, env, service, start, client, wait, prior, captures, births, before_birth = completed_host
    with (root / 'sitecustomize.py').open('a') as stream:
        stream.write('''
if 'orze.service.host' in getattr(sys, 'orig_argv', ()):
    from orze.service.host import Host
    import time
    original_observe = Host._observe
    def observe_after_child_exit(self):
        deadline = time.monotonic() + 10
        while self.children.live() and time.monotonic() < deadline:
            time.sleep(.01)
        return original_observe(self)
    Host._observe = observe_after_child_exit
''')
    process = start()
    code, output = client('status')
    assert code == 0, output
    assert json.loads(output)['state'] == 'stopped'
    assert len(births) == 1 and process.poll() is None
    assert client('stop')[0] == 0 and process.wait(timeout=5) == 0
