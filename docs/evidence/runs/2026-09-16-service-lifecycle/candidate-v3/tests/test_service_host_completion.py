"""Natural CPU completion under the actual persistent service owner."""
import json
import select
import sqlite3
import subprocess
import sys

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
    import subprocess
    from orze.engine.controller_session import _Observer
    spawned = []
    original_spawn, original_observe = subprocess.Popen, _Observer.__init__
    def capture_spawn(*args, **kwargs):
        child = original_spawn(*args, **kwargs)
        spawned.append(child)
        return child
    def observe_after_child_exit(self, *args, **kwargs):
        for child in spawned:
            assert child.wait(timeout=10) == 0
        Path('observed-after-exit').touch()
        return original_observe(self, *args, **kwargs)
    subprocess.Popen, _Observer.__init__ = capture_spawn, observe_after_child_exit
''')
    process = start()
    code, output = client('status')
    assert code == 0, output
    assert json.loads(output)['state'] == 'stopped'
    assert (root / 'observed-after-exit').exists()
    assert len(births) == 1 and process.poll() is None
    assert client('stop')[0] == 0 and process.wait(timeout=5) == 0


@pytest.mark.parametrize('lost', [False, True])
def test_successor_completion_before_started_reply_remains_replayable(completed_host, lost):
    root, config, cfg, env, service, start, client, wait, prior, captures, births, before_birth = completed_host
    with (root / 'sitecustomize.py').open('a') as stream:
        stream.write('''
if 'orze.service.host' in getattr(sys, 'orig_argv', ()):
    import subprocess
    from orze.engine import controller_handoff
    captured = []
    original_spawn, original_receive = subprocess.Popen, controller_handoff._receive
    def capture_spawn(*args, **kwargs):
        child = original_spawn(*args, **kwargs)
        captured.append(child)
        return child
    def receive_after_exit(channel, peer_pid, deadline):
        packet = original_receive(channel, peer_pid, deadline)
        if packet.get('event') == 'STARTED':
            matches = [child for child in captured if child.pid == peer_pid]
            assert len(matches) == 1 and matches[0].wait(timeout=10) == 0
            Path('started-after-exit').touch()
            marker = Path('lose-completed-started')
            if marker.exists():
                marker.rename('lost-completed-started')
                raise OSError('fixture: reply lost after completed successor')
        return packet
    subprocess.Popen, controller_handoff._receive = capture_spawn, receive_after_exit
''')
    marker = root / 'worker-started'
    idea(root, 'idea-0001', f'import time;from pathlib import Path;Path({str(marker)!r}).touch();time.sleep(120)')
    process = start()
    wait(marker.exists)
    source = rows(root, 'SELECT current_id FROM controller_scope_heads')[0][0]
    prior.append(captures[0])
    if lost:
        (root / 'lose-completed-started').touch()
    code, output = client('restart', 'completed-successor', source)
    assert (root / 'started-after-exit').exists()
    assert code == (75 if lost else 0), output
    if lost:
        assert (root / 'lost-completed-started').exists()
        target = rows(root, 'SELECT current_id FROM controller_scope_heads')[0][0]
        assert client('restart', 'another-key', target)[0] == 75
        assert client('watchdog', watchdog=True)[0] == 0
    assert client('restart', 'completed-successor', source)[0] == 0
    code, output = client('status')
    assert code == 0 and json.loads(output)['state'] == 'stopped'
    assert client('watchdog', watchdog=True)[0] == 0
    assert len(births) == 2 and all(select.select([fd], [], [], 0)[0] for fd in captures)
    assert rows(root, 'SELECT state FROM controller_handoffs') == [('STARTED',)]
    assert rows(root, 'SELECT state FROM cpu_action_reservations') == [('SETTLED',)]
    assert client('stop')[0] == 0 and process.wait(timeout=5) == 0


@pytest.mark.parametrize('mutation', ['ack_missing', 'ack_hash', 'orphan_budget', 'foreign_process'])
def test_natural_exit_needs_original_child_and_complete_closure(completed_host, mutation):
    root, config, cfg, env, service, start, client, wait, prior, captures, births, before_birth = completed_host
    process = start()
    wait(lambda: captures and all(select.select([fd], [], [], 0)[0] for fd in captures))
    unrelated = None
    try:
        with sqlite3.connect(root / 'lake.db') as conn:
            if mutation == 'ack_missing':
                conn.execute('UPDATE controller_sessions SET ack_json=NULL')
            elif mutation == 'ack_hash':
                ack = json.loads(conn.execute('SELECT ack_json FROM controller_sessions').fetchone()[0])
                ack['request_sha256'] = '0' * 64
                conn.execute('UPDATE controller_sessions SET ack_json=?', (json.dumps(ack, sort_keys=True, separators=(',', ':')),))
            elif mutation == 'orphan_budget':
                from orze.core import cpu_action_budget
                from orze.idea_lake import IdeaLake
                lake = IdeaLake(str(root / 'lake.db'))
                try:
                    scope = json.loads(conn.execute('SELECT binding_json FROM cpu_action_scopes').fetchone()[0])
                    assert cpu_action_budget.reserve(lake, scope, 'unregistered-action', 1) is not None
                finally:
                    lake.close()
            else:
                unrelated = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
                from orze.engine.supervisor_worker import process_identity
                identity = json.loads(conn.execute('SELECT identity_json FROM controller_instances').fetchone()[0])
                identity['process'] = process_identity(unrelated.pid)[0]
                binding = json.loads(conn.execute('SELECT binding_json FROM controller_sessions').fetchone()[0])
                binding['identity'] = identity
                encode = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'))
                conn.execute('UPDATE controller_instances SET identity_json=?', (encode(identity),))
                conn.execute('UPDATE controller_sessions SET binding_json=?', (encode(binding),))
        assert client('status')[0] == 75
        assert client('stop')[0] == 75
        assert not (root / 'service.json.host.lock/closed.json').exists() and len(births) == 1
        if unrelated is not None:
            assert unrelated.poll() is None
    finally:
        if unrelated is not None:
            unrelated.terminate()
            unrelated.wait(timeout=5)
