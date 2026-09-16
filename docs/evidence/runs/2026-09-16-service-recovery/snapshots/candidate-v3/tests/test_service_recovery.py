"""Explicit fresh host after complete old CPU closure; no existing services."""
import json
import select
import subprocess
import sys

import pytest
import yaml

from test_cpu_controller_stop_product import cpu_controller, rows
from test_service_host_product import _hosted, idea


@pytest.fixture
def recoverable(cpu_controller):
    yield from _hosted(cpu_controller)


@pytest.fixture
def naturally_closed(cpu_controller):
    cpu_controller[2]['action_policy']['idle'] = 'stop'
    yield from _hosted(cpu_controller)


def prepare(root, env, source, target, key):
    result = subprocess.run([sys.executable, '-m', 'orze.service.recovery',
        '--source-service-config', str(source), '--service-config', str(target), '--request-id', key],
        cwd=root, env=env, capture_output=True, text=True, timeout=30)
    (root / (key + '.prepare.stdout')).write_text(result.stdout)
    (root / (key + '.prepare.stderr')).write_text(result.stderr)
    return result


@pytest.mark.parametrize('unbounded', [False, True], ids=['finite-budget', 'no-total-deadline'])
def test_closed_host_recovers_into_owned_successor_without_resetting_budget(recoverable, unbounded):
    root, config, cfg, env, service, start, client, wait, prior, captures, births, before_birth = recoverable
    if unbounded:
        cfg['execution'].update(version=2, wall_budget_seconds=None)
        config.write_text(yaml.safe_dump(cfg))
    idea(root, 'idea-0001', 'print("first incarnation")')
    first = start()
    wait(lambda: rows(root, 'SELECT COUNT(*) FROM execution_attempts WHERE state="TERMINAL"') == [(1,)])
    assert client('stop')[0] == 0 and first.wait(timeout=5) == 0
    prior.extend(captures)
    original = {p.name: p.read_bytes() for p in service.with_name(service.name + '.host.lock').iterdir() if p.is_file()}
    old_scope = rows(root, 'SELECT * FROM cpu_action_scopes')
    target = root / 'recovered.json'
    result = prepare(root, env, service, target, 'recover-001')
    assert result.returncode == 0, result.stderr
    assert len(births) == 1
    idea(root, 'idea-0002', 'print("second incarnation")')
    second = start(service_config=target)
    wait(lambda: rows(root, 'SELECT COUNT(*) FROM execution_attempts WHERE state="TERMINAL"') == [(2,)])
    wait(lambda: rows(root, 'SELECT COUNT(*) FROM cpu_action_reservations WHERE state="SETTLED"') == [(2,)])
    assert second.pid != first.pid and len(births) == 2 and births[-1]['old_exited_before_birth']
    assert rows(root, 'SELECT generation FROM controller_instances ORDER BY generation') == [(0,), (1,)]
    assert rows(root, 'SELECT state FROM controller_handoffs') == [('STARTED',)]
    assert rows(root, 'SELECT state FROM cpu_action_reservations') == [('SETTLED',), ('SETTLED',)]
    assert rows(root, 'SELECT * FROM cpu_action_scopes') == old_scope
    assert {p.name: p.read_bytes() for p in service.with_name(service.name + '.host.lock').iterdir() if p.is_file()} == original
    assert client('stop')[0] == 0 and second.wait(timeout=5) == 0
    assert all(select.select([fd], [], [], 0)[0] for fd in captures)

    # A second cold recovery reads the recovered service's own STARTED lineage.
    prior[:] = captures
    third_service = root / 'recovered-again.json'
    result = prepare(root, env, target, third_service, 'recover-002')
    assert result.returncode == 0, result.stderr
    idea(root, 'idea-0003', 'print("third incarnation")')
    third = start(service_config=third_service)
    if unbounded:
        wait(lambda: rows(root, 'SELECT COUNT(*) FROM cpu_action_reservations WHERE state="SETTLED"') == [(3,)])
    else:
        wait(lambda: captures and select.select([captures[-1]], [], [], 0)[0])
        assert rows(root, 'SELECT COUNT(*) FROM cpu_action_reservations') == [(2,)]
        decision = json.loads(rows(root, 'SELECT record_json FROM cpu_action_decisions ORDER BY rowid DESC LIMIT 1')[0][0])
        assert decision['kind'] == 'Stop' and decision['reason'] == 'wall_envelope_exhausted'
    assert client('stop')[0] == 0 and third.wait(timeout=5) == 0
    assert rows(root, 'SELECT generation FROM controller_instances ORDER BY generation') == [(0,), (1,), (2,)]
    assert [row[:2] for row in rows(root, 'SELECT * FROM cpu_action_scopes')] == [row[:2] for row in old_scope]
    assert len(births) == 3 and all(select.select([fd], [], [], 0)[0] for fd in captures)


@pytest.mark.parametrize('mutation', ['service', 'closure', 'config', 'budget', 'ack', 'latch'])
def test_prepared_recovery_rechecks_source_before_any_new_child(recoverable, mutation):
    import sqlite3
    root, config, cfg, env, service, start, client, wait, prior, captures, births, before_birth = recoverable
    idea(root, 'idea-0001', 'print("closed source")')
    first = start()
    wait(lambda: rows(root, 'SELECT COUNT(*) FROM execution_attempts WHERE state="TERMINAL"') == [(1,)])
    assert client('stop')[0] == 0 and first.wait(timeout=5) == 0
    target = root / 'refused.json'
    result = prepare(root, env, service, target, 'prepare-refusal')
    assert result.returncode == 0, result.stderr
    if mutation == 'service':
        data = json.loads(service.read_bytes());data['stall_threshold'] += 1
        service.write_text(json.dumps(data))
    elif mutation == 'closure':
        path = service.with_name(service.name + '.host.lock') / 'closed.json'
        path.rename(path.with_name('closed.saved.json'))
    elif mutation == 'config':
        config.write_text(config.read_text() + '\n# changed after preparation\n')
    elif mutation == 'budget':
        with sqlite3.connect(root / 'lake.db') as connection:
            connection.execute("UPDATE cpu_action_reservations SET state='BOUND'")
    elif mutation == 'ack':
        with sqlite3.connect(root / 'lake.db') as connection:
            connection.execute("UPDATE controller_sessions SET ack_json='{}'")
    else:
        (root / 'results/.orze_disabled').touch()
    rejected = start(wait_ready=False, service_config=target)
    assert rejected.wait(timeout=10) == 75
    assert len(births) == 1 and rows(root, 'SELECT COUNT(*) FROM controller_instances') == [(1,)]
    assert not target.with_name(target.name + '.host.lock').joinpath('ready.json').exists()


def test_closed_record_does_not_allow_recovery_while_old_host_is_alive(naturally_closed):
    root, config, cfg, env, service, start, client, wait, prior, captures, births, before_birth = naturally_closed
    process = start()
    wait(lambda: captures and select.select([captures[0]], [], [], 0)[0])
    code, output = client('status')
    assert code == 0
    status = json.loads(output)
    assert status['state'] == 'stopped'
    closure = service.with_name(service.name + '.host.lock') / 'closed.json'
    closure.write_text(json.dumps(status['closure']))
    target = root / 'live-refused.json'
    result = prepare(root, env, service, target, 'source-live')
    assert result.returncode == 75 and not target.exists()
    assert process.poll() is None and len(births) == 1
    closure.rename(closure.with_name('closed.simulated.json'))
    assert client('stop')[0] == 0 and process.wait(timeout=5) == 0


def test_recovery_preparation_never_overwrites_existing_configuration(recoverable):
    root, config, cfg, env, service, start, client, wait, prior, captures, births, before_birth = recoverable
    process = start()
    assert client('stop')[0] == 0 and process.wait(timeout=5) == 0
    target = root / 'collision.json'
    target.write_bytes(b'operator-owned configuration\n')
    result = prepare(root, env, service, target, 'collision')
    assert result.returncode == 75 and target.read_bytes() == b'operator-owned configuration\n'
    assert len(births) == 1


def test_competing_prepared_hosts_create_only_one_successor(recoverable):
    root, config, cfg, env, service, start, client, wait, prior, captures, births, before_birth = recoverable
    idea(root, 'idea-0001', 'print("old owner")')
    old = start()
    wait(lambda: rows(root, 'SELECT COUNT(*) FROM execution_attempts WHERE state="TERMINAL"') == [(1,)])
    assert client('stop')[0] == 0 and old.wait(timeout=5) == 0
    prior.extend(captures)
    targets = [root / ('competitor-' + str(i) + '.json') for i in range(2)]
    for i, target in enumerate(targets):
        result = prepare(root, env, service, target, 'competitor-' + str(i))
        assert result.returncode == 0, result.stderr
    # Both actual new hosts finish source validation before either reserves.
    with (root / 'sitecustomize.py').open('a') as stream:
        stream.write('''
if 'orze.service.host' in getattr(sys, 'orig_argv', ()):
    from orze.service.recovery import _RecoveryCoordinator
    import time
    actual_prepare_source = _RecoveryCoordinator.prepare_source
    def prepare_together(self):
        actual_prepare_source(self)
        Path('recovery-ready-' + str(os.getpid())).touch()
        deadline = time.monotonic() + 10
        while len(list(Path('.').glob('recovery-ready-*'))) < 2:
            assert time.monotonic() < deadline
            time.sleep(.01)
    _RecoveryCoordinator.prepare_source = prepare_together
''')
    idea(root, 'idea-0002', 'print("single recovered execution")')
    contenders = [start(wait_ready=False, service_config=target) for target in targets]
    wait(lambda: any(p.poll() == 75 for p in contenders))
    wait(lambda: rows(root, 'SELECT COUNT(*) FROM cpu_action_reservations WHERE state="SETTLED"') == [(2,)])
    assert len(list(root.glob('recovery-ready-*'))) == 2
    assert len(births) == 2 and rows(root, 'SELECT COUNT(*) FROM controller_instances') == [(2,)]
    assert rows(root, 'SELECT state FROM controller_handoffs') == [('STARTED',)]
    winner = next(i for i, process in enumerate(contenders) if process.poll() is None)
    assert client('stop', service_config=targets[winner])[0] == 0
    assert contenders[winner].wait(timeout=5) == 0
    assert all(select.select([fd], [], [], 0)[0] for fd in captures)


@pytest.mark.parametrize('state', ['absent', 'disabled', 'enabled', 'active', 'incomplete'])
def test_source_manager_state_requires_retired_units(monkeypatch, tmp_path, state):
    from orze.engine.controller_control import ControllerHOLD
    from orze.service import recovery, scoped
    from orze.service.host import PROFILE
    svc = {'method': 'systemd', 'service_owner': PROFILE, 'service_config_file': str(tmp_path / 'old.json')}
    names, calls, checked = scoped.unit_names(svc), [], []
    def show(command, **kwargs):
        calls.append(command)
        assert command[:3] == ['systemctl', '--user', 'show'] and command[3] in names
        props = {'Id': command[3], 'LoadState': 'not-found' if state == 'absent' else 'loaded',
                 'ActiveState': 'active' if state == 'active' else 'inactive',
                 'UnitFileState': 'enabled' if state == 'enabled' else 'disabled', 'MainPID': '0', 'ControlGroup': ''}
        if state == 'incomplete':
            props.pop('LoadState')
        return subprocess.CompletedProcess(command, 0, '\n'.join(k + '=' + v for k, v in props.items()), '')
    def require_inactive(value):
        checked.append(value)
        if state == 'active':
            raise RuntimeError('manager still active')
    monkeypatch.setattr(recovery.subprocess, 'run', show)
    monkeypatch.setattr(scoped, '_require_inactive', require_inactive)
    if state in ('absent', 'disabled'):
        recovery._inactive(svc)
        assert len(calls) == 3 and bool(checked) == (state == 'disabled')
    else:
        with pytest.raises((ControllerHOLD, RuntimeError)):
            recovery._inactive(svc)
