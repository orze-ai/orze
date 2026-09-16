"""Explicit fresh host after complete old CPU closure; no existing services."""
import json
import select
import subprocess
import sys

import pytest

from test_cpu_controller_stop_product import cpu_controller, rows
from test_service_host_product import _hosted, idea


@pytest.fixture
def recoverable(cpu_controller):
    yield from _hosted(cpu_controller)


def prepare(root, env, source, target, key):
    result = subprocess.run([sys.executable, '-m', 'orze.service.recovery',
        '--source-service-config', str(source), '--service-config', str(target), '--request-id', key],
        cwd=root, env=env, capture_output=True, text=True, timeout=30)
    (root / (key + '.prepare.stdout')).write_text(result.stdout)
    (root / (key + '.prepare.stderr')).write_text(result.stderr)
    return result


def test_closed_host_recovers_into_owned_successor_without_resetting_budget(recoverable):
    root, config, cfg, env, service, start, client, wait, prior, captures, births, before_birth = recoverable
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
    wait(lambda: rows(root, 'SELECT COUNT(*) FROM cpu_action_reservations WHERE state="SETTLED"') == [(3,)])
    assert client('stop')[0] == 0 and third.wait(timeout=5) == 0
    assert rows(root, 'SELECT generation FROM controller_instances ORDER BY generation') == [(0,), (1,), (2,)]
    assert rows(root, 'SELECT * FROM cpu_action_scopes') == old_scope
    assert len(births) == 3 and all(select.select([fd], [], [], 0)[0] for fd in captures)


@pytest.mark.parametrize('mutation', ['service', 'closure', 'config', 'budget', 'latch'])
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
    else:
        (root / 'results/.orze_disabled').touch()
    rejected = start(wait_ready=False, service_config=target)
    assert rejected.wait(timeout=10) == 75
    assert len(births) == 1 and rows(root, 'SELECT COUNT(*) FROM controller_instances') == [(1,)]
    assert not target.with_name(target.name + '.host.lock').joinpath('ready.json').exists()


def test_closed_record_does_not_allow_recovery_while_old_host_is_alive(recoverable):
    root, config, cfg, env, service, start, client, wait, prior, captures, births, before_birth = recoverable
    process = start()
    # The normal stop command replies before the host's exit. Delay only that
    # exit in an owned helper process would be unnecessary: a live source has
    # no complete host closure and must already refuse preparation.
    target = root / 'live-refused.json'
    result = prepare(root, env, service, target, 'source-live')
    assert result.returncode == 75 and not target.exists()
    assert process.poll() is None and len(births) == 1
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
