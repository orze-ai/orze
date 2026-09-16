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
    assert second.pid != first.pid and len(births) == 2 and births[-1]['old_exited_before_birth']
    assert rows(root, 'SELECT generation FROM controller_instances ORDER BY generation') == [(0,), (1,)]
    assert rows(root, 'SELECT state FROM controller_handoffs') == [('STARTED',)]
    assert rows(root, 'SELECT state FROM cpu_action_reservations') == [('SETTLED',), ('SETTLED',)]
    assert rows(root, 'SELECT * FROM cpu_action_scopes') == old_scope
    assert {p.name: p.read_bytes() for p in service.with_name(service.name + '.host.lock').iterdir() if p.is_file()} == original
    assert client('stop')[0] == 0 and second.wait(timeout=5) == 0
    assert all(select.select([fd], [], [], 0)[0] for fd in captures)
