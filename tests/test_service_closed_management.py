"""Real closed CPU project; manager properties and operations are substitutes."""
import hashlib
import json
from pathlib import Path
import select
import sqlite3

import pytest

from orze.service import host, runtime_contract, scoped, status
from test_cpu_controller_stop_product import cpu_controller
from test_service_host_completion import completed_host
from test_service_host_product import idea
from test_service_scoped import property_map

MANAGER = '''
if 'orze.service.host' in getattr(sys, 'orig_argv', ()):
    from orze.service import runtime_contract, scoped
    import json
    svc = json.loads(Path('service.json').read_text())
    main_unit, timer_unit, watchdog_unit = scoped.unit_names(svc)
    def property_response(unit, **kwargs):
        if unit == timer_unit:
            return {'Id': unit, 'Unit': watchdog_unit, 'ActiveState': 'active', 'UnitFileState': 'enabled'}
        assert unit in (main_unit, watchdog_unit)
        python = svc['python']
        def command(*args):
            return '{ path=' + python + ' ; argv[]=' + ' '.join((python, *args)) + ' ; ignore_errors=no ; }'
        watchdog = unit == watchdog_unit
        return {'Id': unit, 'ExecCondition': '', 'ExecStartPost': '', 'ExecStop': '', 'ExecStopPost': '', 'ExecReload': '',
            'Restart': 'no', 'Type': 'oneshot' if watchdog else 'simple', 'KillMode': 'control-group',
            'MainPID': str(os.getpid()), 'ControlGroup': '/fixture/service-host', 'WorkingDirectory': svc['workdir'],
            'ExecStart': command('-m', 'orze.service.watchdog' if watchdog else 'orze.service.host', '--service-config', svc['service_config_file']),
            'ExecStartPre': '' if watchdog else command('-m', 'orze.service.runtime_contract', '--startup-check', '--service-config', svc['service_config_file']),
            'Environment': '', 'EnvironmentFiles': '', 'PassEnvironment': '',
            'UnsetEnvironment': ' '.join(runtime_contract._RUNTIME_ENVIRONMENT_KEYS),
            'ActiveState': 'active', 'UnitFileState': 'enabled'}
    runtime_contract._systemd_properties = property_response
    runtime_contract.manager_cgroup = lambda pid: '/fixture/service-host'
'''


@pytest.fixture
def closed_project(completed_host, monkeypatch):
    root, config, cfg, env, service, start, client, wait, prior, captures, births, before_birth = completed_host
    svc = json.loads(service.read_text())
    svc['method'] = 'systemd'
    service.write_text(json.dumps(svc, sort_keys=True))
    with (root / 'sitecustomize.py').open('a') as stream:
        stream.write(MANAGER)
    idea(root, 'idea-0001', f'from pathlib import Path;Path({str(root / "done")!r}).write_text("done")')
    process = start()
    wait(lambda: captures and all(select.select([fd], [], [], 0)[0] for fd in captures))
    assert client('stop')[0] == 0 and process.wait(timeout=5) == 0
    assert len(births) == 1 and (root / 'done').read_text() == 'done'
    directory = root / 'units'
    directory.mkdir()
    units = scoped.render_units(svc)
    for name, value in units.items():
        (directory / name).write_text(value)
    other = directory / 'another-project.service'
    other.write_text('independent project')
    effective = property_map(svc)
    for unit, props in effective.items():
        props.update(ActiveState='inactive', UnitFileState='disabled')
        props['_UnitText'] = units[unit]
        if 'MainPID' in props:
            props.update(MainPID='0', ControlGroup='')
    monkeypatch.setattr(runtime_contract, '_systemd_properties', lambda unit, **kwargs: dict(effective[unit]))
    calls = []
    monkeypatch.setattr(scoped, '_checked', lambda *args: calls.append(args))
    yield root, service, svc, directory, units, other, effective, calls, births
    (root / 'manager-observation.json').write_text(json.dumps({'calls': calls, 'effective': effective}, sort_keys=True, indent=2))


def test_closed_host_uninstall_preserves_research_and_other_project(closed_project):
    root, service, svc, directory, units, other, effective, calls, births = closed_project
    db_sha = hashlib.sha256((root / 'lake.db').read_bytes()).hexdigest()
    closed = (root / 'service.json.host.lock/closed.json').read_bytes()
    scoped.uninstall_units(svc, directory)
    assert calls == [('disable', name) for name in scoped.unit_names(svc)] + [('daemon-reload',)]
    assert list(directory.iterdir()) == [other] and other.read_text() == 'independent project'
    assert service.exists() and (root / 'service.json.host.lock/closed.json').read_bytes() == closed
    assert hashlib.sha256((root / 'lake.db').read_bytes()).hexdigest() == db_sha
    assert len(births) == 1


def test_closed_host_status_reports_verified_record(closed_project, monkeypatch, capsys):
    root, service, svc, directory, units, other, effective, calls, births = closed_project
    monkeypatch.setattr(status, '_is_systemd_active', lambda unit: False)
    monkeypatch.setattr(status, '_is_systemd_timer_active', lambda unit: False)
    status.show_status(service_config_file=str(service))
    output = capsys.readouterr().out
    controller = json.loads((root / 'service.json.host.lock/closed.json').read_text())['controller_id']
    assert 'Verified closure record for controller ' + controller in output
    assert calls == [] and len(births) == 1


@pytest.mark.parametrize('mutation', ['active_main', 'active_watchdog', 'active_timer', 'live_pid',
    'remaining_cgroup', 'timer_target', 'install_also', 'ack_missing', 'closed_hash', 'after_first_disable'])
def test_closed_uninstall_rechecks_state_without_process_actions(closed_project, monkeypatch, mutation):
    root, service, svc, directory, units, other, effective, calls, births = closed_project
    main, timer, watchdog = scoped.unit_names(svc)
    if mutation.startswith('active_'):
        effective[{'active_main': main, 'active_watchdog': watchdog, 'active_timer': timer}[mutation]]['ActiveState'] = 'active'
    elif mutation == 'live_pid':
        effective[main]['MainPID'] = '12345'
    elif mutation == 'remaining_cgroup':
        effective[main]['ControlGroup'] = '/fixture/not-empty'
    elif mutation == 'timer_target':
        effective[timer]['Unit'] = 'another-project.service'
    elif mutation == 'install_also':
        effective[main]['_UnitText'] += '\n[Install]\nAlso=another-project.service\n'
    elif mutation == 'ack_missing':
        with sqlite3.connect(root / 'lake.db') as conn:
            conn.execute('UPDATE controller_sessions SET ack_json=NULL')
    elif mutation == 'closed_hash':
        path = root / 'service.json.host.lock/closed.json'
        value = json.loads(path.read_text())
        value['ack_sha256'] = '0' * 64
        path.write_text(json.dumps(value))
    else:
        def disable(*args):
            calls.append(args)
            effective[timer]['ActiveState'] = 'active'
        monkeypatch.setattr(scoped, '_checked', disable)
    with pytest.raises(RuntimeError):
        scoped.uninstall_units(svc, directory)
    assert calls == ([('disable', main)] if mutation == 'after_first_disable' else [])
    assert all((directory / name).read_text() == value for name, value in units.items())
    assert other.read_text() == 'independent project' and len(births) == 1
