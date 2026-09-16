"""Scoped management routing; manager responses here are explicit substitutes.

Actual controller lifetime is covered by test_service_host_product. These
checks do not deploy services or establish a real systemd cgroup result.
"""
import json
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

from orze.service import host, install, runtime_contract, scoped, status
from test_cpu_controller_profile import supported


@pytest.fixture
def projects(tmp_path, monkeypatch):
    packages = [{'name': 'orze', 'root': '/fixture/runtime', 'sha256': 'a' * 64, 'file_count': 1}]
    monkeypatch.setattr(runtime_contract, 'capture_runtime_packages', lambda: packages)
    items = []
    for name in ('a', 'b'):
        root = tmp_path / name
        root.mkdir()
        (root / 'results').mkdir()
        (root / '.env').write_text('')
        cfg = {**supported(), 'controller_control': {'version': 2, 'profile': 'local_cpu_handoff_v1'},
               'results_dir': str(root / 'results')}
        config = root / 'orze.yaml'
        config.write_text(yaml.safe_dump(cfg))
        items.append({'service_owner': host.PROFILE, 'service_config_file': str(root / 'service.json'),
                      'config_file': str(config), 'workdir': str(root), 'results_dir': str(root / 'results'),
                      'python': sys.executable, 'method': 'systemd', 'stall_threshold': 60,
                      'log_file': str(root / 'results/orze.log'), 'runtime_contract_version': 1,
                      'runtime_packages': packages})
    return items


def properties(svc):
    python, config = svc['python'], svc['service_config_file']
    def record(*args):
        return '{ path=' + python + ' ; argv[]=' + ' '.join((python, *args)) + ' ; ignore_errors=no ; }'
    return {'Id': scoped.unit_names(svc)[0],
            'ExecCondition': '', 'ExecStartPost': '', 'ExecStop': '', 'ExecStopPost': '', 'ExecReload': '',
            'Restart': 'no', 'Type': 'simple', 'KillMode': 'control-group', 'MainPID': '12000',
            'ControlGroup': '/fixture/host', 'WorkingDirectory': svc['workdir'],
            'ExecStart': record('-m', 'orze.service.host', '--service-config', config),
            'ExecStartPre': record('-m', 'orze.service.runtime_contract', '--startup-check', '--service-config', config),
            'Environment': '', 'EnvironmentFiles': '', 'PassEnvironment': '',
            'UnsetEnvironment': ' '.join(runtime_contract._RUNTIME_ENVIRONMENT_KEYS),
            'ActiveState': 'active', 'UnitFileState': 'enabled'}


def property_map(svc):
    main, timer, watchdog = scoped.unit_names(svc)
    main_props = properties(svc)
    main_props.update(Id=main, ExecCondition='', ExecStartPost='', ExecStop='', ExecStopPost='', ExecReload='')
    watchdog_props = {**main_props, 'Id': watchdog, 'Type': 'oneshot', 'ExecStartPre': '',
                      'ExecStart': main_props['ExecStart'].replace('orze.service.host', 'orze.service.watchdog')}
    return {main: main_props, watchdog: watchdog_props,
            timer: {'Id': timer, 'Unit': watchdog, 'ActiveState': 'active', 'UnitFileState': 'enabled'}}


@pytest.mark.parametrize('mutation', ['watchdog_config', 'watchdog_pre', 'watchdog_pythonpath',
    'timer_target', 'host_stop', 'watchdog_stop', 'host_post', 'missing_watchdog', 'missing_timer'])
def test_effective_related_units_cannot_escape_selected_project(projects, tmp_path, monkeypatch, mutation):
    svc, other = projects
    effective = property_map(svc)
    main, timer, watchdog = scoped.unit_names(svc)
    if mutation == 'watchdog_config':
        effective[watchdog]['ExecStart'] = property_map(other)[scoped.unit_names(other)[2]]['ExecStart']
    elif mutation == 'watchdog_pre':
        effective[watchdog]['ExecStartPre'] = effective[main]['ExecStartPre']
    elif mutation == 'watchdog_pythonpath':
        effective[watchdog]['Environment'] = 'PYTHONPATH=/fixture/different-runtime'
    elif mutation == 'timer_target':
        effective[timer]['Unit'] = scoped.unit_names(other)[2]
    elif mutation in ('host_stop', 'watchdog_stop', 'host_post'):
        effective[watchdog if mutation == 'watchdog_stop' else main][
            'ExecStartPost' if mutation == 'host_post' else 'ExecStop'] = '{ path=/fixture/other-owner ; argv[]=/fixture/other-owner ; ignore_errors=no ; }'
    else:
        del effective[watchdog if mutation == 'missing_watchdog' else timer]
    calls = []
    def observed(unit, **kwargs):
        calls.append(unit)
        if unit not in effective:
            raise runtime_contract.RuntimeContractError('systemd_properties_unavailable')
        return copy.deepcopy(effective[unit])
    monkeypatch.setattr(runtime_contract, '_systemd_properties', observed)
    report = runtime_contract.audit_runtime_contract(svc)
    (tmp_path / 'effective-observation.json').write_text(json.dumps({'report': report, 'calls': calls, 'mutation': mutation}, sort_keys=True, indent=2))
    assert not report['startup_allowed'], report


@pytest.mark.parametrize('stage', ['entry', 'after_stop'])
def test_effective_drift_blocks_uninstall_manager_mutations(projects, tmp_path, monkeypatch, stage):
    svc = projects[0]
    effective = property_map(svc)
    main, timer, watchdog = scoped.unit_names(svc)
    directory = tmp_path / 'units'
    directory.mkdir()
    original = scoped.render_units(svc)
    for name, text in original.items():
        (directory / name).write_text(text)
    def drift():
        effective[timer]['Unit'] = scoped.unit_names(projects[1])[2]
    if stage == 'entry':
        drift()
    events = []
    def request(path, operation):
        events.append(('host', operation))
        drift()
        return {'kind': 'stopped'}
    monkeypatch.setattr(runtime_contract, '_systemd_properties', lambda unit, **kwargs: copy.deepcopy(effective[unit]))
    monkeypatch.setattr(host, 'request', request)
    monkeypatch.setattr(scoped, '_checked', lambda *args: events.append(('manager', *args)))
    error = None
    try:
        scoped.uninstall_units(svc, directory)
    except RuntimeError as exc:
        error = str(exc)
    (tmp_path / 'uninstall-observation.json').write_text(json.dumps({'error': error, 'events': events}, sort_keys=True, indent=2))
    assert error is not None and not any(row[0] == 'manager' for row in events)
    assert events == ([] if stage == 'entry' else [('host', 'stop')])
    assert all((directory / name).read_text() == value for name, value in original.items())


def test_two_installs_have_distinct_owner_timer_and_configuration(projects, tmp_path, monkeypatch):
    units = tmp_path / 'units'
    legacy = tmp_path / 'legacy.json'
    legacy.write_text('existing global service must remain unchanged')
    monkeypatch.setattr(install, '_SYSTEMD_DIR', units)
    monkeypatch.setattr(install, 'SERVICE_CONFIG_PATH', legacy)
    calls, ready = [], []
    monkeypatch.setattr(scoped, '_checked', lambda *args: calls.append(args))
    monkeypatch.setattr(scoped, '_wait_ready', lambda cfg: ready.append(cfg['service_config_file']))
    effective = {key: value for svc in projects for key, value in property_map(svc).items()}
    monkeypatch.setattr(runtime_contract, '_systemd_properties', lambda unit, **kwargs: copy.deepcopy(effective[unit]))
    for svc in projects:
        install.install(svc['config_file'], method='systemd', service_config_file=svc['service_config_file'])
    assert legacy.read_text() == 'existing global service must remain unchanged'
    assert set(p.name for p in units.iterdir()) == {name for svc in projects for name in scoped.unit_names(svc)}
    assert ready == [x['service_config_file'] for x in projects]
    for svc in projects:
        saved = json.loads(Path(svc['service_config_file']).read_text())
        assert saved['service_owner'] == host.PROFILE and saved['results_dir'] == svc['results_dir']
        main, timer, watchdog = scoped.unit_names(saved)
        assert ('enable', '--now', main) in calls and ('enable', '--now', timer) in calls
        text = (units / main).read_text()
        assert 'orze.service.host --service-config ' + svc['service_config_file'] in text
        assert 'KillMode=control-group' in text and 'Restart=no' in text
        assert 'Unit=' + watchdog in (units / timer).read_text()
        assert svc['service_config_file'] in (units / watchdog).read_text()
    assert not any('orze.service' in call or 'orze-watchdog.timer' in call for call in calls)


@pytest.mark.parametrize('mutation', ['start_other_config', 'pre_other_config', 'killmode', 'type', 'pid', 'cgroup'])
def test_effective_unit_and_owner_drift_are_rejected(projects, monkeypatch, mutation):
    svc, other = projects
    props = properties(svc)
    related = {key: value for key, value in property_map(svc).items() if key != scoped.unit_names(svc)[0]}
    monkeypatch.setattr(runtime_contract, 'manager_cgroup', lambda pid: '/fixture/host')
    assert runtime_contract.audit_runtime_contract(svc, properties=props, related_properties=related, expected_host_pid=12000)['startup_allowed']
    if mutation == 'start_other_config':
        props['ExecStart'] = properties(other)['ExecStart']
    elif mutation == 'pre_other_config':
        props['ExecStartPre'] = properties(other)['ExecStartPre']
    elif mutation == 'killmode':
        props['KillMode'] = 'process'
    elif mutation == 'type':
        props['Type'] = 'oneshot'
    elif mutation == 'pid':
        props['MainPID'] = '12001'
    else:
        props['ControlGroup'] = '/fixture/another-host'
    assert not runtime_contract.audit_runtime_contract(svc, properties=props, related_properties=related, expected_host_pid=12000)['startup_allowed']


def test_unit_collision_preserves_all_existing_bytes(projects, tmp_path, monkeypatch):
    directory = tmp_path / 'units'
    directory.mkdir()
    name = scoped.unit_names(projects[0])[1]
    target = directory / name
    target.write_text('unrelated owner')
    monkeypatch.setattr(scoped, '_checked', lambda *args: pytest.fail('collision reached manager'))
    with pytest.raises(RuntimeError, match='already_present'):
        scoped.install_units(projects[0], directory)
    assert list(directory.iterdir()) == [target] and target.read_text() == 'unrelated owner'


def test_partial_enable_closes_selected_host_before_disabling(projects, tmp_path, monkeypatch):
    svc = projects[0]
    main, timer, watchdog = scoped.unit_names(svc)
    events = []
    def checked(*args):
        events.append(args)
        if args == ('enable', '--now', timer):
            raise RuntimeError('fixture timer failure')
    monkeypatch.setattr(scoped, '_checked', checked)
    monkeypatch.setattr(scoped, '_wait_ready', lambda value: events.append(('ready',)))
    monkeypatch.setattr(runtime_contract, 'audit_runtime_contract', lambda value: {'startup_allowed': True, 'contract_ok': True, 'errors': []})
    monkeypatch.setattr(host, 'request', lambda path, operation: events.append((operation, path)) or {'kind': 'stopped'})
    with pytest.raises(RuntimeError, match='fixture timer failure'):
        scoped.install_units(svc, tmp_path / 'units')
    assert events[-3:] == [('stop', svc['service_config_file']), ('disable', '--now', timer), ('disable', '--now', main)]


def test_status_logs_audit_and_uninstall_select_the_same_project(projects, tmp_path, monkeypatch, capsys):
    directory = tmp_path / 'units'
    directory.mkdir()
    for svc in projects:
        Path(svc['service_config_file']).write_text(json.dumps(svc))
        Path(svc['log_file']).write_text('log for ' + svc['workdir'])
        for name, text in scoped.render_units(svc).items():
            (directory / name).write_text(text)
    observed = []
    monkeypatch.setattr(install, '_SYSTEMD_DIR', directory)
    monkeypatch.setattr(scoped, '_checked', lambda *args: observed.append(args))
    effective = property_map(projects[0])
    monkeypatch.setattr(runtime_contract, '_systemd_properties', lambda unit, **kwargs: observed.append(('audit', unit)) or copy.deepcopy(effective[unit]))
    monkeypatch.setattr(status, '_is_systemd_active', lambda unit: observed.append(('active', unit)) or True)
    monkeypatch.setattr(status, '_is_systemd_timer_active', lambda unit: observed.append(('timer', unit)) or True)
    monkeypatch.setattr(status, '_read_pid', lambda *args: pytest.fail('host status probed a raw PID'))
    def request(path, operation, **kwargs):
        observed.append((operation, path))
        return {'kind': 'stopped'} if operation == 'stop' else {'controller': {'controller_id': 'fixture-controller'}}
    monkeypatch.setattr(host, 'request', request)
    selected = projects[0]
    status.show_status(service_config_file=selected['service_config_file'])
    status.show_logs(service_config_file=selected['service_config_file'])
    assert runtime_contract.main(['--service-config', selected['service_config_file']]) == 0
    before_other = {name: (directory / name).read_bytes() for name in scoped.unit_names(projects[1])}
    install.uninstall(service_config_file=selected['service_config_file'])
    assert all((directory / name).read_bytes() == value for name, value in before_other.items())
    assert not any((directory / name).exists() for name in scoped.unit_names(selected))
    assert Path(selected['service_config_file']).exists()
    assert ('status', selected['service_config_file']) in observed and ('stop', selected['service_config_file']) in observed
    assert all(projects[1]['service_config_file'] not in item for item in observed)
    assert 'log for ' + selected['workdir'] in capsys.readouterr().out


def test_config_collision_and_cron_refusal_precede_manager_calls(projects, tmp_path, monkeypatch):
    svc = projects[0]
    target = Path(svc['service_config_file'])
    target.write_text('old service')
    monkeypatch.setattr(install, '_install_systemd', lambda *args: pytest.fail('existing config reached manager'))
    with pytest.raises(FileExistsError):
        install.install(svc['config_file'], method='systemd', service_config_file=str(target))
    assert target.read_text() == 'old service'
    target.unlink()
    with pytest.raises(RuntimeError, match='requires_systemd'):
        install.install(svc['config_file'], method='crontab', service_config_file=str(target))
    assert not target.exists()
