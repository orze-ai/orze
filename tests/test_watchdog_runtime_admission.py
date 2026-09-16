"""A rejected runtime must not authorize watchdog process operations."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from orze.core.control_outcome import STOP_SENTINELS
from orze.service import runtime_contract, watchdog
from orze.service.failure_loop import read_failure_state


@pytest.fixture
def admission(tmp_path, monkeypatch):
    project = tmp_path / 'project'
    project.mkdir()
    results = project / 'results'
    results.mkdir()
    config = project / 'orze.yaml'
    config.write_text('results_dir: results\nnotifications:\n  enabled: false\n')
    (project / '.env').write_text('')
    packages = [{'name': 'orze', 'root': '/isolated/runtime/orze',
                 'sha256': 'a' * 64, 'file_count': 1}]
    cfg = {'method': 'systemd', 'python': sys.executable, 'workdir': str(project),
           'results_dir': str(results), 'config_file': str(config),
           'log_file': str(results / 'watchdog.log'),
           'runtime_contract_version': runtime_contract.CONTRACT_VERSION,
           'runtime_packages': copy.deepcopy(packages)}
    props = {'Restart': 'no', 'WorkingDirectory': str(project),
             'ExecStart': f'{{ path={sys.executable} ; argv[]={sys.executable} -m orze.cli -c {config} ; ignore_errors=no ; }}',
             'ExecStartPre': f'{{ path={sys.executable} ; argv[]={sys.executable} -m orze.service.runtime_contract --startup-check ; ignore_errors=no ; }}',
             'Environment': '', 'EnvironmentFiles': '', 'PassEnvironment': '',
             'UnsetEnvironment': ' '.join(sorted(runtime_contract._RUNTIME_ENVIRONMENT_KEYS)),
             'ActiveState': 'inactive', 'UnitFileState': 'disabled'}
    events, notifications = [], []
    monkeypatch.setattr(runtime_contract, 'capture_runtime_packages', lambda: copy.deepcopy(packages))
    monkeypatch.setattr(runtime_contract, '_systemd_properties', lambda: dict(props))
    monkeypatch.setattr(watchdog.socket, 'gethostname', lambda: 'admission-host')
    monkeypatch.setattr(watchdog.time, 'sleep', lambda *args: None)
    monkeypatch.setattr(watchdog, '_read_pid', lambda *args: events.append('read_pid'))
    monkeypatch.setattr(watchdog, '_is_pid_alive', lambda *args: events.append('is_alive') or True)
    monkeypatch.setattr(watchdog, '_is_heartbeat_stale', lambda *args: (False, 0))
    monkeypatch.setattr(watchdog, '_is_orze_running', lambda: events.append('pgrep') or False)
    monkeypatch.setattr(watchdog, '_kill_stale', lambda pid: events.append('kill'))
    monkeypatch.setattr(watchdog, '_write_restart_marker', lambda *args: events.append('restart_marker'))
    monkeypatch.setattr(watchdog, '_notify_failure_loop', lambda cfg, event: notifications.append(dict(event)))
    def forbidden(*args, **kwargs):
        pytest.fail('isolated admission test crossed an unmocked process boundary')
    monkeypatch.setattr(watchdog.subprocess, 'run', forbidden)
    monkeypatch.setattr(watchdog.subprocess, 'Popen', forbidden)
    return SimpleNamespace(cfg=cfg, props=props, packages=packages, events=events,
                           notifications=notifications, results=results, project=project)


def flow(admission, monkeypatch, state):
    def read_pid(*args):
        admission.events.append('read_pid')
        return None if state == 'absent' else 424242
    monkeypatch.setattr(watchdog, '_read_pid', read_pid)
    monkeypatch.setattr(watchdog, '_is_heartbeat_stale', lambda *args: (state == 'stale', 99))


def observe(admission, tmp_path, error):
    value = {'events': admission.events, 'error': error.code if error else None,
             'notifications': admission.notifications,
             'state': read_failure_state(admission.results, 'admission-host')}
    (tmp_path / 'observation.json').write_text(json.dumps(value, indent=2, sort_keys=True))
    return value


@pytest.mark.parametrize('state', ['absent', 'healthy', 'stale'])
@pytest.mark.parametrize('invalid', ['missing_version', 'unknown_version', 'missing_packages',
                                    'package_drift', 'python_drift', 'missing_config',
                                    'missing_workdir', 'unit_drift'])
def test_runtime_rejection_precedes_process_inspection(admission, tmp_path, monkeypatch, state, invalid):
    cfg = admission.cfg
    if invalid == 'missing_version':
        cfg.pop('runtime_contract_version')
    elif invalid == 'unknown_version':
        cfg['runtime_contract_version'] = 999
    elif invalid == 'missing_packages':
        cfg.pop('runtime_packages')
    elif invalid == 'package_drift':
        admission.packages[0]['sha256'] = 'b' * 64
    elif invalid == 'python_drift':
        cfg['python'] = '/bin/sh'
    elif invalid == 'missing_config':
        Path(cfg['config_file']).unlink()
    elif invalid == 'missing_workdir':
        cfg['workdir'] = str(tmp_path / 'absent')
    else:
        admission.props['Restart'] = 'always'
    assert runtime_contract.audit_runtime_contract(cfg)['startup_allowed'] is False
    flow(admission, monkeypatch, state)
    error = None
    try:
        watchdog.check_and_restart(cfg)
    except watchdog.WatchdogLaunchError as exc:
        error = exc
    value = observe(admission, tmp_path, error)
    assert value['error'] == 'runtime_contract_rejected', value
    assert admission.events == [], value
    assert value['state']['active'] and value['state']['consecutive_count'] == 1


def test_drift_during_heartbeat_is_rechecked_before_stale_kill(admission, tmp_path, monkeypatch):
    flow(admission, monkeypatch, 'stale')
    def changed_heartbeat(*args):
        admission.packages[0]['sha256'] = 'b' * 64
        return True, 99
    monkeypatch.setattr(watchdog, '_is_heartbeat_stale', changed_heartbeat)
    with pytest.raises(watchdog.WatchdogLaunchError) as caught:
        watchdog.check_and_restart(admission.cfg)
    value = observe(admission, tmp_path, caught.value)
    assert value['error'] == 'runtime_contract_rejected'
    assert 'kill' not in admission.events and 'restart_marker' not in admission.events, value


def test_launch_boundary_rechecks_after_process_scan(admission, tmp_path, monkeypatch):
    flow(admission, monkeypatch, 'absent')
    def changed_scan():
        admission.events.append('pgrep')
        admission.packages[0]['sha256'] = 'b' * 64
        return False
    monkeypatch.setattr(watchdog, '_is_orze_running', changed_scan)
    with pytest.raises(watchdog.WatchdogLaunchError) as caught:
        watchdog.check_and_restart(admission.cfg)
    value = observe(admission, tmp_path, caught.value)
    assert value['error'] == 'runtime_contract_rejected' and 'kill' not in admission.events


@pytest.mark.parametrize('fault', ['exception', 'invalid_report'])
def test_audit_unavailable_blocks_processes_without_raw_text(admission, tmp_path, monkeypatch, fault):
    def audit(*args):
        if fault == 'exception':
            raise RuntimeError('token=private-fixture-value')
        return None
    monkeypatch.setattr(runtime_contract, 'audit_runtime_contract', audit)
    flow(admission, monkeypatch, 'stale')
    with pytest.raises(watchdog.WatchdogLaunchError) as caught:
        watchdog.check_and_restart(admission.cfg)
    value = observe(admission, tmp_path, caught.value)
    assert value['error'] == 'runtime_contract_unavailable' and admission.events == [], value
    assert 'private-fixture-value' not in str(caught.value)
    assert 'private-fixture-value' not in Path(admission.cfg['log_file']).read_text()
    assert 'private-fixture-value' not in json.dumps(value['state'])


@pytest.mark.parametrize('sentinel', STOP_SENTINELS)
def test_operator_stop_precedes_runtime_audit(admission, tmp_path, monkeypatch, sentinel):
    (admission.results / sentinel).write_text('operator stop')
    def unexpected(*args):
        pytest.fail('operator stop should return without a runtime audit')
    monkeypatch.setattr(runtime_contract, 'audit_runtime_contract', unexpected)
    flow(admission, monkeypatch, 'stale')
    watchdog.check_and_restart(admission.cfg)
    observe(admission, tmp_path, None)
    assert admission.events == []


@pytest.mark.parametrize('method', ['systemd', 'crontab'])
@pytest.mark.parametrize('state', ['absent', 'stale'])
def test_admitted_restart_keeps_existing_service_owner(admission, tmp_path, monkeypatch, method, state):
    admission.cfg['method'] = method
    flow(admission, monkeypatch, state)
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout='765432\n', stderr='')
    def popen(args, **kwargs):
        calls.append(args)
        assert kwargs['cwd'] == str(admission.project) and kwargs['start_new_session']
        return SimpleNamespace(pid=765432)
    monkeypatch.setattr(watchdog.subprocess, 'run', run)
    monkeypatch.setattr(watchdog.subprocess, 'Popen', popen)
    watchdog.check_and_restart(admission.cfg)
    observe(admission, tmp_path, None)
    if method == 'systemd':
        assert calls == [
            ['systemctl', '--user', 'reset-failed', 'orze.service'],
            ['systemctl', '--user', 'start', 'orze.service'],
            ['systemctl', '--user', 'show', 'orze.service', '--property=MainPID', '--value']]
    else:
        assert calls == [[sys.executable, '-m', 'orze.cli', '-c', admission.cfg['config_file']]]
    assert ('kill' in admission.events) == (state == 'stale')


@pytest.mark.parametrize('method', ['systemd', 'crontab'])
def test_admitted_healthy_process_is_left_running(admission, tmp_path, monkeypatch, method):
    admission.cfg['method'] = method
    flow(admission, monkeypatch, 'healthy')
    watchdog.check_and_restart(admission.cfg)
    observe(admission, tmp_path, None)
    assert admission.events == ['read_pid', 'is_alive']


def test_repeated_admission_failure_preserves_escalation(admission, tmp_path, monkeypatch):
    admission.packages[0]['sha256'] = 'b' * 64
    flow(admission, monkeypatch, 'absent')
    for _ in range(3):
        with pytest.raises(watchdog.WatchdogLaunchError) as caught:
            watchdog.check_and_restart(admission.cfg)
    value = observe(admission, tmp_path, caught.value)
    assert admission.events == [], value
    assert value['state']['consecutive_count'] == 3
    assert len(admission.notifications) == 1 and admission.notifications[0]['consecutive_count'] == 2
