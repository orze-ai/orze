"""Service startup and watchdog must use the current project's results route."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from orze.core import config as config_module
from orze.service import runtime_contract, watchdog


@pytest.fixture
def route(tmp_path, monkeypatch):
    project = tmp_path / 'project'
    project.mkdir()
    for name in ('a', 'b'):
        (project / name).mkdir()
    config = project / 'orze.yaml'
    config.write_text('results_dir: a\n')
    (project / '.env').write_text('')
    packages = [{'name': 'orze', 'root': '/fixture/runtime',
                 'sha256': 'a' * 64, 'file_count': 1}]
    svc = {'method': 'crontab', 'python': sys.executable, 'workdir': str(project),
           'results_dir': str(project / 'a'), 'config_file': str(config),
           'log_file': str(project / 'a/watchdog.log'),
           'runtime_contract_version': 1, 'runtime_packages': packages}
    monkeypatch.setattr(runtime_contract, 'capture_runtime_packages', lambda: copy.deepcopy(packages))
    return project, config, svc


def audit(route):
    return runtime_contract.audit_runtime_contract(route[2])


@pytest.mark.parametrize('mutation', ['config', 'dotenv', 'inherited_env', 'config_link', 'results_link'])
def test_current_route_change_rejects_startup(route, monkeypatch, mutation):
    project, config, svc = route
    key = 'ORZE_ROUTE_AUDIT_TEST_RESULTS'
    monkeypatch.delenv(key, raising=False)
    if mutation in ('dotenv', 'inherited_env'):
        config.write_text('results_dir: ${' + key + '}\n')
        if mutation == 'dotenv':
            (project / '.env').write_text(key + '=a\n')
        else:
            monkeypatch.setenv(key, 'a')
    elif mutation == 'config_link':
        original = project / 'original.yaml'
        config.rename(original)
        config.symlink_to(original.name)
    elif mutation == 'results_link':
        (project / 'linked').symlink_to('a', target_is_directory=True)
        config.write_text('results_dir: linked\n')
    assert audit(route)['startup_allowed']
    if mutation == 'config':
        config.write_text('results_dir: b\n')
    elif mutation == 'dotenv':
        (project / '.env').write_text(key + '=b\n')
    elif mutation == 'inherited_env':
        monkeypatch.setenv(key, 'b')
    elif mutation == 'config_link':
        (project / 'other.yaml').write_text('results_dir: b\n')
        config.unlink()
        config.symlink_to('other.yaml')
    else:
        (project / 'linked').unlink()
        (project / 'linked').symlink_to('b', target_is_directory=True)
    (project / 'b/.orze_stop_all').write_text('operator stop')
    result = audit(route)
    assert not result['startup_allowed'], result
    assert 'service_results_route_drift' in result['errors'], result
    assert svc['results_dir'] == str(project / 'a')


@pytest.mark.parametrize('text', ['results_dir: [a]\n', 'results_dir: null\n', 'results_dir: 3\n', '[a, b]\n', 'results_dir: [\n'])
def test_unreadable_route_is_not_accepted(route, text):
    route[1].write_text(text)
    result = audit(route)
    assert not result['startup_allowed'], result
    assert 'service_results_route_unavailable' in result['errors'], result


@pytest.mark.parametrize('boundary', ['entry', 'stale', 'launch'])
def test_route_drift_blocks_watchdog_process_actions(route, monkeypatch, boundary):
    project, config, svc = route
    events = []
    monkeypatch.setattr(watchdog, '_notify_failure_loop', lambda *args: None)
    def changed():
        config.write_text('results_dir: b\n')
        (project / 'b/.orze_shutdown').write_text('stop')
    def read_pid(*args):
        events.append('read')
        return 424242 if boundary == 'stale' else None
    def heartbeat(*args):
        changed()
        return True, 9999
    def scan():
        events.append('scan')
        changed()
        return False
    def forbidden(*args, **kwargs):
        pytest.fail('route drift reached a process mutation')
    monkeypatch.setattr(watchdog, '_read_pid', read_pid)
    monkeypatch.setattr(watchdog, '_is_pid_alive', lambda *args: True)
    monkeypatch.setattr(watchdog, '_is_heartbeat_stale', heartbeat)
    monkeypatch.setattr(watchdog, '_is_orze_running', scan)
    monkeypatch.setattr(watchdog, '_kill_stale', forbidden)
    monkeypatch.setattr(watchdog.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(watchdog.subprocess, 'run', forbidden)
    if boundary == 'entry':
        changed()
    with pytest.raises(watchdog.WatchdogLaunchError, match='service_results_route_drift'):
        watchdog.check_and_restart(svc)
    if boundary == 'entry':
        assert events == []


@pytest.mark.parametrize('mutation', ['config', 'dotenv', 'parent', 'environment'])
def test_route_inputs_changed_during_audit_are_rejected(route, monkeypatch, mutation):
    project, config, svc = route
    def capture():
        if mutation == 'config':
            replacement = project / 'new.yaml'
            replacement.write_bytes(config.read_bytes())
            replacement.replace(config)
        elif mutation == 'dotenv':
            replacement = project / 'new.env'
            replacement.write_bytes((project / '.env').read_bytes())
            replacement.replace(project / '.env')
        elif mutation == 'parent':
            (project / 'a').rename(project / 'old-a')
            (project / 'a').mkdir()
        else:
            monkeypatch.setenv('ORZE_ROUTE_AUDIT_TEST_NEW_ENV', 'changed')
        return copy.deepcopy(svc['runtime_packages'])
    monkeypatch.setattr(runtime_contract, 'capture_runtime_packages', capture)
    result = audit(route)
    assert not result['startup_allowed'], result
    assert 'service_results_route_changed' in result['errors'], result


def test_unrelated_config_edit_and_other_directory_stop_remain_compatible(route):
    project, config, svc = route
    config.write_text('results_dir: a\npoll: 27\n')
    (project / 'b/.orze_stop_all').write_text('other project stop')
    assert audit(route)['startup_allowed']


@pytest.mark.parametrize('case', ['default', 'relative', 'parent', 'absolute', 'dotenv',
                                'inherited', 'empty_env', 'unset', 'config_link', 'cwd_dotenv'])
def test_pure_route_matches_actual_loader_in_service_cwd(route, tmp_path, monkeypatch, case):
    project, config, svc = route
    key = 'ORZE_ROUTE_AUDIT_TEST_RESULTS'
    monkeypatch.delenv(key, raising=False)
    text = 'a'
    if case == 'default':
        config.write_text('{}\n')
    else:
        if case == 'parent': text = '../outside'
        if case == 'absolute': text = str(tmp_path / 'absolute')
        if case in ('dotenv', 'inherited', 'empty_env', 'unset', 'cwd_dotenv'):
            text = '${' + key + '}'
        if case in ('dotenv', 'inherited', 'empty_env', 'cwd_dotenv'):
            (project / '.env').write_text(key + '=from-dotenv\n')
        if case == 'inherited': monkeypatch.setenv(key, 'from-environment')
        if case == 'empty_env': monkeypatch.setenv(key, '')
        config.write_text('results_dir: ' + text + '\n')
    if case in ('config_link', 'cwd_dotenv'):
        elsewhere = tmp_path / 'elsewhere'
        elsewhere.mkdir()
        other = elsewhere / 'source.yaml'
        config.rename(other)
        config.symlink_to(other)
        if case == 'config_link':
            (elsewhere / '.env').write_text('')
    before_env, before_cwd = dict(os.environ), Path.cwd()
    pure = config_module.resolve_project_results(str(config), workdir=str(project))
    assert dict(os.environ) == before_env and Path.cwd() == before_cwd
    program = ('import json,sys; from pathlib import Path; '
               'from orze.core.config import load_project_config; '
               'print(json.dumps(str(Path(load_project_config(sys.argv[1])["results_dir"]).resolve())))')
    actual = subprocess.run([sys.executable, '-c', program, str(config)], cwd=project,
                            env=dict(os.environ), text=True, capture_output=True, check=True)
    assert pure == Path(json.loads(actual.stdout))
