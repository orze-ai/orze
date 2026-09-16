"""Install metadata must watch the same result authority as its controller cwd."""
import copy
import json
import os
from pathlib import Path

import pytest
import yaml

from orze.core.control_outcome import STOP_SENTINELS
from orze.service import install


@pytest.fixture
def installation(tmp_path, monkeypatch):
    project, caller = tmp_path / 'project', tmp_path / 'installer-cwd'
    for path in (project, caller):
        path.mkdir()
        (path / '.env').write_text('')
        (path / 'results').mkdir()
    config = project / 'orze.yaml'
    config.write_text('results_dir: results\n')
    service = tmp_path / 'service.json'
    calls = []
    monkeypatch.chdir(caller)
    monkeypatch.setattr(install, 'SERVICE_CONFIG_PATH', service)
    monkeypatch.setattr('orze.service.runtime_contract.capture_runtime_packages', lambda: [])
    monkeypatch.setattr(install, '_install_systemd', lambda cfg: calls.append(('systemd', copy.deepcopy(cfg))))
    monkeypatch.setattr(install, '_install_crontab', lambda cfg: calls.append(('crontab', copy.deepcopy(cfg))))
    def forbidden(*args, **kwargs):
        pytest.fail('isolated route test attempted a real subprocess')
    monkeypatch.setattr(install.subprocess, 'run', forbidden)
    monkeypatch.setattr(install.subprocess, 'Popen', forbidden)
    return project, caller, config, service, calls


def observed(installation, tmp_path):
    project, caller, config, service, calls = installation
    value = {'cwd_after': str(Path.cwd()), 'service_exists': service.exists(),
             'saved': json.loads(service.read_text()) if service.exists() else None,
             'rendered_calls': calls}
    (tmp_path / 'route-observation.json').write_text(json.dumps(value, indent=2, sort_keys=True))
    return value


@pytest.mark.parametrize('method', ['systemd', 'crontab'])
@pytest.mark.parametrize('route', ['child', 'parent', 'absolute', 'env', 'symlink_config'])
def test_installed_routes_match_controller_workdir(installation, tmp_path, monkeypatch, method, route):
    project, caller, config, service, calls = installation
    text = 'results'
    expected = project / 'results'
    if route == 'parent':
        text = '../shared-results'
        expected = tmp_path / 'shared-results'
    elif route == 'absolute':
        expected = tmp_path / 'absolute-results'
        text = str(expected)
    elif route == 'env':
        text = '${ORZE_TEST_SERVICE_INSTALL_RESULTS}'
        monkeypatch.setenv('ORZE_TEST_SERVICE_INSTALL_RESULTS', 'env-results')
        expected = project / 'env-results'
    config.write_text(yaml.safe_dump({'results_dir': text}))
    expected.mkdir(exist_ok=True)
    if route == 'symlink_config':
        alias = caller / 'linked-config.yaml'
        alias.symlink_to(config)
        config = alias
    install.install(os.path.relpath(config, caller), method=method)
    value = observed(installation, tmp_path)
    assert value['cwd_after'] == str(caller)
    assert len(calls) == 1 and calls[0][0] == method
    saved = value['saved']
    assert saved == calls[0][1]
    assert saved['config_file'] == str((project / 'orze.yaml').resolve())
    assert saved['workdir'] == str(project.resolve())
    assert saved['results_dir'] == str(expected.resolve()), value
    assert saved['log_file'] == str(expected.resolve() / 'orze.log'), value


@pytest.mark.parametrize('method', ['systemd', 'crontab'])
def test_install_from_project_directory_keeps_existing_route(installation, tmp_path, monkeypatch, method):
    project, caller, config, service, calls = installation
    monkeypatch.chdir(project)
    install.install('orze.yaml', method=method)
    value = observed(installation, tmp_path)
    assert value['saved']['results_dir'] == str(project / 'results')
    assert value['saved']['log_file'] == str(project / 'results/orze.log')
    assert Path.cwd() == project and len(calls) == 1


@pytest.mark.parametrize('method', ['systemd', 'crontab'])
@pytest.mark.parametrize('sentinel', STOP_SENTINELS)
def test_target_stop_preserves_existing_service_metadata(installation, tmp_path, method, sentinel):
    project, caller, config, service, calls = installation
    (project / 'results' / sentinel).write_text('operator stop')
    original = b'old service record must remain exact\n'
    service.write_bytes(original)
    error = None
    try:
        install.install(str(config), method=method)
    except RuntimeError as exc:
        error = str(exc)
    value = {'error': error, 'service_unchanged': service.read_bytes() == original,
             'rendered_calls': calls, 'cwd_after': str(Path.cwd())}
    (tmp_path / 'route-observation.json').write_text(json.dumps(value, indent=2, sort_keys=True))
    assert error and 'stop latch' in error, value
    assert value['service_unchanged'] and calls == [], value
    assert Path.cwd() == caller


@pytest.mark.parametrize('method', ['systemd', 'crontab'])
@pytest.mark.parametrize('sentinel', STOP_SENTINELS)
def test_unrelated_caller_stop_does_not_block_target(installation, tmp_path, method, sentinel):
    project, caller, config, service, calls = installation
    (caller / 'results' / sentinel).write_text('unrelated project stop')
    error = None
    try:
        install.install(str(config), method=method)
    except RuntimeError as exc:
        error = str(exc)
    value = observed(installation, tmp_path)
    value['error'] = error
    (tmp_path / 'route-observation.json').write_text(json.dumps(value, indent=2, sort_keys=True))
    assert error is None, value
    assert value['saved']['results_dir'] == str(project / 'results')
    assert len(calls) == 1 and Path.cwd() == caller
