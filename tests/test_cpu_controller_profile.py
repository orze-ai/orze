"""Explicit CPU stop metadata; no broadening of existing GPU profiles."""
import argparse
import copy

import pytest
import yaml

from orze.core.config import load_project_config
from orze.core.controller_profile import ControllerProfileError, controller_profile, profile_fingerprint
from orze.core.cpu_execution import CPUExecutionError, cpu_execution, validate_cpu_cli


def supported():
    return {'controller_control': {'version': 1, 'profile': 'local_cpu_stop_v1'},
        'execution': {'version': 1, 'resource': 'cpu', 'slots': 1, 'wall_budget_seconds': 10},
        'telemetry': False, 'auto_upgrade': False, 'metric_harvest': {'enabled': False},
        'max_fix_attempts': 0}


def test_cpu_stop_profile_is_loaded_with_both_fingerprints(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / 'orze.yaml'
    path.write_text(yaml.safe_dump(supported()))
    cfg = load_project_config(str(path))
    assert controller_profile(cfg) == supported()['controller_control']
    assert profile_fingerprint(cfg, []) == cfg['_controller_profile_fingerprint']
    assert cpu_execution(cfg)['resource'] == 'cpu'
    changed = copy.deepcopy(cfg)
    changed['controller_control'] = None
    with pytest.raises(CPUExecutionError, match='changed'):
        cpu_execution(changed)
    with pytest.raises(ControllerProfileError, match='gpu_scope'):
        profile_fingerprint(cfg, [0])


@pytest.mark.parametrize('change', [
    {'controller_control': {'version': True, 'profile': 'local_cpu_stop_v1'}},
    {'controller_control': {'version': 2, 'profile': 'local_cpu_stop_v1'}},
    {'controller_control': {'version': 1, 'profile': 'local_stop_v1'}},
    {'controller_control': {'version': 2, 'profile': 'local_handoff_v1'}},
    {'execution': None}, {'gpu_scheduling': {'allowed_gpus': [0]}},
    {'gpu_scheduling': {'allowed_gpus': False}},
    {'telemetry': True}, {'auto_upgrade': True}, {'roles': {'researcher': {'enabled': True}}},
])
def test_cpu_stop_requires_exact_declaration_and_resource_boundaries(change):
    cfg = {**supported(), **change}
    with pytest.raises((ControllerProfileError, CPUExecutionError)):
        cpu_execution(cfg)
        controller_profile(cfg)


def test_cpu_stop_timeout_is_control_timeout_and_restart_stays_unsupported():
    cfg = supported()
    validate_cpu_cli(cfg, argparse.Namespace(command='stop', timeout=7))
    validate_cpu_cli(cfg, argparse.Namespace(command=None, stop=True, timeout=7))
    for args in (argparse.Namespace(command=None, timeout=7),
                 argparse.Namespace(command='restart', timeout=7),
                 argparse.Namespace(command='stop', gpus='0')):
        with pytest.raises(CPUExecutionError):
            validate_cpu_cli(cfg, args)
    cfg['controller_control'] = None
    with pytest.raises(CPUExecutionError):
        validate_cpu_cli(cfg, argparse.Namespace(command='stop', timeout=7))
