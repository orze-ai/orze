"""Two separate runtime trees, real closed CPU services and unchanged budgets."""
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys

import pytest

from test_cpu_controller_stop_product import cpu_controller, rows
from test_service_backup import hosted_backup
from test_service_host_product import idea
from test_service_recovery import prepare


def command(root, env, label, module, *args):
    result = subprocess.run([sys.executable, '-m', module, *map(str, args)], cwd=root, env=env,
                            capture_output=True, text=True, timeout=60)
    (root/(label+'.stdout')).write_text(result.stdout)
    (root/(label+'.stderr')).write_text(result.stderr)
    return result


@pytest.fixture
def runtimes(hosted_backup):
    hosted = hosted_backup
    root, _, _, env, service, *_ = hosted
    destination = dict(env)
    source = Path(os.environ.get('ORZE_TEST_UPGRADE_SOURCE', Path(__file__).resolve().parents[1]/'src/orze'))
    frozen = root.with_name(root.name+'-runtime')/'orze'
    shutil.copytree(source, frozen, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
    inventory = {str(p.relative_to(frozen)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(frozen.rglob('*')) if p.is_file()}
    (root/'source-runtime-files.json').write_text(json.dumps(inventory, sort_keys=True, indent=2))
    env['PYTHONPATH'] = os.pathsep.join([str(root), str(frozen.parent), destination['PYTHONPATH']])
    capture = command(root, env, 'source-runtime', 'orze.service.runtime_contract', '--capture-controller')
    assert capture.returncode == 0, capture.stderr
    svc = json.loads(service.read_bytes())
    svc['runtime_packages'] = json.loads(capture.stdout)['packages']
    service.write_text(json.dumps(svc, sort_keys=True))
    try:
        yield hosted, destination, frozen
    finally:
        assert inventory == {str(p.relative_to(frozen)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(frozen.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


@pytest.mark.parametrize('previous_handoff', [False, True], ids=['initial-source', 'recovered-source'])
def test_closed_runtime_upgrade_preserves_history_and_exhausted_budget(runtimes, previous_handoff):
    hosted, destination_env, frozen = runtimes
    root, config, cfg, env, service, start, client, wait, prior, captures, births, before_birth = hosted
    idea(root, 'idea-0001', 'print("old runtime action")')
    source = start()
    wait(lambda: rows(root, 'SELECT state FROM cpu_action_reservations') == [('SETTLED',)])
    assert client('stop')[0] == 0 and source.wait(timeout=5) == 0
    prior[:] = captures
    if previous_handoff:
        recovered = root/'source-recovered.json'
        result = prepare(root, env, service, recovered, 'source-recovery')
        assert result.returncode == 0, result.stderr
        process = start(service_config=recovered)
        assert client('stop')[0] == 0 and process.wait(timeout=5) == 0
        prior[:] = captures
        service = recovered
    archive = root.with_name(root.name+'-backup')
    saved = command(root, env, 'backup', 'orze.service.backup', 'create',
                    '--service-config', service, '--destination', archive)
    assert saved.returncode == 0, saved.stderr
    digest = json.loads(saved.stdout)['manifest_sha256']
    scope = rows(root, 'SELECT * FROM cpu_action_scopes')
    original = {p.name: p.read_bytes() for p in service.with_name(service.name+'.host.lock').iterdir() if p.is_file()}
    env.clear(); env.update(destination_env)
    target = root/'upgraded.json'
    result = command(root, env, 'prepare-upgrade', 'orze.service.upgrade', '--source-service-config', service,
        '--service-config', target, '--request-id', 'runtime-upgrade', '--backup', archive, '--manifest-sha256', digest)
    assert result.returncode == 0, result.stderr
    assert json.loads(target.read_bytes())['runtime_packages'] != json.loads(service.read_bytes())['runtime_packages']
    idea(root, 'idea-0002', 'print("new runtime action")')
    upgraded = start(service_config=target)
    wait(lambda: rows(root, 'SELECT state FROM cpu_action_reservations') == [('SETTLED',), ('SETTLED',)])
    assert rows(root, 'SELECT * FROM cpu_action_scopes') == scope
    assert client('stop')[0] == 0 and upgraded.wait(timeout=5) == 0
    assert {p.name: p.read_bytes() for p in service.with_name(service.name+'.host.lock').iterdir() if p.is_file()} == original
    prior[:] = captures
    proof = command(root, env, 'upgraded-closed', 'orze.service.closed_state', '--service-config', target)
    assert proof.returncode == 0, proof.stderr
    # A later ordinary recovery must retain both sides of the runtime boundary.
    final = root/'after-upgrade.json'
    result = prepare(root, env, target, final, 'after-upgrade')
    assert result.returncode == 0, result.stderr
    idea(root, 'idea-0003', 'raise AssertionError("old spend must not be refunded")')
    exhausted = start(service_config=final)
    wait(lambda: captures and select.select([captures[-1]], [], [], 0)[0])
    assert rows(root, 'SELECT COUNT(*) FROM execution_attempts') == [(2,)]
    last = json.loads(rows(root, 'SELECT record_json FROM cpu_action_decisions ORDER BY rowid DESC LIMIT 1')[0][0])
    assert last['kind'] == 'Stop' and last['reason'] == 'wall_envelope_exhausted'
    assert client('stop')[0] == 0 and exhausted.wait(timeout=5) == 0
    assert len(births) == 3 + previous_handoff
    assert all(select.select([fd], [], [], 0)[0] for fd in captures)
    generations = rows(root, 'SELECT generation FROM controller_instances ORDER BY generation')
    assert generations == [(i,) for i in range(3 + previous_handoff)]
    (root/'upgrade-result.json').write_text(json.dumps({'generations': generations, 'source_runtime': str(frozen),
        'source_service': str(service), 'destination_service': str(target), 'backup_sha256': digest,
        'reservations': rows(root, 'SELECT * FROM cpu_action_reservations'), 'scope': scope,
        'exhausted_decision': last}, sort_keys=True, indent=2))
