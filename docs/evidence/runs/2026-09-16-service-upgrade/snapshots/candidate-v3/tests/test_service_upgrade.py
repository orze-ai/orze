"""Two separate runtime trees, real closed CPU services and unchanged budgets."""
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import time

import pytest

from test_cpu_controller_stop_product import cpu_controller, rows
from test_service_backup import hosted_backup
from test_service_host_product import idea
from test_service_recovery import prepare


def command(root, env, label, module, *args):
    started = time.monotonic()
    result = subprocess.run([sys.executable, '-m', module, *map(str, args)], cwd=root, env=env,
                            capture_output=True, text=True, timeout=60)
    finished = time.monotonic()
    (root/(label+'.stdout')).write_text(result.stdout)
    (root/(label+'.stderr')).write_text(result.stderr)
    (root/(label+'.command.json')).write_text(json.dumps({'command': result.args, 'exit_code': result.returncode,
        'pythonpath': env['PYTHONPATH'], 'started_monotonic': started, 'finished_monotonic': finished,
        'wall_seconds': finished-started}, sort_keys=True, indent=2))
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
        expected = root/'expected-source-runtime-files.json'
        expected = json.loads(expected.read_bytes()) if expected.exists() else inventory
        actual = {str(p.relative_to(frozen)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(frozen.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
        (root/'source-runtime-files-after.json').write_text(json.dumps(actual, sort_keys=True, indent=2))
        assert expected == actual


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
    ordinary = prepare(root, env, service, root/'not-an-ordinary-recovery.json', 'wrong-runtime')
    assert ordinary.returncode == 75 and not (root/'not-an-ordinary-recovery.json').exists()
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


def _prepare_upgrade(runtimes):
    hosted, destination, frozen = runtimes
    root, config, cfg, env, service, start, client, wait, prior, captures, births, before_birth = hosted
    original_env = dict(env)
    idea(root, 'idea-0001', 'print("old runtime settled before transition")')
    process = start()
    wait(lambda: rows(root, 'SELECT state FROM cpu_action_reservations') == [('SETTLED',)])
    assert client('stop')[0] == 0 and process.wait(timeout=5) == 0
    prior[:] = captures
    archive = root.with_name(root.name+'-backup')
    saved = command(root, env, 'backup', 'orze.service.backup', 'create',
                    '--service-config', service, '--destination', archive)
    assert saved.returncode == 0, saved.stderr
    digest = json.loads(saved.stdout)['manifest_sha256']
    env.clear(); env.update(destination)
    target = root/'upgraded.json'
    prepared = command(root, env, 'prepare-upgrade', 'orze.service.upgrade', '--source-service-config', service,
        '--service-config', target, '--request-id', 'runtime-upgrade', '--backup', archive, '--manifest-sha256', digest)
    assert prepared.returncode == 0, prepared.stderr
    return target, archive, original_env


@pytest.mark.parametrize('fault', ['backup', 'source-runtime', 'source-closure', 'source-schema',
                                 'stop', 'adapter', 'destination-runtime'])
def test_upgrade_rechecks_inputs_before_any_successor(runtimes, fault):
    import sqlite3
    hosted, destination, frozen = runtimes
    root, config, cfg, env, service, start, client, wait, prior, captures, births, before_birth = hosted
    target, archive, original_env = _prepare_upgrade(runtimes)
    if fault == 'backup':
        manifest = json.loads((archive/'manifest.json').read_bytes())
        entry = next(row for row in manifest['entries'] if row['kind'] == 'database')
        (archive/'objects'/entry['object']).write_bytes(b'changed backup payload')
    elif fault == 'source-runtime':
        path = frozen/'cli.py'
        path.write_bytes(path.read_bytes()+b'\n# changed after upgrade preparation\n')
        expected = json.loads((root/'source-runtime-files.json').read_bytes())
        expected['cli.py'] = hashlib.sha256(path.read_bytes()).hexdigest()
        (root/'expected-source-runtime-files.json').write_text(json.dumps(expected, sort_keys=True))
    elif fault == 'source-closure':
        path = service.with_name(service.name+'.host.lock')/'closed.json'
        path.write_text('{}')
    elif fault == 'source-schema':
        with sqlite3.connect(root/'lake.db') as conn:
            conn.execute('ALTER TABLE controller_scope_heads ADD COLUMN unsupported INTEGER')
    elif fault == 'stop':
        (root/'results/.orze_disabled').touch()
    else:
        value = json.loads(target.read_bytes())
        if fault == 'adapter':
            value['upgrade']['compatibility'] = 'unsupported-state-format'
        else:
            value['runtime_packages'][0]['sha256'] = '0'*64
        target.write_text(json.dumps(value, sort_keys=True))
    rejected = start(wait_ready=False, service_config=target)
    assert rejected.wait(timeout=15) == 75
    assert len(births) == 1 and rows(root, 'SELECT COUNT(*) FROM execution_attempts') == [(1,)]
    assert rows(root, 'SELECT generation FROM controller_instances') == [(0,)]
    assert not target.with_name(target.name+'.host.lock').joinpath('ready.json').exists()
    assert all(select.select([fd], [], [], 0)[0] for fd in captures)


@pytest.mark.parametrize('contender', ['old-recovery', 'second-upgrade'])
def test_upgrade_and_competing_host_have_one_grant_and_one_cpu_action(runtimes, contender):
    hosted, destination, frozen = runtimes
    root, config, cfg, env, service, start, client, wait, prior, captures, births, before_birth = hosted
    target, archive, original_env = _prepare_upgrade(runtimes)
    competitor = root/'competitor.json'
    if contender == 'old-recovery':
        competing_env = original_env
        prepared = prepare(root, original_env, service, competitor, 'competitor')
    else:
        competing_env = destination
        digest = json.loads((target).read_bytes())['upgrade']['manifest_sha256']
        prepared = command(root, destination, 'prepare-competitor', 'orze.service.upgrade',
            '--source-service-config', service, '--service-config', competitor, '--request-id', 'competitor',
            '--backup', archive, '--manifest-sha256', digest)
    assert prepared.returncode == 0, prepared.stderr
    with (root/'sitecustomize.py').open('a') as stream:
        stream.write('''
if 'orze.service.host' in getattr(sys,'orig_argv',()):
    from pathlib import Path
    import time
    from orze.service.recovery import _RecoveryCoordinator
    classes=[_RecoveryCoordinator]
    try:
        from orze.service.upgrade import _UpgradeCoordinator
        classes.append(_UpgradeCoordinator)
    except ImportError:pass
    for cls in classes:
        original=cls.prepare_source
        def together(self,original=original):
            original(self)
            Path('transition-ready-'+str(os.getpid())).touch()
            deadline=time.monotonic()+20
            while len(list(Path('.').glob('transition-ready-*')))<2:
                assert time.monotonic()<deadline
                time.sleep(.01)
        cls.prepare_source=together
''')
    idea(root, 'idea-0002', 'print("only one post-transition execution")')
    env.clear(); env.update(competing_env)
    first = start(wait_ready=False, service_config=competitor)
    env.clear(); env.update(destination)
    second = start(wait_ready=False, service_config=target)
    wait(lambda: any(p.poll() == 75 for p in (first, second)))
    wait(lambda: rows(root, 'SELECT state FROM cpu_action_reservations') == [('SETTLED',), ('SETTLED',)])
    assert len(list(root.glob('transition-ready-*'))) == 2
    assert len(births) == 2 and rows(root, 'SELECT state FROM controller_handoffs') == [('STARTED',)]
    winner, selected, selected_env = ((first, competitor, competing_env) if first.poll() is None
                                     else (second, target, destination))
    env.clear(); env.update(selected_env)
    assert client('stop', service_config=selected)[0] == 0 and winner.wait(timeout=5) == 0
    assert all(select.select([fd], [], [], 0)[0] for fd in captures)
    (root/'race-result.json').write_text(json.dumps({'contender': contender, 'winner': str(selected),
        'returncodes': [first.returncode, second.returncode], 'births': len(births),
        'grant': rows(root, 'SELECT request_id,state FROM controller_handoffs'),
        'reservation_count': len(rows(root, 'SELECT * FROM cpu_action_reservations'))}, sort_keys=True, indent=2))


def test_public_cli_forwards_explicit_upgrade_identity(monkeypatch):
    from orze import cli
    from orze.service import upgrade
    arguments = ['--source-service-config', '/source.json', '--service-config', '/destination.json',
                 '--request-id', 'upgrade-1', '--backup', '/backup', '--manifest-sha256', 'a'*64]
    calls = []
    monkeypatch.setattr(upgrade, 'main', lambda args: calls.append(args) or 75)
    monkeypatch.setattr(sys, 'argv', ['orze', 'service', 'upgrade', *arguments])
    assert cli.main() == 75 and calls == [arguments]
