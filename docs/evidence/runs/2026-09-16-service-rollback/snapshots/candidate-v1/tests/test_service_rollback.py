"""A retained upgrade-aware runtime resumes after a newer runtime did work."""
import hashlib
import json
import os
from pathlib import Path
import select

import pytest
import yaml

from test_service_upgrade import runtimes, cpu_controller, hosted_backup, command, _prepare_upgrade
from test_service_host_product import idea
from test_cpu_controller_stop_product import rows
from test_service_recovery import prepare


def worked_upgrade(runtimes, recovered=False):
    hosted, destination, frozen = runtimes
    root, config, cfg, env, service, start, client, wait, prior, captures, births, before_birth = hosted
    cfg['execution']['wall_budget_seconds'] = 180
    config.write_text(yaml.safe_dump(cfg))
    original_config = config.read_bytes()
    target, old_backup, old_env = _prepare_upgrade(runtimes)
    idea(root, 'idea-0002', 'print("new runtime result retained through rollback")')
    new = start(service_config=target)
    wait(lambda: rows(root, 'SELECT state FROM cpu_action_reservations') == [('SETTLED',), ('SETTLED',)])
    assert client('stop')[0] == 0 and new.wait(timeout=5) == 0
    prior[:] = captures
    if recovered:
        followup = root / 'new-recovered.json'
        result = prepare(root, env, target, followup, 'new-before-rollback')
        assert result.returncode == 0, result.stderr
        process = start(service_config=followup)
        assert client('stop')[0] == 0 and process.wait(timeout=5) == 0
        target = followup
        prior[:] = captures
    current_backup = root.with_name(root.name + '-worked-backup')
    result = command(root, env, 'worked-backup', 'orze.service.backup', 'create',
                     '--service-config', target, '--destination', current_backup)
    assert result.returncode == 0, result.stderr
    digest = json.loads(result.stdout)['manifest_sha256']
    retained = {table: rows(root, 'SELECT * FROM ' + table) for table in
                ('execution_attempts', 'cpu_action_reservations', 'cpu_action_scopes')}
    authority = {p.name: p.read_bytes() for p in target.with_name(target.name + '.host.lock').iterdir() if p.is_file()}
    old_cli = hashlib.sha256((frozen / 'cli.py').read_bytes()).hexdigest()
    new_cli = hashlib.sha256((Path(__file__).resolve().parents[1] / 'src/orze/cli.py').read_bytes()).hexdigest()
    if os.environ.get('ORZE_TEST_UPGRADE_SOURCE'):
        assert old_cli != new_cli
    return target, old_backup, old_env, current_backup, digest, retained, authority, original_config, old_cli, new_cli


def prepare_reverse(root, env, source, target, backup, digest, key, label):
    return command(root, env, label, 'orze.service.upgrade', '--source-service-config', source,
        '--service-config', target, '--request-id', key, '--backup', backup, '--manifest-sha256', digest)


@pytest.mark.parametrize('recovered', [False, True], ids=['upgraded-source', 'recovered-source'])
def test_return_to_fixed_compatible_runtime_preserves_new_work_and_budget(runtimes, recovered):
    hosted, destination, frozen = runtimes
    root, config, cfg, env, service, start, client, wait, prior, captures, births, before_birth = hosted
    source, old_backup, old_env, backup, digest, retained, authority, original_config, old_cli, new_cli = worked_upgrade(runtimes, recovered)
    env.clear(); env.update(old_env)
    ordinary = prepare(root, env, source, root / 'ordinary-rejected.json', 'ordinary-rollback-rejected')
    assert ordinary.returncode == 75 and not (root / 'ordinary-rejected.json').exists()
    old_digest = json.loads((root / 'backup.stdout').read_bytes())['manifest_sha256']
    stale = prepare_reverse(root, env, source, root / 'stale-backup-rejected.json', old_backup, old_digest,
                            'stale-rollback', 'stale-backup')
    assert stale.returncode == 75 and not (root / 'stale-backup-rejected.json').exists()
    target = root / 'rolled-back.json'
    prepared = prepare_reverse(root, env, source, target, backup, digest, 'rollback-001', 'prepare-rollback')
    assert prepared.returncode == 0, prepared.stderr
    assert json.loads(target.read_bytes())['runtime_packages'] == json.loads(service.read_bytes())['runtime_packages']
    idea(root, 'idea-0003', 'print("old compatible runtime after new work")')
    process = start(service_config=target)
    wait(lambda: rows(root, 'SELECT state FROM cpu_action_reservations') == [('SETTLED',)] * 3)
    assert client('stop')[0] == 0 and process.wait(timeout=5) == 0
    assert config.read_bytes() == original_config
    assert {p.name: p.read_bytes() for p in source.with_name(source.name + '.host.lock').iterdir() if p.is_file()} == authority
    for table, original in retained.items():
        assert rows(root, 'SELECT * FROM ' + table)[:len(original)] == original
    prior[:] = captures
    exhausted_service = root / 'after-rollback.json'
    result = prepare(root, env, target, exhausted_service, 'after-rollback')
    assert result.returncode == 0, result.stderr
    idea(root, 'idea-0004', 'raise AssertionError("new-runtime spend was refunded")')
    final = start(service_config=exhausted_service)
    wait(lambda: captures and select.select([captures[-1]], [], [], 0)[0])
    assert rows(root, 'SELECT COUNT(*) FROM execution_attempts') == [(3,)]
    decision = json.loads(rows(root, 'SELECT record_json FROM cpu_action_decisions ORDER BY rowid DESC LIMIT 1')[0][0])
    assert decision['kind'] == 'Stop' and decision['reason'] == 'wall_envelope_exhausted'
    assert client('stop')[0] == 0 and final.wait(timeout=5) == 0
    assert all(select.select([fd], [], [], 0)[0] for fd in captures)
    assert len(births) == 4 + recovered
    assert rows(root, 'SELECT generation FROM controller_instances ORDER BY generation') == [(i,) for i in range(len(births))]
    (root / 'rollback-result.json').write_text(json.dumps({'source': str(source), 'returned': str(target),
        'old_cli_sha256': old_cli, 'new_cli_sha256': new_cli, 'backup_sha256': digest,
        'pre_rollback_records': retained, 'reservations': rows(root, 'SELECT * FROM cpu_action_reservations'),
        'decision': decision, 'births': len(births), 'recovered_new_source': recovered}, sort_keys=True, indent=2))


def test_two_reverse_transitions_cannot_duplicate_new_work(runtimes):
    hosted, destination, frozen = runtimes
    root, config, cfg, env, service, start, client, wait, prior, captures, births, before_birth = hosted
    source, old_backup, old_env, backup, digest, retained, authority, original_config, old_cli, new_cli = worked_upgrade(runtimes)
    env.clear(); env.update(old_env)
    targets = [root / ('rollback-' + str(i) + '.json') for i in range(2)]
    for i, target in enumerate(targets):
        result = prepare_reverse(root, env, source, target, backup, digest, 'reverse-' + str(i), 'prepare-reverse-' + str(i))
        assert result.returncode == 0, result.stderr
    with (root / 'sitecustomize.py').open('a') as stream:
        stream.write('''
if 'orze.service.host' in getattr(sys,'orig_argv',()):
    from pathlib import Path
    import time
    from orze.service.upgrade import _UpgradeCoordinator
    original=_UpgradeCoordinator.prepare_source
    def together(self):
        original(self)
        Path('reverse-ready-'+str(os.getpid())).touch()
        deadline=time.monotonic()+20
        while len(list(Path('.').glob('reverse-ready-*')))<2:
            assert time.monotonic()<deadline
            time.sleep(.01)
    _UpgradeCoordinator.prepare_source=together
''')
    idea(root, 'idea-0003', 'print("one reverse-transition action")')
    contenders = [start(wait_ready=False, service_config=p) for p in targets]
    wait(lambda: any(p.poll() == 75 for p in contenders))
    wait(lambda: rows(root, 'SELECT state FROM cpu_action_reservations') == [('SETTLED',)] * 3)
    winner = next(i for i, p in enumerate(contenders) if p.poll() is None)
    assert client('stop', service_config=targets[winner])[0] == 0 and contenders[winner].wait(timeout=5) == 0
    assert sorted(p.returncode for p in contenders) == [0, 75]
    assert len(list(root.glob('reverse-ready-*'))) == 2
    assert len(births) == 3 and rows(root, 'SELECT state FROM controller_handoffs') == [('STARTED',), ('STARTED',)]
    assert config.read_bytes() == original_config and all(select.select([fd], [], [], 0)[0] for fd in captures)
    for table, original in retained.items():
        assert rows(root, 'SELECT * FROM ' + table)[:len(original)] == original
    (root / 'rollback-race-result.json').write_text(json.dumps({'returncodes': [p.returncode for p in contenders],
        'winner': str(targets[winner]), 'births': len(births), 'old_cli_sha256': old_cli, 'new_cli_sha256': new_cli,
        'pre_rollback_records': retained, 'reservations': rows(root, 'SELECT * FROM cpu_action_reservations')}, sort_keys=True, indent=2))
