"""Real closed CPU service backup, inactive restoration and retained budget."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from orze.engine.controller_control import ControllerHOLD
from orze.service import backup
from test_cpu_controller_stop_product import cpu_controller, rows
from test_service_host_product import _hosted, idea
from test_service_recovery import prepare


@pytest.fixture
def hosted_backup(cpu_controller):
    generator = _hosted(cpu_controller)
    hosted = next(generator)
    root = hosted[0]
    # The test observer is outside the project being backed up. This changes
    # only the fixture's own birth channel, never the service's proof/closure.
    outside = root.with_name(root.name+'-birth.sock')
    (root/'birth.sock').rename(outside)
    site = root/'sitecustomize.py'
    site.write_text(site.read_text().replace("str(__import__('pathlib').Path.cwd() / 'birth.sock')", repr(str(outside))))
    try:
        yield hosted
    finally:
        try:
            next(generator)
        except StopIteration:
            pass
        outside.unlink()


@pytest.fixture
def closed(hosted_backup):
    root, config, cfg, env, service, start, client, wait, prior, captures, births, before_birth = hosted_backup
    idea(root, 'idea-0001', 'print("retained backup result")')
    process = start()
    wait(lambda: rows(root, 'SELECT state FROM cpu_action_reservations') == [('SETTLED',)])
    assert client('stop')[0] == 0 and process.wait(timeout=5) == 0
    prior.extend(captures)
    return hosted_backup


def create(closed, name='backup', **options):
    root, _, _, _, service, *_ = closed
    return backup.capture(service, root.with_name(root.name+'-'+name), **options)


def dump(path):
    with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True) as conn:
        return list(conn.iterdump())


def test_actual_closed_service_roundtrip_preserves_all_sql_and_original_recovery(closed):
    root, config, cfg, env, service, start, client, wait, prior, captures, births, before_birth = closed
    before = dump(root/'lake.db')
    (root/'note-link').symlink_to('ideas.md')
    report = create(closed)
    destination = Path(report['backup'])
    assert report['activation'] is False
    assert backup.verify(destination, report['manifest_sha256'])['activation'] is False
    restored = root.with_name(root.name+'-restored')
    result = backup.restore(destination, report['manifest_sha256'], restored)
    assert result['activation'] is False
    manifest = json.loads((destination/'manifest.json').read_bytes())
    project = restored/next(row['name'] for row in manifest['roots'] if row['source'] == str(root))
    assert dump(project/'lake.db') == before == dump(root/'lake.db')
    assert (project/'note-link').is_symlink() and os.readlink(project/'note-link') == 'ideas.md'
    for row in manifest['entries']:
        if row['kind'] in {'file','database'}:
            actual = project/row['path']
            assert hashlib.sha256(actual.read_bytes()).hexdigest() == row['sha256']
    # A copied historical service file cannot become a runnable new owner.
    rejected = subprocess.run([sys.executable, '-m', 'orze.service.host', '--service-config', str(project/'service.json')],
        cwd=project, env=env, capture_output=True, text=True, timeout=15)
    assert rejected.returncode == 75 and len(births) == 1
    source = root/'after-backup.json'
    prepared = prepare(root, env, service, source, 'after-backup')
    assert prepared.returncode == 0, prepared.stderr
    original_budget = rows(root, 'SELECT * FROM cpu_action_scopes')
    idea(root, 'idea-0002', 'print("original recovered without refund")')
    process = start(service_config=source)
    wait(lambda: rows(root, 'SELECT COUNT(*) FROM cpu_action_reservations WHERE state="SETTLED"') == [(2,)])
    assert rows(root, 'SELECT * FROM cpu_action_scopes') == original_budget
    assert client('stop')[0] == 0 and process.wait(timeout=5) == 0
    assert dump(project/'lake.db') == before
    assert rows(root, 'SELECT generation FROM controller_instances ORDER BY generation') == [(0,), (1,)]


def test_live_host_refused_before_destination_creation(hosted_backup):
    c = hosted_backup
    process = c[5]()
    target = c[0].with_name(c[0].name+'-live-backup')
    with pytest.raises((ControllerHOLD, FileNotFoundError)):
        backup.capture(c[4], target)
    assert not target.exists() and process.poll() is None


@pytest.mark.parametrize('mutation', ['ack','budget','closure','stop'])
def test_unconfirmed_source_is_not_backed_up(closed, mutation):
    root = closed[0]
    if mutation in {'ack','budget'}:
        with sqlite3.connect(root/'lake.db') as conn:
            conn.execute("UPDATE controller_sessions SET ack_json='{}'" if mutation == 'ack' else
                         "UPDATE cpu_action_reservations SET state='BOUND'")
    elif mutation == 'closure':
        path = closed[4].with_name('service.json.host.lock')/'closed.json'
        path.write_text('{}')
    else:
        (root/'results/.orze_disabled').touch()
    with pytest.raises((ControllerHOLD, ValueError)):
        create(closed)
    assert not root.with_name(root.name+'-backup').exists()


@pytest.mark.parametrize('kind', ['source-file','source-link','sync'])
def test_interrupted_or_changed_capture_has_no_completion(closed, monkeypatch, kind):
    root = closed[0]
    file = root/'immutable-note'
    file.write_text('before')
    if kind == 'source-link':
        (root/'note-link').symlink_to('immutable-note')
    actual = backup._stream
    changed = []
    def stream(source, *args, **kwargs):
        result = actual(source, *args, **kwargs)
        if not changed and Path(source) == file:
            changed.append(True)
            if kind == 'source-file':
                file.write_text('after')
            elif kind == 'source-link':
                (root/'note-link').unlink()
                (root/'note-link').symlink_to('different-target')
            else:
                raise OSError('injected copy synchronization uncertainty')
        return result
    monkeypatch.setattr(backup, '_stream', stream)
    with pytest.raises((ControllerHOLD, OSError)):
        create(closed)
    target = root.with_name(root.name+'-backup')
    assert changed and target.exists() and not (target/'complete.json').exists()
    assert (root/'lake.db').exists() and file.exists()


def test_capture_capacity_and_destination_collision_preserve_existing_data(closed):
    root = closed[0]
    with pytest.raises(ControllerHOLD, match='byte_limit'):
        create(closed, 'too-small', max_bytes=1)
    assert not root.with_name(root.name+'-too-small').exists()
    target = root.with_name(root.name+'-backup')
    target.mkdir();(target/'keep').write_text('operator bytes')
    with pytest.raises(FileExistsError):
        create(closed)
    assert (target/'keep').read_text() == 'operator bytes' and not (target/'intent.json').exists()


@pytest.fixture
def captured(closed):
    result = create(closed)
    return Path(result['backup']), result['manifest_sha256'], closed


@pytest.mark.parametrize('mutation', ['payload','missing-complete','extra-object','object-symlink'])
def test_verification_never_accepts_incomplete_or_changed_payload(captured, mutation):
    path, digest, closed = captured
    manifest = json.loads((path/'manifest.json').read_bytes())
    row = next(row for row in manifest['entries'] if row['kind'] == 'file')
    payload = path/'objects'/row['object']
    if mutation == 'payload':
        payload.write_text('replaced')
    elif mutation == 'missing-complete':
        (path/'complete.json').unlink()
    elif mutation == 'extra-object':
        (path/'objects/999999').write_text('unknown')
    else:
        payload.unlink();payload.symlink_to(closed[0]/'ideas.md')
    with pytest.raises((ControllerHOLD, OSError)):
        backup.verify(path, digest)
    target = path.with_name(path.name+'-not-restored')
    with pytest.raises((ControllerHOLD, OSError)):
        backup.restore(path, digest, target)
    assert not target.exists()


@pytest.mark.parametrize('mutation', ['traversal','duplicate','version','link-parent'])
def test_even_explicitly_pinned_invalid_manifest_cannot_restore(captured, mutation):
    path, digest, closed = captured
    manifest = json.loads((path/'manifest.json').read_bytes())
    if mutation == 'traversal':
        manifest['entries'][1]['path'] = '../foreign'
    elif mutation == 'duplicate':
        manifest['entries'].append(copy.deepcopy(manifest['entries'][0]))
    elif mutation == 'version':
        manifest['version'] = True
    else:
        row = manifest['entries'][0]
        row['kind'] = 'symlink';row['target'] = '/tmp'
    raw = backup._encoded(manifest);digest = hashlib.sha256(raw).hexdigest()
    (path/'manifest.json').write_bytes(raw)
    (path/'complete.json').write_bytes(backup._encoded({'version':1,'manifest_sha256':digest}))
    with pytest.raises(ControllerHOLD):
        backup.verify(path, digest)


def test_late_restored_file_change_cannot_receive_completion(captured, monkeypatch):
    path, digest, closed = captured
    target = path.with_name(path.name+'-late-change')
    manifest = json.loads((path/'manifest.json').read_bytes())
    row = next(row for row in manifest['entries'] if row['kind'] == 'file')
    verify = backup._verify
    changed = []
    def change_after_source_recheck(*args, **kwargs):
        result = verify(*args, **kwargs)
        if target.exists() and not changed:
            (target/row['root']/row['path']).write_text('late corrupt restored data')
            changed.append(True)
        return result
    monkeypatch.setattr(backup, '_verify', change_after_source_recheck)
    with pytest.raises(ControllerHOLD):
        backup.restore(path, digest, target)
    assert changed and not (target/'restored.json').exists()


@pytest.mark.parametrize('operation', ['create','verify','restore'])
def test_public_cli_routes_backup_arguments(monkeypatch, operation):
    from orze import cli
    args = [operation, '--service-config', '/source', '--destination', '/backup', '--max-bytes', '5000'] if operation == 'create' else [
        operation, '--backup', '/backup', '--manifest-sha256', 'a'*64]
    if operation == 'restore':
        args += ['--destination', '/restored']
    seen = []
    monkeypatch.setattr(backup, 'main', lambda options: seen.append(options) or 75)
    monkeypatch.setattr(sys, 'argv', ['orze','service','backup',*args])
    assert cli.main() == 75
    assert seen == [args]
