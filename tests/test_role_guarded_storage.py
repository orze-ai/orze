"""Guarded role storage preserves strong closure and never takes stale locks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import pytest

from orze.core.fs import _fs_lock, _fs_unlock
from orze.engine import role_storage as storage, role_supervision as supervision
from orze.engine.process import reconcile_orphaned_role_receipts
from orze.engine.storage_preflight import (
    StoragePreflightError, require_role_storage, require_deployment_storage,
)
from test_role_supervision import project, _prepare, _role


@pytest.fixture
def guarded(project):
    p = project
    _fs_unlock(p.lock)
    lease = storage.acquire_guarded_role(p.lock)
    p.lease = lease
    def begin():
        owner = supervision.begin_role_launch(p.metadata, storage_lease=lease)
        p.owners.append(owner)
        return owner
    p.begin = begin
    yield p
    storage._LEASES.pop(id(lease), None)


def test_actual_worker_closes_and_releases_then_next_owner_can_acquire(guarded, monkeypatch):
    p = guarded
    monkeypatch.setattr(supervision, 'rename_no_replace', lambda *a: pytest.fail('guarded rename'))
    owner = p.begin()
    child = _prepare(p, owner)
    _role(p, owner)
    owner.start()
    assert child.wait(timeout=3) == 0
    closure = owner.require_closed(0)
    assert closure['wait_proof'] == 'ECHILD_WALL'
    assert owner.release(outcome='ok', exit_code=0)
    assert not p.lock.exists()
    assert p.marker.read_text() == 'done'
    assert storage.marker_path(p.lock).is_file()
    assert not storage.transition_path(p.lock).exists()
    next_lease = storage.acquire_guarded_role(p.lock)
    assert next_lease is not None and next_lease.nonce != p.lease.nonce
    storage.cancel_guarded_role(next_lease)
    assert owner.release(outcome='ok', exit_code=0)


def test_unbound_cancel_and_never_executed_owner_release(guarded):
    p = guarded
    storage.cancel_guarded_role(p.lease)
    lease = storage.acquire_guarded_role(p.lock)
    owner = supervision.begin_role_launch(p.metadata, storage_lease=lease)
    p.owners.append(owner)
    with pytest.raises(storage.RoleStorageHOLD, match='after_bind'):
        storage.cancel_guarded_role(lease)
    assert owner.release()
    assert not p.marker.exists()


def test_existing_guarded_lock_has_no_age_takeover_or_generic_unlock(guarded):
    p = guarded
    assert storage.acquire_guarded_role(p.lock) is None
    assert _fs_lock(p.lock, stale_seconds=-1) is False
    before = (p.lock / 'lock.json').read_bytes()
    _fs_unlock(p.lock)
    assert (p.lock / 'lock.json').read_bytes() == before
    with pytest.raises(supervision.RoleSupervisionHOLD):
        supervision.begin_role_launch(p.metadata)


def test_running_worker_cannot_release(guarded):
    p = guarded
    p.command = [sys.executable, '-c', 'import time; time.sleep(30)']
    from orze.engine.supervisor_worker import canonical
    p.metadata['command_sha256'] = hashlib.sha256(canonical(p.command)).hexdigest()
    owner = p.begin()
    child = _prepare(p, owner)
    owner.start()
    with pytest.raises(supervision.RoleSupervisionHOLD):
        owner.release(outcome='ok', exit_code=0)
    assert child.poll() is None and p.lock.is_dir()
    assert storage.acquire_guarded_role(p.lock) is None


def test_storage_release_helper_cannot_bypass_owned_closure(guarded):
    p = guarded
    p.command = [sys.executable, '-c', 'import time; time.sleep(30)']
    from orze.engine.supervisor_worker import canonical
    p.metadata['command_sha256'] = hashlib.sha256(canonical(p.command)).hexdigest()
    owner = p.begin()
    child = _prepare(p, owner)
    owner.start()
    with pytest.raises((storage.RoleStorageHOLD, supervision.RoleSupervisionHOLD)):
        storage.release_guarded_role(p.lease, owner)
    assert child.poll() is None and p.lock.is_dir()


@pytest.mark.parametrize('changed', ['marker', 'lock', 'metadata', 'receipt', 'parent'])
def test_changed_identity_or_bytes_never_releases(guarded, changed):
    p = guarded
    owner = p.begin()
    if changed == 'marker':
        storage.marker_path(p.lock).write_text('different')
    elif changed == 'lock':
        p.lock.rename(p.lock.with_name('retired'))
        p.lock.mkdir()
        (p.lock / 'foreign').write_text('keep')
    elif changed == 'metadata':
        (p.lock / 'lock.json').write_text('{}')
    elif changed == 'receipt':
        (p.lock / 'role-process.json').write_text('{}')
    else:
        parent = p.lock.parent
        parent.rename(parent.with_name('retired-parent'))
        parent.mkdir()
        p.lock.mkdir()
        (p.lock / 'foreign').write_text('keep')
    with pytest.raises(supervision.RoleSupervisionHOLD):
        owner.release()
    assert p.lock.is_dir()
    if (p.lock / 'foreign').exists():
        assert (p.lock / 'foreign').read_text() == 'keep'


def test_unknown_file_retains_transition_for_startup_hold(guarded):
    p = guarded
    owner = p.begin()
    (p.lock / 'foreign').write_text('keep')
    with pytest.raises(supervision.RoleSupervisionHOLD):
        owner.release()
    assert (p.lock / 'foreign').read_text() == 'keep'
    assert storage.transition_path(p.lock).is_dir()
    assert storage.acquire_guarded_role(p.lock) is None
    report = reconcile_orphaned_role_receipts(p.root / '.orze')
    assert 'role_storage_transition_unconfirmed:engineer' in report['errors']
    assert 'role_supervision_recovery_required:engineer' in report['errors']


def test_sync_failure_after_file_removal_retains_guard_and_no_retry(guarded, monkeypatch):
    p = guarded
    owner = p.begin()
    real_sync = os.fsync
    def sync(fd):
        if not (p.lock / 'lock.json').exists():
            raise OSError('injected release sync failure')
        return real_sync(fd)
    monkeypatch.setattr(storage.os, 'fsync', sync)
    with pytest.raises(supervision.RoleSupervisionHOLD):
        owner.release()
    assert storage.transition_path(p.lock).exists()
    assert (p.lock / 'role-process.json').exists()
    assert storage.acquire_guarded_role(p.lock) is None
    with pytest.raises(supervision.RoleSupervisionHOLD):
        owner.release()


def test_exclusive_marker_collision_and_existing_legacy_owner_are_preserved(tmp_path):
    lock = tmp_path / 'engineer'
    assert _fs_lock(lock)
    assert storage.acquire_guarded_role(lock) is None
    assert not storage.marker_path(lock).exists()
    _fs_unlock(lock)
    storage.marker_path(lock).write_text('foreign')
    with pytest.raises(storage.RoleStorageHOLD):
        storage.acquire_guarded_role(lock)
    assert storage.marker_path(lock).read_text() == 'foreign'


def test_actual_cephfs_guarded_preflight_and_owned_release():
    parent = Path('/hot-data/fsx/workspace/erik/orze-p1-continuation-20260914/role-guarded')
    if not parent.is_dir():
        pytest.skip('isolated CephFS evidence root unavailable')
    root = Path(tempfile.mkdtemp(prefix='rg-ceph-', dir=parent))
    cfg = {'_orze_dir': str(root / '.orze')}
    role = {'mode':'script', 'script':'offline.py', 'storage_mode':'guarded'}
    require_role_storage(cfg, role)
    lease = storage.acquire_guarded_role(root / '.orze/locks/engineer')
    assert lease is not None
    storage.cancel_guarded_role(lease)
    assert not lease.lock_dir.exists()


def test_mixed_deployment_modes_check_each_protocol(tmp_path, monkeypatch):
    from orze.engine import storage_preflight as preflight
    seen = []
    monkeypatch.setattr(preflight, 'require_role_storage', lambda cfg, role: seen.append(role['storage_mode']))
    cfg = {'roles': {'a':{'mode':'script','script':'a','storage_mode':'guarded'},
                     'b':{'mode':'script','script':'b','storage_mode':'atomic'}}}
    require_deployment_storage(cfg, tmp_path)
    assert seen == ['guarded', 'atomic']


@pytest.mark.parametrize('where', ['local', 'ceph'])
def test_concurrent_processes_have_one_owner_and_no_stale_takeover(tmp_path, where):
    root = tmp_path
    if where == 'ceph':
        parent = Path('/hot-data/fsx/workspace/erik/orze-p1-continuation-20260914/role-guarded')
        if not parent.is_dir():
            pytest.skip('isolated CephFS evidence root unavailable')
        root = Path(tempfile.mkdtemp(prefix='rg-contend-', dir=parent))
    lock = root / 'engineer'
    initial = storage.acquire_guarded_role(lock)
    storage.cancel_guarded_role(initial)
    script = '''
import json,sys,time
from pathlib import Path
from orze.engine.role_storage import acquire_guarded_role,cancel_guarded_role
root,number=Path(sys.argv[1]),sys.argv[2]
deadline=time.monotonic()+10
while not (root/'start').exists():
    if time.monotonic()>deadline: raise TimeoutError('start')
    time.sleep(.005)
lease=acquire_guarded_role(root/'engineer')
(root/(number+'.json')).write_text(json.dumps({'owned':lease is not None}))
(root/(number+'.ready')).touch()
if lease is not None:
    while not (root/'release').exists():
        if time.monotonic()>deadline: raise TimeoutError('release')
        time.sleep(.005)
    cancel_guarded_role(lease)
'''
    children = [subprocess.Popen([sys.executable, '-c', script, str(root), str(i)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                for i in range(2)]
    try:
        (root / 'start').write_text('go')
        deadline = time.monotonic() + 8
        while not all((root / (str(i)+'.ready')).exists() for i in range(2)):
            assert time.monotonic() < deadline
            time.sleep(.01)
        results = [json.loads((root / (str(i)+'.json')).read_bytes()) for i in range(2)]
        assert sum(row['owned'] for row in results) == 1
        assert not _fs_lock(lock, stale_seconds=-1)
        assert storage.acquire_guarded_role(lock) is None
        (root / 'release').write_text('go')
        for child in children:
            out, err = child.communicate(timeout=10)
            assert child.returncode == 0, (out, err)
    finally:
        (root / 'release').touch()
        for child in children:
            child.communicate(timeout=10)
    next_lease = storage.acquire_guarded_role(lock)
    assert next_lease is not None
    storage.cancel_guarded_role(next_lease)


def test_final_directory_sync_failure_blocks_new_owner_and_is_visible_at_startup(guarded, monkeypatch):
    p = guarded
    owner = p.begin()
    sync = storage._sync
    def fail_after_remove(path):
        if path == p.lock.parent and not p.lock.exists():
            raise OSError('injected namespace sync failure')
        return sync(path)
    monkeypatch.setattr(storage, '_sync', fail_after_remove)
    with pytest.raises(supervision.RoleSupervisionHOLD):
        owner.release()
    assert not p.lock.exists()
    assert storage.transition_path(p.lock).is_dir()
    assert storage.acquire_guarded_role(p.lock) is None
    assert 'role_storage_transition_unconfirmed:engineer' in reconcile_orphaned_role_receipts(p.root / '.orze')['errors']


def test_same_bytes_marker_replacement_invalidates_lease(guarded):
    p = guarded
    path = storage.marker_path(p.lock)
    replacement = path.with_name('replacement')
    replacement.write_bytes(path.read_bytes())
    replacement.replace(path)
    with pytest.raises(storage.RoleStorageHOLD):
        storage.cancel_guarded_role(p.lease)
    assert p.lock.is_dir()


@pytest.mark.parametrize('first', ['source', 'role'])
def test_declared_idle_namespaces_cannot_be_cross_acquired(tmp_path, first):
    from orze.core import idea_source_lock as source
    lock = tmp_path / 'shared-name'
    if first == 'source':
        lease = source._acquire(lock)
        assert lease is not None and source._release(lease)
        assert storage.acquire_guarded_role(lock) is None
        assert not storage.marker_path(lock).exists()
        again = source._acquire(lock)
        assert again is not None and source._release(again)
    else:
        lease = storage.acquire_guarded_role(lock)
        storage.cancel_guarded_role(lease)
        assert source._acquire(lock) is None
        assert not source._marker_path(lock).exists()
        again = storage.acquire_guarded_role(lock)
        assert again is not None
        storage.cancel_guarded_role(again)


@pytest.mark.parametrize('owner_kind', ['source', 'role'])
def test_late_opposite_namespace_declaration_invalidates_current_owner(tmp_path, owner_kind):
    from orze.core import idea_source_lock as source
    lock = tmp_path / 'shared-name'
    if owner_kind == 'source':
        lease = source._acquire(lock)
        before = (lock / 'lock.json').read_bytes()
        storage.marker_path(lock).write_text('uncertain role declaration')
        assert source.idea_source_lock_owned(lease) is False
        assert source._release(lease) is False
    else:
        lease = storage.acquire_guarded_role(lock)
        before = (lock / 'lock.json').read_bytes()
        source._marker_path(lock).write_bytes(source._MARKER_BYTES)
        with pytest.raises(storage.RoleStorageHOLD, match='source_namespace'):
            storage.cancel_guarded_role(lease)
        storage._LEASES.pop(id(lease))
    assert (lock / 'lock.json').read_bytes() == before
