"""Positive no-rename GC plus authority and uncertain-side-effect boundaries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

from orze.engine import gc_guarded, gc_retirement, gc_safety, storage_preflight
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt, attempt_effect_lock
from test_gc_scope_mechanisms import project
from test_gc_native_authority import case, source


def guarded(root):
    scope, task, checkpoint = project(root)
    scope.cfg['gc'] = {'storage_mode': 'guarded'}
    return scope, task, checkpoint


def run(scope, mode='checkpoints', **kwargs):
    return gc_safety.collect(scope, scope.checkpoints_dir if mode == 'checkpoints' else scope.results_dir,
                             set(), mode=mode, **kwargs)


def fresh_claim(task):
    script = '''import json,sys
from pathlib import Path
from orze.engine.scheduler import claim
from orze.engine.termination_hold import TerminationUnconfirmed
task=Path(sys.argv[1])
try:
 result=claim(task.name, task.parent, None); reason=None
except TerminationUnconfirmed as exc:
 result=False; reason=str(exc)
print(json.dumps({'claimed':result,'reason':reason,'claim_exists':(task/'claim.json').exists()}))
'''
    result = subprocess.run([sys.executable, '-c', script, str(task)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed == {'claimed': False, 'reason': 'gc_retirement_unconfirmed', 'claim_exists': False}
    return observed


@pytest.fixture(params=['local', 'ceph'])
def storage_root(request, tmp_path):
    if request.param == 'local':
        return tmp_path
    base = os.environ.get('ORZE_GC_TEST_ROOT')
    if not base:
        pytest.skip('explicit owned CephFS fixture parent not supplied')
    parent = Path(base)
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='case-', dir=parent))
    assert subprocess.check_output(['stat','-f','-c','%T',str(root)], text=True).strip() == 'ceph'
    return root


@pytest.mark.parametrize('mode', ['checkpoints', 'results', 'archive'])
def test_guarded_operations_work_without_rename(storage_root, monkeypatch, mode):
    scope, task, checkpoint = guarded(storage_root)
    if mode == 'archive':
        (task / 'overlays/nested').mkdir(parents=True)
        (task / 'overlays/nested/data').write_bytes(b'archive directory')
    def forbidden(*args, **kwargs):
        pytest.fail('guarded operation attempted a replacing or no-replace rename')
    monkeypatch.setattr(gc_safety, 'rename_no_replace', forbidden)
    monkeypatch.setattr(storage_preflight, 'require_atomic_rename_support', forbidden)
    monkeypatch.setattr(os, 'rename', forbidden)
    monkeypatch.setattr(os, 'replace', forbidden)
    result = run(scope, mode)
    assert result['errors'] == 0, result
    assert not (task / '_attempt_effect.lock').exists()
    gc_retirement.require_quiet(task)
    if mode == 'checkpoints':
        assert result['deleted'] == 1 and not checkpoint.exists()
    elif mode == 'results':
        assert result['deleted_files'] == 1 and not (task / 'scratch.pt').exists()
    else:
        assert result['archived_files'] == 2 and not (task / 'overlays').exists()
        assert (scope.archive_dir / task.name / 'scratch.pt').read_bytes() == b'result bytes'
        assert (scope.archive_dir / task.name / 'overlays/nested/data').read_bytes() == b'archive directory'
    assert (task / 'metrics.json').read_bytes() == b'{"status":"FAILED"}'
    with attempt_effect_lock(task):
        pass


def test_fresh_claim_is_denied_while_bulk_io_holds_no_effect_guard(tmp_path, monkeypatch):
    scope, task, checkpoint = guarded(tmp_path)
    original = gc_guarded.execute
    def observed(*args):
        assert not (task / '_attempt_effect.lock').exists()
        (tmp_path / 'fresh-claim.json').write_text(json.dumps(fresh_claim(task)))
        return original(*args)
    monkeypatch.setattr(gc_guarded, 'execute', observed)
    assert run(scope)['deleted'] == 1
    with attempt_effect_lock(task):
        pass


@pytest.mark.parametrize('point', ['file_delete', 'root_delete', 'delete_sync'])
def test_unknown_removal_keeps_retirement_and_blocks_new_claim(tmp_path, monkeypatch, point):
    scope, task, checkpoint = guarded(tmp_path)
    unlink, rmdir, sync = os.unlink, os.rmdir, os.fsync
    state = {'removed': False}
    def remove_file(name, *args, **kwargs):
        result = unlink(name, *args, **kwargs)
        if name == 'weights.pt':
            state['removed'] = True
            if point == 'file_delete':
                raise OSError('lost unlink acknowledgement')
        return result
    def remove_root(name, *args, **kwargs):
        result = rmdir(name, *args, **kwargs)
        if name == task.name and point == 'root_delete':
            raise OSError('lost rmdir acknowledgement')
        return result
    def fsync(fd):
        result = sync(fd)
        if state['removed'] and point == 'delete_sync':
            raise OSError('lost source-directory fsync acknowledgement')
        return result
    with monkeypatch.context() as changes:
        changes.setattr(os, 'unlink', remove_file)
        changes.setattr(os, 'rmdir', remove_root)
        changes.setattr(os, 'fsync', fsync)
        result = run(scope)
    assert result['deleted'] == 0 and result['errors'] == 1
    assert run(scope)['errors'] == 1
    fresh_claim(task)
    if point == 'root_delete':
        assert not checkpoint.exists()


@pytest.mark.parametrize('kind', ['file', 'empty_directory', 'directory'])
def test_existing_archive_destination_is_preserved(tmp_path, kind):
    scope, task, checkpoint = guarded(tmp_path)
    destination = scope.archive_dir / task.name / 'scratch.pt'
    destination.parent.mkdir(parents=True)
    if kind == 'file':
        destination.write_bytes(b'foreign archive')
    else:
        destination.mkdir()
        if kind == 'directory':
            (destination / 'foreign').write_bytes(b'foreign archive')
    before = destination.stat().st_ino
    assert run(scope, 'archive')['archived_files'] == 0
    assert destination.stat().st_ino == before
    assert (task / 'scratch.pt').read_bytes() == b'result bytes'
    assert not (task / '_gc_retirements').exists()


@pytest.mark.parametrize('mutation', ['late_collision', 'source_changed', 'archive_changed'])
def test_archive_rechecks_before_source_removal(tmp_path, monkeypatch, mutation):
    scope, task, checkpoint = guarded(tmp_path)
    original = gc_guarded.copy_new
    def changed(tree, destination, check):
        if mutation == 'late_collision':
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b'late foreign archive')
        result = original(tree, destination, check)
        if mutation == 'source_changed':
            tree.path.write_bytes(b'changed source')
        elif mutation == 'archive_changed':
            destination.write_bytes(b'changed archive')
        return result
    monkeypatch.setattr(gc_guarded, 'copy_new', changed)
    result = run(scope, 'archive')
    assert result['archived_files'] == 0 and result['errors'] == 1
    assert (task / 'scratch.pt').read_bytes() == (b'changed source' if mutation == 'source_changed' else b'result bytes')
    if mutation == 'late_collision':
        assert (scope.archive_dir / task.name / 'scratch.pt').read_bytes() == b'late foreign archive'
    fresh_claim(task)


def test_archive_copy_failure_leaves_source_and_partial_target(tmp_path, monkeypatch):
    scope, task, checkpoint = guarded(tmp_path)
    source_file = task / 'scratch.pt'
    original = os.read
    def read(fd, count):
        if os.readlink('/proc/self/fd/' + str(fd)) == str(source_file):
            raise OSError('source read unavailable')
        return original(fd, count)
    monkeypatch.setattr(os, 'read', read)
    assert run(scope, 'archive')['errors'] == 1
    assert source_file.read_bytes() == b'result bytes'
    assert (scope.archive_dir / task.name / 'scratch.pt').exists()
    fresh_claim(task)


@pytest.mark.parametrize('latch', ['.orze_disabled', '.orze_stop_all', '.orze_shutdown'])
def test_stop_precedes_retirement_and_mutation(tmp_path, latch):
    scope, task, checkpoint = guarded(tmp_path)
    (scope.results_dir / latch).touch()
    result = run(scope)
    assert result['deleted'] == 0 and result['errors'] == 1
    assert (checkpoint / 'weights.pt').read_bytes() == b'checkpoint bytes'
    assert not (task / '_gc_retirements').exists()


def test_stop_after_retirement_preserves_pending_owner(tmp_path, monkeypatch):
    scope, task, checkpoint = guarded(tmp_path)
    original = gc_guarded.execute
    def stopped(*args):
        (scope.results_dir / '.orze_shutdown').touch()
        return original(*args)
    monkeypatch.setattr(gc_guarded, 'execute', stopped)
    assert run(scope)['errors'] == 1
    assert (checkpoint / 'weights.pt').read_bytes() == b'checkpoint bytes'
    fresh_claim(task)


@pytest.mark.parametrize('kind', ['symlink', 'hardlink'])
def test_redirected_payload_is_refused_before_retirement(tmp_path, kind):
    scope, task, checkpoint = guarded(tmp_path)
    foreign = tmp_path / 'foreign.bin'
    foreign.write_bytes(b'foreign')
    if kind == 'symlink':
        (checkpoint / 'borrowed').symlink_to(foreign)
    else:
        os.link(foreign, checkpoint / 'borrowed')
    assert run(scope)['errors'] == 1
    assert foreign.read_bytes() == b'foreign'
    assert not (task / '_gc_retirements').exists()


def test_keep_and_dry_run_leave_no_retirement(tmp_path):
    scope, task, checkpoint = guarded(tmp_path)
    result = gc_safety.collect(scope, scope.checkpoints_dir, {task.name}, mode='checkpoints')
    assert result['kept'] == 1 and result['deleted'] == 0
    assert run(scope, dry_run=True)['deleted'] == 1
    assert checkpoint.exists() and not (task / '_gc_retirements').exists()
    assert not (scope.checkpoints_dir / '_orze_gc_quarantine').exists()


def test_guarded_mode_preserves_running_native_authority(case):
    from test_gc_native_authority import test_native_running_with_metrics_cannot_lose_checkpoints_or_result_files
    case.cfg['gc'] = {'storage_mode': 'guarded'}
    test_native_running_with_metrics_cannot_lose_checkpoints_or_result_files(case)


def test_guarded_mode_preserves_closed_native_artifacts(source):
    from test_gc_native_authority import test_closed_native_declared_source_and_accepted_artifact_remain_retained
    source.cfg['gc'] = {'storage_mode': 'guarded'}
    test_closed_native_declared_source_and_accepted_artifact_remain_retained(source)


def test_guarded_deployment_does_not_require_atomic_rename_for_gc(tmp_path, monkeypatch):
    scope, task, checkpoint = guarded(tmp_path)
    scope.cfg['gc'].update(enabled=True, checkpoints_dir=str(scope.checkpoints_dir))
    monkeypatch.setattr(storage_preflight, 'require_atomic_rename_support', lambda *_: pytest.fail('unexpected rename gate'))
    storage_preflight.require_deployment_storage(scope.cfg, scope.results_dir)
