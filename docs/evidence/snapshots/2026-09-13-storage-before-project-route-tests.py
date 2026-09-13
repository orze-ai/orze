"""Storage deployment gates; only fresh fixture paths, never user state.

The CephFS regressions exercise the actual local no-replace primitive. They
skip on hosts without this precise unsupported fixture filesystem, not to
convert unsupported deployments into successes. No worker/provider/GPU runs.
"""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import os

import pytest

from orze.engine.gc_tree import GCRefused, rename_no_replace


@pytest.fixture
def unsupported_root():
    parent = Path('/hot-data/fsx/workspace/erik')
    if not parent.is_dir():
        pytest.skip('actual local unsupported filesystem is unavailable')
    root = Path(tempfile.mkdtemp(prefix='orze-storage-boundary-', dir=parent))
    source = root / 'capability-source'
    source.mkdir()
    try:
        rename_no_replace(source, root / 'capability-target')
    except GCRefused as exc:
        if str(exc) != 'gc_atomic_rename_unsupported':
            raise
    else:
        pytest.skip('actual filesystem now supports no-replace; retain fixture')
    # Retain this small unique fixture for diagnosis; no broad cleanup.
    return root


def _runner(root, cfg):
    results = root / 'results'
    results.mkdir()
    events = []
    runner = SimpleNamespace(
        cfg=cfg, results_dir=results, _cpu_execution=None, lake=None,
        _write_pid_file=lambda: events.append('pid'),
        _check_disabled=lambda: True,
        _release_gpu_leases=lambda: events.append('release'),
        _remove_pid_file=lambda: events.append('remove'),
    )
    return runner, events


class TestOriginalBoundary:
    def test_deployment_refuses_before_pid_or_resource_entry(self, unsupported_root):
        from orze.engine.orchestrator import Orze
        root = unsupported_root
        runner, events = _runner(root, {
            '_orze_dir': str(root / '.orze'),
            'roles': {'engineer': {'mode': 'script', 'script': 'unused.py'}},
        })
        with pytest.raises(ValueError, match='storage_atomic_rename_unsupported'):
            Orze.run(runner)
        assert events == []
        assert not (root / '.orze' / 'locks' / 'engineer').exists()

    def test_gc_refuses_before_task_effect_or_quarantine(self, unsupported_root):
        from orze.engine.gc_safety import collect, gc_scope
        root = unsupported_root
        results, checkpoints = root / 'results', root / 'checkpoints'
        task = results / 'idea-disposable'
        task.mkdir(parents=True)
        (task / 'metrics.json').write_bytes(b'{"status":"FAILED"}')
        candidate = checkpoints / task.name
        candidate.mkdir(parents=True)
        (candidate / 'weights.pt').write_bytes(b'original-checkpoint')
        scope = gc_scope(results, {'_project_root': str(root)}, checkpoints_dir=checkpoints)
        stats = collect(scope, checkpoints, set(), mode='checkpoints')
        assert stats['errors'] == 1 and stats['deleted'] == 0
        assert not (task / '_attempt_effect.lock').exists()
        assert not (checkpoints / '_orze_gc_quarantine').exists()
        assert stats['reasons'] == ['storage_atomic_rename_unsupported']
        assert (candidate / 'weights.pt').read_bytes() == b'original-checkpoint'

    def test_no_rename_features_keep_existing_run_route(self, unsupported_root):
        from orze.engine.orchestrator import Orze
        runner, events = _runner(unsupported_root, {'roles': {}})
        Orze.run(runner)
        assert events == ['pid', 'release', 'remove']


class TestProbeMechanisms:
    def test_real_file_directory_success_and_collision_preservation(self, tmp_path, monkeypatch):
        from orze.engine import storage_preflight as storage
        keep = tmp_path / 'user-file'
        keep.write_bytes(b'untouched')
        calls = []
        actual = storage.rename_no_replace

        def observed(source, target):
            collision = target.exists()
            original = (source.stat().st_ino, target.stat().st_ino if collision else None)
            try:
                return actual(source, target)
            finally:
                calls.append((source.is_dir() if source.exists() else target.is_dir(), collision))
                if collision:
                    assert (source.stat().st_ino, target.stat().st_ino) == original
                    if source.is_file():
                        assert source.read_bytes() == b'probe-source'
                        assert target.read_bytes() == b'probe-target'

        monkeypatch.setattr(storage, 'rename_no_replace', observed)
        storage.require_atomic_rename_support(tmp_path)
        assert calls == [(False, False), (False, True), (True, False), (True, True)]
        assert sorted(p.name for p in tmp_path.iterdir()) == ['user-file']
        assert keep.read_bytes() == b'untouched'

    def test_real_unsupported_probe_cleans_only_own_objects(self, unsupported_root):
        from orze.engine.storage_preflight import require_atomic_rename_support, StoragePreflightError
        before = {p.name for p in unsupported_root.iterdir()}
        with pytest.raises(StoragePreflightError, match='storage_atomic_rename_unsupported'):
            require_atomic_rename_support(unsupported_root)
        assert {p.name for p in unsupported_root.iterdir()} == before

    def test_symlink_route_rejected_without_following(self, tmp_path):
        from orze.engine.storage_preflight import require_atomic_rename_support, StoragePreflightError
        foreign = tmp_path / 'foreign'
        foreign.mkdir()
        redirect = tmp_path / 'redirect'
        redirect.symlink_to(foreign, target_is_directory=True)
        with pytest.raises(StoragePreflightError):
            require_atomic_rename_support(redirect)
        assert list(foreign.iterdir()) == []

    def test_parent_replaced_during_operation_never_cleans_replacement(self, tmp_path, monkeypatch):
        from orze.engine import storage_preflight as storage
        root, old = tmp_path / 'root', tmp_path / 'old'
        root.mkdir()
        actual = storage.rename_no_replace

        def replace(source, target):
            root.rename(old)
            root.mkdir()
            (root / 'foreign').write_bytes(b'keep')
            return actual(source, target)

        monkeypatch.setattr(storage, 'rename_no_replace', replace)
        with pytest.raises(storage.StoragePreflightError):
            storage.require_atomic_rename_support(root)
        assert (root / 'foreign').read_bytes() == b'keep'
        assert sorted(p.name for p in root.iterdir()) == ['foreign']

    def test_wrong_overwriting_primitive_cannot_pass_collision_control(self, tmp_path, monkeypatch):
        from orze.engine import storage_preflight as storage
        # Deliberately broken provider only touches newly created probe objects.
        monkeypatch.setattr(storage, 'rename_no_replace', os.replace)
        with pytest.raises(storage.StoragePreflightError):
            storage.require_atomic_rename_support(tmp_path)
        assert list(tmp_path.glob('.orze-storage-probe-*'))

    @pytest.mark.parametrize('stage', ['initial_fsync', 'cleanup_unlink', 'after_cleanup_fsync'])
    def test_storage_or_cleanup_uncertainty_fails_closed(self, tmp_path, monkeypatch, stage):
        from orze.engine import storage_preflight as storage
        keep = tmp_path / 'user-file'
        keep.write_bytes(b'keep')
        actual_sync, actual_unlink, actual_rmdir = os.fsync, os.unlink, os.rmdir
        state = {'syncs': 0, 'removed_probe': False}

        def sync(fd):
            state['syncs'] += 1
            actual_sync(fd)
            if ((stage == 'initial_fsync' and state['syncs'] == 1)
                    or (stage == 'after_cleanup_fsync' and state['removed_probe'])):
                raise OSError('actual fsync response uncertain')

        def unlink(name, *args, **kwargs):
            if stage == 'cleanup_unlink' and name == 'target':
                raise PermissionError('own probe cleanup denied')
            return actual_unlink(name, *args, **kwargs)

        def rmdir(name, *args, **kwargs):
            actual_rmdir(name, *args, **kwargs)
            if str(name).startswith('.orze-storage-probe-'):
                state['removed_probe'] = True

        monkeypatch.setattr(os, 'fsync', sync)
        monkeypatch.setattr(os, 'unlink', unlink)
        monkeypatch.setattr(os, 'rmdir', rmdir)
        with pytest.raises(storage.StoragePreflightError):
            storage.require_atomic_rename_support(tmp_path)
        assert keep.read_bytes() == b'keep'
        if stage == 'after_cleanup_fsync':
            assert state['removed_probe'] is True

    def test_short_probe_write_refuses(self, tmp_path, monkeypatch):
        from orze.engine import storage_preflight as storage
        monkeypatch.setattr(os, 'write', lambda *args: 0)
        with pytest.raises(storage.StoragePreflightError):
            storage.require_atomic_rename_support(tmp_path)

    def test_cpu_route_does_not_require_role_or_gc_storage(self, tmp_path, monkeypatch):
        from orze.engine import orchestrator, cpu_phase
        root = tmp_path / 'project'
        root.mkdir()
        runner, events = _runner(root, {'roles': {'engineer': {'mode': 'script', 'script': 'unused'}}})
        runner._cpu_execution = object()
        runner._run_leased = lambda: events.append('cpu')
        monkeypatch.setattr(cpu_phase, 'close', lambda engine: events.append('close'))
        orchestrator.Orze.run(runner)
        assert events == ['cpu', 'close']
        assert not (root / '.orze').exists()

    def test_dry_run_remains_read_only_on_unsupported_storage(self, unsupported_root):
        from orze.engine.gc_safety import collect, gc_scope
        results, checkpoints = unsupported_root / 'results', unsupported_root / 'checkpoints'
        task = results / 'idea-disposable'
        task.mkdir(parents=True)
        (task / 'metrics.json').write_bytes(b'{"status":"FAILED"}')
        candidate = checkpoints / task.name
        candidate.mkdir(parents=True)
        (candidate / 'weights.pt').write_bytes(b'keep')
        scope = gc_scope(results, {'_project_root': str(unsupported_root)}, checkpoints_dir=checkpoints)
        before = sorted(str(p.relative_to(unsupported_root)) for p in unsupported_root.rglob('*'))
        stats = collect(scope, checkpoints, set(), mode='checkpoints', dry_run=True)
        assert stats['errors'] == 0 and stats['deleted'] == 1
        assert sorted(str(p.relative_to(unsupported_root)) for p in unsupported_root.rglob('*')) == before

    @pytest.mark.parametrize('cfg', [
        {'roles': {'disabled': {'enabled': False, 'mode': 'script', 'script': 'unused'}}},
        {'roles': {'invalid': {'mode': 'script'}}},
        {'roles': {}, 'gc': {'enabled': False, 'checkpoints_dir': '/not-used'}},
    ])
    def test_unused_deployment_features_make_no_probe(self, tmp_path, cfg):
        from orze.engine.storage_preflight import require_deployment_storage
        require_deployment_storage(cfg, tmp_path)
        assert list(tmp_path.iterdir()) == []
