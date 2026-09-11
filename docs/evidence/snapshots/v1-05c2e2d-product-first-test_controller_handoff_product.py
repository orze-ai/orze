"""V2 handoff through real controllers and the real ``python -m orze.cli``.

These are new protocol requirements, not historical stop-only regressions.
Accelerator metadata/capacity, private lease-directory location and offline
license discovery are explicit fixture boundaries. Controller admission,
SQLite, handoff frames, process execution, pidfds and flock remain real.
"""
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import socket
import struct
import subprocess
import sys
import textwrap
import time

import pytest
import yaml

from test_controller_product_boundaries import (
    CONTROLLER, CORE_SOURCE, IDEA, PRO_SOURCE, ProductProcess, WORKER,
    _alive, _line,
)


# Runs only in actual CLI interpreters. In particular, this is not installed
# in the supervisor helper or in CPU workers. No controller method is replaced.
SITECUSTOMIZE = r'''
import json,os,socket,sys
from pathlib import Path
if 'orze.cli' in getattr(sys,'orig_argv',()):
    root=Path(os.environ['ORZE_HANDOFF_FIXTURE_ROOT'])
    channel=None
    owned={os.getpid()}
    if 'ORZE_CONTROLLER_HANDOFF_FD' in os.environ:
        channel=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        channel.settimeout(25);channel.connect(str(root/'birth.sock'))
        channel.sendall((json.dumps({'event':'successor_born','pid':os.getpid(),
            'parent':os.getppid(),'argv':list(sys.orig_argv)})+'\n').encode())
        assert channel.recv(1)==b'L','test did not capture own successor pidfd'
    from orze import extensions
    extensions.has_pro=lambda **kwargs:False
    from orze_pro import _gate
    _gate.require_license=lambda:None
    def tell(event,**values):
        if channel is not None:
            channel.sendall((json.dumps({'event':event,'controller_pid':os.getpid(),
                **values})+'\n').encode())
    from orze.core import gpu_lease
    gpu_lease._lease_dir=lambda:root/'fixture-gpu-leases'
    def idle(gpus):
        assert list(gpus)==[0]
        tell('successor_gpu_idle_spy',gpus=list(gpus))
        return {'physical_scope':[0],'compute_processes':0,
            'accelerator_access':'metadata_only','accelerator_compute_access':'none'}
    gpu_lease.assert_gpu_scope_idle=idle
    from orze.engine import gpu_slots,launcher,process
    from orze.hardware import gpu as hardware_gpu
    def usage(gpus=None):
        assert gpus is not None and set(gpus)<=set([0])
        return {g:(0,100000) for g in gpus}
    gpu_slots._query_all_gpu_usage=usage
    gpu_slots._get_load_per_cpu=lambda:0.0
    gpu_slots._get_free_ram_gb=lambda:1000.0
    gpu_slots._count_user_processes=lambda:len(owned)
    hardware_gpu.get_gpu_memory_used=lambda gpu:0
    def details(gpus=None):
        if gpus is None or not set(gpus)<=set([0]):
            raise RuntimeError('fixture GPU details scope mismatch')
        return [{'index':g,'name':'synthetic metadata only','memory_used_mib':0,
            'memory_total_mib':100000,'utilization_pct':0,'temperature_c':0}
            for g in gpus]
    hardware_gpu._query_gpu_details=details
    from orze.reporting import state as reporting_state
    reporting_state._query_gpu_details=details
    from orze.engine import phases
    phases.get_gpu_memory_used=hardware_gpu.get_gpu_memory_used
    original_iterdir=Path.iterdir
    def own_iterdir(path):
        if path==Path('/proc'):
            return iter(Path('/proc')/str(pid) for pid in sorted(owned))
        return original_iterdir(path)
    Path.iterdir=own_iterdir
    original_listdir=os.listdir
    os.listdir=lambda path:([str(pid) for pid in sorted(owned)]
        if str(path)=='/proc' else original_listdir(path))
    try:
        import psutil
        def own_process_iter(attrs=None):
            for pid in sorted(owned):
                try:
                    value=psutil.Process(pid);value.info=value.as_dict(attrs=attrs)
                    yield value
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        psutil.process_iter=own_process_iter
    except ImportError:pass
    from orze.engine.supervised_process import prepare_supervised
    def captured_prepare(*args,**kwargs):
        handle=prepare_supervised(*args,**kwargs)
        worker,supervisor=handle.pid,handle._supervisor.pid
        owned.update((worker,supervisor))
        tell('successor_worker_ready',worker=worker,supervisor=supervisor)
        assert channel is not None and channel.recv(1)==b'L'
        return handle
    launcher.prepare_supervised=captured_prepare
    process.prepare_supervised=captured_prepare
    from orze_pro.engine import role_runner
    role_runner.prepare_supervised=captured_prepare
'''


RESTART_OBSERVER = r'''
import dataclasses,json,sys
from pathlib import Path
import yaml
from orze.core.config import load_project_config
from orze.engine.controller_control import ControllerHOLD
from orze.engine import controller_handoff as handoff
fault=sys.argv[4] if len(sys.argv)>4 else ''
if fault:
    actual_receive=handoff._receive
    def receive(channel,peer_pid,deadline):
        value=actual_receive(channel,peer_pid,deadline)
        event=value['event']
        if ((fault=='change_config_after_hello' and event=='HELLO') or
                (fault=='lose_final_started' and event=='STARTED')):
            path=Path(sys.argv[1])
            (path.parent/'observer-fault.json').write_text(json.dumps(
                {'fault':fault,'event':event,'after_real_receive':True}))
            if fault=='change_config_after_hello':
                current=yaml.safe_load(path.read_text())
                current['poll']=0.075
                path.write_text(yaml.safe_dump(current))
            else:
                raise OSError('fixture lost final STARTED response after actual receive')
        return value
    handoff._receive=receive
try:
    result=handoff.restart_controller(load_project_config(sys.argv[1]),
        request_id=sys.argv[2],timeout=float(sys.argv[3]))
    record=dataclasses.asdict(result) if dataclasses.is_dataclass(result) else result
    print(json.dumps({'completed':True,'record':record},default=str))
except ControllerHOLD as exc:
    print(json.dumps({'completed':False,'reason':str(exc)}))
'''


class HandoffProcess(ProductProcess):
    """Reuse only existing owned-OS cleanup/read helpers, not V1 admission."""

    def __init__(self, root, mode):
        self.root, self.mode = root, mode
        self.pidfds, self.channels, self.events, self.handles = {}, [], [], []
        self.observers, self.births, self.successor_channels = [], [], []
        self.pause_ready = False
        self.results, self.db = root / 'results', root / '.orze' / 'ideas.db'
        self.results.mkdir(); self.db.parent.mkdir()
        (root / 'base.yaml').write_text('{}\n')
        (root / 'worker.py').write_text(WORKER)
        hooks = root / 'hooks'; hooks.mkdir()
        # Python otherwise logs a sitecustomize import error and keeps running;
        # a failed hardware-boundary installation must instead stop this exec.
        (hooks / 'sitecustomize.py').write_text(
            'try:\n' + textwrap.indent(SITECUSTOMIZE, '    ') +
            '\nexcept BaseException as exc:\n'
            '    import os,sys\n'
            '    print("fixture sitecustomize failed: " + repr(exc), file=sys.stderr, flush=True)\n'
            '    os._exit(97)\n')
        self.worker_listener = self._listener(root / 'worker.sock')
        self.birth_listener = self._listener(root / 'birth.sock')
        listener = self._listener(root / 'controller.sock')
        args = ['--fixture-root', str(root), '--fixture-socket', str(root / 'worker.sock')]
        self.cfg = {
            'controller_control': {'version': 2, 'profile': 'local_handoff_v1'},
            'results_dir': str(self.results), 'idea_lake_db': str(self.db),
            'ideas_file': str(root / '.orze' / 'ideas.md'),
            'base_config': str(root / 'base.yaml'), 'train_script': str(root / 'worker.py'),
            'python': sys.executable, 'train_extra_args': args + ['--fixture-mode', mode],
            'gpu_scheduling': {'allowed_gpus': [0], 'mode': 'exclusive'},
            'poll': 0.05, 'timeout': 30, 'pre_timeout': 30,
            'role_presets': [], 'roles': {}, 'telemetry': False,
            'bot': None, 'telegram_bot': None, 'notifications': {'enabled': False},
            'auto_upgrade': False, 'retrospection': {'enabled': False},
            'cleanup': {'script': None, 'interval': 0, 'patterns': []},
            'metric_harvest': {'enabled': False, 'llm_fallback': False},
            'max_fix_attempts': 0, 'sweep_stray': False, 'auto_seal_eval': False,
            'stall_minutes': 0, 'role_stall_minutes': 0,
            'report': {'primary_metric': 'score'},
        }
        Path(self.cfg['ideas_file']).write_text(
            '# Ideas\n\n## ' + IDEA + ': Tiny handoff CPU fixture\n\n```yaml\nseed: 13\n```\n')
        (root / 'orze.yaml').write_text(yaml.safe_dump(self.cfg))
        self.environment = {
            'PATH': os.environ.get('PATH', ''),
            'PYTHONPATH': os.pathsep.join([str(hooks), str(CORE_SOURCE), str(PRO_SOURCE)]),
            'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '',
            'ORZE_HANDOFF_FIXTURE_ROOT': str(root),
        }
        self.log = (root / 'controller.log').open('wb')
        self.child = subprocess.Popen(
            [sys.executable, '-c', CONTROLLER, str(root), str(root / 'controller.sock')],
            cwd=root, env=self.environment, stdin=subprocess.DEVNULL,
            stdout=self.log, stderr=subprocess.STDOUT)
        self.controller_fd = self.capture(self.child.pid)
        self.channel, _ = listener.accept(); self.channel.settimeout(5)
        self.channels.append(self.channel)
        pid, uid, _ = struct.unpack('3i', self.channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        assert pid == self.child.pid and uid == os.getuid()

    def pump(self):
        choices = [self.birth_listener, *self.successor_channels]
        if _alive(self.controller_fd):
            choices.append(self.channel)
        for channel in select.select(choices, [], [], 0.02)[0]:
            if channel is self.channel:
                value = _line(channel)
                if value['event'] == 'ready':
                    self.capture(value['worker']); self.capture(value['supervisor'])
                    self.handles.append(value)
                    self.command(op='release', token=value['token'])
                self.events.append(value)
            elif channel is self.birth_listener:
                connected, _ = channel.accept(); connected.settimeout(5)
                self.channels.append(connected)
                pid, uid, _ = struct.unpack('3i', connected.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                fd = self.capture(pid)
                value = _line(connected)
                assert value['pid'] == pid and uid == os.getuid()
                assert _alive(fd)
                value['old_controller_alive_at_birth'] = _alive(self.controller_fd)
                value['old_workers_alive_at_birth'] = [
                    owned for owned in getattr(self, 'old_workers', ()) if _alive(self.pidfds[owned])]
                self.births.append(value); self.events.append(value)
                self.successor_channels.append(connected)
                connected.sendall(b'L')
            else:
                if not channel.recv(1, socket.MSG_PEEK):
                    self.successor_channels.remove(channel)
                    continue
                value = _line(channel)
                if value['event'] == 'successor_worker_ready':
                    self.capture(value['worker']); self.capture(value['supervisor'])
                    value['old_controller_alive_at_go'] = _alive(self.controller_fd)
                    value['old_workers_alive_at_go'] = [
                        owned for owned in getattr(self, 'old_workers', ()) if _alive(self.pidfds[owned])]
                    channel.sendall(b'L')
                self.events.append(value)

    def until(self, predicate, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            self.pump()
        assert predicate(), (self.events, (self.root / 'controller.log').read_text()[-16000:])

    def restart(self, request_id, *, cli=False, timeout=12, fault=''):
        if cli:
            command = [sys.executable, '-m', 'orze.cli', 'restart', '-c',
                       str(self.root / 'orze.yaml'), '--request-id', request_id,
                       '--timeout', str(timeout)]
        else:
            command = [sys.executable, '-c', RESTART_OBSERVER,
                       str(self.root / 'orze.yaml'), request_id, str(timeout), fault]
        child = subprocess.Popen(command, cwd=self.root, env=self.environment,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.observers.append(child); self.capture(child.pid)
        return child

    def result(self, child, *, cli=False, timeout=17):
        self.until(lambda: child.poll() is not None, timeout=timeout)
        stdout, stderr = child.communicate(timeout=2)
        assert child.returncode == 0, (stdout.decode(), stderr.decode(), self.events)
        if cli:
            return {'completed': True}
        return json.loads(stdout)

    def assert_successor_holds_real_lease(self):
        descriptor = os.open(self.root / 'fixture-gpu-leases' / 'gpu-0.lock', os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)


@pytest.fixture
def handoff_process(tmp_path):
    if sys.platform != 'linux' or not hasattr(os, 'pidfd_open'):
        pytest.skip('requires Linux own pidfds')
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    instances = []
    def spawn(mode):
        root = tmp_path / mode; root.mkdir()
        instance = HandoffProcess.__new__(HandoffProcess)
        instances.append(instance); instance.__init__(root, mode)
        return instance
    try:
        yield spawn
    finally:
        for instance in reversed(instances):
            instance.close()
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


def _prepare_old(handoff_process, mode):
    controller = handoff_process(mode)
    assert controller.event('constructed')['lake'] == 'IdeaLake'
    _, worker = controller.worker()
    controller.old_workers = [worker['pid']]
    if mode == 'escaped':
        channel, daemon = controller.worker(daemon=True)
        controller.old_workers.append(daemon['pid'])
        assert _alive(controller.pidfds[daemon['pid']])
        channel.sendall(b'W'); assert channel.recv(1) == b'W'
    else:
        controller.wait(lambda: any(
            row['state'] == 'TERMINAL' for row in controller.rows('execution_attempts')))
    return controller


def _assert_started(controller, request_id):
    instances = controller.rows('controller_instances')
    assert sorted(row['generation'] for row in instances) == [0, 1]
    old, new = sorted(instances, key=lambda row: row['generation'])
    assert new['predecessor'] == old['controller_id']
    heads = controller.rows('controller_scope_heads')
    assert len(heads) == 1 and heads[0]['generation'] == 1
    assert heads[0]['current_id'] == new['controller_id']
    grants = controller.rows('controller_handoffs')
    assert len(grants) == 1
    grant = grants[0]
    assert grant['request_id'] == request_id and grant['state'] == 'STARTED'
    assert grant['source_controller_id'] == old['controller_id']
    assert grant['target_controller_id'] == new['controller_id']
    assert grant['ready_json'] and grant['started_json'] and grant['hold_reason'] is None
    sessions = controller.rows('controller_sessions')
    assert len(sessions) == 2
    assert next(row for row in sessions if row['controller_id'] == old['controller_id'])['ack_json']
    assert next(row for row in sessions if row['controller_id'] == new['controller_id'])['ack_json'] is None
    return grant


@pytest.mark.parametrize('mode', ['normal', 'escaped'])
def test_real_api_handoff_requires_old_tree_exit_and_live_successor_resources(handoff_process, mode):
    controller = _prepare_old(handoff_process, mode)
    result = controller.result(controller.restart('api-first'))
    assert result['completed'] is True
    assert not _alive(controller.controller_fd)
    assert len(controller.births) == 1
    born = controller.births[0]
    assert born['old_controller_alive_at_birth'] is False
    assert born['old_workers_alive_at_birth'] == []
    assert _alive(controller.pidfds[born['pid']])
    assert controller.child.wait(timeout=3) == 0
    instances = controller.rows('controller_instances')
    assert sorted(row['generation'] for row in instances) == [0, 1]
    heads = controller.rows('controller_scope_heads')
    assert len(heads) == 1 and heads[0]['generation'] == 1
    assert heads[0]['current_id'] == next(row['controller_id'] for row in instances if row['generation'] == 1)
    assert json.loads(next(row['identity_json'] for row in instances if row['generation'] == 1))['process']['pid'] == born['pid']
    grant = _assert_started(controller, 'api-first')
    assert result['record']['grant_id'] == grant['grant_id']
    assert result['record']['target_controller_id'] == grant['target_controller_id']
    assert result['record']['started_sha256'] == hashlib.sha256(grant['started_json'].encode()).hexdigest()
    controller.assert_successor_holds_real_lease()
    controller.until(lambda: any(event['event'] == 'successor_gpu_idle_spy' for event in controller.events))
    assert _alive(controller.pidfds[born['pid']]), 'observer exit must not terminate accepted successor'


def test_actual_cli_restart_uses_real_successor_not_transport_double(handoff_process):
    controller = _prepare_old(handoff_process, 'normal')
    observer = controller.restart('cli-first', cli=True)
    controller.result(observer, cli=True)
    assert not _alive(controller.pidfds[observer.pid])
    assert not _alive(controller.controller_fd)
    assert len(controller.births) == 1
    born = controller.births[0]
    assert born['old_controller_alive_at_birth'] is False
    assert '-m' in born['argv'] and 'orze.cli' in born['argv']
    assert _alive(controller.pidfds[born['pid']])
    assert controller.rows('controller_scope_heads')[0]['generation'] == 1
    _assert_started(controller, 'cli-first')
    controller.assert_successor_holds_real_lease()


def test_concurrent_same_request_and_replay_create_only_one_real_successor(handoff_process):
    controller = _prepare_old(handoff_process, 'normal')
    observers = [controller.restart('same-request') for _ in range(2)]
    results = [controller.result(observer) for observer in observers]
    assert any(result['completed'] is True for result in results)
    assert len(controller.births) == 1
    instances = controller.rows('controller_instances')
    assert sorted(row['generation'] for row in instances) == [0, 1]
    head = controller.rows('controller_scope_heads')
    assert len(head) == 1 and head[0]['generation'] == 1
    grant = _assert_started(controller, 'same-request')
    replay = controller.result(controller.restart('same-request'))
    assert isinstance(replay['completed'], bool)
    assert controller.rows('controller_instances') == instances
    assert controller.rows('controller_scope_heads') == head
    assert controller.rows('controller_handoffs') == [grant]
    assert len(controller.births) == 1
    assert _alive(controller.pidfds[controller.births[0]['pid']])
    controller.assert_successor_holds_real_lease()


def test_configuration_changed_after_real_hello_cannot_start_successor(handoff_process):
    controller = _prepare_old(handoff_process, 'normal')
    before = controller.rows('controller_scope_heads')
    result = controller.result(controller.restart(
        'config-drift', fault='change_config_after_hello'))
    assert result['completed'] is False
    fault = json.loads((controller.root / 'observer-fault.json').read_text())
    assert fault == {'fault': 'change_config_after_hello', 'event': 'HELLO', 'after_real_receive': True}
    assert not _alive(controller.controller_fd)
    assert len(controller.births) == 1
    assert not any(event['event'] in {'successor_gpu_idle_spy', 'successor_worker_ready'}
                   for event in controller.events)
    heads = controller.rows('controller_scope_heads')
    assert heads[0]['current_id'] == before[0]['current_id']
    assert heads[0]['generation'] == 0
    assert len(controller.rows('controller_instances')) == 1
    grants = controller.rows('controller_handoffs')
    assert len(grants) == 1 and grants[0]['request_id'] == 'config-drift'
    assert grants[0]['state'] not in {'PREPARED', 'STARTED'}
    assert grants[0]['started_json'] is None
    replay = controller.result(controller.restart('config-drift'))
    assert replay['completed'] is False
    assert len(controller.births) == 1
    assert controller.rows('controller_scope_heads') == heads


def test_lost_final_started_response_retains_one_independent_successor(handoff_process):
    controller = _prepare_old(handoff_process, 'normal')
    result = controller.result(controller.restart(
        'lost-response', fault='lose_final_started'))
    assert result['completed'] is False
    fault = json.loads((controller.root / 'observer-fault.json').read_text())
    assert fault == {'fault': 'lose_final_started', 'event': 'STARTED', 'after_real_receive': True}
    assert not _alive(controller.controller_fd)
    assert len(controller.births) == 1
    grant = _assert_started(controller, 'lost-response')
    successor = controller.births[0]['pid']
    assert _alive(controller.pidfds[successor])
    controller.assert_successor_holds_real_lease()
    replay = controller.result(controller.restart('lost-response'))
    assert replay['completed'] is True
    assert replay['record']['grant_id'] == grant['grant_id']
    assert len(controller.births) == 1
    assert controller.rows('controller_handoffs') == [grant]
    assert _alive(controller.pidfds[successor])
    controller.assert_successor_holds_real_lease()
