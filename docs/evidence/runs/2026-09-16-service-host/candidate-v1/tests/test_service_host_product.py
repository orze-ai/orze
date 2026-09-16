"""Persistent service owner, real CPU controllers and short watchdog clients.

Birth channels capture only children of our own host; cleanup uses those pidfds.
No manager, GPU, provider, global PID lookup, or existing service is used.
"""
import json
import os
from pathlib import Path
import select
import signal
import socket
import sqlite3
import struct
import subprocess
import sys
import time

import pytest
import yaml

from test_cpu_controller_stop_product import SITE, cpu_controller, rows

HOOK = r'''
if 'orze.service.host' in getattr(sys, 'orig_argv', ()):
    import subprocess
    from pathlib import Path
    actual_popen = subprocess.Popen
    def popen(command, *args, **kwargs):
        if list(command[:3]) == [sys.executable, '-m', 'orze.cli']:
            with Path('controllers.log').open('ab') as stream:
                kwargs.update(stdout=stream, stderr=subprocess.STDOUT)
                return actual_popen(command, *args, **kwargs)
        raise AssertionError('service host crossed an unexpected spawn boundary')
    subprocess.Popen = popen
if 'orze.cli' in getattr(sys, 'orig_argv', ()):
    import socket, json
    from orze.engine.supervisor_worker import process_identity
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    channel.settimeout(20)
    channel.connect(str(__import__('pathlib').Path.cwd() / 'birth.sock'))
    channel.sendall((json.dumps({'process': process_identity(os.getpid())[0],
                               'parent': os.getppid()}) + '\n').encode())
    assert channel.recv(1) == b'B'
    channel.close()
if 'orze.service.watchdog' in getattr(sys, 'orig_argv', ()):
    from orze.service import watchdog
    def forbidden(*a, **k):
        raise AssertionError('hosted watchdog reached legacy process authority')
    watchdog._read_pid = watchdog._is_pid_alive = watchdog._is_orze_running = forbidden
    watchdog._kill_stale = watchdog._launch_orze = watchdog.check_containers = forbidden
'''


@pytest.fixture
def hosted(cpu_controller):
    from orze.service.runtime_contract import capture_runtime_packages
    root, config, cfg, env, launch, wait_for = cpu_controller
    cfg['controller_control'] = {'version': 2, 'profile': 'local_cpu_handoff_v1'}
    config.write_text(yaml.safe_dump(cfg))
    (root / 'results').mkdir()
    (root / '.env').write_text('')
    (root / 'sitecustomize.py').write_text(SITE + HOOK)
    svc = {'service_owner': {'version': 1, 'profile': 'local_service_host_v1'},
           'service_config_file': str(root / 'service.json'),
           'method': 'process', 'python': sys.executable, 'workdir': str(root),
           'config_file': str(config), 'results_dir': str(root / 'results'),
           'log_file': str(root / 'results/watchdog.log'), 'stall_threshold': .05,
           'runtime_contract_version': 1, 'runtime_packages': capture_runtime_packages()}
    service = root / 'service.json'
    service.write_text(json.dumps(svc, sort_keys=True))
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(root / 'birth.sock'))
    listener.listen()
    host = None
    captures, births, clients = [], [], []
    prior = []
    log = (root / 'host.log').open('wb')

    def receive_birth():
        if select.select([listener], [], [], .01)[0]:
            channel, _ = listener.accept()
            with channel:
                channel.settimeout(3)
                raw = b''
                while not raw.endswith(b'\n'):
                    raw += channel.recv(4096)
                birth = json.loads(raw)
                pid, uid, gid = struct.unpack('3i', channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                assert birth['process']['pid'] == pid and (uid, gid) == (os.getuid(), os.getgid())
                assert birth['parent'] == host.pid
                assert all(select.select([fd], [], [], 0)[0] for fd in prior)
                fd = os.pidfd_open(pid)
                captures.append(fd)
                births.append({**birth, 'old_exited_before_birth': bool(prior)})
                channel.sendall(b'B')

    def wait(predicate, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            receive_birth()
            try:
                if predicate():
                    return
            except (sqlite3.OperationalError, FileNotFoundError, ConnectionError):
                pass
            if host is not None and host.poll() is not None:
                pytest.fail((root / 'host.log').read_text() + (root / 'controllers.log').read_text())
        pytest.fail('service fixture timed out: ' + (root / 'host.log').read_text())

    def start():
        nonlocal host
        host = subprocess.Popen([sys.executable, '-m', 'orze.service.host', '--service-config', str(service)],
                                cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        wait(lambda: (root / 'results/_service_host.lock/ready.json').exists())
        return host

    def client(operation, key=None, source=None, watchdog=False):
        command = [sys.executable, '-m', 'orze.service.watchdog' if watchdog else 'orze.service.host',
                   '--service-config', str(service)]
        if not watchdog:
            command += ['--operation', operation, '--timeout', '12']
            if key:
                command += ['--request-id', key]
            if source:
                command += ['--source-controller-id', source]
        proc = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        clients.append(proc)
        deadline = time.monotonic() + 20
        while proc.poll() is None and time.monotonic() < deadline:
            receive_birth()
        output, _ = proc.communicate(timeout=2)
        (root / ('client-%d.log' % len(clients))).write_text(output)
        return proc.returncode, output

    try:
        yield root, config, cfg, env, service, start, client, wait, prior, captures, births
    finally:
        for proc in clients:
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=5)
        if host is not None:
            if host.poll() is None:
                host.terminate()
                try:
                    host.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    host.kill()
                    host.wait(timeout=5)
            else:
                host.wait()
        for fd in captures:
            if not select.select([fd], [], [], 0)[0]:
                signal.pidfd_send_signal(fd, signal.SIGTERM)
                if not select.select([fd], [], [], 10)[0]:
                    signal.pidfd_send_signal(fd, signal.SIGKILL)
                    assert select.select([fd], [], [], 5)[0]
            os.close(fd)
        log.close()
        listener.close()
        (root / 'births.json').write_text(json.dumps(births, sort_keys=True, indent=2))


def idea(root, name, program):
    action = {'version': 1, 'adapter': 'command', 'purpose': 'service ownership fixture', 'inputs': {},
              'command': [sys.executable, '-c', program], 'timeout_seconds': 60, 'outputs': {}}
    with (root / 'ideas.md').open('a') as stream:
        stream.write('## ' + name + ': service CPU action\n\n```yaml\n' +
                     yaml.safe_dump({'kind': 'native_cpu_action', 'action': action}) + '```\n')


def test_watchdog_handoff_keeps_host_and_successor_after_client_exit(hosted):
    from orze.service.host import request
    root, config, cfg, env, service, start, client, wait, prior, captures, births = hosted
    marker = root / 'worker.json'
    program = ('import os,json,subprocess,sys,time\nfrom pathlib import Path\n'
               'child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(120)"])\n'
               f'Path({str(marker)!r}).write_text(json.dumps([os.getpid(),child.pid]))\n'
               'time.sleep(120)')
    idea(root, 'first-owned', program)
    host = start()
    wait(marker.exists)
    worker_fds = [os.pidfd_open(pid) for pid in json.loads(marker.read_text())]
    captures.extend(worker_fds)
    prior.extend([captures[0], *worker_fds])
    old = rows(root, 'SELECT current_id FROM controller_scope_heads')[0][0]
    idea(root, 'second-owned', f'from pathlib import Path;Path({str(root / "done")!r}).write_text("done")')
    (root / 'results' / ('_host_' + socket.gethostname() + '_fixture.json')).write_text(json.dumps({'epoch': time.time() - 100}))
    code, output = client('watchdog', watchdog=True)
    assert code == 0, output + (root / 'host.log').read_text()
    wait(lambda: (root / 'done').exists() and rows(root, 'SELECT COUNT(*) FROM execution_attempts WHERE state="TERMINAL"') == [(2,)])
    assert host.poll() is None and len(births) == 2 and births[1]['old_exited_before_birth']
    assert rows(root, 'SELECT state FROM controller_handoffs') == [('STARTED',)]
    key = 'watchdog-' + old
    code, output = client('restart', key, old)
    assert code == 0, output
    assert len(births) == 2 and rows(root, 'SELECT COUNT(*) FROM controller_instances') == [(2,)]
    status = request(service, 'status', timeout=5)
    assert status['host_process']['pid'] == host.pid
    assert status['controller']['process']['pid'] == births[1]['process']['pid']
    assert status['live_children'] == 1
    assert client('stop')[0] == 0
    assert host.wait(timeout=5) == 0
    assert all(select.select([fd], [], [], 0)[0] for fd in captures)
    assert rows(root, 'SELECT state FROM cpu_action_reservations') == [('SETTLED',), ('SETTLED',)]


@pytest.mark.parametrize('fault', ['source', 'service', 'wrong_source'])
def test_host_refuses_changed_inputs_without_another_child(hosted, fault):
    root, config, cfg, env, service, start, client, wait, prior, captures, births = hosted
    host = start()
    old = rows(root, 'SELECT current_id FROM controller_scope_heads')[0][0]
    if fault == 'source':
        config.write_text(config.read_text() + '\n# changed after admission\n')
    elif fault == 'service':
        service.write_text(service.read_text() + '\n')
    code, output = client('restart', 'change-refused', 'f' * 48 if fault == 'wrong_source' else old)
    assert code == 75, output
    assert len(births) == 1 and host.poll() is None
    assert rows(root, 'SELECT COUNT(*) FROM controller_instances') == [(1,)]
    assert rows(root, 'SELECT request_json FROM controller_sessions') == [(None,)]
