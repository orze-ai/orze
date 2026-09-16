"""CPU handoff through the real CLI, native workers, budget and protocol.

The existing CPU stop fixture forbids provider/GPU/legacy control boundaries.
Additional hooks only observe child birth/entry or inject named faults. They do
not replace admission, ownership, settlement, or the successor handshake.
"""
import argparse
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
import textwrap
import time

import pytest
import yaml

from test_cpu_controller_stop_product import SITE, cpu_controller, rows

PROFILE = {"version": 2, "profile": "local_cpu_handoff_v1"}

HOOK = r'''
if 'orze.cli' in getattr(sys,'orig_argv',()):
    import json,socket,subprocess
    from pathlib import Path
    root=Path.cwd()
    successor='ORZE_CONTROLLER_HANDOFF_FD' in os.environ
    if successor:
        channel=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        channel.settimeout(15);channel.connect(str(root/'birth.sock'))
        channel.sendall((json.dumps({'pid':os.getpid(),'parent':os.getppid()})+'\n').encode())
        assert channel.recv(1)==b'L'
        channel.close()
    actual_popen=subprocess.Popen
    def logged_popen(command,*args,**kwargs):
        if list(command[:3])==[sys.executable,'-m','orze.cli']:
            with (root/'successor.log').open('ab') as log:
                kwargs['stdout']=log;kwargs['stderr']=subprocess.STDOUT
                return actual_popen(command,*args,**kwargs)
        return actual_popen(command,*args,**kwargs)
    subprocess.Popen=logged_popen
    from orze.engine import cpu_phase,controller_handoff as handoff
    from orze.engine.controller_session import ControllerSession
    actual_fail=ControllerSession.fail
    def fail(self,exc):
        import traceback
        traceback.print_exception(type(exc),exc,exc.__traceback__)
        return actual_fail(self,exc)
    ControllerSession.fail=fail
    actual_start=cpu_phase.start
    def start(engine):
        marker=root/('start-'+str(os.getpid())+'.json')
        assert not marker.exists(),'CPU start called twice'
        actual_start(engine)
        marker.write_text(json.dumps({'pid':os.getpid(),'successor':successor}))
    cpu_phase.start=start
    actual_iteration=cpu_phase.iteration
    def iteration(engine):
        marker=root/('iteration-'+str(os.getpid())+'.json')
        if not marker.exists():
            if successor:
                admission=engine._controller_session._admission
                row=engine.lake.conn.execute('SELECT state FROM controller_handoffs WHERE grant_id=?',
                    (admission.grant_id,)).fetchone()
                assert row[0]=='STARTED','policy reached before COMMIT'
            marker.write_text(json.dumps({'pid':os.getpid(),'successor':successor}))
        return actual_iteration(engine)
    cpu_phase.iteration=iteration
    fault=os.environ.get('CPU_HANDOFF_FAULT','')
    actual_receive=handoff._receive
    def receive(channel,peer_pid,deadline):
        packet=actual_receive(channel,peer_pid,deadline)
        event=packet['event']
        if successor and event=='COMMIT' and fault=='orphan_at_commit':
            from orze.engine.controller_control import current_controller
            from orze.engine.controller_session import _SESSIONS
            from orze.core import cpu_action_budget as budget
            engine=_SESSIONS[current_controller().controller_id].orze
            assert budget.reserve(engine.lake,engine._cpu_scope,'orphan-commit',1) is not None
            (root/'fault.json').write_text(json.dumps({'event':event,'fault':fault}))
        if not successor and event=='STARTED' and fault=='lost_started_reply':
            (root/'fault.json').write_text(json.dumps({'event':event,'fault':fault}))
            raise OSError('fixture: lost final STARTED reply after real receive')
        return packet
    handoff._receive=receive
'''


@pytest.fixture
def project(cpu_controller):
    root, path, cfg, env, launch, wait_for = cpu_controller
    cfg['controller_control'] = PROFILE
    path.write_text(yaml.safe_dump(cfg))
    (root / 'successor.log').touch()
    (root / 'sitecustomize.py').write_text('try:\n' + textwrap.indent(SITE + HOOK, '    ') +
        '\nexcept BaseException as exc:\n'
        '    import os,sys\n'
        '    print("fixture hook failed: "+repr(exc),file=sys.stderr,flush=True)\n'
        '    os._exit(97)\n')
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(root / 'birth.sock'))
    listener.listen()
    listener.settimeout(3)
    captured, controls, births = [], [], []
    expected_old = [None]

    def control(command, key=None, fault=''):
        argv = [sys.executable, '-m', 'orze.cli', command, '-c', str(path), '--timeout', '15']
        if key is not None:
            argv += ['--request-id', key]
        child = subprocess.Popen(argv, cwd=root, env=dict(env, CPU_HANDOFF_FAULT=fault),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        controls.append(child)
        deadline = time.monotonic() + 20
        while child.poll() is None and time.monotonic() < deadline:
            if select.select([listener], [], [], .02)[0]:
                channel, _ = listener.accept()
                with channel:
                    raw = b''
                    while not raw.endswith(b'\n'):
                        raw += channel.recv(1024)
                    birth = json.loads(raw)
                    pid, uid, _ = struct.unpack('3i', channel.getsockopt(
                        socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                    assert pid == birth['pid'] and uid == os.getuid()
                    assert birth['parent'] == child.pid
                    fd = os.pidfd_open(pid)
                    captured.append(fd)
                    # Capture only an actual child of this directly owned issuer.
                    assert expected_old[0] is not None
                    assert select.select([expected_old[0]], [], [], 0)[0]
                    births.append({**birth, 'fd': fd, 'old_exited_before_birth': True})
                    channel.sendall(b'L')
        output, _ = child.communicate(timeout=3)
        (root / ('control-%d.log' % len(controls))).write_text(output)
        return child.returncode, output

    try:
        yield root, path, cfg, env, launch, wait_for, control, births, expected_old
    finally:
        for child in controls:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)
        for fd in captured:
            if not select.select([fd], [], [], 0)[0]:
                signal.pidfd_send_signal(fd, signal.SIGTERM)
                if not select.select([fd], [], [], 10)[0]:
                    signal.pidfd_send_signal(fd, signal.SIGKILL)
                    assert select.select([fd], [], [], 5)[0]
            os.close(fd)
        listener.close()


def test_cpu_handoff_profile_uses_separate_protocol_and_control_cli(tmp_path, monkeypatch):
    from test_cpu_controller_profile import supported
    from orze.core.config import load_project_config
    from orze.core.controller_profile import controller_profile, validate_profile_cli
    from orze.core.cpu_execution import CPUExecutionError, validate_cpu_cli
    monkeypatch.chdir(tmp_path)
    path = tmp_path / 'orze.yaml'
    path.write_text(yaml.safe_dump({**supported(), 'controller_control': PROFILE}))
    cfg = load_project_config(str(path))
    assert controller_profile(cfg) == PROFILE
    for args in (argparse.Namespace(command='restart', timeout=12),
                 argparse.Namespace(command=None, restart=True, timeout=12)):
        validate_cpu_cli(cfg, args)
        assert validate_profile_cli(cfg, args) == PROFILE
    for args in (argparse.Namespace(command='start'), argparse.Namespace(command='resume'),
                 argparse.Namespace(command=None, timeout=12)):
        with pytest.raises(CPUExecutionError):
            validate_cpu_cli(cfg, args)


def _active(root):
    return rows(root, 'SELECT phase FROM controller_instances ORDER BY generation DESC LIMIT 1') == [('ACTIVE',)]


def _settled(root, number):
    return rows(root, "SELECT COUNT(*) FROM cpu_action_reservations WHERE state='SETTLED'") == [(number,)]


def _attempt_count(root):
    exists = rows(root, "SELECT 1 FROM sqlite_master WHERE name='execution_attempts'")
    return rows(root, 'SELECT COUNT(*) FROM execution_attempts')[0][0] if exists else 0


def test_cpu_handoff_replays_once_and_keeps_older_task_generation_valid(project):
    from orze.core.config import load_project_config
    from orze.engine.controller_handoff import _Route, _validate_history
    from orze.idea_lake import IdeaLake
    root, path, cfg, env, launch, wait_for, control, births, expected_old = project
    marker = root / 'first-worker.json'
    program = ("import os,json,subprocess,sys,time\nfrom pathlib import Path\n"
        f"marker=Path({str(marker)!r})\n"
        "if not marker.exists():\n"
        " child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'])\n"
        " marker.write_text(json.dumps([os.getpid(),child.pid]))\n"
        " time.sleep(120)\n")
    old = launch(program)
    wait_for(old, marker.exists)
    fds = [os.pidfd_open(pid) for pid in [old.pid, *json.loads(marker.read_text())]]
    expected_old[0] = fds[0]
    try:
        code, output = control('restart', 'cpu-first')
        assert code == 0, output + (root / 'successor.log').read_text()
        assert old.wait(timeout=5) == 0
        assert all(select.select([fd], [], [], 0)[0] for fd in fds)
        assert len(births) == 1
        assert _settled(root, 1)
        source_id = rows(root, 'SELECT controller_id FROM controller_instances WHERE generation=0')[0][0]
        original_ack = rows(root, 'SELECT ack_json FROM controller_sessions WHERE controller_id=' +
                            repr(source_id))[0][0]
        # Explicit, audited retry of a FAILED task; no automatic adoption/retry.
        lake = IdeaLake(root / 'lake.db')
        try:
            assert lake.record_state_transition('idea-0001', 'FAILED', 'QUEUED',
                                                reason='isolated operator retry')
        finally:
            lake.close()
        deadline = time.monotonic() + 15
        while not _settled(root, 2) and time.monotonic() < deadline:
            time.sleep(.03)
        assert _settled(root, 2), (root / 'successor.log').read_text()
        assert rows(root, 'SELECT generation,state FROM execution_attempts ORDER BY generation') == [
            (1, 'TERMINAL'), (2, 'TERMINAL')]
        assert rows(root, 'SELECT ack_json FROM controller_sessions WHERE controller_id=' +
                    repr(source_id))[0][0] == original_ack
        route = _Route(load_project_config(str(path)))
        _validate_history(route, source_id, 0, closed_inventory=False)
        code, output = control('restart', 'cpu-first')
        assert code == 0, output
        assert len(births) == 1
        assert rows(root, 'SELECT COUNT(*) FROM execution_attempts') == [(2,)]
        expected_old[0] = births[-1]['fd']
        code, output = control('restart', 'cpu-second')
        assert code == 0, output + (root / 'successor.log').read_text()
        assert len(births) == 2
        assert rows(root, 'SELECT state FROM controller_handoffs') == [('STARTED',), ('STARTED',)]
        assert control('stop')[0] == 0
        head = rows(root, 'SELECT current_id,generation FROM controller_scope_heads')[0]
        assert head[1] == 2
        _validate_history(route, head[0], head[1])
        assert rows(root, 'SELECT SUM(CAST(json_extract(permit_json,\'$.reserved_nanoseconds\') AS INTEGER)) '
                    'FROM cpu_action_reservations') == [(120_000_000_000,)]
        assert len(list(root.glob('start-*.json'))) == 3
        assert len(list(root.glob('iteration-*.json'))) == 3
        assert not list((root / 'results').glob('.orze.pid*'))
        (root / 'product-summary.json').write_text(json.dumps({
            'controller_generations': 3, 'native_attempts': 2, 'task_generations': [1, 2],
            'successors': 2, 'replay_extra_successors': 0, 'charged_nanoseconds': 120_000_000_000,
            'old_tree_exited_before_successor': True}))
    finally:
        for fd in fds:
            os.close(fd)


@pytest.mark.parametrize('fault', ['orphan_at_commit', 'lost_started_reply'])
def test_cpu_handoff_unknown_commit_or_reply_never_spawns_again(project, fault):
    root, path, cfg, env, launch, wait_for, control, births, expected_old = project
    old = launch()
    wait_for(old, lambda: _active(root) and (root / ('start-%d.json' % old.pid)).exists())
    old_fd = os.pidfd_open(old.pid)
    expected_old[0] = old_fd
    try:
        code, output = control('restart', 'cpu-fault', fault)
        assert code == 75, output
        assert old.wait(timeout=5) == 0
        assert len(births) == 1
        assert (root / 'fault.json').exists()
        replay, replay_output = control('restart', 'cpu-fault')
        assert len(births) == 1
        assert _attempt_count(root) == 0
        if fault == 'orphan_at_commit':
            assert replay == 75, replay_output
            assert rows(root, 'SELECT state FROM controller_handoffs') == [('PREPARED',)]
            assert rows(root, 'SELECT state FROM cpu_action_reservations') == [('RESERVED',)]
            assert not (root / ('iteration-%d.json' % births[0]['pid'])).exists()
        else:
            assert replay == 0, replay_output
            assert rows(root, 'SELECT state FROM controller_handoffs') == [('STARTED',)]
            assert control('stop')[0] == 0
    finally:
        os.close(old_fd)
