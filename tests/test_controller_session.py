"""Real local session/SQLite/fd protocol tests, not product-loop substitutes.

The tiny host below is a stand-in for orchestration only. Independent product
tests separately invoke actual load_project_config/Orze/run/_run_leased.
These cases use real registration, locks, pidfds, pump and durable ACK paths.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import select
import signal
import sqlite3
import subprocess
import sys
import time

import pytest


SOURCE = str(Path(__file__).resolve().parents[1] / "src")
BOOT = r'''
import json, os, pathlib, sqlite3, sys, threading, time
from types import SimpleNamespace
from orze.idea_lake import IdeaLake
from orze.core import gpu_lease
from orze.engine.controller_control import ControllerHOLD
from orze.engine.controller_session import ControllerSession
replay=os.environ.get('ORZE_TEST_CONTROLLER_SESSION_REPLAY')
if replay:
    module=sys.modules['orze.engine.controller_session']
    exec(compile(pathlib.Path(replay).read_bytes(),replay,'exec'),module.__dict__)
    ControllerSession=module.ControllerSession
root=pathlib.Path(sys.argv[1]); cfg=json.loads(sys.argv[2])
scope=pathlib.Path(cfg['results_dir']); scope.mkdir()
leases=root/'leases'; leases.mkdir()
gpu_lease._lease_dir=lambda: leases
host=SimpleNamespace(cfg=cfg,gpu_ids=[0],results_dir=scope,
    lake=IdeaLake(cfg['idea_lake_db']),_stop_event=threading.Event(),running=True,
    _gpu_leases=None,_leader_handle=None)
session=ControllerSession(host)
host._controller_session=session
pid=scope/'.orze.pid.test'; pid.write_text(str(os.getpid()))
session.bind_pid_file(pid)
owned=gpu_lease.acquire_gpu_leases([0]); session.bind_gpu_leases(owned)
host._gpu_leases=owned
session.start()
def row():
    with sqlite3.connect(cfg['idea_lake_db']) as conn:
        return conn.execute('SELECT request_json,ack_json FROM controller_sessions').fetchone()
'''


def _config(root):
    from orze.core.controller_profile import profile_fingerprint
    cfg = {"controller_control": {"version": 1, "profile": "local_stop_v1"},
        "results_dir": str(root / "results"), "idea_lake_db": str(root / "ideas.db"),
        "gpu_scheduling": {"allowed_gpus": [0]}, "auto_upgrade": False,
        "telemetry": False, "metric_harvest": {"enabled": False},
        "max_fix_attempts": 0, "notifications": {"enabled": False},
        "retrospection": {"enabled": False}}
    cfg["_controller_workdir"] = str(Path.cwd())
    # Unit-level launch-stamp setup, not a claim of actual config-loader use.
    # The independent product fixture uses the real loader instead.
    cfg["_controller_profile_fingerprint"] = profile_fingerprint(cfg)
    return cfg


def _spawn(root, code):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=SOURCE, CUDA_VISIBLE_DEVICES="")
    proc = subprocess.Popen([sys.executable, "-c", BOOT + code, str(root), json.dumps(_config(root))],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pidfd = os.pidfd_open(proc.pid, 0)
    return proc, pidfd


def _close(proc, pidfd):
    try:
        if proc.poll() is None:
            signal.pidfd_send_signal(pidfd, signal.SIGKILL)
        proc.wait(timeout=10)
    finally:
        os.close(pidfd)
        proc.stdout.close()
        proc.stderr.close()


def _run(root, code):
    proc, pidfd = _spawn(root, code)
    try:
        out, err = proc.communicate(timeout=15)
        assert proc.returncode == 0, out + err
    finally:
        _close(proc, pidfd)


def _wait_file(root, name, proc):
    deadline = time.monotonic() + 10
    while not (root / name).exists():
        if proc.poll() is not None:
            out, err = proc.communicate()
            pytest.fail(out + err)
        if time.monotonic() >= deadline:
            pytest.fail("owned test controller handshake timeout")
        time.sleep(0.01)


def test_session_closes_real_resources_and_retains_registered_namespace(tmp_path):
    _run(tmp_path, r'''
fds=owned.fds
ack=session.finish()
assert ack['resources']['gpu_leases']=='closed'
assert ack['members']['member_count']==0
assert host._gpu_leases is None and not pid.exists()
assert (scope/'_controller_registration.lock').is_dir()
assert json.loads(row()[1])==ack
assert session.ctx.quiescing
for fd in fds:
    try: os.fstat(fd)
    except OSError: pass
    else: raise AssertionError('live GPU lease fd')
try: session.finish()
except ControllerHOLD: pass
else: raise AssertionError('finalizer replay accepted')
''')


def test_pinned_stop_pump_does_not_compete_with_action_writer(tmp_path):
    _run(tmp_path, r'''
session.request_local_stop('operator_stop')
expected=row()[0]
peer=sqlite3.connect(cfg['idea_lake_db'])
peer.execute('BEGIN IMMEDIATE')
time.sleep(.7)  # Real writer reservation exceeds the pump connection timeout.
peer.rollback(); peer.close()
assert session.ctx.poll_control()=='QUIESCING'
assert row()[0]==expected and row()[1] is None
assert session.finish()['request_id']==json.loads(expected)['request_id']
''')


@pytest.mark.parametrize("fault", ["gpu_close", "lake_close", "ack_commit", "pid_replaced", "runtime_changed", "request_replaced", "cwd_changed", "session_holder_changed", "gpu_list_changed"])
def test_resource_or_binding_uncertainty_never_writes_ack(tmp_path, fault):
    _run(tmp_path, "fault=" + repr(fault) + r'''
session.request_local_stop('operator_stop')
if fault=='gpu_close':
    owned.close=lambda: None
elif fault=='gpu_list_changed':
    foreign=root/'not_a_controller_lease'
    foreign_fd=os.open(foreign,os.O_RDWR|os.O_CREAT,0o600)
    owned._leases.append(gpu_lease.GpuLease(gpu=1,fd=foreign_fd,path=foreign))
elif fault=='lake_close':
    host.lake.close=lambda: None
elif fault=='ack_commit':
    original_connect=sqlite3.connect
    class LostCommit(sqlite3.Connection):
        def commit(self):
            has_table=self.execute("SELECT 1 FROM sqlite_master WHERE name='controller_sessions'").fetchone()
            has_ack=has_table and self.execute('SELECT 1 FROM controller_sessions WHERE ack_json IS NOT NULL').fetchone()
            if has_ack and self.in_transaction:
                self.rollback()
                return
            super().commit()
    sqlite3.connect=lambda *args,**kwargs: original_connect(*args,factory=LostCommit,**kwargs)
elif fault=='pid_replaced':
    replacement=scope/'replacement'; replacement.write_text(str(os.getpid())); replacement.replace(pid)
elif fault=='runtime_changed':
    cfg['new_public_option']='changed'
elif fault=='cwd_changed':
    os.chdir(root)
elif fault=='session_holder_changed':
    del host._controller_session
elif fault=='request_replaced':
    with sqlite3.connect(cfg['idea_lake_db']) as conn:
        data=json.loads(row()[0]); data['request_id']='f'*48
        conn.execute('UPDATE controller_sessions SET request_json=?',(json.dumps(data),))
try: session.finish()
except ControllerHOLD: pass
else: raise AssertionError('uncertain resource accepted')
assert row()[1] is None
if fault=='gpu_list_changed':
    os.fstat(foreign_fd)  # No authority to close a later-inserted descriptor.
    os.close(foreign_fd)
try: session.ctx.poll_control()
except ControllerHOLD: pass
else: raise AssertionError('HOLD lost')
''')


@pytest.mark.parametrize("mutation", ["none", "bad_digest", "bad_resource", "changed_request"])
def test_observer_requires_bound_ack_and_exact_controller_exit(tmp_path, mutation):
    from orze.engine.controller_control import ControllerHOLD
    from orze.engine.controller_session import CompletedControllerStop, stop_controller
    proc, pidfd = _spawn(tmp_path, r'''
(root/'ready').write_text('ready')
assert host._stop_event.wait(10)
ack=session.finish()
(root/'ack').write_text('ack')
deadline=time.monotonic()+10
while not (root/'exit').exists():
    assert time.monotonic()<deadline
    time.sleep(.01)
''')
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        _wait_file(tmp_path, "ready", proc)
        future = pool.submit(stop_controller, _config(tmp_path), 8)
        _wait_file(tmp_path, "ack", proc)
        assert not select.select([pidfd], [], [], 0)[0]
        assert not future.done(), "ACK was mistaken for controller exit"
        if mutation != "none":
            with sqlite3.connect(tmp_path / "ideas.db") as conn:
                request, ack = conn.execute("SELECT request_json,ack_json FROM controller_sessions").fetchone()
                if mutation == "changed_request":
                    value = json.loads(request)
                    value["request_id"] = "e" * 48
                    conn.execute("UPDATE controller_sessions SET request_json=?", (json.dumps(value),))
                else:
                    value = json.loads(ack)
                    if mutation == "bad_digest":
                        value["members"]["members_sha256"] = "0" * 64
                    else:
                        value["resources"]["gpu_leases"] = "probably_closed"
                    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                    conn.execute("UPDATE controller_sessions SET ack_json=?", (raw,))
        (tmp_path / "exit").write_text("exit")
        proc.wait(timeout=5)
        assert select.select([pidfd], [], [], 0)[0]
        if mutation == "none":
            result = future.result(timeout=5)
            assert type(result) is CompletedControllerStop
            assert result.observed_process["pid"] == proc.pid
            assert result.scope == str(tmp_path / "results")
        else:
            with pytest.raises(ControllerHOLD):
                future.result(timeout=5)
    finally:
        _close(proc, pidfd)
        pool.shutdown(wait=True)


def test_dead_controller_ack_cannot_be_adopted_by_a_fresh_observer(tmp_path):
    from orze.engine.controller_control import ControllerHOLD
    from orze.engine.controller_session import stop_controller
    _run(tmp_path, "session.finish()\n")
    with pytest.raises(ControllerHOLD):
        stop_controller(_config(tmp_path), timeout=1)


def test_configuration_changed_after_loading_cannot_register(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    cfg["poll"] = 123  # Preserve the original loader stamp while inputs drift.
    monkeypatch.setattr(sys.modules[__name__], "_config", lambda root: cfg)
    proc, pidfd = _spawn(tmp_path, "session.finish()\n")
    try:
        out, err = proc.communicate(timeout=10)
        assert proc.returncode != 0, out + err
        assert "controller_loaded_configuration_changed" in err
        assert not (tmp_path / "results" / "_controller_registration.lock").exists()
    finally:
        _close(proc, pidfd)


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), float("inf"), 3601])
def test_observer_rejects_invalid_timeout_without_io(tmp_path, timeout):
    from orze.engine.controller_control import ControllerHOLD
    from orze.engine.controller_session import stop_controller
    with pytest.raises(ControllerHOLD, match="timeout_invalid"):
        stop_controller(_config(tmp_path), timeout)
    assert not list(tmp_path.iterdir())
