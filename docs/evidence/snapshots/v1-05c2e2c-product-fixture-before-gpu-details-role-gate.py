"""Fresh-exec Orze product boundary tests, not a simulated controller loop.

Only accelerator telemetry/capacity and the host GPU lease directory are
isolated. SQLite, claims, GPU lease FDs/flock, Orze lifecycle, worker execution,
supervisor protocol, and the observer's captured controller pidfd are real.
All programs are tiny temporary CPU fixtures, not training or research jobs.
"""
import ctypes
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
import threading
import time

import pytest
import yaml


IDEA = "idea-controller-cpu"
ROLE = "controller_cpu_role"
CORE_SOURCE = Path(__file__).resolve().parents[1] / "src"
PRO_SOURCE = CORE_SOURCE.parent.parent / "orze-pro" / "src"

WORKER = r'''
import argparse,json,os,signal,socket,sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--idea-id',default='idea-controller-cpu')
p.add_argument('--seed',type=int,default=13)
p.add_argument('--results-dir')
p.add_argument('--fixture-root',required=True)
p.add_argument('--fixture-socket',required=True)
p.add_argument('--fixture-phase',default='training')
p.add_argument('--fixture-mode',default='normal')
a,unused=p.parse_known_args()
root=Path(a.fixture_root)
output=root/(a.fixture_phase+'.bin')
fd=os.open(output,os.O_CREAT|os.O_TRUNC|os.O_RDWR,0o600)
os.write(fd,b'owned synthetic CPU output');os.fsync(fd)
(root/(a.fixture_phase+'.go')).write_text(str(os.getpid()))
if a.fixture_phase=='training':
    folder=Path(a.results_dir)/a.idea_id
    with (folder/'metrics.json').open('w') as f:
        json.dump({'status':'COMPLETED','score':0},f);f.flush();os.fsync(f.fileno())
channel=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
channel.settimeout(25);channel.connect(a.fixture_socket)
channel.sendall((json.dumps({'event':'worker','pid':os.getpid(),
    'phase':a.fixture_phase,'inode':os.fstat(fd).st_ino})+'\n').encode())
assert channel.recv(1)==b'L'
channel.close()
if a.fixture_mode=='normal':
    os.close(fd);sys.exit(0)
if a.fixture_mode=='sleep':
    signal.pause();sys.exit(0)
read_end,write_end=os.pipe()
os.set_inheritable(fd,True);os.set_inheritable(write_end,True)
middle=os.fork()
if middle:
    os.close(write_end);os.close(fd)
    assert os.read(read_end,1)==b'L'
    os.close(read_end);os.waitpid(middle,0);os._exit(0)
os.close(read_end);os.setsid()
if os.fork():os._exit(0)
for stream in (0,1,2):
    try:os.close(stream)
    except OSError:pass
daemon=r"""
import json,os,socket,sys
endpoint,descriptor,release,phase=sys.argv[1:]
descriptor,release=int(descriptor),int(release)
c=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);c.settimeout(25);c.connect(endpoint)
c.sendall((json.dumps({'event':'daemon','pid':os.getpid(),'pgid':os.getpgrp(),
    'phase':phase,'inode':os.fstat(descriptor).st_ino,
    'orze_keys':[k for k in os.environ if k.startswith('ORZE_')]})+'\n').encode())
assert c.recv(1)==b'L'
os.write(release,b'L');os.close(release);c.sendall(b'R')
try:
    while True:
        command=c.recv(1)
        if not command or command==b'Q':break
        if command==b'W':
            os.write(descriptor,b' late writer');os.fsync(descriptor);c.sendall(b'W')
finally:os.close(descriptor);c.close()
"""
os.execve(sys.executable,[sys.executable,'-c',daemon,a.fixture_socket,
    str(fd),str(write_end),a.fixture_phase],{})
'''


CONTROLLER = r'''
import json,os,socket,sys,threading,traceback
from pathlib import Path
root=Path(sys.argv[1]);endpoint=sys.argv[2]
channel=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);channel.connect(endpoint)
write_lock=threading.Lock();ready={};controller=None;owned={os.getpid()}
def tell(event,**data):
    with write_lock:channel.sendall((json.dumps({'event':event,**data})+'\n').encode())
def commands():
    stream=channel.makefile('rb')
    for line in stream:
        value=json.loads(line)
        if value['op']=='release':ready[value['token']].set()
        elif value['op']=='snapshot':
            tell('snapshot',active=len(controller.active),evals=len(controller.active_evals),
                 roles=len(controller.active_roles),iteration=controller.iteration)
        elif value['op']=='erase_roles':
            names=list(controller.active_roles);controller.active_roles.clear()
            tell('erased_roles',names=names)
        elif value['op']=='own_pid':owned.add(value['pid'])
threading.Thread(target=commands,daemon=True).start()

# Every accelerator query is an explicit hardware boundary double. The GPU
# lease object, flock, inherited FDs, resource ownership and close stay real.
from orze.core import gpu_lease
lease_dir=root/'fixture-gpu-leases';lease_dir.mkdir(mode=0o700)
gpu_lease._lease_dir=lambda:lease_dir
def idle(gpus):
    assert list(gpus)==[0]
    tell('gpu_idle_spy',gpus=list(gpus))
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
from orze.engine import phases
phases.get_gpu_memory_used=hardware_gpu.get_gpu_memory_used
from orze import extensions
# License/auto-install discovery is an external-service boundary, not the
# role execution path. The actual source role_runner remains installed below.
extensions.has_pro=lambda **kwargs:False

# Any remaining legacy discovery may inspect only already authenticated own
# PIDs. Preserve actual /proc stat/environ and never hide a captured daemon.
actual_iterdir=Path.iterdir
def own_iterdir(path):
    if path==Path('/proc'):return iter(Path('/proc')/str(pid) for pid in sorted(owned))
    return actual_iterdir(path)
Path.iterdir=own_iterdir
actual_listdir=os.listdir
os.listdir=lambda path:([str(pid) for pid in sorted(owned)] if str(path)=='/proc' else actual_listdir(path))
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
    identity=kwargs['identity'];ref=identity.get('attempt_ref') or {}
    phase=ref.get('phase') or ('role' if 'role_name' in identity else 'other')
    token=str(len(ready));ready[token]=threading.Event()
    worker=handle.pid;supervisor=handle._supervisor.pid
    owned.update((worker,supervisor))
    tell('ready',token=token,phase=phase,worker=worker,supervisor=supervisor)
    assert ready[token].wait(20),'fixture observer did not release actual READY'
    return handle
launcher.prepare_supervised=captured_prepare
process.prepare_supervised=captured_prepare
# Explicit offline package-license seam, as in the Pro test harness. No key,
# activation service, provider, role runner, or process proof is substituted.
from orze_pro import _gate
_gate.require_license=lambda:None
from orze_pro.engine import role_runner
role_runner.prepare_supervised=captured_prepare

try:
    from orze.core.config import load_project_config
    from orze.engine.orchestrator import Orze
    cfg=load_project_config(str(root/'orze.yaml'))
    cfg['_config_path']=str(root/'orze.yaml')
    controller=Orze([0],cfg,once=False)
    tell('constructed',pid=os.getpid(),lake=type(controller.lake).__name__)
    controller.run()
    tell('returned',iteration=controller.iteration)
except BaseException as exc:
    tell('controller_error',kind=type(exc).__name__,message=str(exc))
    traceback.print_exc()
    raise
finally:channel.close()
'''


OBSERVER = r'''
from dataclasses import asdict
import json,sys
from orze.core.config import load_project_config
from orze.engine.controller_control import ControllerHOLD
from orze.engine.controller_session import CompletedControllerStop,stop_controller
try:
    result=stop_controller(load_project_config(sys.argv[1]),timeout=float(sys.argv[2]))
    assert isinstance(result,CompletedControllerStop)
    print(json.dumps({'completed':True,'record':asdict(result)}))
except ControllerHOLD as exc:
    print(json.dumps({'completed':False,'reason':str(exc)}))
'''


def _alive(fd):
    poller = select.poll()
    poller.register(fd, select.POLLIN | select.POLLHUP)
    return not poller.poll(0)


def _line(channel):
    raw = bytearray()
    while not raw.endswith(b"\n"):
        piece = channel.recv(1)
        assert piece, "fixture channel closed before complete message"
        raw.extend(piece)
        assert len(raw) <= 16384
    return json.loads(raw)


class ProductProcess:
    def __init__(self, root, mode):
        self.root, self.mode = root, mode
        self.pidfds, self.channels, self.events, self.handles = {}, [], [], []
        self.observers = []
        self.pause_ready = mode == "ready_stop"
        self.results, self.db = root / "results", root / ".orze" / "ideas.db"
        self.results.mkdir()
        self.db.parent.mkdir()
        (root / "base.yaml").write_text("{}\n")
        (root / "worker.py").write_text(WORKER)
        self.worker_listener = self._listener(root / "worker.sock")
        listener = self._listener(root / "controller.sock")
        args = ["--fixture-root", str(root), "--fixture-socket", str(root / "worker.sock")]
        self.cfg = {
            "controller_control": {"version": 1, "profile": "local_stop_v1"},
            "results_dir": str(self.results), "idea_lake_db": str(self.db),
            "ideas_file": str(root / ".orze" / "ideas.md"),
            "base_config": str(root / "base.yaml"), "train_script": str(root / "worker.py"),
            "python": sys.executable, "train_extra_args": args + ["--fixture-mode", mode if mode == "escaped" else "normal"],
            "gpu_scheduling": {"allowed_gpus": [0], "mode": "exclusive"},
            "poll": 0.05, "timeout": 30, "pre_timeout": 30,
            "role_presets": [], "roles": {}, "telemetry": False,
            "bot": None, "telegram_bot": None, "notifications": {"enabled": False},
            "auto_upgrade": False, "retrospection": {"enabled": False},
            "cleanup": {"script": None, "interval": 0, "patterns": []},
            "metric_harvest": {"enabled": False, "llm_fallback": False},
            "max_fix_attempts": 0, "sweep_stray": False, "auto_seal_eval": False,
            "stall_minutes": 0, "role_stall_minutes": 0,
            "report": {"primary_metric": "score"},
        }
        ideas = "# Ideas\n\n## " + IDEA + ": Tiny CPU control fixture\n\n```yaml\nseed: 13\n```\n"
        if mode == "pre_loss":
            self.cfg.update(pre_script=str(root / "worker.py"),
                            pre_args=args + ["--fixture-phase", "pre_script", "--fixture-mode", "sleep"])
        if mode == "role_loss":
            ideas = "# Ideas\n"
            self.cfg["roles"] = {ROLE: {"mode": "script", "script": str(root / "worker.py"),
                "args": args + ["--fixture-phase", "role", "--fixture-mode", "sleep"],
                "timeout": 30, "cooldown": 900, "stall_minutes": 0,
                "triggered_by": "fixture", "auto_trigger_on_stall": False}}
            # Actual orze_path(...,'triggers') is the public results root,
            # not the internal .orze directory.
            (self.results / ("_trigger_" + ROLE)).write_text("Run this tiny CPU fixture once.\n")
        Path(self.cfg["ideas_file"]).write_text(ideas)
        (root / "orze.yaml").write_text(yaml.safe_dump(self.cfg))
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": os.pathsep.join([str(CORE_SOURCE), str(PRO_SOURCE)]),
               "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
        self.environment = env
        self.log = (root / "controller.log").open("wb")
        self.child = subprocess.Popen([sys.executable, "-c", CONTROLLER, str(root), str(root / "controller.sock")],
            cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT)
        self.controller_fd = self.capture(self.child.pid)
        self.channel, _ = listener.accept()
        self.channel.settimeout(5)
        self.channels.append(self.channel)
        pid, uid, _ = struct.unpack("3i", self.channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        assert pid == self.child.pid and uid == os.getuid()

    def _listener(self, path):
        channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        channel.bind(str(path)); channel.listen(4); channel.settimeout(15)
        self.channels.append(channel)
        return channel

    def capture(self, pid):
        if pid not in self.pidfds:
            self.pidfds[pid] = os.pidfd_open(pid)
        return self.pidfds[pid]

    def command(self, **value):
        self.channel.sendall((json.dumps(value) + "\n").encode())

    def pump(self):
        if not select.select([self.channel], [], [], 0.02)[0]:
            return
        value = _line(self.channel)
        if value["event"] == "ready":
            self.capture(value["worker"]); self.capture(value["supervisor"])
            self.handles.append(value)
            if not self.pause_ready:
                self.command(op="release", token=value["token"])
        self.events.append(value)

    def wait(self, predicate, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            if not _alive(self.controller_fd):
                break
            self.pump()
        assert predicate(), (self.events, (self.root / "controller.log").read_text()[-16000:])

    def event(self, name):
        self.wait(lambda: any(x["event"] == name for x in self.events))
        return next(x for x in self.events if x["event"] == name)

    def worker(self, *, daemon=False):
        self.event("ready")
        channel, _ = self.worker_listener.accept(); channel.settimeout(5)
        self.channels.append(channel)
        pid, uid, _ = struct.unpack("3i", channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        value = _line(channel)
        assert value["pid"] == pid and uid == os.getuid()
        assert value["event"] == ("daemon" if daemon else "worker")
        assert value["inode"] == (self.root / (value["phase"] + ".bin")).stat().st_ino
        self.capture(pid)
        self.command(op="own_pid", pid=pid)
        channel.sendall(b"L")
        if daemon:
            assert value["orze_keys"] == []
            assert channel.recv(1) == b"R"
        return channel, value

    def rows(self, table):
        with sqlite3.connect(self.db.as_uri() + "?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(x) for x in conn.execute("SELECT * FROM " + table)]

    def stop(self, timeout=8):
        from orze.engine.controller_control import ControllerHOLD
        child = subprocess.Popen([sys.executable, "-c", OBSERVER,
            str(self.root / "orze.yaml"), str(timeout)], cwd=self.root,
            env=self.environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.observers.append(child)
        self.capture(child.pid)
        stdout, stderr = child.communicate(timeout=timeout + 5)
        assert child.returncode == 0, stderr.decode()[-12000:]
        value = json.loads(stdout)
        if not value["completed"]:
            raise ControllerHOLD(value["reason"])
        return value

    def close(self):
        for channel in self.channels:
            try: channel.close()
            except OSError: pass
        for pid, fd in self.pidfds.items():
            if _alive(fd):
                signal.pidfd_send_signal(fd, signal.SIGKILL)
        if hasattr(self, "child"):
            self.child.wait(timeout=5)
        for child in self.observers:
            child.wait(timeout=5)
        for pid, fd in self.pidfds.items():
            poller = select.poll(); poller.register(fd, select.POLLIN | select.POLLHUP)
            assert poller.poll(5000), ("owned process failed cleanup", pid)
            try: os.waitpid(pid, 0)
            except ChildProcessError: pass
            os.close(fd)
        if hasattr(self, "log"):
            self.log.close()


@pytest.fixture
def product_process(tmp_path):
    if sys.platform != "linux" or not hasattr(os, "pidfd_open"):
        pytest.skip("requires Linux own pidfds")
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    instances = []
    def spawn(mode):
        root = tmp_path / mode
        root.mkdir()
        c = ProductProcess.__new__(ProductProcess)
        instances.append(c)
        c.__init__(root, mode)
        return c
    try:
        yield spawn
    finally:
        for c in reversed(instances):
            c.close()
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


@pytest.mark.parametrize("mode", ["normal", "escaped"])
def test_real_controller_stop_requires_drained_tree_and_observed_exit(product_process, mode):
    c = product_process(mode)
    assert c.event("constructed")["lake"] == "IdeaLake"
    c.worker()
    if mode == "escaped":
        channel, daemon = c.worker(daemon=True)
        assert _alive(c.pidfds[daemon["pid"]])
        channel.sendall(b"W"); assert channel.recv(1) == b"W"
        assert not list((c.results / IDEA).glob("_compute_receipts/*/terminal.json"))
    else:
        c.wait(lambda: any(row["state"] == "TERMINAL" for row in c.rows("execution_attempts")))
    result = c.stop()
    assert result["completed"] is True
    assert not _alive(c.controller_fd), "ACK alone is not captured controller exit"
    assert c.child.wait(timeout=5) == 0
    sessions = c.rows("controller_sessions")
    assert len(sessions) == 1 and sessions[0]["request_json"] and sessions[0]["ack_json"]
    assert all(row["state"] == "TERMINAL" for row in c.rows("execution_attempts"))
    assert all(not _alive(fd) for fd in c.pidfds.values())


def test_lost_sync_pre_script_cannot_ack_from_three_empty_public_maps(product_process):
    from orze.engine.controller_control import ControllerHOLD
    c = product_process("pre_loss")
    c.worker()
    c.command(op="snapshot")
    view = c.event("snapshot")
    assert (view["active"], view["evals"], view["roles"]) == (0, 0, 0)
    ready = next(x for x in c.handles if x["phase"] == "pre_script")
    signal.pidfd_send_signal(c.pidfds[ready["supervisor"]], signal.SIGKILL)
    with pytest.raises(ControllerHOLD):
        c.stop(timeout=2)
    assert not (c.root / "training.go").exists()
    assert c.rows("controller_sessions")[0]["ack_json"] is None
    assert any(row["state"] != "TERMINAL" for row in c.rows("execution_attempts"))


def test_erased_public_role_cannot_hide_lost_private_owner(product_process):
    from orze.engine.controller_control import ControllerHOLD
    c = product_process("role_loss")
    c.worker()
    c.command(op="snapshot")
    assert c.event("snapshot")["roles"] == 1
    c.command(op="erase_roles")
    assert c.event("erased_roles")["names"] == [ROLE]
    ready = c.handles[0]
    signal.pidfd_send_signal(c.pidfds[ready["supervisor"]], signal.SIGKILL)
    with pytest.raises(ControllerHOLD):
        c.stop(timeout=2)
    assert c.rows("controller_sessions")[0]["ack_json"] is None
    assert not (c.root / "training.go").exists()
    assert len(c.handles) == 1


def test_stop_request_at_real_ready_prevents_new_training_go(product_process):
    from orze.engine.controller_control import ControllerHOLD
    c = product_process("ready_stop")
    ready = c.event("ready")
    assert not (c.root / "training.go").exists()
    outcome = []
    def request():
        try: outcome.append(c.stop(timeout=5))
        except ControllerHOLD as exc: outcome.append(exc)
    observer = threading.Thread(target=request)
    observer.start()
    try:
        c.wait(lambda: bool(c.rows("controller_sessions")[0]["request_json"]))
        c.command(op="release", token=ready["token"])
        observer.join(timeout=7)
        assert not observer.is_alive() and len(outcome) == 1
        assert not (c.root / "training.go").exists()
        assert len(c.handles) == 1
    finally:
        observer.join(timeout=7)
