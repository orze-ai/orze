"""Real CPU admission-hook regressions through the actual dispatch phase.

The temporary SQLite task, claim, config materialization and failure/accounting
paths are real. Training and fixer/provider are explicit terminal spy boundaries;
no training/model/GPU workload is run. Tiny scripts own only fixture output.
SO_PEERCRED and retained own pidfds authenticate cleanup, never a host scan.
"""
import ctypes
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from orze.core.evaluation_retry_state import open_existing_lake
from orze.engine import phases, process
from orze.engine.supervised_process import prepare_supervised as REAL_PREPARE
from orze.engine.termination_hold import TerminationUnconfirmed
from orze.idea_lake import IdeaLake
from test_native_training_tree_completion import _alive, _wait_dead
from test_native_posthoc_tree_completion import DAEMON_SOURCE


REAL_POPEN = subprocess.Popen
WORKER_SOURCE = r'''
import json, os, signal, socket, sys
endpoint, path, mode, code = sys.argv[1:]
descriptor = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_RDWR, 0o600)
os.write(descriptor, b"synthetic-pre-script-output")
os.fsync(descriptor)
if mode == "normal":
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    channel.settimeout(20)
    channel.connect(endpoint)
    channel.sendall((json.dumps({"pid": os.getpid(), "pgid": os.getpgrp(),
        "inode": os.fstat(descriptor).st_ino, "path": path,
        "gpu_env": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")}}) + "\n").encode())
    assert channel.recv(1) == b"L"
    channel.sendall(b"R")
    channel.close()
    os.close(descriptor)
    sys.exit(int(code))
read_end, write_end = os.pipe()
os.set_inheritable(write_end, True)
os.set_inheritable(descriptor, True)
middle = os.fork()
if middle:
    os.close(write_end)
    os.close(descriptor)
    assert os.read(read_end, 1) == b"L"
    os.close(read_end)
    os.waitpid(middle, 0)
    if mode == "timeout":
        signal.pause()
    os._exit(int(code))
os.close(read_end)
os.setsid()
if os.fork():
    os._exit(0)
# A surviving writer does not need stdout/stderr. Closing them makes the old
# communicate() return on leader exit, not accidentally wait for daemon EOF.
for stream_fd in (0, 1, 2):
    try:
        os.close(stream_fd)
    except OSError:
        pass
os.execve(sys.executable, [sys.executable, "-c", DAEMON_SOURCE,
    endpoint, str(descriptor), str(write_end), path], {})
'''


@pytest.fixture
def cpu_pre_script(tmp_path, monkeypatch):
    if sys.platform != "linux" or not hasattr(os, "pidfd_open"):
        pytest.skip("requires Linux pidfds for exact own-process cleanup")
    results = tmp_path / "results"
    results.mkdir()
    script, train, base, inbox = (tmp_path / name for name in
                                 ("prepare.py", "unused_train.py", "base.yaml", "ideas.md"))
    script.write_text("DAEMON_SOURCE = " + repr(DAEMON_SOURCE) + "\n" + WORKER_SOURCE)
    train.write_text("# never executed: terminal training boundary is a spy\n")
    base.write_text("{}\n")
    inbox.write_text("# Ideas\n")
    cfg = {"results_dir": str(results), "_project_root": str(tmp_path),
           "_orze_dir": str(tmp_path / ".orze"), "idea_lake_db": str(tmp_path / "ideas.db"),
           "ideas_file": str(inbox), "base_config": str(base), "train_script": str(train),
           "python": sys.executable, "pre_script": str(script), "pre_timeout": 10,
           "train_extra_env": {"ORZE_CPU_TREE_TEST": "must-not-survive-exec",
                               "CUDA_VISIBLE_DEVICES": "4"},
           "sealed_files": [], "gpu_mem_threshold": 2000, "sweep": {}, "gc": {},
           "artifact_preflight": {"enabled": False}, "timeout": 60}
    lake = IdeaLake(cfg["idea_lake_db"])
    idea = "idea-cpu-pre-script"
    lake.insert(idea, "Synthetic CPU admission", "seed: 13\n", "", status="queued")
    c = SimpleNamespace(cfg=cfg, results=results, lake=lake, idea=idea,
                        folder=results / idea, roots=[], pidfds=[], channels=[],
                        worker_pid=None, worker_pidfd=None, daemon_pid=None,
                        daemon_pidfd=None, channel=None, events=[], errors=[],
                        stops=[], thread=None, done=threading.Event())
    c.output = c.folder / "prepared.bin"
    c.runner = SimpleNamespace(cfg=cfg, results_dir=results, lake=None,
        active={}, active_evals={}, gpu_ids=[4], failure_counts={}, fix_counts={}, once=True)
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0

    def live():
        return c.daemon_pidfd is not None and _alive(c.daemon_pidfd)

    def popen(*args, **kwargs):
        child = REAL_POPEN(*args, **kwargs)
        c.roots.append(child)
        fd = os.pidfd_open(child.pid)
        c.pidfds.append((child.pid, fd))
        if c.worker_pid is None:
            c.worker_pid, c.worker_pidfd = child.pid, fd
        return child

    def prepare(*args, **kwargs):
        child = REAL_PREPARE(*args, **kwargs)
        fd = os.pidfd_open(child.pid)
        c.pidfds.append((child.pid, fd))
        if len(c.roots) == 1:
            c.worker_pid, c.worker_pidfd = child.pid, fd
        return child

    reaper = process._terminate_and_reap
    def reap(*args, **kwargs):
        result = reaper(*args, **kwargs)
        c.stops.append({"return": result, "writer_alive_after_return": live()})
        return result

    def train_boundary(idea_id, gpu, *args, **kwargs):
        c.events.append(("training_launch", live()))
        return SimpleNamespace(idea_id=idea_id, gpu=gpu)

    def fixer_boundary(*args, **kwargs):
        c.events.append(("fixer", live()))
        return False  # No provider, code repair, or second worker is executed.

    monkeypatch.setattr(process.subprocess, "Popen", popen)
    # Transparent real READY ownership capture; the old module ignores this
    # seam. No assertion assumes a new API or synthetic closure proof.
    monkeypatch.setattr(process, "prepare_supervised", prepare, raising=False)
    monkeypatch.setattr(process, "_terminate_and_reap", reap)
    monkeypatch.setattr(phases, "run_pre_script", process.run_pre_script)
    monkeypatch.setattr(phases, "get_gpu_memory_used", lambda gpu: 0)
    monkeypatch.setattr(phases, "launch", train_boundary)
    monkeypatch.setattr(phases, "_try_executor_fix", fixer_boundary)
    monkeypatch.setattr("orze.extensions.get_extension", lambda name: None)
    for name in ("_reset_idea_for_retry", "_write_failure", "record_zero_gpu_outcome"):
        actual = getattr(phases, name)
        def observe(*args, _actual=actual, _name=name, **kwargs):
            c.events.append((_name, live()))
            return _actual(*args, **kwargs)
        monkeypatch.setattr(phases, name, observe)
    try:
        yield c
    finally:
        if c.channel is not None and c.daemon_pidfd is not None and _alive(c.daemon_pidfd):
            try:
                c.channel.sendall(b"Q")
            except OSError:
                pass
        for channel in c.channels:
            channel.close()
        for _, fd in c.pidfds:
            if _alive(fd):
                signal.pidfd_send_signal(fd, signal.SIGKILL)
            _wait_dead(fd)
        if c.thread is not None:
            c.thread.join(timeout=5)
            assert not c.thread.is_alive(), "owned CPU admission controller did not finish"
        for child in c.roots:
            child.wait(timeout=5)
        if c.daemon_pid is not None:
            try:
                os.waitpid(c.daemon_pid, 0)
            except ChildProcessError:
                pass
        for _, fd in c.pidfds:
            os.close(fd)
        lake.close()
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


def _start(c, tmp_path, *, mode, code=0):
    endpoint = tmp_path / "pre-script.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(endpoint))
    listener.listen(1)
    listener.settimeout(5)
    c.channels.append(listener)
    c.cfg["pre_args"] = [str(endpoint), str(c.output), mode, str(code)]
    if mode == "timeout":
        c.cfg["pre_timeout"] = 1

    def controller():
        authority = None
        try:
            authority = open_existing_lake(c.lake.db_path)
            c.runner.lake = authority
            phases.OrzePhaseMixin._launch_training(c.runner, [c.idea], True,
                {c.idea: {"title": "Synthetic CPU admission", "priority": "high", "config": {"seed": 13}}})
        except BaseException as exc:
            c.errors.append(exc)
        finally:
            if authority is not None:
                authority.close()
            c.done.set()

    c.thread = threading.Thread(target=controller, daemon=True)
    c.thread.start()
    channel, _ = listener.accept()
    channel.settimeout(5)
    c.channels.append(channel)
    c.channel = channel
    peer_pid, peer_uid, _ = struct.unpack("3i", channel.getsockopt(
        socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
    assert peer_uid == os.geteuid()
    peer_fd = os.pidfd_open(peer_pid)
    c.pidfds.append((peer_pid, peer_fd))
    raw = bytearray()
    while not raw.endswith(b"\n"):
        part = channel.recv(1024)
        assert part, "owned CPU worker ended private handshake"
        raw.extend(part)
        assert len(raw) < 4096
    detail = json.loads(raw)
    assert detail["pid"] == peer_pid
    assert Path(detail["path"]) == c.output
    assert detail["inode"] == c.output.stat().st_ino
    if mode != "normal":
        c.daemon_pid, c.daemon_pidfd = peer_pid, peer_fd
        assert detail["pgid"] != c.worker_pid
        assert detail["inherited_orze_keys"] == []
        assert _alive(peer_fd)
    else:
        assert peer_pid == c.worker_pid
        assert set(detail["gpu_env"].values()) == {""}
    claim = json.loads((c.folder / "claim.json").read_bytes())
    assert claim["lifecycle_db"] == str(Path(c.lake.db_path).absolute())
    assert c.lake.get_fsm_state(c.idea) == "CLAIMED"
    c.claim = claim
    channel.sendall(b"L")
    assert channel.recv(1) == b"R"
    if mode != "timeout":
        _wait_dead(c.worker_pidfd)
    return c


def _finish(c):
    if c.daemon_pidfd is not None and _alive(c.daemon_pidfd):
        c.channel.sendall(b"Q")
        _wait_dead(c.daemon_pidfd)
    assert c.done.wait(5), "admission controller did not finish after owned tree closure"
    c.thread.join(timeout=5)


def test_dispatch_waits_for_pre_script_writer_before_training(cpu_pre_script, tmp_path):
    c = _start(cpu_pre_script, tmp_path, mode="escaped")
    c.done.wait(2)
    c.channel.sendall(b"W")
    assert c.channel.recv(1) == b"W"
    assert c.output.read_bytes() == b"late posthoc writer"
    assert _alive(c.daemon_pidfd)
    reached_while_live = any(name == "training_launch" and live for name, live in c.events)
    _finish(c)
    assert not reached_while_live, "dispatch reached training while the pre-script writer was live"
    assert c.errors == []
    assert c.events == [("training_launch", False)]
    assert c.runner.failure_counts == {}
    assert c.lake.get_fsm_state(c.idea) == "CLAIMED"


def test_timed_out_setup_cannot_repair_or_fail_admission_with_live_writer(cpu_pre_script, tmp_path):
    c = _start(cpu_pre_script, tmp_path, mode="timeout")
    assert c.done.wait(5), "bounded CPU setup timeout did not return or hold"
    illegal = [(name, live) for name, live in c.events if live]
    _finish(c)
    assert illegal == [], ("timeout path treated a surviving setup writer as closed", illegal, c.stops)
    assert all(isinstance(exc, TerminationUnconfirmed) for exc in c.errors)
    assert not any(name == "training_launch" for name, _ in c.events)
    if c.errors:
        assert c.runner.failure_counts == {}
        assert not (c.folder / "_compute_receipts" / c.claim["attempt_id"] / "terminal.json").exists()


@pytest.mark.parametrize("code", [0, 7])
def test_normal_cpu_setup_controls_keep_admission_semantics(cpu_pre_script, tmp_path, code):
    c = _start(cpu_pre_script, tmp_path, mode="normal", code=code)
    _finish(c)
    assert c.errors == []
    assert all(not live for _, live in c.events)
    assert not list((c.folder / "_compute_receipts").glob("*/start.json"))
    if code == 0:
        assert c.events == [("training_launch", False)]
        assert c.runner.failure_counts == {}
        assert not list((c.folder / "_compute_receipts").glob("*/terminal.json"))
    else:
        assert not any(name == "training_launch" for name, _ in c.events)
        assert c.runner.failure_counts == {c.idea: 1}
        receipt = json.loads((c.folder / "_compute_receipts" / c.claim["attempt_id"] / "terminal.json").read_bytes())
        assert receipt["phase"] == "admission"
        assert receipt["allocated_gpu_seconds"] == 0.0
        assert receipt["reason_code"] == "pre_script_failed"
