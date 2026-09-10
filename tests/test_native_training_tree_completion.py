"""C2b public CPU regressions: native training must close its owned tree.

The tiny worker writes synthetic bytes, not a model or a real training job.
Lake/claim/config, launch, accounting, B1 and terminal transactions are real.
Only accelerator allocation/telemetry is replaced. Private socket credentials
and captured pidfds own every fixture cleanup target; there is no host scan.
"""
import contextlib
import ctypes
import json
import os
from pathlib import Path
import select
import signal
import socket
import struct
import subprocess
import sys
import time

import pytest

from orze.core import research_artifacts
from orze.core.execution_attempts import current_attempt
from orze.engine import accounting, launcher, process, training_completion
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
from orze.engine.termination_hold import TerminationUnconfirmed
from test_native_training_caller_boundaries import case as native_case


REAL_POPEN = subprocess.Popen
WORKER_SOURCE = r'''
import argparse, json, os, socket, sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--idea-id")
parser.add_argument("--results-dir")
parser.add_argument("--fixture-socket")
parser.add_argument("--fixture-code", type=int)
parser.add_argument("--fixture-detached")
args, unused = parser.parse_known_args()
folder = Path(args.results_dir) / args.idea_id
with (folder / "metrics.json").open("w") as metrics:
    json.dump({"status": "COMPLETED", "score": 0}, metrics)
    metrics.flush()
    os.fsync(metrics.fileno())
descriptor = os.open(folder / "model.bin", os.O_CREAT | os.O_TRUNC | os.O_RDWR, 0o600)
os.write(descriptor, b"synthetic-native-artifact")
os.fsync(descriptor)
if args.fixture_detached == "no":
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    channel.settimeout(20)
    channel.connect(args.fixture_socket)
    channel.sendall((json.dumps({"pid": os.getpid(), "pgid": os.getpgrp(),
        "inode": os.fstat(descriptor).st_ino}) + "\n").encode())
    assert channel.recv(1) == b"L"
    channel.sendall(b"R")
    channel.close()
    os.close(descriptor)
    sys.exit(args.fixture_code)

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
    os._exit(args.fixture_code)
os.close(read_end)
os.setsid()
last = os.fork()
if last:
    os._exit(0)
daemon = r"""
import json, os, socket, sys
endpoint, descriptor, release_fd = sys.argv[1:]
descriptor, release_fd = int(descriptor), int(release_fd)
channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
channel.settimeout(20)
channel.connect(endpoint)
channel.sendall((json.dumps({"pid": os.getpid(), "pgid": os.getpgrp(),
    "inode": os.fstat(descriptor).st_ino,
    "inherited_orze_keys": [key for key in os.environ if key.startswith("ORZE_")]}) + "\n").encode())
assert channel.recv(1) == b"L"
channel.sendall(b"R")
os.write(release_fd, b"L")
os.close(release_fd)
try:
    while True:
        command = channel.recv(1)
        if not command or command == b"Q":
            break
        if command == b"W":
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, b"late owned writer")
            os.ftruncate(descriptor, len(b"late owned writer"))
            os.fsync(descriptor)
            channel.sendall(b"W")
finally:
    os.close(descriptor)
    channel.close()
"""
# Keep the original artifact inode open across setsid/double-fork/empty-env
# exec; no retrospectively inherited ORZE nonce or process group remains.
os.execve(sys.executable, [sys.executable, "-c", daemon, args.fixture_socket,
                          str(descriptor), str(write_end)], {})
'''


def _alive(pidfd):
    poller = select.poll()
    poller.register(pidfd, select.POLLIN | select.POLLHUP)
    return not poller.poll(0)


def _wait_dead(pidfd):
    poller = select.poll()
    poller.register(pidfd, select.POLLIN | select.POLLHUP)
    assert poller.poll(5000), "owned tiny CPU worker did not exit"


@pytest.fixture
def cpu_training(native_case, monkeypatch):
    if sys.platform != "linux" or not hasattr(os, "pidfd_open"):
        pytest.skip("requires Linux pidfds for exact fixture cleanup")
    from orze.engine.supervised_process import prepare_supervised
    c = native_case
    c.cfg.update({"python": sys.executable, "_project_root": str(c.results.parent),
                  "_orze_dir": str(c.results.parent / "control"),
                  "train_extra_env": {"ORZE_CPU_TREE_TEST": "must-not-survive-exec"},
                  "artifact_contract": {"version": 1, "outputs": {
                      "weights": {"path": "model.bin", "max_bytes": 128}}}})
    Path(c.cfg["train_script"]).write_text(WORKER_SOURCE, encoding="utf-8")
    c.roots, c.pidfds, c.channels, c.handles = [], [], [], []
    c.daemon_pid = c.daemon_pidfd = None
    c.publications, c.observing = [], False
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0

    def popen(*args, **kwargs):
        child = REAL_POPEN(*args, **kwargs)
        c.roots.append(child)
        c.pidfds.append((child.pid, os.pidfd_open(child.pid)))
        return child

    # The C2a primitive already exists on the baseline; old training ignores
    # this slot. Explicit restoration prevents a future inherited OS stub from
    # accidentally turning this real-CPU fixture into simulated closure.
    monkeypatch.setattr(launcher, "prepare_supervised", prepare_supervised, raising=False)
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher, "capture_process_identity", process.capture_process_identity)
    monkeypatch.setattr(launcher, "_terminate_and_reap", process._terminate_and_reap)
    monkeypatch.setattr(launcher, "gpu_execution_lease", lambda *a, **k: contextlib.nullcontext(()))
    monkeypatch.setattr(launcher, "_detect_zombie", lambda *a, **k: False)
    monkeypatch.setattr(launcher, "_watchdog_check", lambda *a, **k: False)
    for target, name in ((accounting, "record_compute_terminal"),
                         (research_artifacts, "register_artifacts"),
                         (c.lake, "_record_state_transition_in_tx"),
                         (c.lake, "_record_stage_transition_in_tx")):
        real = getattr(target, name)
        def observe(*args, _real=real, _name=name, **kwargs):
            if c.observing:
                c.publications.append((_name, c.daemon_pidfd is not None and _alive(c.daemon_pidfd)))
            return _real(*args, **kwargs)
        monkeypatch.setattr(target, name, observe)
    try:
        yield c
    finally:
        for channel in c.channels:
            channel.close()
        for _, descriptor in c.pidfds:
            if _alive(descriptor):
                signal.pidfd_send_signal(descriptor, signal.SIGKILL)
            _wait_dead(descriptor)
        for child in c.roots:
            child.wait(timeout=5)
        if c.daemon_pid is not None:
            try:
                os.waitpid(c.daemon_pid, 0)
            except ChildProcessError:
                pass  # A real product supervisor may already own/reap it.
        for _, descriptor in c.pidfds:
            os.close(descriptor)
        for tp in c.handles:
            tp.close_log()
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


def _launch(c, tmp_path, *, detached, code=0):
    endpoint = tmp_path / "cpu-train.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(endpoint))
    listener.listen(1)
    listener.settimeout(5)
    c.channels.append(listener)
    c.cfg["train_extra_args"] = ["--fixture-socket", str(endpoint),
                                  "--fixture-code", str(code),
                                  "--fixture-detached", "yes" if detached else "no"]
    tp = launcher.launch(c.idea, 0, c.results, c.cfg, lake=c.lake)
    c.handles.append(tp)
    leader_fd = os.pidfd_open(tp.process.pid)
    c.pidfds.append((tp.process.pid, leader_fd))
    channel, _ = listener.accept()
    channel.settimeout(5)
    c.channels.append(channel)
    peer_pid, peer_uid, _ = struct.unpack("3i", channel.getsockopt(
        socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
    assert peer_uid == os.geteuid()
    peer_fd = os.pidfd_open(peer_pid)
    c.pidfds.append((peer_pid, peer_fd))
    if detached:
        c.daemon_pid, c.daemon_pidfd = peer_pid, peer_fd
    raw = bytearray()
    while not raw.endswith(b"\n"):
        chunk = channel.recv(1024)
        assert chunk, "worker ended the private handshake"
        raw.extend(chunk)
        assert len(raw) < 4096
    detail = json.loads(raw)
    assert detail["pid"] == peer_pid
    assert detail["inode"] == (c.folder / "model.bin").stat().st_ino
    if detached:
        assert detail["pgid"] != tp.process.pid
        assert detail["inherited_orze_keys"] == []
        assert _alive(c.daemon_pidfd)
    else:
        assert peer_pid == tp.process.pid
    channel.sendall(b"L")
    assert channel.recv(1) == b"R"
    _wait_dead(leader_fd)
    if not detached:
        deadline = time.monotonic() + 5
        while tp.process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert tp.process.poll() == code
    c.daemon_channel = channel
    c.observing = True
    return tp


@pytest.mark.parametrize("entry", ["poll", "direct"], ids=["native-poll", "native-direct-callback"])
def test_native_training_cannot_publish_before_escaped_artifact_writer_closes(cpu_training, tmp_path, entry):
    c = cpu_training
    tp = _launch(c, tmp_path, detached=True)
    active, failures, finished = {0: tp}, {}, None
    try:
        if entry == "poll":
            finished = launcher.check_active(active, c.results, c.cfg, failures, lake=c.lake)
        else:
            finished = training_completion.finish(c.lake, tp, 0, c.folder, c.cfg, 0, failures)
    except (AttemptEffectBusy, AttemptEffectInDoubt, TerminationUnconfirmed):
        pass
    live = _alive(c.daemon_pidfd)
    if live and any(was_live for _, was_live in c.publications):
        c.daemon_channel.sendall(b"W")
        assert c.daemon_channel.recv(1) == b"W"
        assert (c.folder / "model.bin").read_bytes() == b"late owned writer"
    row = current_attempt(c.lake.conn, c.idea, "training")
    artifacts = research_artifacts.artifacts_for_attempt(c.lake.conn, tp.attempt_ref)
    assert not any(was_live for _, was_live in c.publications), (
        "native training published with a live exact owned artifact writer", c.publications)
    if live:
        assert row["state"] == "RUNNING"
        assert c.lake.get_fsm_state(c.idea) == "IN_PROGRESS"
        assert c.lake.get_stage_state(c.idea, "training") == "IN_PROGRESS"
        assert artifacts == []
        assert not (c.folder / "_compute_receipts" / tp.attempt_id / "terminal.json").exists()
        assert not finished
        assert failures == {}
        if entry == "poll":
            assert active.get(0) is tp
    elif row["state"] == "TERMINAL":
        # The daemon has no natural-exit timer inside this bounded callback;
        # product STOP must not turn a zero leader exit into successful output.
        assert row["terminal"]["outcome"] != "completed"
        assert row["terminal"]["return_code"] == 0
        assert artifacts == []


@pytest.mark.parametrize("code", [0, 7], ids=["exit-zero", "exit-nonzero"])
def test_real_native_training_without_descendants_keeps_actual_exit_and_artifacts(cpu_training, tmp_path, code):
    c = cpu_training
    tp = _launch(c, tmp_path, detached=False, code=code)
    active, failures = {0: tp}, {}
    assert launcher.check_active(active, c.results, c.cfg, failures, lake=c.lake) == [(c.idea, 0)]
    assert active == {}
    row = current_attempt(c.lake.conn, c.idea, "training")
    assert row["state"] == "TERMINAL"
    assert row["terminal"]["outcome"] == ("completed" if code == 0 else "failed")
    assert row["terminal"]["return_code"] == code
    assert c.lake.get_fsm_state(c.idea) == ("COMPLETE" if code == 0 else "FAILED")
    records = research_artifacts.artifacts_for_attempt(c.lake.conn, tp.attempt_ref)
    assert len(records) == (1 if code == 0 else 0)
    if records:
        content = Path(records[0]["path"])
        assert content.read_bytes() == b"synthetic-native-artifact"
        assert content.stat().st_ino != (c.folder / "model.bin").stat().st_ino
    start = json.loads((c.folder / "_compute_receipts" / tp.attempt_id / "start.json").read_bytes())
    terminal = json.loads((c.folder / "_compute_receipts" / tp.attempt_id / "terminal.json").read_bytes())
    assert start["process_pid"] == terminal["process_pid"] == tp.process.pid
    assert terminal["return_code"] == code
