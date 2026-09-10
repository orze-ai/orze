"""Public C2a CPU regressions: an exited leader is not a closed evaluation.

The B1 source, native evaluation launch, validation, SQLite and publication
are real. Training/GPU boundaries reuse the existing B2 fixture; evaluation
Popen and worker/daemon execution are real. Transparent publication spies
record whether the exact daemon pidfd is still live, then call real writers.
No new supervision API is assumed by these tests.
"""

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

from orze.core.execution_attempts import current_attempt
from orze.core import research_artifacts, research_observations
from orze.engine import accounting, evaluator, native_evaluation
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
from orze.engine.termination_hold import TerminationUnconfirmed
from test_observation_snapshot_contract import (
    project as b2_project, artifact_project, native_case,
)


REAL_POPEN = subprocess.Popen
WORKER_SOURCE = r'''
import json, os, socket, sys

endpoint, legacy_output, exit_code, detached = sys.argv[1:]
output = os.environ.get("ORZE_EVALUATION_OUTPUT_PATH", legacy_output)
body = json.dumps({"schema": 1, "observations": [{"name": "measured",
    "values": {"quality": 0},
    "validation": {"status": "valid", "reason_code": "fixture_measured"},
    "comparison_scope": "cpu-tree-v1"}]}, separators=(",", ":"))
if "ORZE_EVALUATION_OUTPUT_PATH" not in os.environ:
    body = '{"status":"COMPLETED","quality":0}'
if detached == "no":
    with open(output, "w") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    sys.exit(int(exit_code))

# The leader cannot exit before the test captures the daemon's exact pidfd.
read_end, write_end = os.pipe()
os.set_inheritable(write_end, True)
middle = os.fork()
if middle:
    os.close(write_end)
    assert os.read(read_end, 1) == b"L"
    os.close(read_end)
    os.waitpid(middle, 0)
    os._exit(int(exit_code))
os.close(read_end)
os.setsid()
last = os.fork()
if last:
    os._exit(0)
daemon = r"""
import json, os, socket, sys
endpoint, output, body, release_fd = sys.argv[1:]
descriptor = os.open(output, os.O_CREAT | os.O_TRUNC | os.O_RDWR, 0o600)
os.write(descriptor, body.encode())
os.fsync(descriptor)
channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
channel.settimeout(20)
channel.connect(endpoint)
channel.sendall((json.dumps({"pid": os.getpid(), "pgid": os.getpgrp(),
    "inode": os.fstat(descriptor).st_ino,
    "inherited_orze_keys": [key for key in os.environ if key.startswith("ORZE_")]}) + "\n").encode())
assert channel.recv(1) == b"L"
channel.sendall(b"R")
os.write(int(release_fd), b"L")
os.close(int(release_fd))
try:
    while True:
        command = channel.recv(1)
        if not command or command == b"Q":
            break
        if command == b"W":
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, b"late daemon output")
            os.ftruncate(descriptor, len(b"late daemon output"))
            os.fsync(descriptor)
            channel.sendall(b"W")
finally:
    os.close(descriptor)
    channel.close()
"""
# A new session, a double fork, and a genuinely empty exec environment defeat
# retrospective group/nonce discovery. The private output FD stays open.
os.execve(sys.executable, [sys.executable, "-c", daemon, endpoint,
                          output, body, str(write_end)], {})
'''


def _alive(pidfd):
    poller = select.poll()
    poller.register(pidfd, select.POLLIN | select.POLLHUP)
    return not poller.poll(0)


def _wait_dead(pidfd, seconds=5):
    poller = select.poll()
    poller.register(pidfd, select.POLLIN | select.POLLHUP)
    assert poller.poll(int(seconds * 1000)), "owned CPU fixture process did not exit"


@pytest.fixture
def cpu_project(b2_project, monkeypatch):
    if sys.platform != "linux" or not hasattr(os, "pidfd_open"):
        pytest.skip("requires Linux pidfd support for exact fixture cleanup")
    c = b2_project
    c.cfg["python"] = sys.executable
    c.cfg["train_extra_env"] = {"ORZE_CPU_TREE_TEST": "must-not-survive-exec"}
    Path(c.cfg["eval_script"]).write_text(WORKER_SOURCE, encoding="utf-8")
    c.cpu_roots, c.cpu_pidfds, c.channels = [], [], []
    c.daemon_pidfd = c.daemon_pid = None
    c.publications = []

    # Own otherwise orphaned fixture daemons for exact waitpid cleanup only.
    # This does not make them descendants of the exited evaluator leader.
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0

    def popen(*args, **kwargs):
        process = REAL_POPEN(*args, **kwargs)
        c.cpu_roots.append(process)
        c.cpu_pidfds.append((process.pid, os.pidfd_open(process.pid)))
        return process

    monkeypatch.setattr(evaluator.subprocess, "Popen", popen)
    for module, name in ((accounting, "record_compute_terminal"),
                         (research_artifacts, "register_artifacts"),
                         (research_observations, "register_observations")):
        real = getattr(module, name)
        def observe(*args, _real=real, _name=name, **kwargs):
            c.publications.append((_name, c.daemon_pidfd is not None and _alive(c.daemon_pidfd)))
            return _real(*args, **kwargs)
        monkeypatch.setattr(module, name, observe)
    try:
        yield c
    finally:
        for channel in c.channels:
            channel.close()
        # Only pidfds opened for actual Popen children or authenticated socket
        # peers are signalled. Never killpg, search usernames, or signal bare PIDs.
        for _, descriptor in c.cpu_pidfds:
            if _alive(descriptor):
                signal.pidfd_send_signal(descriptor, signal.SIGKILL)
            _wait_dead(descriptor)
        for process in c.cpu_roots:
            process.wait(timeout=5)
        if c.daemon_pid is not None:
            try:
                os.waitpid(c.daemon_pid, 0)
            except ChildProcessError:
                pass  # A product supervisor may already have reaped its child.
        for _, descriptor in c.cpu_pidfds:
            os.close(descriptor)
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


def _launch(c, tmp_path, *, detached, code=0, adapter=True):
    if not adapter:
        c.cfg.pop("observation_contract")
    endpoint = tmp_path / "cpu-tree.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(endpoint))
    listener.listen(1)
    listener.settimeout(5)
    c.channels.append(listener)
    c.cfg["eval_args"] = [str(endpoint), str(c.folder / "assessment.json"),
                          str(code), "yes" if detached else "no"]
    ep = evaluator.launch_eval(c.idea, 0, c.results, c.cfg, lake=c.lake,
                               source_event=c.source_event)
    assert ep is not None
    output = (c.folder / "_evaluation_attempts" / ep.attempt_id / "work" / "measurement.json"
              if adapter else c.folder / "assessment.json")
    if detached:
        channel, _ = listener.accept()
        channel.settimeout(5)
        c.channels.append(channel)
        peer_pid, peer_uid, _ = struct.unpack("3i", channel.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
        assert peer_uid == os.geteuid()
        c.daemon_pid, c.daemon_pidfd = peer_pid, os.pidfd_open(peer_pid)
        c.cpu_pidfds.append((peer_pid, c.daemon_pidfd))
        raw = bytearray()
        while not raw.endswith(b"\n"):
            raw.extend(channel.recv(1024))
            assert len(raw) < 4096
        detail = json.loads(raw)
        assert detail["pid"] == peer_pid and detail["pgid"] != ep.process.pid
        assert detail["inherited_orze_keys"] == []
        assert detail["inode"] == output.stat().st_ino
        assert _alive(c.daemon_pidfd)
        # Capture the leader while it is still waiting for this exact signal.
        leader_fd = os.pidfd_open(ep.process.pid)
        c.cpu_pidfds.append((ep.process.pid, leader_fd))
        channel.sendall(b"L")
        assert channel.recv(1) == b"R"
        _wait_dead(leader_fd)
        c.daemon_channel = channel
    else:
        # No production polling until the short-lived real worker has exited.
        deadline = time.monotonic() + 5
        while ep.process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ep.process.poll() == code
    return ep, output


@pytest.mark.parametrize("entry,adapter", [("poll", False), ("poll", True), ("direct", True)],
                         ids=["legacy-native-poll", "b2-native-poll", "b2-direct-callback"])
def test_exited_leader_cannot_publish_while_escaped_output_writer_is_live(cpu_project, tmp_path, entry, adapter):
    c = cpu_project
    ep, output = _launch(c, tmp_path, detached=True, adapter=adapter)
    active = {0: ep}
    result = None
    try:
        if entry == "direct":
            result = native_evaluation.finish(c.lake, ep, c.folder, c.cfg, 0)
        else:
            result = evaluator.check_active_evals(active, c.results, c.cfg, lake=c.lake)
    except (AttemptEffectBusy, AttemptEffectInDoubt, TerminationUnconfirmed):
        pass

    # Make survival concrete: the daemon can still mutate the very same open
    # output inode after the callback. A new implementation may already stop it.
    live = _alive(c.daemon_pidfd)
    if live and any(was_live for _, was_live in c.publications):
        c.daemon_channel.sendall(b"W")
        assert c.daemon_channel.recv(1) == b"W"
        assert output.read_bytes() == b"late daemon output"
    row = current_attempt(c.lake.conn, c.idea, "evaluation")
    artifacts = research_artifacts.artifacts_for_attempt(c.lake.conn, ep.attempt_ref)
    observations = research_observations.observations_for_attempt(c.lake.conn, ep.attempt_ref)
    assert not any(was_live for _, was_live in c.publications), (
        "native publication ran while an exact owned daemon was still live", c.publications)
    if live:
        assert row["state"] == "RUNNING" and not artifacts and not observations
        assert not (c.folder / "_compute_receipts" / ep.attempt_id / "terminal.json").exists()
        assert not result
        if entry == "poll":
            assert active.get(0) is ep
    elif row["state"] == "TERMINAL":
        # This daemon only stops on our test-channel command or a signal;
        # product closure therefore needed forced cleanup, not natural success.
        assert row["terminal"]["outcome"] == "failed"
        assert row["terminal"]["return_code"] == 0
        assert observations == []
    assert current_attempt(c.lake.conn, c.idea, "training") == c.training_before
    assert (c.folder / "metrics.json").read_bytes() == c.metrics_before


@pytest.mark.parametrize("code", [0, 7], ids=["exit-zero", "exit-nonzero"])
def test_real_worker_without_descendants_retains_native_completion_semantics(cpu_project, tmp_path, code):
    c = cpu_project
    ep, _ = _launch(c, tmp_path, detached=False, code=code)
    active = {0: ep}
    assert evaluator.check_active_evals(active, c.results, c.cfg, lake=c.lake) == [(c.idea, 0)]
    assert active == {}
    row = current_attempt(c.lake.conn, c.idea, "evaluation")
    assert row["state"] == "TERMINAL"
    assert row["terminal"]["outcome"] == ("completed" if code == 0 else "failed")
    assert row["terminal"]["return_code"] == code
    expected = 1 if code == 0 else 0
    assert len(research_artifacts.artifacts_for_attempt(c.lake.conn, ep.attempt_ref)) == expected
    assert len(research_observations.observations_for_attempt(c.lake.conn, ep.attempt_ref)) == expected
    start = json.loads((c.folder / "_compute_receipts" / ep.attempt_id / "start.json").read_bytes())
    terminal = json.loads((c.folder / "_compute_receipts" / ep.attempt_id / "terminal.json").read_bytes())
    assert start["process_pid"] == terminal["process_pid"] == ep.process.pid
    assert terminal["return_code"] == code
