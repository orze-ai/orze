"""C2c real CPU posthoc boundaries, not model inference or catalog proof.

The adapter writes a short output only; the real run_posthoc writes metrics.
All cleanup uses authenticated, captured own pidfds from the C2b fixture.
The old posthoc path has no AttemptRef and its optional catalog import is
unavailable. Neither is invented here to manufacture a regression.
"""
import json
import os
from pathlib import Path
import socket
import struct
import sys
import time
from types import SimpleNamespace

import pytest
import yaml

from orze.engine import launcher
from orze.engine.scheduler import claim
from test_native_training_tree_completion import (
    cpu_training, native_case, _alive, _wait_dead,
)


DAEMON_SOURCE = r'''
import json, os, socket, sys
endpoint, descriptor, release_fd, path = sys.argv[1:]
descriptor, release_fd = int(descriptor), int(release_fd)
channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
channel.settimeout(20)
channel.connect(endpoint)
channel.sendall((json.dumps({"pid": os.getpid(), "pgid": os.getpgrp(),
    "inode": os.fstat(descriptor).st_ino, "path": path,
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
            os.write(descriptor, b"late posthoc writer")
            os.ftruncate(descriptor, len(b"late posthoc writer"))
            os.fsync(descriptor)
            channel.sendall(b"W")
finally:
    os.close(descriptor)
    channel.close()
'''

SITE_SOURCE = r'''
import json, os, socket, sys
from pathlib import Path
from orze.engine import posthoc_runner

# Used only by the explicitly requested historical whole-module replay.
# Normal/current tests never set this variable.
snapshot = os.environ.get("ORZE_TEST_POSTHOC_RUNNER_SNAPSHOT")
if snapshot:
    exec(compile(Path(snapshot).read_text(), snapshot, "exec"), posthoc_runner.__dict__)

@posthoc_runner.register_adapter("owned_cpu_posthoc")
def owned_cpu_posthoc(idea_id, cfg, idea_dir):
    path = Path(idea_dir) / "posthoc.bin"
    descriptor = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_RDWR, 0o600)
    os.write(descriptor, b"synthetic-posthoc-output")
    os.fsync(descriptor)
    if not cfg["fixture_detached"]:
        channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        channel.settimeout(20)
        channel.connect(cfg["fixture_socket"])
        channel.sendall((json.dumps({"pid": os.getpid(), "pgid": os.getpgrp(),
            "inode": os.fstat(descriptor).st_ino, "path": str(path)}) + "\n").encode())
        assert channel.recv(1) == b"L"
        channel.sendall(b"R")
        channel.close()
        os.close(descriptor)
    else:
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
        else:
            os.close(read_end)
            os.setsid()
            if os.fork():
                os._exit(0)
            os.execve(sys.executable, [sys.executable, "-c", DAEMON_SOURCE,
                cfg["fixture_socket"], str(descriptor), str(write_end), str(path)], {})
    if cfg["fixture_code"]:
        os._exit(cfg["fixture_code"])
    return {"status": "COMPLETED", "score": 0, "fixture_origin": "actual_adapter_return"}
'''


@pytest.fixture
def cpu_posthoc(cpu_training, tmp_path, monkeypatch):
    c = cpu_training
    c.cfg.pop("artifact_contract")
    c.cfg["idea_lake_db"] = str(c.lake.db_path)
    c.idea = "idea-posthoc-cpu"
    c.folder = c.results / c.idea
    c.adapter_config = {"kind": "posthoc_eval", "adapter": "owned_cpu_posthoc"}
    c.lake.insert(c.idea, "Synthetic CPU posthoc", yaml.safe_dump(c.adapter_config),
                  "", status="queued", kind="posthoc_eval")
    assert claim(c.idea, c.results, 0, lake=c.lake)
    site = tmp_path / "adapter_import"
    site.mkdir()
    (site / "sitecustomize.py").write_text(
        "DAEMON_SOURCE = " + repr(DAEMON_SOURCE) + "\n" + SITE_SOURCE,
        encoding="utf-8")
    package_src = Path(launcher.__file__).resolve().parents[2]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(site), str(package_src))))
    return c


def _launch_posthoc(c, tmp_path, *, detached, code=0, standalone=False):
    endpoint = tmp_path / "cpu-posthoc.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(endpoint))
    listener.listen(1)
    listener.settimeout(5)
    c.channels.append(listener)
    config = dict(c.adapter_config, fixture_socket=str(endpoint),
                  fixture_code=code, fixture_detached=detached)
    (c.folder / "idea_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    if standalone:
        driver = ("from orze.engine.posthoc_runner import run_posthoc; "
                  f"run_posthoc({c.idea!r}, {config!r}, {str(c.folder)!r})")
        child = launcher.subprocess.Popen([sys.executable, "-c", driver], env=os.environ.copy())
        tp = SimpleNamespace(process=child)
    else:
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
        assert chunk, "posthoc worker ended the private handshake"
        raw.extend(chunk)
        assert len(raw) < 4096
    detail = json.loads(raw)
    assert detail["pid"] == peer_pid
    c.adapter_output = Path(detail["path"])
    assert c.adapter_output.is_relative_to(c.folder)
    assert detail["inode"] == c.adapter_output.stat().st_ino
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


def test_posthoc_worker_does_not_publish_public_metrics_with_live_writer(cpu_posthoc, tmp_path):
    c = cpu_posthoc
    tp = _launch_posthoc(c, tmp_path, detached=True)
    c.daemon_channel.sendall(b"W")
    assert c.daemon_channel.recv(1) == b"W"
    assert c.adapter_output.read_bytes() == b"late posthoc writer"
    assert _alive(c.daemon_pidfd)
    assert not (c.folder / "metrics.json").exists(), (
        "real run_posthoc published canonical metrics before its owned writer closed")
    assert not (c.folder / "_compute_receipts" / tp.attempt_id / "terminal.json").exists()


def test_posthoc_parent_does_not_consume_leader_exit_with_live_writer(cpu_posthoc, tmp_path):
    c = cpu_posthoc
    tp = _launch_posthoc(c, tmp_path, detached=True)
    active, failures, error, finished = {0: tp}, {}, None, None
    try:
        finished = launcher.check_active(active, c.results, c.cfg, failures, lake=c.lake)
    except RuntimeError as exc:
        # Baseline also omits started-FSM publication. Preserve that exception
        # for the final assertion; it cannot obscure an earlier real terminal.
        error = exc
    assert _alive(c.daemon_pidfd)
    assert not (c.folder / "_compute_receipts" / tp.attempt_id / "terminal.json").exists(), (
        "parent accepted live posthoc writer", str(error), c.publications)
    assert not any(was_live for _, was_live in c.publications)
    assert active.get(0) is tp
    assert not finished
    assert failures == {}
    assert error is None


@pytest.mark.parametrize("code", [0, 7], ids=["standalone-zero", "standalone-seven"])
def test_real_standalone_posthoc_retains_old_exit_and_metrics(cpu_posthoc, tmp_path, code):
    c = cpu_posthoc
    tp = _launch_posthoc(c, tmp_path, detached=False, code=code, standalone=True)
    assert tp.process.returncode == code
    metrics = c.folder / "metrics.json"
    if code == 0:
        observed = json.loads(metrics.read_text())
        assert observed["status"] == "COMPLETED"
        assert observed["score"] == 0
        assert observed["fixture_origin"] == "actual_adapter_return"
        assert observed["_source"] == "posthoc_runner:owned_cpu_posthoc"
    else:
        assert not metrics.exists()
    assert c.adapter_output.read_bytes() == b"synthetic-posthoc-output"
    assert not list(c.folder.glob("_compute_receipts/*/terminal.json"))
