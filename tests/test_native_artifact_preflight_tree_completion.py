"""CPU-only public resolver admission boundaries, with real claims and SQLite.

Reuse only the existing tiny-worker/own-pidfd cleanup fixture. No dataset/model
resolver runs: its configured script writes one temporary file. The subsequent
pre-script, training and fixer are explicit spies, never real workloads.
These tests assert existing public behavior, not a future attempt/proof API.
"""
import json
import os
from pathlib import Path
import socket
import struct
import threading

import pytest

from orze.core.evaluation_retry_state import open_existing_lake
from orze.engine import phases, process
from orze.engine.termination_hold import TerminationUnconfirmed
from test_native_pre_script_tree_completion import cpu_pre_script, _finish
from test_native_training_tree_completion import _alive, _wait_dead


@pytest.fixture
def cpu_preflight(cpu_pre_script, monkeypatch):
    c = cpu_pre_script
    c.cfg["artifact_preflight"] = {
        "enabled": True, "script": c.cfg["pre_script"],
        "network": "inherit", "timeout": 10, "retry_interval": 60,
    }
    monkeypatch.setattr(phases, "run_artifact_preflight", process.run_artifact_preflight)

    def pre_boundary(*args, **kwargs):
        live = c.daemon_pidfd is not None and _alive(c.daemon_pidfd)
        c.events.append(("pre_script", live))
        return True

    monkeypatch.setattr(phases, "run_pre_script", pre_boundary)
    return c


def _start_preflight(c, tmp_path, *, mode, code=0):
    endpoint = tmp_path / "preflight.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(endpoint))
    listener.listen(1)
    listener.settimeout(5)
    c.channels.append(listener)
    c.cfg["artifact_preflight"]["args"] = [str(endpoint), str(c.output), mode, str(code)]
    if mode == "timeout":
        c.cfg["artifact_preflight"]["timeout"] = 1

    def controller():
        authority = None
        try:
            authority = open_existing_lake(c.lake.db_path)
            c.runner.lake = authority
            phases.OrzePhaseMixin._launch_training(c.runner, [c.idea], True,
                {c.idea: {"title": "Synthetic CPU resolver", "priority": "high",
                          "config": {"seed": 13}}})
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
        assert part, "owned CPU resolver ended private handshake"
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
        assert detail["gpu_env"] == {"CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "none", "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": ""}
    claim = json.loads((c.folder / "claim.json").read_bytes())
    assert claim["lifecycle_db"] == str(Path(c.lake.db_path).absolute())
    assert c.lake.get_fsm_state(c.idea) == "CLAIMED"
    c.claim = claim
    channel.sendall(b"L")
    assert channel.recv(1) == b"R"
    if mode != "timeout":
        _wait_dead(c.worker_pidfd)
    return c


def _receipt(c):
    path = c.folder / "artifact_preflight.json"
    return json.loads(path.read_bytes()) if path.exists() else None


def test_resolver_does_not_publish_pass_or_dispatch_while_writer_live(cpu_preflight, tmp_path):
    c = _start_preflight(cpu_preflight, tmp_path, mode="escaped")
    c.done.wait(2)
    c.channel.sendall(b"W")
    assert c.channel.recv(1) == b"W"
    assert c.output.read_bytes() == b"late posthoc writer"
    assert _alive(c.daemon_pidfd)
    receipt_while_live = _receipt(c)
    downstream_while_live = [(name, live) for name, live in c.events if live]
    _finish(c)
    assert not receipt_while_live or receipt_while_live["status"] != "passed", (
        "resolver published passed before its writable output tree closed", receipt_while_live)
    assert downstream_while_live == []
    assert c.errors == []
    assert c.events == [("pre_script", False), ("training_launch", False)]
    assert _receipt(c)["status"] == "passed"
    assert c.runner.failure_counts == {}


def test_resolver_timeout_cannot_fail_admission_with_live_writer(cpu_preflight, tmp_path):
    c = _start_preflight(cpu_preflight, tmp_path, mode="timeout")
    assert c.done.wait(5), "bounded resolver timeout did not return or hold"
    receipt_while_live = _receipt(c) if _alive(c.daemon_pidfd) else None
    illegal = [(name, live) for name, live in c.events if live]
    _finish(c)
    assert not receipt_while_live or receipt_while_live["status"] not in {
        "timed_out", "failed", "execution_error"}, (
        "timeout classified a surviving resolver writer as closed", receipt_while_live, c.stops)
    assert illegal == [], ("resolver timeout reached business failure before closure", illegal)
    assert all(isinstance(exc, TerminationUnconfirmed) for exc in c.errors)
    assert not any(name in {"pre_script", "training_launch", "fixer"} for name, _ in c.events)
    if c.errors:
        assert c.runner.failure_counts == {}
        assert not (c.folder / "_compute_receipts" / c.claim["attempt_id"] / "terminal.json").exists()


@pytest.mark.parametrize("code", [0, 7])
def test_normal_resolver_exit_controls(cpu_preflight, tmp_path, code):
    c = _start_preflight(cpu_preflight, tmp_path, mode="normal", code=code)
    _finish(c)
    assert c.errors == []
    assert all(not live for _, live in c.events)
    assert not list((c.folder / "_compute_receipts").glob("*/start.json"))
    receipt = _receipt(c)
    assert receipt["status"] == ("passed" if code == 0 else "failed")
    assert receipt["exit_code"] == code
    assert receipt["gpu_visibility"] == "hidden"
    if code == 0:
        assert c.events == [("pre_script", False), ("training_launch", False)]
        assert c.runner.failure_counts == {}
        assert not list((c.folder / "_compute_receipts").glob("*/terminal.json"))
    else:
        assert not any(name in {"pre_script", "training_launch", "fixer"} for name, _ in c.events)
        assert c.runner.failure_counts == {c.idea: 1}
        accounting = json.loads((c.folder / "_compute_receipts" / c.claim["attempt_id"] / "terminal.json").read_bytes())
        assert accounting["phase"] == "admission"
        assert accounting["allocated_gpu_seconds"] == 0.0
        assert accounting["reason_code"] == "artifact_preflight_failed"
