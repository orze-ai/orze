"""Actual registered SQLite + real CPU probe, not a CLI stop ACK.

The only launch seam observes real READY, captures owned pidfds, and optionally
requests real context quiescence. It neither substitutes STOP nor fabricates
closure. Each controller is isolated in a fresh exec process.
"""
import os
from pathlib import Path
import select
import signal
import subprocess
import sys

import pytest


_PROGRAM = '''
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
from orze.idea_lake import IdeaLake
from orze.engine.controller_control import register_controller
from orze.engine import controller_probe as probe
from orze.engine.supervisor_worker import canonical

root = Path(sys.argv[1])
mode = sys.argv[2]
scope = root / "results"
scope.mkdir()
lake = IdeaLake(str(root / "authority.db"))
ctx = register_controller(lake, scope)
handles = []
prepare = probe.prepare_supervised

def observed_prepare(*args, **kwargs):
    process = prepare(*args, **kwargs)
    handles.append((process, os.pidfd_open(process.pid), os.pidfd_open(process.supervisor_pid)))
    assert process.binding["worker"]["pid"] == process.pid
    assert process._started is False
    if mode == "quiesce":
        ctx.quiesce("actual-ready-stop")
    return process

probe.prepare_supervised = observed_prepare
marker = scope / "worker-executed"
command = [sys.executable, "-I", "-c",
    "from pathlib import Path; Path(" + repr(str(marker)) + ").write_text('yes'); print('probe')"]
try:
    if mode == "quiesce":
        try:
            probe.run_probe(command, capture_output=True, timeout=5)
        except probe.ControllerProbeHOLD as error:
            assert str(error) == "controller_probe_stopped", str(error)
        else:
            raise AssertionError("quiesced worker cannot return normal result")
        assert not marker.exists()
        expected_phase, expected_outcome, expected_bytes = "QUIESCING", "interrupted", b""
        assert handles[0][0]._started is False
    else:
        result = probe.run_probe(command, capture_output=True, timeout=5)
        assert result.returncode == 0 and result.stdout == b"probe\\n" and result.stderr == b""
        assert marker.read_text() == "yes"
        expected_phase, expected_outcome, expected_bytes = "ACTIVE", "completed", b"probe\\n"
    state = lake.conn.execute("SELECT phase, hold_reason FROM controller_instances WHERE controller_id=?",
                             (ctx.controller_id,)).fetchone()
    assert tuple(state) == (expected_phase, None), tuple(state)
    rows = lake.conn.execute("SELECT controller_id,payload_json FROM controller_members").fetchall()
    assert len(rows) == 1 and rows[0][0] == ctx.controller_id
    member = json.loads(rows[0][1])
    assert member["identity"]["scope"] == str(scope)
    assert member["kind"] == "controller_probe"
    assert member["os_state"] == "CLOSED" and member["action_state"] == "SETTLED"
    assert member["outcome"] == expected_outcome
    assert member["closure"]["event"] == "TREE_CLOSED"
    assert member["closure"]["wait_proof"] == "ECHILD_WALL"
    assert member["output_bytes"] == len(expected_bytes)
    frames = {k:{"sha256":hashlib.sha256(v).hexdigest(), "bytes":len(v)}
              for k,v in (("stdout", expected_bytes), ("stderr", b""))}
    assert member["output_sha256"] == hashlib.sha256(canonical(frames)).hexdigest()
    print(json.dumps({"phase": state[0], "member": member["outcome"]}))
finally:
    for process, worker_fd, supervisor_fd in handles:
        try:
            if not select.select([worker_fd], [], [], 0)[0]:
                signal.pidfd_send_signal(worker_fd, signal.SIGKILL)
            process._supervisor.wait(timeout=3)
        except subprocess.TimeoutExpired:
            signal.pidfd_send_signal(supervisor_fd, signal.SIGKILL)
            process._supervisor.wait(timeout=3)
        finally:
            os.close(worker_fd)
            os.close(supervisor_fd)
            process._close_descriptors()
'''


@pytest.mark.parametrize("mode", ["normal", "quiesce"])
def test_registered_probe_settles_same_database_only_after_actual_tree_closure(tmp_path, mode):
    child = subprocess.Popen([sys.executable, "-c", _PROGRAM, str(tmp_path), mode],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""})
    descriptor = os.pidfd_open(child.pid)
    try:
        stdout, stderr = child.communicate(timeout=20)
        assert child.returncode == 0, stdout + stderr
    finally:
        if not select.select([descriptor], [], [], 0)[0]:
            signal.pidfd_send_signal(descriptor, signal.SIGKILL)
        child.wait(timeout=5)
        os.close(descriptor)
