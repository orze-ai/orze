"""New Linux supervisor API mechanisms, not released-behavior red tests."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from orze.engine import supervised_process as supervision


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper contract")


def _identity(tmp_path):
    return {"attempt_ref": {"task_id": "idea-test", "phase": "evaluation",
                            "attempt_id": "attempt-test", "generation": 1},
            "scope": str(tmp_path)}


def _prepare(tmp_path, script, **kwargs):
    return supervision.prepare_supervised(
        [sys.executable, "-c", script], identity=_identity(tmp_path),
        cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)


def _until(predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("bounded child handshake did not arrive")


def test_ready_keeps_real_worker_blocked_and_binding_detached(tmp_path):
    process = _prepare(tmp_path, "from pathlib import Path; Path('ran').write_text('yes')")
    try:
        binding = process.binding
        assert process.pid == binding["worker"]["pid"]
        assert process.pid != process.supervisor_pid == binding["supervisor"]["pid"]
        assert binding["identity"] == _identity(tmp_path)
        binding["identity"]["scope"] = "changed"
        assert process.binding["identity"]["scope"] == str(tmp_path)
        assert process.poll() is None
        assert not (tmp_path / "ran").exists()
        process.start()
        assert process.wait(timeout=5) == 0
        receipt = process.closure_receipt()
        assert receipt["binding"] == process.binding
        assert receipt["wait_proof"] == "ECHILD_WALL"
        assert receipt["reaped_children"] == 1
        assert not receipt["stop_requested"]
        assert (tmp_path / "ran").read_text() == "yes"
        assert process.poll() == 0  # exact repeat remains a receipt read
    finally:
        process.stop(timeout=3)


@pytest.mark.parametrize("code", [0, 7, -signal.SIGTERM])
def test_wait_retains_actual_worker_exit_status(tmp_path, code):
    script = (f"import sys; sys.exit({code})" if code >= 0 else
              "import os,signal; os.kill(os.getpid(), signal.SIGTERM)")
    process = _prepare(tmp_path, script)
    process.start()
    assert process.wait(timeout=5) == code
    assert process.closure_receipt()["worker_returncode"] == code


def test_fast_double_fork_setsid_is_not_closed_until_adopted_child_reaped(tmp_path):
    script = """
import os,time
from pathlib import Path
if os.fork(): os._exit(0)
os.setsid()
if os.fork(): os._exit(0)
Path('ready').write_text('yes')
deadline=time.monotonic()+5
while not Path('release').exists() and time.monotonic()<deadline: time.sleep(.01)
Path('late').write_text('finished')
os._exit(0)
"""
    process = _prepare(tmp_path, script)
    try:
        process.start()
        _until(lambda: (tmp_path / "ready").exists())
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=0.05)
        assert process.poll() is None
        assert process.closure_receipt() is None
        (tmp_path / "release").write_text("go")
        assert process.wait(timeout=5) == 0
        receipt = process.closure_receipt()
        assert receipt["reaped_children"] == 3
        assert receipt["stop_requested"] is False
        assert (tmp_path / "late").read_text() == "finished"
    finally:
        process.stop(timeout=3)


def test_stop_after_leader_zero_records_cleanup_without_falsifying_returncode(tmp_path):
    script = """
import os,signal,time
from pathlib import Path
if os.fork(): os._exit(0)
os.setsid()
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path('ready').write_text('yes')
while True: time.sleep(.01)
"""
    process = _prepare(tmp_path, script)
    process.start()
    _until(lambda: (tmp_path / "ready").exists())
    assert process.poll() is None
    assert process.stop(timeout=3) is True
    assert process.poll() == 0
    receipt = process.closure_receipt()
    assert receipt["worker_returncode"] == 0
    assert receipt["stop_requested"] is True
    assert receipt["forced_cleanup"] is True
    assert receipt["reaped_children"] == 2


def test_private_protocol_fds_do_not_survive_exec_but_lease_fd_does(tmp_path):
    lease = tmp_path / "lease"
    lease.write_text("lease")
    fd = os.open(lease, os.O_RDONLY)
    try:
        script = f"""
import json,os
from pathlib import Path
targets=[]
for name in os.listdir('/proc/self/fd'):
    try: targets.append(os.readlink('/proc/self/fd/'+name))
    except FileNotFoundError: pass
Path('fds.json').write_text(json.dumps({{'lease':os.readlink('/proc/self/fd/{fd}'),'all':targets}}))
"""
        process = _prepare(tmp_path, script, pass_fds=(fd,))
        process.start()
        assert process.wait(timeout=5) == 0
        actual = json.loads((tmp_path / "fds.json").read_text())
        assert actual["lease"] == str(lease)
        assert not any(value.startswith("socket:") for value in actual["all"])
    finally:
        os.close(fd)


def test_supervisor_loss_is_uncertain_not_a_failed_integer(tmp_path):
    # Worker is still blocked: killing the supervisor closes its private gate,
    # so this tiny fixture does not leave a running user-code descendant.
    process = _prepare(tmp_path, "raise AssertionError('must not execute')")
    signal.pidfd_send_signal(process.supervisor_pidfd, signal.SIGKILL)
    process._supervisor.wait(timeout=3)
    for operation in (process.poll, process.closure_receipt, process.stop):
        with pytest.raises(supervision.SupervisionUncertain):
            operation()
    assert process.returncode is None
    process._close_descriptors()


def test_unsupported_host_rejects_before_any_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(supervision.sys, "platform", "unsupported")
    monkeypatch.setattr(supervision.subprocess, "Popen",
                        lambda *a, **kw: pytest.fail("must reject before Popen"))
    with pytest.raises(supervision.SupervisionUnavailable):
        _prepare(tmp_path, "pass")


def test_pre_ready_transport_failure_carries_uncertain_provisional_handle(tmp_path, monkeypatch):
    monkeypatch.setattr(supervision, "send_frame",
                        lambda *args: (_ for _ in ()).throw(OSError("injected")))
    with pytest.raises(supervision.SupervisionUncertain) as caught:
        _prepare(tmp_path, "raise AssertionError('must not execute')")
    process = caught.value.process
    assert isinstance(process, supervision.SupervisedProcess)
    assert process.pid is None
    assert process.returncode is None
    process._supervisor.wait(timeout=3)
    process._close_descriptors()
