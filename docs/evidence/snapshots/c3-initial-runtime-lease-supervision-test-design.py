"""C3 new protocol mechanisms; no historical missing-API red claims.

Pure clock/schema cases are distinguished from real Linux child/supervisor
cases. Real children use only private tmp paths and actual owned supervision;
there is no host census, raw PID signalling, provider, or GPU operation.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path
import select
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from orze.core.execution_attempts import AttemptRef
from orze.engine import process_supervision as proof
from orze.engine import supervised_process as api
from orze.engine import supervisor_worker as worker


DESCRIPTOR = {"schema": 1, "clock": "CLOCK_BOOTTIME", "hostname": "fixture-host",
              "boot_id": "12345678-1234-1234-1234-123456789abc",
              "issued_ns": 100, "deadline_ns": 200}


@pytest.mark.parametrize("field,value", [
    ("schema", True), ("issued_ns", 0.0), ("deadline_ns", True),
    ("hostname", "bad/host"), ("boot_id", "12345678-1234-1234-1234-123456789ABC"),
    ("deadline_ns", worker.MAX_NS + 1),
])
def test_descriptor_strict_types_and_identity(field, value):
    descriptor = {**DESCRIPTOR, field: value}
    with pytest.raises(ValueError):
        worker.validate_runtime_lease(descriptor)


def test_historical_descriptor_is_detached_and_has_no_clock_or_os_read(monkeypatch):
    def forbidden():
        pytest.fail("structural historical validation must not access current OS")
    monkeypatch.setattr(worker, "_runtime_lease_identity", forbidden)
    monkeypatch.setattr(worker, "_runtime_lease_clock", forbidden)
    descriptor = deepcopy(DESCRIPTOR)
    validated = worker.validate_runtime_lease(descriptor)
    validated["deadline_ns"] += 1
    assert descriptor == DESCRIPTOR
    for changed in ({k: v for k, v in descriptor.items() if k != "clock"},
                    {**descriptor, "extra": 1}):
        with pytest.raises(ValueError):
            worker.validate_runtime_lease(changed)


@pytest.mark.parametrize("ttl", [True, 0, float("nan"), 1e-12, worker.MAX_NS])
def test_invalid_ttl_never_reads_clock(ttl, monkeypatch):
    monkeypatch.setattr(worker, "_runtime_lease_identity",
                        lambda: pytest.fail("invalid TTL must reject before OS read"))
    with pytest.raises(ValueError):
        worker.capture_runtime_lease(ttl)


def test_capture_floors_float_ns_and_expiry_does_not_disable_clock_reader(monkeypatch):
    monkeypatch.setattr(worker, "_runtime_lease_identity",
                        lambda: (DESCRIPTOR["hostname"], DESCRIPTOR["boot_id"]))
    observed = [100]
    monkeypatch.setattr(worker, "_runtime_lease_clock", lambda: observed[0])
    descriptor = worker.capture_runtime_lease(0.3)
    # IEEE float 0.3 is strictly below 3/10, not an exact 300,000,000 ns.
    assert descriptor["deadline_ns"] - descriptor["issued_ns"] == 299_999_999
    observed[0] = descriptor["deadline_ns"] - 1
    assert worker.require_runtime_lease(descriptor) == observed[0]
    observed[0] += 1
    assert worker.runtime_lease_now(descriptor) == observed[0]
    with pytest.raises(worker.RuntimeLeaseExpired) as caught:
        worker.require_runtime_lease(descriptor)
    assert caught.value.observed_ns == observed[0]


@pytest.mark.parametrize("fault", ["boot", "clock_reverse"])
def test_unknown_clock_identity_is_not_known_expiry(monkeypatch, fault):
    monkeypatch.setattr(worker, "_runtime_lease_identity", lambda: (
        DESCRIPTOR["hostname"], "00000000-0000-0000-0000-000000000000"
        if fault == "boot" else DESCRIPTOR["boot_id"]))
    monkeypatch.setattr(worker, "_runtime_lease_clock", lambda: 99)
    with pytest.raises(ValueError) as caught:
        worker.require_runtime_lease(DESCRIPTOR)
    assert not isinstance(caught.value, worker.RuntimeLeaseExpired)


def _identity(tmp_path):
    return {"attempt_ref": asdict(AttemptRef("idea-lease", "action", "attempt-lease", 1)),
            "scope": str(tmp_path)}


def _prepare(tmp_path, script, *, ttl=2):
    descriptor = worker.capture_runtime_lease(ttl)
    process = api.prepare_supervised(
        [sys.executable, "-c", script], identity=_identity(tmp_path),
        runtime_lease=descriptor, cwd=str(tmp_path),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return process, descriptor


def _marker(path):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.005)
    pytest.fail("test-owned worker did not write its readiness marker")


def _closed_proof(process, tmp_path):
    ref = AttemptRef(**_identity(tmp_path)["attempt_ref"])
    tracked = SimpleNamespace(idea_id=ref.task_id, attempt_id=ref.attempt_id,
                              attempt_ref=ref, process=process)
    ready = process.binding
    row = {"state": "RUNNING", "binding": {
        "process_supervision_protocol": worker.RUNTIME_LEASE_PROTOCOL,
        "runtime_lease": ready["runtime_lease"], "supervision": ready}}
    return proof.require_closed(tracked, row, tmp_path, process.poll(), phase="action")


def test_real_v2_normal_closure_binds_detached_descriptor(tmp_path):
    process, descriptor = _prepare(tmp_path, "pass")
    try:
        descriptor["deadline_ns"] += 1
        assert process.binding["runtime_lease"] != descriptor
        process.start()
        assert process.wait(timeout=3) == 0
        closure = _closed_proof(process, tmp_path)
        assert closure["schema"] == 2 and closure["binding"]["protocol"] == worker.RUNTIME_LEASE_PROTOCOL
        assert closure["lease_expired"] is False
        assert closure["lease_observed_ns"] < closure["binding"]["runtime_lease"]["deadline_ns"]
    finally:
        process.stop(timeout=3)


def test_real_autonomous_expiry_without_parent_poll_closes_escaped_tree(tmp_path, monkeypatch):
    script = """
import os,signal,time
from pathlib import Path
if os.fork(): os._exit(0)
os.setsid()
if os.fork(): os._exit(0)
signal.signal(signal.SIGTERM, lambda *args: os._exit(0))
Path('ready').write_text('ready')
while True: time.sleep(.01)
"""
    process, descriptor = _prepare(tmp_path, script, ttl=1.2)
    try:
        process.start()
        _marker(tmp_path / "ready")
        # Parent neither polls nor sends STOP: actual captured supervisor pidfd
        # must become readable while its control channel remains open.
        with monkeypatch.context() as m:
            m.setattr(process, "poll", lambda: pytest.fail("no parent polling before expiry closure"))
            assert select.select([process.supervisor_pidfd], [], [], 4)[0]
        assert process.wait(timeout=3) == 0
        receipt = _closed_proof(process, tmp_path)
        assert receipt["lease_expired"] is True and receipt["stop_requested"] is True
        assert receipt["lease_observed_ns"] >= descriptor["deadline_ns"]
        assert receipt["reaped_children"] == 3
        assert process._stop_sent is False
    finally:
        process.stop(timeout=3)


def test_real_expired_before_go_is_known_and_still_has_closure(tmp_path):
    process, descriptor = _prepare(tmp_path, "from pathlib import Path;Path('ran').touch()", ttl=.4)
    try:
        assert select.select([process.supervisor_pidfd], [], [], 3)[0]
        with pytest.raises(worker.RuntimeLeaseExpired) as caught:
            process.start()
        assert caught.value.observed_ns >= descriptor["deadline_ns"]
        assert process._uncertainty is None
        assert process.wait(timeout=3) == 125
        assert _closed_proof(process, tmp_path)["lease_expired"] is True
        assert not (tmp_path / "ran").exists()
    finally:
        process.stop(timeout=3)


def test_real_go_crossing_deadline_is_refused_without_protocol_uncertainty(tmp_path, monkeypatch):
    process, descriptor = _prepare(tmp_path, "from pathlib import Path;Path('ran').touch()", ttl=.8)
    original_send = api.send_frame

    def delayed_send(channel, value):
        if value.get("command") == "GO":
            # Queue the actual valid frame after the gate deadline. Blocking
            # the supervisor's owned wait seam is not needed: this frame is
            # sent while the blocked worker is still alive during STOP drain.
            remaining = (descriptor["deadline_ns"] - worker.runtime_lease_now(descriptor)) / 1e9
            if remaining > 0:
                time.sleep(remaining)
        return original_send(channel, value)

    try:
        with monkeypatch.context() as m:
            m.setattr(api, "send_frame", delayed_send)
            process.start()
        assert process.wait(timeout=3) == 125
        assert _closed_proof(process, tmp_path)["lease_expired"] is True
        assert process._uncertainty is None
        assert not (tmp_path / "ran").exists()
    finally:
        process.stop(timeout=3)


def test_real_v2_constructor_failure_preserves_fallback_descriptor(tmp_path, monkeypatch):
    descriptor = worker.capture_runtime_lease(2)
    original = api.SupervisedProcess.__init__

    def reject(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise OSError("controlled constructor failure after Popen")

    monkeypatch.setattr(api.SupervisedProcess, "__init__", reject)
    with pytest.raises(api.SupervisionUncertain) as caught:
        api.prepare_supervised([sys.executable, "-c", "pass"], identity=_identity(tmp_path),
            runtime_lease=descriptor, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    process = caught.value.process
    try:
        assert isinstance(process, api.SupervisedProcess)
        assert process._runtime_lease == descriptor and process._runtime_lease is not descriptor
        assert process.returncode is None
        process._supervisor.wait(timeout=3)
    finally:
        process._close_descriptors()


@pytest.fixture
def closed_process(tmp_path):
    process, _ = _prepare(tmp_path, "pass")
    process.start()
    process.wait(timeout=3)
    receipt = process.closure_receipt()
    yield process, receipt
    # Already observed actual supervisor exit; deliberately malformed wire
    # cases retain local uncertainty but create no live worker to clean up.
    process._close_descriptors()


@pytest.mark.parametrize("fault", ["schema_bool", "expired_int", "observed_float", "descriptor"])
def test_strict_v2_wire_rejects_mutated_real_closure(closed_process, fault):
    process, receipt = closed_process
    mutated = deepcopy(receipt)
    if fault == "schema_bool":
        mutated["schema"] = True
    elif fault == "expired_int":
        mutated["lease_expired"] = 0
    elif fault == "observed_float":
        mutated["lease_observed_ns"] = float(mutated["lease_observed_ns"])
    else:
        mutated["binding"]["runtime_lease"]["deadline_ns"] += 1
    process._closed = None
    with pytest.raises(api.SupervisionUncertain):
        process._accept(mutated)


def test_validated_closure_waits_for_exit_without_redundant_stop(closed_process, monkeypatch):
    process, receipt = closed_process
    calls = []

    def poll():
        calls.append(True)
        return None if len(calls) == 1 else receipt["worker_returncode"]

    # Only the waitpid visibility ordering is a double; receipt was actually
    # received and accepted, and the real supervisor is already reaped.
    monkeypatch.setattr(process, "poll", poll)
    monkeypatch.setattr(process, "_send", lambda command: pytest.fail("already CLOSED must not send STOP"))
    assert process.stop(timeout=1) is True
    assert len(calls) == 2
