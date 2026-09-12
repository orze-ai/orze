"""The real prepare seam must not renew an already-persisted CPU descriptor.

The wrapper changes only its received mutable argument, then delegates to the
actual supervisor. SQL, closure, clock and permission readers remain real.
This is a new C3 candidate boundary, not a missing-API historical defect.
"""
from copy import deepcopy
import os
import select
import signal

from test_native_cpu_action import context
from orze.core import cpu_action_budget as budget
from orze.core.execution_attempts import current_attempt
from orze.engine import native_cpu_action as native
from orze.engine.supervisor_worker import process_identity
from orze.engine.termination_hold import require_no_unconfirmed_stop


def test_prepare_cannot_renew_the_persisted_descriptor_in_place(context, monkeypatch):
    lake, results, scope, cfg, create, handles = context
    cfg["cpu_runtime_lease"] = {"version": 1, "ttl_seconds": 2}
    action, permit = create(
        "from pathlib import Path; import time; Path('executed').write_text('GO'); time.sleep(15)",
        timeout=4)
    actual_prepare = native.prepare_supervised
    captured = []
    original = []

    def prepare(*args, **kwargs):
        row = current_attempt(lake.conn, "idea-cpu", "action")
        original.append(deepcopy(row["binding"]["runtime_lease"]))
        assert row["state"] == "LAUNCHING"
        assert kwargs["runtime_lease"] == original[0]
        # Still inside the four-second budget envelope: rejecting only an
        # oversized permit is not sufficient to protect descriptor identity.
        kwargs["runtime_lease"]["deadline_ns"] += 1_000_000_000
        process = actual_prepare(*args, **kwargs)
        item = {"process": process, "fds": [os.dup(process.supervisor_pidfd)]}
        captured.append(item)
        fd = os.pidfd_open(process.pid, 0)
        try:
            identity, parent = process_identity(process.pid)
            assert identity == process.binding["worker"] and parent == process.supervisor_pid
        except BaseException:
            os.close(fd)
            raise
        item["fds"].append(fd)
        return process

    def admission():
        budget.require_permit(lake, permit)
        require_no_unconfirmed_stop(results / "idea-cpu")

    monkeypatch.setattr(native, "prepare_supervised", prepare)
    try:
        try:
            handle = native.launch("idea-cpu", results, cfg, lake=lake, action=action,
                                   permit=permit, admission=admission)
        except native.CPUActionHOLD as exc:
            handle = exc.cpu_action_handle
        assert len(original) == len(captured) == 1
        row = current_attempt(lake.conn, "idea-cpu", "action")
        assert row["binding"]["runtime_lease"] == original[0], "persisted runtime descriptor was renewed"
        assert captured[0]["process"]._started is False, "GO was issued using a renewed descriptor"
        assert not list((results / "idea-cpu").rglob("executed"))
        assert row["terminal"] is None
        assert budget.snapshot(lake, scope)["active_reservations"] == 1
        assert handle is not None
    finally:
        for item in captured:
            process = item["process"]
            try:
                try:
                    process.stop(timeout=.3)
                except Exception:
                    pass
                for fd in reversed(item["fds"]):
                    if not select.select([fd], [], [], 0)[0]:
                        signal.pidfd_send_signal(fd, signal.SIGKILL)
                process._supervisor.wait(timeout=3)
                assert all(select.select([fd], [], [], 2)[0] for fd in item["fds"])
            finally:
                for fd in item["fds"]:
                    os.close(fd)
                if process._channel is not None:
                    process._channel.close()
                # Own teardown only; never discard the retained native owner.
                if process in handles:
                    handles.remove(process)
