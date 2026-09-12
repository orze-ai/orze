"""Isolated C3 original-behavior diagnostic, deliberately outside tests/.

Reuse the existing real native CPU fixture: real SQLite, budget, claim,
READY/GO, owned Linux supervisor, effect publication and budget settlement.
No authority/closure functions are mocked. Fixture cleanup is explicit STOP,
not evidence that an autonomous runtime lease acted. Worker barriers/timestamps
belong only to this test and are not registered scientific artifacts.
"""

import json
import os
from pathlib import Path
import select
import signal
import time

import pytest

from test_native_cpu_action import context, _finish
from orze.core import cpu_action_budget as budget
from orze.core.execution_attempts import current_attempt
from orze.engine import native_cpu_action as native
from orze.engine.supervisor_worker import process_identity
from orze.engine.termination_hold import require_no_unconfirmed_stop


WORKER = """import json,os,time
from pathlib import Path
started=time.monotonic()
Path('diagnostic-ready.json').write_text(json.dumps({'pid':os.getpid(),'started':started}))
gate=Path('diagnostic-release.json')
while not gate.exists():
 if time.monotonic()-started>8: os._exit(90)
 time.sleep(.005)
finish_at=json.loads(gate.read_text())['finish_at']
while time.monotonic()<finish_at:
 if time.monotonic()-started>8: os._exit(91)
 time.sleep(.005)
Path('diagnostic-done.json').write_text(json.dumps({'pid':os.getpid(),'started':started,'finished':time.monotonic()}))
os._exit(0)
"""


@pytest.fixture
def execution(context):
    lake, results, scope, cfg, create, handles = context
    action, permit = create(WORKER, timeout=2)

    def admission():
        # A real budget/Stop check supplied to the existing required callback;
        # no monkeypatch or fake permission is used for native launch.
        budget.require_permit(lake, permit)
        require_no_unconfirmed_stop(results / "idea-cpu")

    handle = native.launch(
        "idea-cpu", results, cfg, lake=lake, action=action,
        permit=permit, admission=admission,
    )
    process = handle.process
    owner = native._OWNERS[id(handle)]  # Read the real captured clock; do not change it.
    work = Path(owner.binding["work_dir"])
    worker_fd = os.pidfd_open(process.pid, 0)
    supervisor_fd = os.dup(process.supervisor_pidfd)
    trusted_worker = False
    try:
        observed, parent = process_identity(process.pid)
        assert observed == process.binding["worker"]
        assert parent == process.supervisor_pid
        trusted_worker = True
        until = time.monotonic() + 2
        while not (work / "diagnostic-ready.json").exists() and time.monotonic() < until:
            time.sleep(.005)
        assert (work / "diagnostic-ready.json").exists()
        assert current_attempt(lake.conn, "idea-cpu", "action")["state"] == "RUNNING"
        assert budget.snapshot(lake, scope)["active_reservations"] == 1
        yield {
            "lake": lake, "results": results, "scope": scope, "cfg": cfg,
            "permit": permit, "handle": handle, "work": work,
            "started_at": owner.started_at, "deadline": owner.deadline,
            "worker_fd": worker_fd, "supervisor_fd": supervisor_fd,
        }
    finally:
        # This is an owned cleanup operation after the scientific/lease check.
        # Never signal a raw PID or scan unrelated host processes.
        try:
            native.stop(handle, results, cfg, lake=lake, permit=permit)
        finally:
            if trusted_worker and not select.select([worker_fd], [], [], 0)[0]:
                signal.pidfd_send_signal(worker_fd, signal.SIGKILL)
            try:
                supervisor_code = process._supervisor.wait(timeout=4)
                print(json.dumps({"cleanup": "explicit_owned_stop_after_case",
                                  "supervisor_returncode": supervisor_code,
                                  "worker_pidfd_exited": bool(select.select([worker_fd], [], [], 0)[0])}), flush=True)
            finally:
                os.close(worker_fd)
                os.close(supervisor_fd)


def _release(case, finish_at):
    (case["work"] / "diagnostic-release.json").write_text(
        json.dumps({"finish_at": finish_at}), encoding="utf-8",
    )


def _observe_natural_exit_without_harvesting(case):
    # Observe only captured kernel handles; do not receive supervisor frames,
    # invoke process.poll(), harvest(), STOP, or close the owner's channel.
    assert select.select([case["supervisor_fd"]], [], [], 6)[0]
    assert case["handle"].process._supervisor.wait(timeout=1) == 0
    assert select.select([case["worker_fd"]], [], [], 0)[0]
    return json.loads((case["work"] / "diagnostic-done.json").read_text())


def test_normal_completion_before_deadline_is_a_positive_control(execution):
    case = execution
    _release(case, time.monotonic() + .05)
    terminal = _finish(case["handle"], case["results"], case["cfg"], case["lake"], case["permit"])
    done = json.loads((case["work"] / "diagnostic-done.json").read_text())
    print(json.dumps({"case": "normal_control", "worker": done,
                      "started_at": case["started_at"], "deadline": case["deadline"],
                      "terminal": terminal}), flush=True)
    assert done["finished"] < case["deadline"]
    assert terminal["outcome"] == "completed" and terminal["return_code"] == 0
    assert terminal["process_tree"]["wait_proof"] == "ECHILD_WALL"
    assert terminal["process_tree"]["stop_requested"] is False
    assert budget.snapshot(case["lake"], case["scope"])["active_reservations"] == 0


def test_over_deadline_natural_zero_cannot_be_published_as_completed(execution):
    case = execution
    _release(case, case["deadline"] + .35)
    done = _observe_natural_exit_without_harvesting(case)
    assert done["finished"] > case["deadline"]
    assert current_attempt(case["lake"].conn, "idea-cpu", "action")["state"] == "RUNNING"
    terminal = native.harvest(
        case["handle"], case["results"], case["cfg"],
        lake=case["lake"], permit=case["permit"],
    )
    print(json.dumps({"case": "over_deadline_natural_zero", "worker": done,
                      "started_at": case["started_at"], "deadline": case["deadline"],
                      "terminal": terminal}), flush=True)
    assert terminal is not None
    assert terminal["outcome"] != "completed", "actual work exceeded its captured bound before natural exit"


def test_supervisor_enforces_deadline_while_parent_keeps_channel_without_polling(execution):
    case = execution
    _release(case, case["deadline"] + 3)
    channel_fd = case["handle"].process._channel.fileno()
    assert channel_fd >= 0
    wait_seconds = max(0, case["deadline"] + 1 - time.monotonic())
    worker_exited = bool(select.select([case["worker_fd"]], [], [], wait_seconds)[0])
    observed_at = time.monotonic()
    print(json.dumps({"case": "no_parent_poll", "deadline": case["deadline"],
                      "observed_at": observed_at, "worker_pidfd_exited": worker_exited,
                      "owner_channel_open": case["handle"].process._channel.fileno() == channel_fd,
                      "supervisor_pidfd_exited": bool(select.select([case["supervisor_fd"]], [], [], 0)[0]),
                      "state_before_fixture_stop": current_attempt(case["lake"].conn, "idea-cpu", "action")["state"]}), flush=True)
    assert observed_at >= case["deadline"]
    assert worker_exited, "captured worker remained alive after its runtime bound plus one-second cleanup allowance"
