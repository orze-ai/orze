"""C3 real publication windows, not missing-API historical regressions.

Reuse the old SQLite/native fixture without changing it. Time advances on the
actual CLOCK_BOOTTIME; wrappers delegate real artifact registration, SQL commit
and effect confirmation before delaying their response. No fabricated closure,
terminal, lease timestamp or authorization is installed. Fresh recovery below
is a new Python interpreter calling F, not a claim of full CLI coverage.
"""
import json
import os
from pathlib import Path
import select
import signal
import sqlite3
import subprocess
import sys
import time

import pytest

from test_native_cpu_action import context, _finish
from orze.core import cpu_action_budget as budget
from orze.core.execution_attempts import current_attempt
from orze.core.research_artifacts import artifacts_for_attempt
from orze.engine import native_cpu_action as native
from orze.engine import attempt_effect_receipts as effects
from orze.engine.supervisor_worker import process_identity
from orze.engine.termination_hold import require_no_unconfirmed_stop


_GATED_WORKER = """import os,time
from pathlib import Path
Path('lease-ready').write_text('ready')
until=time.monotonic()+15
while not Path('lease-go').exists():
 if time.monotonic()>until: os._exit(91)
 time.sleep(.005)
exec(compile(BODY,'<owned-lease-test>','exec'))
"""
_RESULT = "from pathlib import Path; Path('result.txt').write_text('owned result')"
_TREE = """import json,os,signal,time
from pathlib import Path
signal.signal(signal.SIGTERM,lambda *_:os._exit(0))
child=os.fork()
if child==0:
 os.setsid()
 Path('escaped.json').write_text(json.dumps({'pid':os.getpid()}))
 until=time.monotonic()+15
 while time.monotonic()<until: time.sleep(.01)
 os._exit(92)
until=time.monotonic()+15
while time.monotonic()<until: time.sleep(.01)
os._exit(93)
"""


def _exited(fd):
    return bool(select.select([fd], [], [], 0)[0])


def _wait_file(path):
    until = time.monotonic() + 3
    while not path.exists() and time.monotonic() < until:
        time.sleep(.005)
    assert path.exists(), str(path)


def _now():
    return time.clock_gettime_ns(time.CLOCK_BOOTTIME)


def _expire(descriptor):
    before = _now()
    assert before < descriptor["deadline_ns"], "injection did not enter while authorized"
    target = descriptor["deadline_ns"] + 20_000_000
    while True:
        remaining = target - _now()
        if remaining <= 0:
            break
        time.sleep(min(.01, remaining / 1_000_000_000))
    print(json.dumps({"boundary": "actual_clock_expiry", "before_ns": before,
                      "after_ns": _now(), "lease": descriptor}), flush=True)


@pytest.fixture
def launched(context):
    lake, results, scope, cfg, create, handles = context
    captured = []

    def launch(body=_RESULT, *, ttl=1.5, outputs=True):
        if ttl is not None:
            cfg["cpu_runtime_lease"] = {"version": 1, "ttl_seconds": ttl}
        action, permit = create(_GATED_WORKER.replace("BODY", repr(body)), timeout=4,
            outputs={"result": {"path": "result.txt", "max_bytes": 128}} if outputs else {})

        def admission():
            budget.require_permit(lake, permit)
            require_no_unconfirmed_stop(results / "idea-cpu")

        handle = native.launch("idea-cpu", results, cfg, lake=lake, action=action,
                               permit=permit, admission=admission)
        process = handle.process
        item = {"handle": handle, "process": process, "permit": permit, "fds": []}
        captured.append(item)
        supervisor_fd = os.dup(process.supervisor_pidfd)
        item["fds"].append(supervisor_fd)
        worker_fd = os.pidfd_open(process.pid, 0)
        # This PID came from our actual READY-bound process; authenticate its
        # birth and parent before any possible fixture signal through the fd.
        try:
            identity, parent = process_identity(process.pid)
            assert identity == process.binding["worker"] and parent == process.supervisor_pid
        except BaseException:
            os.close(worker_fd)
            raise
        item["fds"].append(worker_fd)
        work = results / "idea-cpu" / "_action_attempts" / handle.attempt_id / "work"
        _wait_file(work / "lease-ready")
        row = current_attempt(lake.conn, "idea-cpu", "action")
        assert row["state"] == "RUNNING"
        case = {"lake": lake, "results": results, "scope": scope, "cfg": cfg,
                "handle": handle, "permit": permit, "process": process, "work": work,
                "lease": row["binding"]["runtime_lease"], "fds": item["fds"],
                "worker_fd": worker_fd, "supervisor_fd": supervisor_fd}
        (work / "lease-go").write_text("go", encoding="utf-8")
        return case

    yield launch
    for item in captured:
        process = item["process"]
        try:
            try:
                native.stop(item["handle"], results, cfg, lake=lake, permit=item["permit"])
            except Exception:
                # A deliberately lost supervisor is not given a fake closure.
                # Retain product HOLD/strong owner; only clean captured OS fds.
                pass
            for fd in reversed(item["fds"]):
                if not _exited(fd):
                    signal.pidfd_send_signal(fd, signal.SIGKILL)
            process._supervisor.wait(timeout=4)
            assert all(select.select([fd], [], [], 2)[0] for fd in item["fds"])
        finally:
            for fd in item["fds"]:
                os.close(fd)
            if process._channel is not None:
                process._channel.close()
            # The original fixture assumes a confirmed process.poll(). A lost
            # supervisor cannot satisfy it; our own cleanup above substitutes
            # only teardown, not any product proof or native owner removal.
            if process in handles:
                handles.remove(process)


def _finish_case(case):
    return _finish(case["handle"], case["results"], case["cfg"], case["lake"], case["permit"])


def _row(case):
    return current_attempt(case["lake"].conn, "idea-cpu", "action")


def _effect_paths(case):
    root = case["results"] / "idea-cpu" / "_execution_effects" / case["handle"].attempt_id
    return root / "prepared.json", root / "committed.json"


def _fresh_recovery(case):
    """No worker launch: actual separate interpreter/Lake/F plus claim refusal."""
    script = """import json,sys
from pathlib import Path
from orze.idea_lake import IdeaLake
from orze.core import cpu_action_budget as budget
from orze.engine.scheduler import claim
from orze.engine.termination_hold import TerminationUnconfirmed
scope=json.loads(sys.argv[1]); lake=IdeaLake(scope['database'])
try:
 before=lake.conn.execute('SELECT count(*) FROM execution_attempts').fetchone()[0]
 try:
  result=budget.reconcile_confirmed_terminals(lake,scope); held=False
 except budget.CpuBudgetHOLD:
  result=None; held=True
 path=Path(scope['results_dir'])/'idea-cpu'/'claim.json'; original=path.read_bytes()
 try: claimed=claim('idea-cpu',Path(scope['results_dir']),None,lake,resource='cpu')
 except TerminationUnconfirmed: claimed=False
 print(json.dumps({'held':held,'result':result,'claimed':claimed,
  'claim_unchanged':path.read_bytes()==original,'before':before,
  'after':lake.conn.execute('SELECT count(*) FROM execution_attempts').fetchone()[0],
  'reservations':[list(r) for r in lake.conn.execute('SELECT state,terminal_sha256 FROM cpu_action_reservations')]}))
finally: lake.close()
"""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    child = subprocess.Popen([sys.executable, "-c", script, json.dumps(case["scope"])],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    fd = os.pidfd_open(child.pid, 0)
    try:
        stdout, stderr = child.communicate(timeout=10)
        assert child.returncode == 0, (stdout, stderr)
    finally:
        if not _exited(fd):
            signal.pidfd_send_signal(fd, signal.SIGKILL)
        child.wait(timeout=3)
        os.close(fd)
    result = json.loads(stdout)
    print(json.dumps({"fresh_recovery": result}), flush=True)
    assert result["before"] == result["after"] == 1
    assert result["claimed"] is False and result["claim_unchanged"] is True
    assert result["reservations"] == [["BOUND", None]]
    return result


def test_default_lease_completes_and_charges_original_envelope(launched):
    case = launched(ttl=None)
    terminal = _finish_case(case)
    descriptor = case["lease"]
    assert descriptor["deadline_ns"] - descriptor["issued_ns"] == 4_000_000_000
    assert terminal["runtime_lease"]["status"] == "authorized"
    assert descriptor["issued_ns"] <= terminal["runtime_lease"]["observed_ns"] < descriptor["deadline_ns"]
    assert terminal["outcome"] == "completed" and terminal["return_code"] == 0
    assert terminal["process_tree"]["schema"] == 2 and terminal["process_tree"]["lease_expired"] is False
    assert len(artifacts_for_attempt(case["lake"].conn, case["handle"].attempt_ref)) == 1
    snapshot = budget.snapshot(case["lake"], case["scope"])
    assert snapshot["active_reservations"] == 0 and snapshot["reserved_wall_seconds"] == 4


def test_short_lease_autonomously_closes_term_zero_and_escaped_child_without_parent_poll(launched):
    case = launched(_TREE, outputs=False)
    marker = case["work"] / "escaped.json"
    _wait_file(marker)
    child = json.loads(marker.read_text())["pid"]
    fd = os.pidfd_open(child, 0)
    try:
        identity, parent = process_identity(child)
        assert identity["pid"] == child and parent == case["process"].pid
    except BaseException:
        os.close(fd)
        raise
    case["fds"].append(fd)
    # Observe kernel fds only. No process.poll(), harvest(), STOP or channel EOF
    # occurs before this assertion of independently completed supervision.
    channel = case["process"]._channel.fileno()
    assert select.select([case["supervisor_fd"]], [], [], 5)[0]
    assert _exited(case["worker_fd"]) and _exited(fd)
    assert case["process"]._channel.fileno() == channel and channel >= 0
    terminal = _finish_case(case)
    closure = terminal["process_tree"]
    assert terminal["return_code"] == 0 and terminal["outcome"] == "interrupted"
    assert terminal["reason_code"] == "cpu_runtime_lease_expired"
    assert closure["lease_expired"] is True and closure["stop_requested"] is True
    assert closure["reaped_children"] >= 2 and closure["wait_proof"] == "ECHILD_WALL"
    assert terminal["artifact_ids"] == terminal["observation_ids"] == []
    snapshot = budget.snapshot(case["lake"], case["scope"])
    assert snapshot["active_reservations"] == 0 and snapshot["reserved_wall_seconds"] == 4


def test_natural_closure_then_slow_artifact_preparation_expires_without_fake_stop(launched, monkeypatch):
    case = launched()
    original = native.prepare_artifacts
    entered = []

    def slow(*args, **kwargs):
        prepared = original(*args, **kwargs)
        entered.append(case["process"].closure_receipt())
        _expire(case["lease"])
        return prepared

    monkeypatch.setattr(native, "prepare_artifacts", slow)
    terminal = _finish_case(case)
    assert len(entered) == 1 and entered[0]["lease_expired"] is False
    assert terminal["outcome"] == "interrupted" and terminal["reason_code"] == "cpu_runtime_lease_expired"
    assert terminal["runtime_lease"]["status"] == "expired"
    assert terminal["process_tree"]["stop_requested"] is False
    assert terminal["process_tree"]["lease_expired"] is False
    assert terminal["artifact_ids"] == terminal["observation_ids"] == []
    assert budget.snapshot(case["lake"], case["scope"])["active_reservations"] == 0


def test_expiry_after_actual_registration_retains_prepared_effect_without_second_terminal(launched, monkeypatch):
    case = launched()
    original = native.register_artifacts
    observed = []

    def late(conn, ref, records):
        result = original(conn, ref, records)
        observed.extend(artifacts_for_attempt(conn, ref))
        _expire(case["lease"])
        return result

    monkeypatch.setattr(native, "register_artifacts", late)
    with pytest.raises(native.CPUActionHOLD):
        _finish_case(case)
    prepared, committed = _effect_paths(case)
    assert len(observed) == 1 and prepared.is_file() and not committed.exists()
    assert (case["results"] / "idea-cpu" / "_attempt_effect.lock").is_dir()
    assert _row(case)["state"] == "RUNNING" and _row(case)["terminal"] is None
    assert artifacts_for_attempt(case["lake"].conn, case["handle"].attempt_ref) == ()
    before = prepared.read_bytes()
    with pytest.raises(native.CPUActionHOLD):
        _finish_case(case)
    assert prepared.read_bytes() == before
    assert _fresh_recovery(case)["held"] is True


@pytest.mark.parametrize("boundary", ["commit", "confirm"])
def test_actual_publication_response_after_deadline_holds_and_fresh_f_cannot_settle(launched, monkeypatch, boundary):
    case = launched()
    lake, seen = case["lake"], []
    if boundary == "commit":
        class LateCommit(sqlite3.Connection):
            def commit(self):
                row = self.execute("SELECT state FROM execution_attempts WHERE attempt_id=?",
                                   (case["handle"].attempt_id,)).fetchone()
                terminal = self.in_transaction and row is not None and row[0] == "TERMINAL"
                super().commit()
                if terminal and not seen:
                    seen.append(current_attempt(self, "idea-cpu", "action")["terminal"])
                    _expire(case["lease"])

        lake.conn.close()
        lake.conn = sqlite3.connect(str(lake.db_path), factory=LateCommit)
        lake.conn.row_factory = sqlite3.Row
    else:
        original = effects.confirm_effect

        def late(*args, **kwargs):
            result = original(*args, **kwargs)
            assert _effect_paths(case)[1].is_file()
            seen.append(_row(case)["terminal"])
            _expire(case["lease"])
            return result

        monkeypatch.setattr(effects, "confirm_effect", late)
    with pytest.raises(native.CPUActionHOLD):
        _finish_case(case)
    prepared, committed = _effect_paths(case)
    assert len(seen) == 1 and seen[0]["runtime_lease"]["status"] == "authorized"
    assert _row(case)["state"] == "TERMINAL" and _row(case)["terminal"] == seen[0]
    assert prepared.is_file() and committed.exists() is (boundary == "confirm")
    assert (case["results"] / "idea-cpu" / "_attempt_effect.lock").is_dir()
    assert _fresh_recovery(case)["held"] is True


def test_owned_supervisor_loss_after_go_remains_hold_after_expiry_and_fresh_claim(launched):
    case = launched("import time; time.sleep(15)", outputs=False)
    signal.pidfd_send_signal(case["supervisor_fd"], signal.SIGKILL)
    assert select.select([case["supervisor_fd"]], [], [], 2)[0]
    _expire(case["lease"])
    with pytest.raises(native.CPUActionHOLD):
        _finish_case(case)
    assert _row(case)["state"] == "RUNNING" and _row(case)["terminal"] is None
    assert not _effect_paths(case)[1].exists()
    assert budget.snapshot(case["lake"], case["scope"])["active_reservations"] == 1
    _fresh_recovery(case)
