"""Independent internal-registration mechanisms, not product-loop or ACK tests.

Every successful registration lives in a real isolated Python process. We do
not reset the production global, release its durable owner, or pretend process
exit proves action/resource closure. SQLite, process identity and namespace
ownership stay real. Cleanup signals only pidfds of children this test owns.
"""
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not hasattr(os, "pidfd_open"),
    reason="Linux owned-process pidfds required")

_CORE_SOURCE = Path(__file__).resolve().parents[1] / "src"
_COMMON = r'''
import json, os, pathlib, select, signal, sqlite3, sys, threading
from orze.idea_lake import IdeaLake
from orze.engine import controller_control as cc
base = pathlib.Path(sys.argv[1])
scope = base / "results"
def refused(action):
    try:
        action()
    except cc.ControllerHOLD:
        return
    raise AssertionError("controller operation unexpectedly granted authority")
def controller_rows(conn):
    return conn.execute("SELECT controller_id,scope,identity_json,phase,request_id,hold_reason "
                        "FROM main.controller_instances ORDER BY controller_id").fetchall()
'''


def _run_owned(base, body, *args):
    environment = dict(os.environ)
    environment.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(_CORE_SOURCE), environment.get("PYTHONPATH", "")])
    child = subprocess.Popen(
        [sys.executable, "-c", _COMMON + body, str(base), *args],
        cwd=base, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    descriptor = os.pidfd_open(child.pid)
    try:
        try:
            stdout, stderr = child.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            signal.pidfd_send_signal(descriptor, signal.SIGKILL)
            stdout, stderr = child.communicate(timeout=5)
            pytest.fail("owned registration child exceeded its deadline: " +
                        stderr.decode("utf-8", errors="replace")[-8000:])
        assert child.returncode == 0, stderr.decode("utf-8", errors="replace")[-12000:]
        assert select.select([descriptor], [], [], 0)[0] == [descriptor]
        return json.loads(stdout)
    finally:
        if not select.select([descriptor], [], [], 0)[0]:
            signal.pidfd_send_signal(descriptor, signal.SIGKILL)
        child.wait(timeout=5)
        os.close(descriptor)


@pytest.fixture
def project(tmp_path):
    (tmp_path / "results").mkdir()
    return tmp_path


def test_registered_context_is_shared_with_real_thread_and_exact_reentry(project):
    result = _run_owned(project, r'''
lake = IdeaLake(base / "lake.db")
ctx = cc.register_controller(lake, scope)
before = controller_rows(lake.conn)
assert cc.register_controller(lake, scope) is ctx
assert cc.current_controller() is ctx
copy = ctx.identity
copy["controller_id"] = "not-the-owner"
assert ctx.identity["controller_id"] == ctx.controller_id
observed, failures = [], []
def inspect():
    try:
        assert cc.current_controller() is ctx
        with ctx.guard():
            ctx.check_admission()
            assert ctx.poll_control() == "ACTIVE"
        observed.append(ctx.controller_id)
    except BaseException as exc:
        failures.append(repr(exc))
thread = threading.Thread(target=inspect)
thread.start(); thread.join(timeout=3)
assert not thread.is_alive(), "controller context blocked a framework thread"
assert failures == [], failures
assert observed == [ctx.controller_id]
assert controller_rows(lake.conn) == before
identity = ctx.identity
assert identity["process"]["pid"] == os.getpid()
assert type(identity["process"]["start_ticks"]) is int
assert identity["process"]["start_ticks"] > 0
assert identity["scope"] == str(scope)
assert identity["database"] == str(base / "lake.db")
print(json.dumps({"phase": ctx.poll_control(), "thread_seen": len(observed)}))
lake.close()
''')
    assert result == {"phase": "ACTIVE", "thread_seen": 1}


def test_fork_refuses_inherited_context_before_either_inherited_lock(project):
    result = _run_owned(project, r'''
lake = IdeaLake(base / "lake.db")
ctx = cc.register_controller(lake, scope)
before = controller_rows(lake.conn)
locked, release = threading.Event(), threading.Event()
def hold_locks():
    with cc._GLOBAL_GUARD, ctx.guard():
        locked.set()
        release.wait(timeout=8)
holder = threading.Thread(target=hold_locks)
holder.start()
assert locked.wait(timeout=3)
read_fd, write_fd = os.pipe()
child_pid = os.fork()
if child_pid == 0:
    os.close(read_fd)
    try:
        def enter_guard():
            with ctx.guard():
                raise AssertionError("fork acquired parent authority")
        for action in (cc.current_controller,
                       lambda: cc.register_controller(lake, scope),
                       ctx.check_admission, enter_guard,
                       lambda: ctx.hold("child_must_not_write")):
            refused(action)
        os.write(write_fd, b'{"refused":5}')
    except BaseException as exc:
        os.write(write_fd, repr(exc).encode("utf-8")[:2048])
        os._exit(1)
    os._exit(0)
os.close(write_fd)
child_fd = os.pidfd_open(child_pid)
try:
    assert select.select([child_fd], [], [], 3)[0] == [child_fd], "fork touched inherited locked state"
    waited, status = os.waitpid(child_pid, 0)
    assert waited == child_pid and os.waitstatus_to_exitcode(status) == 0
    assert json.loads(os.read(read_fd, 4096)) == {"refused": 5}
finally:
    if not select.select([child_fd], [], [], 0)[0]:
        signal.pidfd_send_signal(child_fd, signal.SIGKILL)
    try:
        os.waitpid(child_pid, 0)
    except ChildProcessError:
        pass
    os.close(child_fd); os.close(read_fd)
    release.set(); holder.join(timeout=3)
assert not holder.is_alive()
assert cc.current_controller() is ctx
ctx.check_admission()
assert controller_rows(lake.conn) == before
print(json.dumps({"fork_refused": 5, "parent_phase": ctx.poll_control()}))
lake.close()
''')
    assert result == {"fork_refused": 5, "parent_phase": "ACTIVE"}


def test_other_lake_or_replaced_connection_cannot_redirect_registration(project):
    result = _run_owned(project, r'''
lake = IdeaLake(base / "lake.db")
ctx = cc.register_controller(lake, scope)
before = controller_rows(lake.conn)
same_database = IdeaLake(base / "lake.db")
other_database = IdeaLake(base / "other.db")
refused(lambda: cc.register_controller(same_database, scope))
refused(lambda: cc.register_controller(other_database, scope))
assert cc.current_controller() is ctx
ctx.check_admission()
assert controller_rows(lake.conn) == before
assert other_database.conn.execute("SELECT 1 FROM main.sqlite_master WHERE name='controller_instances'").fetchone() is None
original_connection = lake.conn
lake.conn = other_database.conn
refused(ctx.check_admission)
assert cc.current_controller() is ctx
lake.conn = original_connection
refused(lambda: cc.register_controller(lake, scope))
refused(ctx.check_admission)
assert controller_rows(original_connection) == before
assert other_database.conn.execute("SELECT 1 FROM main.sqlite_master WHERE name='controller_instances'").fetchone() is None
print(json.dumps({"original_retained": True, "foreign_registration_rows": 0}))
same_database.close(); other_database.close(); lake.close()
''')
    assert result == {"original_retained": True, "foreign_registration_rows": 0}


def test_peer_registration_identity_change_stays_hold_without_new_owner(project):
    result = _run_owned(project, r'''
lake = IdeaLake(base / "lake.db")
ctx = cc.register_controller(lake, scope)
owner_path = scope / "_controller_registration.lock" / "lock.json"
owner_before = owner_path.read_bytes()
peer = sqlite3.connect(base / "lake.db")
changed = ctx.identity
changed["process"]["start_ticks"] += 1
changed_json = cc.canonical(changed).decode("utf-8")
peer.execute("UPDATE main.controller_instances SET identity_json=? WHERE controller_id=?",
             (changed_json, ctx.controller_id))
peer.commit()
tampered = controller_rows(peer)
refused(ctx.poll_control)
refused(ctx.check_admission)
refused(lambda: cc.register_controller(lake, scope))
assert cc.current_controller() is ctx
assert controller_rows(peer) == tampered
assert len(tampered) == 1 and tampered[0][0] == ctx.controller_id
assert owner_path.read_bytes() == owner_before
print(json.dumps({"registered_rows": len(tampered), "owner_retained": True}))
peer.close(); lake.close()
''')
    assert result == {"registered_rows": 1, "owner_retained": True}


def test_exited_process_registration_blocks_same_scope_even_in_another_database(project):
    first = _run_owned(project, r'''
lake = IdeaLake(base / "lake.db")
ctx = cc.register_controller(lake, scope)
ctx.check_admission()
print(json.dumps({"controller_id": ctx.controller_id, "identity": ctx.identity}))
lake.close()
''')
    owner = project / "results" / "_controller_registration.lock" / "lock.json"
    owner_bytes = owner.read_bytes()
    owner_inode = (owner.stat().st_dev, owner.stat().st_ino)
    for filename in ("lake.db", "other.db"):
        result = _run_owned(project, r'''
lake = IdeaLake(base / sys.argv[2])
refused(lambda: cc.register_controller(lake, scope))
pending = cc.current_controller()
assert pending is not None
refused(pending.check_admission)
if sys.argv[2] == "lake.db":
    rows = controller_rows(lake.conn)
    assert len(rows) == 1
    print(json.dumps({"controller_ids": [rows[0][0]], "phase": rows[0][3]}))
else:
    assert lake.conn.execute("SELECT 1 FROM main.sqlite_master WHERE name='controller_instances'").fetchone() is None
    print(json.dumps({"controller_ids": []}))
lake.close()
''', filename)
        if filename == "lake.db":
            assert result == {"controller_ids": [first["controller_id"]], "phase": "ACTIVE"}
        else:
            assert result == {"controller_ids": []}
        assert owner.read_bytes() == owner_bytes
        assert (owner.stat().st_dev, owner.stat().st_ino) == owner_inode
    assert hashlib.sha256(owner_bytes).hexdigest() == first["identity"]["owner_metadata_sha256"]
