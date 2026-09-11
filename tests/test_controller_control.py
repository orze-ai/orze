"""Internal registration mechanisms; not CLI ACK or autoresearch-loop proof.

Every registered context lives in a fresh exec process. Exiting that process
does not release its durable owner, erase its row or certify any drain.
"""
import os
from pathlib import Path
import select
import signal
import subprocess
import sys

import pytest


_PREAMBLE = '''
import json
import os
from pathlib import Path
import sqlite3
import sys
from orze.idea_lake import IdeaLake
from orze.engine import controller_control as control
from orze.engine.supervisor_worker import process_identity

root = Path(sys.argv[1])
scope = root / "results"
scope.mkdir()
lake = IdeaLake(str(root / "ideas.db"))

def refuses(call, kind=control.ControllerHOLD):
    try:
        call()
    except kind:
        return
    raise AssertionError("unexpected controller authority")

def row(ctx):
    return lake.conn.execute("SELECT controller_id, scope, identity_json, phase, "
        "request_id, hold_reason FROM main.controller_instances WHERE controller_id=?",
        (ctx.controller_id,)).fetchone()
'''


def _run(tmp_path, body):
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    environment["PYTHONPATH"] = os.pathsep.join([
        str(Path(__file__).resolve().parents[1] / "src"), environment.get("PYTHONPATH", "")])
    child = subprocess.Popen([sys.executable, "-c", _PREAMBLE + body, str(tmp_path)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=environment)
    descriptor = os.pidfd_open(child.pid)
    try:
        stdout, stderr = child.communicate(timeout=20)
        assert child.returncode == 0, stdout + stderr
    finally:
        if not select.select([descriptor], [], [], 0)[0]:
            signal.pidfd_send_signal(descriptor, signal.SIGKILL)
        child.wait(timeout=5)
        os.close(descriptor)


def test_exact_durable_identity_is_detached_and_idempotent(tmp_path):
    _run(tmp_path, '''
ctx = control.register_controller(lake, scope)
assert control.current_controller() is ctx
assert control.register_controller(lake, scope) is ctx
actual = json.loads(row(ctx)[2])
assert actual == ctx.identity
assert actual["process"] == process_identity(os.getpid())[0]
assert actual["database"] == str(root / "ideas.db")
assert actual["scope_inode"] == scope.stat().st_ino
copy = ctx.identity
copy["process"]["pid"] = 1
assert ctx.identity == actual
assert row(ctx)[3:] == ("ACTIVE", None, None)
ctx.check_admission()
assert not hasattr(ctx, "ack") and not hasattr(ctx, "release")
assert (scope / "_controller_registration.lock" / "lock.json").is_file()
''')
    # A clean Python exit does not erase the registration or owner.
    assert (tmp_path / "results/_controller_registration.lock/lock.json").is_file()


def test_quiesce_is_exact_and_does_not_supply_closure(tmp_path):
    _run(tmp_path, '''
ctx = control.register_controller(lake, scope)
ctx.quiesce("stop-one")
ctx.quiesce("stop-one")
assert ctx.quiescing is True
assert ctx.poll_control() == "QUIESCING"
refuses(ctx.check_admission, control.ControllerQuiescing)
assert control.register_controller(lake, scope) is ctx
assert row(ctx)[3:] == ("QUIESCING", "stop-one", None)
assert lake.conn.execute("SELECT COUNT(*) FROM main.controller_instances").fetchone()[0] == 1
assert not (scope / "controller-ack.json").exists()
''')


def test_replacing_accepted_request_is_sticky_hold(tmp_path):
    _run(tmp_path, '''
ctx = control.register_controller(lake, scope)
ctx.quiesce("stop-one")
refuses(lambda: ctx.quiesce("stop-two"))
assert row(ctx)[3:] == ("HOLD", "stop-one", "controller_request_replaced")
refuses(ctx.check_admission)
refuses(lambda: control.register_controller(lake, scope))
assert control.current_controller() is ctx
''')


def test_registration_does_not_commit_callers_transaction(tmp_path):
    _run(tmp_path, '''
lake.conn.execute("BEGIN IMMEDIATE")
lake.conn.execute("CREATE TABLE caller_pending (value TEXT)")
refuses(lambda: control.register_controller(lake, scope))
assert lake.conn.in_transaction
assert control.current_controller() is None
assert not (scope / "_controller_registration.lock").exists()
lake.conn.rollback()
assert lake.conn.execute("SELECT name FROM main.sqlite_master WHERE name='caller_pending'").fetchone() is None
''')


def test_native_admission_reads_exact_callers_transaction_without_ending_it(tmp_path):
    _run(tmp_path, '''
ctx = control.register_controller(lake, scope)
lake.conn.execute("BEGIN IMMEDIATE")
lake.conn.execute("CREATE TABLE caller_pending (value TEXT)")
ctx.check_admission(lake.conn)
assert lake.conn.in_transaction
lake.conn.rollback()
assert lake.conn.execute("SELECT name FROM main.sqlite_master WHERE name='caller_pending'").fetchone() is None
ctx.check_admission()
''')


@pytest.mark.parametrize("kind", ["memory", "scope_link", "database_link", "database_hardlink"])
def test_registration_refuses_unqualified_storage_before_reserving(tmp_path, kind):
    _run(tmp_path, 'kind = ' + repr(kind) + '''
if kind == "memory":
    lake.close()
    # IdeaLake itself already rejects :memory: under its DELETE-journal
    # policy. Replace this real Lake's closed connection to exercise the
    # registrar's independent persistent-main-route check, not that loader.
    lake.conn = sqlite3.connect(":memory:")
    lake.db_path = ":memory:"
elif kind == "scope_link":
    alias = root / "alias"
    alias.symlink_to(scope, target_is_directory=True)
    scope = alias
elif kind == "database_link":
    alias = root / "alias.db"
    alias.symlink_to(root / "ideas.db")
    lake.db_path = str(alias)
else:
    os.link(root / "ideas.db", root / "alias.db")
refuses(lambda: control.register_controller(lake, scope))
assert control.current_controller() is None
assert not (scope / "_controller_registration.lock").exists()
''')


def test_commit_response_loss_preserves_registration_and_strong_hold(tmp_path):
    _run(tmp_path, '''
original = sqlite3.connect
events = []
class CommitResponseLost(sqlite3.Connection):
    def commit(self):
        super().commit()
        events.append("committed")
        if len(events) == 1:
            raise OSError("controlled response lost after actual commit")

def connection(*args, **kwargs):
    return original(*args, **kwargs, factory=CommitResponseLost)

control.sqlite3.connect = connection
refuses(lambda: control.register_controller(lake, scope))
ctx = control.current_controller()
assert ctx is not None
assert row(ctx)[3] == "HOLD"
assert events == ["committed", "committed"]
refuses(ctx.check_admission)
refuses(lambda: control.register_controller(lake, scope))
assert control.current_controller() is ctx
assert lake.conn.execute("SELECT COUNT(*) FROM main.controller_instances").fetchone()[0] == 1
assert (scope / "_controller_registration.lock" / "lock.json").is_file()
''')


def test_unexpected_schema_is_not_implicitly_migrated_or_retried(tmp_path):
    _run(tmp_path, '''
lake.conn.execute("CREATE TABLE controller_instances (controller_id TEXT)")
lake.conn.commit()
refuses(lambda: control.register_controller(lake, scope))
ctx = control.current_controller()
assert ctx is not None
refuses(ctx.check_admission)
assert lake.conn.execute("PRAGMA main.table_info(controller_instances)").fetchall()[0][1] == "controller_id"
assert lake.conn.execute("SELECT COUNT(*) FROM main.controller_instances").fetchone()[0] == 0
assert (scope / "_controller_registration.lock" / "lock.json").is_file()
refuses(lambda: control.register_controller(lake, scope))
''')


def test_foreign_trigger_revokes_admission_without_running_trigger(tmp_path):
    _run(tmp_path, '''
ctx = control.register_controller(lake, scope)
lake.conn.execute("CREATE TABLE foreign_effect (value TEXT)")
lake.conn.execute("CREATE TRIGGER unexpected_controller_write AFTER UPDATE ON controller_instances "
    "BEGIN INSERT INTO foreign_effect VALUES ('ran'); END")
lake.conn.commit()
refuses(ctx.check_admission)
assert control.current_controller() is ctx
assert lake.conn.execute("SELECT COUNT(*) FROM foreign_effect").fetchone()[0] == 0
refuses(lambda: ctx.quiesce("after-hold"))
assert lake.conn.execute("SELECT COUNT(*) FROM foreign_effect").fetchone()[0] == 0
''')
