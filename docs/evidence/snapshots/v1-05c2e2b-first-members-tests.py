"""Internal membership mechanisms: real isolated controllers and CPU workers.

No Orze lifecycle/CLI ACK is claimed. Faults are explicit SQL/guard boundary
injections; subprocesses never run a provider, GPU probe or host process scan.
"""
import os
from pathlib import Path
import subprocess
import sys

import pytest


COMMON = r'''
import contextlib, hashlib, json, os, pathlib, sqlite3, subprocess, sys
from dataclasses import asdict
from orze.idea_lake import IdeaLake
from orze.engine import controller_control as control, controller_members as members
from orze.engine import supervised_process as primitive
from orze.engine import execution_authority as authority
from orze.core.execution_attempts import create_attempt, mark_running, finish_attempt
base = pathlib.Path(sys.argv[1]); scope = base / 'results'; scope.mkdir()
lake = IdeaLake(base / 'lake.db')
ctx = control.register_controller(lake, scope)
handles = []
def identity(token='probe'):
    return {'schema':1, 'kind':'controller_probe', 'scope':str(scope), 'invocation_id':token}
def prepare(code='pass', ident=None):
    p = primitive.prepare_supervised([sys.executable, '-c', code],
        identity=identity() if ident is None else ident, cwd=scope,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    handles.append(p)
    return p
def rows():
    return [json.loads(r[0]) for r in lake.conn.execute('SELECT payload_json FROM controller_members')]
def refused(fn):
    try: fn()
    except control.ControllerHOLD: return
    raise AssertionError('unexpected admission')
def native():
    folder=scope/'idea-one'; folder.mkdir()
    with authority.execution_transaction(lake, folder) as tx:
        ref=create_attempt(tx.conn, 'idea-one', 'training', 'attempt-one', {})
        tx.watch_attempt(ref)
    p=prepare(ident={'attempt_ref':asdict(ref), 'scope':str(folder)})
    with authority.execution_transaction(lake, folder) as tx:
        mark_running(tx.conn, ref, {'process_supervision':p.binding})
        tx.watch_attempt(ref)
    p.start(); assert p.wait(timeout=5)==0
    return folder,ref,p
'''


def run_case(tmp_path, body):
    cleanup = r'''
finally:
    for p in handles:
        try:
            if p._supervisor.poll() is None:
                if p._uncertainty:
                    p._channel.close()
                elif not p._stop_sent:
                    p._send('STOP'); p._stop_sent=True
            p._supervisor.wait(timeout=5)
        finally:
            p._close_descriptors()
    lake.close()
'''
    script = COMMON + '\ntry:\n' + '\n'.join('    ' + line for line in body.splitlines()) + '\n' + cleanup
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    assert result.returncode == 0, result.stderr.decode()[-16000:]


def test_native_tree_closed_is_not_settled_until_complete_effect_guard_exit(tmp_path):
    run_case(tmp_path, r'''
folder,ref,p=native()
assert rows()[0]['os_state']=='CLOSED'
assert rows()[0]['action_state']=='PENDING'
with authority.execution_transaction(lake, folder) as tx:
    digest=tx.prepare(ref, {'operation':'owned CPU test'})
    assert finish_attempt(tx.conn, ref, {'effect_receipt_sha256':digest})=='committed'
    assert rows()[0]['action_state']=='PENDING'
assert rows()[0]['action_state']=='SETTLED'
assert rows()[0]['closure']==p.closure_receipt()
assert rows()[0]['terminal_sha256']
''')


def test_effect_guard_exit_failure_never_settles_committed_terminal(tmp_path):
    run_case(tmp_path, r'''
folder,ref,p=native()
original=authority.attempt_effect_lock
@contextlib.contextmanager
def failed_exit(*args, **kwargs):
    with original(*args, **kwargs) as lease:
        yield lease
    raise RuntimeError('injected after real effect guard exit')
authority.attempt_effect_lock=failed_exit
try:
    with authority.execution_transaction(lake, folder) as tx:
        digest=tx.prepare(ref, {'operation':'owned CPU test'})
        finish_attempt(tx.conn, ref, {'effect_receipt_sha256':digest})
except RuntimeError: pass
else: raise AssertionError('guard fault swallowed')
assert rows()[0]['action_state']!='SETTLED'
assert members._OWNERS
''')


def test_ready_quiesce_blocks_go_and_settles_interrupted_after_actual_closure(tmp_path):
    run_case(tmp_path, r'''
p=prepare("from pathlib import Path; Path('executed').write_text('yes')")
assert rows()[0]['os_state']=='READY'
ctx.quiesce('request-one')
assert p.start() is False
assert p._started is False
p.wait(timeout=5)
assert not (scope/'executed').exists()
assert rows()[0]['action_state']=='PENDING'
members.settle_process(p, outcome='interrupted')
assert rows()[0]['action_state']=='SETTLED'
assert rows()[0]['closure']['stop_requested'] is True
refused(lambda: prepare(ident=identity('another')))
''')


def test_poll_observes_quiesce_and_sends_only_one_stop(tmp_path):
    run_case(tmp_path, r'''
p=prepare('import time; time.sleep(30)'); p.start()
ctx.quiesce('request-one')
p.wait(timeout=5)
assert p._stop_sent is True
assert p.closure_receipt()['stop_requested'] is True
members.settle_process(p, outcome='interrupted')
assert rows()[0]['action_state']=='SETTLED'
''')


def test_unknown_ready_write_keeps_private_member_and_cannot_settle(tmp_path):
    run_case(tmp_path, r'''
original=members._write
def fault(member, **kwargs):
    if kwargs.get('os_state')=='READY': raise OSError('injected READY persistence loss')
    return original(member, **kwargs)
members._write=fault
try: prepare()
except primitive.SupervisionUncertain as error:
    p=error.process; handles.append(p)
else: raise AssertionError('READY uncertainty swallowed')
assert len(members._OWNERS)==1
assert any(pair[0] is p for pair in members._HANDLES.values())
assert rows()[0]['action_state']!='SETTLED'
refused(lambda: members.settle_process(p, outcome='completed'))
''')


def test_member_intent_sql_refusal_precedes_popen(tmp_path):
    run_case(tmp_path, r'''
lake.conn.execute("CREATE TABLE controller_members (broken TEXT)"); lake.conn.commit()
calls=[]
primitive.subprocess.Popen=lambda *a,**k: calls.append(True)
refused(lambda: prepare())
assert calls==[]
assert members._OWNERS
''')


def test_same_connection_native_create_rollback_does_not_disappear(tmp_path):
    run_case(tmp_path, r'''
folder=scope/'idea-one'; folder.mkdir()
try:
    with authority.execution_transaction(lake, folder) as tx:
        create_attempt(tx.conn, 'idea-one', 'training', 'attempt-one', {})
        raise RuntimeError('injected callback refusal')
except RuntimeError: pass
assert members._OWNERS
assert all(m.payload['action_state']!='SETTLED' for m in members._OWNERS.values())
''')


def test_role_intent_is_enrolled_before_prepare_and_no_os_release_settles(tmp_path):
    run_case(tmp_path, r'''
from orze.core.fs import _fs_lock
from orze.engine.role_supervision import begin_role_launch
lock=base/'role-lock'; assert _fs_lock(lock)
owner=begin_role_launch({'role_name':'engineer','attempt_id':'role-one','scope':str(scope),
    'lock_dir':str(lock),'nonce_sha256':'1'*64,'command_sha256':'2'*64,
    'trigger_delivery':None,'trigger_delivery_db':None})
assert rows()[0]['os_state']=='INTENT'
assert rows()[0]['action_state']=='PENDING'
owner.never_executed_proof(); assert owner.release()
assert rows()[0]['os_state']=='NO_EXECUTION'
assert rows()[0]['action_state']=='SETTLED'
''')


def test_bounded_executor_settles_only_after_output_and_close(tmp_path):
    run_case(tmp_path, r'''
from orze.engine.bounded_executor import run_bounded_executor
result=run_bounded_executor([sys.executable,'-c',"print('ok')"], timeout=3,
    env=dict(os.environ),cwd=scope)
assert result.stdout=='ok\n'
assert rows()[0]['os_state']=='CLOSED'
assert rows()[0]['action_state']=='SETTLED'
''')


def test_foreign_scope_is_not_admitted_as_local_controller_work(tmp_path):
    run_case(tmp_path, r'''
foreign=base/'foreign'; foreign.mkdir()
bad=identity(); bad['scope']=str(foreign)
refused(lambda: prepare(ident=bad))
assert not handles
''')


def test_commit_that_rolls_back_cannot_return_ready_or_settled_member(tmp_path):
    run_case(tmp_path, r'''
p=prepare(); p.start(); p.wait(timeout=5)
original=members.sqlite3.connect
class RollbackCommit(sqlite3.Connection):
    def commit(self): self.rollback()
def connect(*args,**kwargs):
    return original(*args, factory=RollbackCommit, **kwargs)
members.sqlite3.connect=connect
refused(lambda: members.settle_process(p, outcome='completed'))
assert rows()[0]['action_state']=='PENDING'
''')


def test_member_limit_is_explicit_and_does_not_drop_pending_owners(tmp_path):
    run_case(tmp_path, r'''
members.MAX_MEMBERS=1
p=prepare(); p.stop(timeout=3)
members.settle_process(p, outcome='interrupted')
refused(lambda: prepare(ident=identity('another')))
assert len(rows())==1
assert len(members._OWNERS)==1
''')
