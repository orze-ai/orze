"""New V2 registration mechanisms, not public handoff or missing-API reds.

Real isolated exec/SQLite/namespace ownership uses the existing pidfd-cleaned
registration fixture. No reset of production globals, host census or GPU runs.
Grant authentication and actual successor orchestration are tested separately.
"""
import pytest

from test_controller_control import _run


def test_fresh_v2_has_exact_head_generation_owner_and_atomic_session_callback(tmp_path):
    _run(tmp_path, r'''
calls=[]
def session(conn,ctx):
    assert conn.in_transaction
    assert control.current_instance(conn,scope)[0]==ctx.controller_id
    conn.execute('CREATE TABLE session_probe (controller_id TEXT PRIMARY KEY, identity TEXT)')
    conn.execute('INSERT INTO session_probe VALUES (?,?)',(ctx.controller_id,json.dumps(ctx.identity)))
    calls.append(ctx.controller_id)
ctx=control.register_controller(lake,scope,protocol=2,session_registrar=session)
assert ctx.protocol==2 and ctx.generation==0 and ctx.predecessor is None
assert control.registration_version(lake.conn)==2
assert control.current_instance(lake.conn,scope)==tuple(row(ctx))
assert calls==[ctx.controller_id]
assert json.loads(lake.conn.execute('SELECT identity FROM session_probe').fetchone()[0])==ctx.identity
assert tuple(lake.conn.execute('SELECT current_id,generation,pending_grant FROM controller_scope_heads').fetchone())==(ctx.controller_id,0,None)
anchor=scope/'_controller_registration.lock'
owner=anchor/('instance-0-'+ctx.controller_id+'.lock')
assert ctx.identity['owner_directory']==str(owner)
assert ctx.identity['anchor_directory']==str(anchor)
assert ctx.identity['owner_metadata_sha256']==ctx._lease.metadata_sha256
assert ctx.identity['anchor_metadata_sha256']==ctx._anchor_lease.metadata_sha256
assert (anchor/'lock.json').is_file() and (owner/'lock.json').is_file()
before=(anchor/'lock.json').read_bytes(),(owner/'lock.json').read_bytes()
assert control.register_controller(lake,scope,protocol=2) is ctx
refuses(lambda:control.register_controller(lake,scope))
ctx.quiesce('stop-v2'); assert ctx.poll_control()=='QUIESCING'
assert before==((anchor/'lock.json').read_bytes(),(owner/'lock.json').read_bytes())
assert len(control.current_instance(lake.conn,scope))==6
lake.close()
''')


def test_v1_remains_exact_stop_only_without_online_schema_upgrade(tmp_path):
    _run(tmp_path, r'''
ctx=control.register_controller(lake,scope)
assert control.registration_version(lake.conn)==1 and ctx.protocol==1
assert control.current_instance(lake.conn,scope)==tuple(row(ctx))
before=lake.conn.execute("SELECT sql FROM sqlite_master WHERE name='controller_instances'").fetchone()[0]
owner=(scope/'_controller_registration.lock'/'lock.json').read_bytes()
refuses(lambda:control.register_controller(lake,scope,admission=object()))
ctx.check_admission()
refuses(lambda:control.register_controller(lake,scope,protocol=2))
assert lake.conn.execute("SELECT sql FROM sqlite_master WHERE name='controller_instances'").fetchone()[0]==before
assert lake.conn.execute("SELECT 1 FROM sqlite_master WHERE name='controller_scope_heads'").fetchone() is None
assert (scope/'_controller_registration.lock'/'lock.json').read_bytes()==owner
assert list((scope/'_controller_registration.lock').iterdir())==[scope/'_controller_registration.lock'/'lock.json']
lake.close()
''')


def test_fake_successor_admission_cannot_create_namespace_or_registration(tmp_path):
    _run(tmp_path, r'''
for fake in (object(),True,{'target_id':'a'*48,'generation':1}):
    refuses(lambda:control.register_controller(lake,scope,protocol=2,admission=fake))
    assert control.current_controller() is None
    assert list(scope.iterdir())==[]
    assert lake.conn.execute("SELECT 1 FROM sqlite_master WHERE name='controller_instances'").fetchone() is None
for invalid in (True,0,3,'2'):
    refuses(lambda:control.register_controller(lake,scope,protocol=invalid))
assert control.current_controller() is None
lake.close()
''')


@pytest.mark.parametrize('fault', ['missing_head', 'head_trigger', 'foreign_unique'])
def test_v2_schema_is_strict_and_never_falls_back_to_history(tmp_path, fault):
    _run(tmp_path, 'fault='+repr(fault)+r'''
ctx=control.register_controller(lake,scope,protocol=2)
if fault=='missing_head':
    lake.conn.execute('DROP TABLE controller_scope_heads')
elif fault=='head_trigger':
    lake.conn.execute('CREATE TRIGGER bad_head AFTER UPDATE ON controller_scope_heads BEGIN SELECT 1; END')
else:
    lake.conn.execute('CREATE UNIQUE INDEX bad_scope ON controller_instances(scope)')
lake.conn.commit()
refuses(lambda:control.registration_version(lake.conn))
refuses(lambda:control.current_instance(lake.conn,scope))
refuses(ctx.check_admission)
assert control.current_controller() is ctx
assert (ctx._lease.lock_dir/'lock.json').is_file()
lake.close()
''')


@pytest.mark.parametrize('fault', ['pending_grant', 'generation', 'current_id'])
def test_changed_head_revokes_old_context_without_reclaiming_its_owner(tmp_path, fault):
    _run(tmp_path, 'fault='+repr(fault)+r'''
ctx=control.register_controller(lake,scope,protocol=2)
before=(ctx._lease.lock_dir/'lock.json').read_bytes()
if fault=='pending_grant':
    lake.conn.execute("UPDATE controller_scope_heads SET pending_grant='grant-one'")
elif fault=='generation':
    lake.conn.execute('UPDATE controller_scope_heads SET generation=1')
else:
    lake.conn.execute('UPDATE controller_scope_heads SET current_id=?',('f'*48,))
lake.conn.commit()
if fault=='pending_grant':
    # Observer may read the old head while its grant is pending; not admission.
    assert control.current_instance(lake.conn,scope)[0]==ctx.controller_id
else:
    refuses(lambda:control.current_instance(lake.conn,scope))
refuses(ctx.check_admission)
refuses(ctx.poll_control)
assert control.current_controller() is ctx
assert (ctx._lease.lock_dir/'lock.json').read_bytes()==before
lake.close()
''')


@pytest.mark.parametrize('target', ['anchor', 'generation'])
def test_both_anchor_and_generation_owner_metadata_are_required(tmp_path, target):
    _run(tmp_path, 'target='+repr(target)+r'''
ctx=control.register_controller(lake,scope,protocol=2)
owner=ctx._anchor_lease if target=='anchor' else ctx._lease
path=owner.lock_dir/'lock.json'
path.write_bytes(b'{}\n')
refuses(ctx.check_admission)
refuses(lambda:control.register_controller(lake,scope,protocol=2))
assert path.read_bytes()==b'{}\n'
assert ctx._anchor_lease.lock_dir.is_dir() and ctx._lease.lock_dir.is_dir()
lake.close()
''')


def test_session_callback_failure_rolls_back_head_and_instance_but_retains_owner(tmp_path):
    _run(tmp_path, r'''
def failing(conn,ctx):
    assert conn.in_transaction and control.current_instance(conn,scope)[0]==ctx.controller_id
    conn.execute('CREATE TABLE callback_side_effect (id TEXT)')
    raise OSError('explicit callback failure')
refuses(lambda:control.register_controller(lake,scope,protocol=2,session_registrar=failing))
ctx=control.current_controller()
assert ctx is not None
refuses(ctx.check_admission)
for name in ('controller_instances','controller_scope_heads','callback_side_effect'):
    assert lake.conn.execute('SELECT 1 FROM sqlite_master WHERE name=?',(name,)).fetchone() is None
assert (ctx._anchor_lease.lock_dir/'lock.json').is_file()
assert (ctx._lease.lock_dir/'lock.json').is_file()
lake.close()
''')


def test_v2_registration_commit_rollback_cannot_publish_active_context(tmp_path):
    _run(tmp_path, r'''
original=control.sqlite3.connect
class Rollback(sqlite3.Connection):
    def commit(self):
        self.rollback()
def connect(*a,**k):
    k['factory']=Rollback
    return original(*a,**k)
control.sqlite3.connect=connect
try:
    refuses(lambda:control.register_controller(lake,scope,protocol=2))
finally:
    control.sqlite3.connect=original
ctx=control.current_controller()
assert ctx is not None
refuses(ctx.check_admission)
assert lake.conn.execute("SELECT 1 FROM sqlite_master WHERE name='controller_instances'").fetchone() is None
assert (ctx._anchor_lease.lock_dir/'lock.json').is_file()
assert (ctx._lease.lock_dir/'lock.json').is_file()
lake.close()
''')
