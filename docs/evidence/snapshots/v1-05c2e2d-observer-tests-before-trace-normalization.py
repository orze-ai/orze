"""New borrowed-connection proof mechanisms, not historical API-absence reds.

Uses the existing isolated real Session/SQLite/lease fixture with an explicit
orchestration host double. These are not whole handoff process proofs.
"""
import pytest

from test_controller_session import _run


SETUP = r'''
from orze.engine.controller_session import _Observer
ack=session.finish()
observer=_Observer(cfg)
conn=sqlite3.connect(cfg['idea_lake_db'])
def no_connection(*args,**kwargs):
    raise AssertionError('borrowed proof opened another connection')
observer.connection=no_connection
'''


def test_given_writer_is_borrowed_without_new_connection_or_transaction_changes(tmp_path):
    _run(tmp_path, SETUP + r'''
conn.execute('BEGIN IMMEDIATE')
calls=[]
conn.set_trace_callback(calls.append)
before=conn.total_changes
try:
    assert observer.verify_drain(ack,conn=conn) is None
    assert conn.in_transaction and conn.total_changes==before
    assert conn.execute('SELECT 1').fetchone()==(1,)
    assert calls and all(sql.lstrip().upper().startswith(('SELECT','PRAGMA')) for sql in calls)
finally:
    conn.rollback(); conn.close(); observer.close()
''')


@pytest.mark.parametrize('fault', ['memory', 'other_file', 'scope_replaced', 'member_trigger', 'uncommitted_member'])
def test_given_connection_cannot_replace_bound_route_schema_or_current_member_view(tmp_path, fault):
    _run(tmp_path, SETUP + '\nfault=' + repr(fault) + r'''
if fault in {'memory','other_file'}:
    conn.close()
    conn=sqlite3.connect(':memory:' if fault=='memory' else root/'unrelated.db')
conn.execute('BEGIN IMMEDIATE')
if fault=='scope_replaced':
    scope.rename(root/'retained_original_scope'); scope.mkdir()
elif fault=='member_trigger':
    conn.execute('CREATE TRIGGER controlled_member_trigger AFTER INSERT ON controller_members BEGIN SELECT 1; END')
elif fault=='uncommitted_member':
    conn.execute('INSERT INTO controller_members VALUES (?,?,?)',('injected',session.ctx.controller_id,'{}'))
before=conn.total_changes
try:
    try: observer.verify_drain(ack,conn=conn)
    except ControllerHOLD: pass
    else: raise AssertionError('unbound or changed transaction accepted')
    assert conn.in_transaction and conn.total_changes==before
    assert conn.execute('SELECT 1').fetchone()==(1,)
finally:
    conn.rollback(); conn.close(); observer.close()
''')
