"""Profile consumer mechanisms, not a fake process-tree proof.

Real native member/CPU fixtures provide the underlying obligations. Only the
shutdown-publication response is injected, explicitly exercising stale/public
slot handoff behavior. Session end-to-end proof is tested independently.
"""
from test_controller_members import run_case
from test_controller_drain import FINISH


SETUP = r'''
from types import SimpleNamespace
from orze.engine import lifecycle, shutdown_publication
folder,ref,p=native()
tracked=SimpleNamespace(attempt_ref=ref,process=p)
active={0:tracked}; failures=[]
session=SimpleNamespace(fail=failures.append)
def shutdown():
    return lifecycle.graceful_shutdown(scope,{},active,{}, {},0,{},lake,'host','instance',
                                      controller_session=session)
'''


def test_stale_true_is_not_action_settlement_or_resource_release(tmp_path):
    run_case(tmp_path, SETUP + r'''
shutdown_publication.handle_shutdown=lambda *a,**k:True
refused(shutdown)
assert active[0] is tracked and failures
assert rows()[0]['action_state']=='PENDING'
assert lake.conn.execute('SELECT 1').fetchone()[0]==1
assert not (scope/'.orze_shutdown').exists()
''')


def test_replaced_slot_is_retained_after_the_original_action_settles(tmp_path):
    run_case(tmp_path, FINISH + SETUP + r'''
replacement=object()
def settled_then_replaced(*a,**k):
    finish_native(folder,ref)
    active[0]=replacement
    return True
shutdown_publication.handle_shutdown=settled_then_replaced
refused(shutdown)
assert active[0] is replacement and failures
assert rows()[0]['action_state']=='SETTLED'
''')


def test_exact_settled_action_is_removed_but_resources_are_deferred(tmp_path):
    run_case(tmp_path, FINISH + SETUP + r'''
def settled(*a,**k):
    finish_native(folder,ref)
    return True
shutdown_publication.handle_shutdown=settled
pidfile=scope/'.orze.pid.host'; pidfile.write_text(str(os.getpid()))
shutdown()
assert active=={} and failures==[]
assert rows()[0]['action_state']=='SETTLED'
assert lake.conn.execute('SELECT 1').fetchone()[0]==1
assert pidfile.exists() and not (scope/'.orze_shutdown').exists()
''')


def test_legacy_slot_does_not_reach_raw_shutdown_fallback(tmp_path):
    run_case(tmp_path, SETUP + r'''
tracked.attempt_ref=None
called=[]
lifecycle._stop_for_shutdown=lambda *a,**k:called.append(True)
refused(shutdown)
assert called==[] and active[0] is tracked and failures
''')
