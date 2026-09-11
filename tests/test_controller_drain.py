"""New bounded drain mechanisms; actual isolated SQLite and CPU supervisors.

These do not stand in for the real Orze/session/observer end-to-end tests.
The preceding member fixture provides exact owned pidfd subprocess cleanup.
"""
from test_controller_members import run_case


FINISH = r'''
def finish_native(folder,ref):
    with authority.execution_transaction(lake, folder) as tx:
        digest=tx.prepare(ref, {'operation':'drain mechanism'})
        finish_attempt(tx.conn,ref,{'effect_receipt_sha256':digest})
'''


def test_native_history_generations_are_drained_by_exact_full_reference(tmp_path):
    run_case(tmp_path, FINISH + r'''
folder,ref,p=native(); finish_native(folder,ref)
with authority.execution_transaction(lake,folder) as tx:
    second=create_attempt(tx.conn,ref.task_id,ref.phase,'attempt-two',{})
    tx.watch_attempt(second)
p2=prepare(ident={'attempt_ref':asdict(second),'scope':str(folder)})
with authority.execution_transaction(lake,folder) as tx:
    mark_running(tx.conn,second,{'process_supervision':p2.binding})
    tx.watch_attempt(second)
p2.start(); assert p2.wait(timeout=5)==0
finish_native(folder,second)
refused(lambda:members.prove_drained(ctx))
ctx.quiesce('drain-one')
proof=members.prove_drained(ctx)
assert proof['controller_id']==ctx.controller_id
assert proof['member_count']==2 and len(proof['members_sha256'])==64
assert members.prove_drained(ctx)==proof
''')


def test_closed_unsettled_process_and_open_transaction_cannot_prove_drain(tmp_path):
    run_case(tmp_path, r'''
p=prepare(); p.start(); assert p.wait(timeout=5)==0
ctx.quiesce('drain-one')
refused(lambda:members.prove_drained(ctx))
members.settle_process(p,outcome='completed')
folder=scope/'idea-one'; folder.mkdir()
with authority.execution_transaction(lake,folder):
    refused(lambda:members.prove_drained(ctx))
assert members.prove_drained(ctx)['member_count']==1
''')


def test_exact_owned_report_is_admitted_after_quiesce_but_new_os_is_not(tmp_path):
    run_case(tmp_path, FINISH + r'''
folder,ref,p=native(); finish_native(folder,ref)
ctx.quiesce('drain-one')
with authority.execution_transaction(lake,folder) as tx:
    report=create_attempt(tx.conn,ref.task_id,'launch_failure_report','report-one',
                         {'source_attempt':asdict(ref)})
    mark_running(tx.conn,report)
    digest=tx.prepare(report,{'operation':'source-bound drain report'})
    finish_attempt(tx.conn,report,{'effect_receipt_sha256':digest})
assert members.prove_drained(ctx)['member_count']==2
refused(lambda:prepare(ident=identity('new-os')))
''')


def test_foreign_report_source_after_quiesce_is_not_cleanup_authority(tmp_path):
    run_case(tmp_path, FINISH + r'''
folder,ref,p=native(); finish_native(folder,ref)
ctx.quiesce('drain-one')
def wrong_report():
    with authority.execution_transaction(lake,folder) as tx:
        create_attempt(tx.conn,ref.task_id,'pre_script_failure_report','report-one',
                       {'source_attempt':asdict(ref)})
refused(wrong_report)
assert len(rows())==1
assert members.prove_drained(ctx)['member_count']==1
''')


def test_static_rejection_requires_explicit_captured_preparation_proof(tmp_path):
    run_case(tmp_path, r'''
from orze.engine.native_artifact_preflight import _capture, _sha
folder=scope/'idea-one'; folder.mkdir()
captured=_capture(folder,{'artifact_preflight':{'enabled':True,'script':'absent.py'},
                          '_project_root':str(base)})
binding={'preflight_identity':captured.identity,'command_sha256':_sha(captured.command)}
with authority.execution_transaction(lake,folder) as tx:
    ref=create_attempt(tx.conn,'idea-one','artifact_preflight','preflight-one',binding)
    tx.watch_attempt(ref)
ctx.quiesce('drain-one')
members.record_static_preflight_rejection(ref,binding,captured)
with authority.execution_transaction(lake,folder) as tx:
    digest=tx.prepare(ref,{'operation':'configuration_error'})
    finish_attempt(tx.conn,ref,{'effect_receipt_sha256':digest},not_started=True)
proof=members.prove_drained(ctx)
assert proof['member_count']==1
assert rows()[0]['os_state']=='NO_EXECUTION'
assert rows()[0]['closure'] is None and rows()[0]['outcome']=='not_started'
''')


def test_real_ready_worker_cannot_be_relabelled_static_no_execution(tmp_path):
    run_case(tmp_path, r'''
from orze.engine.native_artifact_preflight import _capture, _sha
folder=scope/'idea-one'; folder.mkdir()
captured=_capture(folder,{'artifact_preflight':{'enabled':True,'script':'absent.py'},
                          '_project_root':str(base)})
binding={'preflight_identity':captured.identity,'command_sha256':_sha(captured.command)}
with authority.execution_transaction(lake,folder) as tx:
    ref=create_attempt(tx.conn,'idea-one','artifact_preflight','preflight-one',binding)
    tx.watch_attempt(ref)
p=primitive.prepare_supervised(captured.command,identity={'attempt_ref':asdict(ref),'scope':str(folder)},
    cwd=scope,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
handles.append(p)
refused(lambda:members.record_static_preflight_rejection(ref,binding,captured))
assert rows()[0]['os_state']=='READY'
ctx.quiesce('drain-one'); assert p.start() is False; p.wait(timeout=5)
refused(lambda:members.prove_drained(ctx))
''')


def test_removed_terminal_effect_is_not_a_drained_native_member(tmp_path):
    run_case(tmp_path, FINISH + r'''
folder,ref,p=native(); finish_native(folder,ref)
ctx.quiesce('drain-one')
assert members.prove_drained(ctx)['member_count']==1
effect=folder/'_execution_effects'/ref.attempt_id/'committed.json'
assert effect.is_file(); effect.unlink()
try: members.prove_drained(ctx)
except Exception: pass
else: raise AssertionError('missing committed effect was accepted')
''')


def test_soft_lifetime_limit_does_not_evict_old_settled_owners(tmp_path):
    run_case(tmp_path, r'''
members.MAX_MEMBERS=2
p=prepare(); p.start(); p.wait(timeout=5)
members.settle_process(p,outcome='completed')
assert members.member_limit_reached(ctx) is True
ctx.quiesce('member-limit')
assert members.prove_drained(ctx)['member_count']==1
assert len(members._OWNERS)==1 and len(rows())==1
''')
