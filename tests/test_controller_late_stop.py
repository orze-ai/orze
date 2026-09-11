"""A delayed poll must receive actual closure before sending obsolete STOP."""
from test_controller_members import run_case


def test_quiesce_after_owned_supervisor_exit_accepts_queued_tree_closure(tmp_path):
    run_case(tmp_path, r'''
p=prepare(); p.start()
assert p._supervisor.wait(timeout=5)==0
assert p.returncode is None and p._closed is None
ctx.quiesce('late-stop')
assert p.poll()==0
assert p._stop_sent is False
assert p.closure_receipt()['stop_requested'] is False
members.settle_process(p,outcome='completed')
assert members.prove_drained(ctx)['member_count']==1
''')


def test_real_tree_closes_between_initial_receive_and_actual_stop_send(tmp_path):
    run_case(tmp_path, r'''
from orze.engine import supervisor_worker
p=prepare("from pathlib import Path\nimport time\nwhile not Path('release').exists(): time.sleep(.005)")
p.start()
original=supervisor_worker.send_frame
sent=[]
def close_before_send(channel,message):
    if message.get('command')=='STOP':
        sent.append('STOP')
        (scope/'release').write_text('worker may finish')
        assert p._supervisor.wait(timeout=5)==0
        assert p._closed is None
    return original(channel,message)
supervisor_worker.send_frame=close_before_send
ctx.quiesce('late-stop')
assert p.poll()==0
assert sent==['STOP'] and p._stop_sent is True
assert p.closure_receipt()['stop_requested'] is False
members.settle_process(p,outcome='completed')
assert members.prove_drained(ctx)['member_count']==1
''')
