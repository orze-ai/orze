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
