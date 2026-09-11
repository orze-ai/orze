"""Known CPU-probe cancellation, separate from the earlier GPU observation.

The first case executes the real product controller and observer. Only its
already-explicit GPU-idle boundary runs a tiny CPU diagnostic instead of any
GPU query. The second uses the explicitly declared session host fixture, not
Orze, to isolate no-intent cancellation. Neither substitutes tree closure.
"""
import json

import test_controller_product_boundaries as product
from test_controller_product_boundaries import product_process
from test_controller_session import _run


CPU_PROBE = r'''
from orze.engine import controller_probe
controller_probe.prepare_supervised=captured_prepare
def cpu_probe_idle(gpus):
    if list(gpus)!=[0]:raise RuntimeError('fixture scope changed')
    result=controller_probe.run_probe([sys.executable,str(root/'worker.py'),
        '--fixture-root',str(root),'--fixture-socket',str(root/'worker.sock'),
        '--fixture-phase','probe','--fixture-mode','sleep'],
        cwd=root,capture_output=True,text=True,timeout=20)
    # A cancelled diagnostic is never a usable GPU observation.
    tell('probe_returned_normally',returncode=result.returncode)
    return idle(gpus)
gpu_lease.assert_gpu_scope_idle=cpu_probe_idle
'''


def test_real_product_known_stopped_cpu_probe_drains_without_hold(monkeypatch, product_process):
    from orze.engine.controller_control import ControllerHOLD
    needle = "try:\n    from orze.core.config import load_project_config"
    assert product.CONTROLLER.count(needle) == 1
    monkeypatch.setattr(product, "CONTROLLER", product.CONTROLLER.replace(needle, CPU_PROBE + "\n" + needle))
    c = product_process("probe_stop")
    assert c.event("constructed")["lake"] == "IdeaLake"
    c.worker()
    result = failure = None
    try:
        result = c.stop(timeout=8)
    except ControllerHOLD as exc:
        failure = exc
    members = [json.loads(row["payload_json"]) for row in c.rows("controller_members")]
    assert len(members) == 1 and members[0]["kind"] == "controller_probe"
    member = members[0]
    assert member["os_state"] == "CLOSED" and member["action_state"] == "SETTLED"
    assert member["outcome"] == "interrupted" and member["hold_reason"] is None
    assert member["closure"]["stop_requested"] is True
    assert member["closure"]["wait_proof"] == "ECHILD_WALL"
    assert failure is None, str(failure)
    assert result["completed"] is True and not product._alive(c.controller_fd)
    assert c.child.wait(timeout=5) == 0
    assert c.rows("controller_sessions")[0]["ack_json"] is not None
    assert not (c.root / "training.go").exists()
    assert not any(event["event"] == "probe_returned_normally" for event in c.events)


def test_pre_admission_quiesce_is_known_cancel_without_member_or_process(tmp_path):
    _run(tmp_path, r'''
from orze.engine import controller_probe
from orze.engine.controller_control import ControllerQuiescing
session.request_local_stop('operator_stop')
calls=[]
def forbidden_prepare(*args,**kwargs):
    calls.append((args,kwargs))
    raise AssertionError('quiescent probe crossed process boundary')
controller_probe.prepare_supervised=forbidden_prepare
caught=None
try:controller_probe.run_probe([sys.executable,'-c','pass'],capture_output=True,timeout=3)
except controller_probe.ControllerProbeHOLD as exc:caught=exc
assert calls==[]
assert not any(m.ctx is session.ctx for m in __import__('orze.engine.controller_members',fromlist=['_OWNERS'])._OWNERS.values())
assert isinstance(caught,ControllerQuiescing),repr(caught)
ack=session.finish()
assert ack['members']['member_count']==0
assert json.loads(row()[1])==ack
''')
