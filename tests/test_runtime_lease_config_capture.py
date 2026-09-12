"""Direct native callers cannot change invocation lease policy at callbacks."""
import pytest

from test_native_cpu_action import context
from orze.core.execution_attempts import current_attempt
from orze.engine import native_cpu_action as native


@pytest.mark.parametrize("boundary", ["admission", "prepare"])
def test_direct_native_captures_lease_policy_without_rechecking_stop_admission(context, monkeypatch, boundary):
    lake, results, _, cfg, create, processes = context
    cfg["cpu_runtime_lease"] = {"version": 1, "ttl_seconds": 1}
    action, permit = create("from pathlib import Path; Path('executed').write_text('GO')")
    admission = lambda: None
    if boundary == "admission":
        def admission():
            del cfg["cpu_runtime_lease"]
    else:
        original = native.prepare_supervised

        def changed(*args, **kwargs):
            process = original(*args, **kwargs)
            del cfg["cpu_runtime_lease"]
            return process

        monkeypatch.setattr(native, "prepare_supervised", changed)
    with pytest.raises(native.CPUActionHOLD):
        native.launch("idea-cpu", results, cfg, lake=lake, action=action,
                      permit=permit, admission=admission)
    row = current_attempt(lake.conn, "idea-cpu", "action")
    if boundary == "admission":
        assert row is None and processes == []
    else:
        assert row["state"] == "LAUNCHING" and row["terminal"] is None
        assert len(processes) == 1 and processes[0]._started is False
        assert processes[0].closure_receipt()["stop_requested"] is True
    assert not list(results.rglob("executed"))
