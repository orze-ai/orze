"""New C2a binding/proof mechanisms, not old missing-API regressions.

The explicit SupervisedProcess subclass is a protocol boundary double.
No OS process, artifact, scientific observation or native end-to-end behavior
is inferred from these unit-level validations; the separate real-CPU suite
exercises actual launch and publication.
"""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from orze.core.execution_attempts import AttemptRef
from orze.engine.attempt_effect_lock import AttemptEffectBusy
from orze.engine.evaluation_supervision import (
    PROTOCOL, bound_binding, require_closed, failure_override,
)
from orze.engine.supervised_process import SupervisedProcess


class ProtocolDouble(SupervisedProcess):
    def __init__(self, binding, closure):
        self._test_binding = binding
        self._test_closure = closure
        self._test_returncode = 0

    @property
    def pid(self):
        return self._test_binding["worker"]["pid"]

    @property
    def supervisor_pid(self):
        return self._test_binding["supervisor"]["pid"]

    @property
    def binding(self):
        return deepcopy(self._test_binding)

    def poll(self):
        return self._test_returncode

    def closure_receipt(self):
        return deepcopy(self._test_closure)


def case(tmp_path):
    ref = AttemptRef(task_id="idea-proof", phase="evaluation", attempt_id="eval-proof", generation=1)
    folder = tmp_path / ref.task_id
    binding = {"schema": 1, "protocol": PROTOCOL,
        "identity": {"attempt_ref": asdict(ref), "scope": str(folder)},
        "nonce_sha256": "a" * 64, "command_sha256": "b" * 64,
        "worker": {"pid": 10001, "start_ticks": 11},
        "supervisor": {"pid": 10002, "start_ticks": 10}}
    closure = {"schema": 1, "event": "TREE_CLOSED", "binding": deepcopy(binding),
        "worker_returncode": 0, "stop_requested": False, "forced_cleanup": False,
        "reaped_children": 1, "wait_proof": "ECHILD_WALL"}
    process = ProtocolDouble(binding, closure)
    ep = SimpleNamespace(idea_id=ref.task_id, attempt_id=ref.attempt_id,
                         attempt_ref=ref, process=process)
    row = {"state": "RUNNING", "binding": {
        "process_supervision_protocol": PROTOCOL, "supervision": deepcopy(binding)}}
    return ep, row, folder


@pytest.mark.parametrize("change", [
    "historical_unbound", "other_scope", "other_generation", "changed_nonce",
    "boolean_exit", "wrong_exit", "missing_closure", "boolean_reaped", "truthy_stop",
    "wrong_wait_proof",
])
def test_publisher_binding_requires_exact_current_protocol_and_closed_facts(tmp_path, change):
    ep, row, folder = case(tmp_path)
    ret = 0
    if change == "historical_unbound":
        row["binding"].pop("process_supervision_protocol")
    elif change == "other_scope":
        folder = tmp_path / "other-scope" / ep.idea_id
    elif change == "other_generation":
        ep.attempt_ref = AttemptRef(ep.idea_id, "evaluation", ep.attempt_id, 2)
    elif change == "changed_nonce":
        row["binding"]["supervision"]["nonce_sha256"] = "c" * 64
    elif change == "boolean_exit":
        ret = False
    elif change == "wrong_exit":
        ret = 7
    elif change == "missing_closure":
        ep.process._test_closure = None
    elif change == "boolean_reaped":
        ep.process._test_closure["reaped_children"] = True
    elif change == "truthy_stop":
        ep.process._test_closure["stop_requested"] = 1
    else:
        ep.process._test_closure["wait_proof"] = "proc_scan_empty"
    with pytest.raises(AttemptEffectBusy):
        require_closed(ep, row, folder, ret)


def test_closed_zero_result_remains_zero_and_returned_binding_is_detached(tmp_path):
    ep, row, folder = case(tmp_path)
    assert require_closed(ep, row, folder, 0)["worker_returncode"] == 0
    result = bound_binding(ep, row, folder)
    result["worker"]["pid"] = 99999
    assert ep.process.pid == 10001
    assert failure_override(require_closed(ep, row, folder, 0), None) is None


@pytest.mark.parametrize("confirmed_cleanup", [False, True])
def test_failed_initialization_requires_closed_explicit_stop_before_first_binding(tmp_path, confirmed_cleanup):
    ep, row, folder = case(tmp_path)
    row["state"] = "LAUNCHING"
    row["binding"].pop("supervision")
    ep.process._test_closure["stop_requested"] = confirmed_cleanup
    if confirmed_cleanup:
        closure = require_closed(ep, row, folder, 0, allow_launch_cleanup=True)
        assert failure_override(closure, None)[0] == "failed"
        assert failure_override(closure, ("completed", "bad", "bad"))[0] == "failed"
    else:
        with pytest.raises(AttemptEffectBusy):
            require_closed(ep, row, folder, 0, allow_launch_cleanup=True)
