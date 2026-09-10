"""C2d1 new phase-proof mechanisms, never missing-API or OS-proof reds.

Explicit SupervisedProcess protocol doubles verify the thin adapter contract.
Separate real CPU public tests exercise actual script processes and descendants.
"""
from copy import deepcopy
from dataclasses import asdict

import pytest

from orze.core.execution_attempts import AttemptRef
from orze.engine.attempt_effect_lock import AttemptEffectBusy
from orze.engine import post_script_supervision as proof
from test_evaluation_supervision_binding import case as evaluation_case


def case(tmp_path):
    handle, row, folder = evaluation_case(tmp_path)
    old = handle.attempt_ref
    handle.attempt_ref = AttemptRef(old.task_id, "post_script", old.attempt_id, old.generation)
    handle.process._test_binding["identity"]["attempt_ref"] = asdict(handle.attempt_ref)
    handle.process._test_closure["binding"] = deepcopy(handle.process._test_binding)
    row["binding"]["supervision"] = deepcopy(handle.process._test_binding)
    return handle, row, folder


@pytest.mark.parametrize("change", [
    "historical_unbound", "wrong_scope", "wrong_phase", "changed_worker",
    "missing_closure", "boolean_exit",
])
def test_post_script_requires_its_own_exact_closed_protocol(tmp_path, change):
    handle, row, folder = case(tmp_path)
    ret = 0
    if change == "historical_unbound":
        row["binding"].pop("process_supervision_protocol")
    elif change == "wrong_scope":
        folder = tmp_path / "other" / handle.idea_id
    elif change == "wrong_phase":
        old = handle.attempt_ref
        handle.attempt_ref = AttemptRef(old.task_id, "evaluation", old.attempt_id, old.generation)
    elif change == "changed_worker":
        row["binding"]["supervision"]["worker"]["start_ticks"] += 1
    elif change == "missing_closure":
        handle.process._test_closure = None
    else:
        ret = False
    with pytest.raises(AttemptEffectBusy):
        proof.require_closed(handle, row, folder, ret)


def test_post_script_zero_exit_and_binding_are_detached(tmp_path):
    handle, row, folder = case(tmp_path)
    closure = proof.require_closed(handle, row, folder, 0)
    assert closure["worker_returncode"] == 0
    assert closure["binding"]["identity"]["attempt_ref"]["phase"] == "post_script"
    assert proof.failure_override(closure, None) is None
    returned = proof.bound_binding(handle, row, folder)
    returned["worker"]["pid"] += 1
    assert proof.bound_binding(handle, row, folder) == handle.process.binding


def test_post_script_stop_zero_is_closed_but_not_success(tmp_path):
    handle, row, folder = case(tmp_path)
    handle.process._test_closure["stop_requested"] = True
    closure = proof.require_closed(handle, row, folder, 0)
    assert closure["worker_returncode"] == 0
    override = proof.failure_override(closure, ("completed", "old", "old"))
    assert override[0] == "failed"
    assert override[1] == "post_script_process_tree_stopped"
