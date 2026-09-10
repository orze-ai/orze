"""Native public completion cannot accept self-consistent, wrong-context receipts."""
import json

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import accounting, evaluator
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
from test_stale_evaluation_completion import project, _prepare, _launch, _exit, _lifecycle


@pytest.mark.parametrize("field,wrong", [
    ("return_code", False),
    ("process_pid", 1337),
    ("reason_code", "another_valid_reason"),
    ("started_at_epoch", 0.0),
])
def test_native_success_rejects_planted_receipt_with_wrong_observed_context(project, field, wrong):
    _assert_wrong_terminal_rejected(project, 0, field, wrong)


def test_native_failure_rejects_planted_zero_exit_terminal(project):
    _assert_wrong_terminal_rejected(project, 1, "return_code", 0)


def _assert_wrong_terminal_rejected(p, ret, field, wrong):
    folder = _prepare(p, "idea-compute-context")
    ep = _launch(p, folder.name)
    _exit(p, ep, ret)
    outcome = "completed" if ret == 0 else "failed"
    reason = "evaluation_validated" if ret == 0 else "evaluation_failed_validation_or_process"
    accounting.record_compute_terminal(ep, folder, outcome, reason, phase="evaluation", return_code=ret)
    path = folder / "_compute_receipts" / ep.attempt_id / "terminal.json"
    payload = json.loads(path.read_text())
    payload[field] = wrong
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    original_receipt = path.read_bytes()
    before_lifecycle = _lifecycle(p.lake)
    active = {0: ep}
    with pytest.raises(AttemptEffectInDoubt):
        evaluator.check_active_evals(active, p.results, p.cfg, lake=p.lake)
    assert _lifecycle(p.lake) == before_lifecycle
    assert current_attempt(p.lake.conn, ep.idea_id, "evaluation")["state"] == "RUNNING"
    assert active.get(0) is ep
    assert path.read_bytes() == original_receipt
    assert (folder / "_attempt_effect.lock").is_dir()
    assert not (folder / "_execution_effects" / ep.attempt_id / "committed.json").exists()


@pytest.mark.parametrize("ret", [0, 1], ids=["success", "failure"])
def test_exact_existing_terminal_context_remains_compatible(project, ret):
    p = project
    folder = _prepare(p, "idea-compute-control")
    ep = _launch(p, folder.name)
    _exit(p, ep, ret)
    accounting.record_compute_terminal(ep, folder, "completed" if ret == 0 else "failed",
        "evaluation_validated" if ret == 0 else "evaluation_failed_validation_or_process",
        phase="evaluation", return_code=ret)
    path = folder / "_compute_receipts" / ep.attempt_id / "terminal.json"
    original = path.read_bytes()
    assert evaluator.check_active_evals({0: ep}, p.results, p.cfg, lake=p.lake) == [(ep.idea_id, 0)]
    assert path.read_bytes() == original
