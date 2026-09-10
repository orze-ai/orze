"""Frozen draft boundary: validate actual process binding before DB outcome."""
import pytest

from test_trigger_role_completion import project, _state
from orze.engine import process, roles


@pytest.mark.parametrize("changed", ["nonce", "role"])
def test_complete_but_mismatched_process_binding_cannot_terminalize_delivery(project, changed):
    p = project
    before = p.receipt.read_bytes()
    if changed == "nonce":
        p.rp.process_nonce = "e" * 64
    else:
        p.rp.role_name = "different-worker"
    assert not process.persist_role_process_receipt(p.rp)
    result = roles.check_active_roles(p.active, ideas_file=str(p.root / "ideas.md"))
    assert result == [("worker", roles.OUTCOME_ERROR)]
    assert _state(p) == "STARTED", "Rejected process binding cannot prove this launch terminal"
    assert p.receipt.read_bytes() == before

