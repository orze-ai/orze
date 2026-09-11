"""Accepted legacy receipt metadata must remain consumable by completion.

Real temporary SQLite, receipt publication and Core harvesting reuse the old
delivery fixture. Linux process identity and closure are explicitly simulated:
4096 positive signed-64-bit identity records are not 4096 real processes, nor
evidence of realistic host uptime. No caller-added payload/origin fields exist.
"""
import json

import pytest

from orze.engine import process, roles
from test_trigger_role_completion import project, _state


@pytest.mark.parametrize("descendant_count", [0, 4096])
def test_writer_accepted_legacy_receipt_does_not_strand_terminal_delivery(project, descendant_count):
    p = project
    # Stay within the writer's count cap and the reader's positive-identity
    # contract; use bounded 64-bit values rather than unbounded Python ints.
    p.rp._tracked_descendants = [
        {"pid": 4_000_000 + index, "pgid": 4_000_000,
         "start_ticks": (1 << 63) - 1 - index}
        for index in range(descendant_count)
    ]
    assert len(p.rp._tracked_descendants) <= process._MAX_ROLE_DESCENDANTS
    assert all(process._valid_identity(value) for value in p.rp._tracked_descendants)
    assert process.persist_role_process_receipt(p.rp) is True
    raw = p.receipt.read_bytes()
    receipt = json.loads(raw)
    assert len(receipt["descendants"]) == descendant_count
    assert receipt["trigger_delivery"]["attempt_id"] == p.launch["attempt_id"]
    assert b"PRIVATE REQUEST" not in raw
    if descendant_count:
        assert 256 * 1024 < len(raw) < process._MAX_ROLE_RECEIPT_BYTES
    else:
        assert len(raw) < 256 * 1024

    result = roles.check_active_roles(p.active, ideas_file=str(p.root / "ideas.md"))
    assert _state(p) == "TERMINAL"
    assert not p.receipt.exists(), "Writer-accepted receipt stranded after the real delivery became TERMINAL"
    assert not p.rp.lock_dir.exists()
    assert result == [("worker", roles.OUTCOME_OK)]
    assert not p.active


def test_regular_width_writer_receipt_can_reach_public_delivery_settlement(project):
    p = project
    # Ordinary-width simulated Linux metadata, not 1200 actual processes.
    p.rp._tracked_descendants = [
        {"pid": 4_000_000 + index, "pgid": 4_000_000,
         "start_ticks": 1_234_567_890 + index}
        for index in range(1200)
    ]
    assert len(p.rp._tracked_descendants) <= process._MAX_ROLE_DESCENDANTS
    assert all(process._valid_identity(value) for value in p.rp._tracked_descendants)
    assert process.persist_role_process_receipt(p.rp) is True
    raw = p.receipt.read_bytes()
    assert 64 * 1024 < len(raw) < 256 * 1024
    assert _state(p) == "STARTED"

    result = roles.check_active_roles(p.active, ideas_file=str(p.root / "ideas.md"))
    assert _state(p) == "TERMINAL", f"Writer-accepted {len(raw)}-byte receipt could not reach public settlement"
    assert not p.receipt.exists()
    assert not p.rp.lock_dir.exists()
    assert result == [("worker", roles.OUTCOME_OK)]
    assert not p.active
