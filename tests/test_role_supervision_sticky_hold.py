"""Draft regression: actual GO followed by uncertain STARTED publication."""
import json

import pytest

from orze.engine import process, role_supervision as supervision
from test_role_supervision import project, _prepare


def test_go_receipt_write_failure_cannot_later_grant_terminal_proof(project, monkeypatch):
    p = project
    owner = p.begin()
    child = _prepare(p, owner)
    real_write = process._atomic_private_json
    failed_stages = []

    def fail_started(path, payload):
        if payload.get("stage") == "STARTED":
            failed_stages.append(payload["stage"])
            raise OSError("controlled post-GO STARTED publication failure")
        return real_write(path, payload)

    monkeypatch.setattr(process, "_atomic_private_json", fail_started)
    with pytest.raises(supervision.RoleSupervisionHOLD):
        owner.start()
    assert failed_stages == ["STARTED"]
    assert child.wait(timeout=3) == 0
    assert p.marker.read_text() == "done"
    before = (p.lock / "role-process.json").read_bytes()
    assert json.loads(before)["stage"] == "GO_REQUESTED"

    with pytest.raises(supervision.RoleSupervisionHOLD):
        owner.require_closed(0)

    assert (p.lock / "role-process.json").read_bytes() == before
    assert p.lock.exists()
