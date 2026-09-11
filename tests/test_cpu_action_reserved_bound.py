"""Independent native/real-budget agreement check, no fake budget permit."""
import sys

import pytest
import yaml

from test_native_cpu_action import context
from orze.core import cpu_action_budget as budget
from orze.engine import native_cpu_action as native
from orze.engine.scheduler import claim


def test_smaller_durable_reservation_cannot_authorize_longer_action(context):
    lake, results, scope, cfg, create, handles = context
    action = {"version": 1, "adapter": "command", "purpose": "bounded action",
              "inputs": {}, "command": [sys.executable, "-c", "pass"],
              "timeout_seconds": 2, "outputs": {}}
    admitted = lake.insert("idea-budget", "bound", yaml.safe_dump({
        "kind": "native_cpu_action", "action": action}), "", status="queued",
        kind="native_cpu_action", if_absent=True)
    assert admitted["status"] == "inserted"
    permit = budget.reserve(lake, scope, "idea-budget", 0.1)
    assert permit is not None
    assert claim("idea-budget", results, None, lake, resource="cpu")
    with pytest.raises(native.CPUActionHOLD):
        native.launch("idea-budget", results, cfg, lake=lake, action=action,
                      permit=permit, admission=lambda: None)
    assert handles == []
