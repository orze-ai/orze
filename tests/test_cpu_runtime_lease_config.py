"""Runtime lease invocation policy is independent of action/budget identity."""
import copy

import pytest

from orze.core.cpu_execution import (
    CPUExecutionError, cpu_execution, execution_fingerprint,
    runtime_lease_config, runtime_lease_seconds,
)


def config():
    return {"execution": {"version": 1, "resource": "cpu", "slots": 1,
                          "wall_budget_seconds": 20}}


def test_default_and_shorter_lease_preserve_original_budget_and_action_envelope():
    cfg = config()
    original = copy.deepcopy(cfg)
    default = execution_fingerprint(cfg)
    assert runtime_lease_seconds(cfg, 2) == 2
    assert "cpu_runtime_lease" not in cfg
    cfg["cpu_runtime_lease"] = {"version": 1, "ttl_seconds": 1}
    assert runtime_lease_seconds(cfg, 2) == 1
    assert cpu_execution(cfg) == original["execution"]
    assert execution_fingerprint(cfg) != default
    assert cfg["execution"] == original["execution"]
    with pytest.raises(CPUExecutionError, match="exceeds action"):
        runtime_lease_seconds(cfg, .5)


@pytest.mark.parametrize("value", [None, False, 0, {},
    {"version": True, "ttl_seconds": 1}, {"version": 1.0, "ttl_seconds": 1},
    {"version": 1, "ttl_seconds": True}, {"version": 1, "ttl_seconds": 0},
    {"version": 1, "ttl_seconds": 1e-12}, {"version": 1, "ttl_seconds": float("inf")},
    {"version": 1, "ttl_seconds": float("nan")},
    {"version": 1, "ttl_seconds": 1, "disable": False}])
def test_explicit_malformed_lease_cannot_disable_or_weaken_default(value):
    cfg = config()
    cfg["cpu_runtime_lease"] = value
    with pytest.raises(CPUExecutionError, match="cpu_runtime_lease"):
        cpu_execution(cfg)


def test_normalized_policy_is_fingerprinted_and_cannot_hot_reload():
    cfg = config()
    cfg["cpu_runtime_lease"] = {"version": 1, "ttl_seconds": 1}
    first = execution_fingerprint(cfg)
    cfg["cpu_runtime_lease"]["ttl_seconds"] = 1.0
    assert execution_fingerprint(cfg) == first
    cfg["_cpu_execution_fingerprint"] = first
    assert cpu_execution(cfg) == cfg["execution"]
    del cfg["cpu_runtime_lease"]
    with pytest.raises(CPUExecutionError, match="loaded CPU configuration changed"):
        cpu_execution(cfg)


def test_legacy_profile_cannot_silently_ignore_explicit_runtime_lease():
    assert cpu_execution({}) is None
    assert runtime_lease_config({}) == {"version": 1, "ttl_seconds": None}
    for value in (None, {"version": 1, "ttl_seconds": 1}):
        with pytest.raises(CPUExecutionError, match="requires explicit CPU"):
            cpu_execution({"cpu_runtime_lease": value})
