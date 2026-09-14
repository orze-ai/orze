"""Explicit, default-off CPU resource selection for the existing Orze loop.

The first CPU profile uses a queue policy and a command adapter. No GPU
profile, provider role, implicit evaluator, or background service is inferred.
Wall budgets reserve execution envelopes, not measured CPU utilization.
Version 2 explicitly permits continued authorization without a cumulative wall
ceiling; individual actions still require finite execution bounds.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


class CPUExecutionError(ValueError):
    pass


def _fail(reason):
    raise CPUExecutionError("execution: " + reason)


def cpu_execution(cfg):
    if type(cfg) is not dict:
        _fail("configuration must be a mapping")
    value = cfg.get("execution")
    if value is None:
        if cfg.get("_cpu_execution_fingerprint") is not None:
            _fail("loaded CPU declaration was erased")
        if cfg.get("action_policy") is not None:
            _fail("action_policy requires explicit CPU execution")
        if cfg.get("action_domain") is not None:
            _fail("action_domain requires explicit CPU execution")
        if "cpu_runtime_lease" in cfg:
            _fail("cpu_runtime_lease requires explicit CPU execution")
        return None
    if (type(value) is not dict or set(value) != {
            "version", "resource", "slots", "wall_budget_seconds"}
            or type(value["version"]) is not int or value["version"] not in (1, 2)
            or value["resource"] != "cpu"
            or type(value["slots"]) is not int or not 1 <= value["slots"] <= 64):
        _fail("requires version 1, resource cpu, slots 1..64 and wall_budget_seconds")
    wall = value["wall_budget_seconds"]
    if value["version"] == 2:
        if wall is not None:
            _fail("version 2 requires wall_budget_seconds to be null")
    elif (type(wall) not in (int, float) or not 0 < wall <= 365 * 86400
            or not math.isfinite(wall)):
        _fail("wall_budget_seconds must be finite and in (0, 31536000]")
    result = dict(value)
    if cfg.get("controller_control") is not None or cfg.get("_managed_idea_id"):
        _fail("CPU actions do not use GPU controller profiles or run-idea")
    sched = cfg.get("gpu_scheduling") or {}
    if type(sched) is not dict or sched.get("allowed_gpus"):
        _fail("CPU actions cannot declare a physical GPU scope")
    if cfg.get("role_presets") or cfg.get("research") or cfg.get("evolution"):
        _fail("CPU action policy does not activate legacy role presets")
    roles = cfg.get("roles") or {}
    if type(roles) is not dict or any(
            type(role) is not dict or role.get("enabled", True) is not False
            for role in roles.values()):
        _fail("provider/legacy roles are not enabled by the CPU queue profile")
    for key in ("eval_script", "pre_script", "post_scripts", "cleanup_script",
                "containers", "remote", "fleet", "artifact_contract", "observation_contract"):
        if cfg.get(key):
            _fail(key + " is not an action-local CPU contract")
    action_policy(cfg)
    action_domain(cfg)
    runtime_lease_config(cfg)
    loaded = cfg.get("_cpu_execution_fingerprint")
    if loaded is not None and loaded != execution_fingerprint(cfg, declaration=result):
        _fail("loaded CPU configuration changed")
    return result


def action_policy(cfg):
    value = cfg.get("action_policy", {"version": 1, "kind": "queue",
                                      "idle": "wait", "wait_seconds": 1})
    fields = {"version", "kind", "idle", "wait_seconds"}
    paged = type(value) is dict and type(value.get("version")) is int and value["version"] == 2
    if paged:
        fields.add("evidence_page_size")
    if (type(value) is not dict or set(value) not in (fields, fields | {"config"})
            or type(value["version"]) is not int or value["version"] not in (1, 2)
            or type(value["kind"]) is not str or value["idle"] not in ("wait", "stop")):
        _fail("action_policy requires version 1, registered kind, idle wait/stop, wait_seconds")
    if paged and (value["kind"] == "queue" or type(value["evidence_page_size"]) is not int
                  or not 1 <= value["evidence_page_size"] <= 32):
        _fail("action_policy version 2 requires a custom policy and evidence_page_size 1..32")
    seconds = value["wait_seconds"]
    if (type(seconds) not in (int, float) or not 0.01 <= seconds <= 3600
            or not math.isfinite(seconds)):
        _fail("action_policy.wait_seconds must be finite in [0.01, 3600]")
    from orze.core.research_interfaces import require_policy_declaration
    try:
        return require_policy_declaration(value)
    except ValueError as exc:
        _fail(str(exc))


def action_domain(cfg):
    from orze.core.research_interfaces import domain_declaration
    try:
        return domain_declaration(cfg)
    except ValueError as exc:
        _fail(str(exc))


def execution_fingerprint(cfg, *, declaration=None):
    """Pin semantic resource/policy and actual invocation paths; no hot reload."""
    value = declaration if declaration is not None else cfg.get("execution")
    paths = {key: str(Path(cfg[key]).absolute()) for key in (
        "results_dir", "idea_lake_db", "ideas_file", "_orze_dir", "_project_root") if cfg.get(key)}
    paths["cwd"] = str(Path.cwd())
    raw = json.dumps({"execution": value, "action_policy": action_policy(cfg),
                      "action_domain": action_domain(cfg),
                      "cpu_runtime_lease": runtime_lease_config(cfg),
                      "paths": paths}, sort_keys=True, allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def runtime_lease_config(cfg):
    """Normalize invocation policy without changing the budget namespace."""
    if "cpu_runtime_lease" not in cfg:
        return {"version": 1, "ttl_seconds": None}
    value = cfg["cpu_runtime_lease"]
    if (type(value) is not dict or set(value) != {"version", "ttl_seconds"}
            or type(value["version"]) is not int or value["version"] != 1):
        _fail("cpu_runtime_lease requires version 1 and ttl_seconds")
    seconds = value["ttl_seconds"]
    if (type(seconds) not in (int, float) or not 0 < seconds <= 365 * 86400
            or not math.isfinite(seconds) or seconds < 1e-9):
        _fail("cpu_runtime_lease.ttl_seconds must be finite in [1e-9, 31536000]")
    return {"version": 1, "ttl_seconds": float(seconds)}


def runtime_lease_seconds(cfg, timeout_seconds):
    value = runtime_lease_config(cfg)["ttl_seconds"]
    if value is None:
        return timeout_seconds
    if value > timeout_seconds:
        _fail("cpu_runtime_lease.ttl_seconds exceeds action timeout_seconds")
    return value


def validate_cpu_cli(cfg, args):
    if cpu_execution(cfg) is None:
        return
    # Observation subcommands may load the declaration without executing it;
    # existing mutating managed/GPU/service workflows are not CPU adapters.
    if getattr(args, "command", None) in (
            "run-idea", "start", "resume", "restart", "enable", "upgrade"):
        _fail("CPU actions require the foreground orze -c CONFIG entry")
    for name in ("gpus", "role_only", "train_script", "base_config", "timeout",
                 "results_dir", "ideas_file", "poll"):
        if getattr(args, name, None) is not None:
            _fail("CPU foreground does not support --" + name.replace("_", "-"))
    for name in ("research_only", "admin", "restart", "enable", "upgrade", "reinstall"):
        if getattr(args, name, None):
            _fail("CPU foreground does not support --" + name.replace("_", "-"))
