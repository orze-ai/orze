"""CPU phase of Orze's existing foreground loop (no second runner).

The queue policy is deliberately small. Its decisions and execution envelopes
are durable; no ML roles, GPU discovery, implicit evaluations or auto-repair
are hidden inside this resource adapter. Unknown work is retained, not adopted.
"""
from __future__ import annotations

import atexit
import logging
from pathlib import Path
import signal
import socket
import threading
import time
import uuid

import yaml

from orze.core.cpu_execution import (
    CPUExecutionError, action_policy, cpu_execution, execution_fingerprint,
)

logger = logging.getLogger("orze")


class QueuePolicy:
    """A real policy consumer, not a scientific convergence judgment."""
    def __init__(self, declaration):
        self.declaration = dict(declaration)

    def decide(self, snapshot, budget):
        now = snapshot["now"]
        if budget["stopped"]:
            return {"kind": "Stop", "reason": "scope_stopped", "wakeup": None}
        if snapshot["queue"]:
            eligible = next((task for task in snapshot["queue"] if
                task["action"]["timeout_seconds"] <= budget["remaining_wall_seconds"]), None)
            if eligible is not None and budget["free_slots"] > 0:
                return {"kind": "Execute", "task_id": eligible["idea_id"]}
            if eligible is None and budget["active_reservations"] == 0:
                return {"kind": "Stop", "reason": "wall_envelope_exhausted", "wakeup": None}
            return {"kind": "Wait", "reason": "cpu_resource_or_budget_unavailable",
                    "wakeup": now + self.declaration["wait_seconds"]}
        scope_active = snapshot["active"] or budget["active_reservations"] > 0
        if not scope_active and self.declaration["idle"] == "stop":
            return {"kind": "Stop", "reason": "queue_drained", "wakeup": None}
        return {"kind": "Wait", "reason": "actions_running" if scope_active else "queue_empty",
                "wakeup": now + self.declaration["wait_seconds"]}


def initialize(engine, gpu_ids, cfg, once):
    from orze.core.config import _validate_config
    from orze.core.control_outcome import require_controller_start_allowed
    from orze.idea_lake import IdeaLake
    from orze.core.research_interfaces import BoundPolicy, capture_interfaces
    if type(gpu_ids) is not list or gpu_ids:
        raise CPUExecutionError("execution: CPU actions require an empty physical GPU scope")
    errors, _ = _validate_config(cfg)
    if errors:
        raise CPUExecutionError("execution: invalid CPU configuration: " + "; ".join(errors))
    engine.cfg = cfg
    engine.once = once
    engine.results_dir = Path(cfg["results_dir"]).absolute()
    require_controller_start_allowed(engine.results_dir)
    engine._cpu_cfg_fingerprint = execution_fingerprint(cfg)
    engine.gpu_ids = []
    engine.active = {}
    engine.active_evals = {}
    engine.active_roles = {}
    engine._gpu_leases = None
    engine._controller_profile_enabled = False
    engine._hostname = socket.gethostname()
    engine._instance_uuid = uuid.uuid4().hex
    engine.running = True
    engine.iteration = 0
    engine._stop_event = threading.Event()
    engine._cpu_handles = {}
    engine._cpu_closed = False
    engine._cpu_scope = None
    engine._cpu_wait_until = 0.0
    engine._cpu_once_dispatched = False
    try:
        engine._cpu_interfaces = capture_interfaces(cfg)
        engine._cpu_policy = (QueuePolicy(action_policy(cfg)) if engine._cpu_interfaces is None
                              else BoundPolicy(engine._cpu_interfaces))
    except Exception as exc:
        raise CPUExecutionError("execution: interface initialization rejected") from exc
    engine.results_dir.mkdir(parents=True, exist_ok=True)
    # CPU execution cannot downgrade to text-only lifecycle or copy an old DB.
    if not cfg.get("idea_lake_db"):
        raise CPUExecutionError("execution: CPU actions require an IdeaLake path")
    engine.lake = IdeaLake(cfg["idea_lake_db"])
    signal.signal(signal.SIGINT, engine._shutdown)
    signal.signal(signal.SIGTERM, engine._shutdown)
    atexit.register(engine._atexit_cleanup)


def require_admission(engine):
    from orze.core.control_outcome import require_controller_start_allowed
    from orze.core.cpu_action_budget import snapshot
    if not engine.running or engine._stop_event.is_set():
        raise CPUExecutionError("execution: controller is stopping")
    if (cpu_execution(engine.cfg) is None or execution_fingerprint(engine.cfg)
            != engine._cpu_cfg_fingerprint):
        raise CPUExecutionError("execution: CPU invocation changed")
    require_controller_start_allowed(engine.results_dir)
    if engine._cpu_scope is not None and snapshot(engine.lake, engine._cpu_scope)["stopped"]:
        raise CPUExecutionError("execution: CPU policy has stopped this scope")


def start(engine):
    from orze.core.cpu_action_budget import initialize as initialize_budget
    from orze.engine.health import HealthMonitor
    require_admission(engine)
    engine._cpu_scope = initialize_budget(engine.lake, engine.results_dir, cpu_execution(engine.cfg))
    engine._health_monitor = HealthMonitor(engine.results_dir)
    logger.info("Orze CPU actions: %s, declared slots=%s (no GPU resource allocation)",
                engine.results_dir, engine._cpu_execution["slots"])


def iteration(engine):
    """One existing-loop iteration; waiting uses the shared interruptible event."""
    from orze.core import cpu_action_budget as budget
    from orze.core.cpu_action_contract import validate_action
    from orze.core.control_outcome import ControllerStopHOLD, require_controller_start_allowed
    from orze.engine import native_cpu_action as executor
    from orze.engine.health import check_disk_space
    from orze.engine.idea_ingress import ingest_ideas_source
    from orze.engine.scheduler import claim
    from orze.core.research_interfaces import parse_domain_task, prepare_domain_run

    for key, (handle, permit) in list(engine._cpu_handles.items()):
        terminal = executor.harvest(handle, engine.results_dir, engine.cfg,
                                   lake=engine.lake, permit=permit)
        if terminal is not None:
            del engine._cpu_handles[key]
            logger.info("CPU action %s: %s", handle.idea_id, terminal.get("outcome"))
    try:
        require_controller_start_allowed(engine.results_dir)
    except ControllerStopHOLD:
        budget.record_decision(engine.lake, engine._cpu_scope,
            {"kind": "Stop", "reason": "operator_stop", "wakeup": None})
        return False
    if engine.once and engine._cpu_once_dispatched:
        if not engine._cpu_handles:
            return False
        engine._stop_event.wait(0.05)
        return True
    if not engine._health_monitor.check_before_write():
        delay = engine._health_monitor.retry_delay
        engine._stop_event.wait(min(delay, 0.05) if engine._cpu_handles else delay)
        return True
    # A durable Wait is not a lease expiry or permission to repeat an action.
    remaining = engine._cpu_wait_until - time.monotonic()
    if remaining > 0:
        engine._stop_event.wait(min(remaining, 0.05) if engine._cpu_handles else remaining)
        return True
    if len(engine._cpu_handles) >= engine._cpu_execution["slots"]:
        engine._stop_event.wait(0.05)
        return True
    require_admission(engine)
    ingest_ideas_source(engine, engine.cfg)
    interfaces = engine._cpu_interfaces
    domain_enabled = engine.cfg.get("action_domain") is not None
    queue = engine.lake.get_queue(limit=2000 if interfaces is None else 32)
    for queued in queue:
        row = engine.lake.get(queued["idea_id"])
        if row is None or row["kind"] != "native_cpu_action":
            raise CPUExecutionError("execution: CPU queue contains a non-CPU task")
        try:
            if domain_enabled:
                request = parse_domain_task(row["config"])
                queued["_raw_config"] = row["config"]
                queued["request"] = request
                queued["action"] = {"timeout_seconds": request["timeout_seconds"]}
            else:
                parsed = yaml.safe_load(row["config"])
                if type(parsed) is not dict:
                    raise ValueError("not a mapping")
                if parsed.get("domain_request") is not None:
                    raise ValueError("domain request has no selected domain")
                queued["action"] = validate_action(parsed.get("action"))
        except (ValueError, TypeError, yaml.YAMLError) as exc:
            raise CPUExecutionError("execution: queued CPU action declaration is invalid") from exc
    snapshot = {"queue": queue, "active": bool(engine._cpu_handles), "now": time.time()}
    try:
        if interfaces is not None:
            from orze.engine.cpu_policy_evidence import recorded_evidence
            snapshot["queue"] = [{"idea_id": item["idea_id"],
                "action": {"timeout_seconds": item["action"]["timeout_seconds"]},
                "request": item.get("request", item["action"])} for item in queue]
            snapshot["recorded_evidence"] = recorded_evidence(engine.lake, engine.results_dir)
        decision = engine._cpu_policy.decide(snapshot, budget.snapshot(engine.lake, engine._cpu_scope))
    except Exception as exc:
        raise CPUExecutionError("execution: research policy decision rejected") from exc
    if decision["kind"] == "Execute":
        idea_id = decision["task_id"]
        action = next(task["action"] for task in queue if task["idea_id"] == idea_id)
        if not check_disk_space(engine.results_dir, engine.cfg.get("min_disk_gb", 5)):
            decision = {"kind": "Wait", "reason": "disk_space",
                        "wakeup": time.time() + engine._cpu_policy.declaration["wait_seconds"]}
        else:
            domain_run = None
            if domain_enabled:
                from orze.engine.cpu_action_sources import capture_sources
                item = next(task for task in queue if task["idea_id"] == idea_id)
                try:
                    sources = capture_sources(engine.lake, engine.results_dir,
                                              item["request"]["input_artifact_ids"])
                    domain_run = prepare_domain_run(interfaces, item["_raw_config"], sources)
                    action = domain_run.action
                except Exception as exc:
                    raise CPUExecutionError("execution: domain preparation rejected") from exc
            permit = budget.reserve(engine.lake, engine._cpu_scope, idea_id, action["timeout_seconds"])
            if permit is None:
                decision = {"kind": "Wait", "reason": "cpu_resource_or_budget_unavailable",
                            "wakeup": time.time() + engine._cpu_policy.declaration["wait_seconds"]}
            else:
                require_admission(engine)
                if not claim(idea_id, engine.results_dir, None, lake=engine.lake, resource="cpu"):
                    raise CPUExecutionError("execution: reserved action claim unconfirmed; budget retained")
                try:
                    extension = {} if domain_run is None else {"domain_run": domain_run}
                    handle = executor.launch(idea_id, engine.results_dir, engine.cfg,
                        lake=engine.lake, action=action, permit=permit,
                        admission=lambda: require_admission(engine), **extension)
                except BaseException as exc:
                    handle = getattr(exc, "cpu_action_handle", None)
                    if handle is not None:
                        engine._cpu_handles[permit["reservation_id"]] = (handle, permit)
                    raise
                engine._cpu_handles[permit["reservation_id"]] = (handle, permit)
                engine._cpu_once_dispatched = True
                return True
    budget.record_decision(engine.lake, engine._cpu_scope, decision)
    logger.info("CPU policy %s: %s", decision["kind"], decision["reason"])
    if decision["kind"] == "Stop" or engine.once:
        return False
    engine._cpu_wait_until = time.monotonic() + max(0, decision["wakeup"] - time.time())
    return True


def close(engine):
    """Only captured action owners are stopped; no PID discovery or adoption."""
    if engine._cpu_closed:
        return
    engine._cpu_closed = True
    engine.running = False
    engine._stop_event.set()
    failures = []
    from orze.engine import native_cpu_action as executor
    for key, (handle, permit) in list(engine._cpu_handles.items()):
        try:
            executor.stop(handle, engine.results_dir, engine.cfg,
                          lake=engine.lake, permit=permit)
            del engine._cpu_handles[key]
        except BaseException as exc:
            failures.append(type(exc).__name__)
            logger.error("CPU action %s remains HOLD; reservation retained", handle.idea_id)
    try:
        engine.lake.close()
    except BaseException as exc:
        failures.append(type(exc).__name__)
    if failures:
        raise CPUExecutionError("execution: CPU cleanup remains unconfirmed: " + ",".join(failures))
