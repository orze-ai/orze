"""Explicit local Domain/Policy extensions for the existing CPU product loop.

Registration is trusted Python application setup, never an import path supplied
by a queued task. Each invocation captures its selected implementations. These
callbacks receive detached bounded data, not Lake or executor authority; this is
an API boundary, not an OS sandbox or a time limit for arbitrary trusted Python.
Prepared runs are strong process-local captures, not serializable permissions.
The handles own their captures; lookup registries do not keep unused handles or
callback cycles alive. Collection grants no execution or settlement authority.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from weakref import WeakValueDictionary

import yaml

from orze.core.artifact_contract import get_artifact_contract
from orze.core.cpu_action_contract import action_fingerprint, validate_action
from orze.core.execution_attempts import AttemptAuthorityError, _json


class ResearchInterfaceError(ValueError):
    pass


_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_DOMAINS, _POLICIES = {}, {}
_CONTEXTS, _RUNS = WeakValueDictionary(), WeakValueDictionary()


class _CaptureState(dict):
    """Weakly indexed state, strongly owned by its exact opaque handle.

    The original-handle witness may form a cycle, including through a trusted
    callback. Only weak registry values point into that cycle, so an abandoned
    capture is collectable while external consumers retain the complete state.
    """

    __slots__ = ("__weakref__",)


def _fail(reason):
    raise ResearchInterfaceError("research_interface: " + reason)


def _copy(value):
    try:
        return json.loads(_json({"value": value}))["value"]
    except (AttemptAuthorityError, ValueError, TypeError, RecursionError, UnicodeError, OverflowError) as exc:
        raise ResearchInterfaceError("research_interface: bounded JSON required") from exc


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _same(a, b):
    return _json({"value": a}) == _json({"value": b})


def _token(value):
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        _fail("invalid registered identity")
    return value


def _register(registry, name, implementation_id, factory):
    _token(name)
    _token(implementation_id)
    if not callable(factory) or name in registry or len(registry) >= 64:
        _fail("registration unavailable or duplicate")
    registry[name] = (implementation_id, factory)


def register_domain(name, implementation_id, factory):
    """Register a trusted factory(config)->Domain before selecting it."""
    _register(_DOMAINS, name, implementation_id, factory)


def register_policy(name, implementation_id, factory):
    """Register a trusted factory(declaration)->Policy before selecting it."""
    _register(_POLICIES, name, implementation_id, factory)


def _entry(registry, name):
    _token(name)
    if name not in registry:
        _fail("implementation is not registered: " + name)
    return registry[name]


def domain_declaration(cfg):
    value = cfg.get("action_domain")
    if value is None:
        return None
    if (type(value) is not dict or set(value) != {"version", "kind", "config"}
            or type(value["version"]) is not int or value["version"] != 1
            or type(value["config"]) is not dict):
        _fail("action_domain requires version 1, registered kind and config mapping")
    _entry(_DOMAINS, value["kind"])
    return _copy(value)


def require_policy_declaration(declaration):
    """Validate selection without constructing or invoking the implementation."""
    _entry(_POLICIES, declaration["kind"])
    if type(declaration.get("config", {})) is not dict:
        _fail("policy config must be a mapping")
    if declaration["kind"] == "queue" and declaration.get("config"):
        _fail("queue policy has no additional config")
    return _copy(declaration)


def validate_domain_request(value):
    """Generic task envelope; the domain supplies the actual command."""
    fields = {"version", "purpose", "inputs", "timeout_seconds", "outputs",
              "input_artifact_ids", "payload"}
    if (type(value) is not dict or set(value) != fields
            or type(value["version"]) is not int or value["version"] != 1
            or type(value["purpose"]) is not str or not value["purpose"].strip()
            or type(value["inputs"]) is not dict or type(value["payload"]) is not dict):
        _fail("invalid domain request envelope")
    timeout = value["timeout_seconds"]
    try:
        valid = type(timeout) in (int, float) and timeout > 0 and math.isfinite(timeout)
    except (ValueError, OverflowError):
        valid = False
    if not valid:
        _fail("domain timeout must be positive and finite")
    ids = value["input_artifact_ids"]
    if type(ids) is not list or len(ids) > 32:
        _fail("domain input references must be a bounded list")
    for identity in ids:
        _token(identity)
    if len(set(ids)) != len(ids):
        _fail("duplicate domain input reference")
    result = _copy(value)
    result["outputs"] = get_artifact_contract({"artifact_contract": {
        "version": 1, "outputs": result["outputs"]}})["outputs"]
    return result


def parse_domain_task(raw_config):
    if type(raw_config) is not str or len(raw_config.encode()) > 65536:
        _fail("domain task config is not bounded text")
    try:
        value = yaml.safe_load(raw_config)
    except yaml.YAMLError as exc:
        raise ResearchInterfaceError("research_interface: invalid task YAML") from exc
    if (type(value) is not dict or set(value) != {"kind", "domain_request"}
            or value["kind"] != "native_cpu_action"):
        _fail("explicit domain task requires kind and domain_request only")
    return validate_domain_request(value["domain_request"])


def validate_proposal_decision(value):
    """Validate data-only normal admission; this grants no execution authority."""
    if (type(value) is not dict or set(value) != {
            "kind", "request_id", "task_id", "reason", "domain_request"}
            or value["kind"] != "Propose"
            or type(value["reason"]) is not str or not value["reason"].strip()
            or len(value["reason"].encode()) > 1024):
        _fail("Propose requires stable identities, bounded reason and domain request")
    _token(value["request_id"])
    _token(value["task_id"])
    result = _copy(value)
    result["domain_request"] = validate_domain_request(value["domain_request"])
    return result


def proposal_sources(snapshot, decision):
    """Select original metadata, never authorizing a callback's modified view.

    The coordinator must still verify these records against its actual Lake,
    effects and bytes. Callers retain a private snapshot before invoking Policy.
    """
    selected = validate_proposal_decision(decision)
    results = snapshot.get("recorded_evidence", {}).get("results", [])
    records = []
    for identity in selected["domain_request"]["input_artifact_ids"]:
        matches = []
        for result in results:
            if result.get("outcome") != "completed":
                continue
            ref = result.get("ref")
            if (type(ref) is not dict or set(ref) != {
                    "task_id", "phase", "attempt_id", "generation"}
                    or ref["phase"] != "action" or type(ref["generation"]) is not int
                    or not 0 < ref["generation"] < 2**63):
                continue
            for artifact in result.get("artifact_records", []):
                if (artifact.get("artifact_id") == identity
                        and _same(artifact.get("producer"), ref)):
                    matches.append(artifact)
        if len(matches) != 1:
            _fail("Propose must select unique completed inputs in captured evidence")
        records.append(_copy(matches[0]))
    return tuple(records)


@dataclass(eq=False, frozen=True)
class InterfaceContext:
    """Opaque invocation handle; constructing a lookalike confers no authority."""

    _state: object = field(default=None, init=False, repr=False)


@dataclass(eq=False, frozen=True)
class DomainRun:
    """Opaque prepared-run handle, valid only while its strong owner is retained."""

    _state: object = field(default=None, init=False, repr=False)

    @property
    def action(self):
        return _copy(_run(self)["prepared"]["action"])


def capture_interfaces(cfg):
    from orze.core.cpu_execution import action_policy
    domain = domain_declaration(cfg)
    policy = action_policy(cfg)
    if domain is None and policy["kind"] == "queue":
        return None  # Preserve the complete original A queue/command path.
    pentry = _entry(_POLICIES, policy["kind"])
    policy_object = pentry[1](_copy(policy))
    decide = getattr(policy_object, "decide", None)
    if not callable(decide):
        _fail("policy has no decide method")
    dentry = None if domain is None else _entry(_DOMAINS, domain["kind"])
    domain_object = None if dentry is None else dentry[1](_copy(domain["config"]))
    prepare = getattr(domain_object, "prepare", None)
    interpret = getattr(domain_object, "interpret", None)
    if dentry is not None and (not callable(prepare) or not callable(interpret)):
        _fail("domain must implement prepare and interpret")
    handle = InterfaceContext()
    state = _CaptureState({"handle": handle, "domain": domain, "policy": policy,
        "domain_entry": dentry, "policy_entry": pentry, "domain_object": domain_object,
        "policy_object": policy_object, "prepare": prepare, "interpret": interpret,
        "decide": decide})
    object.__setattr__(handle, "_state", state)
    _CONTEXTS[id(handle)] = state
    return handle


def _context(context):
    state = _CONTEXTS.get(id(context))
    if (type(context) is not InterfaceContext or state is None
            or state["handle"] is not context or context._state is not state):
        _fail("invocation capture unavailable")
    if _entry(_POLICIES, state["policy"]["kind"]) is not state["policy_entry"]:
        _fail("selected policy registration changed")
    if state["domain"] is not None and _entry(_DOMAINS, state["domain"]["kind"]) is not state["domain_entry"]:
        _fail("selected domain registration changed")
    return state


class BoundPolicy:
    def __init__(self, context):
        _context(context)
        self._context = context

    @property
    def declaration(self):
        return _copy(_context(self._context)["policy"])

    def decide(self, snapshot, budget):
        state = _context(self._context)
        captured = _copy(snapshot)
        decision = _copy(state["decide"](_copy(captured), _copy(budget)))
        if type(decision) is not dict:
            _fail("policy must return an explicit decision")
        kind = decision.get("kind")
        if kind == "Execute":
            if (set(decision) != {"kind", "task_id"}
                    or decision["task_id"] not in {r["idea_id"] for r in captured["queue"]}):
                _fail("Execute must select a task in this captured queue")
        elif kind == "Propose":
            if state["domain"] is None:
                _fail("Propose requires a selected Domain")
            decision = validate_proposal_decision(decision)
            proposal_sources(captured, decision)
        elif kind == "Replicate":
            from orze.core.replication_requests import token
            if (set(decision) != {"kind", "source_ref", "request_id", "reason"}
                    or type(decision["reason"]) is not str or not decision["reason"].strip()
                    or len(decision["reason"].encode()) > 1024):
                _fail("Replicate requires an explicit source, stable key and bounded reason")
            ref = decision["source_ref"]
            if (type(ref) is not dict or set(ref) != {"task_id", "phase", "attempt_id", "generation"}
                    or ref["phase"] != "action" or type(ref["generation"]) is not int
                    or not 0 < ref["generation"] < 2**63):
                _fail("Replicate requires an exact action occurrence")
            try:
                for value in (decision["request_id"], ref["task_id"], ref["attempt_id"]):
                    token(value)
            except ValueError as exc:
                raise ResearchInterfaceError("research_interface: invalid replication identity") from exc
            if not any(item.get("outcome") == "completed" and _same(item.get("ref"), ref)
                       for item in captured.get("recorded_evidence", {}).get("results", [])):
                _fail("Replicate must select a completed occurrence in captured evidence")
        elif kind in ("Wait", "Pause", "Stop"):
            if (set(decision) != {"kind", "reason", "wakeup"}
                    or type(decision["reason"]) is not str or not decision["reason"]
                    or len(decision["reason"].encode()) > 128):
                _fail("Wait/Pause/Stop requires a bounded reason and wakeup")
            wakeup = decision["wakeup"]
            if kind in ("Pause", "Stop"):
                if wakeup is not None:
                    _fail("Pause/Stop has no wakeup")
                if kind == "Pause" and (
                        not decision["reason"].strip()
                        or captured.get("active") is not False
                        or type(budget.get("active_reservations")) is not int
                        or budget["active_reservations"] != 0):
                    _fail("Pause requires a reason and confirmed quiescent resources")
            elif (type(wakeup) not in (int, float)
                    or not captured["now"] + 0.01 <= wakeup <= captured["now"] + 3600):
                _fail("Wait must declare a future wakeup within one hour")
        else:
            _fail("unknown policy decision")
        _context(self._context)
        return decision


def _prepared(value, request):
    if type(value) is not dict or set(value) != {"action", "observation"}:
        _fail("Domain.prepare requires action and explicit observation declaration")
    action = validate_action(value["action"])
    for field in ("purpose", "timeout_seconds", "outputs"):
        if not _same(action[field], request[field]):
            _fail("domain changed declared purpose, execution bound or output contract")
    declaration = value["observation"]
    if declaration is not None:
        if type(declaration) is not dict or set(declaration) != {
                "adapter_id", "spec_fingerprint", "protocol_fingerprint", "result_output"}:
            _fail("invalid observation preparation declaration")
        _token(declaration["adapter_id"])
        for field in ("spec_fingerprint", "protocol_fingerprint"):
            if type(declaration[field]) is not str or _SHA.fullmatch(declaration[field]) is None:
                _fail("invalid observation fingerprint")
        name = declaration["result_output"]
        if (type(name) is not str or name not in action["outputs"]
                or action["outputs"][name]["max_bytes"] > 1048576):
            _fail("observation result must name an output bounded to 1 MiB")
    return _copy({"action": action, "observation": declaration})


def prepare_domain_run(context, raw_config, prepared_sources):
    from orze.engine.cpu_action_sources import records, snapshot
    state = _context(context)
    if state["domain"] is None:
        _fail("task requires an explicit domain selection")
    request = parse_domain_task(raw_config)
    source_records = records(prepared_sources)
    if request["input_artifact_ids"] != [r["artifact_id"] for r in source_records]:
        _fail("prepared sources do not match this request")
    prepared = _prepared(state["prepare"](_copy(request), tuple(_copy(source_records))), request)
    metadata = _copy({"schema": 1, "domain_id": state["domain_entry"][0],
        "domain_kind": state["domain"]["kind"],
        "domain_config_sha256": _hash(state["domain"]),
        "request_sha256": hashlib.sha256(raw_config.encode()).hexdigest(),
        "action_sha256": action_fingerprint(prepared["action"]),
        "source_snapshot": snapshot(prepared_sources), "observation": prepared["observation"]})
    _context(context)
    handle = DomainRun()
    captured = _CaptureState({"handle": handle, "context": context, "prepared": prepared,
        "metadata": metadata, "sources": prepared_sources, "interpret": state["interpret"]})
    object.__setattr__(handle, "_state", captured)
    _RUNS[id(handle)] = captured
    return handle


def _run(run):
    state = _RUNS.get(id(run))
    if (type(run) is not DomainRun or state is None
            or state["handle"] is not run or run._state is not state):
        _fail("prepared domain run unavailable")
    _context(state["context"])
    return state


def domain_run_metadata(run):
    return _copy(_run(run)["metadata"])


def domain_sources(run):
    return _run(run)["sources"]


def require_domain_run(run, *, raw_config_sha256, action):
    state = _run(run)
    if (raw_config_sha256 != state["metadata"]["request_sha256"]
            or not _same(validate_action(action), state["prepared"]["action"])):
        _fail("domain request or prepared action changed")
    return _copy(state["metadata"])


def interpret_domain_run(run, envelope):
    state = _run(run)
    claims = state["interpret"](_copy(state["prepared"]), _copy(envelope))
    if type(claims) is not tuple or len(claims) > 32:
        _fail("Domain.interpret must explicitly return zero to 32 claims")
    if state["prepared"]["observation"] is None and claims:
        _fail("undeclared observations are not permitted")
    _run(run)
    return tuple(_copy(list(claims)))


class CommandDomain:
    def __init__(self, config):
        if config != {}:
            _fail("command domain has no settings")

    def prepare(self, request, sources):
        if set(request["payload"]) != {"command"}:
            _fail("command domain payload requires command argv only")
        return {"action": {"version": 1, "adapter": "command",
            **{k: request[k] for k in ("purpose", "inputs", "timeout_seconds", "outputs")},
            "command": request["payload"]["command"]}, "observation": None}

    def interpret(self, prepared, envelope):
        if envelope is not None:
            _fail("command domain has no measurement envelope")
        return ()


class JsonObservationDomain(CommandDomain):
    def prepare(self, request, sources):
        payload = request["payload"]
        if (set(payload) != {"command", "specification", "protocol", "result_output"}
                or type(payload["specification"]) is not dict or type(payload["protocol"]) is not dict):
            _fail("JSON domain requires explicit specification, protocol and result output")
        action = {"version": 1, "adapter": "command",
            **{k: request[k] for k in ("purpose", "inputs", "timeout_seconds", "outputs")},
            "command": payload["command"]}
        return {"action": action, "observation": {
            "adapter_id": "orze.json_cpu_domain.v1",
            "spec_fingerprint": _hash({"schema": "orze.domain_subject.v1", "specification": payload["specification"]}),
            "protocol_fingerprint": _hash({"schema": "orze.domain_protocol.v1", "protocol": payload["protocol"]}),
            "result_output": payload["result_output"]}}

    def interpret(self, prepared, envelope):
        if (type(envelope) is not dict or set(envelope) != {"version", "observations"}
                or type(envelope["version"]) is not int or envelope["version"] != 1
                or type(envelope["observations"]) is not list):
            _fail("JSON domain requires an explicit version-1 observation envelope")
        return tuple(envelope["observations"])


def _queue_factory(declaration):
    from orze.engine.cpu_phase import QueuePolicy
    return QueuePolicy(declaration)


register_domain("command", "orze.command_domain.v1", CommandDomain)
register_domain("json_observations", "orze.json_cpu_domain.v1", JsonObservationDomain)
register_policy("queue", "orze.queue_policy.v1", _queue_factory)
