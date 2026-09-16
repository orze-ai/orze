"""Finite scheduling application; no Core scheduler/resource-policy changes.

Capacity and demand are simulated problem quantities. The deterministic
generators are not optimizers. Only the evaluator assigns scientific validity
and value; producer artifacts never contain an authoritative score.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

MAX_ARTIFACT_BYTES = 16384
PROTOCOLS = ("schedule-feasibility-v1", "schedule-feasibility-v2")
CANDIDATES = ("baseline", "challenger")
PRODUCER_OUTPUTS = {"candidate": {"path": "candidate.json", "max_bytes": MAX_ARTIFACT_BYTES}}
EVALUATOR_OUTPUTS = {"evaluation": {"path": "evaluation.json", "max_bytes": MAX_ARTIFACT_BYTES}}
INVALID_REASONS = (
    "candidate_json_invalid", "candidate_schema_invalid", "unknown_job",
    "duplicate_job", "missing_required", "time_window", "missing_prerequisite",
    "precedence_violation", "capacity_overload",
)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonfinite JSON number")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _topological(jobs):
    remaining = list(jobs)
    done, ordered = set(), []
    while remaining:
        selected = next((job for job in remaining if set(job["after"]) <= done), None)
        if selected is None:
            raise ValueError("instance dependencies must be acyclic")
        remaining.remove(selected)
        ordered.append(selected)
        done.add(selected["id"])
    return ordered


def validate_instance(instance):
    """Return detached, bounded input; reject configuration ambiguity."""
    if (type(instance) is not dict or set(instance) != {"instance_id", "horizon", "capacity", "jobs"}
            or type(instance["instance_id"]) is not str or not _ID.fullmatch(instance["instance_id"])
            or not _integer(instance["horizon"], 1, 10000)
            or not _integer(instance["capacity"], 1, 1024)
            or type(instance["jobs"]) is not list or not 1 <= len(instance["jobs"]) <= 128):
        raise ValueError("invalid scheduling instance")
    known = set()
    fields = {"id", "release", "deadline", "duration", "demand", "value", "required", "after"}
    for job in instance["jobs"]:
        if (type(job) is not dict or set(job) != fields
                or type(job["id"]) is not str or not _ID.fullmatch(job["id"])
                or job["id"] in known
                or not _integer(job["release"], 0, 10000)
                or not _integer(job["deadline"], 1, 10000)
                or not _integer(job["duration"], 1, 10000)
                or not _integer(job["demand"], 1, 1024)
                or not _integer(job["value"], -1000000000, 1000000000)
                or type(job["required"]) is not bool or type(job["after"]) is not list
                or len(job["after"]) > 128
                or any(type(value) is not str or not _ID.fullmatch(value) for value in job["after"])
                or len(set(job["after"])) != len(job["after"])):
            raise ValueError("invalid scheduling job")
        known.add(job["id"])
    if any(job["id"] in job["after"] or not set(job["after"]) <= known for job in instance["jobs"]):
        raise ValueError("invalid scheduling dependency")
    _topological(instance["jobs"])
    if len(_canonical(instance)) > MAX_ARTIFACT_BYTES:
        raise ValueError("scheduling instance exceeds byte bound")
    return copy.deepcopy(instance)


def produce(instance, candidate):
    """Required-only baseline or stable-topological earliest-fit greedy.

    No backtracking or optimality claim. A required job that cannot fit these
    choices raises, rather than returning a purportedly feasible schedule.
    """
    instance = validate_instance(instance)
    if candidate not in CANDIDATES:
        raise ValueError("unknown scheduling candidate")
    usage = [0] * instance["horizon"]
    ends, entries = {}, []
    for job in _topological(instance["jobs"]):
        if candidate == "baseline" and not job["required"]:
            continue
        start = None
        if set(job["after"]) <= set(ends):
            earliest = max([job["release"], *(ends[name] for name in job["after"])])
            latest = min(job["deadline"], instance["horizon"]) - job["duration"]
            for position in range(earliest, latest + 1):
                if all(usage[tick] + job["demand"] <= instance["capacity"]
                       for tick in range(position, position + job["duration"])):
                    start = position
                    break
        if start is None:
            if job["required"]:
                raise ValueError("required job cannot fit deterministic candidate")
            continue
        end = start + job["duration"]
        for tick in range(start, end):
            usage[tick] += job["demand"]
        ends[job["id"]] = end
        entries.append({"job_id": job["id"], "start": start})
    return {"instance_id": instance["instance_id"], "schedule": entries}


def evaluate(instance, raw_bytes, protocol):
    """Classify candidate bytes; invalid candidates have no objective number."""
    instance = validate_instance(instance)
    if protocol not in PROTOCOLS:
        raise ValueError("unknown scheduling protocol")
    def invalid(reason):
        return {"status": "invalid", "reason_code": reason}
    if type(raw_bytes) is not bytes:
        return invalid("candidate_schema_invalid")
    try:
        if len(raw_bytes) > MAX_ARTIFACT_BYTES:
            raise ValueError("candidate byte bound")
        candidate = _json(raw_bytes)
    except (ValueError, UnicodeError, RecursionError, OverflowError):
        return invalid("candidate_json_invalid")
    if (type(candidate) is not dict or set(candidate) != {"instance_id", "schedule"}
            or candidate["instance_id"] != instance["instance_id"]
            or type(candidate["schedule"]) is not list or len(candidate["schedule"]) > 128):
        return invalid("candidate_schema_invalid")
    for entry in candidate["schedule"]:
        if (type(entry) is not dict or set(entry) != {"job_id", "start"}
                or type(entry["job_id"]) is not str
                or not _integer(entry["start"], 0, 2**63 - 1)):
            return invalid("candidate_schema_invalid")
    jobs = {job["id"]: job for job in instance["jobs"]}
    starts = {}
    for entry in candidate["schedule"]:
        name = entry["job_id"]
        if name not in jobs:
            return invalid("unknown_job")
        if name in starts:
            return invalid("duplicate_job")
        starts[name] = entry["start"]
    if any(job["required"] and job["id"] not in starts for job in instance["jobs"]):
        return invalid("missing_required")
    for name, start in starts.items():
        job = jobs[name]
        if start < job["release"] or start + job["duration"] > min(job["deadline"], instance["horizon"]):
            return invalid("time_window")
    if any(not set(jobs[name]["after"]) <= set(starts) for name in starts):
        return invalid("missing_prerequisite")
    for name, start in starts.items():
        if any(starts[parent] + jobs[parent]["duration"] > start for parent in jobs[name]["after"]):
            return invalid("precedence_violation")
    # The only v2 semantic change is the simulated capacity.
    capacity = instance["capacity"] if protocol == PROTOCOLS[0] else 1
    for tick in range(instance["horizon"]):
        demand = sum(jobs[name]["demand"] for name, start in starts.items()
                     if start <= tick < start + jobs[name]["duration"])
        if demand > capacity:
            return invalid("capacity_overload")
    return {"status": "valid", "reason_code": "feasible",
            "scheduled_value": sum(jobs[name]["value"] for name in starts)}


def _payload(payload):
    if type(payload) is not dict:
        raise ValueError("invalid scheduling operation")
    if payload.get("operation") == "produce":
        if (set(payload) not in ({"operation", "candidate"}, {"operation", "candidate", "artifact_utf8"})
                or payload["candidate"] not in CANDIDATES):
            raise ValueError("invalid producer request")
        if "artifact_utf8" in payload and (type(payload["artifact_utf8"]) is not str
                or len(payload["artifact_utf8"].encode("utf-8")) > MAX_ARTIFACT_BYTES):
            raise ValueError("invalid explicit candidate fixture")
    elif payload.get("operation") == "evaluate":
        if set(payload) != {"operation", "protocol"} or payload["protocol"] not in PROTOCOLS:
            raise ValueError("invalid evaluator request")
    else:
        raise ValueError("unknown scheduling operation")
    return copy.deepcopy(payload)


def make_request(operation, candidate=None, protocol=None, source_id=None, artifact_utf8=None):
    """Build an ordinary seven-field Domain request; retries do not add salt."""
    if operation == "produce":
        if protocol is not None or source_id is not None:
            raise ValueError("producer has no protocol/source")
        payload = {"operation": operation, "candidate": "baseline" if candidate is None else candidate}
        if artifact_utf8 is not None:
            payload["artifact_utf8"] = artifact_utf8
        outputs, ids = PRODUCER_OUTPUTS, []
    elif operation == "evaluate":
        if candidate is not None or artifact_utf8 is not None or type(source_id) is not str or not _ID.fullmatch(source_id):
            raise ValueError("evaluation requires one source artifact")
        payload = {"operation": operation, "protocol": protocol}
        outputs, ids = EVALUATOR_OUTPUTS, [source_id]
    else:
        raise ValueError("unknown scheduling operation")
    return {"version": 1, "purpose": operation + " scheduling candidate", "inputs": {},
            "timeout_seconds": 2, "outputs": copy.deepcopy(outputs),
            "input_artifact_ids": ids, "payload": _payload(payload)}


def _bindings(sources):
    result = []
    for source in sources:
        identity, digest = source.get("artifact_id"), source.get("content_sha256")
        if (type(identity) is not str or not _ID.fullmatch(identity)
                or type(digest) is not str or not _SHA.fullmatch(digest)):
            raise ValueError("invalid captured scheduling source")
        result.append({"artifact_id": identity, "content_sha256": digest})
    return result


class SchedulingDomain:
    def __init__(self, config):
        if type(config) is not dict or set(config) != {"instance"}:
            raise ValueError("scheduling config requires only instance")
        self.instance = validate_instance(config["instance"])

    def prepare(self, request, sources):
        fields = {"version", "purpose", "inputs", "timeout_seconds", "outputs", "input_artifact_ids", "payload"}
        if (type(request) is not dict or set(request) != fields or type(request["version"]) is not int
                or request["version"] != 1 or type(request["purpose"]) is not str or not request["purpose"].strip()
                or request["inputs"] != {} or type(request["timeout_seconds"]) not in (int, float)
                or request["timeout_seconds"] != 2 or type(sources) not in (tuple, list)):
            raise ValueError("invalid scheduling request")
        payload = _payload(request["payload"])
        producing = payload["operation"] == "produce"
        if len(sources) != (0 if producing else 1):
            raise ValueError("scheduling source count changed")
        bindings = _bindings(sources)
        if (request["input_artifact_ids"] != [item["artifact_id"] for item in bindings]
                or request["outputs"] != (PRODUCER_OUTPUTS if producing else EVALUATOR_OUTPUTS)):
            raise ValueError("scheduling request binding changed")
        inputs = {"instance": copy.deepcopy(self.instance), "payload": payload, "source_bindings": bindings}
        observation = None
        if not producing:
            observation = {"adapter_id": "holdout.scheduling.v1",
                "spec_fingerprint": _digest({"schema": "scheduling.subject.v1",
                    "instance": self.instance, "sources": bindings}),
                "protocol_fingerprint": _digest({"schema": "scheduling.protocol.v1",
                    "protocol": payload["protocol"]}), "result_output": "evaluation"}
        return {"action": {"version": 1, "adapter": "command", "purpose": request["purpose"],
                "inputs": inputs, "command": [sys.executable, str(Path(__file__).resolve())],
                "timeout_seconds": request["timeout_seconds"], "outputs": copy.deepcopy(request["outputs"])},
                "observation": observation}

    def interpret(self, prepared, envelope):
        inputs = prepared["action"]["inputs"]
        if inputs["payload"]["operation"] == "produce":
            if envelope is not None:
                raise ValueError("producer has no observation envelope")
            return ()
        fields = {"version", "protocol", "instance_sha256", "source_artifact_ids", "verdict",
                  "worker_cpu_seconds", "worker_wall_seconds"}
        if (type(envelope) is not dict or set(envelope) != fields
                or type(envelope["version"]) is not int or envelope["version"] != 1
                or envelope["protocol"] != inputs["payload"]["protocol"]
                or envelope["instance_sha256"] != _digest(inputs["instance"])
                or envelope["source_artifact_ids"] != [item["artifact_id"] for item in inputs["source_bindings"]]):
            raise ValueError("evaluation envelope metadata changed")
        for key in ("worker_cpu_seconds", "worker_wall_seconds"):
            if type(envelope[key]) not in (int, float) or not math.isfinite(envelope[key]) or envelope[key] < 0:
                raise ValueError("invalid evaluator timing")
        verdict = envelope["verdict"]
        if type(verdict) is not dict:
            raise ValueError("invalid evaluation verdict")
        if verdict.get("status") == "valid":
            if (set(verdict) != {"status", "reason_code", "scheduled_value"}
                    or verdict["reason_code"] != "feasible" or type(verdict["scheduled_value"]) is not int):
                raise ValueError("invalid valid verdict")
        elif verdict.get("status") == "invalid":
            if set(verdict) != {"status", "reason_code"} or verdict["reason_code"] not in INVALID_REASONS:
                raise ValueError("invalid rejection verdict")
        else:
            raise ValueError("missing scientific validity")
        values = {"direction": "maximize", "instance_id": inputs["instance"]["instance_id"],
                  "instance_sha256": envelope["instance_sha256"]}
        if verdict["status"] == "valid":
            values["value"] = verdict["scheduled_value"]
        return ({"name": "scheduled_value", "values": values,
                 "validation": {"status": verdict["status"], "reason_code": verdict["reason_code"]},
                 "comparison_scope": _digest({"schema": "scheduling.comparison.v1",
                    "instance_sha256": envelope["instance_sha256"],
                    "protocol_fingerprint": prepared["observation"]["protocol_fingerprint"]})},)


def _read_fd(fd, maximum, *, readonly=False):
    import fcntl
    import stat
    if type(fd) is not int or fd < 0:
        raise ValueError("invalid scheduling descriptor")
    seals = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
    if fcntl.fcntl(fd, fcntl.F_GET_SEALS) & seals != seals:
        raise ValueError("unsealed scheduling descriptor")
    if readonly and fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY:
        raise ValueError("source descriptor is not read-only")
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= maximum:
        raise ValueError("scheduling descriptor size invalid")
    raw = os.pread(fd, maximum + 1, 0)
    after = os.fstat(fd)
    if len(raw) != before.st_size or (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size):
        raise ValueError("scheduling descriptor changed")
    return raw


def main():
    started_cpu, started_wall = time.process_time(), time.monotonic()
    raw = _read_fd(int(os.environ["ORZE_ACTION_INPUT_FD"]), 65536)
    inputs = _json(raw)
    if type(inputs) is not dict or set(inputs) != {"instance", "payload", "source_bindings"}:
        raise ValueError("invalid scheduling worker inputs")
    instance, payload = validate_instance(inputs["instance"]), _payload(inputs["payload"])
    descriptors = _json(os.environ.get("ORZE_ACTION_SOURCE_FDS", "{}").encode("utf-8"))
    bindings = _bindings(inputs["source_bindings"])
    if type(descriptors) is not dict or set(descriptors) != {item["artifact_id"] for item in bindings}:
        raise ValueError("source descriptor identities changed")
    if payload["operation"] == "produce":
        if bindings:
            raise ValueError("producer must not read candidate sources")
        content = (payload["artifact_utf8"].encode("utf-8") if "artifact_utf8" in payload
                   else _canonical(produce(instance, payload["candidate"])))
        if len(content) > MAX_ARTIFACT_BYTES:
            raise ValueError("candidate output exceeds bound")
        Path("candidate.json").write_bytes(content)
        return
    if len(bindings) != 1:
        raise ValueError("evaluator requires one candidate source")
    source = bindings[0]
    content = _read_fd(descriptors[source["artifact_id"]], MAX_ARTIFACT_BYTES, readonly=True)
    if hashlib.sha256(content).hexdigest() != source["content_sha256"]:
        raise ValueError("candidate source bytes changed")
    # Explicit acceptance-only process fault, never a research request field.
    # It runs after actual source verification and before any complete envelope.
    if os.environ.get("ORZE_HOLDOUT_EVALUATOR_FAULT") == "partial_exit_71":
        Path("evaluation.json").write_bytes(b'{"version":1,"verdict":')
        os._exit(71)
    verdict = evaluate(instance, content, payload["protocol"])
    envelope = {"version": 1, "protocol": payload["protocol"],
        "instance_sha256": _digest(instance), "source_artifact_ids": [source["artifact_id"]],
        "verdict": verdict, "worker_cpu_seconds": max(0.0, time.process_time() - started_cpu),
        "worker_wall_seconds": max(0.0, time.monotonic() - started_wall)}
    encoded = _canonical(envelope)
    if len(encoded) > MAX_ARTIFACT_BYTES:
        raise ValueError("evaluation output exceeds bound")
    Path("evaluation.json").write_bytes(encoded)


if __name__ == "__main__":
    main()
