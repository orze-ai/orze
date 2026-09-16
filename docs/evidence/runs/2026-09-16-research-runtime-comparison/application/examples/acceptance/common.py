"""Example-application transport only; not a second executor or Core API.

Dataset digests hash canonical JSON (compression uses an explicit hex string).
Only prepare/claim run in the controller; worker helpers use the sealed FDs
provided by the existing native adapter. This module has no Orze import.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

OUTPUTS = {"result": {"path": "result.json", "max_bytes": 16384}}
CANDIDATES = ("baseline", "challenger", "unchecked")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def prepare(request, sources, *, dataset, worker_path, adapter_id, protocol):
    payload = request["payload"]
    if (type(payload) is not dict or set(payload) != {"candidate", "operation"}
            or payload["candidate"] not in CANDIDATES
            or payload["operation"] not in ("measure", "analyze")
            or request["inputs"] != {} or request["outputs"] != OUTPUTS
            or request["timeout_seconds"] != 2):
        raise ValueError("acceptance request does not match the frozen protocol")
    if payload["operation"] == "analyze" and payload["candidate"] != "unchecked":
        raise ValueError("this finite protocol analyzes the unchecked candidate")
    expected = 3 if payload["operation"] == "analyze" else 0
    if len(sources) != expected:
        raise ValueError("acceptance request requires its exact source set")
    bindings = [{"artifact_id": item["artifact_id"], "content_sha256": item["content_sha256"]}
                for item in sources]
    if request["input_artifact_ids"] != [item["artifact_id"] for item in bindings]:
        raise ValueError("acceptance source order changed")
    inputs = {**payload, "dataset": dataset, "source_bindings": bindings}
    return {"action": {"version": 1, "adapter": "command",
        "purpose": request["purpose"], "inputs": inputs,
        "command": [sys.executable, str(Path(worker_path).resolve())],
        "timeout_seconds": request["timeout_seconds"], "outputs": request["outputs"]},
        "observation": {"adapter_id": adapter_id,
            "spec_fingerprint": digest({"schema": "acceptance.subject.v1",
                "candidate": payload["candidate"], "dataset": dataset}),
            "protocol_fingerprint": digest({"schema": "acceptance.protocol.v1", "protocol": protocol}),
            "result_output": "result"}}


def _read_descriptor(fd, maximum):
    if type(fd) is not int or fd < 0:
        raise ValueError("invalid sealed descriptor")
    raw = os.pread(fd, maximum + 1, 0)
    if len(raw) > maximum:
        raise ValueError("sealed example input exceeds its declared bound")
    return raw


def read_inputs():
    value = json.loads(_read_descriptor(int(os.environ["ORZE_ACTION_INPUT_FD"]), 65536))
    if type(value) is not dict or set(value) != {"dataset", "candidate", "operation", "source_bindings"}:
        raise ValueError("invalid example worker input")
    return value


def read_source_results(inputs):
    bindings = inputs["source_bindings"]
    descriptors = json.loads(os.environ.get("ORZE_ACTION_SOURCE_FDS", "{}"))
    expected = [item["artifact_id"] for item in bindings]
    if (type(descriptors) is not dict or len(set(expected)) != len(expected)
            or set(descriptors) != set(expected)):
        raise ValueError("source descriptors do not match the captured bindings")
    result = {}
    for binding in bindings:
        identity = binding["artifact_id"]
        raw = _read_descriptor(descriptors[identity], OUTPUTS["result"]["max_bytes"])
        if hashlib.sha256(raw).hexdigest() != binding["content_sha256"]:
            raise ValueError("actual source bytes differ from captured content")
        value = json.loads(raw)
        if type(value) is not dict:
            raise ValueError("source is not a result envelope")
        result[identity] = value
    return result


def write_result(inputs, cost, details, *, started_cpu, started_wall):
    value = {"version": 1, "candidate": inputs["candidate"], "operation": inputs["operation"],
        "dataset_sha256": digest(inputs["dataset"]), "cost": cost, "details": details,
        "source_artifact_ids": [item["artifact_id"] for item in inputs["source_bindings"]],
        "worker_cpu_seconds": max(0.0, time.process_time() - started_cpu),
        "worker_wall_seconds": max(0.0, time.monotonic() - started_wall)}
    raw = canonical(value)
    if len(raw) > OUTPUTS["result"]["max_bytes"]:
        raise ValueError("result exceeds frozen output bound")
    Path("result.json").write_bytes(raw)
    return value


def claim(prepared, envelope, *, status, reason_code):
    inputs = prepared["action"]["inputs"]
    fields = {"version", "candidate", "operation", "dataset_sha256", "cost", "details",
              "source_artifact_ids", "worker_cpu_seconds", "worker_wall_seconds"}
    if (type(envelope) is not dict or set(envelope) != fields
            or type(envelope["version"]) is not int or envelope["version"] != 1
            or envelope["candidate"] != inputs["candidate"]
            or envelope["operation"] != inputs["operation"]
            or envelope["dataset_sha256"] != digest(inputs["dataset"])
            or envelope["source_artifact_ids"] != [b["artifact_id"] for b in inputs["source_bindings"]]
            or type(envelope["cost"]) is not int or envelope["cost"] < 0
            or type(envelope["details"]) is not dict):
        raise ValueError("result metadata does not match its actual prepared action")
    for key in ("worker_cpu_seconds", "worker_wall_seconds"):
        if (type(envelope[key]) not in (int, float)
                or not math.isfinite(envelope[key]) or envelope[key] < 0):
            raise ValueError("invalid worker timing")
    if status not in ("valid", "invalid", "unknown"):
        raise ValueError("domain must explicitly classify its result")
    values = {key: envelope[key] for key in ("cost", "candidate", "operation", "dataset_sha256",
                                             "worker_cpu_seconds", "worker_wall_seconds")}
    return ({"name": "objective", "values": values,
        "validation": {"status": status, "reason_code": reason_code},
        "comparison_scope": digest({"schema": "acceptance.comparison.v1",
            "protocol": prepared["observation"]["protocol_fingerprint"],
            "dataset_sha256": envelope["dataset_sha256"]})},)
