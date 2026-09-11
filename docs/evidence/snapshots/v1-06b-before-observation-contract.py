"""Explicit evaluator IO and observation binding metadata, without file IO.

The JSON adapter records a pinned evaluator's reported validation, not a core
scientific verdict. Legacy configurations remain off. Unsupported combinations
are rejected explicitly instead of dropping existing qualification policies.
"""
from __future__ import annotations

import os
from pathlib import Path
import re

from orze.core.artifact_contract import get_artifact_contract

JSON_OBSERVATION_ADAPTER = "orze.json_observations.v1"
MAX_OBSERVATIONS = 32
MAX_OBSERVATION_OUTPUT_BYTES = 1048576
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _fail(reason):
    raise ValueError("observation_contract: " + reason)


def _text(value, maximum, label):
    if (type(value) is not str or not value or value.strip() != value
            or len(value) > maximum or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        _fail("invalid " + label)
    try:
        if len(value.encode("utf-8")) > maximum:
            _fail(label + " is too long")
    except UnicodeError:
        _fail(label + " is not UTF-8")
    return value


def _ids(value):
    if (type(value) is not list or len(value) > 32
            or any(type(item) is not str or not _TOKEN.fullmatch(item) for item in value)
            or len(set(value)) != len(value)):
        _fail("input artifact IDs must be unique bounded tokens")
    return list(value)


def get_observation_contract(cfg: dict | None) -> dict | None:
    """Return a detached declaration; dynamic source/ownership checks are later."""
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        _fail("project configuration must be a mapping")
    value = cfg.get("observation_contract")
    if value is None:
        return None
    if (type(value) is not dict
            or set(value) != {"version", "adapter", "protocol_id", "inputs", "output"}
            or type(value["version"]) is not int or value["version"] != 1
            or value["adapter"] != JSON_OBSERVATION_ADAPTER):
        _fail("requires version 1 and the explicit JSON observation adapter")
    protocol_id = _text(value["protocol_id"], 256, "protocol_id")
    inputs = value["inputs"]
    if (type(inputs) is not list or len(inputs) > 32
            or any(type(name) is not str or not _NAME.fullmatch(name) for name in inputs)
            or len(set(inputs)) != len(inputs)):
        _fail("inputs must be unique declared logical artifact names")
    try:
        declaration = get_artifact_contract({"artifact_contract": {
            "version": 1, "outputs": {"result": value["output"]}}})
    except ValueError as exc:
        _fail("invalid output declaration: " + str(exc))
    output = declaration["outputs"]["result"]
    if output["max_bytes"] > MAX_OBSERVATION_OUTPUT_BYTES:
        _fail("output exceeds the 1 MiB envelope limit")
    # Current benchmark/bundle/lineage adapters resolve canonical task paths.
    # This first isolated adapter does not silently reinterpret those paths.
    report = cfg.get("report")
    if report is not None and type(report) is not dict:
        _fail("report must be a mapping")
    if report is not None and report.get("benchmark_contract") is not None:
        _fail("benchmark_contract combination is not implemented")
    for key in ("evaluation_bundle", "model_lineage"):
        companion = cfg.get(key)
        if companion is None:
            continue
        if type(companion) is not dict or type(companion.get("enabled", False)) is not bool:
            _fail("invalid " + key + " declaration")
        if companion.get("enabled", False):
            _fail(key + " combination is not implemented")
    managed = cfg.get("managed_run")
    if managed is not None and (type(managed) is not dict
                                or any(item is not False for item in managed.values())):
        _fail("managed_run qualification combination is not implemented")
    return {"version": 1, "adapter": JSON_OBSERVATION_ADAPTER,
            "protocol_id": protocol_id, "inputs": list(inputs), "output": dict(output)}


def validate_observation_publication_binding(binding: dict) -> dict:
    """Metadata-only generic binding; no phase, adapter or domain inference."""
    if type(binding) is not dict or set(binding) != {
            "adapter_id", "protocol_fingerprint", "spec_fingerprint", "scope", "input_artifact_ids"}:
        _fail("invalid publication binding")
    adapter = _text(binding["adapter_id"], 128, "adapter_id")
    for key in ("protocol_fingerprint", "spec_fingerprint"):
        if type(binding[key]) is not str or not _SHA.fullmatch(binding[key]):
            _fail("invalid " + key)
    scope = _text(binding["scope"], 4096, "scope")
    if "\\" in scope or not Path(scope).is_absolute() or os.path.normpath(scope) != scope:
        _fail("scope must be a normalized absolute path")
    return {"adapter_id": adapter, "protocol_fingerprint": binding["protocol_fingerprint"],
            "spec_fingerprint": binding["spec_fingerprint"], "scope": scope,
            "input_artifact_ids": _ids(binding["input_artifact_ids"])}
