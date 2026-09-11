"""Explicit, bounded CPU command declarations and artifact binding metadata.

validate_action(value) returns a detached exact version-1 command declaration.
action_fingerprint(action) hashes its canonical specification, including the
command and inline inputs, without adding task/attempt or storage identities.
artifact_binding(cfg, folder, action) pins the existing artifact contract/root
with that action-specific fingerprint; it performs no filesystem writes.

The declaration is bounded to the existing 64 KiB execution JSON contract.
Inputs are inline JSON data, not promises that arbitrary executable bytes,
imports, files or environment dependencies have been captured hermetically.
No shell expansion, process launch, closure, measurement or publication rights
are conferred here. Native execution must supply argv without shell=True and
transport its pinned inline inputs through the sealed worker-only FD boundary.
Output paths remain declared relative files; empty outputs are explicitly valid.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from orze.core.artifact_contract import (
    get_artifact_contract, validate_artifact_publication_binding,
)
from orze.core.execution_attempts import AttemptAuthorityError, _json


_FIELDS = {"version", "adapter", "purpose", "inputs", "command",
           "timeout_seconds", "outputs"}
SPECIFICATION_SCHEMA = "orze.native_cpu_action.v1"


def _fail(reason):
    raise ValueError("cpu_action_contract: " + reason)


def validate_action(value) -> dict:
    """Validate exact declaration/JSON types without execution or path IO."""
    if type(value) is not dict or set(value) != _FIELDS:
        _fail("requires exactly version, adapter, purpose, inputs, command, timeout_seconds and outputs")
    if type(value["version"]) is not int or value["version"] != 1:
        _fail("version must be integer 1")
    if type(value["adapter"]) is not str or value["adapter"] != "command":
        _fail("adapter must be command")
    if type(value["purpose"]) is not str or not value["purpose"].strip():
        _fail("purpose must be nonempty text")
    if type(value["inputs"]) is not dict:
        _fail("inputs must be an inline JSON mapping")
    command = value["command"]
    if (type(command) is not list or not command
            or any(type(item) is not str or not item or "\0" in item for item in command)):
        _fail("command must be a nonempty argv list of nonempty strings without NUL")
    timeout = value["timeout_seconds"]
    if type(timeout) not in (int, float):
        _fail("timeout_seconds must be a positive finite number")
    try:
        valid_timeout = timeout > 0 and math.isfinite(timeout)
    except (OverflowError, ValueError):
        valid_timeout = False
    if not valid_timeout:
        _fail("timeout_seconds must be a positive finite number")
    try:
        detached = json.loads(_json(value))
        contract = get_artifact_contract({"artifact_contract": {
            "version": 1, "outputs": detached["outputs"]}})
    except (AttemptAuthorityError, ValueError, TypeError, UnicodeError,
            OverflowError, RecursionError) as exc:
        raise ValueError("cpu_action_contract: invalid bounded JSON or output contract") from exc
    detached["outputs"] = contract["outputs"]
    return detached


def action_fingerprint(action) -> str:
    """Domain-separated canonical specification hash, not a content proof."""
    normalized = validate_action(action)
    encoded = json.dumps({"specification_schema": SPECIFICATION_SCHEMA,
                          "action": normalized},
                         sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_binding(cfg, folder, action) -> dict:
    """Reuse configured artifact storage with a native-action specification."""
    if type(cfg) is not dict:
        _fail("project configuration must be a mapping")
    normalized = validate_action(action)
    folder = Path(os.path.abspath(folder))
    project_root = Path(os.path.abspath(cfg.get("_project_root") or folder.parent.parent))
    control_root = Path(cfg.get("_orze_dir") or project_root / ".orze")
    if not control_root.is_absolute():
        control_root = project_root / control_root
    return validate_artifact_publication_binding({
        "contract": {"version": 1, "outputs": normalized["outputs"]},
        "root": os.path.abspath(control_root / "artifacts"),
        "scope": str(folder.parent),
        "spec_fingerprint": action_fingerprint(normalized),
    })
