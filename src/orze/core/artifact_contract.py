"""Explicit file-output contracts and their launch-time binding; no file I/O.

An artifact occurrence is separate from its content hash. These declarations
authorize neither execution nor scientific validity. Legacy callers remain
disabled unless they opt into the versioned contract before launch.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re

MAX_ARTIFACT_OUTPUTS = 32
MAX_ARTIFACT_BYTES = 2**40
MAX_CONTRACT_BYTES = 16384
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _fail(reason):
    raise ValueError("artifact_contract: " + reason)


def _encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def get_artifact_contract(cfg: dict | None) -> dict | None:
    """Normalize a detached bounded declaration, or None for explicit legacy."""
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        _fail("project configuration must be a mapping")
    value = cfg.get("artifact_contract")
    if value is None:
        return None
    if (type(value) is not dict or set(value) != {"version", "outputs"}
            or type(value["version"]) is not int or value["version"] != 1
            or type(value["outputs"]) is not dict
            or len(value["outputs"]) > MAX_ARTIFACT_OUTPUTS):
        _fail("requires version 1 and at most 32 named outputs")
    outputs, total = {}, 0
    for name, item in value["outputs"].items():
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            _fail("invalid logical output name")
        if type(item) is not dict or set(item) != {"path", "max_bytes"}:
            _fail("each output requires only path and max_bytes")
        path, maximum = item["path"], item["max_bytes"]
        if (not isinstance(path, str) or not path or len(path) > 1024
                or "\0" in path or "\\" in path):
            _fail("invalid relative output path")
        try:
            if len(path.encode("utf-8")) > 1024:
                _fail("output path is too long")
        except UnicodeError:
            _fail("output path is not UTF-8")
        parsed = PurePosixPath(path)
        if (parsed.is_absolute() or path in {".", ".."}
                or ".." in parsed.parts or parsed.as_posix() != path):
            _fail("output path must be canonical and task-relative")
        if type(maximum) is not int or not 1 <= maximum <= MAX_ARTIFACT_BYTES:
            _fail("max_bytes must be a positive integer at most 1 TiB")
        total += maximum
        outputs[name] = {"path": path, "max_bytes": maximum}
    normalized = {"version": 1, "outputs": outputs}
    if total > MAX_ARTIFACT_BYTES or len(_encoded(normalized)) > MAX_CONTRACT_BYTES:
        _fail("aggregate artifact or contract metadata budget exceeded")
    return normalized


def _absolute(value):
    if not isinstance(value, str) or not value or len(value) > 4096 or "\0" in value:
        _fail("invalid absolute binding path")
    try:
        if len(value.encode("utf-8")) > 4096:
            _fail("binding path is too long")
    except UnicodeError:
        _fail("binding path is not UTF-8")
    if not Path(value).is_absolute() or os.path.normpath(value) != value:
        _fail("binding path must be normalized and absolute")
    return value


def validate_artifact_publication_binding(binding: dict) -> dict:
    """Validate metadata only; filesystem ownership belongs to publication."""
    if type(binding) is not dict or set(binding) != {"contract", "root", "scope", "spec_fingerprint"}:
        _fail("invalid publication binding")
    contract = get_artifact_contract({"artifact_contract": binding["contract"]})
    fingerprint = binding["spec_fingerprint"]
    if contract is None or not isinstance(fingerprint, str) or not _SHA.fullmatch(fingerprint):
        _fail("publication requires a declared contract and specification fingerprint")
    return {"contract": contract, "root": _absolute(binding["root"]),
            "scope": _absolute(binding["scope"]), "spec_fingerprint": fingerprint}


def artifact_publication_binding(cfg, idea_dir, execution_identity) -> dict | None:
    """Pin the output contract and adapter-defined specification before Popen.

    The legacy training adapter fingerprints its existing semantic execution
    identity plus this output contract. Task/attempt IDs and storage locations
    are absent; an explicit future replication does not need fingerprint salt.
    """
    contract = get_artifact_contract(cfg)
    if contract is None:
        return None
    if not isinstance(execution_identity, str) or not _SHA.fullmatch(execution_identity):
        _fail("a declared artifact requires the launch execution identity")
    idea_dir = Path(os.path.abspath(idea_dir))
    project_root = Path(os.path.abspath(cfg.get("_project_root") or idea_dir.parent.parent))
    control_root = Path(cfg.get("_orze_dir") or project_root / ".orze")
    if not control_root.is_absolute():
        control_root = project_root / control_root
    fingerprint = hashlib.sha256(_encoded({
        "specification_schema": "orze.legacy_training_artifacts.v1",
        "execution_identity": execution_identity, "artifact_contract": contract,
    })).hexdigest()
    return validate_artifact_publication_binding({
        "contract": contract, "root": os.path.abspath(control_root / "artifacts"),
        "scope": str(idea_dir.parent), "spec_fingerprint": fingerprint,
    })
