"""New explicit CPU action contract, not historical missing-API regressions."""
from copy import deepcopy
import hashlib
import json

import pytest

from orze.core.cpu_action_contract import (
    action_fingerprint, artifact_binding, validate_action,
)


def action():
    return {"version": 1, "adapter": "command", "purpose": "Build a summary",
            "inputs": {"count": 2, "options": {"enabled": True, "note": "雪"}},
            "command": ["python3", "worker.py", "literal;argument"],
            "timeout_seconds": 3.5,
            "outputs": {"summary": {"path": "summary.json", "max_bytes": 128}}}


def test_valid_action_is_detached_and_preserves_explicit_inline_inputs():
    raw = action()
    normalized = validate_action(raw)
    assert normalized == raw and normalized is not raw
    raw["inputs"]["options"]["note"] = "changed"
    raw["command"].append("later")
    raw["outputs"]["summary"]["path"] = "later.json"
    assert normalized["inputs"]["options"]["note"] == "雪"
    assert normalized["command"] == ["python3", "worker.py", "literal;argument"]
    assert normalized["outputs"]["summary"]["path"] == "summary.json"


@pytest.mark.parametrize("field,value", [
    ("version", True), ("version", 2), ("adapter", "shell"),
    ("purpose", "  "), ("inputs", []), ("command", []),
    ("command", "python3 worker.py"), ("command", ["python3", ""]),
    ("command", ["python3", 3]), ("command", ["python3", "bad\0arg"]),
    ("timeout_seconds", True), ("timeout_seconds", 0),
    ("timeout_seconds", -1), ("timeout_seconds", float("inf")),
    ("timeout_seconds", float("nan")),
    ("outputs", {"escape": {"path": "../escape", "max_bytes": 1}}),
])
def test_strict_declaration_rejects_invalid_fields(field, value):
    candidate = action()
    candidate[field] = value
    with pytest.raises(ValueError, match="cpu_action_contract"):
        validate_action(candidate)


@pytest.mark.parametrize("mutation", ["missing", "extra", "nonmapping"])
def test_exact_keyset_does_not_infer_action_defaults(mutation):
    candidate = action()
    if mutation == "missing":
        del candidate["inputs"]
    elif mutation == "extra":
        candidate["shell"] = True
    else:
        candidate = None
    with pytest.raises(ValueError, match="cpu_action_contract"):
        validate_action(candidate)


@pytest.mark.parametrize("payload", [float("nan"), object(), "x" * 65536])
def test_inline_inputs_must_be_bounded_finite_json(payload):
    candidate = action()
    candidate["inputs"] = {"value": payload}
    with pytest.raises(ValueError, match="cpu_action_contract"):
        validate_action(candidate)


def test_recursive_inline_inputs_are_rejected_without_mutation():
    candidate = action()
    candidate["inputs"]["self"] = candidate["inputs"]
    with pytest.raises(ValueError, match="cpu_action_contract"):
        validate_action(candidate)
    assert candidate["inputs"]["self"] is candidate["inputs"]


def test_canonical_fingerprint_binds_command_inputs_and_output_contract():
    candidate = action()
    reordered = dict(reversed(list(candidate.items())))
    assert action_fingerprint(candidate) == action_fingerprint(reordered)
    expected = json.dumps({"specification_schema": "orze.native_cpu_action.v1",
                           "action": validate_action(candidate)},
                          sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    assert action_fingerprint(candidate) == hashlib.sha256(expected).hexdigest()
    fingerprints = {action_fingerprint(candidate)}
    for field, value in [("purpose", "Different purpose"), ("inputs", {}),
                         ("command", ["python3", "another.py"]),
                         ("timeout_seconds", 4), ("outputs", {})]:
        changed = deepcopy(candidate)
        changed[field] = value
        fingerprints.add(action_fingerprint(changed))
    assert len(fingerprints) == 6


def test_storage_and_task_locations_do_not_salt_action_specification(tmp_path):
    candidate = action()
    first = artifact_binding({"_project_root": str(tmp_path), "_orze_dir": "control"},
                             tmp_path / "results" / "one", candidate)
    second = artifact_binding({"_project_root": str(tmp_path / "other")},
                              tmp_path / "different-results" / "two", candidate)
    assert first["root"] == str(tmp_path / "control" / "artifacts")
    assert first["scope"] == str(tmp_path / "results")
    assert first["root"] != second["root"] and first["scope"] != second["scope"]
    assert first["spec_fingerprint"] == second["spec_fingerprint"] == action_fingerprint(candidate)
    candidate["outputs"].clear()
    assert first["contract"]["outputs"]["summary"]["path"] == "summary.json"
    assert list(tmp_path.iterdir()) == []


def test_empty_outputs_are_explicit_and_keep_native_action_binding(tmp_path):
    candidate = action()
    candidate["outputs"] = {}
    candidate["timeout_seconds"] = 1
    publication = artifact_binding({}, tmp_path / "results" / "action-a", candidate)
    assert publication["contract"] == {"version": 1, "outputs": {}}
    assert publication["spec_fingerprint"] == action_fingerprint(candidate)
    assert validate_action(candidate)["timeout_seconds"] == 1
