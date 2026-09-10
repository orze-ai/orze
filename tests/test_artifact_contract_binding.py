"""New launch metadata semantics; no filesystem or execution authority claim."""
from copy import deepcopy

import pytest

from orze.core.artifact_contract import (
    artifact_publication_binding, validate_artifact_publication_binding,
)


def _cfg(root):
    return {"_project_root": str(root), "_orze_dir": ".orze",
            "artifact_contract": {"version": 1, "outputs": {
                "result": {"path": "output.bin", "max_bytes": 128}}}}


def test_binding_is_detached_and_task_or_storage_names_do_not_salt_specification(tmp_path):
    cfg = _cfg(tmp_path)
    a = artifact_publication_binding(cfg, tmp_path / "results" / "task-a", "a" * 64)
    moved = _cfg(tmp_path / "different-project")
    b = artifact_publication_binding(moved, tmp_path / "other-results" / "task-b", "a" * 64)
    assert a["scope"] != b["scope"] and a["root"] != b["root"]
    assert a["spec_fingerprint"] == b["spec_fingerprint"]
    assert a["root"] == str(tmp_path / ".orze" / "artifacts")
    cfg["artifact_contract"]["outputs"]["result"]["path"] = "replacement.bin"
    assert a["contract"]["outputs"]["result"]["path"] == "output.bin"
    assert not list(tmp_path.iterdir()), "binding metadata does not create its storage paths"


def test_execution_or_output_contract_changes_are_distinct_specifications(tmp_path):
    cfg = _cfg(tmp_path)
    folder = tmp_path / "results" / "task-a"
    a = artifact_publication_binding(cfg, folder, "a" * 64)
    b = artifact_publication_binding(cfg, folder, "b" * 64)
    cfg["artifact_contract"]["outputs"]["result"]["max_bytes"] += 1
    c = artifact_publication_binding(cfg, folder, "a" * 64)
    assert len({a["spec_fingerprint"], b["spec_fingerprint"], c["spec_fingerprint"]}) == 3


def test_disabled_legacy_binding_does_not_require_or_invent_execution_identity():
    assert artifact_publication_binding(None, object(), None) is None
    assert artifact_publication_binding({}, object(), None) is None


@pytest.mark.parametrize("damage", ["fingerprint", "path", "contract"])
def test_stored_binding_validation_rejects_changed_types_or_ambiguous_paths(tmp_path, damage):
    original = artifact_publication_binding(_cfg(tmp_path), tmp_path / "results" / "task-a", "a" * 64)
    changed = deepcopy(original)
    if damage == "fingerprint":
        changed["spec_fingerprint"] = True
    elif damage == "path":
        changed["root"] = str(tmp_path) + "/../escape"
    else:
        changed["contract"] = None
    with pytest.raises(ValueError, match="artifact_contract"):
        validate_artifact_publication_binding(changed)
    assert validate_artifact_publication_binding(original) == original
