"""Architecture audit fixes: unblock_when enforcement + rejection backedge.

Covers two related fixes to the method-validator subsystem:

- Fix A (problem #3): `unblock_when: {artifact_exists: <path>}` is now
  honored. Previously the field was documented in many validator YAMLs but
  the engine ignored it, forcing manual `.disabled` renames once the
  unblocking method-spec artifact was published.

- Fix B (problem #5b): validator rejections append a JSONL row to
  `results/_validator_rejections.jsonl` so the research role can read
  recent rejections and stop re-proposing the same blocked idea family.
"""
import json
import os
from pathlib import Path

import yaml

from orze.engine.launcher import (
    log_validator_rejection,
    validate_idea_against_method_validators,
)


def _write_validator(vdir: Path, name: str, *, unblock_artifact: str = "",
                     field: str = "continuation_parent",
                     bad_value: str = "idea-02e83b") -> None:
    spec = {
        "name": name,
        "severity": "error",
        "rules": [
            {"field": field, "operator": "not_equals", "value": bad_value},
        ],
    }
    if unblock_artifact:
        spec["unblock_when"] = {"artifact_exists": unblock_artifact}
    (vdir / f"{name}.yaml").write_text(yaml.safe_dump(spec))


def test_unblock_when_artifact_missing_still_rejects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # avoid cwd-relative false positive
    vdir = tmp_path / "results/_validators"
    vdir.mkdir(parents=True)
    _write_validator(
        vdir, "block_x",
        unblock_artifact="results/_methods/x_resolved.yaml",
    )
    err = validate_idea_against_method_validators(
        {"continuation_parent": "idea-02e83b"}, vdir)
    assert err is not None
    assert "block_x" in err


def test_unblock_when_artifact_present_short_circuits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    vdir = tmp_path / "results/_validators"
    mdir = tmp_path / "results/_methods"
    vdir.mkdir(parents=True)
    mdir.mkdir(parents=True)
    _write_validator(
        vdir, "block_y",
        unblock_artifact="results/_methods/y_resolved.yaml",
    )
    (mdir / "y_resolved.yaml").write_text("name: y_resolved\nstatus: resolved\n")
    err = validate_idea_against_method_validators(
        {"continuation_parent": "idea-02e83b"}, vdir)
    assert err is None


def test_unblock_when_resolves_relative_to_validators_dir(tmp_path):
    # Even when cwd is not the repo root, validators_dir.parent.parent
    # should resolve the artifact (orze can be invoked from anywhere).
    vdir = tmp_path / "results/_validators"
    mdir = tmp_path / "results/_methods"
    vdir.mkdir(parents=True)
    mdir.mkdir(parents=True)
    _write_validator(
        vdir, "block_z",
        unblock_artifact="results/_methods/z_resolved.yaml",
    )
    (mdir / "z_resolved.yaml").write_text("name: z_resolved\n")
    cwd = os.getcwd()
    os.chdir("/")
    try:
        err = validate_idea_against_method_validators(
            {"continuation_parent": "idea-02e83b"}, vdir)
        assert err is None
    finally:
        os.chdir(cwd)


def test_rejection_log_writes_row(tmp_path):
    log_validator_rejection(
        tmp_path, "idea-deadbeef", "block_x",
        "validator[block_x]: continuation_parent='idea-02e83b' must not equal 'idea-02e83b'",
        {
            "kind": "train",
            "continuation_parent": "idea-02e83b",
            "lora_path": "",
            "irrelevant_key": "stripped",
        },
    )
    log = tmp_path / "_validator_rejections.jsonl"
    assert log.exists()
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 1
    rec = rows[0]
    assert rec["idea_id"] == "idea-deadbeef"
    assert rec["validator"] == "block_x"
    assert "block_x" in rec["rejection"]
    assert rec["config_summary"]["continuation_parent"] == "idea-02e83b"
    # Stray keys are not included in the summary.
    assert "irrelevant_key" not in rec["config_summary"]
    assert rec["ts"].endswith("Z")


def test_rejection_log_append_does_not_truncate(tmp_path):
    for i in range(3):
        log_validator_rejection(
            tmp_path, f"idea-{i:06d}", "block_x", "reason", {"kind": "train"})
    log = tmp_path / "_validator_rejections.jsonl"
    rows = log.read_text().strip().splitlines()
    assert len(rows) == 3


def test_rejection_log_never_raises(tmp_path):
    # Pass garbage; helper is best-effort.
    log_validator_rejection(
        tmp_path / "nonexistent_subdir", "idea-x", "v", "r", None)
