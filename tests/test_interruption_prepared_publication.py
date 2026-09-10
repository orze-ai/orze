"""New short-publication API contracts; absent APIs are skips, not old reds."""
import importlib
import importlib.util
import json
import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from orze.engine import accounting, resume
from test_resume import resume_case


@pytest.fixture
def api():
    name = "orze.engine.interruption_publication"
    if importlib.util.find_spec(name) is None:
        pytest.skip("new prepare/publish capability not present in baseline")
    return importlib.import_module(name)


@pytest.fixture
def regular_case(resume_case):
    project, results, idea_dir, checkpoint, cfg, tp = resume_case
    progress = json.loads((idea_dir / "progress.json").read_text())
    progress["checkpoint_path"] += "/model.bin"
    (idea_dir / "progress.json").write_text(json.dumps(progress))
    tp.attempt_id = "native-interruption-1"
    tp.execution_identity = "a" * 64
    accounting.record_compute_start(tp, idea_dir)
    return project, results, idea_dir, checkpoint / "model.bin", cfg, tp


def prepare(api, case, reason="timeout"):
    _, results, _, _, cfg, tp = case
    return api.prepare_interruption(tp, results, cfg, reason, "SIGTERM", -15)


def publish(api, case, prepared):
    _, results, _, _, cfg, tp = case
    return api.publish_interruption(prepared, tp, results, cfg)


def terminal(case):
    return case[2] / "_compute_receipts" / case[5].attempt_id / "terminal.json"


def test_legacy_wrapper_still_supports_directory_checkpoint(resume_case):
    _, results, idea_dir, _, cfg, tp = resume_case
    receipt = resume.write_interruption_receipt(tp, results, cfg, "timeout", "SIGTERM", -15)
    assert receipt["resume_eligible"] is True
    assert receipt["checkpoint"]["kind"] == "directory"
    assert json.loads((idea_dir / "interruption.json").read_text()) == receipt


def test_prepare_is_frozen_readonly_and_publish_does_not_rehash(api, regular_case, monkeypatch):
    before = {str(p): p.read_bytes() for p in regular_case[0].rglob("*") if p.is_file()}
    with monkeypatch.context() as scope:
        scope.setattr(Path, "mkdir", lambda *a, **k: pytest.fail("prepare mkdir"))
        scope.setattr(resume, "atomic_write", lambda *a, **k: pytest.fail("prepare write"))
        scope.setattr(accounting, "record_compute_terminal", lambda *a, **k: pytest.fail("prepare accounting"))
        prepared = prepare(api, regular_case)
    assert before == {str(p): p.read_bytes() for p in regular_case[0].rglob("*") if p.is_file()}
    with pytest.raises(FrozenInstanceError):
        prepared.payload_json = "{}"
    monkeypatch.setattr(resume, "_hash_path", lambda *a, **k: pytest.fail("publisher large hash"))
    monkeypatch.setattr(resume, "_receipt_contract", lambda *a, **k: pytest.fail("publisher contract"))
    receipt = publish(api, regular_case, prepared)
    assert receipt["resume_eligible"] is True
    assert receipt["checkpoint"]["kind"] == "file"
    assert json.loads((regular_case[2] / "interruption.json").read_text()) == receipt
    assert json.loads(terminal(regular_case).read_text())["reason_code"] == "interruption_timeout"


@pytest.mark.parametrize("target", ["checkpoint", "input", "progress", "config", "script"])
def test_attested_file_drift_rejects_before_any_publication(api, regular_case, target):
    project, _, idea_dir, checkpoint, _, _ = regular_case
    prepared = prepare(api, regular_case)
    path = {"checkpoint": checkpoint, "input": project / "dataset.lock",
            "progress": idea_dir / "progress.json", "config": idea_dir / "idea_config.yaml",
            "script": project / "train.py"}[target]
    original = path.stat()
    path.write_bytes(path.read_bytes() + b"changed")
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    with pytest.raises(resume.ResumeValidationError):
        publish(api, regular_case, prepared)
    assert not (idea_dir / "interruption.json").exists()
    assert not terminal(regular_case).exists()


@pytest.mark.parametrize("change", ["attempt", "policy", "results"])
def test_preparation_is_bound_to_scope_and_execution(api, regular_case, change):
    prepared = prepare(api, regular_case)
    case = list(regular_case)
    if change == "attempt":
        case[5].attempt_id = "replacement-attempt"
    elif change == "policy":
        case[4]["resume"]["args"].append("--different")
    else:
        case[1] = case[0] / "other-results"
    with pytest.raises(resume.ResumeValidationError):
        publish(api, case, prepared)
    assert not (regular_case[2] / "interruption.json").exists()


@pytest.mark.parametrize("unsupported", ["checkpoint_directory", "input_directory", "input_limit"])
def test_unsupported_resume_does_not_block_stop_or_hash_directory(api, regular_case, monkeypatch, unsupported):
    project, _, idea_dir, checkpoint, cfg, _ = regular_case
    if unsupported == "checkpoint_directory":
        progress = json.loads((idea_dir / "progress.json").read_text())
        progress["checkpoint_path"] = str(checkpoint.parent)
        (idea_dir / "progress.json").write_text(json.dumps(progress))
    elif unsupported == "input_directory":
        cfg["resume"]["immutable_inputs"] = [str(checkpoint.parent)]
    else:
        cfg["resume"]["immutable_inputs"] = [str(project / f"input-{i}") for i in range(65)]
    monkeypatch.setattr(resume, "_receipt_contract", lambda *a, **k: pytest.fail("unsupported slow contract"))
    prepared = prepare(api, regular_case)
    receipt = publish(api, regular_case, prepared)
    assert receipt["resume_eligible"] is False
    assert "unsupported" in receipt["resume_reason"]
    assert "checkpoint" not in receipt
    assert json.loads(terminal(regular_case).read_text())["outcome"] == "interrupted"


def test_disabled_resume_requires_no_contract_input(api, regular_case, monkeypatch):
    regular_case[4]["resume"]["enabled"] = False
    (regular_case[2] / "progress.json").unlink()
    monkeypatch.setattr(resume, "_receipt_contract", lambda *a, **k: pytest.fail("disabled contract"))
    receipt = publish(api, regular_case, prepare(api, regular_case))
    assert receipt["resume_eligible"] is False
    assert receipt["resume_reason"] == "resume_policy_disabled"


def test_mutation_during_slow_contract_never_attests_mixed_input(api, regular_case, monkeypatch):
    original = resume._receipt_contract
    def change(*args, **kwargs):
        payload = original(*args, **kwargs)
        regular_case[3].write_bytes(b"after contract")
        return payload
    monkeypatch.setattr(resume, "_receipt_contract", change)
    prepared = prepare(api, regular_case)
    receipt = publish(api, regular_case, prepared)
    assert receipt["resume_eligible"] is False
    assert "changed" in receipt["resume_reason"]


def test_legitimate_effect_directory_creation_does_not_invalidate_inputs(api, regular_case):
    prepared = prepare(api, regular_case)
    (regular_case[2] / "_effect_guard").mkdir()
    (regular_case[2] / "_effect_guard" / "intent.json").write_text("{}")
    assert publish(api, regular_case, prepared)["resume_eligible"] is True


def test_silent_interruption_write_is_not_success(api, regular_case, monkeypatch):
    prepared = prepare(api, regular_case)
    monkeypatch.setattr(resume, "atomic_write", lambda *a, **k: None)
    with pytest.raises((OSError, resume.ResumeValidationError)):
        publish(api, regular_case, prepared)
    assert not terminal(regular_case).exists()


def test_terminal_write_shortfall_is_not_success(api, regular_case, monkeypatch):
    prepared = prepare(api, regular_case)
    real_write = os.write
    def short_write(fd, data):
        if Path(f"/proc/self/fd/{fd}").resolve() == terminal(regular_case):
            return real_write(fd, data[:1])
        return real_write(fd, data)
    monkeypatch.setattr(os, "write", short_write)
    with pytest.raises((OSError, resume.ResumeValidationError)):
        publish(api, regular_case, prepared)
    assert (regular_case[2] / "interruption.json").exists()
    assert terminal(regular_case).read_bytes() == b"{"


def test_parent_fsync_failure_is_not_success(api, regular_case, monkeypatch):
    prepared = prepare(api, regular_case)
    real_fsync = os.fsync
    def broken(fd):
        if Path(f"/proc/self/fd/{fd}").resolve() == regular_case[2]:
            raise OSError("injected parent sync failure")
        return real_fsync(fd)
    monkeypatch.setattr(os, "fsync", broken)
    with pytest.raises((OSError, resume.ResumeValidationError)):
        publish(api, regular_case, prepared)


def test_output_leaf_symlink_cannot_write_outside(api, regular_case):
    outside = regular_case[0] / "outside.txt"
    outside.write_bytes(b"preserve")
    prepared = prepare(api, regular_case)
    (regular_case[2] / "interruption.json").symlink_to(outside)
    with pytest.raises(resume.ResumeValidationError):
        publish(api, regular_case, prepared)
    assert outside.read_bytes() == b"preserve"


def test_nonfreeform_reason_mapping_is_preserved(api, regular_case):
    receipt = publish(api, regular_case, prepare(api, regular_case, reason="arbitrary reason"))
    assert receipt["reason"] == "arbitrary reason"
    compute = json.loads(terminal(regular_case).read_text())
    assert compute["reason_code"] == "interruption_other"
    assert "arbitrary reason" not in terminal(regular_case).read_text()
