"""New strict receipt mechanism, with real accounting and filesystem IO."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from orze.engine import accounting
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
from orze.engine.compute_publication import verify_compute_receipt


@pytest.fixture
def case(tmp_path):
    folder = tmp_path / "idea-compute-receipt"
    folder.mkdir()
    process = SimpleNamespace(idea_id=folder.name, attempt_id="native-A", gpu=0,
        execution_identity="a" * 64, start_time=10, process=SimpleNamespace(pid=42))
    start = accounting.record_compute_start(process, folder, phase="evaluation")
    terminal = accounting.record_compute_terminal(process, folder, "completed",
        "evaluation_validated", phase="evaluation", return_code=0)
    return folder, process, start, terminal


def _verify(case, *, require_start=True):
    folder, process, _, payload = case
    verify_compute_receipt(folder, payload, process=process, phase="evaluation",
        event="terminal", outcome="completed", reason_code="evaluation_validated",
        return_code=0, require_start=require_start)


def _start_path(case):
    return case[0] / "_compute_receipts" / case[1].attempt_id / "start.json"


def test_exact_start_terminal_and_integer_clock_use_immutable_context(case):
    folder, process, start, _ = case
    before = {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    verify_compute_receipt(folder, start, process=process, phase="evaluation",
                          event="start", outcome="started")
    _verify(case)
    assert {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("require_start", [False, True])
def test_missing_start_is_allowed_only_for_explicit_launch_initialization_failure(case, require_start):
    _start_path(case).unlink()
    if require_start:
        with pytest.raises(AttemptEffectInDoubt):
            _verify(case, require_start=True)
    else:
        _verify(case, require_start=False)
    assert not _start_path(case).exists()


@pytest.mark.parametrize("field,wrong", [("started_at_epoch", False),
    ("phase", "training"), ("process_pid", 43), ("started_at_epoch", 11.0)])
def test_even_optional_existing_start_must_match_context(case, field, wrong):
    path = _start_path(case)
    value = json.loads(path.read_text())
    value[field] = wrong
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    before = path.read_bytes()
    with pytest.raises(AttemptEffectInDoubt):
        _verify(case, require_start=False)
    assert path.read_bytes() == before


@pytest.mark.parametrize("damage", ["partial", "oversized", "symlink", "hardlink", "fifo"])
def test_start_receipt_cannot_redirect_block_or_exceed_bounds(case, damage):
    path = _start_path(case)
    before = path.read_bytes()
    if damage == "partial":
        path.write_bytes(b"{")
    elif damage == "oversized":
        path.write_bytes(b" " * 65537)
    elif damage == "hardlink":
        os.link(path, path.with_name("start-copy.json"))
    else:
        path.unlink()
        if damage == "symlink":
            external = case[0].parent / "external.json"
            external.write_bytes(before)
            path.symlink_to(external)
        else:
            os.mkfifo(path)
    with pytest.raises(AttemptEffectInDoubt):
        _verify(case)


def test_start_changed_during_terminal_read_cannot_form_a_mixed_receipt_pair(case, monkeypatch):
    start = _start_path(case)
    terminal = start.with_name("terminal.json")
    real_read = os.read
    changed = []

    def read(fd, amount):
        raw = real_read(fd, amount)
        if not changed and Path(f"/proc/self/fd/{fd}").resolve() == terminal:
            changed.append(True)
            value = json.loads(start.read_text())
            value["process_pid"] = 44
            start.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        return raw

    monkeypatch.setattr(os, "read", read)
    with pytest.raises(AttemptEffectInDoubt):
        _verify(case)
    assert changed


def test_parent_fsync_failure_never_acknowledges_receipt(case, monkeypatch):
    real_fsync = os.fsync
    seen = []

    def sync(fd):
        if Path(f"/proc/self/fd/{fd}").resolve() == case[0]:
            seen.append(True)
            raise OSError("synthetic compute parent durability failure")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", sync)
    with pytest.raises(AttemptEffectInDoubt):
        _verify(case)
    assert seen
