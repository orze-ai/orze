"""D1 new mechanism contracts; missing old APIs are not behavioral reds.

Real receipt publication/reading is used. Reapers are CPU-only callables, never
OS process operations. Explicit publication/fsync faults exercise storage
failure; neither qualification nor receipt validation is mocked.
"""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orze.engine import termination_hold as stop


@pytest.fixture
def execution(tmp_path):
    folder = tmp_path / "idea-stop"
    folder.mkdir()
    process = SimpleNamespace(pid=None, returncode=-9)
    process.poll = lambda: process.returncode
    tp = SimpleNamespace(
        idea_id=folder.name, attempt_id="evaluation-first", gpu=0,
        start_time=100.0, process=process)
    return SimpleNamespace(folder=folder, tp=tp)


def _paths(p):
    folder = p.folder / "_execution_stops" / p.tp.attempt_id
    return folder / "requested.json", folder / "confirmed.json"


def _snapshot(folder):
    snapshot = {}
    for path in sorted(folder.rglob("*")):
        key = str(path.relative_to(folder))
        if path.is_symlink():
            snapshot[key] = ("symlink", os.readlink(path))
        elif path.is_dir():
            snapshot[key] = ("directory",)
        else:
            metadata = path.stat()
            snapshot[key] = ("file", path.read_bytes(), metadata.st_mtime_ns, metadata.st_nlink)
    return snapshot


def _hold_readonly(p):
    before = _snapshot(p.folder)
    with pytest.raises(stop.TerminationUnconfirmed, match="^execution_termination_unconfirmed$"):
        stop.require_no_unconfirmed_stop(p.folder)
    assert _snapshot(p.folder) == before


def _confirm(p):
    assert stop.terminate_execution(
        p.tp, p.folder, phase="evaluation", reaper=lambda *a, **k: True) == -9
    request, confirmed = _paths(p)
    assert request.is_file() and confirmed.is_file()
    return request, confirmed


@pytest.mark.parametrize("reaper_result", [False, None, 1, True],
                         ids=["false", "none", "truthy_integer", "true"])
def test_reaper_success_requires_exact_boolean_true(execution, reaper_result):
    p = execution
    reaper = Mock(return_value=reaper_result)
    if reaper_result is True:
        assert stop.terminate_execution(p.tp, p.folder, phase="evaluation", reaper=reaper) == -9
        stop.require_no_unconfirmed_stop(p.folder)
        assert _paths(p)[1].exists()
    else:
        with pytest.raises(stop.TerminationUnconfirmed, match="^evaluation_termination_unconfirmed$"):
            stop.terminate_execution(p.tp, p.folder, phase="evaluation", reaper=reaper)
        assert p.tp._termination_unconfirmed is True
        assert _paths(p)[0].exists() and not _paths(p)[1].exists()
        _hold_readonly(p)
    reaper.assert_called_once()


@pytest.mark.parametrize("return_code", [True, None, 0.5], ids=["bool", "none", "float"])
def test_true_reaper_does_not_authorize_noninteger_exit(execution, return_code):
    p = execution
    p.tp.process.returncode = return_code
    with pytest.raises(stop.TerminationUnconfirmed):
        stop.terminate_execution(p.tp, p.folder, phase="evaluation", reaper=lambda *a, **k: True)
    assert not _paths(p)[1].exists()
    _hold_readonly(p)


def test_reaper_exception_leaves_a_durable_request_not_confirmation(execution):
    p = execution
    with pytest.raises(stop.TerminationUnconfirmed):
        stop.terminate_execution(
            p.tp, p.folder, phase="evaluation", reaper=Mock(side_effect=OSError("reaper failed")))
    assert _paths(p)[0].exists() and not _paths(p)[1].exists()
    assert p.tp._termination_unconfirmed is True
    _hold_readonly(p)


def test_request_precedes_reaper_and_false_latches_across_ticks_and_restart(execution):
    p = execution
    observations = []

    def reaper(*args, **kwargs):
        request, confirmed = _paths(p)
        payload = json.loads(request.read_text(encoding="utf-8"))
        assert payload["idea_id"] == p.tp.idea_id
        assert payload["attempt_id"] == p.tp.attempt_id
        assert payload["phase"] == "evaluation"
        assert not confirmed.exists()
        _hold_readonly(p)
        observations.append(payload)
        return False

    with pytest.raises(stop.TerminationUnconfirmed):
        stop.terminate_execution(p.tp, p.folder, phase="evaluation", reaper=reaper)
    before = _snapshot(p.folder)
    # An integer leader exit on a later tick cannot trigger rediscovery of a
    # now-smaller process tree. A fresh process descriptor also remains held.
    for tp in (p.tp, SimpleNamespace(**{
            key: value for key, value in vars(p.tp).items() if key != "_termination_unconfirmed"})):
        forbidden = Mock(side_effect=AssertionError("must not rediscover writers"))
        with pytest.raises(stop.TerminationUnconfirmed):
            stop.terminate_execution(tp, p.folder, phase="evaluation", reaper=forbidden)
        forbidden.assert_not_called()
    assert len(observations) == 1
    assert _snapshot(p.folder) == before


@pytest.mark.parametrize("field,value", [
    ("request_sha256", "0" * 64), ("idea_id", "idea-other"),
    ("phase", "training"), ("attempt_id", "evaluation-other"),
    ("return_code", True), ("tree_stopped", 1),
])
def test_restart_requires_exact_request_identity_and_typed_confirmation(execution, field, value):
    p = execution
    request, confirmed = _confirm(p)
    data = json.loads(confirmed.read_text(encoding="utf-8"))
    data[field] = value
    confirmed.write_text(json.dumps(data), encoding="utf-8")
    _hold_readonly(p)
    assert request.exists()


@pytest.mark.parametrize("damage", ["partial", "symlink", "hardlink", "oversize"])
def test_partial_redirected_or_unbounded_receipts_fail_closed_without_repair(execution, damage):
    p = execution
    request, confirmed = _confirm(p)
    if damage == "partial":
        confirmed.rename(p.folder / "retained-confirmation.json")
    elif damage == "symlink":
        saved = p.folder / "retained-confirmation.json"
        confirmed.rename(saved)
        confirmed.symlink_to(saved)
    elif damage == "hardlink":
        os.link(confirmed, p.folder / "confirmation-link.json")
    else:
        confirmed.write_bytes(b" " * 8193)
    _hold_readonly(p)


def test_history_limit_is_bounded_without_deleting_or_scanning_into_extra_attempts(execution, monkeypatch):
    p = execution
    request, _ = _confirm(p)
    assert stop._MAX_STOPS == 1024
    monkeypatch.setattr(stop, "_MAX_STOPS", 1)
    (request.parent.parent / "second-attempt").mkdir()
    _hold_readonly(p)


@pytest.mark.parametrize("failure", ["fsync", "atomic_create"])
def test_storage_failure_never_signals_or_publishes_confirmation(execution, monkeypatch, failure):
    p = execution
    reaper = Mock(return_value=True)
    if failure == "fsync":
        monkeypatch.setattr(stop.os, "fsync", Mock(side_effect=OSError("disk full")))
    else:
        monkeypatch.setattr(stop, "atomic_create", Mock(return_value=False))
    with pytest.raises(stop.TerminationUnconfirmed):
        stop.terminate_execution(p.tp, p.folder, phase="evaluation", reaper=reaper)
    reaper.assert_not_called()
    assert p.tp._termination_unconfirmed is True
    assert not _paths(p)[1].exists()
    _hold_readonly(p)


def test_valid_history_and_missing_legacy_history_are_readonly(execution):
    p = execution
    before = _snapshot(p.folder)
    assert stop.require_no_unconfirmed_stop(p.folder) is None
    assert _snapshot(p.folder) == before
    assert not (p.folder / "_execution_stops").exists()
    request, confirmed = _confirm(p)
    payload = json.loads(confirmed.read_text(encoding="utf-8"))
    assert payload["request_sha256"] == hashlib.sha256(request.read_bytes()).hexdigest()
    before = _snapshot(p.folder)
    for _ in range(2):
        assert stop.require_no_unconfirmed_stop(p.folder) is None
    assert _snapshot(p.folder) == before


def test_deep_invalid_receipt_uses_the_same_closed_authority_exception(execution):
    p = execution
    request, _ = _confirm(p)
    # Below the byte limit, but beyond the JSON decoder recursion limit.
    request.write_bytes(b"[" * 1500 + b"0" + b"]" * 1500)
    _hold_readonly(p)
