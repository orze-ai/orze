"""New explicit status/read-only gate mechanisms, not missing-API old reds."""
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest

from orze.core.control_outcome import (
    ControllerStopHOLD, STOP_SENTINELS, StopOutcome,
    require_controller_start_allowed,
)


def test_explicit_status_is_frozen_metadata_not_implicit_stop_authority():
    for status in ("requested", "hold", "confirmed"):
        result = StopOutcome(status, "controller_stop_unconfirmed")
        assert asdict(result) == {"status": status, "reason_code": "controller_stop_unconfirmed"}
        with pytest.raises(FrozenInstanceError):
            result.status = "confirmed"


def test_status_is_an_exact_known_string():
    for status in (None, True, "done", "CONFIRMED"):
        with pytest.raises(ValueError, match="stop_outcome_status_invalid"):
            StopOutcome(status, "test_reason")


def test_reason_is_a_bounded_stable_token():
    for reason in (None, True, "", "Bad-reason", "r" * 65):
        with pytest.raises(ValueError, match="stop_outcome_reason_invalid"):
            StopOutcome("requested", reason)
    assert StopOutcome("hold", "r" * 64).reason_code == "r" * 64


def test_no_status_label_can_be_used_as_a_boolean():
    for status in ("requested", "hold", "confirmed"):
        with pytest.raises(TypeError, match="no implicit truth value"):
            bool(StopOutcome(status, "test_reason"))


def test_absent_markers_are_read_only_and_missing_results_are_not_created(tmp_path):
    ordinary = tmp_path / "ordinary"
    ordinary.write_bytes(b"unchanged")
    before = tmp_path.stat(), ordinary.stat()
    assert require_controller_start_allowed(tmp_path) is None
    missing = tmp_path / "not-created"
    assert require_controller_start_allowed(missing) is None
    assert not missing.exists()
    assert ordinary.read_bytes() == b"unchanged"
    after = tmp_path.stat(), ordinary.stat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_mtime_ns, item.st_ctime_ns)
    assert [identity(item) for item in before] == [identity(item) for item in after]


@pytest.mark.parametrize("name", STOP_SENTINELS)
def test_every_existing_marker_refuses_without_deleting_or_interpreting_contents(tmp_path, name):
    marker = tmp_path / name
    marker.write_bytes(b"old or malformed is not closure evidence\n")
    before = marker.stat()
    with pytest.raises(ControllerStopHOLD, match="controller_start_blocked_by_sentinel"):
        require_controller_start_allowed(tmp_path)
    after = marker.stat()
    assert marker.read_bytes() == b"old or malformed is not closure evidence\n"
    assert (before.st_ino, before.st_mtime_ns, before.st_ctime_ns) == (
        after.st_ino, after.st_mtime_ns, after.st_ctime_ns)


def test_dangling_marker_is_present_not_a_missing_target(tmp_path):
    marker = tmp_path / ".orze_stop_all"
    marker.symlink_to(tmp_path / "missing-target")
    with pytest.raises(ControllerStopHOLD, match="controller_start_blocked_by_sentinel"):
        require_controller_start_allowed(tmp_path)
    assert marker.is_symlink() and not marker.exists()


def test_unreadable_marker_probe_refuses_without_treating_it_as_absent(tmp_path, monkeypatch):
    original = Path.lstat
    calls = []
    def unreadable(path, *args, **kwargs):
        calls.append(path.name)
        if path == tmp_path / ".orze_stop_all":
            raise PermissionError("controlled permission fault")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "lstat", unreadable)
    with pytest.raises(ControllerStopHOLD, match="controller_stop_state_unverifiable"):
        require_controller_start_allowed(tmp_path)
    assert calls == [".orze_disabled", ".orze_stop_all"]
