"""Tests for F13 cross-host fleet scheduler."""

from unittest.mock import patch

from orze.engine.gpu_slots import (
    _parse_nvidia_smi_stdout,
    is_fleet_eligible_kind,
    pick_least_loaded,
    poll_fleet,
    wrap_remote_cmd,
)


SMI_A = "0, 5, 1000, 32510\n1, 95, 30000, 32510\n"
SMI_B = "0, 50, 15000, 32510\n"


def test_parse_nvidia_smi_stdout():
    parsed = _parse_nvidia_smi_stdout(SMI_A)
    assert parsed[0]["util"] == 5
    assert parsed[0]["free_mib"] == 32510 - 1000
    assert parsed[1]["util"] == 95
    assert parsed[1]["free_mib"] == 32510 - 30000


def test_poll_fleet_local_plus_remote():
    import subprocess as sp

    class FakeProc:
        def __init__(self, stdout, rc=0):
            self.stdout = stdout
            self.returncode = rc
            self.stderr = ""

    def fake_run(cmd, *a, **kw):
        if cmd[0] == "ssh":
            return FakeProc(SMI_B)
        return FakeProc(SMI_A)

    with patch("orze.engine.gpu_slots.subprocess.run", side_effect=fake_run):
        fleet = poll_fleet(["localhost", "host-b"])
    assert "localhost" in fleet
    assert "host-b" in fleet
    assert fleet["host-b"][0]["util"] == 50


def test_pick_least_loaded_picks_lowest_util():
    fleet = {
        "a": {0: {"util": 50, "free_mib": 20000, "total_mib": 32000},
              1: {"util": 90, "free_mib": 5000, "total_mib": 32000}},
        "b": {0: {"util": 10, "free_mib": 15000, "total_mib": 32000}},
    }
    host, gpu = pick_least_loaded(fleet)
    assert (host, gpu) == ("b", 0)


def test_pick_respects_min_free_mib():
    fleet = {
        "a": {0: {"util": 5, "free_mib": 500, "total_mib": 32000}},
        "b": {0: {"util": 20, "free_mib": 10000, "total_mib": 32000}},
    }
    # a has lower util but not enough free VRAM → b wins.
    host, _ = pick_least_loaded(fleet, min_free_mib=1024)
    assert host == "b"


def test_pick_returns_none_when_nothing_fits():
    fleet = {"a": {0: {"util": 5, "free_mib": 100, "total_mib": 32000}}}
    assert pick_least_loaded(fleet, min_free_mib=1024) is None


def test_wrap_remote_cmd_local_passthrough():
    assert wrap_remote_cmd("localhost", ["python", "a.py"]) == ["python", "a.py"]


def test_wrap_remote_cmd_builds_ssh_invocation():
    wrapped = wrap_remote_cmd("node-b", ["python", "a.py", "--x=1"])
    assert wrapped[0] == "ssh"
    assert "node-b" in wrapped
    # every user arg is shell-quoted, so present verbatim or with quotes
    joined = " ".join(wrapped)
    assert "python" in joined and "a.py" in joined


def test_is_fleet_eligible_kind():
    assert is_fleet_eligible_kind("posthoc_eval")
    assert is_fleet_eligible_kind("bundle_combine")
    assert is_fleet_eligible_kind("tta_sweep")
    assert is_fleet_eligible_kind("agg_search")
    assert not is_fleet_eligible_kind("train")
    assert not is_fleet_eligible_kind(None)
