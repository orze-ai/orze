"""Actual cluster orphan entry point with a fully synthetic /proc surface."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import orze.engine.cluster as cluster
from orze.engine.termination_hold import TerminationUnconfirmed


@pytest.fixture
def orphan(tmp_path, monkeypatch):
    results = tmp_path / "results"
    task = results / "idea-one"
    task.mkdir(parents=True)
    state = SimpleNamespace(
        results=results, task=task, pid=830001, signals=[],
        args=["python", "train.py", "--results-dir", str(results)],
    )
    original_text = Path.read_text
    original_bytes = Path.read_bytes

    def listdir(path):
        assert str(path) == "/proc", "unexpected enumeration"
        return [str(state.pid)]

    def read_text(path, *args, **kwargs):
        if str(path).startswith("/proc/"):
            assert str(path) == f"/proc/{state.pid}/stat"
            return f"{state.pid} (synthetic child) S 1 830011 830011"
        return original_text(path, *args, **kwargs)

    def read_bytes(path):
        if str(path).startswith("/proc/"):
            assert str(path) == f"/proc/{state.pid}/cmdline"
            return ("\0".join(state.args) + "\0").encode()
        return original_bytes(path)

    def getpgid(pid):
        assert pid == state.pid, "unexpected PID identity probe"
        return pid + 10

    def killpg(pgid, sig):
        assert pgid == state.pid + 10, "unexpected synthetic PGID"
        state.signals.append(("pgid", pgid, sig))

    def kill(pid, sig):
        assert pid == state.pid, "unexpected synthetic PID"
        state.signals.append(("pid", pid, sig))

    os_double = SimpleNamespace(**vars(os))
    os_double.listdir = listdir
    os_double.getpgid = getpgid
    os_double.killpg = killpg
    os_double.kill = kill
    monkeypatch.setattr(cluster, "os", os_double)
    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    return state


def cleanup_or_hold(state):
    try:
        return cluster.kill_orphans(state.results, {"train_script": "train.py"})
    except TerminationUnconfirmed:
        return None


@pytest.mark.parametrize("claim_state", ["no_task_id", "malformed", "bare_pid"])
def test_unverified_orphan_cannot_authorize_process_group_kill(orphan, claim_state):
    if claim_state != "no_task_id":
        orphan.args.extend(["--idea-id", orphan.task.name])
        claim = orphan.task / "claim.json"
        claim.write_text(
            "{not-json" if claim_state == "malformed" else '{"pid": 840001}',
            encoding="utf-8",
        )
        before = claim.read_bytes()
    else:
        before = None

    cleanup_or_hold(orphan)

    assert orphan.signals == [], f"{claim_state} authorized an unowned group"
    if before is not None:
        assert (orphan.task / "claim.json").read_bytes() == before


def test_explicit_task_without_claim_is_left_alone_control(orphan):
    orphan.args.extend(["--idea-id", orphan.task.name])

    cleanup_or_hold(orphan)

    assert orphan.signals == []
    assert not (orphan.task / "claim.json").exists()
