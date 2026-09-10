"""C2c candidate protocol mechanisms using owned, real blocked CPU workers.

No old-API absence is a regression here. The inherited fixture explicitly
restores real supervision, Popen, identities and stop. Only temporary adapter
code, final authorization and actual parent memfd-close seams are interleaved.
The adapter sleeps only when a GO/cleanup test must keep it live for STOP.
"""
from contextlib import contextmanager
import errno
import json
import os
from pathlib import Path
import time

import pytest
import yaml

from orze.core.execution_attempts import current_attempt
from orze.engine import launcher, native_posthoc
from orze.engine.sealed_payload import SealedPayloadError
from orze.engine.supervised_process import SupervisionUncertain
from orze.engine.termination_hold import TerminationUnconfirmed
from test_native_posthoc_tree_completion import cpu_posthoc, cpu_training, native_case


def _adapter(c, tmp_path, *, pause=False):
    marker = tmp_path / "posthoc-adapter-executed"
    site = Path(os.environ["PYTHONPATH"].split(os.pathsep)[0]) / "sitecustomize.py"
    site.write_text(
        "from pathlib import Path\nimport time\n"
        "from orze.engine.posthoc_runner import register_adapter\n"
        "@register_adapter('owned_cpu_posthoc')\n"
        "def adapter(idea_id, cfg, idea_dir):\n"
        "    Path(" + repr(str(marker)) + ").write_text('executed')\n"
        + ("    time.sleep(30)\n" if pause else "")
        + "    return {'score': 0}\n", encoding="utf-8")
    (c.folder / "idea_config.yaml").write_text(
        yaml.safe_dump(c.adapter_config), encoding="utf-8")
    return marker


def _remember(c, handles, process):
    c.pidfds.append((process.pid, os.pidfd_open(process.pid)))
    handles.append(process)
    assert process.poll() is None


def _no_publication(c):
    assert not (c.folder / "metrics.json").exists()
    assert not list((Path(c.cfg["_orze_dir"]) / "artifacts").glob("*/content"))


def _closed_failed(c, process, *, task_state):
    process.wait(timeout=5)
    closure = process.closure_receipt()
    assert closure["stop_requested"] is True
    assert closure["wait_proof"] == "ECHILD_WALL"
    row = current_attempt(c.lake.conn, c.idea, "posthoc")
    assert row["state"] == "TERMINAL"
    assert row["terminal"]["outcome"] == "failed"
    assert row["terminal"]["process_tree"] == closure
    assert row["terminal"]["return_code"] == closure["worker_returncode"]
    assert c.lake.get_fsm_state(c.idea) == task_state
    starts = list(c.folder.glob("_compute_receipts/*/start.json"))
    terminals = list(c.folder.glob("_compute_receipts/*/terminal.json"))
    assert len(starts) == len(terminals) == 1
    receipt = json.loads(terminals[0].read_text())
    assert receipt["phase"] == "posthoc"
    assert receipt["process_pid"] == process.pid
    assert receipt["outcome"] == "failed"
    assert receipt["return_code"] == closure["worker_returncode"]
    _no_publication(c)


@pytest.mark.parametrize("change", ["config", "runtime"])
def test_posthoc_ready_does_not_authorize_changed_launch(cpu_posthoc, tmp_path, monkeypatch, change):
    c = cpu_posthoc
    marker = _adapter(c, tmp_path)
    real_prepare = launcher.prepare_supervised
    real_attest = launcher._assert_controller_runtime_attested
    handles, attest_rejections = [], []

    def prepare(*args, **kwargs):
        process = real_prepare(*args, **kwargs)
        _remember(c, handles, process)
        assert not marker.exists()
        if change == "config":
            value = dict(c.adapter_config, changed_after_ready=True)
            (c.folder / "idea_config.yaml").write_text(yaml.safe_dump(value), encoding="utf-8")
        return process

    def attest(cfg):
        if handles and change == "runtime":
            attest_rejections.append(True)
            raise launcher.LaunchIntegrityError("fixture posthoc runtime rejected after READY")
        return real_attest(cfg)

    monkeypatch.setattr(launcher, "prepare_supervised", prepare)
    monkeypatch.setattr(launcher, "_assert_controller_runtime_attested", attest)
    try:
        with pytest.raises(launcher.LaunchIntegrityError):
            launcher.launch(c.idea, 0, c.results, c.cfg, lake=c.lake)
        assert len(handles) == 1
        process = handles[0]
        process.wait(timeout=5)
        assert not marker.exists()
        closure = process.closure_receipt()
        assert closure["stop_requested"] is True
        assert closure["wait_proof"] == "ECHILD_WALL"
        if change == "runtime":
            assert attest_rejections == [True]
        row = current_attempt(c.lake.conn, c.idea, "posthoc")
        assert row["state"] == "RUNNING"
        assert row["terminal"] is None
        assert not list(c.folder.glob("_compute_receipts/*/terminal.json"))
        _no_publication(c)
    finally:
        for process in handles:
            process.stop(timeout=3)


def test_posthoc_constructor_fault_stops_ready_owner_before_failed_launch(cpu_posthoc, tmp_path, monkeypatch):
    c = cpu_posthoc
    marker = _adapter(c, tmp_path)
    real_prepare = launcher.prepare_supervised
    handles = []

    def prepare(*args, **kwargs):
        process = real_prepare(*args, **kwargs)
        _remember(c, handles, process)
        return process

    def construct(**kwargs):
        assert kwargs["process"] is handles[0]
        assert not marker.exists()
        raise ValueError("fixture posthoc constructor failed")

    monkeypatch.setattr(launcher, "prepare_supervised", prepare)
    monkeypatch.setattr(launcher, "TrainingProcess", construct)
    try:
        with pytest.raises(ValueError, match="fixture posthoc constructor failed"):
            launcher.launch(c.idea, 0, c.results, c.cfg, lake=c.lake)
        assert len(handles) == 1
        assert not marker.exists()
        _closed_failed(c, handles[0], task_state="CLAIMED")
    finally:
        for process in handles:
            process.stop(timeout=3)


def _fail_payload_close(monkeypatch):
    actual_payload = native_posthoc.sealed_payload
    actual_close = os.close
    captured, calls = {}, []

    @contextmanager
    def payload(raw):
        with actual_payload(raw) as fd:
            captured["fd"] = fd
            yield fd

    def close(fd):
        if fd == captured.get("fd") and not calls:
            calls.append(fd)
            actual_close(fd)
            raise OSError(errno.EIO, "fixture close completed but confirmation failed")
        return actual_close(fd)

    monkeypatch.setattr(native_posthoc, "sealed_payload", payload)
    monkeypatch.setattr(os, "close", close)
    return calls


def test_posthoc_payload_close_fault_after_go_requires_confirmed_stop(cpu_posthoc, tmp_path, monkeypatch):
    c = cpu_posthoc
    marker = _adapter(c, tmp_path, pause=True)
    real_prepare = launcher.prepare_supervised
    handles = []
    closes = _fail_payload_close(monkeypatch)

    def prepare(*args, **kwargs):
        process = real_prepare(*args, **kwargs)
        _remember(c, handles, process)
        real_start = process.start

        def start():
            real_start()
            deadline = time.monotonic() + 5
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert marker.exists(), "real user adapter must execute before the parent close fault"

        monkeypatch.setattr(process, "start", start)
        return process

    monkeypatch.setattr(launcher, "prepare_supervised", prepare)
    try:
        with pytest.raises(SealedPayloadError, match="sealed_payload_close_failed"):
            launcher.launch(c.idea, 0, c.results, c.cfg, lake=c.lake)
        assert len(handles) == len(closes) == 1
        assert marker.exists()
        _closed_failed(c, handles[0], task_state="IN_PROGRESS")
    finally:
        for process in handles:
            process.stop(timeout=3)


def test_posthoc_ready_uncertainty_survives_secondary_payload_close_fault(cpu_posthoc, tmp_path, monkeypatch):
    c = cpu_posthoc
    marker = _adapter(c, tmp_path)
    real_prepare = launcher.prepare_supervised
    handles = []
    closes = _fail_payload_close(monkeypatch)

    def prepare(*args, **kwargs):
        process = real_prepare(*args, **kwargs)
        _remember(c, handles, process)
        assert not marker.exists()
        raise SupervisionUncertain("fixture posthoc READY response lost", process=process)

    monkeypatch.setattr(launcher, "prepare_supervised", prepare)
    try:
        with pytest.raises(TerminationUnconfirmed, match="posthoc_supervision_unconfirmed"):
            launcher.launch(c.idea, 0, c.results, c.cfg, lake=c.lake)
        assert len(handles) == len(closes) == 1
        assert not marker.exists()
        row = current_attempt(c.lake.conn, c.idea, "posthoc")
        assert row["state"] == "LAUNCHING"
        assert row["terminal"] is None
        assert c.lake.get_fsm_state(c.idea) == "CLAIMED"
        assert not list(c.folder.glob("_compute_receipts/*/terminal.json"))
        assert handles[0].closure_receipt() is None
        _no_publication(c)
    finally:
        for process in handles:
            process.stop(timeout=3)
