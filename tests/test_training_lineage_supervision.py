"""Native lineage pipe/GO integration, not a kernel-isolation proof.

Actual lineage preparation, nonce pipe, receipt, native launch and tree stop
run on tiny CPU workers. Namespace setup is explicitly replaced by a controlled
worker that emits (or deliberately omits) the nonce; no model training occurs.
"""
import json
import os
from pathlib import Path
import sys
import time

import pytest

from orze.core.model_lineage import ModelLineageError
from orze.core.execution_attempts import current_attempt
from orze.engine import launcher
from test_model_lineage import _config
from test_native_training_tree_completion import cpu_training, native_case


@pytest.mark.parametrize("attests", [True, False], ids=["worker-nonce-after-go", "early-eof-not-timeout"])
def test_lineage_handshake_uses_worker_pid_and_worker_only_writer(cpu_training, tmp_path, monkeypatch, attests):
    c = cpu_training
    c.cfg.update(_config(tmp_path))
    c.cfg["model_lineage"]["attestation_timeout"] = 10
    script = Path(c.cfg["train_script"])
    script.write_text("""
import json, os, time
from pathlib import Path
fd = int(os.environ['ORZE_BOUNDARY_ATTEST_FD'])
attests = os.environ['FIXTURE_ATTESTS'] == 'yes'
if attests:
    os.write(fd, (os.environ['ORZE_BOUNDARY_ATTEST_NONCE'] + '\\n').encode())
os.close(fd)
folder = Path(os.environ['FIXTURE_FOLDER'])
if not attests:
    if os.fork():
        os._exit(0)
    os.setsid()
    time.sleep(20)
    os._exit(0)
(folder / 'model.bin').write_bytes(b'synthetic-not-a-model')
(folder / 'metrics.json').write_text(json.dumps({'status':'COMPLETED'}))
""", encoding="utf-8")
    c.cfg["train_extra_env"].update({"FIXTURE_ATTESTS": "yes" if attests else "no",
                                    "FIXTURE_FOLDER": str(c.folder)})
    # This deliberately does NOT exercise actual mount/network namespaces.
    monkeypatch.setattr(launcher, "_probe_kernel_boundary", lambda **kw: None)
    monkeypatch.setattr(launcher, "_build_isolated_cmd", lambda *a, **kw: [sys.executable, str(script)])
    real_prepare = launcher.prepare_supervised
    handles, received = [], []

    def prepare(*args, **kwargs):
        received.append(kwargs)
        process = real_prepare(*args, **kwargs)
        handles.append(process)
        c.pidfds.append((process.pid, os.pidfd_open(process.pid)))
        assert process.poll() is None
        assert not (c.folder / "model.bin").exists()
        return process

    monkeypatch.setattr(launcher, "prepare_supervised", prepare)
    started_at = time.monotonic()
    try:
        if attests:
            tp = launcher.launch(c.idea, 0, c.results, c.cfg, lake=c.lake)
            c.handles.append(tp)
            assert tp.process.wait(timeout=5) == 0
            assert tp.process.pid == handles[0].binding["worker"]["pid"]
            assert tp.process.pid != handles[0].supervisor_pid
            boundary = json.loads((c.folder / "_compute_receipts" / tp.attempt_id / "boundary.json").read_text())
            assert boundary["payload"]["process_pid"] == tp.process.pid
            start = json.loads((c.folder / "_compute_receipts" / tp.attempt_id / "start.json").read_text())
            assert start["process_pid"] == tp.process.pid
            assert current_attempt(c.lake.conn, c.idea, "training")["state"] == "RUNNING"
        else:
            with pytest.raises(ModelLineageError, match="attestation_invalid"):
                launcher.launch(c.idea, 0, c.results, c.cfg, lake=c.lake)
            assert time.monotonic() - started_at < 3, "EOF must not wait for the 10s attestation timeout"
            row = current_attempt(c.lake.conn, c.idea, "training")
            assert row["state"] == "TERMINAL" and row["terminal"]["outcome"] == "failed"
            assert row["terminal"]["process_tree"]["stop_requested"] is True
            assert row["terminal"]["process_tree"]["wait_proof"] == "ECHILD_WALL"
            assert not (c.folder / "_compute_receipts" / row["attempt_id"] / "boundary.json").exists()
        assert len(received) == 1
        assert len(received[0]["worker_only_fds"]) == 1
        assert received[0]["pass_fds"] == ()  # Accelerator lease is stubbed.
    finally:
        for process in handles:
            process.stop(timeout=3)
