"""C2b draft regression: a started launch failure is not a stale claim.

Actual CPU supervisor/GO/stop and Core accounting/SQLite/reporting are used.
Only the parent's post-GO lineage callback raises a synthetic ordinary error;
there is no fake reaper result, model training, GPU, provider or host scan.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from orze.core import model_lineage
from orze.core.execution_attempts import current_attempt
from orze.engine import launcher
from orze.engine.attempt_effect_receipts import require_closed_effects
from orze.engine.launch_failure_report import report_launch_failure
from orze.engine.termination_hold import require_no_unconfirmed_stop
from test_native_training_tree_completion import cpu_training, native_case


def test_real_started_launch_failure_reports_from_running_once(cpu_training, monkeypatch):
    c = cpu_training
    Path(c.cfg["train_script"]).write_text("pass\n", encoding="utf-8")
    real_prepare = launcher.prepare_supervised
    handles, go_sent = [], []
    failure = RuntimeError("synthetic parent callback failure after actual GO")
    real_reaper = Mock(wraps=launcher._terminate_and_reap)

    def prepare(*args, **kwargs):
        process = real_prepare(*args, **kwargs)
        c.pidfds.append((process.pid, os.pidfd_open(process.pid)))
        handles.append(process)
        real_go = process.start

        def go():
            real_go()
            go_sent.append(process.pid)

        monkeypatch.setattr(process, "start", go)
        return process

    def reject_after_go(context, *, process_pid):
        assert len(handles) == 1 and go_sent == [process_pid]
        assert current_attempt(c.lake.conn, c.idea, "training")["state"] == "RUNNING"
        raise failure

    monkeypatch.setattr(launcher, "prepare_supervised", prepare)
    monkeypatch.setattr(launcher, "_terminate_and_reap", real_reaper)
    monkeypatch.setattr(model_lineage, "receive_model_lineage_attestation", reject_after_go)
    try:
        with pytest.raises(RuntimeError) as raised:
            launcher.launch(c.idea, 0, c.results, c.cfg, lake=c.lake)
        assert raised.value is failure
        assert len(handles) == 1 and go_sent == [handles[0].pid]
        assert real_reaper.call_count == 1
        assert real_reaper.call_args.args[0] is handles[0]
        closure = handles[0].closure_receipt()
        assert closure is not None and closure["wait_proof"] == "ECHILD_WALL"
        ref = failure._orze_launch_attempt_ref
        row = current_attempt(c.lake.conn, c.idea, "training")
        assert row["attempt_id"] == ref.attempt_id and row["state"] == "TERMINAL"
        assert row["terminal"]["outcome"] == "failed"
        assert row["terminal"]["process_tree"] == closure
        assert c.lake.get_fsm_state(c.idea) == "IN_PROGRESS"
        compute = json.loads((c.folder / "_compute_receipts" / ref.attempt_id / "terminal.json").read_bytes())
        assert compute["outcome"] == "failed"
        assert compute["process_pid"] == handles[0].pid
        assert compute["return_code"] == closure["worker_returncode"]
        require_no_unconfirmed_stop(c.folder)
        require_closed_effects(c.folder)

        counts = {}
        report = report_launch_failure(c.lake, c.folder, failure, counts, c.cfg)

        assert report["status"] == "reported", "current started failure must not be classified stale"
        assert counts == {c.idea: 1}
        assert c.lake.get_fsm_state(c.idea) == "FAILED"
        action = current_attempt(c.lake.conn, c.idea, "launch_failure_report")
        assert action["state"] == "TERMINAL"
        assert action["terminal"]["source_attempt"] == asdict(ref)
        assert action["terminal"]["failure_count_after"] == 1
        assert action["terminal"]["repair_status"] == "pending_explicit_action"
        history = c.lake.get_fsm_history(c.idea)
        metrics = (c.folder / "metrics.json").read_bytes()
        duplicate_counts = {}

        duplicate = report_launch_failure(c.lake, c.folder, failure, duplicate_counts, c.cfg)

        assert duplicate["status"] == "duplicate"
        assert duplicate_counts == {c.idea: 1}
        assert current_attempt(c.lake.conn, c.idea, "launch_failure_report") == action
        assert c.lake.get_fsm_history(c.idea) == history
        assert (c.folder / "metrics.json").read_bytes() == metrics
    finally:
        for process in handles:
            process.stop(timeout=3)
