"""An unwritable diagnostic marker cannot prevent failed evaluation closure.

The failure is a real temporary parent-path/file conflict. Only external
process and GPU boundaries are doubled; validator, marker IO, IdeaLake, and
compute receipt creation are production code.
"""

import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orze.engine import evaluator
from orze.idea_lake import IdeaLake


@pytest.fixture
def project(tmp_path, monkeypatch):
    results = tmp_path / "results"
    folder = results / "idea-marker-io"
    folder.mkdir(parents=True)
    training_bytes = b'{"status":"COMPLETED","quality":0}'
    checkpoint_bytes = b"finished-training\x00checkpoint\xff"
    (folder / "metrics.json").write_bytes(training_bytes)
    (folder / "checkpoint.pt").write_bytes(checkpoint_bytes)
    lake = IdeaLake(tmp_path / "ideas.db")
    lake.insert(folder.name, "marker IO", "{}", "", status="queued")
    assert lake.reconcile_training_complete(folder.name, "reconcile_test_training")
    cfg = {
        "_project_root": str(tmp_path), "idea_lake_db": str(lake.db_path),
        "eval_script": "unused-evaluator.py", "eval_output": "blocked/report.json",
        "eval_timeout": 60,
        "report": {"primary_metric": "quality", "sort": "ascending",
                   "columns": [{"key": "quality", "source": "metrics.json:quality"}]},
    }

    class FailedProcess:
        pid = None
        returncode = None

        def finish(self):
            if self.returncode is None:
                # Created only by the fake evaluator after real launch. The
                # marker needs a directory at this exact path, so mkdir fails.
                (folder / "blocked").write_bytes(b"not-a-directory")
            self.returncode = 1
            return 1

        def poll(self):
            return self.finish()

        def wait(self, timeout=None):
            assert timeout == cfg["eval_timeout"]
            return self.finish()

    popen = Mock(return_value=FailedProcess())
    gpu_check = Mock()
    lease = Mock(side_effect=lambda *args, **kwargs: nullcontext(()))
    monkeypatch.setattr(evaluator.subprocess, "Popen", popen)
    monkeypatch.setattr(evaluator, "_verify_gpu_free", gpu_check)
    monkeypatch.setattr(evaluator, "gpu_execution_lease", lease)
    monkeypatch.setattr(evaluator, "_terminate_and_reap",
                        Mock(side_effect=AssertionError("No actual process to terminate")))
    try:
        yield SimpleNamespace(
            results=results, folder=folder, idea_id=folder.name, lake=lake,
            cfg=cfg, popen=popen, gpu_check=gpu_check, lease=lease,
            training_bytes=training_bytes, checkpoint_bytes=checkpoint_bytes)
    finally:
        lake.close()


@pytest.mark.parametrize("mode", ["async", "sync"])
def test_marker_parent_file_conflict_does_not_prevent_failed_fsm_and_receipt(project, mode):
    p = project
    if mode == "sync":
        evaluator.run_eval(p.idea_id, 0, p.results, p.cfg, lake=p.lake)
    else:
        ep = evaluator.launch_eval(p.idea_id, 0, p.results, p.cfg, lake=p.lake)
        assert ep is not None
        active = {0: ep}
        assert evaluator.check_active_evals(
            active, p.results, p.cfg, lake=p.lake) == [(p.idea_id, 0)]
        assert active == {}

    p.popen.assert_called_once()
    p.gpu_check.assert_called_once()
    p.lease.assert_called_once_with(0, require_idle=True)
    assert (p.folder / "blocked").read_bytes() == b"not-a-directory"
    assert not (p.folder / "blocked" / "report.json").exists()
    assert p.lake.get_fsm_state(p.idea_id) == "FAILED"
    assert p.lake.get_stage_state(p.idea_id, "evaluation") == "FAILED"
    assert p.lake.get_stage_state(p.idea_id, "training") == "COMPLETE"
    assert (p.folder / "metrics.json").read_bytes() == p.training_bytes
    assert (p.folder / "checkpoint.pt").read_bytes() == p.checkpoint_bytes
    starts = list((p.folder / "_compute_receipts").glob("*/start.json"))
    terminals = list((p.folder / "_compute_receipts").glob("*/terminal.json"))
    assert len(starts) == len(terminals) == 1
    start = json.loads(starts[0].read_text(encoding="utf-8"))
    terminal = json.loads(terminals[0].read_text(encoding="utf-8"))
    assert start["attempt_id"] == terminal["attempt_id"]
    assert start["phase"] == terminal["phase"] == "evaluation"
    assert terminal["outcome"] == "failed"
    assert terminal["return_code"] == 1
