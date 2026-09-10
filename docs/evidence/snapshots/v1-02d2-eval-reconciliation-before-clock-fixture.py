"""G1: recovery must verify seals; unconfirmed termination is not completion.

All files and the real IdeaLake are temporary. Host GPU access, evaluator
creation, and process termination are replaced at their external boundaries.
No real subprocess, provider, GPU lock, or process signal is used.
"""

import hashlib
import json
import subprocess
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orze.engine import evaluator
from orze.engine.sealed import write_sealed_manifest
from orze.idea_lake import IdeaLake


@pytest.fixture
def project(tmp_path, monkeypatch):
    results = tmp_path / "results"
    folder = results / "idea-eval-recovery"
    folder.mkdir(parents=True)
    (folder / "metrics.json").write_bytes(b'{"status":"COMPLETED","quality":1}')
    (folder / "checkpoint.pt").write_bytes(b"completed-training\x00checkpoint\xff")
    lake = IdeaLake(tmp_path / "ideas.db")
    lake.insert(folder.name, "recovery", "{}", "", status="queued")
    assert lake.reconcile_training_complete(folder.name, "reconcile_test_training")
    cfg = {
        "_project_root": str(tmp_path), "idea_lake_db": str(lake.db_path),
        "eval_script": "unused-evaluator.py", "eval_output": "assessment.json",
        "eval_timeout": 60,
        "report": {"primary_metric": "quality", "sort": "ascending",
                   "columns": [{"key": "quality", "source": "assessment.json:quality"}]},
    }
    p = SimpleNamespace(
        results=results, folder=folder, idea_id=folder.name, lake=lake, cfg=cfg,
        training_bytes=(folder / "metrics.json").read_bytes(),
        checkpoint_bytes=(folder / "checkpoint.pt").read_bytes(), log_handles=[],
    )

    class UnreapedProcess:
        pid = None
        returncode = None

        def poll(self):
            return None

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("unused-evaluator.py", timeout)

    p.process = UnreapedProcess()

    def popen(*args, **kwargs):
        p.log_handles.append(kwargs["stdout"])
        return p.process

    p.popen = Mock(side_effect=popen)
    p.gpu_check = Mock()
    p.lease = Mock(side_effect=lambda *args, **kwargs: nullcontext(()))
    p.terminate = Mock()  # The owned child remains unconfirmed after cleanup.
    monkeypatch.setattr(evaluator.subprocess, "Popen", p.popen)
    monkeypatch.setattr(evaluator, "_verify_gpu_free", p.gpu_check)
    monkeypatch.setattr(evaluator, "gpu_execution_lease", p.lease)
    monkeypatch.setattr(evaluator, "_terminate_and_reap", p.terminate)
    try:
        yield p
    finally:
        for handle in p.log_handles:
            handle.close()
        lake.close()


def _assert_training_unchanged(p):
    assert p.lake.get_stage_state(p.idea_id, "training") == "COMPLETE"
    assert (p.folder / "metrics.json").read_bytes() == p.training_bytes
    assert (p.folder / "checkpoint.pt").read_bytes() == p.checkpoint_bytes


def test_existing_valid_output_cannot_reconcile_across_a_changed_sealed_file(project):
    p = project
    output = p.folder / "assessment.json"
    output.write_text('{"status":"COMPLETED","quality":0}', encoding="utf-8")
    output_before = output.read_bytes()
    sealed = p.results / "sealed-evaluator-policy.json"
    sealed.write_bytes(b'{"split":"held-out"}')
    write_sealed_manifest(p.results, {
        str(sealed): hashlib.sha256(sealed.read_bytes()).hexdigest()})
    p.cfg["sealed_files"] = [str(sealed)]
    sealed.write_bytes(b'{"split":"changed"}')

    assert evaluator.launch_eval(
        p.idea_id, 0, p.results, p.cfg, lake=p.lake) is None

    p.popen.assert_not_called()
    p.gpu_check.assert_not_called()
    p.lease.assert_not_called()
    assert p.lake.get_fsm_state(p.idea_id) != "COMPLETE"
    assert p.lake.get_stage_state(p.idea_id, "evaluation") != "COMPLETE"
    _assert_training_unchanged(p)
    assert output.read_bytes() == output_before
    assert not (p.folder / "_compute_receipts").exists()


@pytest.mark.parametrize("mode", ["async", "sync"])
def test_timeout_without_confirmed_exit_keeps_attempt_in_progress_without_terminal_receipt(
        project, mode):
    p = project
    active = None
    if mode == "async":
        ep = evaluator.launch_eval(p.idea_id, 0, p.results, p.cfg, lake=p.lake)
        assert ep is not None
        ep.start_time -= 120  # Deterministic timeout; never wait or use a GPU.
        active = {0: ep}
        with pytest.raises(RuntimeError, match="^evaluation_termination_unconfirmed$"):
            evaluator.check_active_evals(active, p.results, p.cfg, lake=p.lake)
        assert active == {0: ep}
    else:
        with pytest.raises(RuntimeError, match="^evaluation_termination_unconfirmed$"):
            evaluator.run_eval(p.idea_id, 0, p.results, p.cfg, lake=p.lake)

    p.popen.assert_called_once()
    p.gpu_check.assert_called_once()
    p.lease.assert_called_once_with(0, require_idle=True)
    p.terminate.assert_called_once()
    assert p.process.returncode is None
    assert p.lake.get_fsm_state(p.idea_id) == "IN_PROGRESS"
    assert p.lake.get_stage_state(p.idea_id, "evaluation") == "IN_PROGRESS"
    _assert_training_unchanged(p)
    assert not (p.folder / "assessment.json").exists()
    starts = list((p.folder / "_compute_receipts").glob("*/start.json"))
    assert len(starts) == 1
    assert json.loads(starts[0].read_text(encoding="utf-8"))["phase"] == "evaluation"
    assert list((p.folder / "_compute_receipts").glob("*/terminal.json")) == []
