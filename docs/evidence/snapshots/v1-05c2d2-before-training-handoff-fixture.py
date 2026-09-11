"""Public phase handoff must retain execution ownership when stop is unknown.

Real launch/claim/Lake/compute receipts; only process, GPU, provider, and the
registration failure boundary are simulated. No new API absence is a red.
"""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from orze.engine import launcher, phases, process
from orze.idea_lake import IdeaLake


class FakeChild:
    pid = 987654301
    # An exited leader alone must not count as a fully stopped execution.
    returncode = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


class RefusingActive(dict):
    def __setitem__(self, key, value):
        raise RuntimeError("injected slot registration refusal")


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    train, base, inbox = (tmp_path / name for name in
                          ("train.py", "base.yaml", "ideas.md"))
    train.write_text("# no execution in this fixture\n", encoding="utf-8")
    base.write_text("{}\n", encoding="utf-8")
    inbox.write_text("# Ideas\n", encoding="utf-8")
    cfg = {
        "results_dir": str(results), "train_script": str(train),
        "base_config": str(base), "ideas_file": str(inbox),
        "idea_lake_db": str(tmp_path / "ideas.db"),
        "_project_root": str(tmp_path), "_orze_dir": str(tmp_path / ".orze"),
        "sealed_files": [], "gpu_mem_threshold": 2000,
        "sweep": {}, "gc": {}, "timeout": 60,
        "artifact_preflight": {"enabled": False},
    }
    lake = IdeaLake(cfg["idea_lake_db"])
    lake.insert("idea-handoff", "Fixture", "seed: 1\n", "fixture", status="queued")
    runner = SimpleNamespace(
        cfg=cfg, results_dir=results, active_evals={}, active={}, gpu_ids=[4],
        lake=lake, failure_counts={}, fix_counts={},
    )
    child = FakeChild()
    popen = Mock(return_value=child)
    fixer = Mock(return_value=False)
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    capture_identity = launcher.capture_process_identity

    def simulated_child_identity(pid):
        if pid == child.pid:
            return {"pid": pid, "pgid": pid, "start_ticks": 123456}
        return capture_identity(pid)

    monkeypatch.setattr(launcher, "capture_process_identity", simulated_child_identity)
    monkeypatch.setattr(launcher, "_verify_gpu_free", lambda *args: None)
    monkeypatch.setattr(phases, "get_gpu_memory_used", lambda gpu: 0)
    monkeypatch.setattr(phases, "run_pre_script", lambda *args: True)
    monkeypatch.setattr(phases, "_try_executor_fix", fixer)
    # No real provider can be reached through optional proposal hooks.
    monkeypatch.setattr("orze.extensions.get_extension", lambda name: None)
    from supervision_fixture import install_training
    install_training(monkeypatch)
    try:
        yield runner, child, popen, fixer
    finally:
        lake.close()


@pytest.mark.parametrize("stop_confirmed", [False, True])
@pytest.mark.parametrize("boundary", ["initialization", "slot_registration"])
def test_phase_never_releases_unknown_started_execution(
        scenario, monkeypatch, boundary, stop_confirmed):
    runner, child, popen, fixer = scenario
    stops = []

    def controlled_stop(proc, *args, **kwargs):
        assert proc is child
        stops.append(proc.pid)
        if stop_confirmed:
            proc.returncode = -15
        return stop_confirmed

    monkeypatch.setattr(launcher, "_terminate_and_reap", controlled_stop)
    monkeypatch.setattr(process, "_terminate_and_reap", controlled_stop)
    from supervision_fixture import adapt_training_reaper
    adapt_training_reaper(monkeypatch, launcher)
    adapt_training_reaper(monkeypatch, process)
    if boundary == "initialization":
        def reject_attestation(*args, **kwargs):
            raise RuntimeError("injected post-Popen initialization failure")
        monkeypatch.setattr(
            "orze.core.model_lineage.receive_model_lineage_attestation",
            reject_attestation)
    else:
        runner.active = RefusingActive()

    raised = None
    try:
        phases.OrzePhaseMixin._launch_training(
            runner, ["idea-handoff"], True,
            {"idea-handoff": {"title": "Fixture", "config": {"seed": 1}}})
    except RuntimeError as exc:
        raised = exc

    assert popen.call_count == 1, "the real launch must reach exactly one Popen boundary"
    assert stops == [child.pid], "the actual post-launch cleanup seam must execute"
    idea_dir = runner.results_dir / "idea-handoff"
    starts = list((idea_dir / "_compute_receipts").glob("*/start.json"))
    terminals = list((idea_dir / "_compute_receipts").glob("*/terminal.json"))
    assert len(starts) == 1
    if not stop_confirmed:
        assert terminals == [], "unconfirmed cleanup cannot mint a terminal allocation receipt"
        assert (idea_dir / "claim.json").exists()
        assert runner.lake.get_fsm_state("idea-handoff") in {"CLAIMED", "IN_PROGRESS"}
        assert not (idea_dir / "metrics.json").exists()
        assert runner.failure_counts == {}
        fixer.assert_not_called()
        assert raised is not None, "unknown post-Popen effects must stop ordinary phase dispatch"
    else:
        assert raised is None
        assert len(terminals) == 1
        terminal = json.loads(terminals[0].read_text(encoding="utf-8"))
        assert terminal["return_code"] == -15
        if boundary == "initialization":
            assert terminal["outcome"] == "failed"
            assert runner.lake.get_fsm_state("idea-handoff") == "FAILED"
            # Native repair is now a separate explicit action; this is the
            # documented D2 fixture migration, not another historical red.
            fixer.assert_not_called()
            from orze.core.execution_attempts import current_attempt
            report = current_attempt(runner.lake.conn, "idea-handoff", "launch_failure_report")
            assert report["terminal"]["repair_status"] == "pending_explicit_action"
        else:
            assert terminal["outcome"] == "requeued"
            assert runner.lake.get_fsm_state("idea-handoff") == "QUEUED"
            assert not (idea_dir / "claim.json").exists()
            fixer.assert_not_called()
