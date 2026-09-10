"""Popen succeeded even when releasing the parent's GPU lease FD failed.

This public-launch regression is frozen separately from the original D1
training-stop tests. The lease boundary still acquires the real isolated
fixture lease; only its post-yield OS close failure is injected.
"""
import contextlib
import io
import json

import pytest

from orze.engine import launcher
from orze.engine.scheduler import claim
from orze.idea_lake import IdeaLake


@pytest.mark.parametrize("phase", ["training", "posthoc"])
@pytest.mark.parametrize("confirmed", [False, True])
def test_created_child_requires_cleanup_after_lease_exit_failure(
        tmp_path, monkeypatch, phase, confirmed):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    idea_id = "idea-lease-exit"
    lake.insert(idea_id, "Lease exit fixture", "seed: 13", "", status="queued")
    assert claim(idea_id, results, 0, lake=lake)
    idea_dir = results / idea_id
    kind = "train" if phase == "training" else "evaluation"
    (idea_dir / "idea_config.yaml").write_text(f"kind: {kind}\nseed: 13\n")
    train, base, ideas = (tmp_path / name for name in
                          ["train.py", "base.yaml", "ideas.md"])
    train.write_text("# Not executed\n")
    base.write_text("{}\n")
    ideas.write_text("")
    cfg = {"train_script": str(train), "base_config": str(base),
           "ideas_file": str(ideas), "python": "python3"}

    class Child:
        pid = 931452
        returncode = None
        stdin = io.BytesIO()

        def poll(self):
            return self.returncode

    child = Child()
    launches, stops = [], []
    real_lease = launcher.gpu_execution_lease

    @contextlib.contextmanager
    def lease_with_close_failure(*args, **kwargs):
        with real_lease(*args, **kwargs) as fds:
            yield fds
        raise OSError("synthetic parent lease FD close failure")

    def popen(*args, **kwargs):
        launches.append(True)
        return child

    def reap(process, stopped_id, **kwargs):
        assert process is child and stopped_id == idea_id
        stops.append(True)
        child.returncode = -15
        return confirmed

    monkeypatch.setattr(launcher, "gpu_execution_lease", lease_with_close_failure)
    monkeypatch.setattr(launcher, "_verify_gpu_free", lambda *a, **k: None)
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher, "_terminate_and_reap", reap)
    try:
        with pytest.raises(Exception) as error:
            launcher.launch(idea_id, 0, results, cfg, lake=lake)
        assert launches == [True]
        assert stops == [True], "Popen returned: the failed launch still owns effects"
        starts = list(idea_dir.glob("_compute_receipts/*/start.json"))
        terminals = list(idea_dir.glob("_compute_receipts/*/terminal.json"))
        assert len(starts) == 1, "an allocated child is not a zero-GPU rejection"
        assert json.loads(starts[0].read_text())["phase"] == phase
        if confirmed:
            assert isinstance(error.value, OSError)
            assert len(terminals) == 1
            terminal = json.loads(terminals[0].read_text())
            assert terminal["phase"] == phase
            assert terminal["outcome"] == "failed"
            assert terminal["return_code"] == -15
        else:
            assert str(error.value) == f"{phase}_termination_unconfirmed"
            assert terminals == []
        assert (idea_dir / "claim.json").exists()
        assert lake.get_fsm_state(idea_id) == "CLAIMED"
        assert not (idea_dir / "metrics.json").exists()
    finally:
        lake.close()
