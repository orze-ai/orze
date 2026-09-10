"""READY-to-GO attestation rejection with a real blocked CPU worker.

The attestation function is the explicit fault boundary; this does not claim
to alter installed packages or a real controller runtime pin.
"""
from pathlib import Path

from orze.engine import evaluator
from test_native_eval_tree_completion import (
    cpu_project, b2_project, artifact_project, native_case,
)


def test_go_attestation_rejection_preserves_error_and_confirms_blocked_worker_stop(
        cpu_project, tmp_path, monkeypatch):
    c = cpu_project
    marker = tmp_path / "runtime-rejected-user-code"
    Path(c.cfg["eval_script"]).write_text(
        "from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('executed')\n",
        encoding="utf-8")
    c.cfg["eval_args"] = []
    real_prepare = evaluator.prepare_supervised
    real_attest = evaluator._assert_controller_runtime_attested
    ready, attestations = [], []

    def prepare(*args, **kwargs):
        process = real_prepare(*args, **kwargs)
        assert process.poll() is None and not marker.exists()
        ready.append(process)
        return process

    def attest(cfg):
        attestations.append(bool(ready))
        if ready:
            raise evaluator.LaunchIntegrityError("injected_ready_runtime_rejection")
        return real_attest(cfg)

    monkeypatch.setattr(evaluator, "prepare_supervised", prepare)
    monkeypatch.setattr(evaluator, "_assert_controller_runtime_attested", attest)
    observed = None
    try:
        try:
            evaluator.launch_eval(c.idea, 0, c.results, c.cfg, lake=c.lake,
                                  source_event=c.source_event)
        except evaluator.LaunchIntegrityError as exc:
            observed = exc
        assert len(ready) == 1
        ready[0].wait(timeout=5)
        assert isinstance(observed, evaluator.LaunchIntegrityError), (
            "actual GO skipped the now-rejecting runtime attestation", attestations)
        assert str(observed) == "injected_ready_runtime_rejection"
        assert attestations[0] is False and attestations[-1] is True
        assert ready[0].closure_receipt()["stop_requested"] is True
        assert not marker.exists()
    finally:
        for process in ready:
            process.stop(timeout=3)
