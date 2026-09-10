"""Actual native CPU launch: READY does not renew a stale entrypoint grant."""
from pathlib import Path

from orze.engine import evaluator
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt
from orze.engine.termination_hold import TerminationUnconfirmed
from test_native_eval_tree_completion import (
    cpu_project, b2_project, artifact_project, native_case,
)


def test_ready_wait_does_not_authorize_changed_copied_entrypoint(cpu_project, tmp_path, monkeypatch):
    c = cpu_project
    marker = tmp_path / "unauthorized-user-code"
    Path(c.cfg["eval_script"]).write_text("# original authorized evaluator\n", encoding="utf-8")
    c.cfg["eval_args"] = []
    real_prepare = evaluator.prepare_supervised
    ready = []

    def prepare(cmd, **kwargs):
        process = real_prepare(cmd, **kwargs)
        assert process.poll() is None and not marker.exists()
        copied = Path(cmd[1])
        assert copied.name == "entrypoint.py" and "_evaluation_attempts" in copied.parts
        replacement = copied.with_name("replacement-entrypoint.py")
        replacement.write_text(
            "from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('executed')\n",
            encoding="utf-8")
        replacement.replace(copied)
        ready.append(process)
        return process

    monkeypatch.setattr(evaluator, "prepare_supervised", prepare)
    try:
        try:
            evaluator.launch_eval(c.idea, 0, c.results, c.cfg, lake=c.lake,
                                  source_event=c.source_event)
        except (AttemptEffectBusy, AttemptEffectInDoubt, TerminationUnconfirmed):
            pass
        assert len(ready) == 1, "must cross actual READY, not reject an unused API"
        # A corrected launcher stops its known blocked process on revalidation
        # failure. The candidate executes the changed script and returns zero.
        ready[0].wait(timeout=5)
        assert not marker.exists(), "GO executed a copied script changed after its final validation"
    finally:
        for process in ready:
            process.stop(timeout=3)
