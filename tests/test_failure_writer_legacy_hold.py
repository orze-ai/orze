"""D2 draft: a retained legacy publication owner also fences bare writers."""
import pytest

from orze.engine import evaluator, launcher
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt, attempt_effect_lock
from orze.engine.termination_hold import TerminationUnconfirmed
from test_stale_evaluation_completion import _files


@pytest.mark.parametrize("phase", ["training", "evaluation"])
def test_bare_failure_writer_cannot_ignore_retained_legacy_owner(tmp_path, phase):
    folder = tmp_path / "idea-legacy-held"
    folder.mkdir()
    (folder / "metrics.json").write_text('{"status":"IN_PROGRESS","step":3}')
    with pytest.raises(AttemptEffectInDoubt):
        with attempt_effect_lock(folder):
            (folder / "partial-diagnostic.txt").write_text("publication interrupted")
            raise AttemptEffectInDoubt("partial legacy publication")
    before = _files(folder)
    try:
        if phase == "training":
            launcher._write_failure(folder, "late failure")
        else:
            evaluator._write_eval_failure_marker(tmp_path, folder.name, "assessment.json", "late failure")
    except TerminationUnconfirmed:
        pass
    assert _files(folder) == before
    assert (folder / "_attempt_effect.lock").is_dir()
