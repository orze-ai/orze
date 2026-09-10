"""C2d1: native skip policy cannot resolve an unknown action.

Actual source receipts and SQLite action intents; the unknown Popen effect is
an explicit fault injection. These assertions claim a HOLD result, not an
observed second child launch or scientific artifact acceptance.
"""
import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import evaluator
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt

from test_completion_event_authority import accepted
from test_stale_evaluation_completion import project


@pytest.mark.parametrize("skip", ["output_exists", "ineligible"])
def test_native_skip_does_not_hide_a_durable_unclosed_post_script(project, skip):
    p = project
    folder, source = accepted(p)
    p.cfg["post_scripts"] = [{
        "script": "unused-unclosed-post.py", "output": "post-result.json",
    }]
    p.popen.side_effect = RuntimeError("injected unknown subprocess handoff")
    with pytest.raises(AttemptEffectInDoubt):
        evaluator.run_post_scripts(folder.name, 0, p.results, p.cfg,
                                   lake=p.lake, source_event=source)
    row = current_attempt(p.lake.conn, folder.name, "post_script")
    assert row["state"] == "LAUNCHING"
    if skip == "output_exists":
        (folder / "post-result.json").write_text("{}")
    else:
        (folder / "metrics.json").write_text('{"status":"IN_PROGRESS"}')
    before = p.popen.call_count

    with pytest.raises(AttemptEffectBusy):
        evaluator.run_post_scripts(folder.name, 0, p.results, p.cfg,
                                   lake=p.lake, source_event=source)

    assert current_attempt(p.lake.conn, folder.name, "post_script") == row
    assert not (folder / "_compute_receipts" / row["attempt_id"] / "terminal.json").exists()
    assert p.popen.call_count == before
    p.reaper.assert_not_called()


def test_native_existing_output_without_pending_action_remains_a_policy_skip(project):
    p = project
    folder, source = accepted(p)
    (folder / "post-result.json").write_text("{}")
    p.cfg["post_scripts"] = [{
        "script": "unused-skipped-post.py", "output": "post-result.json",
    }]
    before = p.popen.call_count
    evaluator.run_post_scripts(folder.name, 0, p.results, p.cfg,
                               lake=p.lake, source_event=source)
    assert current_attempt(p.lake.conn, folder.name, "post_script") is None
    assert p.popen.call_count == before
    assert not (folder / "_eval_audit.jsonl").exists()
    p.reaper.assert_not_called()

