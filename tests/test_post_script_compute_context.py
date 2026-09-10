"""Real action callers verify receipt bytes against their process context."""
import json

import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import evaluator
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt

from test_completion_event_authority import accepted
from test_stale_evaluation_completion import project


@pytest.mark.parametrize("event", ["start", "terminal"])
def test_post_script_rejects_self_consistent_receipt_for_wrong_gpu_type(project, monkeypatch, event):
    p = project
    folder, source = accepted(p)
    p.cfg["post_scripts"] = [{"script": "unused-context-post.py"}]
    actual_popen = p.popen.side_effect

    def popen(*args, **kwargs):
        child = actual_popen(*args, **kwargs)
        child.returncode = 0
        return child

    p.popen.side_effect = popen
    name = "record_compute_" + event
    original = getattr(evaluator, name)
    corrupted = []

    def record(handle, idea_dir, *args, **kwargs):
        payload = original(handle, idea_dir, *args, **kwargs)
        assert payload["physical_gpu"] == 0
        payload["physical_gpu"] = False
        path = idea_dir / "_compute_receipts" / handle.attempt_id / (event + ".json")
        path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        corrupted.append(path)
        return payload

    monkeypatch.setattr(evaluator, name, record)
    with pytest.raises(AttemptEffectInDoubt):
        evaluator.run_post_scripts(folder.name, 0, p.results, p.cfg, lake=p.lake, source_event=source)

    assert len(corrupted) == 1
    assert current_attempt(p.lake.conn, folder.name, "post_script")["state"] != "TERMINAL"
    assert p.lake.get_fsm_state(folder.name) == "FAILED"
    assert current_attempt(p.lake.conn, folder.name, "evaluation")["attempt_id"] == source.attempt_ref.attempt_id
    assert p.popen.call_count == 2
    p.reaper.assert_not_called()
