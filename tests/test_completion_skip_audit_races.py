"""Draft behavior: a skipped old source must not append to retry diagnostics."""
from pathlib import Path

import pytest

from orze.engine import evaluator
from orze.engine.evaluation_retry import request_evaluation_retry

from test_completion_event_authority import accepted
from test_stale_evaluation_completion import project, _launch


@pytest.mark.parametrize("boundary", ["eligibility", "output_exists"])
def test_old_post_script_skip_cannot_append_diagnostics_after_real_retry(project, monkeypatch, boundary):
    p = project
    folder, event = accepted(p)
    output = folder / "post-result.json"
    output.write_text("{}")
    p.cfg["post_scripts"] = [{"script": "unused-post.py", "output": output.name}]
    captured = {}

    def retry():
        assert request_evaluation_retry(folder.name, p.results, p.cfg, p.lake)["status"] == "evaluation_retry_pending"
        current = _launch(p, folder.name)
        assert current.attempt_id != event.attempt_ref.attempt_id
        captured["process"] = current
        audit = folder / "_eval_audit.jsonl"
        captured["audit"] = audit.read_bytes() if audit.exists() else None
        captured["popen_count"] = p.popen.call_count

    if boundary == "eligibility":
        original = evaluator.is_training_complete_for_downstream
        fired = []

        def eligibility(idea_dir, cfg):
            if fired:
                return original(idea_dir, cfg)
            fired.append(True)
            retry()
            # The real downstream predicate, not a mocked qualification, reads
            # a current incomplete training artifact and returns a skip.
            (folder / "metrics.json").write_text('{"status":"IN_PROGRESS"}')
            return original(idea_dir, cfg)

        monkeypatch.setattr(evaluator, "is_training_complete_for_downstream", eligibility)
    else:
        original = Path.exists
        fired = []

        def exists(path):
            result = original(path)
            if path == output and not fired:
                fired.append(True)
                retry()
            return result

        monkeypatch.setattr(Path, "exists", exists)

    evaluator.run_post_scripts(folder.name, 0, p.results, p.cfg, lake=p.lake, source_event=event)

    assert captured["process"].process.returncode is None
    audit = folder / "_eval_audit.jsonl"
    assert (audit.read_bytes() if audit.exists() else None) == captured["audit"], "old source appended a skip into the new attempt's diagnostics"
    assert p.popen.call_count == captured["popen_count"]
    p.reaper.assert_not_called()
