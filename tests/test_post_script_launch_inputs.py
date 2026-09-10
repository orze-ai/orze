"""C2d1 input requirements at the real source-bound public action entry.

These are newly specified invalid-budget requirements, not claims of observed
production incidents. The source catalog/receipts/attempt writes are real;
Popen and the already-completed child are explicit existing OS doubles.
"""
import pytest

from orze.core.execution_attempts import current_attempt
from orze.engine import evaluator

from test_completion_event_authority import accepted
from test_stale_evaluation_completion import project


@pytest.mark.parametrize("timeout", [
    float("nan"), float("inf"), True, 0, -1, "60", 10 ** 400,
], ids=["nan", "infinity", "boolean", "zero", "negative", "string", "overflow"])
def test_invalid_native_post_script_budget_is_rejected_before_intent_and_popen(project, timeout):
    p = project
    folder, source = accepted(p)
    p.cfg["post_scripts"] = [{"script": "unused-budget-post.py", "timeout": timeout}]
    original = p.popen.side_effect

    def completed_child(*args, **kwargs):
        child = original(*args, **kwargs)
        child.returncode = 0
        return child

    p.popen.side_effect = completed_child
    before = p.popen.call_count
    error = None
    try:
        evaluator.run_post_scripts(folder.name, 0, p.results, p.cfg,
                                   lake=p.lake, source_event=source)
    except (ValueError, RuntimeError) as exc:
        error = exc

    assert current_attempt(p.lake.conn, folder.name, "post_script") is None, (
        "an invalid post-script execution budget acquired durable action intent")
    assert p.popen.call_count == before, "an invalid budget reached process creation"
    assert error is not None, "the invalid execution budget was silently accepted"
    p.reaper.assert_not_called()
