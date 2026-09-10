"""C2c draft resource-boundary regressions; real tiny CPU launch if admitted."""
import pytest
import yaml

from orze.core.execution_attempts import current_attempt
from orze.engine import launcher
from test_native_posthoc_tree_completion import cpu_posthoc, cpu_training, native_case


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), True, 0, -1, "60"],
                         ids=["nan", "infinite", "bool", "zero", "negative", "string"])
def test_invalid_posthoc_budget_rejects_before_intent_or_actual_process(cpu_posthoc, timeout):
    c = cpu_posthoc
    c.cfg["posthoc_timeout"] = timeout
    (c.folder / "idea_config.yaml").write_text(
        yaml.safe_dump({"kind": "posthoc_eval", "adapter": "null"}))
    before_claim = (c.folder / "claim.json").read_bytes()
    before_fsm = c.lake.get_fsm_state(c.idea)
    tp, error = None, None
    try:
        try:
            tp = launcher.launch(c.idea, 0, c.results, c.cfg, lake=c.lake)
            c.handles.append(tp)
        except launcher.LaunchIntegrityError as exc:
            error = exc
        assert c.roots == [], "invalid execution limit already created a real subprocess"
        assert error is not None and str(error) == "posthoc_timeout_invalid"
        assert current_attempt(c.lake.conn, c.idea, "posthoc") is None
        assert (c.folder / "claim.json").read_bytes() == before_claim
        assert c.lake.get_fsm_state(c.idea) == before_fsm
        assert not (c.folder / "_compute_receipts").exists()
    finally:
        # The pre-fix null adapter is a finite real CPU action. Reap exactly
        # its returned supervisor/tree; no metrics are accepted by this test.
        if tp is not None:
            tp.process.wait(timeout=5)
