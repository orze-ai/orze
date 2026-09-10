"""An existing compute receipt cannot substitute bool for the observed exit."""
import json

import pytest

from orze.engine import accounting, resume
from test_interruption_prepared_publication import api, regular_case, resume_case


def test_existing_boolean_return_code_is_not_the_integer_exit_receipt(api, regular_case):
    _, results, folder, _, cfg, tp = regular_case
    accounting.record_compute_terminal(tp, folder, "interrupted", "interruption_timeout",
                                       phase="training", return_code=0)
    terminal = folder / "_compute_receipts" / tp.attempt_id / "terminal.json"
    value = json.loads(terminal.read_text())
    value["return_code"] = False
    terminal.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    before = terminal.read_bytes()
    prepared = api.prepare_interruption(tp, results, cfg, "timeout", "SIGTERM", 0)
    with pytest.raises(resume.ResumeValidationError):
        api.publish_interruption(prepared, tp, results, cfg)
    assert terminal.read_bytes() == before, "conflicting prior accounting is preserved, never repaired"
