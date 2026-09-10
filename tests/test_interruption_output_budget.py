"""Existing accounting output cannot enter an unbounded short-lock read."""
import json
from pathlib import Path

import pytest

from orze.engine import accounting, resume
from test_interruption_prepared_publication import (
    api, prepare, publish, regular_case, resume_case, terminal,
)


def test_existing_oversize_compute_receipt_rejected_before_legacy_json_read(api, regular_case, monkeypatch):
    _, _, folder, _, _, tp = regular_case
    accounting.record_compute_terminal(tp, folder, "interrupted", "interruption_timeout",
                                       phase="training", return_code=-15)
    path = terminal(regular_case)
    value = json.loads(path.read_text())
    value["unused_metadata"] = "x" * (64 * 1024)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    before = path.read_bytes()
    real_read_text = Path.read_text

    def read_text(target, *args, **kwargs):
        if target == path:
            pytest.fail("short publication entered unbounded accounting receipt read")
        return real_read_text(target, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    with pytest.raises(resume.ResumeValidationError):
        publish(api, regular_case, prepare(api, regular_case))
    assert path.read_bytes() == before
    assert not (folder / "interruption.json").exists()
