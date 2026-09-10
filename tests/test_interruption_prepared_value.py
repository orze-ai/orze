"""Value ownership and clock normalization for the new prepare/publish API."""
import json

import pytest

from orze.engine import resume
from test_interruption_prepared_publication import (
    api, prepare, publish, regular_case, resume_case, terminal,
)


def test_integer_start_clock_uses_accounting_float_normalization(api, regular_case):
    regular_case[5].start_time = 0
    payload = publish(api, regular_case, prepare(api, regular_case))
    assert payload["resume_eligible"] is True
    receipt = json.loads(terminal(regular_case).read_text())
    assert type(receipt["started_at_epoch"]) is float
    assert receipt["started_at_epoch"] == 0.0


def test_clock_changed_after_prepare_cannot_publish(api, regular_case):
    prepared = prepare(api, regular_case)
    regular_case[5].start_time += 1
    with pytest.raises(resume.ResumeValidationError):
        publish(api, regular_case, prepared)
    assert not (regular_case[2] / "interruption.json").exists()
    assert not terminal(regular_case).exists()


def test_deserialized_payload_is_not_a_mutable_prepared_alias(api, regular_case):
    prepared = prepare(api, regular_case)
    edited_copy = json.loads(prepared.payload_json)
    edited_copy["checkpoint"]["sha256"] = "0" * 64
    edited_copy["resume_eligible"] = False
    payload = publish(api, regular_case, prepared)
    assert payload["resume_eligible"] is True
    assert payload["checkpoint"]["sha256"] != "0" * 64
    payload["checkpoint"]["sha256"] = "changed returned payload"
    assert json.loads(prepared.payload_json)["checkpoint"]["sha256"] != "changed returned payload"
    assert json.loads((regular_case[2] / "interruption.json").read_text())["checkpoint"]["sha256"] != "changed returned payload"
