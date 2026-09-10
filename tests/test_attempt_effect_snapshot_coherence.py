"""Reject a mixed receipt read that was never a valid filesystem snapshot."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from orze.core.execution_attempts import AttemptRef
from orze.engine import attempt_effect_receipts as receipts
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt, attempt_effect_lock


def _raw(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


@pytest.mark.parametrize("consumer", ["closed_gate", "next_prepare"])
def test_receipt_scan_rejects_a_pair_that_was_never_consistent(tmp_path, monkeypatch, consumer):
    folder = tmp_path / "idea-mixed"
    first = AttemptRef(folder.name, "evaluation", "attempt-1", 1)
    with attempt_effect_lock(folder) as lease:
        digest_a = receipts.prepare_effect(lease, first, {"revision": "A"})
        receipts.confirm_effect(lease, first, digest_a)
    attempt = folder / "_execution_effects" / first.attempt_id
    prepared, committed = attempt / "prepared.json", attempt / "committed.json"
    declaration = json.loads(prepared.read_bytes())
    confirmation = json.loads(committed.read_bytes())
    # Start invalid: prepared A / confirmation B. Neither edit below creates
    # a matching pair: A/B -> C/B -> C/A. Reading A before and A after the edit
    # must not turn that history into a fictitious closed effect.
    confirmation["prepared_sha256"] = "b" * 64
    raw_b = _raw(confirmation)
    declaration["plan"] = {"revision": "C"}
    raw_c = _raw(declaration)
    digest_c = hashlib.sha256(raw_c).hexdigest()
    assert len({digest_a, digest_c, "b" * 64}) == 3
    confirmation["prepared_sha256"] = digest_a
    seen = []
    open_fd = os.open

    def change_between_reads(path, flags, *args, **kwargs):
        if Path(path) == committed and not flags & os.O_CREAT and not seen:
            prepared.write_bytes(raw_c)
            assert hashlib.sha256(prepared.read_bytes()).hexdigest() != json.loads(
                committed.read_bytes())["prepared_sha256"]
            committed.write_bytes(_raw(confirmation))
            assert hashlib.sha256(prepared.read_bytes()).hexdigest() != json.loads(
                committed.read_bytes())["prepared_sha256"]
            seen.append("in-place pair changed")
        return open_fd(path, flags, *args, **kwargs)

    with attempt_effect_lock(folder) as lease:
        # Acquire the real lock before damage. Its entry history gate must
        # remain enabled; this test targets the subsequent actual consumer.
        committed.write_bytes(raw_b)
        with monkeypatch.context() as faults:
            faults.setattr(os, "open", change_between_reads)
            with pytest.raises(AttemptEffectInDoubt):
                if consumer == "closed_gate":
                    receipts.require_closed_effects(folder)
                else:
                    second = AttemptRef(folder.name, "evaluation", "attempt-2", 2)
                    receipts.prepare_effect(lease, second, {"revision": "next"})
        assert seen == ["in-place pair changed"]
        assert not (folder / "_execution_effects" / "attempt-2").exists()
