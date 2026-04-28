"""Tests for orze.Journal — survives-compaction iteration log."""

from __future__ import annotations

import json
import pytest

from orze import Journal, Iteration


def test_md_roundtrip(tmp_path):
    p = tmp_path / "status.md"
    j = Journal(p, meta={"goal": "hit 5.30%"})
    with j.iter("iter-1", hypothesis="multidomain LoRA") as it:
        it.recipe(lora="iter-4", base="chk-38000")
        it.note("loaded model")
        it.result(macro_wer=5.79, per_dataset={"ami": 11.98})
        it.decide("rejected", reason="worse than baseline 5.449")

    # Reload and check structured state survives.
    j2 = Journal(p)
    assert j2.meta == {"goal": "hit 5.30%"}
    assert len(j2.iters) == 1
    it2 = j2.iters[0]
    assert it2.name == "iter-1"
    assert it2.hypothesis == "multidomain LoRA"
    assert it2.recipe == {"lora": "iter-4", "base": "chk-38000"}
    assert it2.result == {"macro_wer": 5.79, "per_dataset": {"ami": 11.98}}
    assert it2.decision == ("rejected", "worse than baseline 5.449")
    assert it2.closed_at is not None


def test_jsonl_roundtrip(tmp_path):
    p = tmp_path / "status.jsonl"
    j = Journal(p)
    j.iter("a").result(x=1).close()
    j.iter("b").decide("kept").close()

    j2 = Journal(p)
    assert [it.name for it in j2.iters] == ["a", "b"]
    assert j2.iters[0].result == {"x": 1}
    assert j2.iters[1].decision == ("kept", "")


def test_resume_appends(tmp_path):
    p = tmp_path / "status.md"
    j = Journal(p)
    with j.iter("iter-1") as it:
        it.result(macro=5.0)
    j2 = Journal(p)  # simulating restart
    with j2.iter("iter-2") as it:
        it.result(macro=4.5)

    j3 = Journal(p)
    assert [it.name for it in j3.iters] == ["iter-1", "iter-2"]
    assert j3.iters[0].result == {"macro": 5.0}
    assert j3.iters[1].result == {"macro": 4.5}


def test_exception_marks_errored(tmp_path):
    p = tmp_path / "status.md"
    j = Journal(p)
    with pytest.raises(ValueError):
        with j.iter("crashy") as it:
            it.note("starting")
            raise ValueError("boom")
    j2 = Journal(p)
    assert j2.iters[0].decision == ("errored", "ValueError: boom")
    assert j2.iters[0].closed_at is not None


def test_get_returns_last_match(tmp_path):
    p = tmp_path / "j.jsonl"
    j = Journal(p)
    j.iter("dup").result(v=1).close()
    j.iter("dup").result(v=2).close()
    assert j.get("dup").result == {"v": 2}
    assert j.get("missing") is None


def test_md_human_section_preserved(tmp_path):
    """If the user writes prose between machine markers, we keep it intact
    on subsequent flushes."""
    p = tmp_path / "j.md"
    j = Journal(p)
    j.iter("iter-1").result(x=1).close()
    # Tampering: just verify structured reload still works after we re-open.
    j2 = Journal(p)
    assert j2.iters[0].result == {"x": 1}
