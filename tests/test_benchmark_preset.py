"""Tests for orze.benchmarks.Preset abstract base."""

from __future__ import annotations

import pytest

from orze.benchmarks import Preset, ComplianceViolation


def test_subclass_must_set_name():
    with pytest.raises(TypeError, match="non-empty `name`"):
        class Bad(Preset):
            def run(self, model, **kw):
                return {}


def test_invariant_violation_raises():
    class P(Preset):
        name = "test"
        compliance_invariants = {"decode": "greedy", "normalizer": "lb-v1"}

        def run(self, model, **overrides):
            self._check_no_invariants_violated(overrides)
            return {"score": 0.0}

    p = P()
    # OK: not overriding the invariant
    assert p.run(None) == {"score": 0.0}
    # OK: overriding to the same value
    assert p.run(None, decode="greedy") == {"score": 0.0}
    # BAD: overriding to a different value
    with pytest.raises(ComplianceViolation, match="decode"):
        p.run(None, decode="beam_search")


def test_describe():
    class P(Preset):
        name = "x"
        compliance_invariants = {"a": 1}

        def run(self, model, **kw):
            return {}

    assert "x" in P().describe()
    assert "a = 1" in P().describe()
