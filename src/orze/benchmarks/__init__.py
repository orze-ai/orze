"""orze.benchmarks — preset framework for compliance-locked benchmark evaluation.

A `Preset` codifies the rules for a specific benchmark so the eval cannot
silently drift out of compliance. Concrete maintained presets ship in
`orze_pro.benchmarks` (HF Open ASR Leaderboard, MMLU, HumanEval, etc.);
this module ships the abstract base so anyone can build their own.

Pattern::

    from orze.benchmarks import Preset

    class MyBenchmark(Preset):
        name = "mybenchmark-v1"
        compliance_invariants = {
            "decode": "greedy",
            "normalizer": "myorg/normalizer-v1",
            "datasets": ["dataset_a", "dataset_b"],
        }

        def run(self, model, **overrides):
            self._check_no_invariants_violated(overrides)
            ...  # actual eval
            return {"macro_metric": 0.0, "per_dataset": {...}}

The `_check_no_invariants_violated` helper raises if the caller passed any
override that conflicts with the preset's locked invariants.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ComplianceViolation(RuntimeError):
    """Raised when an eval call would violate a preset's locked invariants."""


class Preset(ABC):
    """Base class for benchmark presets.

    Subclasses must set ``name`` and ``compliance_invariants`` and implement
    ``run``. The invariants dict is the contract the preset guarantees: keys
    are config knobs that MUST hold a specific value (or be in a specific set)
    for any run produced by the preset to count as compliant.
    """

    name: str = ""
    compliance_invariants: dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Don't enforce on the abstract base itself.
        if cls.__name__ == "Preset":
            return
        if not cls.name:
            raise TypeError(f"{cls.__name__} must set a non-empty `name` class attr")

    @abstractmethod
    def run(self, model: Any, **overrides: Any) -> dict[str, Any]:
        """Run the benchmark. Returns a dict at minimum with the headline metric.

        Implementations should call ``self._check_no_invariants_violated(overrides)``
        before doing anything expensive.
        """

    def _check_no_invariants_violated(self, overrides: dict[str, Any]) -> None:
        violations = []
        for key, expected in self.compliance_invariants.items():
            if key in overrides and overrides[key] != expected:
                violations.append(f"  {key}: locked to {expected!r}, got {overrides[key]!r}")
        if violations:
            raise ComplianceViolation(
                f"Preset {self.name!r} compliance violation:\n" + "\n".join(violations)
            )

    def describe(self) -> str:
        """Human-readable summary of the preset's rules."""
        lines = [f"Preset: {self.name}"]
        if self.compliance_invariants:
            lines.append("Locked invariants:")
            for k, v in self.compliance_invariants.items():
                lines.append(f"  {k} = {v!r}")
        return "\n".join(lines)


__all__ = ["Preset", "ComplianceViolation"]
