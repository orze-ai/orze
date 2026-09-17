"""One project rule for model instructions, selection and an optional bound.

The caller supplies already verified, valid development candidates with a
finite ``loss``. This example neither qualifies evidence nor establishes a
metric's lower bound. A bound must follow from the task definition, never from
the smallest observed loss. Confirmation still follows a search stop.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MinimumLoss:
    metric: str
    lower_bound: float | None = None

    def __post_init__(self):
        if type(self.metric) is not str or not self.metric.strip():
            raise ValueError("a metric name is required")
        if self.lower_bound is not None:
            self._finite(self.lower_bound)

    @staticmethod
    def _finite(value):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("loss and bound must be finite numbers")
        return value

    def select(self, valid_candidates):
        """Keep input order for exact ties; return the original candidate."""
        if not valid_candidates:
            raise ValueError("at least one valid development candidate is required")

        def loss(candidate):
            value = self._finite(candidate["loss"])
            if self.lower_bound is not None and value < self.lower_bound:
                raise ValueError("measured loss contradicts the declared bound")
            return value

        return min(valid_candidates, key=loss)

    def at_declared_bound(self, valid_candidates):
        """Exact equality only: further search cannot change this selection."""
        selected = self.select(valid_candidates)
        return self.lower_bound is not None and selected["loss"] == self.lower_bound

    def instructions(self):
        return (
            f"Actual selection rule: select the valid candidate with the smallest "
            f"development {self.metric}, including all earlier candidates; exact "
            "ties retain the earlier candidate. Only strict development-loss "
            "improvement can replace that selection. Robustness, simplicity, "
            "runtime and other secondary objectives do not break ties or change "
            "the selection in this task. Heldout outcomes never guide the selection."
        )
