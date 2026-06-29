"""SOP (Standard Operating Procedure) registry for research workflow types.

This module defines all supported research workflow types and their configurations.
Each SOP type (training, analysis, data_prep, etc.) is registered here with its
workflow class, transition rules, and executor functions.

Design principle: FSM is orthogonal to SOP. The FSM tracks state transitions
uniformly (CLAIMED → IN_PROGRESS → COMPLETE/FAILED) regardless of SOP type.
The SOP defines what work is done and how transitions flow through substeps.
"""

from typing import Dict, Any, Callable, List, Tuple


class SOPDefinition:
    """Configuration for a single SOP type."""

    def __init__(
        self,
        name: str,
        description: str,
        launcher_func: Callable,
        checker_func: Callable,
        transitions: List[Tuple[str, str, str]],
        substates: List[str] = None,
    ):
        """
        Args:
            name: SOP type identifier (e.g., "training")
            description: Human-readable description
            launcher_func: Function to launch work (callable(idea_id, gpu, results_dir, cfg, lake))
            checker_func: Function to check active processes (callable(active_dict, results_dir, cfg, lake))
            transitions: List of (from_state, to_state, reason_template) tuples
            substates: Optional detailed state tracking (e.g., ["training", "evaluating"])
        """
        self.name = name
        self.description = description
        self.launcher = launcher_func
        self.checker = checker_func
        self.transitions = transitions
        self.substates = substates or []

    def __repr__(self):
        return f"SOPDefinition({self.name}, substates={len(self.substates)})"


class SOPRegistry:
    """Global registry of all SOP types."""

    def __init__(self):
        self._registry: Dict[str, SOPDefinition] = {}

    def register(self, sop: SOPDefinition) -> None:
        """Register a new SOP type."""
        if sop.name in self._registry:
            raise ValueError(f"SOP '{sop.name}' already registered")
        self._registry[sop.name] = sop

    def get(self, sop_type: str) -> SOPDefinition:
        """Get SOP definition by type. Falls back to 'training' if not found."""
        if sop_type not in self._registry:
            if sop_type is not None:
                import logging
                logging.warning(f"SOP type '{sop_type}' not found, using 'training'")
            return self._registry.get("training")
        return self._registry[sop_type]

    def list_sops(self) -> Dict[str, SOPDefinition]:
        """List all registered SOPs."""
        return dict(self._registry)

    def is_registered(self, sop_type: str) -> bool:
        """Check if SOP type is registered."""
        return sop_type in self._registry


# Global registry instance
_global_registry = SOPRegistry()


def register_sop(sop: SOPDefinition) -> None:
    """Register a SOP globally."""
    _global_registry.register(sop)


def get_sop(sop_type: str) -> SOPDefinition:
    """Get a SOP definition globally."""
    return _global_registry.get(sop_type)


def list_sops() -> Dict[str, SOPDefinition]:
    """List all registered SOPs globally."""
    return _global_registry.list_sops()


def is_sop_registered(sop_type: str) -> bool:
    """Check if a SOP type is registered globally."""
    return _global_registry.is_registered(sop_type)


# Note: Actual SOP registrations happen in sop/training_sop.py, sop/analysis_sop.py, etc.
# This module provides only the registry infrastructure.
