"""v4.5 SOP Dispatcher - Lightweight workflow type dispatcher.

Simple dispatch based on sop_type from idea config.
No framework, no registry, no singleton - just functions.

Orchestrator reads cfg.get("sop", "training"), calls get_sop_launcher() or get_sop_checker()
to get the appropriate function, and invokes it. That's it.
"""

import importlib
import logging
from typing import Callable, Dict, Optional, Any

logger = logging.getLogger("orze.sop_dispatcher")

# Built-in SOP definitions (no YAML dependency for core functionality)
_BUILTIN_SOPS = {
    "training": {
        "description": "Model training with post-training evaluation",
        "launcher_module": "orze.engine.launcher",
        "launcher_function": "launch",
        "checker_module": "orze.engine.launcher",
        "checker_function": "check_active",
    },
    "analysis": {
        "description": "Data/model analysis workflow (compute-only, no training)",
        "launcher_module": "orze.engine.analysis",
        "launcher_function": "launch_analysis",
        "checker_module": "orze.engine.analysis",
        "checker_function": "check_active_analysis",
    },
}


def get_sop_launcher(sop_type: Optional[str]) -> Callable:
    """Get launcher function for a SOP type.

    Args:
        sop_type: SOP type name (e.g., "training", "analysis")
                  If None or unknown, defaults to "training"

    Returns:
        Callable launcher function with signature:
          launcher(idea_id, gpu, results_dir, cfg, lake=None) -> Process

    Raises:
        ImportError/AttributeError: If launcher function cannot be loaded
    """
    sop_type = sop_type or "training"
    sop_def = _BUILTIN_SOPS.get(sop_type)

    if not sop_def:
        logger.warning("SOP type '%s' not found, using 'training'", sop_type)
        sop_def = _BUILTIN_SOPS.get("training")

    try:
        module = importlib.import_module(sop_def["launcher_module"])
        return getattr(module, sop_def["launcher_function"])
    except (ImportError, AttributeError) as e:
        logger.error("Failed to load launcher for SOP '%s': %s", sop_type, e)
        raise


def get_sop_checker(sop_type: Optional[str]) -> Callable:
    """Get checker function for a SOP type.

    Args:
        sop_type: SOP type name (e.g., "training", "analysis")
                  If None or unknown, defaults to "training"

    Returns:
        Callable checker function with signature:
          checker(active_dict, results_dir, cfg, lake=None, ...) -> list[(idea_id, gpu)]

    Raises:
        ImportError/AttributeError: If checker function cannot be loaded
    """
    sop_type = sop_type or "training"
    sop_def = _BUILTIN_SOPS.get(sop_type)

    if not sop_def:
        logger.warning("SOP type '%s' not found, using 'training'", sop_type)
        sop_def = _BUILTIN_SOPS.get("training")

    try:
        module = importlib.import_module(sop_def["checker_module"])
        return getattr(module, sop_def["checker_function"])
    except (ImportError, AttributeError) as e:
        logger.error("Failed to load checker for SOP '%s': %s", sop_type, e)
        raise


def register_sop(sop_type: str, launcher_module: str, launcher_func: str,
                 checker_module: str, checker_func: str) -> None:
    """Register a new SOP type at runtime.

    Args:
        sop_type: Name of the SOP (e.g., "analysis", "data_prep")
        launcher_module: Module name containing launcher function (e.g., "orze.engine.analysis")
        launcher_func: Function name for launcher (e.g., "launch_analysis")
        checker_module: Module name containing checker function
        checker_func: Function name for checker (e.g., "check_active_analysis")
    """
    _BUILTIN_SOPS[sop_type] = {
        "launcher_module": launcher_module,
        "launcher_function": launcher_func,
        "checker_module": checker_module,
        "checker_function": checker_func,
    }
    logger.info("Registered SOP type: %s", sop_type)


def get_sop_type(cfg: Dict[str, Any]) -> str:
    """Extract SOP type from idea config.

    Args:
        cfg: Idea configuration dict (from ideas.md or database)

    Returns:
        SOP type string (defaults to "training" if not specified)
    """
    return cfg.get("sop", "training")


def list_registered_sops() -> Dict[str, Dict[str, str]]:
    """List all registered SOP types.

    Returns:
        Dict of SOP type -> definition
    """
    return dict(_BUILTIN_SOPS)
