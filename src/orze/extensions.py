"""Extension point for orze-pro (autopilot features).

orze (open) is a complete experiment orchestration tool:
  scheduling, leaderboard, notifications, retrospection, analysis, admin.

orze-pro adds autonomy: research agents that propose ideas, auto-fix
failures, and evolve code on plateaus — so experiments run while you sleep.

CALLING SPEC:
    get_extension(name: str) -> module or None
    has_pro() -> bool
    pro_features() -> list of available feature names

Extension points (all in the autonomy layer):
    "role_runner"              — multi-agent orchestration
    "agents.research"          — autonomous idea generation
    "agents.research_context"  — context builder for research LLM
    "agents.research_llm"      — LLM backends (Gemini/OpenAI/Anthropic)
    "agents.code_evolution"    — auto-evolve train script on plateau
    "agents.meta_research"     — meta-level strategy adjustment
    "agents.bug_fixer"         — auto-fix failed experiments
    "agents.bot"               — interactive Telegram/Slack bot
"""

import importlib
import logging
from typing import List, Optional

logger = logging.getLogger("orze")

_cache = {}

# Only autonomy features are gated behind pro.
# Everything else (retrospection, analysis, notifications, admin) is open.
_PRO_MODULES = {
    "role_runner": "orze_pro.engine.role_runner",
    "agents.research": "orze_pro.agents.research",
    "agents.research_context": "orze_pro.agents.research_context",
    "agents.research_llm": "orze_pro.agents.research_llm",
    "agents.code_evolution": "orze_pro.agents.code_evolution",
    "agents.meta_research": "orze_pro.agents.meta_research",
    "agents.bug_fixer": "orze_pro.agents.bug_fixer",
    "agents.bot": "orze_pro.agents.bot",
    "agents.professor_bootstrap": "orze_pro.agents.professor_bootstrap",
    "idea_filter": "orze_pro.engine.idea_filter",
}

# No built-in fallbacks — these modules live in orze-pro only.
_BUILTIN_FALLBACK = {}


def get_extension(name: str) -> Optional[object]:
    """Load a pro extension module. Returns None if not available."""
    if name in _cache:
        return _cache[name]

    # Try orze-pro first
    pro_path = _PRO_MODULES.get(name)
    if pro_path:
        try:
            mod = importlib.import_module(pro_path)
            _cache[name] = mod
            return mod
        except ImportError as exc:
            err_msg = str(exc)
            if "not activated" in err_msg or "ORZE_PRO_KEY" in err_msg:
                logger.error("orze-pro license check failed: %s. Roles will not run.", err_msg)
            pass

    # Fallback to built-in (transition period)
    fallback_path = _BUILTIN_FALLBACK.get(name)
    if fallback_path:
        try:
            mod = importlib.import_module(fallback_path)
            _cache[name] = mod
            return mod
        except ImportError:
            pass

    _cache[name] = None
    return None


_auto_install_attempted = False


def _find_pro_key() -> str:
    """Find ORZE_PRO_KEY from env var, .env file, or ~/.orze-pro.key."""
    import os
    from pathlib import Path
    # 1. Environment variable
    key = os.environ.get("ORZE_PRO_KEY", "").strip()
    if key:
        return key
    # 2. .env file in current directory
    env_path = Path(".env")
    if env_path.exists():
        try:
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("ORZE_PRO_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            pass
    # 3. ~/.orze-pro.key
    home_key = Path.home() / ".orze-pro.key"
    if home_key.exists():
        try:
            return home_key.read_text().strip()
        except OSError:
            pass
    return ""


def _auto_install_pro() -> bool:
    """Auto-install orze-pro from private PyPI if ORZE_PRO_KEY is set."""
    global _auto_install_attempted
    if _auto_install_attempted:
        return False
    _auto_install_attempted = True

    import subprocess
    import sys
    key = _find_pro_key()
    if not key or not key.startswith("ORZE-PRO-"):
        return False
    logger.info("Pro license key detected. Installing orze-pro...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "orze-pro", "--quiet",
         "--extra-index-url", f"https://__token__:{key}@pypi.orze.ai/simple/"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        logger.info("orze-pro installed successfully.")
        return True
    logger.warning("Failed to install orze-pro. Check your license key or visit https://orze.ai/pro")
    return False


def has_pro() -> bool:
    """Check if orze-pro is installed and licensed."""
    try:
        mod = importlib.import_module("orze_pro")
    except ImportError:
        if _auto_install_pro():
            try:
                mod = importlib.import_module("orze_pro")
            except ImportError:
                return False
        else:
            return False
    # orze_pro is importable, but check license status too
    is_licensed_fn = getattr(mod, "is_licensed", None)
    if is_licensed_fn is not None and not is_licensed_fn():
        logger.warning("orze-pro installed but not licensed — pro features disabled")
        return False
    return True


def pro_version() -> Optional[str]:
    """Return orze-pro version if installed."""
    try:
        mod = importlib.import_module("orze_pro")
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return None


def pro_features() -> List[str]:
    """List available pro features (checks importability without full load)."""
    if not has_pro():
        return []
    available = []
    for name, path in _PRO_MODULES.items():
        try:
            # Use find_module to check availability without triggering full import
            import importlib.util
            spec = importlib.util.find_spec(path)
            if spec is not None:
                available.append(name)
        except (ImportError, ModuleNotFoundError, ValueError):
            pass
    return available


def check_pro_status() -> str:
    """Return a human-readable pro status string for --check output."""
    if has_pro():
        ver = pro_version()
        features = pro_features()
        return f"orze-pro {ver} — {len(features)} features: {', '.join(features)}"
    else:
        # Check if built-in fallbacks exist (transition period)
        fallbacks = []
        for name, path in _BUILTIN_FALLBACK.items():
            try:
                importlib.import_module(path)
                fallbacks.append(name)
            except ImportError:
                pass
        if fallbacks:
            return f"orze-pro not installed (using {len(fallbacks)} built-in modules)"
        return ("orze-pro not installed — running in manual mode. "
                "Install orze-pro for autonomous research agents.")
