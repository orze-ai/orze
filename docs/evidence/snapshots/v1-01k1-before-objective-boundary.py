#!/usr/bin/env python3
"""FSM runner: discovers and steps all procedures.

Loads from two tiers:
  1. orze-pro package: built-in procedures & plugins (pro features)
  2. Project-level: user overrides in procedures/ and fsm/plugins/

Pro-tier files live in orze-pro/src/orze_pro/{fsm/plugins,procedures,prompts}.
Project-level files can override or extend pro defaults.

Run as an orze script role:
    roles:
      fsm:
        mode: script
        script: fsm/runner.py
        args: ["--results-dir", "{results_dir}"]
        cooldown: 120
        timeout: 30
"""

from __future__ import annotations

import importlib
import importlib.util
import argparse
import logging
import os
import re
import sys
from pathlib import Path
from collections.abc import Mapping

# Ensure project root is on path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FSM] %(levelname)s %(message)s",
)
logger = logging.getLogger("fsm")


def _find_pro_package() -> Path | None:
    """Locate the orze-pro package's fsm directory."""
    # Try installed package first
    try:
        import orze_pro
        pkg_dir = Path(orze_pro.__file__).parent
        if (pkg_dir / "fsm").exists():
            return pkg_dir
    except ImportError:
        pass

    # Try local submodule
    submodule = Path(_project_root) / "orze-pro" / "src" / "orze_pro"
    if submodule.exists():
        return submodule

    return None


def _load_plugins(dirs: list[Path]):
    """Auto-discover and import plugins from multiple directories.

    Project-level plugins (first in dirs) override pro plugins by filename.
    All plugins are loaded via importlib.util to avoid module path conflicts.
    """
    seen = set()
    for plugins_dir in dirs:
        if not plugins_dir.exists():
            continue
        for py_file in sorted(plugins_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            if py_file.name in seen:
                logger.debug("Plugin %s already loaded (project override)", py_file.name)
                continue
            seen.add(py_file.name)
            try:
                spec = importlib.util.spec_from_file_location(
                    f"_fsm_plugin_{py_file.stem}", str(py_file))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                logger.debug("Loaded plugin: %s from %s", py_file.stem, plugins_dir)
            except Exception as e:
                logger.error("Failed to load plugin %s: %s", py_file, e)


def _load_procedures(dirs: list[Path], results_dir: Path, *,
                     extras: dict | None = None,
                     selected: dict[Path, set[str]] | None = None) -> list:
    """Load procedure YAMLs from multiple directories. Project overrides pro."""
    from orze.fsm.engine import FSM

    seen = set()
    fsms = []
    for proc_dir in dirs:
        if not proc_dir.exists():
            continue
        for yaml_file in sorted(proc_dir.glob("*.yaml")):
            if (selected is not None and proc_dir in selected
                    and yaml_file.stem not in selected[proc_dir]):
                continue
            if yaml_file.name in seen:
                logger.debug("Procedure %s already loaded (project override)", yaml_file.name)
                continue
            seen.add(yaml_file.name)
            try:
                fsm = FSM.from_yaml(str(yaml_file), results_dir, extras=extras)
                fsms.append(fsm)
                logger.debug("Loaded procedure: %s from %s", fsm.name, proc_dir)
            except Exception as e:
                logger.error("Failed to load %s: %s", yaml_file, e)
    return fsms


def _load_config(path: Path) -> dict:
    """Resolve child-relative paths while retaining full invocation policy."""
    from orze.core.config import load_project_config
    import yaml

    if not path.is_file():
        raise ValueError("fsm_config_unavailable")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("fsm_config_not_mapping")
    previous = Path.cwd()
    try:
        os.chdir(path.parent)
        cfg = load_project_config(str(path))
        cfg["_config_path"] = str(path)
        for key in ("results_dir", "ideas_file", "idea_lake_db"):
            value = cfg.get(key)
            if not isinstance(value, str) or not value or "${" in value:
                raise ValueError("fsm_config_path_unresolved")
            cfg[key] = str(Path(value).absolute())
        cfg["_env_ORZE_RESULTS_DIR"] = cfg["results_dir"]
        cfg["_env_ORZE_IDEAS_FILE"] = cfg["ideas_file"]
        return cfg
    finally:
        os.chdir(previous)


def _builtin_procedures(cfg: dict | None) -> set[str]:
    """Do not reconnect previously broken automation without an opt-in."""
    policy = (cfg or {}).get("fsm", {})
    if not isinstance(policy, Mapping):
        raise ValueError("fsm_policy_not_mapping")
    names = policy.get("procedures", [])
    if (not isinstance(names, list) or any(
            not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]+", name) is None
            for name in names)):
        raise ValueError("fsm_procedures_must_be_names_without_suffix")
    selected = {"activity_log", *names}
    selected.discard("idea_verifier")
    review = (cfg or {}).get("idea_review", {})
    if isinstance(review, Mapping) and review.get("enabled") is True:
        selected.add("idea_verifier")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir")
    parser.add_argument("--procedures-dir")
    parser.add_argument("--config", "-c")
    args = parser.parse_args()
    invocation_root = Path.cwd()
    config_argument = args.config if args.config is not None else os.environ.get("ORZE_CONFIG_PATH")
    config_path = Path(config_argument).absolute() if config_argument else None
    if config_path is None and (invocation_root / "orze.yaml").is_file():
        config_path = invocation_root / "orze.yaml"
    cfg = None
    try:
        if args.config is not None and not args.config.strip():
            raise ValueError("fsm_config_unavailable")
        if config_path is not None:
            cfg = _load_config(config_path)
        selected_names = _builtin_procedures(cfg)
        configured_results = Path(cfg["results_dir"]) if cfg is not None else None
        requested_results = Path(args.results_dir).absolute() if args.results_dir else None
        if (configured_results is not None and requested_results is not None
                and configured_results.resolve() != requested_results.resolve()):
            raise ValueError("fsm_results_scope_conflict")
        results_dir = requested_results or configured_results or invocation_root / "results"
    except Exception as exc:
        # Invalid configuration must not fall back to legacy default authority.
        logger.error("FSM configuration rejected (%s)", type(exc).__name__)
        raise SystemExit(1) from exc
    project_root = config_path.parent if config_path is not None else invocation_root
    project_procedures = (Path(args.procedures_dir).absolute() if args.procedures_dir
                          else project_root / "procedures")

    if not results_dir.exists():
        logger.error("Results dir not found: %s", results_dir)
        sys.exit(1)

    # Find orze-pro package
    pro_pkg = _find_pro_package()
    pro_tier = "pro" if pro_pkg else "basic"
    logger.info("Tier: %s%s", pro_tier,
                f" ({pro_pkg})" if pro_pkg else "")

    # Load plugins: project-level first (overrides), then pro
    plugin_dirs = [project_root / "fsm" / "plugins", Path(__file__).parent / "plugins"]
    if pro_pkg:
        plugin_dirs.append(pro_pkg / "fsm" / "plugins")
    _load_plugins(plugin_dirs)

    # Set up JSONL activity log
    from orze.fsm.engine import set_activity_log
    set_activity_log(results_dir / "_fsm_activity.jsonl")

    # Load procedures: project-level first (overrides), then pro
    proc_dirs = [project_procedures]
    if pro_pkg:
        proc_dirs.append(pro_pkg / "procedures")
    selected = {pro_pkg / "procedures": selected_names} if pro_pkg else None
    fsms = _load_procedures(proc_dirs, results_dir,
                            extras={"cfg": cfg} if cfg is not None else {},
                            selected=selected)

    if not fsms:
        logger.warning("No procedures found (checked: %s)",
                        ", ".join(str(d) for d in proc_dirs))
        sys.exit(0)

    logger.info("Loaded %d procedures: %s",
                len(fsms), ", ".join(f.name for f in fsms))

    # Step each FSM
    for fsm in fsms:
        status = fsm.status()
        logger.info("[%s] state=%s, transitions=%d",
                    fsm.name, status["state"], status["transitions"])
        fsm.step()


if __name__ == "__main__":
    main()
