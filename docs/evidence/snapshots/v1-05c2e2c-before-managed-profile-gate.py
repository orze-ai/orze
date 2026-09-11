"""Strict opt-in boundary for the local, stop-only controller profile.

This module is metadata-only: it does not register a controller, open a
database, discover GPUs, stop processes, clear markers or authorize restart.
Disabled legacy configurations are not revalidated against this profile.
Fingerprints bind parsed public JSON plus resolved launch/control paths and
the exact physical GPU set; private runtime additions are excluded. A caller
must recompute against its own captured fingerprint, not trust a public stamp.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


class ControllerProfileError(ValueError):
    """Stable, content-free refusal before unsupported profile side effects."""


def _reject(reason):
    raise ControllerProfileError("controller_profile_" + reason)


def _mapping(cfg, name):
    value = cfg.get(name)
    if value is None:
        return {}
    if type(value) is not dict:
        _reject(name + "_invalid")
    return value


def _gpus(value):
    if (type(value) is not list or not value
            or any(type(item) is not int or item < 0 for item in value)
            or len(value) != len(set(value))):
        _reject("explicit_gpus_required")
    return sorted(value)


def controller_profile(cfg):
    """Return a detached exact declaration, or None for the disabled profile."""
    if cfg is None:
        return None
    if type(cfg) is not dict:
        _reject("config_invalid")
    declaration = cfg.get("controller_control")
    if declaration is None:
        return None
    if (type(declaration) is not dict or set(declaration) != {"version", "profile"}
            or type(declaration["version"]) is not int or declaration["version"] != 1
            or type(declaration["profile"]) is not str
            or declaration["profile"] != "local_stop_v1"):
        _reject("declaration_invalid")
    _gpus(_mapping(cfg, "gpu_scheduling").get("allowed_gpus"))
    if cfg.get("telemetry", True) is not False:
        _reject("telemetry_unsupported")
    if cfg.get("auto_upgrade", True) is not False:
        _reject("auto_upgrade_unsupported")
    for name in ("bot", "telegram_bot"):
        if cfg.get(name) is not None:
            _reject(name + "_unsupported")
    for name in ("notifications", "retrospection"):
        if _mapping(cfg, name).get("enabled", False) is not False:
            _reject(name + "_unsupported")
    cleanup = _mapping(cfg, "cleanup")
    if cleanup.get("script") not in (None, ""):
        _reject("cleanup_script_unsupported")
    harvest = _mapping(cfg, "metric_harvest")
    if (harvest.get("enabled", True) is not False
            and harvest.get("llm_fallback", True) is not False):
        _reject("metric_llm_unsupported")
    if type(cfg.get("max_fix_attempts", 0)) is not int or cfg.get("max_fix_attempts", 0) != 0:
        _reject("implicit_repair_unsupported")
    for name in ("containers", "remote", "fleet"):
        if cfg.get(name) not in (None, {}, []):
            _reject(name + "_unsupported")
    if _mapping(cfg, "substrate").get("elo_ranking_enabled", False) is not False:
        _reject("elo_ranking_unsupported")
    if any(type(key) is not str for key in cfg):
        _reject("json_invalid")
    _json_value({key: value for key, value in cfg.items() if not key.startswith("_")})
    return {"version": 1, "profile": "local_stop_v1"}


def _json_value(value, depth=0):
    if depth > 32:
        _reject("json_invalid")
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _json_value(item, depth + 1)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _json_value(item, depth + 1)
        return
    _reject("json_invalid")


def profile_fingerprint(cfg, gpu_ids=None):
    """Hash the complete supported parsed configuration without publishing it."""
    if controller_profile(cfg) is None:
        return None
    allowed = _gpus(cfg["gpu_scheduling"]["allowed_gpus"])
    if gpu_ids is not None and _gpus(gpu_ids) != allowed:
        _reject("gpu_scope_changed")
    public = {key: value for key, value in cfg.items()
              if type(key) is str and not key.startswith("_")}
    if any(type(key) is not str for key in cfg):
        _reject("json_invalid")
    _json_value(public)
    workdir = cfg.get("_controller_workdir", str(Path.cwd()))
    if type(workdir) is not str or not workdir:
        _reject("workdir_invalid")
    workdir = Path(workdir).absolute()
    def absolute(value):
        if type(value) is not str or not value or "\0" in value:
            _reject("path_invalid")
        path = Path(value)
        return str((path if path.is_absolute() else workdir / path).absolute())
    paths = {name: absolute(cfg.get(name, default)) for name, default in (
        ("_config_path", "orze.yaml"), ("results_dir", "orze_results"),
        ("idea_lake_db", ".orze/idea_lake.db"), ("_orze_dir", ".orze"),
        ("_project_root", str(workdir)))}
    payload = {"schema": 1, "public_config": public, "paths": paths,
               "workdir": str(workdir), "physical_gpus": allowed}
    try:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError):
        _reject("json_invalid")
    return hashlib.sha256(raw).hexdigest()


def validate_profile_cli(cfg, args):
    """Validate controller-mode CLI overrides before GPU/admin/launch effects."""
    profile = controller_profile(cfg)
    if profile is None:
        return None
    command = getattr(args, "command", None)
    if (command == "run-idea" or getattr(args, "role_only", None)
            or getattr(args, "research_only", False) or getattr(args, "admin", False)
            or command == "start"):
        _reject("cli_mode_unsupported")
    raw = getattr(args, "gpus", None)
    if raw is not None:
        if (type(raw) is not str or not raw or any(not part.strip().isascii()
                or not part.strip().isdigit() for part in raw.split(","))):
            _reject("explicit_gpus_required")
        if _gpus([int(part.strip()) for part in raw.split(",")]) != _gpus(cfg["gpu_scheduling"]["allowed_gpus"]):
            _reject("gpu_scope_changed")
    stopping = command in {"stop", "restart"} or getattr(args, "stop", False) or getattr(args, "restart", False)
    for arg, key in (("timeout", "timeout"), ("poll", "poll"), ("results_dir", "results_dir"),
                     ("train_script", "train_script"), ("ideas_md", "ideas_file"), ("base_config", "base_config")):
        value = getattr(args, arg, None)
        if value is not None and not (arg == "timeout" and stopping) and value != cfg.get(key):
            _reject("cli_override_unsupported")
    return profile
