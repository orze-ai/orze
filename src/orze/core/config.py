"""Config loading, validation, merging, and sanitization for Orze projects.

CALLING SPEC:
    DEFAULT_CONFIG -> dict
        Module-level dict with all default orze.yaml keys and their defaults.

    load_project_config(path: Optional[str] = None) -> dict
        Load orze.yaml (or path), merge with DEFAULT_CONFIG, auto-discover
        research backends from env vars, load .env. Returns full config dict.

    _validate_config(cfg: dict) -> tuple[list[str], list[str]]
        Validate a loaded config. Returns (errors, warnings) where errors
        are fatal and warnings are informational.

    _sanitize_config(config: dict) -> dict
        Deep-copy config and replace invalid non-numeric values in known
        numeric fields (e.g. sequence_length, batch_size) with safe defaults.

    find_dotenv(config_path: Optional[str] = None) -> Optional[Path]
        Locate .env file next to config or in CWD. Returns path or None.

    _load_dotenv(config_path: Optional[str] = None) -> int
        Load .env into os.environ (only sets vars not already present).
        Returns count of vars loaded.
"""
import os
import re
import math
import logging
import copy
from typing import Optional
from pathlib import Path
import yaml

logger = logging.getLogger("orze")

import sys

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")

# Canonical claude-mode role skill defaults. Users who declare one of these
# roles with ``mode: claude`` but omit ``skills:`` get the bundled SOP stack
# auto-injected instead of a hard config error. Keep in sync with the lists
# in orze_pro.engine.role_runner — this registry is the user-facing
# self-healing layer so projects that override (e.g. pin a different model)
# don't have to restate every skill.
CANONICAL_CLAUDE_SKILL_DEFAULTS: dict = {
    "professor": [
        "@sop:file_layout",
        "@sop:professor_base",
        "@sop:professor_paper_lake",
        "@sop:professor_web_search",
        "@sop:professor_cross_domain_query",
        "@sop:professor_idea_review",
        "@sop:professor_diversity_enforcement",
        "@sop:professor_gap_closure",
        "@sop:professor_strategy_review",
        "@sop:professor_regression_detection",
        "@sop:professor_steering",
    ],
    "thinker": [
        "@sop:file_layout",
        "@sop:thinker_synthesis",
        "@sop:thinker_base",
        "@sop:thinker_phase_a_reformulation",
        "@sop:thinker_phase_b_root_cause",
        "@sop:thinker_axiom_removal",
        "@sop:thinker_phase_c_constraints",
        "@sop:thinker_phase_d_cross_domain",
        "@sop:thinker_phase_e_proposals",
        "@sop:thinker_phase_f_implementation",
    ],
    "data_analyst": [
        "@sop:file_layout",
        "@sop:data_analyst_base",
        "@sop:data_analyst_error_analysis",
        "@sop:data_analyst_visualization",
        "@sop:data_analyst_insights",
        "@sop:data_analyst_anomaly_hypotheses",
    ],
    "engineer": [
        "@sop:file_layout",
        "@sop:engineer_base",
        "@sop:engineer_implement",
        "@sop:engineer_fix_bugs",
    ],
}


def _expand_env_vars(obj):
    """Recursively expand ${VAR} references in string values using os.environ."""
    if isinstance(obj, str):
        def _replace(m):
            return os.environ.get(m.group(1), m.group(0))
        return _ENV_VAR_RE.sub(_replace, obj)
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    return obj


# Strict re-scan after substitution: any ${VAR} left in a string value
# means the env var was not set when config loaded. We don't error
# (that would break first-run setups where the user is wiring secrets
# in stages) but we surface the *path* to the offending key so the
# operator knows exactly which value will silently 404 / 401 at use
# time. This is the post-mortem fix for the 5-day silent campaign
# where ${TELEGRAM_BOT_TOKEN} stayed unresolved in
# notifications.channels and every notify() 404'd.
_UNRESOLVED_VAR_RE = re.compile(r"\$\{[A-Z_][A-Z0-9_]*\}")


def _find_unresolved_env_vars(obj, path: str = "") -> list:
    """Return list of (dotted_path, raw_value) for any ${VAR} placeholder
    that survived _expand_env_vars."""
    found: list = []
    if isinstance(obj, str):
        if _UNRESOLVED_VAR_RE.search(obj):
            found.append((path or "<root>", obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            sub = f"{path}.{k}" if path else str(k)
            found.extend(_find_unresolved_env_vars(v, sub))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            sub = f"{path}[{i}]"
            found.extend(_find_unresolved_env_vars(item, sub))
    return found


def find_dotenv(config_path: Optional[str] = None) -> Optional[Path]:
    """Find .env file: next to config or CWD. Returns path or None."""
    candidates = []
    if config_path:
        candidates.append(Path(config_path).resolve().parent / ".env")
    candidates.append(Path.cwd() / ".env")
    for c in candidates:
        if c.is_file():
            return c
    return None


def _parse_dotenv(env_file: Path) -> dict:
    """Parse a .env file into a dict of key-value pairs."""
    result = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            result[key] = value
    return result


def _load_dotenv(config_path: Optional[str] = None) -> int:
    """Load .env file. Only sets vars NOT already in os.environ. Returns count loaded."""
    env_file = find_dotenv(config_path)
    if not env_file:
        return 0

    loaded = 0
    for key, value in _parse_dotenv(env_file).items():
        if not os.environ.get(key):
            os.environ[key] = value
            loaded += 1

    if loaded:
        logger.info("Loaded %d env var(s) from %s", loaded, env_file)
    return loaded


def reload_dotenv(config_path: Optional[str] = None) -> int:
    """Re-read .env and update os.environ with changed values. Returns count updated."""
    env_file = find_dotenv(config_path)
    if not env_file:
        return 0

    updated = 0
    for key, value in _parse_dotenv(env_file).items():
        if os.environ.get(key) != value:
            os.environ[key] = value
            updated += 1
            logger.info(".env hot-reload: %s changed", key)

    return updated


DEFAULT_CONFIG = {
    "train_script": "train.py",
    "ideas_file": None,  # None = derive to {orze_dir}/ideas.md
    "base_config": "configs/base.yaml",
    "results_dir": "orze_results",
    "python": sys.executable,
    "train_extra_args": [],
    "train_extra_env": {},
    "timeout": 3600,
    "poll": 30,
    "gpu_mem_threshold": 2000,
    "gpu_scheduling": {
        "max_vram_pct": 90,        # stop filling GPU at this VRAM %
        "min_free_vram_mib": 1000, # require this much free VRAM
        "max_jobs_per_gpu": 1,     # safety cap (1 = backward compat)
        "allowed_gpus": [],         # optional hard physical-GPU allowlist
    },
    "launcher": {
        "paused": False,
        "paused_flag_path": None,
    },
    "pre_script": None,
    "pre_args": [],
    "pre_timeout": 3600,
    "artifact_preflight": {
        "enabled": False,
        "script": None,
        "args": [],
        "timeout": 300,
        "network": "inherit",
        "retry_interval": 300,
    },
    "resume": {
        "enabled": False,
        "progress_file": "progress.json",
        "args": ["--resume-from", "{checkpoint}"],
        "checkpoint_roots": [],
        "immutable_inputs": [],
        "input_roots": [],
        "max_files": 10000,
        "max_bytes": 0,
    },
    "eval_script": None,
    "eval_args": [],
    "eval_timeout": 3600,
    "eval_output": "eval_report.json",
    "post_scripts": [],
    "cleanup": {
        "script": None,
        "interval": 100,
        "patterns": [],
    },
    "report": {
        "title": "Orze Report",
        "primary_metric": "test_accuracy",
        "sort": "descending",
        "columns": [
            {"key": "test_accuracy", "label": "Accuracy", "fmt": ".4f"},
            {"key": "test_loss", "label": "Loss", "fmt": ".4f"},
            {"key": "training_time", "label": "Time(s)", "fmt": ".0f"},
        ],
        "ceiling_k": 20,
        "ceiling_std_threshold": 0.015,
        "ceiling_min_ideas": 30,
        # Optional fail-closed contract for benchmark-comparable local ranks.
        # See docs/benchmark-contract.md.
        "benchmark_contract": None,
    },
    "stall_minutes": 0,         # 0 = disabled
    "role_stall_minutes": 5,    # composite agent no-progress watchdog
    "max_idea_failures": 0,     # 0 = disabled (never skip)
    "max_fix_attempts": 0,      # 0 = disabled; executor LLM fix attempts per idea
    "min_disk_gb": 0,           # 0 = disabled
    "orphan_timeout_hours": 6,  # reclaim stale claims after 6 hours
    "plateau_threshold": 50,    # fire plateau notification after N completions w/o improvement
    "roles": {},
    # Enforce OS sandboxing and deterministic tool-call denials for managed
    # Claude roles. Older/unsupported sandbox runtimes fail closed.
    "agent_tool_policy": {
        "enabled": True,
    },
    "auto_upgrade": True,
    "sweep_stray": True,        # sweep stray files to .orze/stray/ by default
    # Data boundary guardrails. Hard path blocks use a verified private mount
    # namespace; network denial uses a private network namespace. watch_paths
    # is explicitly Python-level audit only.
    "data_boundaries": {
        "forbidden_in_training": [],  # list[str] — abort training on read
        "watch_paths": [],            # list[str] — log-only audit
        # ``deny`` runs the trainer and all descendants in a private network
        # namespace. ``inherit`` is not sufficient for a no-leakage claim.
        "training_network": "inherit",
    },
    # Optional keyed-fingerprint manifest audit. This proves only the declared
    # manifests are within the configured overlap bounds; see
    # docs/data-separation.md.
    "data_separation": {
        "enabled": False,
        "train_manifest": None,
        "train_manifest_sha256": None,
        "evaluation_manifest": None,
        "evaluation_manifest_sha256": None,
        "fingerprint_namespace_sha256": None,
        "normalization_contract_sha256": None,
        "fields": ["sample"],
        "max_overlap": {"sample": 0},
        "max_records": 10_000_000,
        "max_bytes": 2 * 1024 * 1024 * 1024,
        "max_line_bytes": 4096,
    },
    # Bind one declared output artifact to its managed attempt, kernel-boundary
    # activation, and data-separation receipt. Disabled for external/pretrained
    # artifacts whose training did not run under Orze.
    "model_lineage": {
        "enabled": False,
        "artifact": None,
        "max_files": 100_000,
        "max_bytes": 100 * 1024 * 1024 * 1024,
        "attestation_timeout": 10,
    },
    # Auto-seal eval scripts. When true, any file matching eval_*.py or
    # eval_*.sh in the project root is added to sealed_files at config
    # load time, preventing silent mutation by LLM agents.
    "auto_seal_eval": True,
    "notifications": {
        "enabled": False,
        "on": ["completed", "failed", "new_best", "watchdog_restart", "plateau",
               "needs_intervention", "role_circuit_breaker", "role_degraded"],
        "channels": [],
    },
    "retrospection": {
        "enabled": False,
        "script": "",
        "interval": 50,
        "timeout": 120,
    },
}
def load_project_config(path: Optional[str] = None) -> dict:
    """Load orze.yaml and merge with defaults. Returns full config dict."""
    _load_dotenv(path)
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    if not path and Path("orze.yaml").exists():
        path = "orze.yaml"

    if path and Path(path).exists():
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for k, v in raw.items():
            if k == "report" and isinstance(v, dict):
                cfg["report"] = {**cfg["report"], **v}
            else:
                cfg[k] = v
        # Fix YAML 'on:' boolean parsing — YAML interprets 'on' as True
        # so notifications.on becomes notifications[True] instead of notifications["on"]
        ncfg = cfg.get("notifications")
        if isinstance(ncfg, dict) and True in ncfg and "on" not in ncfg:
            ncfg["on"] = ncfg.pop(True)
            logger.info("Fixed YAML 'on:' boolean key in notifications config")

        logger.info("Loaded config from %s", path)
    elif path:
        logger.warning("Config file %s not found, using defaults", path)

    # Expand ${VAR} references in config values using os.environ
    cfg = _expand_env_vars(cfg)

    # Loud-warn on unresolved ${VAR} placeholders. Calls relying on these
    # (notifications, webhooks) will silently fail at runtime — make the
    # diagnosis a one-liner instead of a 5-day silent campaign.
    unresolved = _find_unresolved_env_vars(cfg)
    for path, raw in unresolved:
        logger.warning(
            "Unresolved ${VAR} placeholder at %s: %r — env var not set; "
            "any call relying on this value will silently fail "
            "(set the variable in .env or shell, then reload).",
            path, raw,
        )

    # Round-2 F1: top-level ``evolution.enabled`` /
    # ``evolution.max_attempts_per_plateau`` were a separate naming for
    # the same concept the ``code_evolution`` role implements. Migrate
    # them into ``roles.code_evolution`` and emit one DeprecationWarning
    # at config-load so operators have a release to update orze.yaml.
    # Old keys keep working until removed in a future release.
    legacy_evolution = cfg.get("evolution")
    if isinstance(legacy_evolution, dict):
        import warnings as _warnings
        _warnings.warn(
            "Top-level 'evolution:' is deprecated; move 'evolution.enabled' "
            "and 'evolution.max_attempts_per_plateau' under "
            "'roles.code_evolution.{enabled, max_attempts_per_plateau}'. "
            "Old keys will be honored for one release.",
            DeprecationWarning, stacklevel=2,
        )
        logger.warning(
            "Deprecated config key 'evolution:' detected — fold into "
            "roles.code_evolution.{enabled,max_attempts_per_plateau}. "
            "Migrating in-place for this run.")
        if not isinstance(cfg.get("roles"), dict):
            cfg["roles"] = {}
        roles = cfg["roles"]
        ce = roles.setdefault("code_evolution", {})
        if isinstance(ce, dict):
            for src, dst in (("enabled", "enabled"),
                              ("max_attempts_per_plateau",
                               "max_attempts_per_plateau"),
                              ("model", "model"),
                              ("timeout", "timeout"),
                              ("claude_bin", "claude_bin")):
                if src in legacy_evolution and dst not in ce:
                    ce[dst] = legacy_evolution[src]

    # Migrate legacy research: into roles: dict
    if "research" in cfg and isinstance(cfg["research"], dict):
        logger.warning("Migrating legacy 'research:' config to 'roles: {research: ...}'. "
                        "Update orze.yaml to use the 'roles:' format directly.")
        if not cfg.get("roles"):
            cfg["roles"] = {"research": cfg["research"]}
        elif "research" not in cfg["roles"]:
            cfg["roles"]["research"] = cfg["research"]

    # Compute project_root and orze_dir from results_dir
    results_path = Path(cfg["results_dir"])
    if not results_path.is_absolute():
        results_path = Path.cwd() / results_path
    project_root = results_path.parent
    orze_dir = project_root / ".orze"
    
    cfg["_orze_dir"] = str(orze_dir)
    cfg["_project_root"] = str(project_root)
    
    # Resolve ideas_file: None → .orze/ideas.md
    if not cfg.get("ideas_file"):
        cfg["ideas_file"] = str(orze_dir / "ideas.md")
    
    # Resolve idea_lake_db default → .orze/idea_lake.db (NOT results_dir)
    if not cfg.get("idea_lake_db"):
        cfg["idea_lake_db"] = str(orze_dir / "idea_lake.db")
    
    # Environment variable exposures for subprocess injection
    cfg["_env_ORZE_DIR"] = str(orze_dir)
    cfg["_env_ORZE_RESULTS_DIR"] = str(results_path)
    cfg["_env_ORZE_IDEAS_FILE"] = cfg["ideas_file"]
    cfg["_env_ORZE_RULES_DIR"] = str(orze_dir / "rules")
    cfg["_env_ORZE_METHODS_DIR"] = str(results_path / "methods")
    cfg["_env_ORZE_KNOWLEDGE_DIR"] = str(results_path / "knowledge")
    cfg["_env_ORZE_FEEDBACK_DIR"] = str(orze_dir / "feedback")

    # Auto-discover research backends from environment API keys.
    # Only activates if NO roles are explicitly configured at all.
    # If the user defined any roles (even mode: script), respect that
    # and don't inject auto-discovered backends alongside them.
    roles = cfg.get("roles") or {}
    if not roles:
        _AUTO_BACKENDS = [
            ("GEMINI_API_KEY", "gemini", "gemini-2.5-flash"),
            ("OPENAI_API_KEY", "openai", "gpt-4o"),
            ("ANTHROPIC_API_KEY", "anthropic", None),
        ]
        discovered = []
        for env_var, backend, default_model in _AUTO_BACKENDS:
            if os.environ.get(env_var):
                role_name = f"research_{backend}"
                role_cfg = {"mode": "research", "backend": backend}
                if default_model:
                    role_cfg["model"] = default_model
                if "roles" not in cfg:
                    cfg["roles"] = {}
                cfg["roles"][role_name] = role_cfg
                discovered.append(f"{backend} ({env_var})")
        if discovered:
            logger.info("Auto-discovered research backends: %s",
                        ", ".join(discovered))
        else:
            logger.info("No API keys found in environment — research agent will not run. "
                        "Add GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY to .env")

    # Auto-seal eval scripts (data leakage guardrail). Any file in the project
    # root matching eval_*.py or eval_*.sh is added to sealed_files unless
    # auto_seal_eval is explicitly set to false.
    if cfg.get("auto_seal_eval", True):
        sealed = list(cfg.get("sealed_files") or [])
        existing = set(sealed)
        auto_added = []
        try:
            for pattern in ("eval_*.py", "eval_*.sh"):
                for match in sorted(Path(".").glob(pattern)):
                    name = str(match)
                    if name not in existing:
                        sealed.append(name)
                        existing.add(name)
                        auto_added.append(name)
        except Exception as e:
            logger.warning("auto_seal_eval glob failed: %s", e)
        if auto_added:
            cfg["sealed_files"] = sealed
            logger.info("auto_seal_eval: sealed %d eval script(s): %s",
                        len(auto_added), ", ".join(auto_added))

    # A plain sealed_files entry snapshots whatever happens to exist at
    # startup.  sealed_hashes additionally pins a preregistered SHA-256 so
    # drift that happened before startup is caught as well.  Pinned paths are
    # automatically included in every existing sealed-file check.
    pinned = cfg.get("sealed_hashes") or {}
    if isinstance(pinned, dict):
        sealed = list(cfg.get("sealed_files") or [])
        existing = set(sealed)
        for fpath in pinned:
            if fpath not in existing:
                sealed.append(fpath)
                existing.add(fpath)
        if sealed:
            cfg["sealed_files"] = sealed

    return cfg


def _validate_config(cfg: dict) -> tuple:
    """Validate orze config on startup. Returns (errors, warnings) tuple."""
    errors = []
    warnings = []

    # --- Errors: things that will break ---

    # Validate roles
    roles = cfg.get("roles")
    if roles and isinstance(roles, dict):
        for rname, rcfg in roles.items():
            if not isinstance(rcfg, dict):
                errors.append(f"roles.{rname}: expected dict, got {type(rcfg).__name__}")
                continue
            mode = rcfg.get("mode", "script")
            if mode not in ("script", "claude", "research"):
                errors.append(f"roles.{rname}.mode: '{mode}' is invalid "
                              f"(expected 'script', 'claude', or 'research')")
            if mode == "claude" and not rcfg.get("skills"):
                default_skills = CANONICAL_CLAUDE_SKILL_DEFAULTS.get(rname)
                if default_skills:
                    rcfg["skills"] = list(default_skills)
                    warnings.append(
                        f"roles.{rname}: no 'skills' set — auto-injected "
                        f"{len(default_skills)} canonical SOPs for known role "
                        f"'{rname}'")
                else:
                    errors.append(
                        f"roles.{rname}: mode 'claude' requires 'skills' "
                        f"(no canonical defaults for role name '{rname}'; "
                        f"known canonicals: "
                        f"{', '.join(sorted(CANONICAL_CLAUDE_SKILL_DEFAULTS))})")
            # rules_file was removed in 3.4.0 — reject explicitly rather
            # than silently ignoring a stale legacy key.
            if "rules_file" in rcfg:
                errors.append(
                    f"roles.{rname}: 'rules_file' was removed in orze 3.4.0. "
                    f"Replace with 'skills: [./{rcfg['rules_file']}]' "
                    f"(or add bundled '@sop:<name>' entries).")
            if mode == "claude":
                import shutil as _shutil
                _claude_bin = rcfg.get("claude_bin", "claude")
                if not _shutil.which(_claude_bin):
                    errors.append(
                        f"roles.{rname}: mode 'claude' requires Claude CLI "
                        f"but '{_claude_bin}' not found on PATH. "
                        f"Install: npm install -g @anthropic-ai/claude-code")
            if mode == "script" and not rcfg.get("script"):
                errors.append(f"roles.{rname}: mode 'script' requires 'script'")
            if mode == "research" and not rcfg.get("backend"):
                errors.append(f"roles.{rname}: mode 'research' requires 'backend' "
                              f"(gemini, openai, anthropic, ollama, custom)")
            for field_name in ("stall_minutes", "stall_warmup_seconds"):
                value = rcfg.get(field_name)
                if (value is not None and (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        or value < 0)):
                    errors.append(
                        f"roles.{rname}.{field_name}: must be a "
                        "non-negative number"
                    )

    agent_policy = cfg.get(
        "agent_tool_policy", DEFAULT_CONFIG["agent_tool_policy"])
    if not isinstance(agent_policy, dict):
        errors.append("agent_tool_policy: must be a mapping")
    elif not isinstance(agent_policy.get("enabled", True), bool):
        errors.append("agent_tool_policy.enabled: must be true or false")
    elif agent_policy.get("enabled", True) and isinstance(roles, dict):
        protected_options = (
            "--add-dir", "--allow-dangerously-skip-permissions",
            "--dangerously-skip-permissions", "--permission-mode",
            "--setting-sources", "--settings",
        )
        for role_name, role_cfg in roles.items():
            if (not isinstance(role_cfg, dict)
                    or role_cfg.get("mode") != "claude"):
                continue
            if role_cfg.get("dangerously_skip_permissions", False):
                errors.append(
                    f"roles.{role_name}.dangerously_skip_permissions: "
                    "conflicts with enabled agent_tool_policy"
                )
            raw_args = role_cfg.get("claude_args") or []
            args = [raw_args] if isinstance(raw_args, str) else raw_args
            if isinstance(args, list):
                conflicts = [
                    str(arg) for arg in args
                    if any(str(arg) == option
                           or str(arg).startswith(option + "=")
                           for option in protected_options)
                ]
                if conflicts:
                    errors.append(
                        f"roles.{role_name}.claude_args: cannot override "
                        "enabled agent_tool_policy: " + ", ".join(conflicts)
                    )

    executor_fix = cfg.get("executor_fix", {})
    if not isinstance(executor_fix, dict):
        errors.append("executor_fix: must be a mapping")
    else:
        if "dangerously_skip_permissions" in executor_fix:
            errors.append(
                "executor_fix.dangerously_skip_permissions: forbidden")
        try:
            executor_turns = int(executor_fix.get("max_turns", 20))
            if executor_turns < 1:
                errors.append("executor_fix.max_turns: must be at least 1")
        except (TypeError, ValueError):
            errors.append("executor_fix.max_turns: must be an integer")
        try:
            executor_timeout = float(executor_fix.get("timeout", 300))
            if executor_timeout <= 0:
                errors.append("executor_fix.timeout: must be positive")
        except (TypeError, ValueError):
            errors.append("executor_fix.timeout: must be numeric")

    try:
        executor_enabled = int(cfg.get("max_fix_attempts", 0)) > 0
    except (TypeError, ValueError):
        executor_enabled = False
    if executor_enabled and (
        not isinstance(agent_policy, dict)
        or agent_policy.get("enabled", True) is not True
    ):
        errors.append(
            "max_fix_attempts: executor fixes require enabled "
            "agent_tool_policy")

    # Validate numeric fields
    for key in ("timeout", "poll", "eval_timeout", "stall_minutes",
                "role_stall_minutes",
                "max_idea_failures", "max_fix_attempts", "min_disk_gb",
                "orphan_timeout_hours"):
        val = cfg.get(key)
        if val is not None and (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(float(val))
                or val < 0):
            errors.append(f"{key}: must be a non-negative number, got {val!r}")

    gpu_cfg = cfg.get("gpu_scheduling", DEFAULT_CONFIG["gpu_scheduling"])
    if not isinstance(gpu_cfg, dict):
        errors.append("gpu_scheduling: must be a mapping")
    else:
        normalized_gpu_lists = {}
        for key in ("allowed_gpus", "reserved_gpus"):
            values = gpu_cfg.get(key, []) or []
            if (not isinstance(values, list)
                    or any(isinstance(value, bool)
                           or not isinstance(value, int)
                           or value < 0 for value in values)
                    or len(values) != len(set(values))):
                errors.append(
                    f"gpu_scheduling.{key}: must be a list of unique "
                    "non-negative integer GPU IDs")
            else:
                normalized_gpu_lists[key] = set(values)
        # ``allowed_gpus`` is a hard positive scope and ``reserved_gpus`` is
        # an independent negative scope.  Redundantly reserving a device that
        # is already outside the allowlist is safe and makes operator intent
        # explicit; rejecting that combination prevents exact least-privilege
        # configurations such as allow 4-7 while reserving 0-3.
        min_free = gpu_cfg.get("min_free_vram_mib", 1000)
        if (isinstance(min_free, bool) or not isinstance(min_free, int)
                or min_free < 0):
            errors.append(
                "gpu_scheduling.min_free_vram_mib: must be a "
                "non-negative integer")

    launcher_cfg = cfg.get("launcher", DEFAULT_CONFIG["launcher"])
    if not isinstance(launcher_cfg, dict):
        errors.append("launcher: must be a mapping")
    else:
        if not isinstance(launcher_cfg.get("paused", False), bool):
            errors.append("launcher.paused: must be true or false")
        paused_path = launcher_cfg.get("paused_flag_path")
        if (paused_path is not None
                and (not isinstance(paused_path, str)
                     or not paused_path.strip()
                     or any(ord(char) < 32 for char in paused_path))):
            errors.append(
                "launcher.paused_flag_path: must be null or a non-empty "
                "path without control characters")

    # Validate eval config consistency
    if cfg.get("eval_script") and not cfg.get("eval_output"):
        errors.append("eval_script is set but eval_output is missing")

    # Resolve datasets/models before allocating a training process. The
    # resolver is deliberately explicit because silently inheriting offline
    # flags caused repeated launches that could never fetch missing metadata.
    artifact_preflight = cfg.get(
        "artifact_preflight", DEFAULT_CONFIG["artifact_preflight"])
    if not isinstance(artifact_preflight, dict):
        errors.append("artifact_preflight: must be a mapping")
    else:
        enabled = artifact_preflight.get("enabled", False)
        if not isinstance(enabled, bool):
            errors.append("artifact_preflight.enabled: must be true or false")
        script = artifact_preflight.get("script")
        if enabled and (not isinstance(script, str) or not script.strip()):
            errors.append(
                "artifact_preflight.script: required when preflight is enabled"
            )
        elif enabled:
            project_root = Path(cfg.get("_project_root", "."))
            script_path = Path(script)
            if not script_path.is_absolute():
                script_path = project_root / script_path
            if not script_path.is_file():
                errors.append(
                    f"artifact_preflight.script not found: {script_path}"
                )
        args = artifact_preflight.get("args", [])
        if not isinstance(args, list):
            errors.append("artifact_preflight.args: must be a list")
        policy = artifact_preflight.get("network", "inherit")
        if policy not in ("inherit", "required", "offline"):
            errors.append(
                "artifact_preflight.network: must be 'inherit', 'required', "
                "or 'offline'"
            )
        for key in ("timeout", "retry_interval"):
            value = artifact_preflight.get(key, 300)
            if (isinstance(value, bool)
                    or not isinstance(value, (int, float)) or value < 0):
                errors.append(
                    f"artifact_preflight.{key}: must be a non-negative number"
                )
        if policy == "required":
            extra_env = cfg.get("train_extra_env") or {}
            if isinstance(extra_env, dict):
                offline_keys = [
                    key for key in (
                        "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE",
                        "TRANSFORMERS_OFFLINE",
                    )
                    if str(extra_env.get(key, "")).strip().lower()
                    in {"1", "true", "yes", "on"}
                ]
                if offline_keys:
                    errors.append(
                        "artifact_preflight.network is 'required' but "
                        "train_extra_env enables offline mode via: "
                        + ", ".join(offline_keys)
                    )

    resume_cfg = cfg.get("resume", DEFAULT_CONFIG["resume"])
    if not isinstance(resume_cfg, dict):
        errors.append("resume: must be a mapping")
    else:
        resume_enabled = resume_cfg.get("enabled", False)
        if not isinstance(resume_enabled, bool):
            errors.append("resume.enabled: must be true or false")
        progress_file = resume_cfg.get("progress_file", "progress.json")
        if (not isinstance(progress_file, str) or not progress_file
                or Path(progress_file).is_absolute()
                or ".." in Path(progress_file).parts):
            errors.append(
                "resume.progress_file: must be a relative path inside the idea directory"
            )
        resume_args = resume_cfg.get(
            "args", ["--resume-from", "{checkpoint}"])
        if (not isinstance(resume_args, list)
                or not all(isinstance(arg, str) for arg in resume_args)
                or not any("{checkpoint}" in str(arg)
                           for arg in resume_args)):
            errors.append(
                "resume.args: must be a list containing {checkpoint}"
            )
        for key in ("checkpoint_roots", "immutable_inputs", "input_roots"):
            paths = resume_cfg.get(key, [])
            if (not isinstance(paths, list)
                    or not all(isinstance(path, str) and path
                               for path in paths)):
                errors.append(f"resume.{key}: must be a list")
        if resume_enabled and not resume_cfg.get("immutable_inputs"):
            errors.append(
                "resume.immutable_inputs: at least one pinned model/dataset "
                "input is required when resume is enabled"
            )
        for key, default in (("max_files", 10000), ("max_bytes", 0)):
            value = resume_cfg.get(key, default)
            if (isinstance(value, bool) or not isinstance(value, int)
                    or value < 0):
                errors.append(f"resume.{key}: must be a non-negative integer")

    pinned = cfg.get("sealed_hashes")
    if pinned is not None:
        if not isinstance(pinned, dict):
            errors.append("sealed_hashes: must be a mapping of path to SHA-256")
        else:
            for fpath, digest in pinned.items():
                if not isinstance(fpath, str) or not fpath:
                    errors.append("sealed_hashes: every path must be a non-empty string")
                if (not isinstance(digest, str)
                        or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None):
                    errors.append(
                        f"sealed_hashes.{fpath}: expected a 64-character SHA-256"
                    )

    # Data-boundary configuration controls whether held-out paths and the
    # network are visible to training. Validate it here and again at the final
    # launcher boundary so a direct caller cannot bypass startup validation.
    boundaries = cfg.get("data_boundaries", {})
    if not isinstance(boundaries, dict):
        errors.append("data_boundaries: must be a mapping")
    else:
        for key in ("forbidden_in_training", "watch_paths"):
            paths = boundaries.get(key, [])
            if (not isinstance(paths, list)
                    or any(not isinstance(path, str) or not path.strip()
                           for path in paths)):
                errors.append(
                    f"data_boundaries.{key}: must be a list of non-empty paths"
                )
            elif any(not Path(path).is_absolute() for path in paths):
                errors.append(
                    f"data_boundaries.{key}: every path must be absolute"
                )
            elif any(":" in path or any(ord(char) < 32 for char in path)
                     for path in paths):
                errors.append(
                    f"data_boundaries.{key}: paths cannot contain ':' or "
                    "control characters"
                )
        network = boundaries.get("training_network", "inherit")
        if network not in ("inherit", "deny"):
            errors.append(
                "data_boundaries.training_network: expected 'inherit' or 'deny'"
            )

    from orze.core.data_separation import validate_data_separation_config
    errors.extend(validate_data_separation_config(cfg))
    from orze.core.model_lineage import validate_model_lineage_config
    errors.extend(validate_model_lineage_config(cfg))

    # Report columns are consumed as mappings by the leaderboard. Reject
    # shorthand strings during --check instead of crashing after compute has
    # already been spent on a completed experiment.
    report_cfg = cfg.get("report", {})
    if not isinstance(report_cfg, dict):
        errors.append("report: must be a mapping")
    else:
        columns = report_cfg.get("columns", [])
        if not isinstance(columns, list):
            errors.append("report.columns: must be a list of mappings")
        else:
            for index, column in enumerate(columns):
                if not isinstance(column, dict) or not column.get("key"):
                    errors.append(
                        f"report.columns[{index}]: must be a mapping with a non-empty 'key'"
                    )
        from orze.core.benchmark_contract import (
            validate_benchmark_contract_config,
        )
        errors.extend(validate_benchmark_contract_config(cfg))

    # train_script must exist
    ts = cfg.get("train_script")
    if ts and not Path(ts).exists():
        errors.append(f"train_script not found: {ts}")

    # Contract check: verify train_script reads idea_config.yaml or idea_lake.db
    if ts and Path(ts).exists():
        try:
            script_text = Path(ts).read_text()
            reads_config = ("idea_config" in script_text
                            or "idea_lake" in script_text
                            or "sweep_config" in script_text)
            if not reads_config:
                warnings.append(
                    f"train_script '{ts}' does not reference idea_config.yaml "
                    f"or idea_lake.db — idea-specific config overrides may be ignored")
        except OSError:
            pass

    # --- Warnings: things that might be unintentional ---

    bc = cfg.get("base_config")
    if bc and not Path(bc).exists():
        warnings.append(f"base_config not found: {bc}")

    es = cfg.get("eval_script")
    if es and not Path(es).exists():
        warnings.append(f"eval_script not found: {es}")

    if not roles:
        warnings.append("No research agent configured — idea generation disabled. "
                        "Add an API key to .env (GEMINI_API_KEY, OPENAI_API_KEY, or "
                        "ANTHROPIC_API_KEY) for auto-discovery, or configure roles: in orze.yaml")

    # Check for API keys if research roles exist
    has_research = roles and any(
        isinstance(rc, dict) and rc.get("mode") == "research"
        for rc in roles.values()
    )
    if has_research:
        has_key = any(os.environ.get(k) for k in
                      ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"))
        if not has_key:
            warnings.append("Research role configured but no API keys found in environment")

    gc_cfg = cfg.get("gc") or {}
    if not gc_cfg.get("enabled"):
        warnings.append("GC disabled — checkpoint dirs will accumulate indefinitely. "
                        "Add gc: {enabled: true, checkpoints_dir: ...} to enable.")

    ncfg = cfg.get("notifications", {})
    if not isinstance(ncfg, dict):
        errors.append("notifications: must be a mapping")
        ncfg = {}
    if not ncfg.get("enabled"):
        warnings.append("Notifications disabled")
    else:
        channels = ncfg.get("channels", [])
        if not channels:
            warnings.append("Notifications enabled but no channels configured")
        for ch in channels:
            if not isinstance(ch, dict):
                continue
            ch_type = ch.get("type", "?")
            if ch_type == "telegram":
                if not ch.get("bot_token"):
                    warnings.append(f"Notification channel '{ch_type}': missing bot_token")
                if not ch.get("chat_id"):
                    warnings.append(f"Notification channel '{ch_type}': missing chat_id")
            elif ch_type in ("slack", "discord"):
                if not ch.get("webhook_url"):
                    warnings.append(f"Notification channel '{ch_type}': missing webhook_url")
            elif ch_type == "webhook":
                if not ch.get("url"):
                    warnings.append(f"Notification channel '{ch_type}': missing url")

    # --- Issue A: Warn about unknown/misspelled config keys ---
    # NOTE: this allowlist must stay in lockstep with the keys we
    # emit from the `orze setup` template (cli_setup.py) AND with
    # any keys consumed by `cfg.get(...)` across the engine. When
    # we add a new top-level config key, add it here too — otherwise
    # users get a noisy "Unknown config key" warning on a key orze
    # itself shipped them. See Round-3: nested_config_whitelist /
    # metric_harvest / sweep_allowlist were emitted by the template
    # but absent from this list, producing false-positive validator
    # warnings on every fresh `orze setup` install.
    _KNOWN_EXTRAS = {
        "_config_path", "research", "gc", "metric_validation", "sealed_files",
        "sealed_hashes",
        "min_expected_results", "goal_file", "gpu_scheduling", "roles",
        "notifications", "evolution", "retrospection", "cleanup",
        "train_extra_args", "train_extra_env", "pre_script", "pre_args",
        "pre_timeout", "eval_script", "eval_args", "eval_timeout",
        "eval_output", "post_scripts", "report",
        "admin_port", "idea_lake_db", "bot", "telegram_bot",
        "sops", "containers",
        # Round-3: shipped by `orze setup` template; consumed by
        # phases (nested_config_whitelist), reporting/leaderboard
        # (metric_harvest), and reserved for sweep guardrails
        # (sweep_allowlist; emitted by template, consumed by future
        # sweep_stray hardening — keep allowlisted so users can
        # configure ahead of time without warnings).
        "nested_config_whitelist", "nested_config_normalize",
        "metric_harvest", "sweep_allowlist",
        "telemetry", "research_budget",
        # Runtime-consumed framework controls.  Keep these recognised so
        # valid production configs do not emit typo warnings at startup.
        "role_stall_minutes", "executor_fix",
    }
    known_keys = set(DEFAULT_CONFIG.keys()) | _KNOWN_EXTRAS
    for key in cfg:
        if key.startswith("_"):
            continue  # computed/internal keys (prefix _)
        if key not in known_keys:
            known_list = ", ".join(sorted(known_keys))
            warnings.append(
                f"Unknown config key '{key}' in orze.yaml — possible typo? "
                f"(known keys: {known_list})"
            )

    # --- Multi-tenant hint ---
    ideas_val = cfg.get("ideas_file", "ideas.md")
    if ideas_val == "ideas.md" or (not Path(ideas_val).is_absolute()
                                    and not ideas_val.startswith(cfg.get("results_dir", "orze_results"))):
        logger.debug("ideas_file is '%s' (relative, not under results_dir). "
                      "Multi-instance setups should use distinct ideas_file paths.",
                      ideas_val)

    # --- Issue B: Warn if ideas_file does not exist yet ---
    ideas_path = cfg.get("ideas_file")
    if ideas_path and not Path(ideas_path).exists():
        warnings.append(
            f"ideas_file '{ideas_path}' does not exist yet — the system will "
            f"have no ideas to run until it is created"
        )

    # --- Issue C: Error if python interpreter path does not exist ---
    python_path = cfg.get("python")
    if python_path and not Path(python_path).exists():
        errors.append(f"python interpreter not found: {python_path}")

    # --- Unresolved ${VAR} placeholders: surface in `orze --check` output ---
    # Skip the computed/internal keys (prefix _) and the canonical-skill
    # registry which contains literal @sop:... (no ${} but also we don't
    # need to walk it).
    scan_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    for path, raw in _find_unresolved_env_vars(scan_cfg):
        warnings.append(
            f"Unresolved env var placeholder at '{path}': {raw!r} — "
            f"variable not set in environment; any call relying on this "
            f"value will silently fail at runtime"
        )

    return errors, warnings

def _sanitize_config(config: dict) -> dict:
    """Sanitize config by replacing invalid numeric values with defaults.

    Handles cases where AI-generated ideas use strings like 'variable' or 'auto'
    in fields that expect integers (e.g., sequence_length, max_frames).
    """
    if not isinstance(config, dict):
        return config

    # Known numeric fields that should be integers
    numeric_fields = {
        ("training", "sequence_length"): 32,
        ("training", "batch_size"): 16,
        ("training", "epochs"): 10,
        ("data", "batch_size"): 16,
        ("data", "frame_sampling", "max_frames"): 32,
        ("optimizer", "max_epochs"): 10,
    }

    # Known intermediate fields that must be dicts (not lists/scalars)
    dict_fields = {
        ("data", "frame_sampling"): {},
    }

    def sanitize_value(value, default):
        """Try to convert to int; if it fails, return default."""
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                logger.warning(f"Replacing invalid numeric value '{value}' with {default}")
                return default
        return value

    # Deep copy to avoid modifying original
    config = copy.deepcopy(config)

    # Ensure known intermediate fields are dicts
    for path, default in dict_fields.items():
        current = config
        for key in path[:-1]:
            if key not in current or not isinstance(current[key], dict):
                break
            current = current[key]
        else:
            final_key = path[-1]
            if final_key in current and not isinstance(current[final_key], dict):
                logger.warning(
                    f"Replacing non-dict value for '{'.'.join(path)}' "
                    f"(was {type(current[final_key]).__name__}) with {default}"
                )
                current[final_key] = default

    # Sanitize known numeric fields
    for path, default in numeric_fields.items():
        current = config
        for i, key in enumerate(path[:-1]):
            if key not in current or not isinstance(current[key], dict):
                break
            current = current[key]
        else:
            # We successfully navigated to the parent dict
            final_key = path[-1]
            if final_key in current:
                current[final_key] = sanitize_value(current[final_key], default)

    return config


# Path helper for .orze/ and orze_results/ layout
_ORZE_DIR_KINDS = {"logs", "receipts", "locks", "triggers", "mcp", "state", 
                    "heartbeats", "backups", "feedback", "tmp", "rules"}
_RESULTS_KINDS = {"methods", "knowledge", "stray"}


def orze_path(cfg: dict, kind: str, name: str = "") -> Path:
    """Return path for orze-internal or results subdirectories.
    
    Args:
        cfg: Config dict (must have _orze_dir and _env_ORZE_RESULTS_DIR)
        kind: One of: logs, receipts, locks, triggers, mcp, state, heartbeats,
              backups, feedback, tmp, stray, rules (→ .orze/kind/)
              OR methods, knowledge (→ orze_results/kind/)
        name: Optional subdirectory or filename under kind/
    
    Returns:
        Path object. Parent directory is created if it doesn't exist.
    
    Raises:
        ValueError: If kind is not recognized.
    """
    # 2026-05-22: trigger files were path-split — role_runner.py + bot.py +
    # sop_tier2.py wrote to .orze/triggers/_trigger_<role>, but every RULES
    # file, SOP skill, and orze/cli.py write/read results/_trigger_<role>.
    # Unify on the user-visible results/ path that the RULES files already
    # use as the contract. _RESULTS_KINDS would prefix a "triggers/" subdir,
    # so special-case to return results_dir directly.
    if kind == "triggers":
        base = Path(cfg["_env_ORZE_RESULTS_DIR"])
    elif kind in _ORZE_DIR_KINDS:
        base = Path(cfg["_orze_dir"]) / kind
    elif kind in _RESULTS_KINDS:
        base = Path(cfg["_env_ORZE_RESULTS_DIR"]) / kind
    else:
        raise ValueError(f"Unknown kind: {kind}")
    
    base.mkdir(parents=True, exist_ok=True)
    
    if name:
        return base / name
    return base
