"""Fail-closed verification for the installed Orze service runtime."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

from orze.service.watchdog import load_service_config


CONTRACT_VERSION = 1
CONTROLLER_CONTRACT_VERSION = 1
_STOP_SENTINELS = (".orze_disabled", ".orze_stop_all", ".orze_shutdown")
# Development-only dependency trees are not imported or served by the Orze
# runtime; the admin server serves the separately hashed ``ui/dist`` build.
# Including node_modules made every launch re-attestation read thousands of
# irrelevant files and tied a runtime identity to local frontend tooling.
_IGNORED_PARTS = {"__pycache__", "node_modules"}
_RUNTIME_ENVIRONMENT_KEYS = {
    "BASH_ENV",
    "ENV",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PYTHONHOME",
    "PYTHONINSPECT",
    "PYTHONPATH",
    "PYTHONSTARTUP",
}


class RuntimeContractError(RuntimeError):
    """Raised when runtime identity cannot be captured safely."""


def _hash_package_tree(root: Path) -> tuple[str, int]:
    """Hash package-relative paths and bytes without following escape links."""
    root = Path(root).resolve(strict=True)
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in _IGNORED_PARTS for part in relative.parts):
            continue
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise RuntimeContractError("runtime_package_symlink_escape") from exc
        payload = resolved.read_bytes()
        encoded_name = relative.as_posix().encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        count += 1
    if count == 0:
        raise RuntimeContractError("runtime_package_empty")
    return digest.hexdigest(), count


def capture_runtime_packages(
        names: Iterable[str] = ("orze", "orze_pro")) -> list[dict]:
    """Capture import roots and content hashes for installed runtime packages."""
    captured = []
    for name in names:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, AttributeError, ValueError):
            spec = None
        if spec is None or not spec.submodule_search_locations:
            if name == "orze":
                raise RuntimeContractError("orze_runtime_not_importable")
            continue
        locations = list(spec.submodule_search_locations)
        if len(locations) != 1:
            raise RuntimeContractError(
                f"{name}_runtime_location_ambiguous")
        root = Path(locations[0]).resolve(strict=True)
        tree_hash, file_count = _hash_package_tree(root)
        captured.append({
            "name": name,
            "root": str(root),
            "sha256": tree_hash,
            "file_count": file_count,
        })
    return captured


def _systemd_properties() -> dict:
    keys = (
        "Restart", "WorkingDirectory", "ExecStart", "ExecStartPre",
        "Environment", "EnvironmentFiles", "PassEnvironment",
        "UnsetEnvironment", "ActiveState", "UnitFileState",
    )
    result = subprocess.run(
        ["systemctl", "--user", "show", "orze.service",
         "--all", f"--property={','.join(keys)}"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeContractError("systemd_properties_unavailable")
    properties = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            # ExecStartPre may legally occur more than once. Preserve every
            # effective record so an expected command cannot hide an extra
            # pre-start command by appearing last in the property stream.
            if key in properties:
                properties[key] += "\n" + value
            else:
                properties[key] = value
    required = {
        "Restart", "WorkingDirectory", "ExecStart", "Environment",
        "PassEnvironment", "ActiveState", "UnitFileState",
    }
    if any(key not in properties for key in required):
        raise RuntimeContractError("systemd_properties_incomplete")
    properties.setdefault("ExecStartPre", "")
    properties.setdefault("EnvironmentFiles", "")
    properties.setdefault("UnsetEnvironment", "")
    unit_text = subprocess.run(
        ["systemctl", "--user", "cat", "orze.service", "--no-pager"],
        capture_output=True, text=True, timeout=10,
    )
    if unit_text.returncode != 0:
        raise RuntimeContractError("systemd_unit_text_unavailable")
    properties["_UnitText"] = unit_text.stdout
    return properties


def _runtime_errors(expected: object, observed: list[dict]) -> list[str]:
    if not isinstance(expected, list) or not expected:
        return ["runtime_identity_missing"]
    errors = []
    by_name = {
        record.get("name"): record
        for record in observed if isinstance(record, dict)
    }
    for record in expected:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            errors.append("runtime_identity_invalid")
            continue
        current = by_name.get(record["name"])
        if current is None:
            errors.append(f"runtime_package_missing:{record['name']}")
            continue
        for field in ("root", "sha256", "file_count"):
            if current.get(field) != record.get(field):
                errors.append(f"runtime_package_{field}_drift:{record['name']}")
    expected_names = {
        record.get("name") for record in expected if isinstance(record, dict)
    }
    for name in by_name.keys() - expected_names:
        errors.append(f"runtime_package_unpinned:{name}")
    return errors


def audit_controller_runtime_contract(
    expected: object,
    *,
    observed_packages: Optional[list[dict]] = None,
    observed_python: Optional[str] = None,
) -> dict:
    """Verify an opt-in project pin for direct controller launches.

    Unlike the service contract, this check is executed by the controller
    package itself.  It detects path, interpreter, and content drift in
    contract-aware runtimes, but cannot bootstrap trust in an older runtime
    that predates this check.  Managed systemd launches use the independent
    ``ExecStartPre`` service contract for that downgrade boundary.
    """
    errors = []
    if not isinstance(expected, dict):
        errors.append("controller_runtime_invalid")
        expected = {}
    if expected.get("contract_version") != CONTROLLER_CONTRACT_VERSION:
        errors.append(
            "controller_runtime_contract_version_missing_or_unsupported")

    configured_python = expected.get("python")
    if not isinstance(configured_python, str) or not configured_python.strip():
        errors.append("controller_runtime_python_invalid")
    else:
        try:
            current_python = Path(
                observed_python or sys.executable).resolve(strict=True)
            if Path(configured_python).resolve(strict=True) != current_python:
                errors.append("controller_runtime_python_identity_drift")
        except (OSError, RuntimeError):
            errors.append("controller_runtime_python_invalid")

    try:
        observed = (observed_packages if observed_packages is not None
                    else capture_runtime_packages())
        errors.extend(_runtime_errors(expected.get("packages"), observed))
    except (OSError, RuntimeContractError) as exc:
        errors.append(str(exc))

    errors = sorted(set(errors))
    return {
        "schema_version": 1,
        "contract_ok": not errors,
        "errors": errors,
    }


def require_controller_runtime_contract(expected: object) -> dict:
    """Require an opt-in controller pin and raise only stable error codes.

    The no-contract case remains backward compatible. Callers use this shared
    boundary before any state transition that can authorize later work, not
    only at controller startup or immediately before GPU telemetry.
    """
    if expected is None:
        return {"schema_version": 1, "contract_ok": True, "errors": []}
    report = audit_controller_runtime_contract(expected)
    if not report["contract_ok"]:
        reasons = ",".join(sorted(set(report.get("errors") or [])))
        raise RuntimeContractError(
            "controller_runtime_contract_rejected:"
            f"{reasons or 'unknown_runtime_drift'}")
    return report


def capture_controller_runtime_contract() -> dict:
    """Return a ready-to-paste exact pin for the current controller."""
    return {
        "contract_version": CONTROLLER_CONTRACT_VERSION,
        "python": str(Path(sys.executable).resolve(strict=True)),
        "packages": capture_runtime_packages(),
    }


_EXEC_RECORD_RE = re.compile(
    r"\{\s*path=(?P<path>.*?)\s*;\s*"
    r"argv\[\]=(?P<argv>.*?)\s*;\s*"
    r"ignore_errors=(?P<ignore>[^;\s]+)\s*;",
    re.DOTALL,
)


def _exact_exec_property(value: object, expected: tuple[str, ...]) -> bool:
    """Return whether a systemd show-property is one exact command record."""
    if not isinstance(value, str) or not value.strip():
        return False
    records = list(_EXEC_RECORD_RE.finditer(value))
    if len(records) != 1 or value.count("{ path=") != 1:
        return False
    record = records[0]
    return (
        record.group("path").strip() == expected[0]
        and record.group("argv").strip() == " ".join(expected)
        and record.group("ignore").strip() == "no"
    )


def _environment_keys(value: object, *, assignments: bool) -> set[str]:
    if not isinstance(value, str):
        return set()
    if assignments:
        return set(re.findall(
            r"(?:^|\s)[\"']?([A-Za-z_][A-Za-z0-9_]*)=", value))
    return {
        token.strip("\"'").split("=", 1)[0]
        for token in value.split()
        if token.strip("\"'")
    }


def _unit_errors(svc_cfg: dict, properties: dict) -> list[str]:
    errors = []
    if properties.get("Restart") != "no":
        errors.append("systemd_restart_policy_drift")
    try:
        expected_workdir = str(Path(svc_cfg["workdir"]).resolve(strict=True))
        observed_workdir = str(Path(
            properties.get("WorkingDirectory", "")
        ).resolve(strict=False))
        if observed_workdir != expected_workdir:
            errors.append("systemd_workdir_drift")
    except (KeyError, OSError, RuntimeError):
        errors.append("systemd_workdir_invalid")

    required_start = (
        str(svc_cfg.get("python", "")), "-m", "orze.cli", "-c",
        str(svc_cfg.get("config_file", "")),
    )
    if (any(not token for token in required_start)
            or not _exact_exec_property(
                properties.get("ExecStart", ""), required_start)):
        errors.append("systemd_exec_start_drift")
    exec_pre = properties.get("ExecStartPre", "")
    required_pre = (
        str(svc_cfg.get("python", "")), "-m",
        "orze.service.runtime_contract", "--startup-check",
    )
    if not exec_pre:
        errors.append("systemd_exec_start_pre_missing")
    elif (any(not token for token in required_pre)
          or not _exact_exec_property(exec_pre, required_pre)):
        errors.append("systemd_exec_start_pre_drift")

    environment = properties.get("Environment", "")
    pass_environment = properties.get("PassEnvironment", "")
    explicit_keys = _environment_keys(environment, assignments=True)
    passed_keys = _environment_keys(pass_environment, assignments=False)
    unset_keys = _environment_keys(
        properties.get("UnsetEnvironment", ""), assignments=False)
    if "PYTHONPATH" in explicit_keys or "PYTHONPATH" in passed_keys:
        errors.append("systemd_pythonpath_override")
    if ((explicit_keys | passed_keys) &
            (_RUNTIME_ENVIRONMENT_KEYS - {"PYTHONPATH"})):
        errors.append("systemd_runtime_environment_override")
    # User services otherwise inherit their manager's environment. Explicitly
    # remove interpreter/linker injection keys after all environment sources
    # have been assembled, including manager state and future drop-ins.
    if not _RUNTIME_ENVIRONMENT_KEYS.issubset(unset_keys):
        errors.append("systemd_runtime_environment_unsealed")
    if "orze_subscription_limit_actions=off" in environment.lower():
        errors.append("systemd_shutdown_actions_disabled")
    if properties.get("EnvironmentFiles", "").strip():
        errors.append("systemd_environment_files_unpinned")
    if re.search(r"(?mi)^\s*EnvironmentFile\s*=\s*\S", properties.get(
            "_UnitText", "")):
        errors.append("systemd_environment_files_unpinned")
    return errors


def audit_runtime_contract(
    svc_cfg: dict,
    *,
    properties: Optional[dict] = None,
    observed_packages: Optional[list[dict]] = None,
) -> dict:
    """Return contract and startup decisions using stable, non-secret codes."""
    errors = []
    if not isinstance(svc_cfg, dict):
        return {
            "schema_version": 1,
            "contract_ok": False,
            "startup_allowed": False,
            "errors": ["service_config_invalid"],
            "active_latches": [],
        }
    if svc_cfg.get("runtime_contract_version") != CONTRACT_VERSION:
        errors.append("runtime_contract_version_missing_or_unsupported")
    try:
        configured_python = Path(svc_cfg["python"]).resolve(strict=True)
        if configured_python != Path(sys.executable).resolve(strict=True):
            errors.append("runtime_python_identity_drift")
    except (KeyError, OSError, RuntimeError):
        errors.append("runtime_python_invalid")
    for key, kind in (
        ("workdir", "directory"),
        ("results_dir", "directory"),
        ("config_file", "file"),
    ):
        try:
            path = Path(svc_cfg[key]).resolve(strict=True)
            valid = path.is_dir() if kind == "directory" else path.is_file()
            if not valid:
                errors.append(f"service_{key}_invalid")
        except (KeyError, OSError, RuntimeError):
            errors.append(f"service_{key}_invalid")
    try:
        observed = (observed_packages if observed_packages is not None
                    else capture_runtime_packages())
        errors.extend(_runtime_errors(svc_cfg.get("runtime_packages"), observed))
    except (OSError, RuntimeContractError) as exc:
        errors.append(str(exc))

    if svc_cfg.get("method") == "systemd":
        try:
            effective = properties if properties is not None else _systemd_properties()
            errors.extend(_unit_errors(svc_cfg, effective))
        except (OSError, RuntimeContractError, subprocess.TimeoutExpired) as exc:
            errors.append(str(exc))
            effective = {}
    else:
        effective = {}

    results_text = svc_cfg.get("results_dir")
    results_dir = Path(results_text) if results_text else None
    active_latches = [
        name for name in _STOP_SENTINELS
        if results_dir is not None and (results_dir / name).exists()
    ]
    if active_latches and svc_cfg.get("method") == "systemd":
        if effective.get("ActiveState") == "active":
            errors.append("latched_systemd_unit_active")
        if effective.get("UnitFileState") in {"enabled", "enabled-runtime"}:
            errors.append("latched_systemd_unit_enabled")

    errors = sorted(set(errors))
    return {
        "schema_version": 1,
        "contract_ok": not errors,
        "startup_allowed": not errors and not active_latches,
        "errors": errors,
        "active_latches": active_latches,
    }


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    startup_check = argv == ["--startup-check"]
    capture_controller = argv == ["--capture-controller"]
    if argv and not startup_check and not capture_controller:
        print(
            "usage: python -m orze.service.runtime_contract "
            "[--startup-check|--capture-controller]",
            file=sys.stderr,
        )
        return 2
    if capture_controller:
        try:
            report = capture_controller_runtime_contract()
        except (OSError, RuntimeContractError) as exc:
            print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
            return 1
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    svc_cfg = load_service_config()
    if not svc_cfg:
        report = {
            "schema_version": 1,
            "contract_ok": False,
            "startup_allowed": False,
            "errors": ["service_config_missing"],
            "active_latches": [],
        }
    else:
        report = audit_runtime_contract(svc_cfg)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if (report["startup_allowed"] if startup_check
                 else report["contract_ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
