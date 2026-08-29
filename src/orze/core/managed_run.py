"""Read-only admission for one explicitly selected managed idea run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


class ManagedRunError(RuntimeError):
    """Raised before orchestration state or GPU telemetry may be touched."""


def _present(path: Path) -> bool:
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return True


_EVIDENCE_REQUIREMENTS = (
    "require_data_separation",
    "require_model_lineage",
    "require_benchmark_contract",
    "require_explicit_untainted_metrics",
    "require_clean_training_access_log",
)


def _evidence_policy(cfg: Mapping) -> Mapping:
    policy = cfg.get("managed_run", {})
    if not isinstance(policy, Mapping) or any(
            not isinstance(policy.get(key, False), bool)
            for key in _EVIDENCE_REQUIREMENTS):
        raise ManagedRunError("managed_run_evidence_policy_invalid")
    if set(policy) - set(_EVIDENCE_REQUIREMENTS):
        raise ManagedRunError("managed_run_evidence_policy_invalid")
    return policy


def _require_evidence_contracts(cfg: Mapping) -> Mapping:
    policy = _evidence_policy(cfg)
    separation = cfg.get("data_separation") or {}
    lineage = cfg.get("model_lineage") or {}
    report = cfg.get("report") or {}
    if (policy.get("require_data_separation") is True
            and (not isinstance(separation, Mapping)
                 or separation.get("enabled") is not True)):
        raise ManagedRunError("managed_run_data_separation_required")
    if (policy.get("require_model_lineage") is True
            and (not isinstance(lineage, Mapping)
                 or lineage.get("enabled") is not True)):
        raise ManagedRunError("managed_run_model_lineage_required")
    if (policy.get("require_benchmark_contract") is True
            and (not isinstance(report, Mapping)
                 or not isinstance(report.get("benchmark_contract"), Mapping))):
        raise ManagedRunError("managed_run_benchmark_contract_required")
    if (policy.get("require_clean_training_access_log") is True
            and policy.get("require_model_lineage") is not True):
        raise ManagedRunError(
            "managed_run_clean_access_log_requires_model_lineage")
    return policy


def prepare_managed_idea_run(
    cfg: Mapping,
    idea_id: str,
    gpu: int,
) -> dict:
    """Authorize one queued idea without mutating project or GPU state.

    This is the entry boundary for ``orze run-idea``. The ordinary launcher
    independently repeats all time-sensitive checks before GPU telemetry.
    """
    if not isinstance(cfg, dict):
        raise ManagedRunError("managed_run_config_invalid")

    from orze.core.ideas import IDEA_ID_PATTERN
    import re
    if (not isinstance(idea_id, str) or len(idea_id) > 128
            or re.fullmatch(IDEA_ID_PATTERN, idea_id) is None):
        raise ManagedRunError("managed_run_idea_id_invalid")
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise ManagedRunError("managed_run_gpu_invalid")

    # Stop state is the cheapest and most authoritative refusal. Check it
    # before config/runtime hashing so a deliberately stopped project stays
    # inert even while its prospective evidence policy is incomplete.
    results_dir = Path(cfg.get("results_dir", "orze_results"))
    for sentinel in (".orze_disabled", ".orze_stop_all", ".orze_shutdown"):
        if _present(results_dir / sentinel):
            raise ManagedRunError(
                f"managed_run_blocked_by_sentinel:{sentinel}")
    launcher = cfg.get("launcher")
    if not isinstance(launcher, dict):
        raise ManagedRunError("managed_run_launcher_policy_invalid")
    if launcher.get("paused") is True:
        raise ManagedRunError("managed_run_blocked_by_pause_policy")
    pause_path = launcher.get("paused_flag_path")
    if isinstance(pause_path, str) and pause_path:
        pause_path = Path(pause_path)
        if not pause_path.is_absolute():
            pause_path = results_dir / pause_path
    else:
        pause_path = results_dir / "_launcher_paused.flag"
    if _present(pause_path):
        raise ManagedRunError("managed_run_blocked_by_pause_policy")

    from orze.core.config import _validate_config
    errors, _ = _validate_config(cfg)
    if errors:
        raise ManagedRunError("managed_run_config_validation_failed")

    contract = cfg.get("controller_runtime")
    if contract is None:
        raise ManagedRunError("managed_run_controller_runtime_pin_required")
    from orze.service.runtime_contract import (
        RuntimeContractError,
        require_controller_runtime_contract,
    )
    try:
        require_controller_runtime_contract(contract)
    except RuntimeContractError as exc:
        raise ManagedRunError(str(exc)) from exc
    _require_evidence_contracts(cfg)

    # Pure scope validation; this helper never calls nvidia-smi.
    from orze.engine.launcher import LaunchIntegrityError, _assert_gpu_authorized
    scoped = dict(cfg)
    scoped["_managed_gpu_ids"] = [gpu]
    try:
        _assert_gpu_authorized(gpu, scoped)
    except LaunchIntegrityError as exc:
        raise ManagedRunError(str(exc)) from exc

    lake_path = Path(cfg.get("idea_lake_db") or results_dir / "idea_lake.db")
    from orze.reporting.evidence import authoritative_idea_lifecycle
    lifecycle, reason = authoritative_idea_lifecycle(lake_path, [idea_id])
    if reason != "authoritative_lifecycle_loaded":
        raise ManagedRunError(f"managed_run_{reason}")
    row = lifecycle.get(idea_id) if isinstance(lifecycle, Mapping) else None
    if not isinstance(row, Mapping):
        raise ManagedRunError("managed_run_authoritative_lifecycle_rows_missing")
    if row.get("state") != "QUEUED":
        raise ManagedRunError("managed_run_idea_not_queued")

    idea_dir = results_dir / idea_id
    if (_present(idea_dir / "claim.json")
            or _present(idea_dir / "metrics.json")):
        raise ManagedRunError("managed_run_idea_already_attempted")

    from orze.core.decision_batches import validate_idea_decision_admission
    decision_error = validate_idea_decision_admission(
        results_dir, cfg, idea_id)
    if decision_error:
        raise ManagedRunError(decision_error)

    return {
        "schema_version": 1,
        "authorized": True,
        "idea_id": idea_id,
        "gpu": gpu,
        "lifecycle_state": "QUEUED",
        "approach_family": row.get("family") or "other",
    }


def verify_managed_idea_outcome(cfg: Mapping, idea_id: str) -> dict:
    """Require a truthful terminal outcome before ``run-idea`` exits zero."""
    policy = _require_evidence_contracts(cfg)
    results_dir = Path(cfg.get("results_dir", "orze_results"))
    idea_dir = results_dir / idea_id
    metrics_path = idea_dir / "metrics.json"
    if metrics_path.is_symlink():
        raise ManagedRunError("managed_run_metrics_redirected")
    try:
        if metrics_path.stat().st_nlink != 1:
            raise ManagedRunError("managed_run_metrics_redirected")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManagedRunError("managed_run_metrics_missing") from exc
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise ManagedRunError("managed_run_metrics_invalid") from exc
    if not isinstance(metrics, dict) or metrics.get("status") != "COMPLETED":
        raise ManagedRunError("managed_run_training_not_completed")
    if metrics.get("tainted_leakage") is True:
        raise ManagedRunError("managed_run_tainted_leakage")
    if (policy.get("require_explicit_untainted_metrics") is True
            and metrics.get("tainted_leakage") is not False):
        raise ManagedRunError(
            "managed_run_explicit_untainted_metrics_required")
    access_log = None
    if policy.get("require_clean_training_access_log") is True:
        from orze.data_boundaries import audit_training_access_log
        access_log = audit_training_access_log(idea_dir)
        if access_log.get("status") != "CLEAN":
            raise ManagedRunError(
                "managed_run_training_access_log_not_clean")

    lake_path = Path(cfg.get("idea_lake_db") or results_dir / "idea_lake.db")
    from orze.reporting.evidence import authoritative_idea_lifecycle
    lifecycle, reason = authoritative_idea_lifecycle(lake_path, [idea_id])
    if reason != "authoritative_lifecycle_loaded":
        raise ManagedRunError(f"managed_run_{reason}")
    row = lifecycle.get(idea_id) if isinstance(lifecycle, Mapping) else None
    if not isinstance(row, Mapping):
        raise ManagedRunError("managed_run_authoritative_lifecycle_rows_missing")
    if row.get("state") != "COMPLETE":
        raise ManagedRunError("managed_run_lifecycle_not_complete")

    if cfg.get("eval_script"):
        eval_output = idea_dir / str(
            cfg.get("eval_output") or "eval_report.json")
        if (not eval_output.is_file() or eval_output.is_symlink()
                or eval_output.stat().st_nlink != 1):
            raise ManagedRunError("managed_run_evaluation_output_missing")

    for post_script in (cfg.get("post_scripts") or []):
        if not isinstance(post_script, Mapping):
            raise ManagedRunError("managed_run_post_script_policy_invalid")
        output = post_script.get("output")
        if not isinstance(output, str) or not output:
            continue
        output_path = idea_dir / output
        if (not output_path.is_file() or output_path.is_symlink()
                or output_path.stat().st_nlink != 1):
            raise ManagedRunError("managed_run_post_script_output_missing")

    lineage = cfg.get("model_lineage") or {}
    if isinstance(lineage, Mapping) and lineage.get("enabled") is True:
        from orze.core.model_lineage import (
            ModelLineageError,
            validate_model_lineage_for_evaluation,
        )
        try:
            validate_model_lineage_for_evaluation(idea_dir, cfg)
        except ModelLineageError as exc:
            raise ManagedRunError(str(exc)) from exc

    if policy.get("require_data_separation") is True:
        from orze.core.data_separation import (
            DataSeparationError,
            ensure_data_separation,
        )
        try:
            ensure_data_separation(cfg)
        except DataSeparationError as exc:
            raise ManagedRunError(str(exc)) from exc

    report = cfg.get("report") or {}
    if isinstance(report, Mapping) and report.get("benchmark_contract"):
        from orze.core.benchmark_contract import validate_benchmark_receipt
        valid, benchmark_reason = validate_benchmark_receipt(idea_dir, cfg)
        if not valid:
            raise ManagedRunError(
                str(benchmark_reason or "managed_run_benchmark_receipt_invalid"))

    return {
        "schema_version": 1,
        "completed": True,
        "idea_id": idea_id,
        "lifecycle_state": "COMPLETE",
        "evaluation_required": bool(cfg.get("eval_script")),
        "benchmark_contract_required": bool(
            isinstance(report, Mapping) and report.get("benchmark_contract")),
        "model_lineage_required": bool(
            isinstance(lineage, Mapping) and lineage.get("enabled") is True),
        "explicit_untainted_metrics_required": bool(
            policy.get("require_explicit_untainted_metrics") is True),
        "clean_training_access_log_required": bool(
            policy.get("require_clean_training_access_log") is True),
        "training_access_log": access_log,
    }
