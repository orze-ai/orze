"""Fail-closed provenance for benchmark-comparable evaluation results.

The contract deliberately does not try to infer an external leaderboard rank.
It validates that a local result carries nonce-bound evidence from the configured
evaluator for the exact benchmark revision/view and model form. A durable
exposure ledger also distinguishes adaptive development from one untouched
confirmation and enforces the preregistered look budget. Reporting code may then
compute a *local* ordering among contract-compliant runs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import socket
import time
from pathlib import Path
from typing import Mapping, Optional

from orze.core.fs import _fs_lock, _fs_unlock, atomic_write, deep_get


SCHEMA_VERSION = 1
PROVENANCE_FILE = "_benchmark_evaluation.json"
EXPOSURE_LEDGER_FILE = "_benchmark_exposures.jsonl"
EXPOSURE_LOCK_DIR = ".benchmark_exposure.lock"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMMUTABLE_REVISION_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_MODEL_FORM = "single_model_single_pass"
_EVIDENCE_SCOPES = {"development_proxy", "local_reproduction"}
_SELECTION_MODES = {"adaptive", "confirmation"}


class BenchmarkContractError(RuntimeError):
    """Raised before evaluation when benchmark provenance cannot be trusted."""


def get_benchmark_contract(cfg: Mapping) -> Optional[dict]:
    report = cfg.get("report") or {}
    contract = report.get("benchmark_contract")
    return contract if isinstance(contract, dict) else None


def _project_path(cfg: Mapping, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(cfg.get("_project_root") or ".") / path
    return path.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _idea_evidence_path(idea_dir: Path, relative_name: str) -> Optional[Path]:
    """Resolve an idea-local evidence path without following any symlink."""
    base = Path(idea_dir)
    relative = Path(str(relative_name))
    if (base.is_symlink() or relative.is_absolute() or relative == Path(".")
            or ".." in relative.parts):
        return None
    current = base
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return None
    return current


def validate_benchmark_contract_config(cfg: Mapping) -> list[str]:
    """Return configuration errors for ``report.benchmark_contract``."""
    report = cfg.get("report") or {}
    contract = report.get("benchmark_contract")
    if contract is None:
        return []
    prefix = "report.benchmark_contract"
    if not isinstance(contract, dict):
        return [f"{prefix}: must be a mapping"]

    errors: list[str] = []
    for key in ("benchmark_id", "view"):
        if not isinstance(contract.get(key), str) or not contract[key].strip():
            errors.append(f"{prefix}.{key}: must be a non-empty string")

    revision = contract.get("revision")
    if (not isinstance(revision, str)
            or _IMMUTABLE_REVISION_RE.fullmatch(revision.lower()) is None):
        errors.append(
            f"{prefix}.revision: must be an immutable 40- or 64-character hex revision"
        )

    required = contract.get("required_metrics")
    if (not isinstance(required, list) or not required
            or not all(isinstance(key, str) and key for key in required)
            or len(required) != len(set(required))):
        errors.append(
            f"{prefix}.required_metrics: must be a non-empty list of unique strings"
        )
        required = []

    receipt = contract.get("receipt")
    if (not isinstance(receipt, str) or not receipt
            or Path(receipt) == Path(".") or Path(receipt).is_absolute()
            or ".." in Path(receipt).parts):
        errors.append(
            f"{prefix}.receipt: must be a relative path inside each idea directory"
        )

    if contract.get("model_form") != _MODEL_FORM:
        errors.append(
            f"{prefix}.model_form: must be '{_MODEL_FORM}'"
        )
    if contract.get("aggregate") != "macro_mean":
        errors.append(f"{prefix}.aggregate: must be 'macro_mean'")
    managed_lineage = contract.get("managed_model_lineage", False)
    if not isinstance(managed_lineage, bool):
        errors.append(
            f"{prefix}.managed_model_lineage: must be true or false")
    elif managed_lineage:
        lineage = cfg.get("model_lineage", {})
        if (not isinstance(lineage, Mapping)
                or lineage.get("enabled") is not True):
            errors.append(
                f"{prefix}.managed_model_lineage: requires "
                "model_lineage.enabled: true")
    evidence_scope = contract.get("evidence_scope")
    if evidence_scope not in _EVIDENCE_SCOPES:
        errors.append(
            f"{prefix}.evidence_scope: must be one of "
            + ", ".join(sorted(_EVIDENCE_SCOPES))
        )
    selection_mode = contract.get("selection_mode")
    if selection_mode not in _SELECTION_MODES:
        errors.append(
            f"{prefix}.selection_mode: must be one of "
            + ", ".join(sorted(_SELECTION_MODES))
        )
    prior_exposures = contract.get("prior_exposures")
    if (isinstance(prior_exposures, bool)
            or not isinstance(prior_exposures, int) or prior_exposures < 0):
        errors.append(
            f"{prefix}.prior_exposures: must be a non-negative integer"
        )
    max_evaluations = contract.get("max_evaluations")
    if (isinstance(max_evaluations, bool)
            or not isinstance(max_evaluations, int) or max_evaluations < 1):
        errors.append(
            f"{prefix}.max_evaluations: must be a positive integer"
        )
    if (isinstance(prior_exposures, int)
            and not isinstance(prior_exposures, bool)
            and isinstance(max_evaluations, int)
            and not isinstance(max_evaluations, bool)
            and prior_exposures > max_evaluations):
        errors.append(
            f"{prefix}: prior_exposures cannot exceed max_evaluations"
        )
    if selection_mode == "confirmation":
        if prior_exposures != 0:
            errors.append(
                f"{prefix}: confirmation requires prior_exposures: 0; "
                "otherwise the benchmark is already "
                "adaptive/benchmark-fitted"
            )
        if max_evaluations != 1:
            errors.append(
                f"{prefix}: confirmation requires max_evaluations: 1; "
                "additional looks must be labeled adaptive"
            )
    tolerance = contract.get("aggregate_tolerance", 1e-6)
    if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
            or not math.isfinite(float(tolerance)) or tolerance < 0):
        errors.append(
            f"{prefix}.aggregate_tolerance: must be a finite non-negative number"
        )

    for digest_key in (
        "evaluator_sha256", "dataset_manifest_sha256", "scorer_sha256",
    ):
        digest_value = contract.get(digest_key)
        if (not isinstance(digest_value, str)
                or _SHA256_RE.fullmatch(digest_value.lower()) is None):
            errors.append(f"{prefix}.{digest_key}: must be a SHA-256 digest")
    evaluator_digest = contract.get("evaluator_sha256")

    eval_script = cfg.get("eval_script")
    if not isinstance(eval_script, str) or not eval_script:
        errors.append(f"{prefix}: requires eval_script")
    else:
        eval_path = _project_path(cfg, eval_script)
        if not eval_path.is_file():
            errors.append(f"{prefix}: eval_script not found: {eval_path}")
        elif isinstance(evaluator_digest, str) and _SHA256_RE.fullmatch(
                evaluator_digest.lower()):
            actual = _sha256_file(eval_path)
            if actual != evaluator_digest.lower():
                errors.append(
                    f"{prefix}.evaluator_sha256: configured {evaluator_digest.lower()} "
                    f"does not match {eval_path} ({actual})"
                )

        pinned = cfg.get("sealed_hashes") or {}
        pinned_digest = pinned.get(eval_script) if isinstance(pinned, dict) else None
        if (not isinstance(pinned_digest, str)
                or not isinstance(evaluator_digest, str)
                or pinned_digest.lower() != evaluator_digest.lower()):
            errors.append(
                f"{prefix}: eval_script must be pinned to the same digest in sealed_hashes"
            )

    columns = report.get("columns") or []
    column_keys = {
        column.get("key") for column in columns if isinstance(column, dict)
    }
    missing_columns = sorted(set(required) - column_keys)
    if missing_columns:
        errors.append(
            f"{prefix}.required_metrics missing from report.columns: "
            + ", ".join(missing_columns)
        )
    for column in columns:
        if (not isinstance(column, dict)
                or column.get("key") not in set(required)):
            continue
        source = column.get("source", "")
        if source and ":" in source:
            filename = source.split(":", 1)[0]
            source_path = Path(filename)
            if (not filename or source_path == Path(".")
                    or source_path.is_absolute()
                    or ".." in source_path.parts):
                errors.append(
                    f"report.columns source for {column['key']}: must stay "
                    "inside the idea directory"
                )
    primary = report.get("primary_metric")
    if primary in set(required):
        errors.append(
            f"{prefix}.required_metrics: primary_metric must be the aggregate, "
            "not one of its component metrics"
        )
    min_datasets = report.get("min_datasets", 0)
    if (isinstance(min_datasets, bool) or not isinstance(min_datasets, int)
            or min_datasets < len(required)):
        errors.append(
            f"report.min_datasets: must be at least {len(required)} when the "
            "benchmark contract is enabled"
        )
    return errors


def _exposure_dataset_identity(contract: Mapping) -> dict:
    """Identity that cannot be reset by relabeling the same sealed dataset."""
    return {
        "dataset_manifest_sha256": str(
            contract["dataset_manifest_sha256"]
        ).lower(),
    }


def _exposure_base_identity(contract: Mapping) -> dict:
    return {
        **_exposure_dataset_identity(contract),
        "benchmark_id": contract["benchmark_id"],
        "benchmark_revision": str(contract["revision"]).lower(),
        "benchmark_view": contract["view"],
        "scorer_sha256": str(contract["scorer_sha256"]).lower(),
        "evidence_scope": contract["evidence_scope"],
        "selection_mode": contract["selection_mode"],
    }


def _exposure_identity(contract: Mapping) -> dict:
    return {
        **_exposure_base_identity(contract),
        "prior_exposures": int(contract["prior_exposures"]),
        "max_evaluations": int(contract["max_evaluations"]),
    }


def benchmark_exposure_ledger_path(cfg: Mapping) -> Path:
    """Return the one project-scoped exposure ledger path.

    The location is derived rather than configurable so changing a campaign's
    results directory cannot silently create fresh benchmark history.
    """
    project_root = Path(cfg.get("_project_root") or ".").resolve()
    expected_orze_dir = project_root / ".orze"
    configured_orze_dir = Path(
        cfg.get("_orze_dir") or expected_orze_dir
    ).resolve()
    if configured_orze_dir != expected_orze_dir:
        raise BenchmarkContractError(
            "benchmark_exposure_control_directory_drift"
        )
    if expected_orze_dir.is_symlink():
        raise BenchmarkContractError(
            "benchmark_exposure_control_directory_symlink_forbidden"
        )
    return expected_orze_dir / EXPOSURE_LEDGER_FILE


def _legacy_exposure_ledger_paths(
        results_dir: Path, cfg: Mapping) -> list[Path]:
    """Find ledgers written by the brief result-local v1 implementation."""
    project_path = benchmark_exposure_ledger_path(cfg)
    project_root = project_path.parent.parent
    candidates = {Path(results_dir) / EXPOSURE_LEDGER_FILE}
    try:
        candidates.update(project_root.glob(f"*/{EXPOSURE_LEDGER_FILE}"))
    except OSError as exc:
        raise BenchmarkContractError(
            "benchmark_exposure_project_directory_unreadable"
        ) from exc
    return sorted(
        (path for path in candidates
         if path != project_path and (path.exists() or path.is_symlink())),
        key=lambda path: str(path),
    )


def benchmark_exposure_evidence_paths(
        results_dir: Path, cfg: Mapping) -> list[Path]:
    """Return every current or legacy ledger that affects report integrity."""
    return [
        benchmark_exposure_ledger_path(cfg),
        *_legacy_exposure_ledger_paths(results_dir, cfg),
    ]


def _read_exposure_ledger_path(path: Path) -> list[dict]:
    if path.is_symlink():
        raise BenchmarkContractError(
            "benchmark_exposure_ledger_symlink_forbidden"
        )
    if not path.exists():
        return []
    records = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise BenchmarkContractError(
            "benchmark_exposure_ledger_unreadable"
        ) from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise BenchmarkContractError(
                f"benchmark_exposure_ledger_blank_line:{line_number}"
            )
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchmarkContractError(
                f"benchmark_exposure_ledger_corrupt:{line_number}"
            ) from exc
        if not isinstance(record, dict) or record.get("schema_version") != 1:
            raise BenchmarkContractError(
                f"benchmark_exposure_ledger_schema_invalid:{line_number}"
            )
        declared_hash = record.get("record_sha256")
        canonical_record = dict(record)
        canonical_record.pop("record_sha256", None)
        actual_hash = hashlib.sha256(json.dumps(
            canonical_record, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        if (not isinstance(declared_hash, str)
                or _SHA256_RE.fullmatch(declared_hash) is None
                or declared_hash != actual_hash):
            raise BenchmarkContractError(
                f"benchmark_exposure_ledger_integrity_invalid:{line_number}"
            )
        records.append(record)
    hashes = [record["record_sha256"] for record in records]
    if len(hashes) != len(set(hashes)):
        raise BenchmarkContractError(
            "benchmark_exposure_ledger_duplicate_record"
        )
    return records


def _write_initial_project_ledger(path: Path, content: bytes) -> None:
    """Create the project ledger once when migrating result-local history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise BenchmarkContractError(
            "benchmark_exposure_control_directory_symlink_forbidden"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(path), flags, 0o600)
    except FileExistsError:
        return
    except OSError as exc:
        raise BenchmarkContractError(
            "benchmark_exposure_ledger_unwritable"
        ) from exc
    try:
        written = 0
        while written < len(content):
            count = os.write(fd, content[written:])
            if count <= 0:
                raise BenchmarkContractError(
                    "benchmark_exposure_ledger_short_write"
                )
            written += count
        os.fsync(fd)
    finally:
        os.close(fd)
    parent_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _read_exposure_records(
        results_dir: Path, cfg: Mapping, *, migrate: bool = False) -> list[dict]:
    """Read one project history and fail closed on divergent legacy copies."""
    project_path = benchmark_exposure_ledger_path(cfg)
    project_records = _read_exposure_ledger_path(project_path)
    legacy_paths = _legacy_exposure_ledger_paths(results_dir, cfg)
    legacy_records = [
        (path, _read_exposure_ledger_path(path)) for path in legacy_paths
    ]

    if project_path.exists():
        project_hashes = [record["record_sha256"] for record in project_records]
        for _, records in legacy_records:
            hashes = [record["record_sha256"] for record in records]
            if hashes != project_hashes[:len(hashes)]:
                raise BenchmarkContractError(
                    "benchmark_exposure_legacy_ledger_conflict"
                )
        return project_records

    nonempty = [(path, records) for path, records in legacy_records if records]
    if not nonempty:
        return []
    reference_hashes = [
        record["record_sha256"] for record in nonempty[0][1]
    ]
    if any(
        [record["record_sha256"] for record in records] != reference_hashes
        for _, records in nonempty[1:]
    ):
        raise BenchmarkContractError(
            "benchmark_exposure_legacy_ledger_conflict"
        )
    if migrate:
        try:
            content = nonempty[0][0].read_bytes()
        except OSError as exc:
            raise BenchmarkContractError(
                "benchmark_exposure_ledger_unreadable"
            ) from exc
        _write_initial_project_ledger(project_path, content)
        migrated = _read_exposure_ledger_path(project_path)
        if ([record["record_sha256"] for record in migrated]
                != reference_hashes):
            raise BenchmarkContractError(
                "benchmark_exposure_legacy_migration_mismatch"
            )
        return migrated
    return nonempty[0][1]


def _matching_exposures(records: list[dict], contract: Mapping) -> list[dict]:
    identity = _exposure_identity(contract)
    return [
        record for record in records
        if all(record.get(key) == value for key, value in identity.items())
    ]


def _validated_matching_exposures(
        records: list[dict], contract: Mapping) -> list[dict]:
    dataset_identity = _exposure_dataset_identity(contract)
    policy_records = [
        record for record in records
        if all(record.get(key) == value
               for key, value in dataset_identity.items())
    ]
    expected_policy = _exposure_identity(contract)
    if any(any(record.get(key) != value
               for key, value in expected_policy.items())
           for record in policy_records):
        raise BenchmarkContractError(
            "benchmark_exposure_policy_drift"
        )
    matching = _matching_exposures(records, contract)
    prior = int(contract["prior_exposures"])
    raw_ordinals = [record.get("exposure_ordinal") for record in matching]
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in raw_ordinals):
        raise BenchmarkContractError(
            "benchmark_exposure_ledger_ordinal_sequence_invalid"
        )
    ordinals = sorted(raw_ordinals)
    expected = list(range(prior + 1, prior + len(matching) + 1))
    if ordinals != expected:
        raise BenchmarkContractError(
            "benchmark_exposure_ledger_ordinal_sequence_invalid"
        )
    maximum = int(contract["max_evaluations"])
    if prior + len(matching) > maximum:
        raise BenchmarkContractError(
            f"benchmark_exposure_budget_exhausted:"
            f"{prior + len(matching)}/{maximum}"
        )
    return matching


def _audit_exposure_provenance_links(
        results_dir: Path, records: list[dict], contract: Mapping) -> None:
    """Reject deletion of ledger rows still referenced by idea provenance."""
    matching_hashes = {
        record["record_sha256"]
        for record in _matching_exposures(records, contract)
    }
    identity = _exposure_identity(contract)
    try:
        children = list(Path(results_dir).iterdir())
    except OSError as exc:
        raise BenchmarkContractError(
            "benchmark_exposure_results_directory_unreadable"
        ) from exc
    for child in children:
        if child.is_symlink():
            if (child / PROVENANCE_FILE).exists():
                raise BenchmarkContractError(
                    f"benchmark_exposure_provenance_symlink:{child.name}"
                )
            continue
        if not child.is_dir() or child.name == EXPOSURE_LOCK_DIR:
            continue
        provenance_path = child / PROVENANCE_FILE
        if not provenance_path.exists():
            continue
        try:
            provenance = json.loads(
                provenance_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            raise BenchmarkContractError(
                f"benchmark_exposure_provenance_invalid:{child.name}"
            ) from exc
        if not isinstance(provenance, dict):
            raise BenchmarkContractError(
                f"benchmark_exposure_provenance_invalid:{child.name}"
            )
        if not all(provenance.get(key) == value
                   for key, value in identity.items()):
            continue
        if provenance.get("exposure_record_sha256") not in matching_hashes:
            raise BenchmarkContractError(
                f"benchmark_exposure_record_deleted:{child.name}"
            )


def _reserve_benchmark_exposure(
    idea_dir: Path,
    cfg: Mapping,
    contract: Mapping,
    evaluator_sha256: str,
    nonce: str,
) -> tuple[int, str]:
    """Atomically reserve one benchmark look before the evaluator starts."""
    results_dir = Path(idea_dir).parent
    ledger_path = benchmark_exposure_ledger_path(cfg)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lock_dir = ledger_path.parent / EXPOSURE_LOCK_DIR
    if lock_dir.is_symlink():
        raise BenchmarkContractError(
            "benchmark_exposure_lock_symlink_forbidden"
        )
    if not _fs_lock(lock_dir, stale_seconds=300):
        raise BenchmarkContractError("benchmark_exposure_ledger_locked")
    try:
        records = _read_exposure_records(results_dir, cfg, migrate=True)
        matching = _validated_matching_exposures(records, contract)
        _audit_exposure_provenance_links(results_dir, records, contract)
        prior = int(contract["prior_exposures"])
        ordinal = prior + len(matching) + 1
        maximum = int(contract["max_evaluations"])
        if ordinal > maximum:
            raise BenchmarkContractError(
                f"benchmark_exposure_budget_exhausted:{ordinal - 1}/{maximum}"
            )
        nonce_sha256 = hashlib.sha256(nonce.encode("utf-8")).hexdigest()
        record = {
            "schema_version": 1,
            **_exposure_identity(contract),
            "exposure_ordinal": ordinal,
            "idea_id": Path(idea_dir).name,
            "evaluation_nonce_sha256": nonce_sha256,
            "evaluator_sha256": evaluator_sha256,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "reserved_at_unix_ns": time.time_ns(),
        }
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
        record_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        record["record_sha256"] = record_sha256
        existed = ledger_path.exists()
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(str(ledger_path), flags, 0o600)
        except OSError as exc:
            raise BenchmarkContractError(
                "benchmark_exposure_ledger_unwritable"
            ) from exc
        try:
            line = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
            written = 0
            while written < len(line):
                count = os.write(fd, line[written:])
                if count <= 0:
                    raise BenchmarkContractError(
                        "benchmark_exposure_ledger_short_write"
                    )
                written += count
            os.fsync(fd)
        finally:
            os.close(fd)
        if not existed:
            parent_fd = os.open(str(ledger_path.parent), os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        return ordinal, record_sha256
    finally:
        _fs_unlock(lock_dir)


def benchmark_exposure_summary(results_dir: Path, cfg: Mapping) -> dict:
    """Return non-secret exposure accounting for reporting and status."""
    contract = get_benchmark_contract(cfg)
    if contract is None:
        return {"enabled": False}
    try:
        all_records = _read_exposure_records(Path(results_dir), cfg)
        records = _validated_matching_exposures(all_records, contract)
        _audit_exposure_provenance_links(
            Path(results_dir), all_records, contract,
        )
    except BenchmarkContractError as exc:
        return {
            "enabled": True,
            "valid": False,
            "reason": str(exc),
        }
    prior = int(contract.get("prior_exposures", 0))
    observed = len(records)
    maximum = int(contract.get("max_evaluations", 0))
    return {
        "enabled": True,
        "valid": True,
        "evidence_scope": contract.get("evidence_scope"),
        "selection_mode": contract.get("selection_mode"),
        "prior_exposures": prior,
        "managed_exposures": observed,
        "total_exposures": prior + observed,
        "max_evaluations": maximum,
        "remaining": max(0, maximum - prior - observed),
        "benchmark_fitted": (
            contract.get("selection_mode") == "adaptive" or prior > 0
        ),
    }


def prepare_benchmark_evaluation(idea_dir: Path, cfg: Mapping) -> dict[str, str]:
    """Pin evaluator identity and return nonce-bearing child environment.

    A pre-existing receipt is rejected.  This prevents training or a previous
    attempt from placing a document that the evaluator never produced.
    """
    contract = get_benchmark_contract(cfg)
    if contract is None:
        return {}
    if (contract.get("selection_mode") == "confirmation"
            and (contract.get("prior_exposures") != 0
                 or contract.get("max_evaluations") != 1)):
        raise BenchmarkContractError(
            "benchmark_confirmation_policy_invalid"
        )

    receipt_path = _idea_evidence_path(
        Path(idea_dir), str(contract["receipt"]))
    provenance_path = _idea_evidence_path(Path(idea_dir), PROVENANCE_FILE)
    if receipt_path is None or provenance_path is None:
        raise BenchmarkContractError(
            "benchmark evidence path redirected before evaluation"
        )
    if receipt_path.exists():
        raise BenchmarkContractError(
            f"benchmark receipt existed before evaluation: {receipt_path}"
        )

    lineage = None
    lineage_sha256 = None
    if contract.get("managed_model_lineage") is True:
        try:
            from orze.core.model_lineage import (
                validate_model_lineage_for_evaluation,
            )
            lineage, lineage_sha256 = validate_model_lineage_for_evaluation(
                Path(idea_dir), cfg)
        except Exception as exc:
            raise BenchmarkContractError(
                "benchmark_managed_model_lineage_invalid") from exc

    eval_path = _project_path(cfg, str(cfg["eval_script"]))
    actual_digest = _sha256_file(eval_path)
    expected_digest = str(contract["evaluator_sha256"]).lower()
    if actual_digest != expected_digest:
        raise BenchmarkContractError(
            "benchmark evaluator hash drift: "
            f"expected {expected_digest}, got {actual_digest}"
        )

    nonce = secrets.token_hex(32)
    exposure_ordinal, exposure_record_sha256 = _reserve_benchmark_exposure(
        Path(idea_dir), cfg, contract, actual_digest, nonce,
    )
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": contract["benchmark_id"],
        "benchmark_revision": str(contract["revision"]).lower(),
        "benchmark_view": contract["view"],
        "evaluator_sha256": actual_digest,
        "dataset_manifest_sha256": str(
            contract["dataset_manifest_sha256"]).lower(),
        "scorer_sha256": str(contract["scorer_sha256"]).lower(),
        "evidence_scope": contract["evidence_scope"],
        "selection_mode": contract["selection_mode"],
        "prior_exposures": int(contract["prior_exposures"]),
        "max_evaluations": int(contract["max_evaluations"]),
        "exposure_ordinal": exposure_ordinal,
        "exposure_record_sha256": exposure_record_sha256,
        "evaluation_nonce": nonce,
        "pid": os.getpid(),
    }
    if lineage is not None:
        provenance.update({
            "managed_model_lineage_sha256": lineage_sha256,
            "model_artifact_sha256": lineage["artifact_sha256"],
        })
    atomic_write(
        provenance_path,
        json.dumps(provenance, sort_keys=True, indent=2) + "\n",
    )
    child_env = {
        "ORZE_BENCHMARK_EVALUATION_NONCE": nonce,
        "ORZE_BENCHMARK_RECEIPT": str(receipt_path.resolve()),
        "ORZE_BENCHMARK_EXPOSURE_ORDINAL": str(exposure_ordinal),
        "ORZE_BENCHMARK_EXPOSURE_RECORD_SHA256": exposure_record_sha256,
    }
    if lineage is not None:
        child_env.update({
            "ORZE_MANAGED_MODEL_LINEAGE_SHA256": str(lineage_sha256),
            "ORZE_MODEL_ARTIFACT_SHA256": str(lineage["artifact_sha256"]),
        })
    return child_env


def load_benchmark_values(idea_dir: Path, cfg: Mapping) -> dict:
    """Load configured report values without fallback or inferred aliases."""
    values = {}
    metrics = {}
    metrics_path = _idea_evidence_path(Path(idea_dir), "metrics.json")
    try:
        if metrics_path is None:
            raise OSError("redirected metrics path")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        metrics = {}
    if not isinstance(metrics, dict):
        metrics = {}

    for column in (cfg.get("report") or {}).get("columns") or []:
        if not isinstance(column, dict) or not column.get("key"):
            continue
        key = column["key"]
        source = column.get("source", "")
        if source and ":" in source:
            filename, dotpath = source.split(":", 1)
            source_path = _idea_evidence_path(Path(idea_dir), filename)
            try:
                if source_path is None:
                    raise OSError("redirected metric source")
                document = json.loads(
                    source_path.read_text(encoding="utf-8")
                )
                values[key] = deep_get(document, dotpath)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                values[key] = None
        else:
            values[key] = deep_get(metrics, key) if "." in key else metrics.get(key)
    return values


def validate_benchmark_receipt(
    idea_dir: Path,
    cfg: Mapping,
    values: Optional[Mapping] = None,
) -> tuple[bool, str]:
    """Validate evaluator provenance, exact coverage, and single-model form."""
    contract = get_benchmark_contract(cfg)
    if contract is None:
        return True, "benchmark_contract_disabled"

    if Path(idea_dir).is_symlink():
        return False, "benchmark_idea_dir_symlink_forbidden"
    provenance_path = _idea_evidence_path(Path(idea_dir), PROVENANCE_FILE)
    if provenance_path is None:
        return False, "benchmark_provenance_symlink_forbidden"

    try:
        provenance = json.loads(
            provenance_path.read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        return False, "benchmark_provenance_missing"
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False, "benchmark_provenance_invalid"

    receipt_path = _idea_evidence_path(
        Path(idea_dir), str(contract["receipt"]))
    if receipt_path is None:
        return False, "benchmark_receipt_symlink_forbidden"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False, "benchmark_receipt_missing"
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False, "benchmark_receipt_invalid"
    if not isinstance(provenance, dict) or not isinstance(receipt, dict):
        return False, "benchmark_receipt_invalid"

    expected = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": contract["benchmark_id"],
        "benchmark_revision": str(contract["revision"]).lower(),
        "benchmark_view": contract["view"],
        "evaluator_sha256": str(contract["evaluator_sha256"]).lower(),
        "dataset_manifest_sha256": str(
            contract["dataset_manifest_sha256"]).lower(),
        "scorer_sha256": str(contract["scorer_sha256"]).lower(),
        "evidence_scope": contract["evidence_scope"],
        "selection_mode": contract["selection_mode"],
        "prior_exposures": int(contract["prior_exposures"]),
        "max_evaluations": int(contract["max_evaluations"]),
    }
    for key, expected_value in expected.items():
        if provenance.get(key) != expected_value:
            return False, f"benchmark_provenance_{key}_mismatch"
        if receipt.get(key) != expected_value:
            return False, f"benchmark_receipt_{key}_mismatch"
    if (not isinstance(provenance.get("evaluation_nonce"), str)
            or receipt.get("evaluation_nonce") != provenance["evaluation_nonce"]):
        return False, "benchmark_receipt_nonce_mismatch"
    exposure_ordinal = provenance.get("exposure_ordinal")
    if (isinstance(exposure_ordinal, bool)
            or not isinstance(exposure_ordinal, int) or exposure_ordinal < 1
            or receipt.get("exposure_ordinal") != exposure_ordinal):
        return False, "benchmark_receipt_exposure_ordinal_mismatch"
    record_sha256 = provenance.get("exposure_record_sha256")
    if (not isinstance(record_sha256, str)
            or _SHA256_RE.fullmatch(record_sha256) is None
            or receipt.get("exposure_record_sha256") != record_sha256):
        return False, "benchmark_receipt_exposure_record_mismatch"
    try:
        ledger_records = _read_exposure_records(Path(idea_dir).parent, cfg)
        _validated_matching_exposures(ledger_records, contract)
        _audit_exposure_provenance_links(
            Path(idea_dir).parent, ledger_records, contract,
        )
    except BenchmarkContractError as exc:
        return False, str(exc)
    ledger_match = [
        record for record in _matching_exposures(ledger_records, contract)
        if record.get("record_sha256") == record_sha256
    ]
    if len(ledger_match) != 1:
        return False, "benchmark_exposure_record_missing_or_duplicated"
    exposure_record = ledger_match[0]
    canonical_record = dict(exposure_record)
    canonical_record.pop("record_sha256", None)
    actual_record_sha256 = hashlib.sha256(json.dumps(
        canonical_record, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    if (actual_record_sha256 != record_sha256
            or exposure_record.get("exposure_ordinal") != exposure_ordinal
            or exposure_record.get("idea_id") != Path(idea_dir).name
            or exposure_record.get("evaluation_nonce_sha256") != hashlib.sha256(
                provenance["evaluation_nonce"].encode("utf-8")).hexdigest()):
        return False, "benchmark_exposure_record_integrity_mismatch"

    if receipt.get("model_form") != _MODEL_FORM:
        return False, "benchmark_receipt_model_form_mismatch"
    if receipt.get("component_model_count") != 1:
        return False, "benchmark_receipt_component_count_mismatch"
    if receipt.get("inference_passes_per_sample") != 1:
        return False, "benchmark_receipt_inference_pass_count_mismatch"
    if receipt.get("dataset_specific_routing") is not False:
        return False, "benchmark_receipt_dataset_routing_not_disabled"
    model_digest = receipt.get("model_artifact_sha256")
    if (not isinstance(model_digest, str)
            or _SHA256_RE.fullmatch(model_digest.lower()) is None):
        return False, "benchmark_receipt_model_artifact_sha256_invalid"
    if contract.get("managed_model_lineage") is True:
        try:
            from orze.core.model_lineage import (
                validate_model_lineage_for_evaluation,
            )
            lineage, lineage_sha256 = validate_model_lineage_for_evaluation(
                Path(idea_dir), cfg)
        except Exception:
            return False, "benchmark_managed_model_lineage_invalid"
        if provenance.get(
                "managed_model_lineage_sha256") != lineage_sha256:
            return False, "benchmark_provenance_model_lineage_mismatch"
        if provenance.get(
                "model_artifact_sha256") != lineage["artifact_sha256"]:
            return False, "benchmark_provenance_model_artifact_mismatch"
        if receipt.get(
                "managed_model_lineage_sha256") != lineage_sha256:
            return False, "benchmark_receipt_model_lineage_mismatch"
        if model_digest.lower() != lineage["artifact_sha256"]:
            return False, "benchmark_receipt_model_artifact_mismatch"
    decoding_digest = receipt.get("decoding_config_sha256")
    if (not isinstance(decoding_digest, str)
            or _SHA256_RE.fullmatch(decoding_digest.lower()) is None):
        return False, "benchmark_receipt_decoding_config_sha256_invalid"

    expected_metrics = set(contract["required_metrics"])
    receipt_metrics = receipt.get("metric_keys")
    if (not isinstance(receipt_metrics, list)
            or not all(isinstance(key, str) and key for key in receipt_metrics)
            or len(receipt_metrics) != len(set(receipt_metrics))
            or set(receipt_metrics) != expected_metrics):
        return False, "benchmark_receipt_metric_coverage_mismatch"

    if values is not None:
        numbers = []
        for key in contract["required_metrics"]:
            value = values.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))):
                return False, f"benchmark_metric_missing_or_nonfinite:{key}"
            numbers.append(float(value))
        primary = (cfg.get("report") or {}).get("primary_metric")
        primary_value = values.get(primary)
        if (isinstance(primary_value, bool)
                or not isinstance(primary_value, (int, float))
                or not math.isfinite(float(primary_value))):
            return False, "benchmark_primary_metric_missing_or_nonfinite"
        macro = sum(numbers) / len(numbers)
        try:
            tolerance = float(contract.get("aggregate_tolerance", 1e-6))
        except (TypeError, ValueError):
            return False, "benchmark_aggregate_tolerance_invalid"
        if not math.isfinite(tolerance) or tolerance < 0:
            return False, "benchmark_aggregate_tolerance_invalid"
        if abs(float(primary_value) - macro) > tolerance:
            return False, "benchmark_primary_metric_not_macro_mean"

    return True, "benchmark_contract_verified"
