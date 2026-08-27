"""Fail-closed provenance for benchmark-comparable evaluation results.

The contract deliberately does not try to infer an external leaderboard rank.
It proves only that a local result was produced by the configured evaluator for
the exact benchmark revision/view and model form.  Reporting code may then
compute a *local* ordering among contract-compliant runs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
from pathlib import Path
from typing import Mapping, Optional

from orze.core.fs import atomic_write, deep_get


SCHEMA_VERSION = 1
PROVENANCE_FILE = "_benchmark_evaluation.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMMUTABLE_REVISION_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_MODEL_FORM = "single_model_single_pass"


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


def prepare_benchmark_evaluation(idea_dir: Path, cfg: Mapping) -> dict[str, str]:
    """Pin evaluator identity and return nonce-bearing child environment.

    A pre-existing receipt is rejected.  This prevents training or a previous
    attempt from placing a document that the evaluator never produced.
    """
    contract = get_benchmark_contract(cfg)
    if contract is None:
        return {}

    receipt_path = Path(idea_dir) / contract["receipt"]
    if receipt_path.exists():
        raise BenchmarkContractError(
            f"benchmark receipt existed before evaluation: {receipt_path}"
        )

    eval_path = _project_path(cfg, str(cfg["eval_script"]))
    actual_digest = _sha256_file(eval_path)
    expected_digest = str(contract["evaluator_sha256"]).lower()
    if actual_digest != expected_digest:
        raise BenchmarkContractError(
            "benchmark evaluator hash drift: "
            f"expected {expected_digest}, got {actual_digest}"
        )

    nonce = secrets.token_hex(32)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": contract["benchmark_id"],
        "benchmark_revision": str(contract["revision"]).lower(),
        "benchmark_view": contract["view"],
        "evaluator_sha256": actual_digest,
        "dataset_manifest_sha256": str(
            contract["dataset_manifest_sha256"]).lower(),
        "scorer_sha256": str(contract["scorer_sha256"]).lower(),
        "evaluation_nonce": nonce,
        "pid": os.getpid(),
    }
    atomic_write(
        Path(idea_dir) / PROVENANCE_FILE,
        json.dumps(provenance, sort_keys=True, indent=2) + "\n",
    )
    return {
        "ORZE_BENCHMARK_EVALUATION_NONCE": nonce,
        "ORZE_BENCHMARK_RECEIPT": str(receipt_path.resolve()),
    }


def load_benchmark_values(idea_dir: Path, cfg: Mapping) -> dict:
    """Load configured report values without fallback or inferred aliases."""
    values = {}
    metrics = {}
    metrics_path = Path(idea_dir) / "metrics.json"
    try:
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
            try:
                document = json.loads(
                    (Path(idea_dir) / filename).read_text(encoding="utf-8")
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

    try:
        provenance = json.loads(
            (Path(idea_dir) / PROVENANCE_FILE).read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        return False, "benchmark_provenance_missing"
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False, "benchmark_provenance_invalid"

    receipt_path = Path(idea_dir) / contract["receipt"]
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
    }
    for key, expected_value in expected.items():
        if provenance.get(key) != expected_value:
            return False, f"benchmark_provenance_{key}_mismatch"
        if receipt.get(key) != expected_value:
            return False, f"benchmark_receipt_{key}_mismatch"
    if (not isinstance(provenance.get("evaluation_nonce"), str)
            or receipt.get("evaluation_nonce") != provenance["evaluation_nonce"]):
        return False, "benchmark_receipt_nonce_mismatch"

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
