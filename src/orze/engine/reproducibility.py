"""Fail-closed preregistered reproduction-question verification."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Mapping

import yaml

from orze.core.ideas import IDEA_ID_PATTERN
from orze.core.integrity import canonical_config_for_execution, hash_config
from orze.core.sqlite_policy import (
    SQLitePolicyError,
    inspect_shared_database_policy,
)
from orze.reporting.evidence import (
    authoritative_completed_idea_ids,
    authoritative_idea_lifecycle,
    qualify_authoritative_report_evidence,
)


_IDEA_RE = re.compile(IDEA_ID_PATTERN)
_PATH_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,63}")
_MAX_GROUPS = 64
_MAX_GROUP_SIZE = 16
_MAX_CONFIG_BYTES = 4 * 1024 * 1024
_MAX_TOTAL_CONFIG_BYTES = 64 * 1024 * 1024
_QUALIFIED_REASONS = frozenset({
    "authoritative_local_evidence_verified",
    "benchmark_evidence_verified",
})
_REPLICATION_METADATA_FIELDS = frozenset({
    "_replicate_of", "replication_role", "replication_index",
})


def config_identity_sha256(config: Mapping) -> str:
    """Return the exact canonical identity used by reproduction contracts."""
    canonical = json.dumps(
        config, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _bounded_statement(value) -> bool:
    return (
        isinstance(value, str)
        and 20 <= len(value.strip()) <= 500
        and not any(ord(character) < 32 for character in value)
    )


def validate_reproducibility_contract(
    contract,
    expected_idea_ids,
) -> str | None:
    """Validate one reproduction declaration before campaign registration."""
    if (not isinstance(expected_idea_ids, list) or not expected_idea_ids
            or len(expected_idea_ids) != len(set(expected_idea_ids))):
        return "reproducibility_expected_idea_ids_invalid"
    expected = set(expected_idea_ids)
    if not isinstance(contract, dict) or contract.get("mode") not in {
            "groups", "not_applicable"}:
        return "reproducibility_contract_invalid"
    config_identities = contract.get("expected_config_identity_sha256")
    if (not isinstance(config_identities, dict)
            or set(config_identities) != expected
            or any(not isinstance(value, str) or len(value) != 64
                   or any(character not in "0123456789abcdef"
                          for character in value)
                   for value in config_identities.values())):
        return "reproducibility_config_identities_invalid"
    if contract["mode"] == "not_applicable":
        if (set(contract) != {
                    "mode", "rationale", "expected_config_identity_sha256"}
                or not _bounded_statement(contract.get("rationale"))):
            return "reproducibility_not_applicable_invalid"
        return None
    if set(contract) != {
            "mode", "groups", "expected_config_identity_sha256"}:
        return "reproducibility_contract_fields_invalid"
    groups = contract.get("groups")
    if not isinstance(groups, list) or not 1 <= len(groups) <= _MAX_GROUPS:
        return "reproducibility_groups_invalid"
    seen = set()
    for group in groups:
        if (not isinstance(group, dict)
                or set(group) != {
                    "question", "idea_ids", "varying_config_paths",
                    "max_absolute_metric_delta",
                }
                or not _bounded_statement(group.get("question"))):
            return "reproducibility_group_invalid"
        idea_ids = group.get("idea_ids")
        if (not isinstance(idea_ids, list)
                or not 2 <= len(idea_ids) <= _MAX_GROUP_SIZE
                or len(idea_ids) != len(set(idea_ids))
                or any(not isinstance(idea_id, str)
                       or _IDEA_RE.fullmatch(idea_id) is None
                       or idea_id not in expected
                       for idea_id in idea_ids)
                or seen.intersection(idea_ids)):
            return "reproducibility_group_idea_ids_invalid"
        seen.update(idea_ids)
        paths = group.get("varying_config_paths")
        if (not isinstance(paths, list) or not 1 <= len(paths) <= 32
                or len(paths) != len(set(paths))
                or any(not isinstance(path, str)
                       or not 1 <= len(path) <= 256
                       or any(_PATH_RE.fullmatch(part) is None
                              for part in path.split("."))
                       for path in paths)):
            return "reproducibility_group_paths_invalid"
        delta = group.get("max_absolute_metric_delta")
        if (isinstance(delta, bool) or not isinstance(delta, (int, float))
                or not math.isfinite(float(delta)) or delta < 0):
            return "reproducibility_group_tolerance_invalid"
    return None


def _open_read_only(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).absolute()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError("reproducibility_database_redirected")
    if not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError("reproducibility_database_unavailable")
    connection = sqlite3.connect(
        path.as_uri() + "?mode=ro", uri=True, timeout=5,
    )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        policy = inspect_shared_database_policy(connection)
        if not policy["compliant"]:
            raise ValueError("reproducibility_database_policy_invalid")
        return connection
    except (sqlite3.Error, SQLitePolicyError, ValueError):
        connection.close()
        raise


def _load_configs(db_path, expected_idea_ids) -> tuple[dict, str | None]:
    try:
        connection = _open_read_only(db_path)
    except (OSError, sqlite3.Error, SQLitePolicyError, ValueError):
        return {}, "reproducibility_database_invalid"
    rows = []
    try:
        for offset in range(0, len(expected_idea_ids), 500):
            chunk = expected_idea_ids[offset:offset + 500]
            marks = ",".join("?" for _ in chunk)
            rows.extend(connection.execute(
                "SELECT idea_id, config, config_hash, config_source_sha256 "
                f"FROM ideas WHERE idea_id IN ({marks})",
                chunk,
            ).fetchall())
    except sqlite3.Error:
        return {}, "reproducibility_database_invalid"
    finally:
        connection.close()
    if len(rows) != len(expected_idea_ids):
        return {}, "reproducibility_config_rows_missing"
    configs = {}
    total = 0
    for row in rows:
        raw = row["config"]
        if not isinstance(raw, str):
            return {}, "reproducibility_config_invalid"
        size = len(raw.encode("utf-8"))
        total += size
        if size > _MAX_CONFIG_BYTES or total > _MAX_TOTAL_CONFIG_BYTES:
            return {}, "reproducibility_config_limit_exceeded"
        if (row["config_source_sha256"]
                != hashlib.sha256(raw.encode("utf-8")).hexdigest()):
            return {}, "reproducibility_config_integrity_invalid"
        try:
            parsed = yaml.safe_load(raw)
            json.dumps(parsed, allow_nan=False)
        except (TypeError, ValueError, yaml.YAMLError):
            return {}, "reproducibility_config_invalid"
        if not isinstance(parsed, dict):
            return {}, "reproducibility_config_invalid"
        if row["config_hash"] != hash_config(parsed):
            return {}, "reproducibility_config_integrity_invalid"
        configs[str(row["idea_id"])] = parsed
    if set(configs) != set(expected_idea_ids):
        return {}, "reproducibility_config_rows_invalid"
    return configs, None


def _canonical(value) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


def _pop_path(config: dict, dotted: str):
    parts = dotted.split(".")
    current = config
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted)
        current = current[part]
    if not isinstance(current, dict) or parts[-1] not in current:
        raise KeyError(dotted)
    value = current.pop(parts[-1])
    if isinstance(value, (dict, list)):
        raise ValueError(dotted)
    return value


def audit_campaign_reproducibility(
    db_path: str | Path,
    results_dir: str | Path,
    cfg: Mapping,
    *,
    expected_idea_ids: list[str],
    contract: dict,
) -> dict:
    """Verify declared replica isolation and qualified metric tolerance."""
    error = validate_reproducibility_contract(contract, expected_idea_ids)
    receipt = {
        "schema_version": 1,
        "status": "UNVERIFIED",
        "reason": "reproducibility_evidence_incomplete",
        "mode": contract.get("mode") if isinstance(contract, dict) else None,
        "checks": {},
        "groups": [],
        "rank_claim_proven": False,
    }
    if error:
        receipt["reason"] = error
        return receipt
    configs, config_error = _load_configs(db_path, expected_idea_ids)
    if config_error:
        receipt["reason"] = config_error
        return receipt
    execution_identities = {
        idea_id: _canonical(canonical_config_for_execution(config))
        for idea_id, config in configs.items()
    }
    current_identity_sha256 = {
        idea_id: config_identity_sha256(config)
        for idea_id, config in configs.items()
    }
    identity_matches = {
        idea_id: (
            current_identity_sha256[idea_id]
            == contract["expected_config_identity_sha256"][idea_id]
        )
        for idea_id in expected_idea_ids
    }
    receipt["checks"]["preregistered_config_identities_match"] = {
        "passed": all(identity_matches.values()),
        "idea_ids": sorted(
            idea_id for idea_id, passed in identity_matches.items()
            if not passed
        ),
    }
    if not all(identity_matches.values()):
        receipt["reason"] = "reproducibility_config_identity_mismatch"
        return receipt
    declared_ids = {
        idea_id
        for group in contract.get("groups", [])
        for idea_id in group["idea_ids"]
    }
    declared_replica_ids = {
        idea_id for idea_id, config in configs.items()
        if _REPLICATION_METADATA_FIELDS.intersection(config)
    }
    unpreregistered_replica_ids = sorted(
        declared_replica_ids - declared_ids
    )
    receipt["checks"]["declared_replicas_preregistered"] = {
        "passed": not unpreregistered_replica_ids,
        "declared_idea_ids": sorted(declared_replica_ids),
        "unpreregistered_idea_ids": unpreregistered_replica_ids,
    }
    exact_duplicates = []
    by_identity = {}
    for idea_id, identity in execution_identities.items():
        by_identity.setdefault(identity, []).append(idea_id)
    for idea_ids in by_identity.values():
        if len(idea_ids) > 1:
            exact_duplicates.append(sorted(idea_ids))
    receipt["checks"]["no_exact_duplicate_configs"] = {
        "passed": not exact_duplicates,
        "idea_id_groups": exact_duplicates,
    }
    if contract["mode"] == "not_applicable":
        if unpreregistered_replica_ids:
            receipt["status"] = "FAILED"
            receipt["reason"] = "declared_replicas_without_question"
        elif exact_duplicates:
            receipt["status"] = "FAILED"
            receipt["reason"] = "exact_duplicate_configs_detected"
        else:
            receipt["status"] = "VERIFIED"
            receipt["reason"] = "no_replicates_declared_or_detected"
        return receipt

    lifecycle = {}
    ordered_declared_ids = sorted(declared_ids)
    for offset in range(0, len(ordered_declared_ids), 64):
        lifecycle_chunk, lifecycle_reason = authoritative_idea_lifecycle(
            Path(db_path), ordered_declared_ids[offset:offset + 64]
        )
        if lifecycle_reason != "authoritative_lifecycle_loaded":
            receipt["reason"] = lifecycle_reason
            return receipt
        lifecycle.update(lifecycle_chunk)
    completed_ids, completed_reason = authoritative_completed_idea_ids(
        Path(db_path)
    )
    if completed_reason != "authoritative_lifecycle_loaded":
        receipt["reason"] = completed_reason
        return receipt

    evidence_complete = True
    targets_pass = not exact_duplicates and not unpreregistered_replica_ids
    for group in contract["groups"]:
        idea_ids = group["idea_ids"]
        normalized = []
        varying_values = {path: [] for path in group["varying_config_paths"]}
        isolation_complete = True
        try:
            for idea_id in idea_ids:
                config = copy.deepcopy(configs[idea_id])
                for path in group["varying_config_paths"]:
                    varying_values[path].append(_canonical(_pop_path(config, path)))
                normalized.append(_canonical(
                    canonical_config_for_execution(config)
                ))
        except (KeyError, TypeError, ValueError):
            isolation_complete = False
        only_declared_variables_changed = (
            isolation_complete
            and len(set(normalized)) == 1
            and all(len(set(values)) > 1 for values in varying_values.values())
            and len({execution_identities[idea_id] for idea_id in idea_ids})
            == len(idea_ids)
        )

        terminal = all(
            lifecycle[idea_id]["state"] in {
                "COMPLETE", "FAILED", "SKIPPED", "ARCHIVED",
            }
            for idea_id in idea_ids
        )
        all_complete = all(
            lifecycle[idea_id]["state"] == "COMPLETE" for idea_id in idea_ids
        )
        values = {}
        metric_evidence_complete = True
        if all_complete:
            for idea_id in idea_ids:
                try:
                    _, _, value, reason = qualify_authoritative_report_evidence(
                        idea_id, Path(results_dir), cfg, completed_ids
                    )
                except Exception:
                    reason = "reproducibility_metric_audit_failed"
                    value = None
                if (reason not in _QUALIFIED_REASONS
                        or isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))):
                    metric_evidence_complete = False
                    break
                values[idea_id] = float(value)
        metric_delta = (
            max(values.values()) - min(values.values())
            if len(values) == len(idea_ids) else None
        )
        tolerance_passed = (
            metric_delta is not None
            and metric_delta <= group["max_absolute_metric_delta"]
        )
        group_evidence_complete = (
            terminal and (metric_evidence_complete if all_complete else True)
        )
        group_passed = (
            group_evidence_complete
            and only_declared_variables_changed
            and all_complete
            and tolerance_passed
        )
        evidence_complete = evidence_complete and group_evidence_complete
        targets_pass = targets_pass and group_passed
        receipt["groups"].append({
            "question": group["question"],
            "idea_ids": idea_ids,
            "varying_config_paths": group["varying_config_paths"],
            "only_declared_variables_changed": only_declared_variables_changed,
            "all_terminal": terminal,
            "all_complete": all_complete,
            "metric_values": values,
            "metric_delta": metric_delta,
            "max_absolute_metric_delta": group[
                "max_absolute_metric_delta"
            ],
            "passed": group_passed,
        })
    receipt["checks"]["group_evidence_complete"] = {
        "passed": evidence_complete,
    }
    if evidence_complete:
        receipt["status"] = "VERIFIED" if targets_pass else "FAILED"
        receipt["reason"] = (
            "reproducibility_targets_verified"
            if targets_pass else "reproducibility_targets_failed"
        )
    return receipt
