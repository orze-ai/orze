"""Fail-closed aggregate acceptance evidence for a research harness.

This module does not turn a local readiness check into campaign or leaderboard
proof.  It binds an immutable requirement universe to exact JSON receipts and
derives each result from the receipt's own status.  Missing, redirected,
rewritten, or ambiguously scoped evidence remains ``UNVERIFIED``.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

from orze.core.fs import atomic_write


SCHEMA_VERSION = 1
_MAX_JSON_BYTES = 8 * 1024 * 1024
_MAX_EVIDENCE_PER_REQUIREMENT = 16
_MAX_ASSERTIONS_PER_EVIDENCE = 64
_SYSTEM_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_STATUSES = frozenset({"VERIFIED", "FAILED", "UNVERIFIED"})

# This is deliberately code-owned, not manifest-owned.  A project cannot make
# itself green by deleting an inconvenient campaign or public-result row.
REQUIRED_TARGETS = {
    "cpu_control_plane_efficiency": (
        "production-path 50,000-idea control benchmark meets every bound"
    ),
    "zero_compute_rejection": (
        "100% invalid or ineligible work rejected before GPU allocation"
    ),
    "eligible_queue_to_claim_latency": (
        "p95 no more than two configured poll intervals with a free slot"
    ),
    "terminal_to_next_claim_latency": (
        "p95 no more than one poll interval while eligible work remains"
    ),
    "gpu_ownership": "zero cross-scheduler GPU lease collisions",
    "gpu_scope": "100% launches use only the declared physical GPU allowlist",
    "gpu_duty_cycle": (
        "at least 90% allocation duty while eligible work exists"
    ),
    "evaluation_code_identity": "zero mixed-version evaluation executions",
    "data_separation": "zero forbidden train/evaluation sample identities",
    "result_honesty": (
        "100% decisions and reported rows use contract-qualified evidence"
    ),
    "model_eligibility": "one model, one inference pass, no routing",
    "reproduction_efficiency": (
        "no redundant replicas without a preregistered reproduction question"
    ),
    "recovery_correctness": (
        "zero lifecycle/transition/process identity contradictions"
    ),
    "environment_identity": "one exact resolved dependency graph per run",
    "operator_visibility": (
        "artifact, blocker, and next deadline update gap no more than 10 minutes"
    ),
    "time_to_first_valid_decision": "no more than 4 hours",
    "time_to_all_decisions": "no more than 24 hours",
    "research_yield": "qualified success rate at least 25%",
    "gpu_hours_per_qualified_success": "no more than 8 GPU-hours",
    "duplicate_training": "zero duplicate training attempts",
    "official_leaderboard_outcome": (
        "accepted public eligible single-model result on every default track"
    ),
}


class AcceptanceManifestError(ValueError):
    """Raised when an acceptance manifest is structurally ambiguous."""


def _open_regular_single_link(path: Path) -> tuple[int, os.stat_result]:
    """Open an absolute path component-wise without following symlinks."""
    absolute = Path(path).absolute()
    parts = absolute.parts
    if not parts or not absolute.anchor:
        raise OSError("acceptance_path_not_absolute")
    flags_dir = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd = os.open(absolute.anchor, flags_dir)
    try:
        for component in parts[1:-1]:
            next_fd = os.open(
                component, flags_dir, dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
    finally:
        os.close(directory_fd)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(descriptor)
        raise OSError("acceptance_evidence_identity_invalid")
    return descriptor, info


def _read_stable_bytes(path: Path) -> tuple[bytes, str]:
    descriptor, before = _open_regular_single_link(path)
    try:
        if not 1 <= before.st_size <= _MAX_JSON_BYTES:
            raise OSError("acceptance_evidence_size_invalid")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            data = handle.read(_MAX_JSON_BYTES + 1)
            after = os.fstat(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    identity_before = (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
        before.st_ctime_ns, before.st_nlink,
    )
    identity_after = (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
        after.st_ctime_ns, after.st_nlink,
    )
    if len(data) > _MAX_JSON_BYTES or identity_before != identity_after:
        raise OSError("acceptance_evidence_changed_during_read")
    rebound_fd, rebound = _open_regular_single_link(path)
    os.close(rebound_fd)
    rebound_identity = (
        rebound.st_dev, rebound.st_ino, rebound.st_size,
        rebound.st_mtime_ns, rebound.st_ctime_ns, rebound.st_nlink,
    )
    if rebound_identity != identity_before:
        raise OSError("acceptance_evidence_namespace_changed")
    return data, hashlib.sha256(data).hexdigest()


def _read_stable_json(path: Path) -> tuple[dict, str]:
    data, digest = _read_stable_bytes(path)

    def strict_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AcceptanceManifestError(
                    f"acceptance_json_duplicate_key:{key}"
                )
            result[key] = value
        return result

    def reject_constant(value):
        raise AcceptanceManifestError(
            f"acceptance_json_nonfinite_number:{value}"
        )

    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcceptanceManifestError("acceptance_json_invalid") from exc
    if not isinstance(payload, dict):
        raise AcceptanceManifestError("acceptance_json_not_object")
    return payload, digest


def _resolve_pointer(payload, pointer: str):
    if pointer == "":
        return payload
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise AcceptanceManifestError("acceptance_json_pointer_invalid")
    current = payload
    for encoded in pointer[1:].split("/"):
        if re.search(r"~(?![01])", encoded):
            raise AcceptanceManifestError("acceptance_json_pointer_invalid")
        token = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            if token not in current:
                raise KeyError(pointer)
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                raise KeyError(pointer)
            index = int(token)
            if index >= len(current):
                raise KeyError(pointer)
            current = current[index]
        else:
            raise KeyError(pointer)
    return current


def _json_equal(left, right) -> bool:
    """Compare JSON values without Python's ``True == 1`` coercion."""
    try:
        return json.dumps(
            left, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ) == json.dumps(
            right, sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
    except (TypeError, ValueError):
        return False


def _validate_manifest(manifest: Mapping) -> None:
    if set(manifest) != {"schema_version", "system_id", "requirements"}:
        raise AcceptanceManifestError("acceptance_manifest_fields_invalid")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise AcceptanceManifestError("acceptance_manifest_schema_invalid")
    system_id = manifest.get("system_id")
    if not isinstance(system_id, str) or _SYSTEM_ID_RE.fullmatch(system_id) is None:
        raise AcceptanceManifestError("acceptance_manifest_system_id_invalid")
    requirements = manifest.get("requirements")
    if not isinstance(requirements, Mapping):
        raise AcceptanceManifestError("acceptance_manifest_requirements_invalid")
    if set(requirements) != set(REQUIRED_TARGETS):
        raise AcceptanceManifestError("acceptance_manifest_universe_invalid")
    for requirement_id, declaration in requirements.items():
        if not isinstance(declaration, Mapping) or not set(declaration) <= {
            "evidence", "note",
        } or "evidence" not in declaration:
            raise AcceptanceManifestError(
                f"acceptance_requirement_invalid:{requirement_id}"
            )
        note = declaration.get("note")
        if note is not None and (not isinstance(note, str) or len(note) > 1000):
            raise AcceptanceManifestError(
                f"acceptance_requirement_note_invalid:{requirement_id}"
            )
        evidence = declaration["evidence"]
        if (not isinstance(evidence, list)
                or len(evidence) > _MAX_EVIDENCE_PER_REQUIREMENT):
            raise AcceptanceManifestError(
                f"acceptance_requirement_evidence_invalid:{requirement_id}"
            )
        for item in evidence:
            if not isinstance(item, Mapping) or set(item) != {
                "path", "sha256", "status_pointer", "assertions",
            }:
                raise AcceptanceManifestError("acceptance_evidence_fields_invalid")
            if not isinstance(item["path"], str) or not item["path"]:
                raise AcceptanceManifestError("acceptance_evidence_path_invalid")
            if (_SHA256_RE.fullmatch(item["sha256"])
                    is None):
                raise AcceptanceManifestError("acceptance_evidence_sha256_invalid")
            if not isinstance(item["status_pointer"], str):
                raise AcceptanceManifestError("acceptance_status_pointer_invalid")
            assertions = item["assertions"]
            if (not isinstance(assertions, list)
                    or len(assertions) > _MAX_ASSERTIONS_PER_EVIDENCE):
                raise AcceptanceManifestError("acceptance_assertions_invalid")
            for assertion in assertions:
                if not isinstance(assertion, Mapping) or set(assertion) != {
                    "pointer", "equals",
                } or not isinstance(assertion["pointer"], str):
                    raise AcceptanceManifestError("acceptance_assertion_invalid")


def _requirement_status(evidence_statuses: list[str]) -> str:
    if "FAILED" in evidence_statuses:
        return "FAILED"
    if not evidence_statuses or "UNVERIFIED" in evidence_statuses:
        return "UNVERIFIED"
    return "VERIFIED"


def _validate_official_rank_evidence(payload: Mapping) -> None:
    """Require an explicit public single-model proof for a green rank row."""
    from orze.engine import public_rank

    if payload.get("rank_claim_proven") is not True:
        raise AcceptanceManifestError("official_rank_proof_missing")
    if (payload.get("schema_version") != public_rank.SCHEMA_VERSION
            or payload.get("verification_method")
            != public_rank.VERIFICATION_METHOD):
        raise AcceptanceManifestError("official_rank_verification_method_invalid")
    verifier_path = Path(public_rank.__file__).resolve(strict=True)
    if payload.get("verifier_source_sha256") != hashlib.sha256(
            verifier_path.read_bytes()).hexdigest():
        raise AcceptanceManifestError("official_rank_verifier_identity_invalid")
    verified_at = payload.get("verified_at")
    try:
        observed_at = datetime.datetime.fromisoformat(verified_at)
    except (TypeError, ValueError) as exc:
        raise AcceptanceManifestError("official_rank_time_invalid") from exc
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise AcceptanceManifestError("official_rank_time_invalid")
    age = datetime.datetime.now(datetime.timezone.utc) - observed_at.astimezone(
        datetime.timezone.utc)
    if age < datetime.timedelta(minutes=-5) or age > datetime.timedelta(hours=24):
        raise AcceptanceManifestError("official_rank_evidence_stale")
    for key in ("public_submission_url", "public_leaderboard_url"):
        value = payload.get(key)
        parsed = urlsplit(value) if isinstance(value, str) else None
        if (parsed is None or parsed.scheme != "https" or not parsed.hostname
                or parsed.username is not None or parsed.password is not None):
            raise AcceptanceManifestError("official_rank_public_url_invalid")
    if payload.get("public_leaderboard_url") != public_rank.LEADERBOARD_PUBLIC_URL:
        raise AcceptanceManifestError("official_rank_leaderboard_url_invalid")
    if (payload.get("model_form") != "single_model_single_pass"
            or payload.get("ensemble") is not False
            or payload.get("routing") is not False):
        raise AcceptanceManifestError("official_rank_model_eligibility_invalid")
    eligibility = payload.get("eligibility_evidence")
    eligibility_fields = {
        "verification_method", "verifier_source_sha256", "model_id",
        "idea_id", "attempt_id", "execution_identity_sha256",
        "model_artifact_sha256", "model_lineage_sha256",
        "benchmark_receipt_sha256", "evaluation_bundle_sha256",
    }
    if (not isinstance(eligibility, Mapping)
            or set(eligibility) != eligibility_fields
            or eligibility.get("verification_method")
            != public_rank.ELIGIBILITY_METHOD
            or eligibility.get("verifier_source_sha256")
            != payload.get("verifier_source_sha256")
            or eligibility.get("model_id") != payload.get("model_id")
            or not isinstance(eligibility.get("idea_id"), str)
            or _SYSTEM_ID_RE.fullmatch(eligibility["idea_id"]) is None
            or not isinstance(eligibility.get("attempt_id"), str)
            or _SYSTEM_ID_RE.fullmatch(eligibility["attempt_id"]) is None
            or any(_SHA256_RE.fullmatch(eligibility.get(key, "")) is None
                   for key in {
                       "verifier_source_sha256", "execution_identity_sha256",
                       "model_artifact_sha256", "model_lineage_sha256",
                       "benchmark_receipt_sha256", "evaluation_bundle_sha256",
                   })):
        raise AcceptanceManifestError("official_rank_model_eligibility_invalid")
    model_id = payload.get("model_id")
    if (not isinstance(model_id, str)
            or public_rank._MODEL_ID_RE.fullmatch(model_id) is None):
        raise AcceptanceManifestError("official_rank_model_id_invalid")
    default_columns = payload.get("default_dataset_columns")
    if (not isinstance(default_columns, list) or not default_columns
            or any(not isinstance(item, str) for item in default_columns)
            or len(set(default_columns)) != len(default_columns)
            or not set(public_rank.DEFAULT_TRACK_COLUMNS.values()).issubset(
                default_columns)):
        raise AcceptanceManifestError("official_rank_default_columns_invalid")
    for key in ("landing_rank", "ranked_models"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise AcceptanceManifestError("official_rank_landing_invalid")
    average = payload.get("landing_average_wer")
    if (isinstance(average, bool) or not isinstance(average, (int, float))
            or not math.isfinite(float(average)) or float(average) < 0
            or payload["landing_rank"] > payload["ranked_models"]):
        raise AcceptanceManifestError("official_rank_landing_invalid")

    endpoint_evidence = payload.get("public_endpoint_evidence")
    expected_endpoints = {
        "submission": ("GET", payload["public_submission_url"]),
        "leaderboard_info": ("GET", public_rank.LEADERBOARD_INFO_URL),
        "leaderboard_call": ("POST", public_rank.LEADERBOARD_CALL_URL),
    }
    if (not isinstance(endpoint_evidence, Mapping)
            or set(endpoint_evidence) != {
                *expected_endpoints, "leaderboard_result",
            }):
        raise AcceptanceManifestError("official_rank_endpoint_evidence_invalid")
    for name, (method, request_url) in expected_endpoints.items():
        evidence = endpoint_evidence[name]
        required = {
            "method", "request_url", "final_url", "http_status",
            "content_type", "response_bytes", "response_sha256",
        }
        if method == "POST":
            required.add("request_sha256")
        if (not isinstance(evidence, Mapping) or set(evidence) != required
                or evidence.get("method") != method
                or evidence.get("request_url") != request_url
                or evidence.get("http_status") != 200
                or not isinstance(evidence.get("final_url"), str)
                or evidence.get("final_url") != request_url
                or not isinstance(evidence.get("content_type"), str)
                or (name != "submission" and not evidence[
                    "content_type"].startswith("application/json"))
                or isinstance(evidence.get("response_bytes"), bool)
                or not isinstance(evidence.get("response_bytes"), int)
                or not 1 <= evidence["response_bytes"] <= 8 * 1024 * 1024
                or _SHA256_RE.fullmatch(evidence.get("response_sha256", ""))
                is None
                or (method == "POST" and _SHA256_RE.fullmatch(
                    evidence.get("request_sha256", "")) is None)):
            raise AcceptanceManifestError(
                "official_rank_endpoint_evidence_invalid")
    result = endpoint_evidence["leaderboard_result"]
    required_result = {
        "method", "request_url", "final_url", "http_status", "content_type",
        "response_bytes", "response_sha256",
    }
    if (not isinstance(result, Mapping) or set(result) != required_result
            or result.get("method") != "GET"
            or not isinstance(result.get("request_url"), str)
            or not result["request_url"].startswith(
                public_rank.LEADERBOARD_CALL_URL + "/")
            or result.get("final_url") != result.get("request_url")
            or result.get("http_status") != 200
            or not isinstance(result.get("final_url"), str)
            or not isinstance(result.get("content_type"), str)
            or not result["content_type"].startswith("text/event-stream")
            or isinstance(result.get("response_bytes"), bool)
            or not isinstance(result.get("response_bytes"), int)
            or not 1 <= result["response_bytes"] <= 8 * 1024 * 1024
            or _SHA256_RE.fullmatch(result.get("response_sha256", "")) is None):
        raise AcceptanceManifestError("official_rank_endpoint_evidence_invalid")
    tracks = payload.get("default_tracks")
    if (not isinstance(tracks, Mapping)
            or set(tracks) != set(public_rank.DEFAULT_TRACK_COLUMNS)):
        raise AcceptanceManifestError("official_rank_default_tracks_missing")
    for track_id, column in public_rank.DEFAULT_TRACK_COLUMNS.items():
        track = tracks.get(track_id)
        if (not isinstance(track, Mapping)
                or set(track) != {
                    "accepted", "column", "rank", "score", "ranked_models",
                }
                or track.get("accepted") is not True
                or track.get("column") != column
                or isinstance(track.get("rank"), bool)
                or not isinstance(track.get("rank"), int)
                or track["rank"] <= 0
                or isinstance(track.get("ranked_models"), bool)
                or not isinstance(track.get("ranked_models"), int)
                or track["ranked_models"] <= 0
                or track["rank"] > track["ranked_models"]
                or isinstance(track.get("score"), bool)
                or not isinstance(track.get("score"), (int, float))
                or not math.isfinite(float(track["score"]))
                or float(track["score"]) < 0):
            raise AcceptanceManifestError("official_rank_default_track_invalid")


def audit_acceptance_matrix(manifest_path: str | Path) -> dict:
    """Derive the complete harness status from exact pinned evidence."""
    generated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    source_path = Path(__file__).absolute()
    path = Path(manifest_path).absolute()
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "UNVERIFIED",
        "reason": "acceptance_evidence_incomplete",
        "system_id": None,
        "requirements": {},
        "counts": {
            "required": len(REQUIRED_TARGETS),
            "verified": 0,
            "failed": 0,
            "unverified": len(REQUIRED_TARGETS),
        },
        "all_required_targets_verified": False,
        "rank_claim_proven": False,
    }
    try:
        source_data, source_digest = _read_stable_bytes(source_path)
        manifest, manifest_digest = _read_stable_json(path)
        _validate_manifest(manifest)
    except (OSError, AcceptanceManifestError) as exc:
        receipt["reason"] = str(exc)
        return receipt
    receipt["source_sha256"] = source_digest
    receipt["manifest_sha256"] = manifest_digest
    receipt["system_id"] = manifest["system_id"]
    manifest_dir = path.parent
    observed_files: dict[Path, str] = {}

    for requirement_id, target in REQUIRED_TARGETS.items():
        declaration = manifest["requirements"][requirement_id]
        evidence_results = []
        statuses = []
        for evidence in declaration["evidence"]:
            evidence_path = Path(evidence["path"])
            if not evidence_path.is_absolute():
                evidence_path = (manifest_dir / evidence_path).absolute()
            item_result = {
                "path": str(evidence_path),
                "expected_sha256": evidence["sha256"],
                "status_pointer": evidence["status_pointer"],
                "status": "UNVERIFIED",
                "reason": "evidence_unavailable",
            }
            try:
                payload, observed_digest = _read_stable_json(evidence_path)
                item_result["observed_sha256"] = observed_digest
                if observed_digest != evidence["sha256"]:
                    raise AcceptanceManifestError("evidence_sha256_mismatch")
                for assertion in evidence["assertions"]:
                    if not _json_equal(
                        _resolve_pointer(payload, assertion["pointer"]),
                        assertion["equals"],
                    ):
                        raise AcceptanceManifestError(
                            "evidence_assertion_mismatch:"
                            + assertion["pointer"]
                        )
                evidence_status = _resolve_pointer(
                    payload, evidence["status_pointer"]
                )
                if evidence_status not in _STATUSES:
                    raise AcceptanceManifestError("evidence_status_invalid")
                if (requirement_id == "official_leaderboard_outcome"
                        and evidence_status == "VERIFIED"):
                    _validate_official_rank_evidence(payload)
                item_result["status"] = evidence_status
                item_result["reason"] = "evidence_status_derived"
                statuses.append(evidence_status)
                observed_files[evidence_path] = observed_digest
            except (OSError, AcceptanceManifestError, KeyError) as exc:
                item_result["reason"] = str(exc)
                statuses.append("UNVERIFIED")
            evidence_results.append(item_result)
        requirement_status = _requirement_status(statuses)
        receipt["requirements"][requirement_id] = {
            "target": target,
            "status": requirement_status,
            "note": declaration.get("note"),
            "evidence": evidence_results,
        }

    # Rebind every input after the full audit, not just during its own read.
    try:
        manifest_after, manifest_after_digest = _read_stable_json(path)
        source_after, source_after_digest = _read_stable_bytes(source_path)
        if (manifest_after_digest != manifest_digest
                or manifest_after != manifest
                or source_after_digest != source_digest
                or source_after != source_data):
            raise OSError("acceptance_inputs_changed_during_audit")
        for evidence_path, expected_digest in observed_files.items():
            _, observed_digest = _read_stable_bytes(evidence_path)
            if observed_digest != expected_digest:
                raise OSError("acceptance_inputs_changed_during_audit")
    except (OSError, AcceptanceManifestError):
        receipt["reason"] = "acceptance_inputs_changed_during_audit"
        return receipt

    statuses = [
        item["status"] for item in receipt["requirements"].values()
    ]
    receipt["counts"] = {
        "required": len(statuses),
        "verified": statuses.count("VERIFIED"),
        "failed": statuses.count("FAILED"),
        "unverified": statuses.count("UNVERIFIED"),
    }
    if "FAILED" in statuses:
        receipt["status"] = "FAILED"
        receipt["reason"] = "one_or_more_targets_failed"
    elif statuses and all(status == "VERIFIED" for status in statuses):
        receipt["status"] = "VERIFIED"
        receipt["reason"] = "all_required_targets_verified"
        receipt["all_required_targets_verified"] = True
    receipt["rank_claim_proven"] = (
        receipt["requirements"]["official_leaderboard_outcome"]["status"]
        == "VERIFIED"
    )
    return receipt


def write_acceptance_matrix(
    manifest_path: str | Path,
    output_path: str | Path,
) -> dict:
    receipt = audit_acceptance_matrix(manifest_path)
    atomic_write(
        Path(output_path),
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate exact research-harness acceptance evidence"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt = write_acceptance_matrix(args.manifest, args.output)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
