import datetime
import hashlib
import json
import os
from pathlib import Path

import orze.engine.acceptance_matrix as acceptance_module
import orze.engine.public_rank as public_rank
from orze.engine.acceptance_matrix import (
    REQUIRED_TARGETS,
    audit_acceptance_matrix,
)


def _write_json(path, payload):
    data = (json.dumps(payload, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _endpoint(method, url, *, post=False, content_type="application/json"):
    result = {
        "method": method,
        "request_url": url,
        "final_url": url,
        "http_status": 200,
        "content_type": content_type,
        "response_bytes": 100,
        "response_sha256": "a" * 64,
    }
    if post:
        result["request_sha256"] = "b" * 64
    return result


def _official_receipt(receipt_status):
    submission_url = "https://huggingface.co/org/standalone-asr"
    result_url = public_rank.LEADERBOARD_CALL_URL + "/0123456789abcdef"
    artifact_manifest_sha256 = "e" * 64
    declaration = {
        "schema_version": 1,
        "model_id": "org/standalone-asr",
        "model_form": "single_model_single_pass",
        "component_model_count": 1,
        "inference_passes_per_sample": 1,
        "dataset_specific_routing": False,
        "artifact_manifest_sha256": artifact_manifest_sha256,
    }
    return {
        "schema_version": 1,
        "status": receipt_status,
        "rank_claim_proven": True,
        "verification_method": public_rank.VERIFICATION_METHOD,
        "verifier_source_sha256": hashlib.sha256(
            Path(public_rank.__file__).read_bytes()).hexdigest(),
        "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "public_submission_url": submission_url,
        "public_leaderboard_url": public_rank.LEADERBOARD_PUBLIC_URL,
        "model_id": "org/standalone-asr",
        "model_form": "single_model_single_pass",
        "ensemble": False,
        "routing": False,
        "eligibility_evidence": {
            "verification_method": (
                "managed_model_lineage_and_single_pass_preflight_v1"),
            "verifier_source_sha256": hashlib.sha256(
                Path(public_rank.__file__).read_bytes()).hexdigest(),
            "model_id": "org/standalone-asr",
            "idea_id": "managed-idea",
            "attempt_id": "attempt-1",
            "execution_identity_sha256": "f" * 64,
            "model_artifact_sha256": "a" * 64,
            "model_lineage_sha256": "b" * 64,
            "benchmark_receipt_sha256": "c" * 64,
            "evaluation_bundle_sha256": "d" * 64,
            "artifact_manifest_sha256": artifact_manifest_sha256,
            "artifact_files": 1,
            "artifact_bytes": 5,
        },
        "publication_identity_evidence": {
            "verification_method": (
                public_rank.PUBLICATION_IDENTITY_METHOD),
            "model_id": "org/standalone-asr",
            "public_submission_url": submission_url,
            "hub_commit_sha": "1" * 40,
            "artifact_manifest_sha256": artifact_manifest_sha256,
            "artifact_files": 1,
            "artifact_bytes": 5,
            "hub_repository_file_count": 1,
            "matched_payload_file_count": 1,
            "lfs_payload_file_count": 1,
            "regular_payload_file_count": 0,
            "ignored_metadata_files": [],
            "hub_repository_identity_sha256": "f" * 64,
            "hub_api_evidence": _endpoint(
                "GET",
                "https://huggingface.co/api/models/"
                "org/standalone-asr?blobs=true"),
            "model_card_evidence": {
                **_endpoint(
                    "GET",
                    "https://huggingface.co/org/standalone-asr/resolve/"
                    + "1" * 40 + "/README.md",
                    content_type="text/markdown"),
                "declaration_sha256": hashlib.sha256(json.dumps(
                    declaration, sort_keys=True, separators=(",", ":"),
                    allow_nan=False,
                ).encode()).hexdigest(),
            },
            "regular_file_evidence": [],
        },
        "landing_rank": 2,
        "landing_average_wer": 5.0,
        "ranked_models": 3,
        "default_dataset_columns": [
            "AMI-Cleaned", "Private (scripted)",
            "Private (conversational)",
        ],
        "default_tracks": {
            "private_scripted": {
                "accepted": True,
                "column": "Private (scripted)",
                "rank": 1,
                "score": 2.0,
                "ranked_models": 3,
            },
            "private_conversational": {
                "accepted": True,
                "column": "Private (conversational)",
                "rank": 3,
                "score": 10.0,
                "ranked_models": 3,
            },
        },
        "public_endpoint_evidence": {
            "submission": _endpoint("GET", submission_url),
            "leaderboard_info": _endpoint(
                "GET", public_rank.LEADERBOARD_INFO_URL),
            "leaderboard_call": _endpoint(
                "POST", public_rank.LEADERBOARD_CALL_URL, post=True),
            "leaderboard_result": _endpoint(
                "GET", result_url, content_type="text/event-stream"),
        },
    }


def _manifest(tmp_path, *, receipt_status="VERIFIED"):
    evidence_path = tmp_path / "evidence.json"
    digest = _write_json(evidence_path, {
        "status": receipt_status,
        "scope": {"single_model": True},
    })
    requirements = {}
    for requirement_id in REQUIRED_TARGETS:
        requirements[requirement_id] = {
            "evidence": [{
                "path": str(evidence_path),
                "sha256": digest,
                "status_pointer": "/status",
                "assertions": [{
                    "pointer": "/scope/single_model",
                    "equals": True,
                }],
            }],
        }
    official_path = tmp_path / "official.json"
    official_digest = _write_json(
        official_path, _official_receipt(receipt_status))
    requirements["official_leaderboard_outcome"] = {
        "evidence": [{
            "path": str(official_path),
            "sha256": official_digest,
            "status_pointer": "/status",
            "assertions": [{
                "pointer": "/rank_claim_proven",
                "equals": True,
            }],
        }],
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, {
        "schema_version": 1,
        "system_id": "test-harness",
        "requirements": requirements,
    })
    return manifest_path, evidence_path


def test_acceptance_matrix_requires_every_verified_dimension(tmp_path):
    manifest_path, _ = _manifest(tmp_path)

    receipt = audit_acceptance_matrix(manifest_path)

    assert receipt["status"] == "VERIFIED"
    assert receipt["all_required_targets_verified"] is True
    assert receipt["counts"] == {
        "required": len(REQUIRED_TARGETS),
        "verified": len(REQUIRED_TARGETS),
        "failed": 0,
        "unverified": 0,
    }
    assert receipt["rank_claim_proven"] is True


def test_acceptance_matrix_cannot_delete_an_open_requirement(tmp_path):
    manifest_path, _ = _manifest(tmp_path)
    payload = json.loads(manifest_path.read_text())
    del payload["requirements"]["official_leaderboard_outcome"]
    _write_json(manifest_path, payload)

    receipt = audit_acceptance_matrix(manifest_path)

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["reason"] == "acceptance_manifest_universe_invalid"
    assert receipt["rank_claim_proven"] is False


def test_acceptance_matrix_keeps_missing_evidence_unverified(tmp_path):
    manifest_path, _ = _manifest(tmp_path)
    payload = json.loads(manifest_path.read_text())
    payload["requirements"]["research_yield"] = {
        "evidence": [],
        "note": "campaign paused",
    }
    _write_json(manifest_path, payload)

    receipt = audit_acceptance_matrix(manifest_path)

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["requirements"]["research_yield"]["status"] == "UNVERIFIED"
    assert receipt["all_required_targets_verified"] is False


def test_acceptance_matrix_propagates_failed_target(tmp_path):
    manifest_path, _ = _manifest(tmp_path, receipt_status="FAILED")

    receipt = audit_acceptance_matrix(manifest_path)

    assert receipt["status"] == "FAILED"
    assert receipt["counts"]["failed"] == len(REQUIRED_TARGETS)
    assert receipt["rank_claim_proven"] is False


def test_acceptance_matrix_rejects_generic_green_as_rank_proof(tmp_path):
    manifest_path, evidence_path = _manifest(tmp_path)
    payload = json.loads(manifest_path.read_text())
    generic_digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    payload["requirements"]["official_leaderboard_outcome"] = {
        "evidence": [{
            "path": str(evidence_path),
            "sha256": generic_digest,
            "status_pointer": "/status",
            "assertions": [],
        }],
    }
    _write_json(manifest_path, payload)

    receipt = audit_acceptance_matrix(manifest_path)

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["rank_claim_proven"] is False
    evidence = receipt["requirements"]["official_leaderboard_outcome"][
        "evidence"
    ][0]
    assert evidence["reason"] == "official_rank_proof_missing"


def test_acceptance_matrix_rejects_hand_authored_public_endpoint_rank(tmp_path):
    manifest_path, _ = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    official_path = tmp_path / "official.json"
    fabricated = {
        "status": "VERIFIED",
        "rank_claim_proven": True,
        "verification_method": "public_endpoint",
        "public_submission_url": "https://example.test/submission",
        "public_leaderboard_url": "https://example.test/leaderboard",
        "model_form": "single_model_single_pass",
        "ensemble": False,
        "routing": False,
        "default_tracks": {
            "private_scripted": {"accepted": True, "rank": 1},
            "private_conversational": {"accepted": True, "rank": 1},
        },
    }
    digest = _write_json(official_path, fabricated)
    manifest["requirements"]["official_leaderboard_outcome"]["evidence"][0][
        "sha256"
    ] = digest
    _write_json(manifest_path, manifest)

    receipt = audit_acceptance_matrix(manifest_path)

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["rank_claim_proven"] is False
    evidence = receipt["requirements"]["official_leaderboard_outcome"][
        "evidence"
    ][0]
    assert evidence["reason"] == "official_rank_verification_method_invalid"


def test_acceptance_matrix_rejects_stale_public_rank_receipt(tmp_path):
    manifest_path, _ = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    official_path = tmp_path / "official.json"
    payload = json.loads(official_path.read_text())
    payload["verified_at"] = (
        datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(hours=25)
    ).isoformat()
    digest = _write_json(official_path, payload)
    manifest["requirements"]["official_leaderboard_outcome"]["evidence"][0][
        "sha256"
    ] = digest
    _write_json(manifest_path, manifest)

    receipt = audit_acceptance_matrix(manifest_path)

    evidence = receipt["requirements"]["official_leaderboard_outcome"][
        "evidence"
    ][0]
    assert evidence["status"] == "UNVERIFIED"
    assert evidence["reason"] == "official_rank_evidence_stale"


def test_acceptance_matrix_rejects_public_rank_verifier_drift(tmp_path):
    manifest_path, _ = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    official_path = tmp_path / "official.json"
    payload = json.loads(official_path.read_text())
    payload["verifier_source_sha256"] = "0" * 64
    digest = _write_json(official_path, payload)
    manifest["requirements"]["official_leaderboard_outcome"]["evidence"][0][
        "sha256"
    ] = digest
    _write_json(manifest_path, manifest)

    receipt = audit_acceptance_matrix(manifest_path)

    evidence = receipt["requirements"]["official_leaderboard_outcome"][
        "evidence"
    ][0]
    assert evidence["status"] == "UNVERIFIED"
    assert evidence["reason"] == "official_rank_verifier_identity_invalid"


def test_acceptance_matrix_rejects_public_rank_eligibility_model_mismatch(
        tmp_path):
    manifest_path, _ = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    official_path = tmp_path / "official.json"
    payload = json.loads(official_path.read_text())
    payload["eligibility_evidence"]["model_id"] = "org/different-model"
    digest = _write_json(official_path, payload)
    manifest["requirements"]["official_leaderboard_outcome"]["evidence"][0][
        "sha256"
    ] = digest
    _write_json(manifest_path, manifest)

    receipt = audit_acceptance_matrix(manifest_path)

    evidence = receipt["requirements"]["official_leaderboard_outcome"][
        "evidence"
    ][0]
    assert evidence["status"] == "UNVERIFIED"
    assert evidence["reason"] == "official_rank_model_eligibility_invalid"


def test_acceptance_matrix_rejects_publication_artifact_manifest_mismatch(
        tmp_path):
    manifest_path, _ = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    official_path = tmp_path / "official.json"
    payload = json.loads(official_path.read_text())
    payload["publication_identity_evidence"][
        "artifact_manifest_sha256"] = "0" * 64
    digest = _write_json(official_path, payload)
    manifest["requirements"]["official_leaderboard_outcome"]["evidence"][0][
        "sha256"
    ] = digest
    _write_json(manifest_path, manifest)

    receipt = audit_acceptance_matrix(manifest_path)

    evidence = receipt["requirements"]["official_leaderboard_outcome"][
        "evidence"
    ][0]
    assert evidence["status"] == "UNVERIFIED"
    assert evidence["reason"] == "official_rank_publication_identity_invalid"


def test_acceptance_matrix_rejects_receipt_rewrite(tmp_path):
    manifest_path, evidence_path = _manifest(tmp_path)
    _write_json(evidence_path, {"status": "VERIFIED", "scope": {}})

    receipt = audit_acceptance_matrix(manifest_path)

    assert receipt["status"] == "UNVERIFIED"
    first = receipt["requirements"]["cpu_control_plane_efficiency"]["evidence"][0]
    assert first["reason"] == "evidence_sha256_mismatch"


def test_acceptance_matrix_rejects_duplicate_json_keys(tmp_path):
    manifest_path, evidence_path = _manifest(tmp_path)
    data = b'{"status":"FAILED","status":"VERIFIED","scope":{}}\n'
    evidence_path.write_bytes(data)
    payload = json.loads(manifest_path.read_text())
    digest = hashlib.sha256(data).hexdigest()
    for requirement_id, declaration in payload["requirements"].items():
        if requirement_id != "official_leaderboard_outcome":
            declaration["evidence"][0]["sha256"] = digest
    _write_json(manifest_path, payload)

    receipt = audit_acceptance_matrix(manifest_path)

    assert receipt["status"] == "UNVERIFIED"
    first = receipt["requirements"]["cpu_control_plane_efficiency"]["evidence"][0]
    assert first["reason"] == "acceptance_json_duplicate_key:status"


def test_acceptance_matrix_does_not_equate_boolean_and_integer(tmp_path):
    manifest_path, _ = _manifest(tmp_path)
    payload = json.loads(manifest_path.read_text())
    payload["requirements"]["cpu_control_plane_efficiency"]["evidence"][0][
        "assertions"
    ][0]["equals"] = 1
    _write_json(manifest_path, payload)

    receipt = audit_acceptance_matrix(manifest_path)

    assert receipt["status"] == "UNVERIFIED"
    first = receipt["requirements"]["cpu_control_plane_efficiency"]["evidence"][0]
    assert first["reason"].startswith("evidence_assertion_mismatch:")


def test_acceptance_matrix_rejects_parent_symlink(tmp_path):
    manifest_path, evidence_path = _manifest(tmp_path)
    real = tmp_path / "real"
    real.mkdir()
    moved = real / "evidence.json"
    evidence_path.replace(moved)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    payload = json.loads(manifest_path.read_text())
    for declaration in payload["requirements"].values():
        declaration["evidence"][0]["path"] = str(linked / "evidence.json")
    _write_json(manifest_path, payload)

    receipt = audit_acceptance_matrix(manifest_path)

    assert receipt["status"] == "UNVERIFIED"
    first = receipt["requirements"]["cpu_control_plane_efficiency"]["evidence"][0]
    assert first["reason"] != "evidence_status_derived"


def test_acceptance_matrix_rejects_hardlinked_evidence(tmp_path):
    manifest_path, evidence_path = _manifest(tmp_path)
    os.link(evidence_path, tmp_path / "alias.json")

    receipt = audit_acceptance_matrix(manifest_path)

    assert receipt["status"] == "UNVERIFIED"
    first = receipt["requirements"]["cpu_control_plane_efficiency"]["evidence"][0]
    assert first["reason"] == "acceptance_evidence_identity_invalid"


def test_acceptance_matrix_rejects_source_change(tmp_path, monkeypatch):
    manifest_path, _ = _manifest(tmp_path)
    original = acceptance_module._read_stable_bytes
    source_reads = 0

    def changed_after_start(path):
        nonlocal source_reads
        data, digest = original(path)
        if path == acceptance_module.Path(acceptance_module.__file__).absolute():
            source_reads += 1
            if source_reads > 1:
                return data + b"changed", "0" * 64
        return data, digest

    monkeypatch.setattr(
        acceptance_module, "_read_stable_bytes", changed_after_start
    )

    receipt = audit_acceptance_matrix(manifest_path)

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["reason"] == "acceptance_inputs_changed_during_audit"
