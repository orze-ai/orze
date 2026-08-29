import hashlib
import json
from pathlib import Path

import orze.engine.public_rank as public_rank
import pytest
from orze.engine.public_rank import (
    EndpointResponse,
    LEADERBOARD_CALL_URL,
    LEADERBOARD_INFO_URL,
    LEADERBOARD_ORIGIN,
    verify_open_asr_public_rank,
)


MODEL_ID = "org/standalone-asr"
SUBMISSION_URL = "https://huggingface.co/org/standalone-asr"
REAL_MANAGED_CONTEXT = public_rank._managed_eligibility_context
REAL_PUBLICATION_IDENTITY = public_rank._publication_identity_evidence
DEFAULTS = [
    "RTFx ⬆️️", "AMI-Cleaned", "Private (scripted)",
    "Private (conversational)",
]


def _eligibility(model_id=MODEL_ID):
    return {
        "verification_method": public_rank.ELIGIBILITY_METHOD,
        "verifier_source_sha256": hashlib.sha256(
            Path(public_rank.__file__).read_bytes()).hexdigest(),
        "model_id": model_id,
        "idea_id": "managed-idea",
        "attempt_id": "attempt-1",
        "execution_identity_sha256": "f" * 64,
        "model_artifact_sha256": "a" * 64,
        "model_lineage_sha256": "b" * 64,
        "benchmark_receipt_sha256": "c" * 64,
        "evaluation_bundle_sha256": "d" * 64,
        "artifact_manifest_sha256": "e" * 64,
        "artifact_files": 1,
        "artifact_bytes": 5,
    }


def _artifact_manifest(contents=b"model"):
    core = {
        "schema_version": 1,
        "hash_method": "sha256_bytes_v1",
        "files": [{
            "path": "model.bin",
            "size": len(contents),
            "sha256": hashlib.sha256(contents).hexdigest(),
        }],
    }
    return {
        **core,
        "manifest_sha256": public_rank._canonical_sha256(core),
    }


def _publication(model_id=MODEL_ID):
    return {
        "verification_method": public_rank.PUBLICATION_IDENTITY_METHOD,
        "model_id": model_id,
        "public_submission_url": SUBMISSION_URL,
        "hub_commit_sha": "1" * 40,
        "artifact_manifest_sha256": "e" * 64,
        "artifact_files": 1,
        "artifact_bytes": 5,
        "hub_repository_file_count": 1,
        "matched_payload_file_count": 1,
        "lfs_payload_file_count": 1,
        "regular_payload_file_count": 0,
        "ignored_metadata_files": [],
        "hub_repository_identity_sha256": "2" * 64,
        "hub_api_evidence": {},
        "model_card_evidence": {},
        "regular_file_evidence": [],
    }


@pytest.fixture(autouse=True)
def _managed_eligibility_stub(monkeypatch):
    def derive(idea_dir, cfg, model_id):
        if idea_dir is None and cfg is None:
            return None
        assert idea_dir == Path("/managed")
        assert cfg == {}
        return public_rank.ManagedEligibilityContext(
            _eligibility(model_id), _artifact_manifest())
    monkeypatch.setattr(
        public_rank, "_managed_eligibility_context", derive)
    monkeypatch.setattr(
        public_rank, "_publication_identity_evidence",
        lambda manifest, model_id, url, transport: _publication(model_id))


def _info(defaults=DEFAULTS):
    return {
        "named_endpoints": {
            "/_main_table_update": {
                "parameters": [
                    {"parameter_name": "search_query", "parameter_default": None},
                    {"parameter_name": "show_proprietary", "parameter_default": True},
                    {"parameter_name": "show_llm", "parameter_default": False},
                    {"parameter_name": "selected_columns", "parameter_default": defaults},
                ]
            }
        }
    }


def _table(rows=None):
    headers = [
        "Rank", "model", "Average WER ⬇️", "Private (scripted)",
        "Private (conversational)",
    ]
    rows = rows or [
        [1, "org/leader", 4.0, 3.0, 8.0],
        [2, (
            '<a target="_blank" href="https://huggingface.co/'
            + MODEL_ID + '">' + MODEL_ID + "</a>"
        ), 5.0, 2.0, 10.0],
        [3, "org/trailer", 6.0, 4.0, 9.0],
    ]
    payload = [{"datatype": [], "value": {"headers": headers, "data": rows}}]
    return b"event: complete\ndata: " + json.dumps(payload).encode() + b"\n\n"


class FakeTransport:
    def __init__(self, *, info=None, table=None, submission=None):
        self.info = info or _info()
        self.table = table or _table()
        self.submission = submission or ("model " + MODEL_ID).encode()
        self.calls = []

    def __call__(self, method, url, body, timeout, max_bytes):
        self.calls.append((method, url, body, timeout, max_bytes))
        if url == SUBMISSION_URL:
            return EndpointResponse(
                self.submission, url, content_type="text/html")
        if url == LEADERBOARD_INFO_URL:
            return EndpointResponse(
                json.dumps(self.info).encode(), url,
                content_type="application/json")
        if url == LEADERBOARD_CALL_URL:
            return EndpointResponse(
                b'{"event_id":"0123456789abcdef0123456789abcdef"}', url,
                content_type="application/json")
        if url == LEADERBOARD_CALL_URL + "/0123456789abcdef0123456789abcdef":
            return EndpointResponse(
                self.table, url, content_type="text/event-stream")
        raise AssertionError(url)


def test_public_rank_is_derived_from_default_public_table():
    transport = FakeTransport()

    receipt = verify_open_asr_public_rank(
        MODEL_ID, SUBMISSION_URL, idea_dir=Path("/managed"), cfg={},
        transport=transport)

    assert receipt["status"] == "VERIFIED"
    assert receipt["rank_claim_proven"] is True
    assert receipt["landing_rank"] == 2
    assert receipt["landing_average_wer"] == 5.0
    assert receipt["default_tracks"] == {
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
    }
    assert receipt["default_dataset_columns"] == DEFAULTS
    assert receipt["publication_identity_evidence"] == _publication()
    evidence = receipt["public_endpoint_evidence"]
    assert set(evidence) == {
        "submission", "leaderboard_info", "leaderboard_call",
        "leaderboard_result",
    }
    assert evidence["leaderboard_result"]["response_sha256"] == (
        hashlib.sha256(transport.table).hexdigest()
    )
    assert [call[0] for call in transport.calls] == ["GET", "GET", "POST", "GET"]


def test_public_rank_rejects_defaults_without_both_private_tracks():
    transport = FakeTransport(info=_info(["AMI-Cleaned", "Private (scripted)"]))

    receipt = verify_open_asr_public_rank(
        MODEL_ID, SUBMISSION_URL, idea_dir=Path("/managed"), cfg={},
        transport=transport)

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["rank_claim_proven"] is False
    assert receipt["reason"] == "public_rank_private_defaults_missing"


def test_public_rank_rejects_absent_model_instead_of_inventing_rank():
    transport = FakeTransport(table=_table([
        [1, "org/other", 4.0, 3.0, 8.0],
    ]))

    receipt = verify_open_asr_public_rank(
        MODEL_ID, SUBMISSION_URL, idea_dir=Path("/managed"), cfg={},
        transport=transport)

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["rank_claim_proven"] is False
    assert receipt["reason"] == "public_rank_model_absent"
    assert receipt["default_dataset_columns"] == DEFAULTS


def test_public_rank_rejects_duplicate_model_rows():
    cell = (
        '<a href="https://huggingface.co/' + MODEL_ID + '">' + MODEL_ID + "</a>"
    )
    transport = FakeTransport(table=_table([
        [1, cell, 4.0, 3.0, 8.0],
        [2, cell, 5.0, 4.0, 9.0],
    ]))

    receipt = verify_open_asr_public_rank(
        MODEL_ID, SUBMISSION_URL, idea_dir=Path("/managed"), cfg={},
        transport=transport)

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["reason"] == "public_rank_model_row_duplicated"


def test_public_rank_rejects_submission_without_model_identity():
    transport = FakeTransport(submission=b"some other public model")

    receipt = verify_open_asr_public_rank(
        MODEL_ID, SUBMISSION_URL, idea_dir=Path("/managed"), cfg={},
        transport=transport)

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["reason"] == "public_rank_submission_model_missing"


def test_public_rank_rejects_fabricated_or_nonpublic_submission_url():
    receipt = verify_open_asr_public_rank(
        MODEL_ID, "https://example.test/invented",
        idea_dir=Path("/managed"), cfg={}, transport=FakeTransport())

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["rank_claim_proven"] is False
    assert receipt["reason"] == "public_rank_host_not_allowed"


def test_public_rank_rejects_table_rank_inconsistent_with_scores():
    transport = FakeTransport(table=_table([
        [1, "org/leader", 4.0, 3.0, 8.0],
        [9, MODEL_ID, 5.0, 2.0, 10.0],
    ]))

    receipt = verify_open_asr_public_rank(
        MODEL_ID, SUBMISSION_URL, idea_dir=Path("/managed"), cfg={},
        transport=transport)

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["reason"] == "public_rank_landing_rank_mismatch"


def test_public_rank_observation_without_eligibility_cannot_green():
    receipt = verify_open_asr_public_rank(
        MODEL_ID, SUBMISSION_URL, transport=FakeTransport())

    assert receipt["landing_rank"] == 2
    assert receipt["status"] == "UNVERIFIED"
    assert receipt["rank_claim_proven"] is False
    assert receipt["model_form"] == "UNVERIFIED"
    assert receipt["ensemble"] is None
    assert receipt["routing"] is None
    assert receipt["reason"] == "public_rank_model_eligibility_unverified"


def _managed_receipt(artifact="a" * 64):
    return {
        "model_form": "single_model_single_pass",
        "component_model_count": 1,
        "inference_passes_per_sample": 1,
        "dataset_specific_routing": False,
        "model_artifact_sha256": artifact,
        "managed_model_lineage_sha256": "b" * 64,
        "evaluation_bundle_sha256": "d" * 64,
    }


def test_managed_eligibility_is_derived_from_validated_production_receipts(
        tmp_path, monkeypatch):
    from orze.core import benchmark_contract, model_lineage

    idea_dir = tmp_path / "managed-idea"
    idea_dir.mkdir()
    receipt_bytes = json.dumps(
        _managed_receipt(), sort_keys=True).encode()
    (idea_dir / "benchmark.json").write_bytes(receipt_bytes)
    monkeypatch.setattr(
        benchmark_contract, "validate_benchmark_receipt",
        lambda idea, cfg: (True, "verified"))
    monkeypatch.setattr(
        benchmark_contract, "get_benchmark_contract",
        lambda cfg: {"receipt": "benchmark.json"})
    monkeypatch.setattr(
        model_lineage, "validate_model_lineage_for_evaluation",
        lambda idea, cfg, include_artifact_manifest=False: ({
            "idea_id": "managed-idea",
            "attempt_id": "attempt-1",
            "execution_identity_sha256": "f" * 64,
            "artifact_sha256": "a" * 64,
            "artifact_files": 1,
            "artifact_bytes": 5,
        }, "b" * 64, _artifact_manifest()))

    evidence = REAL_MANAGED_CONTEXT(
        idea_dir, {}, MODEL_ID).evidence

    assert evidence == {
        "verification_method": public_rank.ELIGIBILITY_METHOD,
        "verifier_source_sha256": hashlib.sha256(
            Path(public_rank.__file__).read_bytes()).hexdigest(),
        "model_id": MODEL_ID,
        "idea_id": "managed-idea",
        "attempt_id": "attempt-1",
        "execution_identity_sha256": "f" * 64,
        "model_artifact_sha256": "a" * 64,
        "model_lineage_sha256": "b" * 64,
        "benchmark_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "evaluation_bundle_sha256": "d" * 64,
        "artifact_manifest_sha256": public_rank._canonical_sha256({
            "schema_version": 1,
            "hash_method": "sha256_bytes_v1",
            "files": [{
                "path": "model.bin",
                "size": 5,
                "sha256": hashlib.sha256(b"model").hexdigest(),
            }],
        }),
        "artifact_files": 1,
        "artifact_bytes": 5,
    }


def test_managed_eligibility_rejects_benchmark_lineage_artifact_mismatch(
        tmp_path, monkeypatch):
    from orze.core import benchmark_contract, model_lineage

    idea_dir = tmp_path / "managed-idea"
    idea_dir.mkdir()
    (idea_dir / "benchmark.json").write_text(
        json.dumps(_managed_receipt(artifact="e" * 64)))
    monkeypatch.setattr(
        benchmark_contract, "validate_benchmark_receipt",
        lambda idea, cfg: (True, "verified"))
    monkeypatch.setattr(
        benchmark_contract, "get_benchmark_contract",
        lambda cfg: {"receipt": "benchmark.json"})
    monkeypatch.setattr(
        model_lineage, "validate_model_lineage_for_evaluation",
        lambda idea, cfg, include_artifact_manifest=False: ({
            "idea_id": "managed-idea",
            "attempt_id": "attempt-1",
            "execution_identity_sha256": "f" * 64,
            "artifact_sha256": "a" * 64,
            "artifact_files": 1,
            "artifact_bytes": 5,
        }, "b" * 64, _artifact_manifest()))

    with pytest.raises(
            public_rank.PublicRankError,
            match="public_rank_managed_eligibility_mismatch"):
        REAL_MANAGED_CONTEXT(idea_dir, {}, MODEL_ID)


def test_managed_eligibility_rejects_unverified_benchmark(
        tmp_path, monkeypatch):
    from orze.core import benchmark_contract

    idea_dir = tmp_path / "managed-idea"
    idea_dir.mkdir()
    monkeypatch.setattr(
        benchmark_contract, "validate_benchmark_receipt",
        lambda idea, cfg: (False, "benchmark_receipt_missing"))

    with pytest.raises(
            public_rank.PublicRankError,
            match=("public_rank_benchmark_invalid:"
                   "benchmark_receipt_missing")):
        REAL_MANAGED_CONTEXT(idea_dir, {}, MODEL_ID)


def _publication_fixture(*, extra=None, regular_body=b"{}", lfs_body=b"weights"):
    files = [
        {
            "path": "config.json",
            "size": len(regular_body),
            "sha256": hashlib.sha256(regular_body).hexdigest(),
        },
        {
            "path": "model.safetensors",
            "size": len(lfs_body),
            "sha256": hashlib.sha256(lfs_body).hexdigest(),
        },
    ]
    core = {
        "schema_version": 1,
        "hash_method": "sha256_bytes_v1",
        "files": files,
    }
    manifest = {**core, "manifest_sha256": public_rank._canonical_sha256(core)}
    declaration = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_form": "single_model_single_pass",
        "component_model_count": 1,
        "inference_passes_per_sample": 1,
        "dataset_specific_routing": False,
        "artifact_manifest_sha256": manifest["manifest_sha256"],
    }
    readme_body = (
        b"# Standalone model\n\n<!-- orze-publication-identity-v1\n"
        + json.dumps(
            declaration, sort_keys=True, separators=(",", ":")
        ).encode()
        + b"\n-->\n"
    )
    siblings = [
        {
            "rfilename": ".gitattributes",
            "size": 10,
            "blobId": "1" * 40,
            "lfs": None,
        },
        {
            "rfilename": "README.md",
            "size": len(readme_body),
            "blobId": "2" * 40,
            "lfs": None,
        },
        {
            "rfilename": "config.json",
            "size": len(regular_body),
            "blobId": "3" * 40,
            "lfs": None,
        },
        {
            "rfilename": "model.safetensors",
            "size": len(lfs_body),
            "blobId": "4" * 40,
            "lfs": {
                "sha256": hashlib.sha256(lfs_body).hexdigest(),
                "size": len(lfs_body),
                "pointerSize": 128,
            },
        },
    ]
    if extra is not None:
        siblings.append(extra)
    api = {
        "id": MODEL_ID,
        "sha": "5" * 40,
        "private": False,
        "gated": False,
        "siblings": siblings,
    }
    api_url = "https://huggingface.co/api/models/" + MODEL_ID + "?blobs=true"
    file_url = (
        "https://huggingface.co/" + MODEL_ID + "/resolve/" + "5" * 40
        + "/config.json"
    )
    readme_url = (
        "https://huggingface.co/" + MODEL_ID + "/resolve/" + "5" * 40
        + "/README.md"
    )

    def transport(method, url, body, timeout, max_bytes):
        assert method == "GET"
        assert body is None
        if url == api_url:
            return EndpointResponse(
                json.dumps(api).encode(), url,
                content_type="application/json")
        if url == file_url:
            return EndpointResponse(
                regular_body, url, content_type="application/json")
        if url == readme_url:
            return EndpointResponse(
                readme_body, url, content_type="text/markdown")
        raise AssertionError(url)

    return manifest, transport


def test_publication_identity_binds_exact_hub_revision_and_file_bytes():
    manifest, transport = _publication_fixture()

    evidence = REAL_PUBLICATION_IDENTITY(
        manifest, MODEL_ID, SUBMISSION_URL, transport)

    assert evidence["verification_method"] == (
        public_rank.PUBLICATION_IDENTITY_METHOD)
    assert evidence["hub_commit_sha"] == "5" * 40
    assert evidence["artifact_manifest_sha256"] == manifest["manifest_sha256"]
    assert evidence["artifact_files"] == 2
    assert evidence["matched_payload_file_count"] == 2
    assert evidence["lfs_payload_file_count"] == 1
    assert evidence["regular_payload_file_count"] == 1
    assert evidence["ignored_metadata_files"] == [
        ".gitattributes", "README.md"]
    assert evidence["model_card_evidence"]["response_sha256"]
    assert evidence["model_card_evidence"]["declaration_sha256"] == (
        public_rank._canonical_sha256({
            "schema_version": 1,
            "model_id": MODEL_ID,
            "model_form": "single_model_single_pass",
            "component_model_count": 1,
            "inference_passes_per_sample": 1,
            "dataset_specific_routing": False,
            "artifact_manifest_sha256": manifest["manifest_sha256"],
        })
    )
    assert evidence["regular_file_evidence"][0]["path"] == "config.json"


def test_publication_identity_rejects_extra_ensemble_payload():
    manifest, transport = _publication_fixture(extra={
        "rfilename": "consensus.py",
        "size": 1,
        "blobId": "6" * 40,
        "lfs": None,
    })

    with pytest.raises(
            public_rank.PublicRankError,
            match="public_rank_hub_artifact_file_set_mismatch:extra=consensus.py"):
        REAL_PUBLICATION_IDENTITY(
            manifest, MODEL_ID, SUBMISSION_URL, transport)


def test_publication_identity_rejects_remote_byte_mismatch():
    manifest, base_transport = _publication_fixture(regular_body=b"{}")

    def changed_remote(method, url, body, timeout, max_bytes):
        response = base_transport(method, url, body, timeout, max_bytes)
        if url.endswith("/config.json"):
            return EndpointResponse(
                b"[]", response.final_url,
                content_type=response.content_type)
        return response

    with pytest.raises(
            public_rank.PublicRankError,
            match="public_rank_hub_artifact_bytes_mismatch"):
        REAL_PUBLICATION_IDENTITY(
            manifest, MODEL_ID, SUBMISSION_URL, changed_remote)


def test_publication_identity_requires_canonical_model_url():
    manifest, transport = _publication_fixture()

    with pytest.raises(
            public_rank.PublicRankError,
            match="public_rank_submission_url_not_canonical_model"):
        REAL_PUBLICATION_IDENTITY(
            manifest, MODEL_ID,
            "https://huggingface.co/org/different-model", transport)


def test_publication_identity_rejects_regular_file_redirect_off_hub():
    manifest, base_transport = _publication_fixture()

    def redirected(method, url, body, timeout, max_bytes):
        response = base_transport(method, url, body, timeout, max_bytes)
        if url.endswith("/config.json"):
            return EndpointResponse(
                response.body, "https://github.com/org/repo/config.json",
                content_type=response.content_type)
        return response

    with pytest.raises(
            public_rank.PublicRankError,
            match="public_rank_hub_file_redirect_invalid"):
        REAL_PUBLICATION_IDENTITY(
            manifest, MODEL_ID, SUBMISSION_URL, redirected)


def test_publication_identity_hashes_zero_byte_regular_files():
    manifest, transport = _publication_fixture(regular_body=b"")

    evidence = REAL_PUBLICATION_IDENTITY(
        manifest, MODEL_ID, SUBMISSION_URL, transport)

    regular = evidence["regular_file_evidence"]
    assert regular[0]["response_bytes"] == 0
    assert regular[0]["response_sha256"] == hashlib.sha256(b"").hexdigest()


def test_publication_identity_rejects_model_card_composite_declaration():
    manifest, base_transport = _publication_fixture()

    def composite_card(method, url, body, timeout, max_bytes):
        response = base_transport(method, url, body, timeout, max_bytes)
        if url.endswith("/README.md"):
            changed = response.body.replace(
                b'"component_model_count":1',
                b'"component_model_count":2',
            )
            assert len(changed) == len(response.body)
            return EndpointResponse(
                changed, response.final_url,
                content_type=response.content_type)
        return response

    with pytest.raises(
            public_rank.PublicRankError,
            match="public_rank_model_card_declaration_mismatch"):
        REAL_PUBLICATION_IDENTITY(
            manifest, MODEL_ID, SUBMISSION_URL, composite_card)
