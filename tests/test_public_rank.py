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
REAL_MANAGED_ELIGIBILITY = public_rank._managed_eligibility_evidence
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
    }


@pytest.fixture(autouse=True)
def _managed_eligibility_stub(monkeypatch):
    def derive(idea_dir, cfg, model_id):
        if idea_dir is None and cfg is None:
            return None
        assert idea_dir == Path("/managed")
        assert cfg == {}
        return _eligibility(model_id)
    monkeypatch.setattr(
        public_rank, "_managed_eligibility_evidence", derive)


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
        lambda idea, cfg: ({
            "idea_id": "managed-idea",
            "attempt_id": "attempt-1",
            "execution_identity_sha256": "f" * 64,
            "artifact_sha256": "a" * 64,
        }, "b" * 64))

    evidence = REAL_MANAGED_ELIGIBILITY(
        idea_dir, {}, MODEL_ID)

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
        lambda idea, cfg: ({
            "idea_id": "managed-idea",
            "attempt_id": "attempt-1",
            "execution_identity_sha256": "f" * 64,
            "artifact_sha256": "a" * 64,
        }, "b" * 64))

    with pytest.raises(
            public_rank.PublicRankError,
            match="public_rank_managed_eligibility_mismatch"):
        REAL_MANAGED_ELIGIBILITY(idea_dir, {}, MODEL_ID)


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
        REAL_MANAGED_ELIGIBILITY(idea_dir, {}, MODEL_ID)
