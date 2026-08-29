import hashlib
import json
from pathlib import Path

import orze.engine.public_rank as public_rank
from orze.engine.public_rank import (
    EndpointResponse,
    LEADERBOARD_CALL_URL,
    LEADERBOARD_INFO_URL,
    LEADERBOARD_ORIGIN,
    verify_open_asr_public_rank,
)


MODEL_ID = "org/standalone-asr"
SUBMISSION_URL = "https://huggingface.co/org/standalone-asr"
DEFAULTS = [
    "RTFx ⬆️️", "AMI-Cleaned", "Private (scripted)",
    "Private (conversational)",
]


def _eligibility(model_id=MODEL_ID):
    return json.dumps({
        "schema_version": 1,
        "status": "VERIFIED",
        "verification_method": (
            "managed_model_lineage_and_single_pass_preflight_v1"),
        "verifier_source_sha256": hashlib.sha256(
            Path(public_rank.__file__).read_bytes()).hexdigest(),
        "model_id": model_id,
        "model_form": "single_model_single_pass",
        "component_model_count": 1,
        "inference_passes_per_sample": 1,
        "ensemble": False,
        "routing": False,
        "model_artifact_sha256": "a" * 64,
        "model_lineage_sha256": "b" * 64,
        "benchmark_receipt_sha256": "c" * 64,
        "evaluation_bundle_sha256": "d" * 64,
    }, sort_keys=True).encode()


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
        MODEL_ID, SUBMISSION_URL, eligibility_receipt=_eligibility(),
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
        MODEL_ID, SUBMISSION_URL, eligibility_receipt=_eligibility(),
        transport=transport)

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["rank_claim_proven"] is False
    assert receipt["reason"] == "public_rank_private_defaults_missing"


def test_public_rank_rejects_absent_model_instead_of_inventing_rank():
    transport = FakeTransport(table=_table([
        [1, "org/other", 4.0, 3.0, 8.0],
    ]))

    receipt = verify_open_asr_public_rank(
        MODEL_ID, SUBMISSION_URL, eligibility_receipt=_eligibility(),
        transport=transport)

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["rank_claim_proven"] is False
    assert receipt["reason"] == "public_rank_model_absent"


def test_public_rank_rejects_duplicate_model_rows():
    cell = (
        '<a href="https://huggingface.co/' + MODEL_ID + '">' + MODEL_ID + "</a>"
    )
    transport = FakeTransport(table=_table([
        [1, cell, 4.0, 3.0, 8.0],
        [2, cell, 5.0, 4.0, 9.0],
    ]))

    receipt = verify_open_asr_public_rank(
        MODEL_ID, SUBMISSION_URL, eligibility_receipt=_eligibility(),
        transport=transport)

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["reason"] == "public_rank_model_row_duplicated"


def test_public_rank_rejects_submission_without_model_identity():
    transport = FakeTransport(submission=b"some other public model")

    receipt = verify_open_asr_public_rank(
        MODEL_ID, SUBMISSION_URL, eligibility_receipt=_eligibility(),
        transport=transport)

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["reason"] == "public_rank_submission_model_missing"


def test_public_rank_rejects_fabricated_or_nonpublic_submission_url():
    receipt = verify_open_asr_public_rank(
        MODEL_ID, "https://example.test/invented",
        eligibility_receipt=_eligibility(), transport=FakeTransport())

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["rank_claim_proven"] is False
    assert receipt["reason"] == "public_rank_host_not_allowed"


def test_public_rank_rejects_table_rank_inconsistent_with_scores():
    transport = FakeTransport(table=_table([
        [1, "org/leader", 4.0, 3.0, 8.0],
        [9, MODEL_ID, 5.0, 2.0, 10.0],
    ]))

    receipt = verify_open_asr_public_rank(
        MODEL_ID, SUBMISSION_URL, eligibility_receipt=_eligibility(),
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
