"""Verify Open ASR leaderboard rank from the public rendered data surface.

The verifier intentionally uses the same public Gradio endpoint that renders
the landing page.  It derives the default column set from the endpoint schema,
requires both private aggregates to be defaults, and computes ranks from the
returned rows.  A receipt therefore records observed public evidence rather
than accepting caller-supplied rank numbers.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import html
import ipaddress
import json
import math
import re
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping
import urllib.request
from urllib.parse import urljoin, urlsplit

from orze.core.fs import atomic_write


SCHEMA_VERSION = 1
VERIFICATION_METHOD = "open_asr_public_gradio_v1"
LEADERBOARD_ORIGIN = "https://hf-audio-open-asr-leaderboard.hf.space"
LEADERBOARD_INFO_URL = LEADERBOARD_ORIGIN + "/gradio_api/info"
LEADERBOARD_CALL_URL = (
    LEADERBOARD_ORIGIN + "/gradio_api/call/_main_table_update"
)
LEADERBOARD_PUBLIC_URL = "https://huggingface.co/spaces/hf-audio/open_asr_leaderboard"
DEFAULT_TRACK_COLUMNS = {
    "private_scripted": "Private (scripted)",
    "private_conversational": "Private (conversational)",
}
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_MAX_ROWS = 10000
_EVENT_ID_RE = re.compile(r"[A-Za-z0-9_-]{16,128}")
_MODEL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")
_HF_HREF_RE = re.compile(
    r'''href=["']https://huggingface\.co/([^"'?#]+)["']''', re.IGNORECASE,
)
_ELIGIBILITY_FIELDS = {
    "schema_version", "status", "verification_method", "model_id",
    "model_form", "component_model_count", "inference_passes_per_sample",
    "ensemble", "routing", "model_artifact_sha256", "model_lineage_sha256",
    "benchmark_receipt_sha256", "evaluation_bundle_sha256",
    "verifier_source_sha256",
}
ELIGIBILITY_METHOD = "managed_model_lineage_and_single_pass_preflight_v1"


class PublicRankError(ValueError):
    """Raised when public evidence cannot support a rank claim."""


@dataclass(frozen=True)
class EndpointResponse:
    body: bytes
    final_url: str
    status: int = 200
    content_type: str = "application/json"


Transport = Callable[[str, str, bytes | None, int, int], EndpointResponse]


def _strict_json(data: bytes, *, allow_nan_as_none: bool = False):
    def strict_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise PublicRankError(f"public_json_duplicate_key:{key}")
            result[key] = value
        return result

    def reject_constant(value):
        if allow_nan_as_none and value == "NaN":
            return None
        raise PublicRankError(f"public_json_nonfinite_number:{value}")

    try:
        return json.loads(
            data.decode("utf-8"), object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicRankError("public_json_invalid") from exc


def _allowed_url(url: str, *, leaderboard: bool) -> None:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.port not in (None, 443) or parsed.fragment):
        raise PublicRankError("public_rank_url_invalid")
    host = parsed.hostname.lower().rstrip(".")
    if leaderboard:
        allowed = host == "hf-audio-open-asr-leaderboard.hf.space"
    else:
        allowed = host in {
            "huggingface.co", "www.huggingface.co", "github.com",
            "www.github.com", "api.github.com",
        }
    if not allowed:
        raise PublicRankError("public_rank_host_not_allowed")


def _require_public_dns(url: str) -> None:
    host = urlsplit(url).hostname
    assert host is not None
    try:
        addresses = {
            item[4][0] for item in socket.getaddrinfo(
                host, 443, type=socket.SOCK_STREAM)
        }
    except OSError as exc:
        raise PublicRankError("public_rank_dns_failed") from exc
    if not addresses:
        raise PublicRankError("public_rank_dns_empty")
    for address in addresses:
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError as exc:
            raise PublicRankError("public_rank_dns_invalid") from exc
        if not parsed.is_global:
            raise PublicRankError("public_rank_dns_not_global")


class _ValidatedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # The fixed Gradio endpoints do not redirect. Public model/PR pages may
        # redirect only within their explicitly allowed public host set.
        _allowed_url(newurl, leaderboard=False)
        _require_public_dns(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _default_transport(
    method: str,
    url: str,
    body: bytes | None,
    timeout: int,
    max_bytes: int,
) -> EndpointResponse:
    leaderboard = url.startswith(LEADERBOARD_ORIGIN + "/")
    _allowed_url(url, leaderboard=leaderboard)
    _require_public_dns(url)
    headers = {
        "Accept": "application/json, text/event-stream, text/html",
        "User-Agent": "orze-public-rank/1.0",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url, data=body, headers=headers, method=method,
    )
    opener = urllib.request.build_opener(_ValidatedRedirectHandler())
    try:
        with opener.open(request, timeout=timeout) as response:
            payload = response.read(max_bytes + 1)
            result = EndpointResponse(
                body=payload,
                final_url=response.geturl(),
                status=response.status,
                content_type=response.headers.get("Content-Type", ""),
            )
    except OSError as exc:
        raise PublicRankError("public_rank_request_failed") from exc
    if result.status != 200:
        raise PublicRankError("public_rank_http_status_invalid")
    if not result.body or len(result.body) > max_bytes:
        raise PublicRankError("public_rank_response_size_invalid")
    _allowed_url(result.final_url, leaderboard=leaderboard)
    return result


def _endpoint_evidence(
    response: EndpointResponse,
    request_url: str,
    method: str,
    request_body: bytes | None = None,
) -> dict:
    evidence = {
        "method": method,
        "request_url": request_url,
        "final_url": response.final_url,
        "http_status": response.status,
        "content_type": response.content_type,
        "response_bytes": len(response.body),
        "response_sha256": hashlib.sha256(response.body).hexdigest(),
    }
    if request_body is not None:
        evidence["request_sha256"] = hashlib.sha256(request_body).hexdigest()
    return evidence


def _request(
    transport: Transport,
    method: str,
    url: str,
    body: bytes | None = None,
    timeout: int = 30,
) -> EndpointResponse:
    response = transport(method, url, body, timeout, _MAX_RESPONSE_BYTES)
    if (not isinstance(response, EndpointResponse) or response.status != 200
            or not response.body
            or len(response.body) > _MAX_RESPONSE_BYTES):
        raise PublicRankError("public_rank_transport_response_invalid")
    return response


def _main_endpoint_defaults(info: Mapping) -> list[str]:
    endpoints = info.get("named_endpoints")
    endpoint = endpoints.get("/_main_table_update") if isinstance(
        endpoints, Mapping) else None
    parameters = endpoint.get("parameters") if isinstance(endpoint, Mapping) else None
    if not isinstance(parameters, list):
        raise PublicRankError("public_rank_endpoint_schema_missing")
    by_name = {}
    for item in parameters:
        if not isinstance(item, Mapping):
            raise PublicRankError("public_rank_endpoint_parameter_invalid")
        name = item.get("parameter_name")
        if not isinstance(name, str) or name in by_name:
            raise PublicRankError("public_rank_endpoint_parameter_invalid")
        by_name[name] = item.get("parameter_default")
    if set(by_name) != {
        "search_query", "show_proprietary", "show_llm", "selected_columns",
    }:
        raise PublicRankError("public_rank_endpoint_parameters_changed")
    selected = by_name["selected_columns"]
    if (by_name["show_proprietary"] is not True
            or by_name["show_llm"] is not False
            or not isinstance(selected, list)
            or not selected
            or any(not isinstance(item, str) for item in selected)
            or len(set(selected)) != len(selected)):
        raise PublicRankError("public_rank_endpoint_defaults_invalid")
    if not set(DEFAULT_TRACK_COLUMNS.values()).issubset(selected):
        raise PublicRankError("public_rank_private_defaults_missing")
    return selected


def _parse_complete_sse(body: bytes):
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PublicRankError("public_rank_sse_utf8_invalid") from exc
    complete = []
    for block in re.split(r"\r?\n\r?\n", text.strip()):
        event = None
        data = []
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data.append(line[6:])
        if event == "error":
            raise PublicRankError("public_rank_sse_error")
        if event == "complete" and data:
            complete.append("\n".join(data).encode("utf-8"))
    if len(complete) != 1:
        raise PublicRankError("public_rank_sse_complete_invalid")
    # Gradio serializes absent dataframe cells as the non-standard JSON token
    # NaN. Treat only that token as a missing value; infinities remain invalid.
    return _strict_json(complete[0], allow_nan_as_none=True)


def _number(value, reason: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PublicRankError(reason)
    result = float(value)
    if not math.isfinite(result):
        raise PublicRankError(reason)
    return result


def _positive_rank(value, reason: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PublicRankError(reason)
    return value


def _model_id_from_cell(cell) -> str | None:
    if not isinstance(cell, str):
        return None
    plain = html.unescape(cell).strip()
    if _MODEL_ID_RE.fullmatch(plain):
        return plain
    matches = [html.unescape(match).rstrip("/") for match in _HF_HREF_RE.findall(plain)]
    matches = [match for match in matches if _MODEL_ID_RE.fullmatch(match)]
    if len(set(matches)) != 1:
        return None
    return matches[0]


def _derive_table_claim(table_payload, model_id: str) -> dict:
    if not isinstance(table_payload, list) or len(table_payload) != 1:
        raise PublicRankError("public_rank_table_output_invalid")
    output = table_payload[0]
    value = output.get("value") if isinstance(output, Mapping) else None
    headers = value.get("headers") if isinstance(value, Mapping) else None
    rows = value.get("data") if isinstance(value, Mapping) else None
    if (not isinstance(headers, list) or not headers
            or any(not isinstance(item, str) for item in headers)
            or len(headers) != len(set(headers))
            or not isinstance(rows, list) or not rows or len(rows) > _MAX_ROWS):
        raise PublicRankError("public_rank_table_shape_invalid")
    required = {
        "Rank", "model", "Average WER ⬇️", *DEFAULT_TRACK_COLUMNS.values(),
    }
    if not required.issubset(headers):
        raise PublicRankError("public_rank_table_columns_missing")
    indexed = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(headers):
            raise PublicRankError("public_rank_table_row_invalid")
        indexed.append(dict(zip(headers, row)))
    matches = [
        row for row in indexed if _model_id_from_cell(row["model"]) == model_id
    ]
    if not matches:
        raise PublicRankError("public_rank_model_absent")
    if len(matches) != 1:
        raise PublicRankError("public_rank_model_row_duplicated")
    selected = matches[0]
    average = _number(selected["Average WER ⬇️"], "public_rank_average_invalid")
    displayed_rank = _positive_rank(selected["Rank"], "public_rank_landing_rank_invalid")
    averages = []
    for row in indexed:
        try:
            averages.append(_number(
                row["Average WER ⬇️"], "public_rank_average_invalid"))
        except PublicRankError:
            continue
    derived_landing_rank = 1 + sum(value < average for value in averages)
    if displayed_rank != derived_landing_rank:
        raise PublicRankError("public_rank_landing_rank_mismatch")
    tracks = {}
    for track_id, column in DEFAULT_TRACK_COLUMNS.items():
        score = _number(selected[column], "public_rank_track_score_invalid")
        population = []
        for row in indexed:
            try:
                population.append(_number(
                    row[column], "public_rank_track_score_invalid"))
            except PublicRankError:
                continue
        tracks[track_id] = {
            "accepted": True,
            "column": column,
            "rank": 1 + sum(value < score for value in population),
            "score": score,
            "ranked_models": len(population),
        }
    return {
        "landing_rank": displayed_rank,
        "landing_average_wer": average,
        "ranked_models": len(averages),
        "default_tracks": tracks,
    }


def _eligibility_evidence(data: bytes | None, model_id: str) -> dict | None:
    if data is None:
        return None
    if not data or len(data) > 1024 * 1024:
        raise PublicRankError("public_rank_eligibility_size_invalid")
    payload = _strict_json(data)
    if not isinstance(payload, Mapping) or set(payload) != _ELIGIBILITY_FIELDS:
        raise PublicRankError("public_rank_eligibility_schema_invalid")
    if (payload.get("schema_version") != 1
            or payload.get("status") != "VERIFIED"
            or payload.get("verification_method") != ELIGIBILITY_METHOD
            or payload.get("verifier_source_sha256") != hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest()
            or payload.get("model_id") != model_id
            or payload.get("model_form") != "single_model_single_pass"
            or payload.get("component_model_count") != 1
            or payload.get("inference_passes_per_sample") != 1
            or payload.get("ensemble") is not False
            or payload.get("routing") is not False):
        raise PublicRankError("public_rank_model_eligibility_invalid")
    for key in (
        "model_artifact_sha256", "model_lineage_sha256",
        "benchmark_receipt_sha256", "evaluation_bundle_sha256",
    ):
        value = payload.get(key)
        if not isinstance(value, str) or re.fullmatch(
                r"[0-9a-f]{64}", value) is None:
            raise PublicRankError("public_rank_eligibility_identity_invalid")
    return {
        "receipt_sha256": hashlib.sha256(data).hexdigest(),
        "verification_method": payload["verification_method"],
        "verifier_source_sha256": payload["verifier_source_sha256"],
        "model_artifact_sha256": payload["model_artifact_sha256"],
        "model_lineage_sha256": payload["model_lineage_sha256"],
        "benchmark_receipt_sha256": payload["benchmark_receipt_sha256"],
        "evaluation_bundle_sha256": payload["evaluation_bundle_sha256"],
    }


def verify_open_asr_public_rank(
    model_id: str,
    public_submission_url: str,
    *,
    eligibility_receipt: bytes | None = None,
    transport: Transport | None = None,
) -> dict:
    """Fetch public endpoints and derive one model's current default ranks."""
    now = datetime.datetime.now(datetime.timezone.utc)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "UNVERIFIED",
        "reason": "public_rank_evidence_incomplete",
        "verified_at": now.isoformat(),
        "rank_claim_proven": False,
        "verification_method": VERIFICATION_METHOD,
        "verifier_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "public_submission_url": public_submission_url,
        "public_leaderboard_url": LEADERBOARD_PUBLIC_URL,
        "model_id": model_id,
        "model_form": "UNVERIFIED",
        "ensemble": None,
        "routing": None,
        "public_endpoint_evidence": {},
    }
    try:
        if _MODEL_ID_RE.fullmatch(model_id) is None:
            raise PublicRankError("public_rank_model_id_invalid")
        _allowed_url(public_submission_url, leaderboard=False)
        eligibility = _eligibility_evidence(eligibility_receipt, model_id)
        request = transport or _default_transport

        submission = _request(request, "GET", public_submission_url)
        if model_id.encode("utf-8") not in submission.body:
            raise PublicRankError("public_rank_submission_model_missing")
        receipt["public_endpoint_evidence"]["submission"] = _endpoint_evidence(
            submission, public_submission_url, "GET")

        info_response = _request(request, "GET", LEADERBOARD_INFO_URL)
        defaults = _main_endpoint_defaults(_strict_json(info_response.body))
        receipt["default_dataset_columns"] = defaults
        receipt["public_endpoint_evidence"]["leaderboard_info"] = (
            _endpoint_evidence(info_response, LEADERBOARD_INFO_URL, "GET")
        )

        call_body = json.dumps(
            {"data": ["", True, False, defaults]},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        call_response = _request(
            request, "POST", LEADERBOARD_CALL_URL, call_body)
        call_payload = _strict_json(call_response.body)
        event_id = call_payload.get("event_id") if isinstance(
            call_payload, Mapping) else None
        if not isinstance(event_id, str) or _EVENT_ID_RE.fullmatch(event_id) is None:
            raise PublicRankError("public_rank_event_id_invalid")
        receipt["public_endpoint_evidence"]["leaderboard_call"] = (
            _endpoint_evidence(
                call_response, LEADERBOARD_CALL_URL, "POST", call_body)
        )

        result_url = urljoin(LEADERBOARD_CALL_URL + "/", event_id)
        result_response = _request(request, "GET", result_url, timeout=60)
        receipt["public_endpoint_evidence"]["leaderboard_result"] = (
            _endpoint_evidence(result_response, result_url, "GET")
        )
        table_claim = _derive_table_claim(
            _parse_complete_sse(result_response.body), model_id)
        receipt.update(table_claim)
        if eligibility is None:
            raise PublicRankError("public_rank_model_eligibility_unverified")
        receipt["eligibility_evidence"] = eligibility
        receipt["model_form"] = "single_model_single_pass"
        receipt["ensemble"] = False
        receipt["routing"] = False
        receipt["status"] = "VERIFIED"
        receipt["reason"] = "public_rank_derived_from_default_table"
        receipt["rank_claim_proven"] = True
    except (PublicRankError, OSError, ValueError) as exc:
        receipt["reason"] = str(exc)[:300] or type(exc).__name__
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a model against the public Open ASR landing table"
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--submission-url", required=True)
    parser.add_argument("--eligibility-receipt", type=Path)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt = verify_open_asr_public_rank(
        args.model_id, args.submission_url,
        eligibility_receipt=(
            args.eligibility_receipt.read_bytes()
            if args.eligibility_receipt is not None else None
        ),
    )
    atomic_write(
        Path(args.output), json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
