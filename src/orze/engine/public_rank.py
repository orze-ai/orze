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
import os
import re
import socket
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping
import urllib.request
from urllib.parse import quote, unquote, urljoin, urlsplit

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
ELIGIBILITY_METHOD = "managed_model_lineage_and_single_pass_preflight_v1"
PUBLICATION_IDENTITY_METHOD = "hf_public_model_revision_exact_files_v1"
_HUB_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_BLOB_RE = re.compile(r"[0-9a-f]{40}")
_MODEL_CARD_DECLARATION_RE = re.compile(
    rb"<!--\s*orze-publication-identity-v1\s*\r?\n"
    rb"(\{.*?\})\s*\r?\n-->",
    re.DOTALL,
)


class PublicRankError(ValueError):
    """Raised when public evidence cannot support a rank claim."""


@dataclass(frozen=True)
class EndpointResponse:
    body: bytes
    final_url: str
    status: int = 200
    content_type: str = "application/json"


Transport = Callable[[str, str, bytes | None, int, int], EndpointResponse]


@dataclass(frozen=True)
class ManagedEligibilityContext:
    evidence: dict
    artifact_manifest: dict


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
    if len(result.body) > max_bytes:
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
    allow_empty: bool = False,
) -> EndpointResponse:
    response = transport(method, url, body, timeout, _MAX_RESPONSE_BYTES)
    if (not isinstance(response, EndpointResponse) or response.status != 200
            or (not response.body and not allow_empty)
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


def _read_stable_regular(path: Path, max_bytes: int = 1024 * 1024) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PublicRankError("public_rank_eligibility_receipt_unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or not 1 <= before.st_size <= max_bytes):
            raise PublicRankError("public_rank_eligibility_receipt_invalid")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            data = handle.read(max_bytes + 1)
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
    if len(data) > max_bytes or identity_before != identity_after:
        raise PublicRankError("public_rank_eligibility_receipt_changed")
    return data


def _managed_eligibility_context(
    idea_dir: Path | None,
    cfg: Mapping | None,
    model_id: str,
) -> ManagedEligibilityContext | None:
    """Derive eligibility only from the managed production evidence chain."""
    if idea_dir is None and cfg is None:
        return None
    if idea_dir is None or not isinstance(cfg, Mapping):
        raise PublicRankError("public_rank_managed_eligibility_context_invalid")
    from orze.core.benchmark_contract import (
        get_benchmark_contract,
        validate_benchmark_receipt,
    )
    from orze.core.model_lineage import validate_model_lineage_for_evaluation

    idea_dir = Path(idea_dir)
    if idea_dir.is_symlink() or not idea_dir.is_dir():
        raise PublicRankError("public_rank_managed_idea_invalid")
    valid, reason = validate_benchmark_receipt(idea_dir, cfg)
    if not valid:
        raise PublicRankError("public_rank_benchmark_invalid:" + reason)
    try:
        lineage, lineage_sha256, artifact_manifest = (
            validate_model_lineage_for_evaluation(
                idea_dir, cfg, include_artifact_manifest=True)
        )
    except Exception as exc:
        raise PublicRankError("public_rank_model_lineage_invalid") from exc
    contract = get_benchmark_contract(cfg)
    if not isinstance(contract, Mapping):
        raise PublicRankError("public_rank_benchmark_contract_missing")
    receipt_name = contract.get("receipt")
    if not isinstance(receipt_name, str) or not receipt_name:
        raise PublicRankError("public_rank_benchmark_receipt_invalid")
    root = idea_dir.resolve(strict=True)
    receipt_path = (idea_dir / receipt_name).resolve(strict=True)
    try:
        receipt_path.relative_to(root)
    except ValueError as exc:
        raise PublicRankError("public_rank_benchmark_receipt_escaped") from exc
    receipt_bytes = _read_stable_regular(receipt_path)
    receipt = _strict_json(receipt_bytes)
    artifact_sha256 = lineage.get("artifact_sha256")
    evaluation_bundle_sha256 = receipt.get("evaluation_bundle_sha256")
    if (not isinstance(receipt, Mapping)
            or receipt.get("model_form") != "single_model_single_pass"
            or receipt.get("component_model_count") != 1
            or receipt.get("inference_passes_per_sample") != 1
            or receipt.get("dataset_specific_routing") is not False
            or receipt.get("model_artifact_sha256") != artifact_sha256
            or receipt.get("managed_model_lineage_sha256") != lineage_sha256
            or not isinstance(evaluation_bundle_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", evaluation_bundle_sha256) is None):
        raise PublicRankError("public_rank_managed_eligibility_mismatch")
    evidence = {
        "verification_method": ELIGIBILITY_METHOD,
        "verifier_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "model_id": model_id,
        "idea_id": lineage["idea_id"],
        "attempt_id": lineage["attempt_id"],
        "execution_identity_sha256": lineage["execution_identity_sha256"],
        "model_artifact_sha256": artifact_sha256,
        "model_lineage_sha256": lineage_sha256,
        "benchmark_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "evaluation_bundle_sha256": evaluation_bundle_sha256,
        "artifact_manifest_sha256": artifact_manifest["manifest_sha256"],
        "artifact_files": lineage["artifact_files"],
        "artifact_bytes": lineage["artifact_bytes"],
    }
    return ManagedEligibilityContext(evidence, artifact_manifest)


def _managed_eligibility_evidence(
    idea_dir: Path | None,
    cfg: Mapping | None,
    model_id: str,
) -> dict | None:
    """Compatibility wrapper returning only receipt-safe eligibility fields."""
    context = _managed_eligibility_context(idea_dir, cfg, model_id)
    return None if context is None else context.evidence


def _canonical_sha256(value: Mapping) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PublicRankError("public_rank_publication_manifest_invalid") from exc
    return hashlib.sha256(encoded).hexdigest()


def _safe_hub_path(value) -> str:
    if (not isinstance(value, str) or not value or "\\" in value
            or "\x00" in value):
        raise PublicRankError("public_rank_hub_file_path_invalid")
    path = PurePosixPath(value)
    if (path.is_absolute() or value.startswith("./") or value.endswith("/")
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.as_posix() != value):
        raise PublicRankError("public_rank_hub_file_path_invalid")
    return value


def _is_publication_metadata(path: str) -> bool:
    if path in {".gitattributes", "README.md"}:
        return True
    parts = PurePosixPath(path).parts
    return (
        len(parts) >= 2
        and parts[0] == ".eval_results"
        and PurePosixPath(path).suffix.lower() in {".yaml", ".yml"}
    )


def _canonical_submission_url(model_id: str, url: str) -> None:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname != "huggingface.co"
            or parsed.username is not None or parsed.password is not None
            or parsed.port not in (None, 443) or parsed.query or parsed.fragment
            or parsed.path.rstrip("/") != "/" + model_id):
        raise PublicRankError("public_rank_submission_url_not_canonical_model")


def _validate_hub_file_response_url(
    final_url: str, *, model_id: str, commit: str, path: str,
) -> None:
    parsed = urlsplit(final_url)
    decoded_path = unquote(parsed.path)
    allowed_paths = {
        f"/{model_id}/resolve/{commit}/{path}",
        f"/api/resolve-cache/models/{model_id}/{commit}/{path}",
    }
    if (parsed.scheme != "https" or parsed.hostname != "huggingface.co"
            or parsed.username is not None or parsed.password is not None
            or parsed.port not in (None, 443) or parsed.fragment
            or decoded_path not in allowed_paths):
        raise PublicRankError("public_rank_hub_file_redirect_invalid")


def _publication_identity_evidence(
    artifact_manifest: Mapping,
    model_id: str,
    public_submission_url: str,
    transport: Transport,
) -> dict:
    """Bind managed local artifact bytes to one immutable public Hub commit."""
    _canonical_submission_url(model_id, public_submission_url)
    if not isinstance(artifact_manifest, Mapping):
        raise PublicRankError("public_rank_local_artifact_manifest_invalid")
    files = artifact_manifest.get("files")
    if (artifact_manifest.get("schema_version") != 1
            or artifact_manifest.get("hash_method") != "sha256_bytes_v1"
            or not isinstance(files, list) or not files
            or len(files) > 100_000):
        raise PublicRankError("public_rank_local_artifact_manifest_invalid")
    local_by_path = {}
    for item in files:
        path = _safe_hub_path(item.get("path") if isinstance(item, Mapping) else None)
        size = item.get("size") if isinstance(item, Mapping) else None
        digest = item.get("sha256") if isinstance(item, Mapping) else None
        if (set(item) != {"path", "size", "sha256"}
                or path in local_by_path or isinstance(size, bool)
                or not isinstance(size, int) or size < 0
                or not isinstance(digest, str)
                or _SHA256_RE.fullmatch(digest) is None):
            raise PublicRankError("public_rank_local_artifact_manifest_invalid")
        local_by_path[path] = {"path": path, "size": size, "sha256": digest}
    local_core = {
        "schema_version": 1,
        "hash_method": "sha256_bytes_v1",
        "files": [local_by_path[path] for path in sorted(local_by_path)],
    }
    local_manifest_sha256 = _canonical_sha256(local_core)
    if artifact_manifest.get("manifest_sha256") != local_manifest_sha256:
        raise PublicRankError("public_rank_local_artifact_manifest_invalid")

    api_url = "https://huggingface.co/api/models/" + model_id + "?blobs=true"
    api_response = _request(transport, "GET", api_url)
    if (api_response.final_url != api_url
            or not api_response.content_type.startswith("application/json")):
        raise PublicRankError("public_rank_hub_api_response_invalid")
    api_payload = _strict_json(api_response.body)
    commit = api_payload.get("sha") if isinstance(api_payload, Mapping) else None
    siblings = api_payload.get("siblings") if isinstance(
        api_payload, Mapping) else None
    if (api_payload.get("id") != model_id
            or api_payload.get("private") is not False
            or not isinstance(commit, str)
            or _HUB_COMMIT_RE.fullmatch(commit) is None
            or not isinstance(siblings, list) or not siblings
            or len(siblings) > 100_000):
        raise PublicRankError("public_rank_hub_model_metadata_invalid")

    remote_by_path = {}
    repository_identity = []
    for sibling in siblings:
        if not isinstance(sibling, Mapping):
            raise PublicRankError("public_rank_hub_file_metadata_invalid")
        path = _safe_hub_path(sibling.get("rfilename"))
        size = sibling.get("size")
        blob_id = sibling.get("blobId")
        lfs = sibling.get("lfs")
        if (path in remote_by_path or isinstance(size, bool)
                or not isinstance(size, int) or size < 0
                or not isinstance(blob_id, str)
                or _GIT_BLOB_RE.fullmatch(blob_id) is None):
            raise PublicRankError("public_rank_hub_file_metadata_invalid")
        lfs_sha256 = None
        if lfs is not None:
            if (not isinstance(lfs, Mapping)
                    or not isinstance(lfs.get("sha256"), str)
                    or _SHA256_RE.fullmatch(lfs["sha256"]) is None
                    or isinstance(lfs.get("size"), bool)
                    or lfs.get("size") != size):
                raise PublicRankError("public_rank_hub_lfs_metadata_invalid")
            lfs_sha256 = lfs["sha256"]
        remote_by_path[path] = {
            "path": path,
            "size": size,
            "blob_id": blob_id,
            "lfs_sha256": lfs_sha256,
        }
        repository_identity.append(remote_by_path[path])

    ignored = sorted(
        path for path in remote_by_path if _is_publication_metadata(path))
    payload_paths = set(remote_by_path) - set(ignored)
    if payload_paths != set(local_by_path):
        extra = sorted(payload_paths - set(local_by_path))
        missing = sorted(set(local_by_path) - payload_paths)
        reason = "public_rank_hub_artifact_file_set_mismatch"
        if extra:
            reason += ":extra=" + quote(",".join(extra[:8]), safe="")
        if missing:
            reason += ":missing=" + quote(",".join(missing[:8]), safe="")
        raise PublicRankError(reason)

    model_card = remote_by_path.get("README.md")
    if (model_card is None or model_card["lfs_sha256"] is not None
            or not 1 <= model_card["size"] <= 1024 * 1024):
        raise PublicRankError("public_rank_model_card_metadata_invalid")
    model_card_url = (
        "https://huggingface.co/" + model_id + "/resolve/" + commit
        + "/README.md"
    )
    model_card_response = _request(transport, "GET", model_card_url)
    _validate_hub_file_response_url(
        model_card_response.final_url,
        model_id=model_id,
        commit=commit,
        path="README.md",
    )
    if len(model_card_response.body) != model_card["size"]:
        raise PublicRankError("public_rank_model_card_size_mismatch")
    declarations = _MODEL_CARD_DECLARATION_RE.findall(
        model_card_response.body)
    if len(declarations) != 1:
        raise PublicRankError("public_rank_model_card_declaration_missing")
    declaration = _strict_json(declarations[0])
    expected_declaration = {
        "schema_version": 1,
        "model_id": model_id,
        "model_form": "single_model_single_pass",
        "component_model_count": 1,
        "inference_passes_per_sample": 1,
        "dataset_specific_routing": False,
        "artifact_manifest_sha256": local_manifest_sha256,
    }
    if declaration != expected_declaration:
        raise PublicRankError("public_rank_model_card_declaration_mismatch")
    model_card_evidence = {
        **_endpoint_evidence(
            model_card_response, model_card_url, "GET"),
        "declaration_sha256": _canonical_sha256(declaration),
    }

    remote_files = []
    regular_file_evidence = []
    lfs_files = 0
    for path in sorted(payload_paths):
        remote = remote_by_path[path]
        if remote["lfs_sha256"] is not None:
            digest = remote["lfs_sha256"]
            lfs_files += 1
        else:
            if remote["size"] > _MAX_RESPONSE_BYTES:
                raise PublicRankError(
                    "public_rank_hub_regular_file_too_large_to_hash")
            file_url = (
                "https://huggingface.co/" + model_id + "/resolve/" + commit
                + "/" + quote(path, safe="/")
            )
            response = _request(
                transport, "GET", file_url, allow_empty=True)
            _validate_hub_file_response_url(
                response.final_url, model_id=model_id, commit=commit, path=path)
            if len(response.body) != remote["size"]:
                raise PublicRankError("public_rank_hub_file_size_mismatch")
            digest = hashlib.sha256(response.body).hexdigest()
            regular_file_evidence.append({
                "path": path,
                **_endpoint_evidence(response, file_url, "GET"),
            })
        remote_files.append({
            "path": path,
            "size": remote["size"],
            "sha256": digest,
        })
    remote_core = {
        "schema_version": 1,
        "hash_method": "sha256_bytes_v1",
        "files": remote_files,
    }
    remote_manifest_sha256 = _canonical_sha256(remote_core)
    if remote_manifest_sha256 != local_manifest_sha256:
        raise PublicRankError("public_rank_hub_artifact_bytes_mismatch")
    return {
        "verification_method": PUBLICATION_IDENTITY_METHOD,
        "model_id": model_id,
        "public_submission_url": public_submission_url,
        "hub_commit_sha": commit,
        "artifact_manifest_sha256": local_manifest_sha256,
        "artifact_files": len(local_by_path),
        "artifact_bytes": sum(item["size"] for item in local_by_path.values()),
        "hub_repository_file_count": len(remote_by_path),
        "matched_payload_file_count": len(remote_files),
        "lfs_payload_file_count": lfs_files,
        "regular_payload_file_count": len(regular_file_evidence),
        "ignored_metadata_files": ignored,
        "hub_repository_identity_sha256": _canonical_sha256({
            "schema_version": 1,
            "commit": commit,
            "files": sorted(repository_identity, key=lambda item: item["path"]),
        }),
        "hub_api_evidence": _endpoint_evidence(
            api_response, api_url, "GET"),
        "model_card_evidence": model_card_evidence,
        "regular_file_evidence": regular_file_evidence,
    }


def verify_open_asr_public_rank(
    model_id: str,
    public_submission_url: str,
    *,
    idea_dir: Path | None = None,
    cfg: Mapping | None = None,
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
        eligibility_context = _managed_eligibility_context(
            idea_dir, cfg, model_id)
        eligibility = (
            None if eligibility_context is None
            else eligibility_context.evidence
        )
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
        publication_identity = _publication_identity_evidence(
            eligibility_context.artifact_manifest,
            model_id,
            public_submission_url,
            request,
        )
        receipt["eligibility_evidence"] = eligibility
        receipt["publication_identity_evidence"] = publication_identity
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
    parser.add_argument("--project-config", type=Path)
    parser.add_argument("--idea-dir", type=Path)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if (args.project_config is None) != (args.idea_dir is None):
        parser.error("--project-config and --idea-dir must be supplied together")
    cfg = None
    if args.project_config is not None:
        from orze.core.config import load_project_config
        cfg = load_project_config(str(args.project_config))
    receipt = verify_open_asr_public_rank(
        args.model_id, args.submission_url, idea_dir=args.idea_dir, cfg=cfg,
    )
    atomic_write(
        Path(args.output), json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
