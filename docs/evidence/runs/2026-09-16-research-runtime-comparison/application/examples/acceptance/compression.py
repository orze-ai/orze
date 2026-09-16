"""Finite lossless-container acceptance application, not Core codec policy.

The worker imports only this file and the stdlib-only sibling common module.
The three deliberately fixed candidates are compared on complete container
bytes, never JSON hex transport size or only their compressed payload.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time


DEFAULT_DATASET = (b"A" * 64 + b"\x00" + b"B" * 64).hex()
COUNTERFACTUAL_DATASET = (b"A" * 64 + b"B" * 64).hex()
PROTOCOL = "container-raw1-rle1-hex1-be32-v1"
ADAPTER_ID = "acceptance.compression.v1"
MAX_DATASET_BYTES = 2048
_HEX = re.compile(r"(?:[0-9a-f]{2})*\Z")
_CANDIDATES = {"baseline", "challenger", "unchecked"}


def _dataset_digest(value):
    # Same explicit JSON-data identity used by the common example transport.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def dataset_bytes(value):
    """JSON binary transport is explicit, bounded, canonical lower-case hex."""
    if (type(value) is not str or len(value) > 2 * MAX_DATASET_BYTES
            or _HEX.fullmatch(value) is None):
        raise ValueError("compression dataset requires bounded canonical hex")
    return bytes.fromhex(value)


def encode_container(data, candidate):
    if type(data) is not bytes or len(data) > MAX_DATASET_BYTES:
        raise ValueError("compression dataset is not bounded bytes")
    if candidate == "baseline":
        magic, payload = b"RAW1", data
    elif candidate == "unchecked":
        magic, payload = b"HEX1", data.hex().encode("ascii")
    elif candidate == "challenger":
        magic = b"RLE1"
        # This intentionally lossy candidate is a scientific negative control.
        # The original declared length is not adjusted to hide discarded NULs.
        stripped = data.replace(b"\x00", b"")
        encoded = bytearray()
        offset = 0
        while offset < len(stripped):
            end = offset + 1
            while end < len(stripped) and stripped[end] == stripped[offset] and end - offset < 255:
                end += 1
            encoded.extend((end - offset, stripped[offset]))
            offset = end
        payload = bytes(encoded)
    else:
        raise ValueError("unknown compression candidate")
    return magic + len(data).to_bytes(4, "big") + payload


def decode_container(container):
    """Strict versioned decoder; exact framing excludes trailing garbage."""
    if type(container) is not bytes or not 8 <= len(container) <= 8 + 2 * MAX_DATASET_BYTES:
        raise ValueError("container_size")
    magic, declared, payload = container[:4], int.from_bytes(container[4:8], "big"), container[8:]
    if declared > MAX_DATASET_BYTES:
        raise ValueError("decoded_size")
    if magic == b"RAW1":
        if len(payload) != declared:
            raise ValueError("raw_length")
        return payload
    if magic == b"HEX1":
        if len(payload) != declared * 2:
            raise ValueError("hex_length")
        try:
            text = payload.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("hex_alphabet") from exc
        if _HEX.fullmatch(text) is None:
            raise ValueError("hex_alphabet")
        return bytes.fromhex(text)
    if magic == b"RLE1":
        if len(payload) % 2:
            raise ValueError("rle_truncated_pair")
        decoded = bytearray()
        for offset in range(0, len(payload), 2):
            count, value = payload[offset:offset + 2]
            if count == 0:
                raise ValueError("rle_zero_count")
            if len(decoded) + count > declared:
                raise ValueError("rle_length")
            decoded.extend(bytes((value,)) * count)
        if len(decoded) != declared:
            raise ValueError("rle_length")
        return bytes(decoded)
    raise ValueError("container_magic")


def check_roundtrip(container, dataset):
    try:
        decoded = decode_container(container)
    except ValueError as exc:
        return "invalid", str(exc)
    if decoded != dataset:
        return "invalid", "roundtrip_mismatch"
    return "valid", "exact_roundtrip"


def _container(details):
    if type(details) is not dict or type(details.get("container_hex")) is not str:
        raise ValueError("missing container hex")
    text = details["container_hex"]
    if len(text) > 2 * (8 + 2 * MAX_DATASET_BYTES) or _HEX.fullmatch(text) is None:
        raise ValueError("container hex is not bounded canonical encoding")
    return bytes.fromhex(text)


def _analyze(inputs, source_results):
    dataset = dataset_bytes(inputs["dataset"])
    if len(source_results) != 3:
        raise ValueError("analysis requires the three actual source artifacts")
    expected_digest = _dataset_digest(inputs["dataset"])
    checks, candidates = [], set()
    unchecked = None
    for artifact_id, result in source_results.items():
        if (type(result) is not dict or type(result.get("version")) is not int
                or result["version"] != 1 or result.get("operation") != "measure"
                or result.get("candidate") not in _CANDIDATES
                or result.get("candidate") in candidates
                or result.get("dataset_sha256") != expected_digest
                or result.get("source_artifact_ids") != []):
            raise ValueError("source is not a distinct measurement of this dataset")
        candidates.add(result["candidate"])
        container = _container(result.get("details"))
        if type(result.get("cost")) is not int or result["cost"] != len(container):
            raise ValueError("source cost excludes container bytes")
        status, reason = check_roundtrip(container, dataset)
        checks.append({"artifact_id": artifact_id, "candidate": result["candidate"],
            "container_hex": container.hex(), "cost": len(container),
            "status": status, "reason_code": reason})
        if result["candidate"] == "unchecked":
            unchecked = container
    if candidates != _CANDIDATES or unchecked is None:
        raise ValueError("analysis source candidates incomplete")
    return {"container_hex": unchecked.hex(), "source_checks": checks}


class CompressionDomain:
    def __init__(self, config):
        if type(config) is not dict or set(config) != {"dataset"}:
            raise ValueError("compression config requires only dataset")
        self.dataset = dataset_bytes(config["dataset"]).hex()

    def prepare(self, request, sources):
        from .common import prepare
        return prepare(request, sources, dataset=self.dataset,
            worker_path=Path(__file__).resolve(), adapter_id=ADAPTER_ID, protocol=PROTOCOL)

    def interpret(self, prepared, envelope):
        from .common import claim
        inputs = prepared["action"]["inputs"]
        data = dataset_bytes(inputs["dataset"])
        container = _container(envelope["details"])
        if type(envelope.get("cost")) is not int or envelope["cost"] != len(container):
            raise ValueError("measurement cost is not full container bytes")
        if inputs["operation"] == "measure" and inputs["candidate"] == "unchecked":
            return claim(prepared, envelope, status="unknown", reason_code="validation_deferred")
        if inputs["operation"] == "analyze":
            checks = envelope["details"].get("source_checks")
            bindings = inputs["source_bindings"]
            if type(checks) is not list or len(checks) != 3:
                raise ValueError("analysis omitted source checks")
            reconstructed = {}
            for check in checks:
                if type(check) is not dict or set(check) != {
                        "artifact_id", "candidate", "container_hex", "cost", "status", "reason_code"}:
                    raise ValueError("invalid analysis check")
                value = _container(check)
                if check["cost"] != len(value) or (check["status"], check["reason_code"]) != check_roundtrip(value, data):
                    raise ValueError("analysis misstates a roundtrip check")
                reconstructed[check["artifact_id"]] = {"version": 1, "operation": "measure",
                    "candidate": check["candidate"], "dataset_sha256": _dataset_digest(inputs["dataset"]),
                    "source_artifact_ids": [], "cost": check["cost"], "details": {"container_hex": check["container_hex"]}}
            if list(reconstructed) != [item["artifact_id"] for item in bindings]:
                raise ValueError("analysis did not check the actual source ID list")
            verified = _analyze(inputs, reconstructed)
            if verified != envelope["details"]:
                raise ValueError("analysis changed the unchecked container")
        status, reason = check_roundtrip(container, data)
        return claim(prepared, envelope, status=status, reason_code=reason)


def main():
    from common import read_inputs, read_source_results, write_result
    started_cpu, started_wall = time.process_time(), time.monotonic()
    inputs = read_inputs()
    dataset = dataset_bytes(inputs["dataset"])
    if inputs["operation"] == "measure":
        details = {"container_hex": encode_container(dataset, inputs["candidate"]).hex()}
    elif inputs["operation"] == "analyze" and inputs["candidate"] == "unchecked":
        details = _analyze(inputs, read_source_results(inputs))
    else:
        raise ValueError("unknown compression operation")
    write_result(inputs, len(_container(details)), details,
                 started_cpu=started_cpu, started_wall=started_wall)


if __name__ == "__main__":
    main()
