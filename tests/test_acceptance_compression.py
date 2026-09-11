"""Finite codec oracle controls, separate from the four full acceptance runs.

Expected bytes are hand-written protocol fixtures, not produced by the tested
decoder. One isolated real supervised worker checks sealed source-FD plumbing;
that unit fixture is not claimed to be a published SQLite source or a campaign.
"""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

from examples.acceptance import compression as codec


@pytest.mark.parametrize("wire,expected", [
    (b"RAW1\x00\x00\x00\x03A\x00B", b"A\x00B"),
    (b"RLE1\x00\x00\x00\x04\x02A\x01\x00\x01B", b"AA\x00B"),
    (b"HEX1\x00\x00\x00\x03" + b"410042", b"A\x00B"),
    (b"RAW1\x00\x00\x00\x00", b""),
])
def test_strict_decoder_known_wire_bytes(wire, expected):
    assert codec.decode_container(wire) == expected


@pytest.mark.parametrize("wire", [
    b"RAW1\x00\x00\x00",
    b"RAW2\x00\x00\x00\x01A",
    b"RAW1\x00\x00\x00\x01AB",
    b"RAW1\x00\x00\x00\x02A",
    b"RLE1\x00\x00\x00\x01\x00A",
    b"RLE1\x00\x00\x00\x01\x01A\x01",
    b"RLE1\x00\x00\x00\x01\x02A",
    b"RLE1\x00\x00\x00\x02\x01A",
    b"RLE1\x00\x00\x00\x01\x01A\x01B",
    b"HEX1\x00\x00\x00\x01ff00",
    b"HEX1\x00\x00\x00\x01FF",
    b"HEX1\x00\x00\x00\x01  ",
    b"HEX1\x00\x00\x00\x01gg",
    b"HEX1\xff\xff\xff\xff",
])
def test_decoder_rejects_bad_version_length_counts_alphabet_and_trailing(wire):
    with pytest.raises(ValueError):
        codec.decode_container(wire)


def test_actual_encoded_cost_counts_header_and_payload_not_hex_transport():
    data = b"A" * 64 + b"\x00" + b"B" * 64
    raw = codec.encode_container(data, "baseline")
    rle = codec.encode_container(data, "challenger")
    hexd = codec.encode_container(data, "unchecked")
    assert raw == b"RAW1\x00\x00\x00\x81" + data
    assert rle == b"RLE1\x00\x00\x00\x81\x40A\x40B"
    assert hexd == b"HEX1\x00\x00\x00\x81" + b"41" * 64 + b"00" + b"42" * 64
    assert (len(raw), len(rle), len(hexd)) == (137, 12, 266)


def test_lossy_challenger_only_becomes_valid_when_actual_input_changes():
    default, changed = bytes.fromhex(codec.DEFAULT_DATASET), bytes.fromhex(codec.COUNTERFACTUAL_DATASET)
    assert default == b"A" * 64 + b"\x00" + b"B" * 64
    assert changed == b"A" * 64 + b"B" * 64
    assert codec.check_roundtrip(b"RLE1\x00\x00\x00\x81\x40A\x40B", default) == ("invalid", "rle_length")
    assert codec.check_roundtrip(b"RLE1\x00\x00\x00\x80\x40A\x40B", changed) == ("valid", "exact_roundtrip")
    assert codec.check_roundtrip(b"RAW1\x00\x00\x00\x01X", b"Y") == ("invalid", "roundtrip_mismatch")


@pytest.mark.parametrize("config", [{"dataset": "FF"}, {"dataset": "0"}, {"dataset": "00 01"}, {"dataset": "", "codec": "rle"}])
def test_config_rejects_implicit_or_ambiguous_binary_encoding(config):
    with pytest.raises(ValueError):
        codec.CompressionDomain(config)


def sources():
    data = b"A" * 64 + b"\x00" + b"B" * 64
    wires = {
        "baseline": b"RAW1\x00\x00\x00\x81" + data,
        "challenger": b"RLE1\x00\x00\x00\x81\x40A\x40B",
        "unchecked": b"HEX1\x00\x00\x00\x81" + b"41" * 64 + b"00" + b"42" * 64,
    }
    return {"source-" + candidate: {"version": 1, "candidate": candidate,
        "operation": "measure", "dataset_sha256": hashlib.sha256(json.dumps(data.hex()).encode()).hexdigest(),
        "cost": len(wire), "details": {"container_hex": wire.hex()},
        "source_artifact_ids": [], "worker_cpu_seconds": 0.01, "worker_wall_seconds": 0.02}
        for candidate, wire in wires.items()}


def test_analysis_checks_all_sources_without_changing_unchecked_measurement():
    measurements = sources()
    original = copy.deepcopy(measurements)
    details = codec._analyze({"dataset": codec.DEFAULT_DATASET}, measurements)
    assert [check["candidate"] for check in details["source_checks"]] == ["baseline", "challenger", "unchecked"]
    assert [check["status"] for check in details["source_checks"]] == ["valid", "invalid", "valid"]
    assert [check["cost"] for check in details["source_checks"]] == [137, 12, 266]
    assert details["container_hex"] == measurements["source-unchecked"]["details"]["container_hex"]
    assert measurements == original


@pytest.mark.parametrize("change", ["missing", "foreign_dataset", "payload_only_cost"])
def test_analysis_rejects_missing_foreign_or_misreported_source(change):
    measurements = sources()
    if change == "missing":
        measurements.pop("source-baseline")
    elif change == "foreign_dataset":
        measurements["source-baseline"]["dataset_sha256"] = "0" * 64
    else:
        measurements["source-baseline"]["cost"] = 129
    with pytest.raises(ValueError):
        codec._analyze({"dataset": codec.DEFAULT_DATASET}, measurements)


def test_real_analysis_worker_reads_three_sealed_fds_and_returns_all_checks(tmp_path):
    from orze.engine.supervised_process import prepare_supervised, SupervisionUncertain
    descriptors, process = [], None
    def sealed(value):
        raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        writable = os.memfd_create("compression-unit", os.MFD_ALLOW_SEALING)
        try:
            assert os.write(writable, raw) == len(raw)
            fcntl.fcntl(writable, fcntl.F_ADD_SEALS,
                fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
            readonly = os.open("/proc/self/fd/" + str(writable), os.O_RDONLY)
            descriptors.append(readonly)
        finally:
            os.close(writable)
        return readonly, hashlib.sha256(raw).hexdigest()
    try:
        values = sources()
        source_fds, bindings = {}, []
        for identity, envelope in values.items():
            fd, digest = sealed(envelope)
            source_fds[identity] = fd
            bindings.append({"artifact_id": identity, "content_sha256": digest})
        inputs = {"dataset": codec.DEFAULT_DATASET, "candidate": "unchecked", "operation": "analyze", "source_bindings": bindings}
        input_fd, _ = sealed(inputs)
        env = {**os.environ, "ORZE_ACTION_INPUT_FD": str(input_fd), "ORZE_ACTION_SOURCE_FDS": json.dumps(source_fds),
               "CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": ""}
        try:
            process = prepare_supervised([sys.executable, str(Path(codec.__file__).resolve())],
                identity={"test": "compression-stdlib-worker"}, env=env, cwd=str(tmp_path),
                worker_only_fds=tuple(descriptors))
        except SupervisionUncertain as exc:
            process = exc.process
            raise
        process.start()
        assert process.wait(timeout=4) == 0
        result = json.loads((tmp_path / "result.json").read_text())
        assert result["source_artifact_ids"] == list(values)
        assert result["cost"] == 266
        assert [item["status"] for item in result["details"]["source_checks"]] == ["valid", "invalid", "valid"]
        assert result["worker_cpu_seconds"] >= 0
        assert result["worker_wall_seconds"] >= 0
    finally:
        if process is not None:
            if process.poll() is None:
                process.stop(timeout=.5)
            assert type(process.poll()) is int
        for fd in descriptors:
            os.close(fd)


def test_domain_keeps_original_unknown_and_emits_a_new_valid_worse_claim():
    from examples.acceptance.common import OUTPUTS
    domain = codec.CompressionDomain({"dataset": codec.DEFAULT_DATASET})
    measurements = sources()
    requested = {"version": 1, "purpose": "finite compression validation", "inputs": {},
        "timeout_seconds": 2, "outputs": copy.deepcopy(OUTPUTS), "input_artifact_ids": [],
        "payload": {"candidate": "unchecked", "operation": "measure"}}
    prepared = domain.prepare(requested, ())
    unknown = domain.interpret(prepared, measurements["source-unchecked"])
    captured_unknown = copy.deepcopy(unknown)
    assert prepared["action"]["command"] == [sys.executable, str(Path(codec.__file__).resolve())]
    assert unknown[0]["validation"] == {"status": "unknown", "reason_code": "validation_deferred"}
    assert unknown[0]["values"]["cost"] == 266
    source_records = tuple({"artifact_id": key, "content_sha256": hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
        for key, value in measurements.items())
    analysis_request = {**requested, "input_artifact_ids": list(measurements),
        "payload": {"candidate": "unchecked", "operation": "analyze"}}
    analysis = domain.prepare(analysis_request, source_records)
    envelope = {**measurements["source-unchecked"], "operation": "analyze",
        "source_artifact_ids": list(measurements),
        "details": codec._analyze(analysis["action"]["inputs"], measurements)}
    checked = domain.interpret(analysis, envelope)
    assert checked[0]["validation"] == {"status": "valid", "reason_code": "exact_roundtrip"}
    assert checked[0]["values"]["cost"] == 266 > measurements["source-baseline"]["cost"]
    assert checked[0]["values"]["operation"] == "analyze"
    assert checked[0]["comparison_scope"] == unknown[0]["comparison_scope"]
    assert unknown == captured_unknown


def test_unchecked_measurement_does_not_call_roundtrip_oracle(monkeypatch):
    from examples.acceptance.common import OUTPUTS
    calls = []
    def forbidden(*args):
        calls.append(args)
        raise AssertionError("unchecked measurement must defer scientific validation")
    monkeypatch.setattr(codec, "check_roundtrip", forbidden)
    domain = codec.CompressionDomain({"dataset": codec.DEFAULT_DATASET})
    requested = {"version": 1, "purpose": "defer the unchecked validation", "inputs": {},
        "timeout_seconds": 2, "outputs": copy.deepcopy(OUTPUTS), "input_artifact_ids": [],
        "payload": {"candidate": "unchecked", "operation": "measure"}}
    prepared = domain.prepare(requested, ())
    claims = domain.interpret(prepared, sources()["source-unchecked"])
    assert calls == []
    assert claims[0]["validation"] == {"status": "unknown", "reason_code": "validation_deferred"}
    assert claims[0]["values"]["cost"] == 266
