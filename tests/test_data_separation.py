"""Privacy-safe, fail-closed train/evaluation separation tests."""

import hashlib
import hmac
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import orze.core.data_separation as separation_module
from orze.core.config import _validate_config
from orze.core.data_separation import (
    DataSeparationError,
    ensure_data_separation,
)
from orze.engine.execution_identity import compute_execution_identity
from orze.engine.launcher import LaunchIntegrityError, launch


NAMESPACE = "a" * 64
NORMALIZATION = "b" * 64
KEY = b"test-only-manifest-key"


def _fingerprint(value: str) -> str:
    return hmac.new(KEY, value.encode("utf-8"), hashlib.sha256).hexdigest()


def _write_manifest(path: Path, role: str, rows: list[dict],
                    fields=("sample", "speaker", "source"),
                    **header_overrides) -> str:
    header = {
        "schema_version": 1,
        "role": role,
        "fingerprint_algorithm": "hmac-sha256",
        "fingerprint_namespace_sha256": NAMESPACE,
        "normalization_contract_sha256": NORMALIZATION,
        "fields": list(fields),
    }
    header.update(header_overrides)
    content = "\n".join(
        json.dumps(value, sort_keys=True, separators=(",", ":"))
        for value in [header, *rows]
    ) + "\n"
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _row(sample: str, speaker: str, source: str) -> dict:
    return {
        "sample": _fingerprint(sample),
        "speaker": _fingerprint(speaker),
        "source": _fingerprint(source),
    }


def _config(tmp_path: Path, train_rows=None, evaluation_rows=None) -> dict:
    train = tmp_path / "train.manifest.jsonl"
    evaluation = tmp_path / "evaluation.manifest.jsonl"
    train_digest = _write_manifest(
        train, "train", train_rows or [_row("train-1", "speaker-1", "source-1")])
    evaluation_digest = _write_manifest(
        evaluation, "evaluation",
        evaluation_rows or [_row("eval-1", "speaker-2", "source-2")],
    )
    return {
        "_project_root": str(tmp_path),
        "_orze_dir": str(tmp_path / ".orze"),
        "data_separation": {
            "enabled": True,
            "train_manifest": str(train),
            "train_manifest_sha256": train_digest,
            "evaluation_manifest": str(evaluation),
            "evaluation_manifest_sha256": evaluation_digest,
            "fingerprint_namespace_sha256": NAMESPACE,
            "normalization_contract_sha256": NORMALIZATION,
            "fields": ["sample", "speaker", "source"],
            "max_overlap": {"sample": 0, "speaker": 0, "source": 0},
            "max_records": 100,
            "max_bytes": 1024 * 1024,
            "max_line_bytes": 4096,
        },
    }


def test_disjoint_keyed_manifests_write_aggregate_only_receipt(tmp_path):
    cfg = _config(
        tmp_path,
        [_row("train-1", "speaker-1", "source-1"),
         _row("train-2", "speaker-1", "source-1")],
        [_row("eval-1", "speaker-2", "source-2")],
    )

    receipt = ensure_data_separation(cfg)

    assert receipt["status"] == "passed"
    assert receipt["records"] == {"train": 2, "evaluation": 1}
    assert receipt["overlap"] == {"sample": 0, "speaker": 0, "source": 0}
    assert receipt["fingerprints_persisted"] is False
    assert receipt["rank_claim_proven"] is False
    receipt_text = (
        tmp_path / ".orze" / "state" / "data_separation.json"
    ).read_text(encoding="utf-8")
    for raw in ("train-1", "eval-1", _fingerprint("train-1")):
        assert raw not in receipt_text
    assert not list((tmp_path / ".orze" / "state").glob("data-separation-*"))


@pytest.mark.parametrize("field", ["sample", "speaker", "source"])
def test_cross_manifest_overlap_fails_by_configured_dimension(tmp_path, field):
    train = _row("train", "train-speaker", "train-source")
    evaluation = _row("eval", "eval-speaker", "eval-source")
    evaluation[field] = train[field]
    cfg = _config(tmp_path, [train], [evaluation])

    with pytest.raises(
            DataSeparationError, match=f"overlap_exceeded:{field}"):
        ensure_data_separation(cfg)


def test_duplicate_sample_fingerprint_within_manifest_fails(tmp_path):
    first = _row("same", "speaker-1", "source-1")
    second = _row("same", "speaker-2", "source-2")
    cfg = _config(tmp_path, [first, second], [_row("eval", "s3", "src3")])

    with pytest.raises(DataSeparationError, match="duplicate_sample"):
        ensure_data_separation(cfg)


def test_declared_nonzero_population_overlap_is_audited_not_hidden(tmp_path):
    train = _row("train", "speaker-1", "shared-source")
    evaluation = _row("eval", "speaker-2", "shared-source")
    cfg = _config(tmp_path, [train], [evaluation])
    cfg["data_separation"]["max_overlap"]["source"] = 1

    receipt = ensure_data_separation(cfg)

    assert receipt["overlap"]["source"] == 1
    assert receipt["max_overlap"]["source"] == 1


def test_same_size_backdated_manifest_rewrite_invalidates_cache(tmp_path):
    cfg = _config(tmp_path)
    ensure_data_separation(cfg)
    path = Path(cfg["data_separation"]["evaluation_manifest"])
    before = path.stat()
    original = path.read_text(encoding="utf-8")
    changed = original.replace(_fingerprint("eval-1"), _fingerprint("eval-9"))
    assert len(changed) == len(original)
    path.write_text(changed, encoding="utf-8")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))

    with pytest.raises(DataSeparationError, match="digest_mismatch"):
        ensure_data_separation(cfg)


def test_unchanged_manifests_reuse_hash_bound_receipt(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    first = ensure_data_separation(cfg)
    monkeypatch.setattr(
        separation_module,
        "_read_manifest",
        lambda *args, **kwargs: pytest.fail("unchanged manifests were reread"),
    )

    second = ensure_data_separation(cfg)

    assert second == first


def test_corrupt_cached_receipt_is_rebuilt_from_manifests(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    ensure_data_separation(cfg)
    receipt_path = tmp_path / ".orze" / "state" / "data_separation.json"
    envelope = json.loads(receipt_path.read_text(encoding="utf-8"))
    envelope["payload"]["overlap"]["sample"] = 999
    envelope["payload_sha256"] = separation_module._payload_hash(
        envelope["payload"])
    receipt_path.write_text(json.dumps(envelope), encoding="utf-8")
    calls = 0
    original = separation_module._read_manifest

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(separation_module, "_read_manifest", counted)

    rebuilt = ensure_data_separation(cfg)

    assert calls == 2
    assert rebuilt["overlap"]["sample"] == 0


def test_concurrent_launchers_share_one_manifest_audit(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    calls = 0
    calls_lock = threading.Lock()
    original = separation_module._read_manifest

    def delayed(*args, **kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        return original(*args, **kwargs)

    monkeypatch.setattr(separation_module, "_read_manifest", delayed)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(
            lambda _: ensure_data_separation(cfg), range(2)))

    assert calls == 2
    assert all(result["status"] == "passed" for result in results)


def test_manifest_record_and_byte_limits_fail_closed(tmp_path):
    cfg = _config(
        tmp_path,
        [_row("train-1", "speaker-1", "source-1"),
         _row("train-2", "speaker-2", "source-2")],
    )
    cfg["data_separation"]["max_records"] = 1
    with pytest.raises(DataSeparationError, match="record_limit"):
        ensure_data_separation(cfg)

    cfg = _config(tmp_path)
    cfg["data_separation"]["max_bytes"] = 1
    with pytest.raises(DataSeparationError, match="size_invalid"):
        ensure_data_separation(cfg)


@pytest.mark.parametrize(
    "mutation, reason",
    [
        (lambda cfg: Path(cfg["data_separation"]["train_manifest"]).unlink(),
         "manifest_unavailable"),
        (lambda cfg: cfg["data_separation"].update(max_records=0),
         "policy_invalid"),
        (lambda cfg: cfg["data_separation"].update(fields=["speaker"]),
         "policy_invalid"),
    ],
)
def test_invalid_or_missing_policy_fails_closed(tmp_path, mutation, reason):
    cfg = _config(tmp_path)
    mutation(cfg)

    with pytest.raises(DataSeparationError, match=reason):
        ensure_data_separation(cfg)


def test_redirected_manifest_is_rejected(tmp_path):
    cfg = _config(tmp_path)
    original = Path(cfg["data_separation"]["train_manifest"])
    outside = tmp_path / "outside.jsonl"
    original.replace(outside)
    original.symlink_to(outside)

    with pytest.raises(DataSeparationError, match="manifest_redirected"):
        ensure_data_separation(cfg)


def test_header_namespace_and_role_are_exact(tmp_path):
    cfg = _config(tmp_path)
    train = Path(cfg["data_separation"]["train_manifest"])
    cfg["data_separation"]["train_manifest_sha256"] = _write_manifest(
        train,
        "evaluation",
        [_row("train", "speaker", "source")],
    )

    with pytest.raises(DataSeparationError, match="header_invalid"):
        ensure_data_separation(cfg)


def test_record_schema_rejects_raw_or_extra_content(tmp_path):
    cfg = _config(tmp_path)
    train = Path(cfg["data_separation"]["train_manifest"])
    cfg["data_separation"]["train_manifest_sha256"] = _write_manifest(
        train,
        "train",
        [{**_row("train", "speaker", "source"),
          "transcript": "private transcript must not persist"}],
    )

    with pytest.raises(DataSeparationError, match="record_invalid"):
        ensure_data_separation(cfg)

    receipt = tmp_path / ".orze" / "state" / "data_separation.json"
    assert not receipt.exists()


def test_config_validation_requires_complete_exact_contract(tmp_path):
    cfg = _config(tmp_path)
    cfg["data_separation"].update({
        "fields": ["sample", "sample"],
        "max_overlap": {"sample": -1},
        "fingerprint_namespace_sha256": "not-a-digest",
    })

    errors, _ = _validate_config(cfg)

    assert any("data_separation.fields" in error for error in errors)
    assert any("data_separation.max_overlap" in error for error in errors)
    assert any("fingerprint_namespace_sha256" in error for error in errors)


def test_launch_rejects_overlap_before_gpu_telemetry(tmp_path, monkeypatch):
    shared = _row("shared", "shared-speaker", "shared-source")
    cfg = _config(tmp_path, [shared], [shared])
    results = tmp_path / "results"
    (results / "idea-overlap").mkdir(parents=True)
    train_script = tmp_path / "train.py"
    train_script.write_text("# trainer\n", encoding="utf-8")
    base = tmp_path / "base.yaml"
    base.write_text("{}\n", encoding="utf-8")
    ideas = tmp_path / "ideas.md"
    ideas.write_text("", encoding="utf-8")
    cfg.update({
        "train_script": str(train_script),
        "base_config": str(base),
        "ideas_file": str(ideas),
        "python": sys.executable,
    })
    gpu_checks = []
    monkeypatch.setattr(
        "orze.engine.launcher._verify_gpu_free",
        lambda *args: gpu_checks.append(True),
    )

    with pytest.raises(LaunchIntegrityError, match="overlap_exceeded"):
        launch("idea-overlap", 4, results, cfg)

    assert gpu_checks == []


def test_execution_identity_stays_stable_when_disabled_and_binds_when_enabled(
        tmp_path):
    config = tmp_path / "config.yaml"
    base = tmp_path / "base.yaml"
    script = tmp_path / "train.py"
    config.write_text("seed: 1\n", encoding="utf-8")
    base.write_text("{}\n", encoding="utf-8")
    script.write_text("# train\n", encoding="utf-8")
    kwargs = {
        "config_path": config,
        "base_config_path": base,
        "train_script": script,
        "python": sys.executable,
        "train_extra_args": [],
        "train_extra_env": {},
        "data_boundaries": {},
    }

    original = compute_execution_identity(**kwargs)
    disabled = compute_execution_identity(
        **kwargs, data_separation={"enabled": False, "changed": True})
    enabled = compute_execution_identity(
        **kwargs, data_separation={"enabled": True, "policy": "one"})
    changed = compute_execution_identity(
        **kwargs, data_separation={"enabled": True, "policy": "two"})

    assert disabled == original
    assert enabled != original
    assert changed != enabled
