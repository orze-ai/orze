"""Receipt-only API contracts; these are new mechanism tests, not old reds."""

import json
import os
from pathlib import Path

import pytest

import orze.core.data_separation as separation
from test_data_separation import _config


def _forbid_audit(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("receipt observer attempted an audit or filesystem write")

    for name in ("_read_manifest", "_fs_lock", "atomic_write"):
        monkeypatch.setattr(separation, name, forbidden)
    monkeypatch.setattr(separation.tempfile, "TemporaryDirectory", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)


def test_valid_receipt_read_preserves_bytes_and_metadata(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    expected = separation.ensure_data_separation(cfg)
    receipt = tmp_path / ".orze/state/data_separation.json"
    before = receipt.read_bytes(), receipt.stat().st_mtime_ns
    _forbid_audit(monkeypatch)
    manifest_paths = {Path(cfg["data_separation"][f"{role}_manifest"])
                      for role in ("train", "evaluation")}
    original_open = Path.open

    def observed_open(path, *args, **kwargs):
        assert path not in manifest_paths, "observer read manifest content"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observed_open)

    assert separation.read_data_separation_receipt(cfg) == expected
    assert (receipt.read_bytes(), receipt.stat().st_mtime_ns) == before


def test_missing_control_directory_is_not_created(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    _forbid_audit(monkeypatch)

    with pytest.raises(separation.DataSeparationError, match="receipt_unavailable"):
        separation.read_data_separation_receipt(cfg)
    assert not (tmp_path / ".orze").exists()


@pytest.mark.parametrize("mutation", ["missing", "invalid_json", "digest", "overlap", "policy", "metadata"])
def test_unavailable_receipt_is_never_rebuilt(tmp_path, monkeypatch, mutation):
    cfg = _config(tmp_path)
    separation.ensure_data_separation(cfg)
    receipt = tmp_path / ".orze/state/data_separation.json"
    if mutation == "missing":
        receipt.unlink()
    elif mutation == "invalid_json":
        receipt.write_text("{", encoding="utf-8")
    elif mutation in {"digest", "overlap"}:
        envelope = json.loads(receipt.read_text(encoding="utf-8"))
        if mutation == "digest":
            envelope["payload_sha256"] = "0" * 64
        else:
            envelope["payload"]["overlap"]["sample"] = 99
            envelope["payload_sha256"] = separation._payload_hash(envelope["payload"])
        receipt.write_text(json.dumps(envelope), encoding="utf-8")
    elif mutation == "policy":
        cfg["data_separation"]["max_records"] += 1
    else:
        manifest = Path(cfg["data_separation"]["train_manifest"])
        previous = manifest.stat()
        os.utime(manifest, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000))
    before = receipt.read_bytes() if receipt.exists() else None
    _forbid_audit(monkeypatch)

    with pytest.raises(separation.DataSeparationError, match="receipt_unavailable"):
        separation.read_data_separation_receipt(cfg)
    assert (receipt.read_bytes() if receipt.exists() else None) == before


@pytest.mark.parametrize("redirect", ["root", "state", "receipt"])
def test_redirected_control_paths_are_rejected(tmp_path, monkeypatch, redirect):
    cfg = _config(tmp_path)
    separation.ensure_data_separation(cfg)
    original = {"root": tmp_path / ".orze", "state": tmp_path / ".orze/state",
                "receipt": tmp_path / ".orze/state/data_separation.json"}[redirect]
    moved = tmp_path / "original"
    original.rename(moved)
    original.symlink_to(moved, target_is_directory=redirect != "receipt")
    _forbid_audit(monkeypatch)

    with pytest.raises(separation.DataSeparationError):
        separation.read_data_separation_receipt(cfg)


def test_disabled_policy_is_readonly_without_control_paths(monkeypatch):
    _forbid_audit(monkeypatch)
    assert separation.read_data_separation_receipt({}) == {"status": "disabled"}


@pytest.mark.parametrize("spec", [[], {"enabled": True}, {"enabled": "yes"},
                                  {"enabled": 0}, {"enabled": ""}, {"enabled": None}])
def test_invalid_policy_cannot_be_promoted_to_passed(spec, monkeypatch):
    _forbid_audit(monkeypatch)
    with pytest.raises(separation.DataSeparationError):
        separation.read_data_separation_receipt({"data_separation": spec})


def test_manifest_change_during_receipt_read_is_rejected(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    separation.ensure_data_separation(cfg)
    original_cached = separation._cached_receipt
    receipt = tmp_path / ".orze/state/data_separation.json"
    before = receipt.read_bytes()

    def race(*args, **kwargs):
        payload = original_cached(*args, **kwargs)
        assert payload is not None
        manifest = Path(cfg["data_separation"]["evaluation_manifest"])
        previous = manifest.stat()
        os.utime(manifest, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000))
        return payload

    monkeypatch.setattr(separation, "_cached_receipt", race)
    _forbid_audit(monkeypatch)
    with pytest.raises(separation.DataSeparationError, match="manifest_changed"):
        separation.read_data_separation_receipt(cfg)
    assert receipt.read_bytes() == before
