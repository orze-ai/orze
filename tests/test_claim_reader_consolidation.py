"""Bounded claim-reader controls and explicitly new snapshot requirements.

Only TestLegacyClaimReader is a baseline behavior suite. TestNewSnapshot runs
after the new API exists; its absence is not evidence of an old product defect.
All filesystem mutations are confined to pytest's private temporary directory.
"""

import errno
import hashlib
import json
import math
import os
from pathlib import Path

import pytest

from orze.engine import claim_authority as authority
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt


def _sized_json(size):
    prefix, suffix = b'{"padding":"', b'"}'
    return prefix + b"x" * (size - len(prefix) - len(suffix)) + suffix


def _claim_fd(monkeypatch, path):
    """Observe actual opens; other descriptors remain entirely untouched."""
    real_open = os.open
    captured = []

    def observe_open(candidate, flags, *args, **kwargs):
        fd = real_open(candidate, flags, *args, **kwargs)
        if Path(candidate) == path:
            captured.append(fd)
        return fd

    monkeypatch.setattr(authority.os, "open", observe_open)
    return captured


class TestLegacyClaimReader:
    @pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
    def test_existing_json_formats_and_values(self, tmp_path, encoding):
        path = tmp_path / "claim.json"
        data = {
            "note": "排序 Δ",
            "unknown": {"accepted": [1, None, False]},
            "nan": float("nan"),
            "positive": float("inf"),
            "negative": float("-inf"),
        }
        path.write_bytes(json.dumps(data, ensure_ascii=False, indent=2).encode(encoding))

        result = authority.read_claim(path)

        assert result["note"] == data["note"]
        assert result["unknown"] == data["unknown"]
        assert math.isnan(result["nan"])
        assert result["positive"] == float("inf")
        assert result["negative"] == float("-inf")

    def test_initial_absence_is_optional(self, tmp_path):
        assert authority.read_claim(tmp_path / "missing.json") is None

    @pytest.mark.parametrize("raw", [b"", b"{", b'{"a":1,"a":2}', b"[]"])
    def test_existing_invalid_metadata_type_and_reason(self, tmp_path, raw):
        path = tmp_path / "claim.json"
        path.write_bytes(raw)

        with pytest.raises(AttemptEffectInDoubt, match="^claim_metadata_invalid$"):
            authority.read_claim(path)

    @pytest.mark.parametrize("size", [65536, 65537])
    def test_existing_default_size_boundary(self, tmp_path, size):
        path = tmp_path / "claim.json"
        raw = _sized_json(size)
        path.write_bytes(raw)

        if size == 65536:
            assert authority.read_claim(path) == json.loads(raw)
        else:
            with pytest.raises(AttemptEffectInDoubt, match="^claim_metadata_invalid$"):
                authority.read_claim(path)


class TestNewSnapshot:
    def test_raw_sha_single_open_read_and_close(self, tmp_path, monkeypatch):
        path = tmp_path / "claim.json"
        raw = ' \n { "note" : "排序 Δ", "unknown" : [ 1, null ] } \n'.encode("utf-8")
        path.write_bytes(raw)
        real_read, real_close = os.read, os.close
        opened = _claim_fd(monkeypatch, path)
        reads, closes = [], []

        def observe_read(fd, size):
            if fd in opened:
                reads.append((fd, size))
            return real_read(fd, size)

        def observe_close(fd):
            if fd in opened:
                closes.append(fd)
            return real_close(fd)

        monkeypatch.setattr(authority.os, "read", observe_read)
        monkeypatch.setattr(authority.os, "close", observe_close)

        value, raw_sha256 = authority.read_claim_snapshot(path)

        assert value == {"note": "排序 Δ", "unknown": [1, None]}
        assert raw_sha256 == hashlib.sha256(raw).hexdigest()
        assert raw_sha256 != hashlib.sha256(json.dumps(value).encode()).hexdigest()
        assert len(opened) == 1
        assert reads == [(opened[0], 65537)]
        assert closes == opened
        with pytest.raises(OSError) as caught:
            os.fstat(opened[0])
        assert caught.value.errno == errno.EBADF

    def test_initial_missing_required_and_optional(self, tmp_path):
        path = tmp_path / "missing.json"

        assert authority.read_claim_snapshot(path) is None
        with pytest.raises(FileNotFoundError):
            authority.read_claim_snapshot(path, required=True)

    @pytest.mark.parametrize("limit", [8192, 65536])
    def test_explicit_size_limits(self, tmp_path, limit):
        path = tmp_path / "claim.json"
        raw = _sized_json(limit)
        path.write_bytes(raw)

        assert authority.read_claim_snapshot(path, limit=limit, required=True) == (
            json.loads(raw), hashlib.sha256(raw).hexdigest()
        )

        path.write_bytes(_sized_json(limit + 1))
        with pytest.raises(AttemptEffectBusy):
            authority.read_claim_snapshot(path, limit=limit, required=True)
        if limit == 8192:
            assert authority.read_claim(path) == json.loads(path.read_bytes())

    @pytest.mark.parametrize(
        "options", [{"limit": True}, {"limit": 0}, {"limit": 65537}, {"required": 1}]
    )
    def test_invalid_options_are_value_errors_before_open(self, tmp_path, monkeypatch, options):
        path = tmp_path / "claim.json"
        path.write_bytes(b"{}")

        def unexpected_open(*args, **kwargs):
            raise AssertionError("invalid reader options must not open a file")

        monkeypatch.setattr(authority.os, "open", unexpected_open)
        with pytest.raises(ValueError):
            authority.read_claim_snapshot(path, **options)

    def test_snapshot_format_rejection_is_busy_not_in_doubt(self, tmp_path):
        path = tmp_path / "claim.json"
        path.write_bytes(b'{"a":1,"a":2}')

        with pytest.raises(AttemptEffectBusy) as caught:
            authority.read_claim_snapshot(path, required=True)

        assert not isinstance(caught.value, AttemptEffectInDoubt)
        with pytest.raises(AttemptEffectInDoubt, match="^claim_metadata_invalid$"):
            authority.read_claim(path)

    @pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "parent_symlink"])
    def test_redirected_or_nonregular_files_are_busy(self, tmp_path, kind):
        path = tmp_path / "claim.json"
        target = tmp_path / "target.json"
        if kind == "fifo":
            os.mkfifo(path)
        elif kind == "parent_symlink":
            directory = tmp_path / "real"
            directory.mkdir()
            (directory / "claim.json").write_bytes(b"{}")
            link = tmp_path / "linked"
            link.symlink_to(directory, target_is_directory=True)
            path = link / "claim.json"
        else:
            target.write_bytes(b"{}")
            if kind == "symlink":
                path.symlink_to(target)
            else:
                os.link(target, path)

        with pytest.raises(AttemptEffectBusy):
            authority.read_claim_snapshot(path)

    @pytest.mark.parametrize("change", ["short_read", "same_bytes_inode", "mode", "disappear"])
    def test_after_read_identity_changes_never_return_a_snapshot(self, tmp_path, monkeypatch, change):
        path = tmp_path / "claim.json"
        path.write_bytes(b"{}\n")
        path.chmod(0o600)
        replacement = tmp_path / "replacement.json"
        replacement.write_bytes(b"{}\n")
        real_read = os.read
        opened = _claim_fd(monkeypatch, path)
        injected = []

        def changed_read(fd, size):
            raw = real_read(fd, size)
            if fd in opened and not injected:
                injected.append(change)
                if change == "short_read":
                    # Still valid JSON: rejection must not rely on a parse error.
                    return raw[:-1]
                if change == "same_bytes_inode":
                    replacement.replace(path)
                elif change == "mode":
                    # This real permission change also changes ctime; it is not
                    # presented as a synthetic mode-only filesystem event.
                    path.chmod(0o640)
                else:
                    path.unlink()
            return raw

        monkeypatch.setattr(authority.os, "read", changed_read)
        rejection = OSError if change == "disappear" else AttemptEffectBusy
        with pytest.raises(rejection):
            authority.read_claim_snapshot(path, required=False)
        assert injected == [change]

    @pytest.mark.parametrize("operation", ["read", "close"])
    def test_read_and_close_os_errors_propagate(self, tmp_path, monkeypatch, operation):
        path = tmp_path / "claim.json"
        path.write_bytes(b"{}")
        real_read, real_close = os.read, os.close
        opened = _claim_fd(monkeypatch, path)

        def failed_read(fd, size):
            if fd in opened and operation == "read":
                raise OSError(errno.EIO, "injected claim read failure")
            return real_read(fd, size)

        def failed_close(fd):
            # Close the actual owned descriptor before simulating loss of its
            # completion response, so even this negative fixture leaks no FD.
            result = real_close(fd)
            if fd in opened and operation == "close":
                raise OSError(errno.EIO, "injected claim close failure")
            return result

        monkeypatch.setattr(authority.os, "read", failed_read)
        monkeypatch.setattr(authority.os, "close", failed_close)

        with pytest.raises(OSError) as caught:
            authority.read_claim_snapshot(path, required=False)

        assert caught.value.errno == errno.EIO
        assert len(opened) == 1
        with pytest.raises(OSError) as closed:
            os.fstat(opened[0])
        assert closed.value.errno == errno.EBADF

    def test_legacy_wrapper_uses_public_snapshot_projection(self, tmp_path, monkeypatch):
        path = tmp_path / "claim.json"
        path.write_bytes(b'{"unknown":1}')
        missing = tmp_path / "missing.json"
        real_snapshot = authority.read_claim_snapshot
        calls = []

        def observed(candidate, *, limit=65536, required=False):
            calls.append((Path(candidate), limit, required))
            return real_snapshot(candidate, limit=limit, required=required)

        monkeypatch.setattr(authority, "read_claim_snapshot", observed)

        assert authority.read_claim(path) == {"unknown": 1}
        assert authority.read_claim(missing) is None
        assert calls == [(path, 65536, False), (missing, 65536, False)]


class TestArchitecture:
    def test_public_snapshot_api_is_available(self):
        # Deliberate baseline architecture assertion, not a missing import or
        # an assertion that any historical execution behavior was defective.
        assert callable(getattr(authority, "read_claim_snapshot", None))
