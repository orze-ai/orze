"""New legacy receipt storage mechanisms; no real OS tree proof is claimed.

The inherited fixture uses real SQLite/files and explicit identity/poll/reaper
doubles. These tests never enumerate or signal host processes.
"""
import json
import os

import pytest

from orze.engine import process, roles
from test_trigger_role_completion import project, _state


def _held_before_poll(p):
    polls = []
    p.rp.process.poll = lambda: polls.append("polled") or 0
    result = roles.check_active_roles(p.active, ideas_file=str(p.root / "ideas.md"))
    assert result == []
    assert polls == []
    assert _state(p) == "STARTED"
    assert p.active == {"worker": p.rp}
    assert p.rp.lock_dir.is_dir()


def _is_receipt(fd, path):
    try:
        descriptor, named = os.fstat(fd), path.stat(follow_symlinks=False)
    except OSError:
        return False
    return (descriptor.st_dev, descriptor.st_ino) == (named.st_dev, named.st_ino)


def test_oversized_serialized_unicode_writer_keeps_existing_receipt(project):
    p = project
    before = p.receipt.read_bytes()
    entries = sorted(path.name for path in p.rp.lock_dir.iterdir())
    # This is an API metadata-size test, not a realistic long scope name.
    # JSON's escaped Unicode representation exceeds the byte budget although
    # the source field's UTF-8 bytes remain below it.
    value = "é" * (process._MAX_ROLE_RECEIPT_BYTES // 6 + 1)
    p.rp.trigger_launch["scope"] = value
    encoded_field = json.dumps({"scope": value}, separators=(",", ":")).encode("utf-8")
    assert len(value.encode("utf-8")) < process._MAX_ROLE_RECEIPT_BYTES
    assert len(encoded_field) > process._MAX_ROLE_RECEIPT_BYTES
    assert process.persist_role_process_receipt(p.rp) is False
    assert p.receipt.read_bytes() == before
    assert sorted(path.name for path in p.rp.lock_dir.iterdir()) == entries
    assert _state(p) == "STARTED"


@pytest.mark.parametrize("schema", [True, 2], ids=["bool-not-version-one", "v2-without-owner"])
def test_nonlegacy_schema_never_downgrades_to_legacy_completion(project, schema):
    p = project
    receipt = json.loads(p.receipt.read_bytes())
    receipt["schema_version"] = schema
    raw = (json.dumps(receipt, sort_keys=True) + "\n").encode()
    p.receipt.write_bytes(raw)
    _held_before_poll(p)
    assert p.receipt.read_bytes() == raw


@pytest.mark.parametrize("link", ["symlink", "hardlink"])
def test_redirected_receipt_cannot_be_classified_or_removed(project, link):
    p = project
    original = p.receipt.read_bytes()
    retained = p.root / "owned-retained-receipt.json"
    p.receipt.rename(retained)
    if link == "symlink":
        p.receipt.symlink_to(retained)
    else:
        os.link(retained, p.receipt)
    _held_before_poll(p)
    assert p.receipt.read_bytes() == original
    assert retained.read_bytes() == original
    if link == "symlink":
        assert p.receipt.is_symlink()
    else:
        assert retained.stat().st_nlink == 2


@pytest.mark.parametrize("fault", ["early_eof", "same_bytes_replacement"])
def test_incomplete_or_replaced_receipt_read_holds_before_poll(project, monkeypatch, fault):
    p = project
    original = p.receipt.read_bytes()
    original_inode = p.receipt.stat().st_ino
    retained = p.root / "owned-displaced-receipt.json"
    real_read = os.read
    injected = []

    def read(fd, size):
        target = _is_receipt(fd, p.receipt)
        if target and not injected:
            injected.append(fault)
            if fault == "early_eof":
                return b""
            chunk = real_read(fd, size)
            p.receipt.rename(retained)
            p.receipt.write_bytes(original)
            return chunk
        return real_read(fd, size)

    with monkeypatch.context() as patch:
        patch.setattr(os, "read", read)
        _held_before_poll(p)
    assert injected == [fault]
    assert p.receipt.read_bytes() == original
    if fault == "same_bytes_replacement":
        assert p.receipt.stat().st_ino != original_inode
        assert retained.stat().st_ino == original_inode
        assert retained.read_bytes() == original


@pytest.mark.parametrize("field", ["nonce", "attempt_id", "boolean_generation"])
def test_large_foreign_receipt_is_kept_after_own_terminal(project, field):
    p = project
    # Ordinary-width metadata, not an actual 1200-process tree.
    p.rp._tracked_descendants = [
        {"pid": 4_000_000 + index, "pgid": 4_000_000,
         "start_ticks": 1_234_567_890 + index}
        for index in range(1200)
    ]
    assert process.persist_role_process_receipt(p.rp) is True
    receipt = json.loads(p.receipt.read_bytes())
    if field == "nonce":
        receipt["nonce_sha256"] = "b" * 64
    elif field == "attempt_id":
        receipt["trigger_delivery"]["attempt_id"] = "different-attempt"
    else:
        assert receipt["trigger_delivery"]["generation"] == 1
        receipt["trigger_delivery"]["generation"] = True
    foreign = (json.dumps(receipt, sort_keys=True) + "\n").encode()
    assert 64 * 1024 < len(foreign) < process._MAX_ROLE_RECEIPT_BYTES
    p.receipt.write_bytes(foreign)
    result = roles.check_active_roles(p.active, ideas_file=str(p.root / "ideas.md"))
    assert _state(p) == "TERMINAL"
    assert result == [("worker", roles.OUTCOME_ERROR)]
    assert p.receipt.read_bytes() == foreign
    assert p.rp.lock_dir.is_dir()
