"""Source preservation through the real Orze._sync_ideas public phase.

The seven existing-behavior cases exercise the old phase as well as the new
source protocol. Names containing new_mechanism are separate acceptance for
the newly declared atomic publisher and finite source/block budgets; they are
excluded from old behavioral-baseline counts. No provider/GPU is executed.
"""
import errno
import os
from pathlib import Path

import pytest
import yaml

from orze.core.fs import locked_append
from orze.core.ideas import parse_ideas
from orze.engine import phases
from test_idea_ingress_contract import _block, engine  # Shared frozen fixture, unchanged.


def test_same_mtime_replacement_does_not_admit_the_previous_parse_cache(engine):
    instance, cfg, source = engine
    original = "# Ideas\n\n" + _block("idea-cache", 1)
    replacement = "# Ideas\n\n" + _block("idea-cache", 2)
    assert len(original.encode()) == len(replacement.encode())
    source.write_text(original, encoding="utf-8")
    metadata = source.stat()
    assert parse_ideas(str(source))["idea-cache"]["config"] == {"seed": 1}
    candidate = source.with_name("new-source.md")
    candidate.write_text(replacement, encoding="utf-8")
    os.utime(candidate, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    os.replace(candidate, source)
    assert source.stat().st_mtime_ns == metadata.st_mtime_ns
    assert source.stat().st_ino != metadata.st_ino

    instance._sync_ideas(cfg)

    admitted = instance.lake.get("idea-cache")
    assert admitted is not None
    assert yaml.safe_load(admitted["config"]) == {"seed": 2}


def test_ambiguous_duplicate_id_blocks_are_preserved_in_full(engine):
    instance, cfg, source = engine
    original = "# Ideas\n\n" + _block("idea-duplicate", 11) + _block("idea-duplicate", 22)
    source.write_bytes(original.encode("utf-8"))

    instance._sync_ideas(cfg)

    # A dict collapse is not permission to acknowledge either original block.
    assert source.read_bytes() == original.encode("utf-8")


def test_observed_source_replacement_during_admission_is_not_cleared(engine, monkeypatch):
    instance, cfg, source = engine
    source.write_text("# Ideas\n\n" + _block("idea-first", 1), encoding="utf-8")
    replacement = "# Replaced by another writer\n\n" + _block("idea-replacement", 2)
    real_insert = instance.lake.insert
    interleavings = []

    def insert_then_replace(*args, **kwargs):
        result = real_insert(*args, **kwargs)
        if not interleavings:
            candidate = source.with_name("external-replacement.md")
            candidate.write_bytes(replacement.encode("utf-8"))
            os.replace(candidate, source)
            interleavings.append(source.stat().st_ino)
        return result

    monkeypatch.setattr(instance.lake, "insert", insert_then_replace)

    instance._sync_ideas(cfg)

    assert len(interleavings) == 1, "the real admission interleaving must run"
    assert source.read_bytes() == replacement.encode("utf-8")
    assert instance.lake.get("idea-first") is not None


def test_enospc_at_source_publication_preserves_the_entire_original_source(engine, monkeypatch):
    instance, cfg, source = engine
    original = "# Ideas\n\n" + _block("idea-space", 3)
    source.write_bytes(original.encode("utf-8"))
    real_open, real_replace = Path.open, os.replace
    faults = []

    class FullDeviceWriter:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def write(self, value):
            # Real old open('w') has already truncated the original. Inject
            # only the ensuing device-write failure, not the whole consumer.
            faults.append("direct_write_after_real_truncation")
            raise OSError(errno.ENOSPC, "test-only full source filesystem")

    def full_source_open(path, *args, **kwargs):
        handle = real_open(path, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == source and "w" in mode and not faults:
            return FullDeviceWriter(handle)
        return handle

    def full_source_replace(old, new, *args, **kwargs):
        if Path(new) == source and not faults:
            faults.append("atomic_replace_rejected")
            raise OSError(errno.ENOSPC, "test-only full source filesystem")
        return real_replace(old, new, *args, **kwargs)

    # Covers the real old destructive write boundary and the new atomic
    # publication boundary without prescribing the new temporary-file API.
    monkeypatch.setattr(Path, "open", full_source_open)
    monkeypatch.setattr(os, "replace", full_source_replace)

    instance._sync_ideas(cfg)

    assert len(faults) == 1, "a source-publication I/O boundary must fail"
    assert instance.lake.get("idea-space") is not None
    assert source.read_bytes() == original.encode("utf-8")


def test_header_and_unadmitted_invalid_content_keep_their_exact_original_bytes(engine):
    instance, cfg, source = engine
    header = b" \t# Human-owned Ideas\r\n\r\n<!-- retain whitespace -->\r\n"
    accepted = _block("idea-valid", 4).encode("utf-8")
    invalid = (b"## idea-invalid: Needs correction\r\n```yaml\r\nseed: [\r\n```\r\n"
               b"\r\n# Human follow-up\r\n  retain these spaces  \r\n")
    source.write_bytes(header + accepted + invalid)

    instance._sync_ideas(cfg)

    assert instance.lake.get_all_ids() == {"idea-valid"}
    assert source.read_bytes() == header + invalid


@pytest.mark.parametrize("optional", ["elo", "reflection"])
def test_optional_provider_hook_runs_after_source_unlock_and_cannot_lose_its_append(
    engine, monkeypatch, optional,
):
    instance, cfg, source = engine
    source.write_text("# Ideas\n\n" + _block("idea-before-hook", 5), encoding="utf-8")
    flag = "elo_ranking_enabled" if optional == "elo" else "reflection_enabled"
    function = ("_run_elo_tournament_for_ingested" if optional == "elo"
                else "_run_reflection_for_ingested")
    cfg["substrate"] = {flag: True}
    lock = instance.results_dir / ".ideas_md.lock"
    appended = _block("idea-from-hook", 6)
    observations = []

    def provider_boundary(lake, ingested_ids, substrate_cfg, *args):
        acquired = not lock.exists()
        observation = {"acquired": acquired, "ids": list(ingested_ids), "appended": False}
        observations.append(observation)
        if acquired:
            observation["appended"] = locked_append(source, appended, lock)

    # This is the optional provider-call boundary, not the source lock or
    # producer. Assertions are outside the best-effort hook's broad except.
    monkeypatch.setattr(phases, function, provider_boundary)

    instance._sync_ideas(cfg)

    assert observations == [{"acquired": True, "ids": ["idea-before-hook"], "appended": True}]
    assert appended.encode("utf-8") in source.read_bytes(), (
        "source ACK must finish before external/provider hooks can append new work")


def test_new_mechanism_atomic_replace_failure_leaves_original_source_intact(engine, monkeypatch):
    instance, cfg, source = engine
    original = ("# Ideas\n\n" + _block("idea-replace-failure", 7)).encode("utf-8")
    source.write_bytes(original)
    real_replace = os.replace
    faults = []

    def refuse_replace(old, new, *args, **kwargs):
        if Path(new) == source:
            faults.append((Path(old), Path(new)))
            raise OSError(errno.EACCES, "test-only atomic publication denied")
        return real_replace(old, new, *args, **kwargs)

    monkeypatch.setattr(os, "replace", refuse_replace)

    instance._sync_ideas(cfg)

    assert len(faults) == 1
    assert instance.lake.get("idea-replace-failure") is not None
    assert source.read_bytes() == original


def test_new_mechanism_source_over_four_mib_is_preserved_without_partial_admission(engine):
    instance, cfg, source = engine
    original = (_block("idea-oversized-source", 8).encode("utf-8")
                + b"x" * (4 * 1024 * 1024))
    source.write_bytes(original)
    assert len(original) > 4 * 1024 * 1024

    instance._sync_ideas(cfg)

    assert instance.lake.get("idea-oversized-source") is None
    assert source.read_bytes() == original


def test_new_mechanism_block_budget_defers_remainder_without_erasing_it(engine):
    instance, cfg, source = engine
    blocks = [_block(f"idea-bounded-{number:03d}", number) for number in range(129)]
    header = "# Ideas\n\n"
    source.write_text(header + "".join(blocks), encoding="utf-8")

    instance._sync_ideas(cfg)

    assert len(instance.lake.get_all_ids()) == 128
    assert instance.lake.get("idea-bounded-128") is None
    assert source.read_bytes() == (header + blocks[128]).encode("utf-8")

    instance._sync_ideas(cfg)

    assert len(instance.lake.get_all_ids()) == 129
    assert source.read_bytes() == header.encode("utf-8")
