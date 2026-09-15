"""Loss-resistant proposal source admission under the shared producer lock.

CALLING SPEC:
    ingest_ideas_source(engine, cfg) -> (raw_ideas, inserted_ids)
        Capture fresh primary source bytes under the producer lock, admit an
        immutable bounded batch, and acknowledge only exact committed blocks.
        Requires a lake; the caller keeps the legacy no-lake path. Optional
        enrichment/provider work belongs after this function releases the lock.

The SQLite commit precedes source acknowledgement. Failed source publication
can therefore be retried without replacing an admitted task. Files are not an
exactly-once bus: non-cooperating writers and unresolved source-lock owners are
not silently reconciled. Sidecars remain additive and are never consumed.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from itertools import islice
import logging
import os
from pathlib import Path
import re
import stat
import uuid

import yaml

from orze.core.ideas import (
    IDEA_ID_PATTERN, _iter_sidecar_ideas, _sidecar_sections, parse_ideas_text,
)
from orze.core.idea_source_lock import (
    idea_source_lock, idea_source_lock_owned,
)

logger = logging.getLogger("orze")
_MAX_SOURCE_BYTES = 4 * 1024 * 1024
_MAX_ADMISSIONS = 128
_HEADINGS = re.compile(r"^## [^\r\n]*", re.MULTILINE)
_IDEA = re.compile(rf"## ({IDEA_ID_PATTERN}):\s*(.+)")


def _read_source(path):
    """Single descriptor, bounded fresh UTF-8, no cache or redirected file."""
    current = Path(path.absolute().anchor)
    for part in path.absolute().parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError("idea_source_redirected")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_size > _MAX_SOURCE_BYTES):
            raise ValueError("idea_source_invalid_or_oversize")
        chunks, remaining = [], _MAX_SOURCE_BYTES + 1
        while remaining:
            chunk = os.read(fd, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
        value.st_ctime_ns, value.st_nlink,
    )
    if (len(data) > _MAX_SOURCE_BYTES or identity(before) != identity(after)
            or identity(after) != identity(path.lstat())):
        raise ValueError("idea_source_changed_during_read")
    return data.decode("utf-8"), identity(after), before.st_mode & 0o777


def _blocks(text):
    """Unknown/malformed section headings delimit retained bytes too."""
    blocks = []
    for heading, end in _sidecar_sections(_HEADINGS.finditer(text), len(text)):
        match = _IDEA.fullmatch(heading.group())
        blocks.append((match.group(1) if match else None, heading.start(), end))
    return blocks


def _read_sidecar(path):
    """Same fresh/regular/size boundary as primary; never consume a sidecar."""
    try:
        return _read_source(path)[0]
    except (OSError, ValueError, UnicodeError) as exc:
        logger.warning("Sidecar %s not admitted: %s", path.name, type(exc).__name__)
        return ""


def _batch(path, text, candidates, occurrences, offset, *, prefix=None):
    """Bound decoded records and raw payload; offset is only an inspection hint.

    Primary pages need no sidecar scan. Continuation may skip an identity-pinned
    prefix; selected sidecars are freshly read. No cached YAML or admission.
    """
    batch = candidates[offset:offset + _MAX_ADMISSIONS]
    sidecars = {}
    if len(batch) == _MAX_ADMISSIONS:
        # A full primary page leaves possible sidecars for the next tick. An
        # empty extra tick can reset the hint; enumeration is not convergence.
        return batch, sidecars, offset + len(batch)
    size = sum(len(text[start:end].encode("utf-8")) for _, start, end in batch)
    skip = max(0, offset - len(candidates))
    stream = (_iter_sidecar_ideas(str(path), occurrences, read_text=_read_sidecar)
              if prefix is None else prefix.stream(path, occurrences, skip, read_text=_read_sidecar))
    more = False
    try:
        for idea_id, idea in islice(stream, skip, None):
            amount = len(idea["raw"].encode("utf-8"))
            if len(batch) == _MAX_ADMISSIONS or size + amount > _MAX_SOURCE_BYTES:
                more = True
                break
            batch.append((idea_id, None, None))
            sidecars[idea_id] = idea
            size += amount
        if prefix is not None:
            prefix.verify()
    except BaseException:
        if prefix is not None:
            prefix.clear()
        raise
    finally:
        stream.close()
    return batch, sidecars, offset + len(batch) if more else 0


def _publish_ack(path, original, original_identity, mode, updated, lease):
    """Strict publication: no swallowed ENOSPC or assumed rename success."""
    temporary = path.with_name(f".{path.name}.ingest-{uuid.uuid4().hex}.tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        try:
            data, offset = updated.encode("utf-8"), 0
            while offset < len(data):
                written = os.write(fd, data[offset:])
                if written <= 0:
                    raise OSError("idea_source_short_write")
                offset += written
            os.fsync(fd)
        finally:
            os.close(fd)
        current, identity, _ = _read_source(path)
        if (current != original or identity != original_identity
                or not idea_source_lock_owned(lease)):
            raise ValueError("idea_source_or_owner_changed_before_ack")
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if _read_source(path)[0] != updated:
            raise ValueError("idea_source_ack_readback_failed")
    finally:
        temporary.unlink(missing_ok=True)


def _proposal_fields(idea):
    raw = idea.get("raw", "")

    def field(name):
        match = re.search(rf"\*\*{re.escape(name)}\*\*:\s*(.+)", raw)
        return match.group(1).strip() if match else None

    priority = idea.get("priority", "medium")
    declared = field("Kind")
    configured = (idea.get("config") or {}).get("kind")
    if declared is not None and configured is not None and declared != configured:
        raise ValueError("proposal_kind_conflict")
    kind = configured if configured is not None else declared if declared is not None else "train"
    if priority == "critical" and idea.get("_overlay_source") != "sidecar":
        priority = "high"
    return {
        "status": "queued", "priority": priority,
        "category": field("Category"), "parent": field("Parent"),
        "hypothesis": field("Hypothesis"),
        "approach_family": idea.get("approach_family", field("Approach Family") or "other"),
        "kind": kind,
    }


def ingest_ideas_source(engine, cfg):
    path = Path(cfg["ideas_file"])
    raw_ideas, inserted = {}, []
    try:
        with idea_source_lock(engine.results_dir / ".ideas_md.lock") as lease:
            if lease is None:
                return {}, []
            text, source_identity, mode = _read_source(path)
            blocks = _blocks(text)
            occurrences = Counter(block[0] for block in blocks if block[0])
            candidates = [block for block in blocks if block[0]
                          and occurrences[block[0]] == 1]
            # This cursor only rotates a bounded inspection batch; it grants no
            # admission/ACK authority. Primary changes reset it, while retained
            # conflicts cannot permanently starve later proposals in a live run.
            scope = (str(path.absolute()), hashlib.sha256(text.encode("utf-8")).hexdigest())
            cursor = getattr(engine, "_idea_ingress_cursor", None)
            offset = cursor[2] if cursor and cursor[:2] == scope else 0
            from orze.engine.sidecar_prefix import SidecarPrefix
            prefix = getattr(engine, "_idea_sidecar_prefix", None)
            if prefix is None or not cursor or cursor[:2] != scope or not offset:
                prefix = engine._idea_sidecar_prefix = SidecarPrefix()
            batch, sidecars, next_offset = _batch(path, text, candidates, occurrences, offset, prefix=prefix)
            engine._idea_ingress_cursor = (*scope, next_offset)
            for idea_id, start, end in batch:
                if start is None:
                    raw_ideas[idea_id] = sidecars[idea_id]
                    continue
                try:
                    raw_ideas.update(parse_ideas_text(text[start:end]))
                except (ValueError, TypeError, AttributeError, RecursionError, yaml.YAMLError) as exc:
                    logger.warning("Retaining unparsed proposal %s: %s", idea_id, type(exc).__name__)
            if not raw_ideas:
                return raw_ideas, inserted
            find_ids = getattr(engine.lake, "find_existing_ids", None)
            if callable(find_ids):
                db_ids = find_ids(raw_ideas)
            else:
                # Lightweight lake adapters retain the existing public get()
                # contract; never fall back to enumerating all historical IDs.
                db_ids = {key for key in raw_ideas if engine.lake.get(key) is not None}
            pending = {key: engine._config_override_hash(value.get("config", {}))
                       for key, value in raw_ideas.items() if key not in db_ids}
            try:
                # Prepare legacy hashes without constructing an unused owner
                # map. Lightweight adapters may expose only the old lookup API.
                # Normal insert still checks current status/kind/YAML in its
                # own writer transaction; neither API grants duplicate or ACK.
                prepare = getattr(engine.lake, "prepare_admitted_config_hashes", None)
                if not callable(prepare):
                    prepare = engine.lake.find_admitted_config_hashes
                prepare(set(pending.values()))
            except Exception as exc:
                logger.warning("Proposal dedup index unavailable: %s", type(exc).__name__)
            acknowledged = set()
            for idea_id, idea in list(raw_ideas.items())[:_MAX_ADMISSIONS]:
                if not idea_source_lock_owned(lease):
                    break
                # Both same-ID identity and cross-ID config duplicates reach
                # the atomic boundary. An old JSON cache cannot suppress a
                # legal retry or select a stale/different-config "winner".
                try:
                    outcome = engine.lake.insert(
                        idea_id, idea["title"], yaml.dump(idea.get("config", {})),
                        idea.get("raw", ""), if_absent=True, **_proposal_fields(idea),
                    )
                except Exception as exc:
                    logger.warning("Proposal %s not acknowledged: %s", idea_id, type(exc).__name__)
                    continue
                if not isinstance(outcome, dict):
                    continue
                result = outcome.get("status")
                if result == "inserted":
                    inserted.append(idea_id)
                if result in ("inserted", "already_present_exact"):
                    acknowledged.add(idea_id)
                else:
                    logger.warning("Retaining proposal %s: %s", idea_id, result)
            spans = [(start, end) for idea_id, start, end in batch
                     if start is not None and idea_id in acknowledged and
                     raw_ideas.get(idea_id, {}).get("_overlay_source") != "sidecar"]
            if spans:
                parts, previous = [], 0
                for start, end in spans:
                    parts.append(text[previous:start])
                    previous = end
                parts.append(text[previous:])
                updated = "".join(parts)
                _publish_ack(path, text, source_identity, mode, updated, lease)
                from orze.engine.roles import mark_ingest
                mark_ingest(path)
                for role in engine.active_roles.values():
                    role.ideas_pre_size = len(updated.encode("utf-8"))
                    role.ideas_pre_count = len(re.findall(r"^## idea-", updated, re.MULTILINE))
                    if getattr(role, "writes_ideas_file", True) and inserted:
                        role.ideas_consumed_during_run += len(inserted)
            if inserted:
                logger.info("Admitted %d new proposals from %s", len(inserted), path)
    except FileNotFoundError:
        pass  # Empty historical source remains compatible; never manufacture it.
    except Exception as exc:
        logger.warning("Proposal source not acknowledged: %s", type(exc).__name__)
    return raw_ideas, inserted
