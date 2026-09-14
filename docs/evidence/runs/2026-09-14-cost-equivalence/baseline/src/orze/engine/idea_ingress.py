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
import logging
import os
from pathlib import Path
import re
import stat
import uuid

import yaml

from orze.core.ideas import (
    IDEA_ID_PATTERN, _overlay_sidecar_ideas, parse_ideas_text,
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
    headings = list(_HEADINGS.finditer(text))
    blocks = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        match = _IDEA.fullmatch(heading.group())
        blocks.append((match.group(1) if match else None, heading.start(), end))
    return blocks


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
            sidecars = {key: value for key, value in
                        _overlay_sidecar_ideas(str(path), {}).items()
                        if key not in occurrences}
            candidates.extend((key, None, None) for key in sidecars)
            # This cursor only rotates a bounded inspection batch; it grants no
            # admission/ACK authority. Source changes reset it, while retained
            # conflicts cannot permanently starve later proposals in a live run.
            scope = (str(path.absolute()), hashlib.sha256(text.encode("utf-8")).hexdigest())
            cursor = getattr(engine, "_idea_ingress_cursor", None)
            offset = cursor[2] if cursor and cursor[:2] == scope else 0
            batch = candidates[offset:offset + _MAX_ADMISSIONS]
            engine._idea_ingress_cursor = (*scope, (
                offset + _MAX_ADMISSIONS if offset + _MAX_ADMISSIONS < len(candidates) else 0))
            for idea_id, start, end in batch:
                if start is None:
                    raw_ideas[idea_id] = sidecars[idea_id]
                    continue
                try:
                    raw_ideas.update(parse_ideas_text(text[start:end]))
                except (ValueError, TypeError, AttributeError, RecursionError, yaml.YAMLError) as exc:
                    logger.warning("Retaining unparsed proposal %s: %s", idea_id, type(exc).__name__)
            db_ids = engine.lake.get_all_ids()
            config_hashes = engine._load_config_hashes()
            pending = {key: engine._config_override_hash(value.get("config", {}))
                       for key, value in raw_ideas.items() if key not in db_ids}
            try:
                known = engine.lake.find_admitted_config_hashes(set(pending.values()))
                for fingerprint, idea_id in known.items():
                    config_hashes.setdefault(fingerprint, idea_id)
            except Exception as exc:
                logger.warning("Proposal dedup index unavailable: %s", type(exc).__name__)
            acknowledged = set()
            for idea_id, idea in list(raw_ideas.items())[:_MAX_ADMISSIONS]:
                if not idea_source_lock_owned(lease):
                    break
                # A known same-ID task always reaches the atomic identity
                # boundary. Legacy cross-ID config dedup is retained, but is
                # not a durable rejection authorizing deletion of its source.
                fingerprint = pending.get(idea_id)
                if (fingerprint and config_hashes.get(fingerprint)
                        and config_hashes[fingerprint] != idea_id):
                    # CPU and legacy executable kinds have different admission
                    # domains even when their raw config happens to hash alike.
                    winner = engine.lake.get(config_hashes[fingerprint])
                    try:
                        cpu = _proposal_fields(idea)["kind"] == "native_cpu_action"
                    except ValueError:
                        continue
                    if winner and (winner.get("kind") == "native_cpu_action") == cpu:
                        continue
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
                    if fingerprint:
                        config_hashes[fingerprint] = idea_id
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
