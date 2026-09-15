"""Versioned, bounded research memory declarations, always pending verification.

A syntactically valid author/derived claim is not qualified evidence. A source
binding must be rechecked by its versioned resolver before any consumer uses
it. This module neither resolves sources nor publishes files or execution
rights. Every read captures fresh bytes from the fixed project memory path.

Schema 1: at most 32 records, 8 references per record, 64 references overall,
and 64 KiB UTF-8 for the complete document. Limits reject the complete input;
no record, counterexample or field is silently truncated or dropped.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat

MAX_BYTES = 64 * 1024
MAX_ENTRIES = 32
MAX_SOURCES = 64


class MemoryUnavailable(ValueError):
    pass


def project_scope(results_dir):
    path = Path(results_dir).absolute()
    if '..' in path.parts:
        raise MemoryUnavailable('memory_scope_invalid')
    return hashlib.sha256(str(path).encode('utf-8')).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise MemoryUnavailable('memory_duplicate_json_key')
        result[key] = value
    return result


def _constant(value):
    raise MemoryUnavailable('memory_nonfinite_json_number')


def _keys(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise MemoryUnavailable('memory_fields_invalid')


def _text(value, maximum):
    if type(value) is not str or not value.strip() or len(value.encode('utf-8')) > maximum:
        raise MemoryUnavailable('memory_text_invalid')
    if any(ord(c) < 32 and c not in '\n\t' for c in value):
        raise MemoryUnavailable('memory_text_invalid')


def decode_memory(raw, expected_scope):
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_BYTES:
        raise MemoryUnavailable('memory_size_invalid')
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs, parse_constant=_constant)
        _keys(value, ('schema', 'project_scope', 'entries'))
        if type(value['schema']) is not int or value['schema'] != 1:
            raise MemoryUnavailable('memory_schema_invalid')
        if type(value['project_scope']) is not str or value['project_scope'] != expected_scope:
            raise MemoryUnavailable('memory_scope_mismatch')
        entries = value['entries']
        if type(entries) is not list or len(entries) > MAX_ENTRIES:
            raise MemoryUnavailable('memory_entry_count_invalid')
        seen, total_sources = set(), 0
        for entry in entries:
            _keys(entry, ('id', 'question', 'hypothesis', 'rationale', 'claim', 'claimed_state', 'origin', 'sources'))
            key = entry['id']
            if type(key) is not str or not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', key) or key in seen:
                raise MemoryUnavailable('memory_record_id_invalid')
            seen.add(key)
            for name, maximum in [('question', 512), ('hypothesis', 512), ('rationale', 1024), ('claim', 1024)]:
                _text(entry[name], maximum)
            if type(entry['claimed_state']) is not str or entry['claimed_state'] not in ('confirmed', 'refuted', 'unknown'):
                raise MemoryUnavailable('memory_claimed_state_invalid')
            if type(entry['origin']) is not str or entry['origin'] not in ('authored', 'derived'):
                raise MemoryUnavailable('memory_origin_invalid')
            sources = entry['sources']
            if type(sources) is not list or len(sources) > 8:
                raise MemoryUnavailable('memory_source_count_invalid')
            if entry['claimed_state'] != 'unknown' and not sources:
                raise MemoryUnavailable('memory_claim_sources_missing')
            total_sources += len(sources)
            if total_sources > MAX_SOURCES:
                raise MemoryUnavailable('memory_source_count_invalid')
            references = set()
            for source in sources:
                _keys(source, ('kind', 'binding_schema', 'idea_id', 'binding_sha256'))
                if source['kind'] != 'report' or type(source['kind']) is not str:
                    raise MemoryUnavailable('memory_source_kind_invalid')
                if type(source['binding_schema']) is not int or source['binding_schema'] != 1:
                    raise MemoryUnavailable('memory_binding_schema_invalid')
                key, digest = source['idea_id'], source['binding_sha256']
                if type(key) is not str or not re.fullmatch(r'idea-[A-Za-z0-9][A-Za-z0-9_-]{0,121}', key):
                    raise MemoryUnavailable('memory_source_id_invalid')
                if type(digest) is not str or not re.fullmatch(r'[0-9a-f]{64}', digest):
                    raise MemoryUnavailable('memory_source_digest_invalid')
                if (key, digest) in references:
                    raise MemoryUnavailable('memory_source_duplicate')
                references.add((key, digest))
        return value
    except (UnicodeError, json.JSONDecodeError, RecursionError, OverflowError) as exc:
        raise MemoryUnavailable('memory_json_invalid') from exc


def _identity(info):
    return info.st_dev, info.st_ino, info.st_mode


def _file_stamp(info):
    return (*_identity(info), info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def read_memory(results_dir):
    """Return unverified author claims after a fresh bounded descriptor read."""
    descriptors = []
    try:
        root = Path(results_dir).absolute()
        expected_scope = project_scope(root)
        path = root / 'knowledge/research_memory.json'
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        current = os.open('/', flags | os.O_DIRECTORY)
        descriptors.append(current)
        parents = []
        for part in path.parts[1:-1]:
            child = os.open(part, flags | os.O_DIRECTORY, dir_fd=current)
            descriptors.append(child)
            parents.append((current, part, _identity(os.fstat(child))))
            current = child
        descriptor = os.open(path.name, flags, dir_fd=current)
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise MemoryUnavailable('memory_file_unsafe')
        if not 0 < before.st_size <= MAX_BYTES:
            raise MemoryUnavailable('memory_size_invalid')
        chunks, size = [], 0
        while size <= MAX_BYTES:
            chunk = os.read(descriptor, min(8192, MAX_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        raw = b''.join(chunks)
        if _file_stamp(os.fstat(descriptor)) != _file_stamp(before):
            raise MemoryUnavailable('memory_changed_during_read')
        if _file_stamp(os.stat(path.name, dir_fd=current, follow_symlinks=False)) != _file_stamp(before):
            raise MemoryUnavailable('memory_changed_during_read')
        for parent, name, identity in parents:
            if _identity(os.stat(name, dir_fd=parent, follow_symlinks=False)) != identity:
                raise MemoryUnavailable('memory_changed_during_read')
        value = decode_memory(raw, expected_scope)
        return {'availability': 'pending_verification', 'document_sha256': hashlib.sha256(raw).hexdigest(), 'document': value}
    except MemoryUnavailable as exc:
        return {'availability': 'unavailable', 'reason': str(exc)}
    except FileNotFoundError:
        return {'availability': 'unavailable', 'reason': 'memory_absent'}
    except (OSError, AttributeError, ValueError, UnicodeError):
        return {'availability': 'unavailable', 'reason': 'memory_read_unavailable'}
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
