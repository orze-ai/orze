"""Bounded, process-local hints for already inspected sidecar prefixes.

Only file identities, first-valid IDs and one namespace-bound name window
survive a controller tick. No YAML, raw source, admission result or execution
right is cached. A hinted prefix is
used only to skip records before the current inspection offset, after checking
every preceding file and the directory namespace. Selected files are read and
parsed again. A new traversal, changed primary source, PID, directory or file
discards the hints. Oversized prefixes continue via fresh parsing.
"""
from __future__ import annotations

import heapq
import os
from pathlib import Path
import stat

from orze.core.ideas import _iter_sidecar_ideas, _iter_sidecar_text

MAX_FILES = 8192
MAX_IDS = 32768
MAX_ID_BYTES = 1024 * 1024
NAME_WINDOW = 512


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _file_stamp(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("sidecar_prefix_file_unconfirmed")
    return _identity(info)


def _directory_stamp(path):
    ancestors = []
    for parent in path.parents:
        info = parent.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("sidecar_prefix_parent_redirected")
        ancestors.append((info.st_dev, info.st_ino))
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("sidecar_prefix_directory_unconfirmed")
    return _identity(info), tuple(ancestors)


def _names(directory, *, after=""):
    """Bound directory-name memory; each new window enumerates the directory.

    This is not a filesystem snapshot or a constant-I/O scan. Namespace
    identity is checked by the caller before publishing a batch.
    """
    while True:
        with os.scandir(directory) as entries:
            names = heapq.nsmallest(NAME_WINDOW, (entry.name for entry in entries
                                  if entry.name > after and entry.name.endswith(".md")))
        if not names:
            return
        yield from names
        after = names[-1]
        if len(names) < NAME_WINDOW:
            return


class SidecarPrefix:
    def __init__(self):
        self.scope = self.directory = self.revision = None
        self.files = []
        self.id_count = self.id_bytes = 0
        self.extend = True
        self.name_window = ()
        self.name_window_after = self.name_window_capacity = None

    def _names(self, directory, *, after=""):
        """Reuse one ordered name window under this stream's namespace checks.

        Retain at most NAME_WINDOW candidate names, one starting key and the
        window capacity. A backward seek or changed capacity needs a new scan.
        Payloads and preceding file identities are still verified separately.
        """
        while True:
            if (self.name_window_capacity != NAME_WINDOW
                    or self.name_window_after is None or after < self.name_window_after
                    or (len(self.name_window) == NAME_WINDOW
                        and after >= self.name_window[-1])):
                with os.scandir(directory) as entries:
                    names = heapq.nsmallest(NAME_WINDOW, (entry.name for entry in entries
                        if entry.name > after and entry.name.endswith(".md")))
                self.name_window = tuple(names)
                self.name_window_after = after
                self.name_window_capacity = NAME_WINDOW
            for name in self.name_window:
                if name > after:
                    yield name
            if len(self.name_window) < NAME_WINDOW:
                return
            after = self.name_window[-1]

    def clear(self):
        self.__init__()

    def verify(self):
        if self.revision is None:
            return
        try:
            if (os.getpid() != self.scope[1]
                    or _directory_stamp(self.directory) != self.revision
                    or any(_file_stamp(self.directory / name) != identity
                           for name, identity, _ in self.files)
                    or _directory_stamp(self.directory) != self.revision):
                raise ValueError("sidecar_prefix_changed")
        except (OSError, ValueError):
            self.clear()
            raise

    def stream(self, path, excluded, skip, *, read_text):
        directory = Path(path).absolute().parent / "ideas.d"
        scope = (str(Path(path).absolute()), os.getpid(), frozenset(excluded))
        # The engine also resets this object when primary source bytes change.
        if self.scope != scope or not skip:
            self.clear()
        try:
            self.verify()
        except (OSError, ValueError):
            pass  # Stale hints grant nothing; fresh prefix parsing below.
        if self.revision is None:
            try:
                revision = _directory_stamp(directory)
            except (OSError, ValueError):
                yield from _iter_sidecar_ideas(str(path), excluded, read_text=read_text)
                return
            self.scope, self.directory, self.revision = scope, directory, revision
        seen = set(excluded)
        remaining = skip
        try:
            # Same namespace and every cached file were just verified. Their
            # names need not be selected again from each directory window.
            after = ""
            for name, _, ids in self.files:
                if len(ids) > remaining:
                    raise ValueError("sidecar_prefix_offset_unconfirmed")
                seen.update(ids)
                remaining -= len(ids)
                after = name
                for idea_id in ids:
                    yield idea_id, None  # Only consumed by islice's skip.
            for name in self._names(directory, after=after):
                path = directory / name
                try:
                    before = _file_stamp(path)
                except (OSError, ValueError):
                    before = None
                text = read_text(path)
                ids = []
                for idea_id, idea in _iter_sidecar_text(text, seen):
                    ids.append(idea_id)
                    remaining = max(0, remaining - 1)
                    yield idea_id, idea
                # A partially yielded file never reaches this publication.
                # Empty/error reads do not authorize a cached empty prefix.
                amount = sum(len(key.encode()) for key in ids)
                if (not self.extend or not text or before is None
                        or len(self.files) >= MAX_FILES
                        or self.id_count + len(ids) > MAX_IDS
                        or self.id_bytes + amount > MAX_ID_BYTES):
                    self.extend = False
                    continue
                if _file_stamp(path) != before:
                    raise ValueError("sidecar_prefix_file_changed")
                self.files.append((name, before, tuple(ids)))
                self.id_count += len(ids)
                self.id_bytes += amount
        except Exception:
            self.clear()
            raise
