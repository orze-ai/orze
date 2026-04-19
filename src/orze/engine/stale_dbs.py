"""Startup guard: relocate zero-byte DB files that collide with canonical
orze DB names.

Why this matters
----------------
``sqlite3.connect('./queue.db')`` silently creates the file if it
doesn't exist. A zero-byte ``queue.db`` at the project root will
"succeed" at opening but return zero rows, hiding the fact that the
real DB lives at ``results/idea_lake.db``. See RCA #4 in the wired-up
findings report.

Contract
--------
    relocate_zero_byte_dbs(cwd, stale_dir, names=...) -> list[(src, dest)]
        Move each zero-byte file in ``cwd`` whose name is in ``names``
        to ``stale_dir/<name>.<timestamp>``. Returns the list of
        (src, dest) tuples for the moves actually performed.
        Idempotent: safe to call on every startup.
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Iterable, List, Tuple

logger = logging.getLogger("orze")

CANONICAL_DB_NAMES = (
    "idea_lake.db",
    "queue.db",
    "orze.db",
    "lake.db",
    "orze_queue.db",
    "ideas.db",
)


def relocate_zero_byte_dbs(cwd: Path,
                           stale_dir: Path,
                           names: Iterable[str] = CANONICAL_DB_NAMES
                           ) -> List[Tuple[Path, Path]]:
    moved: List[Tuple[Path, Path]] = []
    for name in names:
        p = Path(cwd) / name
        if not p.exists() or p.is_dir():
            continue
        try:
            if p.stat().st_size != 0:
                continue
        except OSError:
            continue
        try:
            stale_dir.mkdir(parents=True, exist_ok=True)
            dest = stale_dir / f"{name}.{int(time.time())}"
            shutil.move(str(p), str(dest))
            moved.append((p, dest))
        except OSError as e:
            logger.warning("Could not move stale %s: %s", p, e)
    return moved
