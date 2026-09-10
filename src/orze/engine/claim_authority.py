"""Short claim/reset ownership and read-only native-attempt admission.

New filesystem claims may bind the absolute main SQLite path supplied by the
framework's IdeaLake. This is routing metadata, not a cryptographic capability.
No-Lake repair uses that exact claim/native catalog route in mode=ro and never
creates/migrates a database. Only absence of every route retains legacy
behavior; an unavailable, conflicting or redirected route never downgrades.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import stat

from orze.core.execution_attempts import AttemptAuthorityError, current_attempt
from orze.engine.attempt_effect_lock import (
    AttemptEffectBusy, AttemptEffectInDoubt, attempt_effect_lock, require_effect_lease,
)

_MAX_CLAIM = 65536
_MAX_PHASES = 64


def safe_file(path, *, missing=True):
    path = Path(path).absolute()
    if ".." in path.parts:
        raise AttemptEffectInDoubt("claim_path_invalid")
    for parent in path.parents:
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise AttemptEffectInDoubt("claim_parent_invalid")
    try:
        info = path.lstat()
    except FileNotFoundError:
        if missing:
            return None
        raise AttemptEffectInDoubt("claim_required_file_missing")
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise AttemptEffectInDoubt("claim_file_redirected")
    return info


def read_claim(path):
    info = safe_file(path)
    if info is None:
        return None
    if not 0 < info.st_size <= _MAX_CLAIM:
        raise AttemptEffectInDoubt("claim_metadata_invalid")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size,
                             item.st_mtime_ns, item.st_ctime_ns, item.st_nlink)
    try:
        raw = os.read(fd, _MAX_CLAIM + 1)
        if (len(raw) != info.st_size or identity(os.fstat(fd)) != identity(info)
                or identity(Path(path).lstat()) != identity(info)):
            raise AttemptEffectInDoubt("claim_metadata_changed")
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("duplicate claim key")
                result[key] = value
            return result
        value = json.loads(raw, object_pairs_hook=pairs)
        if not isinstance(value, dict):
            raise ValueError("claim must be an object")
        return value
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise AttemptEffectInDoubt("claim_metadata_invalid") from exc
    finally:
        os.close(fd)


def _lake_path(lake):
    if lake is None:
        return None
    rows = lake.conn.execute("PRAGMA database_list").fetchall()
    paths = [row[2] for row in rows if row[1] == "main"]
    if len(paths) != 1 or not paths[0]:
        raise AttemptEffectBusy("claim_persistent_catalog_unavailable")
    return str(Path(paths[0]).absolute())


def _closed(conn, task_id):
    # The public reader verifies the exact main table schema, including views,
    # collation and bounded stored JSON. Absence remains an explicit legacy case.
    current_attempt(conn, task_id, "training")
    if conn.execute("SELECT 1 FROM main.sqlite_master WHERE name='execution_attempts'").fetchone() is None:
        return
    phases = conn.execute(
        "SELECT DISTINCT phase FROM main.execution_attempts "
        "WHERE task_id=? COLLATE BINARY LIMIT ?", (task_id, _MAX_PHASES + 1),
    ).fetchall()
    if len(phases) > _MAX_PHASES:
        raise AttemptEffectBusy("claim_attempt_phase_limit")
    for row in phases:
        attempt = current_attempt(conn, task_id, row[0])
        if attempt is not None and attempt["state"] not in ("TERMINAL", "NOT_STARTED"):
            raise AttemptEffectBusy("claim_execution_attempt_unclosed")


def require_closed_claim_attempts(idea_dir, *, lake=None, claim_data=None, effect_lease=None):
    """Return a validated main DB routing path; never create storage."""
    from orze.engine.execution_catalog import declared_catalog

    supplied = _lake_path(lake)
    bound = (claim_data or {}).get("lifecycle_db")
    if claim_data is not None and "lifecycle_db" in claim_data:
        if type(bound) is not str or not bound or not Path(bound).is_absolute():
            raise AttemptEffectBusy("claim_catalog_binding_invalid")
    declared = declared_catalog(idea_dir)
    routes = [value for value in (supplied, bound, declared) if value is not None]
    if routes and any(value != routes[0] for value in routes[1:]):
        raise AttemptEffectBusy("claim_catalog_scope_mismatch")
    db_path = routes[0] if routes else None
    if db_path is None:
        return None
    path = Path(db_path)
    before = safe_file(path, missing=False)
    if lake is not None and lake.conn.in_transaction:
        if effect_lease is None:
            raise AttemptEffectBusy("claim_explicit_transaction_lease_required")
        require_effect_lease(effect_lease, idea_dir)
        # The coordinator owns this exact lease and SQL transaction. It may
        # have just closed the old attempt, which another connection cannot
        # see until commit. Never manage its transaction or infer ownership.
        _closed(lake.conn, Path(idea_dir).name)
        after = safe_file(path, missing=False)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise AttemptEffectBusy("claim_catalog_replaced")
        return str(path)
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=1)
    try:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        _closed(conn, Path(idea_dir).name)
        after = safe_file(path, missing=False)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise AttemptEffectBusy("claim_catalog_replaced")
    finally:
        conn.close()
    return str(path)


@contextmanager
def claim_change_guard(idea_dir, *, lake=None, effect_lease=None):
    """Explicit lease re-entry only; all claim/evidence changes occur inside."""
    idea_dir = Path(idea_dir).absolute()
    with attempt_effect_lock(idea_dir, lease=effect_lease) as acquired:
        try:
            for name in ("metrics.json", "train_output.log"):
                safe_file(idea_dir / name)
            claim_data = read_claim(idea_dir / "claim.json")
            db_path = require_closed_claim_attempts(
                idea_dir, lake=lake, claim_data=claim_data,
                effect_lease=effect_lease)
        except (OSError, sqlite3.Error, ValueError, TypeError, UnicodeError,
                AttemptAuthorityError, AttemptEffectInDoubt) as exc:
            raise AttemptEffectBusy("claim_authority_unavailable") from exc
        yield acquired, claim_data, db_path
