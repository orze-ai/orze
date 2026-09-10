"""Durable routing declaration: a native task may not downgrade to no-Lake IO.

This is not a capability or an attempt receipt. It records which existing
main database the framework supplied; native entry points still verify current
attempt authority. Publication happens under the task's explicit effect lease.
"""
import json
import os
from pathlib import Path
import stat

from orze.core.fs import atomic_create
from orze.engine.attempt_effect_lock import AttemptEffectBusy, AttemptEffectInDoubt, require_effect_lease

CATALOG_FILE = "_execution_catalog.json"


def declared_catalog(idea_dir):
    path = Path(idea_dir) / CATALOG_FILE
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or not 0 < info.st_size <= 8192:
        raise AttemptEffectBusy("execution_catalog_declaration_invalid")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    identity = lambda st: (st.st_dev, st.st_ino, st.st_mode, st.st_nlink,
                           st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    try:
        raw = os.read(fd, 8193)
        if (len(raw) != info.st_size or identity(os.fstat(fd)) != identity(info)
                or identity(path.lstat()) != identity(info)):
            raise AttemptEffectBusy("execution_catalog_declaration_changed")
        value = json.loads(raw)
        if (type(value) is not dict or set(value) != {"schema", "task_id", "database"}
                or type(value["schema"]) is not int or value["schema"] != 1
                or value["task_id"] != Path(idea_dir).name
                or type(value["database"]) is not str or not Path(value["database"]).is_absolute()
                or ".." in Path(value["database"]).parts
                or raw != (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()):
            raise AttemptEffectBusy("execution_catalog_declaration_invalid")
        return value["database"]
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise AttemptEffectBusy("execution_catalog_declaration_invalid") from exc
    finally:
        os.close(fd)


def bind_catalog(lake, idea_dir, lease):
    require_effect_lease(lease, idea_dir)
    paths = [row[2] for row in lake.conn.execute("PRAGMA database_list") if row[1] == "main"]
    if len(paths) != 1 or not paths[0]:
        raise AttemptEffectBusy("execution_persistent_catalog_required")
    database = str(Path(paths[0]).absolute())
    existing = declared_catalog(idea_dir)
    if existing is not None:
        if existing != database:
            raise AttemptEffectBusy("execution_catalog_scope_mismatch")
        return
    payload = {"schema": 1, "task_id": Path(idea_dir).name, "database": database}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    if len(encoded.encode()) > 8192:
        raise AttemptEffectBusy("execution_catalog_declaration_oversized")
    try:
        atomic_create(Path(idea_dir) / CATALOG_FILE, encoded)
        if declared_catalog(idea_dir) != database:
            raise AttemptEffectInDoubt("execution_catalog_declaration_not_published")
        fd = os.open(idea_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except BaseException as exc:
        raise AttemptEffectInDoubt("execution_catalog_publication_unconfirmed") from exc
