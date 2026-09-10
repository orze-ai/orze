"""Legacy recovery publication gate; never adopts a native execution.

CALLING SPEC: process observation/termination happens outside publication.
Each bounded file/DB callback reacquires the same task guard used by native
admission and rechecks catalog/history before effects. Native restart adoption
is deliberately unavailable here, even when metrics or dead PIDs look useful.
"""
from pathlib import Path
import logging
import sqlite3

from orze.core.execution_attempts import AttemptAuthorityError, current_attempt
from orze.engine.attempt_effect_lock import (
    AttemptEffectInDoubt, attempt_effect_lock, require_effect_lease,
)
from orze.engine.termination_hold import TerminationUnconfirmed

logger = logging.getLogger("orze")


def legacy_recovery_allowed(lake, idea_dir, *, lease=None):
    from orze.engine.execution_catalog import declared_catalog
    from orze.engine.attempt_effect_receipts import require_closed_effects
    idea_dir = Path(idea_dir)
    try:
        if declared_catalog(idea_dir) is not None:
            return False
        if lease is None:
            owner = idea_dir / "_attempt_effect.lock"
            if owner.exists() or owner.is_symlink():
                return False
        else:
            require_effect_lease(lease, idea_dir)
        require_closed_effects(idea_dir)
        # Exercise the strict read-only schema check, then reject any native
        # phase history for this task, not only currently running training.
        current_attempt(lake.conn, idea_dir.name, "training")
        table = lake.conn.execute(
            "SELECT 1 FROM main.sqlite_master WHERE name='execution_attempts'").fetchone()
        if table and lake.conn.execute(
                "SELECT 1 FROM main.execution_attempts WHERE task_id=? COLLATE BINARY LIMIT 1",
                (idea_dir.name,)).fetchone():
            return False
        return True
    except (TerminationUnconfirmed, AttemptAuthorityError, OSError, sqlite3.Error):
        return False


def publish_legacy_recovery(lake, idea_dir, callback):
    """Run a short legacy-only callback; reject partial/unknown publication."""
    if not legacy_recovery_allowed(lake, idea_dir):
        return False
    try:
        with attempt_effect_lock(Path(idea_dir)) as lease:
            if not legacy_recovery_allowed(lake, idea_dir, lease=lease):
                return False
            try:
                result = callback()
                require_effect_lease(lease, idea_dir)
                return result
            except BaseException as exc:
                raise AttemptEffectInDoubt("legacy_recovery_publication_unconfirmed") from exc
    except TerminationUnconfirmed:
        logger.error("Legacy recovery held for %s; native adoption is unavailable", Path(idea_dir).name)
        return False
