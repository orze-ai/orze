"""Short per-task ownership guards for controller effect publication.

Uses the already audited nonce-bound, no-age-takeover directory protocol in
an independent namespace. This lock is not a process lease: never hold it
while waiting for a child, provider, or large artifact hash. Re-entry is only
by explicitly passing the exact lease, not by thread-local implicit ownership.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from orze.core.idea_source_lock import (
    SourceLockInDoubt, SourceLockLease, idea_source_lock, idea_source_lock_owned,
)
from orze.engine.termination_hold import (
    TerminationUnconfirmed, require_no_unconfirmed_stop,
)


class AttemptEffectBusy(TerminationUnconfirmed):
    """No permission to publish, retry, rotate a claim, or free ownership."""


class AttemptEffectInDoubt(TerminationUnconfirmed):
    """A partly published effect must retain its guard across restart."""


@dataclass(frozen=True)
class AttemptEffectLease:
    idea_dir: Path
    owner: SourceLockLease


def require_effect_lease(lease: AttemptEffectLease, idea_dir: Path) -> None:
    if (not isinstance(lease, AttemptEffectLease)
            or lease.idea_dir != Path(idea_dir).absolute()
            or not idea_source_lock_owned(lease.owner)):
        raise AttemptEffectInDoubt("attempt_effect_ownership_lost")
    require_no_unconfirmed_stop(lease.idea_dir)


@contextmanager
def attempt_effect_lock(idea_dir: Path, *, lease: AttemptEffectLease | None = None):
    """Yield exclusive authority or fail closed, without waiting or takeover.

    Raising AttemptEffectInDoubt after a partial file/DB effect retains the
    owner directory. Ordinary errors before effects release it. The coordinator
    must persist an effect intent before mutations; this primitive itself does
    not imply filesystem/SQLite atomicity or current-attempt validation.
    """
    idea_dir = Path(idea_dir).absolute()
    if lease is not None:
        require_effect_lease(lease, idea_dir)
        yield lease
        require_effect_lease(lease, idea_dir)
        return
    require_no_unconfirmed_stop(idea_dir)
    try:
        with idea_source_lock(idea_dir / "_attempt_effect.lock") as owner:
            if owner is None:
                raise AttemptEffectBusy("attempt_effect_lock_unavailable")
            acquired = AttemptEffectLease(idea_dir, owner)
            try:
                require_effect_lease(acquired, idea_dir)
                yield acquired
                require_effect_lease(acquired, idea_dir)
            except AttemptEffectInDoubt as exc:
                raise SourceLockInDoubt("attempt_effect_publication_unconfirmed") from exc
    except SourceLockInDoubt as exc:
        raise AttemptEffectInDoubt("attempt_effect_publication_unconfirmed") from exc
