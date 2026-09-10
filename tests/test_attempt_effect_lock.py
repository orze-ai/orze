"""New write-guard mechanism; not old API-absence regressions."""
from pathlib import Path

import pytest

from orze.engine.attempt_effect_lock import (
    AttemptEffectBusy, AttemptEffectInDoubt, attempt_effect_lock, require_effect_lease,
)


def test_only_explicit_exact_lease_may_reenter(tmp_path):
    folder = tmp_path / "idea-effects"
    with attempt_effect_lock(folder) as lease:
        with pytest.raises(AttemptEffectBusy):
            with attempt_effect_lock(folder):
                pytest.fail("implicit re-entry bypassed task ownership")
        with attempt_effect_lock(folder, lease=lease) as nested:
            assert nested is lease
            require_effect_lease(nested, folder)
    with attempt_effect_lock(folder) as other:
        assert other.owner.owner_nonce != lease.owner.owner_nonce


def test_foreign_task_cannot_borrow_lease(tmp_path):
    with attempt_effect_lock(tmp_path / "idea-a") as lease:
        with pytest.raises(AttemptEffectInDoubt):
            with attempt_effect_lock(tmp_path / "idea-b", lease=lease):
                pytest.fail("lease escaped task scope")
    assert not (tmp_path / "idea-b").exists()


def test_partial_publication_retains_owner_across_object_loss(tmp_path):
    folder = tmp_path / "idea-effects"
    with pytest.raises(AttemptEffectInDoubt):
        with attempt_effect_lock(folder):
            raise AttemptEffectInDoubt("fixture publication failed after prepare")
    original = {p.name: p.read_bytes() for p in
                (folder / "_attempt_effect.lock").iterdir() if p.is_file()}
    with pytest.raises(AttemptEffectBusy):
        with attempt_effect_lock(folder):
            pytest.fail("uncertain effects automatically resumed")
    assert {p.name: p.read_bytes() for p in
            (folder / "_attempt_effect.lock").iterdir() if p.is_file()} == original


def test_pre_effect_error_releases_guard(tmp_path):
    folder = tmp_path / "idea-effects"
    with pytest.raises(ValueError):
        with attempt_effect_lock(folder):
            raise ValueError("fixture validation failed before effects")
    with attempt_effect_lock(folder) as lease:
        require_effect_lease(lease, folder)


def test_changed_owner_cannot_authorize_writes_or_erase_replacement(tmp_path):
    folder = tmp_path / "idea-effects"
    with pytest.raises(AttemptEffectInDoubt):
        with attempt_effect_lock(folder) as lease:
            metadata = lease.owner.lock_dir / "lock.json"
            metadata.write_text('{"fixture":"replacement"}')
            require_effect_lease(lease, folder)
    assert metadata.read_text() == '{"fixture":"replacement"}'


def test_held_stop_refuses_effect_guard_before_publication(tmp_path):
    folder = tmp_path / "idea-effects"
    (folder / "_execution_stops").mkdir(parents=True)
    from orze.engine.termination_hold import TerminationUnconfirmed
    with pytest.raises(TerminationUnconfirmed):
        with attempt_effect_lock(folder):
            pytest.fail("stopping execution admitted new control-plane writes")
    assert not (folder / "_attempt_effect.lock").exists()
    assert not (folder / "_attempt_effect.lock.source-lock").exists()
