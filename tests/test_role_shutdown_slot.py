"""Shutdown must not overwrite a role slot replaced during other cleanup."""
from unittest.mock import Mock

from orze.engine import lifecycle, role_supervision
from test_role_supervision import project


def test_shutdown_preserves_replacement_slot_and_original_strong_hold(project):
    p = project
    owner = p.begin()
    active = {"engineer": owner}
    replacement = object()
    lake = Mock()
    lake.close.side_effect = lambda: active.update(engineer=replacement)
    lifecycle.graceful_shutdown(
        p.results, {}, {}, {}, active, 1, {}, lake, "test-host", "test-instance",
        kill_all=True, managed=True)
    assert active["engineer"] is replacement
    assert role_supervision._OWNERS[id(owner)] is owner
    assert (p.lock / "role-process.json").exists()
    assert owner.process is None
