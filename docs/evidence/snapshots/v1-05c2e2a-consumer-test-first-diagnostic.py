"""Owned role consumer contracts; real CPU/SQLite, not controller ACK tests."""
from unittest.mock import Mock

import pytest

from orze.engine import lifecycle, role_supervision, roles, trigger_delivery
from orze.engine.orchestrator import Orze
from orze.engine.process import RoleProcess
from orze.idea_lake import IdeaLake
from test_role_supervision import project, _prepare


@pytest.mark.parametrize("route", ["shutdown", "kill_all", "atexit", "upgrade"])
def test_pending_role_survives_every_consumer_without_inventing_closure(project, route):
    p = project
    owner = p.begin()
    active = {"engineer": owner}
    assert roles.check_active_roles(active) == []
    lake = Mock()
    if route in {"shutdown", "kill_all"}:
        lifecycle.graceful_shutdown(
            p.results, {}, {}, {}, active, 1, {}, lake, "test-host", "test-instance",
            kill_all=route == "kill_all", managed=True)
    elif route == "atexit":
        lifecycle.atexit_cleanup({}, {}, active, p.results)
    else:
        controller = Orze.__new__(Orze)
        controller.active_roles, controller.active, controller.active_evals = active, {}, {}
        controller.lake = lake
        with pytest.raises(role_supervision.RoleSupervisionHOLD):
            controller._kill_and_save()
        lake.close.assert_not_called()
    assert active == {"engineer": owner}
    assert owner.process is None
    assert (p.lock / "role-process.json").exists()
    assert not p.marker.exists()


def test_closed_tree_waits_for_actual_trigger_settlement_before_finished(project, monkeypatch):
    p = project
    lake = IdeaLake(p.root / "delivery.db")
    try:
        source = p.root / "trigger"
        source.write_text("consumer fixture")
        pending = trigger_delivery.observe_trigger(lake.db_path, "scope", "engineer", source)["pending"]
        lease = trigger_delivery.lease_trigger(
            lake.db_path, pending["delivery_id"], scope="scope", role_name="engineer",
            expected_sha256=pending["payload_sha256"], owner="controller")
        launch = trigger_delivery.begin_launch(
            lake.db_path, lease, attempt_id=p.metadata["attempt_id"],
            nonce_sha256=p.metadata["nonce_sha256"], command_sha256=p.metadata["command_sha256"])
        p.metadata["trigger_delivery"] = {key: launch[key] for key in role_supervision._REF}
        p.metadata["trigger_delivery_db"] = str(lake.db_path)
        owner = p.begin()
        child = _prepare(p, owner)
        rp = RoleProcess(
            role_name="engineer", process=child, start_time=0, log_path=p.root / "role.log",
            timeout=30, lock_dir=p.lock, cycle_num=1, writes_ideas_file=False,
            process_nonce=p.nonce, supervision_owner=owner,
            trigger_launch=launch, trigger_delivery_db=lake.db_path)
        assert trigger_delivery.record_started(lake.db_path, launch, process_pid=child.pid)
        owner.start()
        assert child.wait(timeout=3) == 0
        active = {"engineer": rp}
        monkeypatch.setattr(roles, "_fs_unlock", lambda *args: pytest.fail("v2 bare unlock"))
        monkeypatch.setattr(roles, "_consecutive_soft_failures", {"engineer": 2})
        actual_terminal = trigger_delivery.record_terminal
        monkeypatch.setattr(trigger_delivery, "record_terminal", lambda *args, **kwargs: False)
        assert roles.check_active_roles(active) == []
        assert active["engineer"] is rp
        assert roles._consecutive_soft_failures == {"engineer": 2}
        assert p.lock.exists()
        assert trigger_delivery.get_delivery(lake.db_path, launch["delivery_id"])["state"] == "STARTED"
        monkeypatch.setattr(trigger_delivery, "record_terminal", actual_terminal)
        assert roles.check_active_roles(active) == [("engineer", roles.OUTCOME_OK)]
        assert active == {} and not p.lock.exists()
        assert roles._consecutive_soft_failures == {}
        assert roles.check_active_roles(active) == []
        assert trigger_delivery.get_delivery(lake.db_path, launch["delivery_id"])["state"] == "TERMINAL"
    finally:
        lake.close()
