"""A consumer must never settle a public role's substituted database."""
import sqlite3

from orze.engine import role_delivery, role_supervision, trigger_delivery
from orze.engine.process import RoleProcess
from orze.idea_lake import IdeaLake
from test_role_supervision import project, _prepare


def _state(path, delivery_id):
    with sqlite3.connect(path) as conn:
        return conn.execute("SELECT state FROM trigger_deliveries WHERE delivery_id=?",
                            (delivery_id,)).fetchone()[0]


def test_settlement_does_not_use_database_replaced_after_first_holder_check(project, monkeypatch):
    p = project
    lake = IdeaLake(p.root / "owned.db")
    try:
        source = p.root / "trigger"
        source.write_text("snapshot fixture")
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
        foreign = p.root / "foreign.db"
        with sqlite3.connect(lake.db_path) as src, sqlite3.connect(foreign) as dst:
            src.backup(dst)
        owner.start()
        assert child.wait(timeout=3) == 0
        actual_closed = owner.require_closed

        def replace_after_check(*args, **kwargs):
            closure = actual_closed(*args, **kwargs)
            rp.trigger_delivery_db = str(foreign)
            return closure

        monkeypatch.setattr(owner, "require_closed", replace_after_check)
        assert role_delivery.settle_role_delivery(rp, "ok", 0, True) is False
        assert _state(foreign, launch["delivery_id"]) == "STARTED"
        assert _state(lake.db_path, launch["delivery_id"]) == "STARTED"
        assert (p.lock / "role-process.json").exists()
    finally:
        lake.close()
