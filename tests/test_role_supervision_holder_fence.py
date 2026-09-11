"""Candidate regression: READY holder cannot redirect trigger settlement."""
import copy
from pathlib import Path
import time

import pytest

from orze.engine import process, role_supervision as supervision, trigger_delivery
from orze.idea_lake import IdeaLake
from test_role_supervision import project, _prepare


def test_ready_holder_database_change_is_rejected_before_consumer(project):
    p = project
    lake = IdeaLake(p.root / "authority.db")
    try:
        source = p.root / "_trigger_engineer"
        source.write_text("PRIVATE request")
        pending = trigger_delivery.observe_trigger(lake.db_path, "scope", "engineer", source)["pending"]
        lease = trigger_delivery.lease_trigger(lake.db_path, pending["delivery_id"],
            scope="scope", role_name="engineer", expected_sha256=pending["payload_sha256"], owner="controller")
        launch = trigger_delivery.begin_launch(lake.db_path, lease, attempt_id=p.metadata["attempt_id"],
            nonce_sha256=p.metadata["nonce_sha256"], command_sha256=p.metadata["command_sha256"])
        p.metadata["trigger_delivery"] = {key: launch[key] for key in supervision._REF}
        p.metadata["trigger_delivery_db"] = str(lake.db_path)
        owner = p.begin()
        _prepare(p, owner)
        rp = process.RoleProcess(role_name="engineer", process=owner.process,
            start_time=time.time(), log_path=p.root / "role.log", timeout=30,
            lock_dir=p.lock, cycle_num=1, writes_ideas_file=False,
            process_nonce=p.nonce, trigger_launch=copy.deepcopy(launch),
            trigger_delivery_db=str(lake.db_path), supervision_owner=owner)
        assert supervision.supervised_role_owner(rp) is owner
        before = (p.lock / "role-process.json").read_bytes()
        rp.trigger_delivery_db = str(p.root / "different.db")

        with pytest.raises(supervision.RoleSupervisionHOLD):
            supervision.supervised_role_owner(rp)

        assert (p.lock / "role-process.json").read_bytes() == before
        assert not Path(rp.trigger_delivery_db).exists()
        assert not p.marker.exists()
    finally:
        lake.close()
