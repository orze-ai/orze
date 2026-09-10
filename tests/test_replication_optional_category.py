"""A lawful task without a taxonomy remains eligible for explicit replication."""
from orze.engine.replication import request_replication
from test_replication_requests import completed
from test_artifact_snapshot_contract import project
from test_native_training_caller_boundaries import case as native_case


def test_optional_category_none_preserves_confirmed_source_and_new_task(completed):
    c = completed
    c.lake.conn.execute("UPDATE ideas SET category=NULL WHERE idea_id=?", (c.idea,))
    c.lake.conn.commit()
    result = request_replication(c.idea, c.results, c.cfg, c.lake, request_id="without-taxonomy")
    assert result["status"] == "created"
    source = c.lake.conn.execute("SELECT config,category FROM ideas WHERE idea_id=?", (c.idea,)).fetchone()
    target = c.lake.conn.execute("SELECT config,category FROM ideas WHERE idea_id=?", (result["task_id"],)).fetchone()
    assert tuple(source) == tuple(target)
    assert target["category"] is None
