def _owned(lake, ep, *, states=("RUNNING",)):
    row = require_current(lake.conn, _ref(ep), states=states)
    if not canonical_identity_equal(
            lifecycle_fence(lake, ep.idea_id, "evaluation"), row["binding"].get("lifecycle")):
        raise StaleAttempt("evaluation_lifecycle_revision_changed")
    return row
