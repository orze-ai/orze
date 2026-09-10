"""Two concrete draft scope gaps, separate from the frozen new API suite."""
from copy import deepcopy

import pytest

from orze.engine.replication import ReplicationError, request_replication
from test_replication_requests import completed
from test_artifact_snapshot_contract import project
from test_native_training_caller_boundaries import case as native_case


@pytest.mark.parametrize("declaration", [None, ""])
def test_explicit_default_catalog_declaration_cannot_authorize_different_lake(completed, declaration):
    c = completed
    cfg = deepcopy(c.cfg)
    cfg["idea_lake_db"] = declaration
    before = list(c.lake.conn.iterdump())
    with pytest.raises(ReplicationError, match="scope"):
        request_replication(c.idea, c.results, cfg, c.lake, request_id="wrong-catalog")
    assert list(c.lake.conn.iterdump()) == before
