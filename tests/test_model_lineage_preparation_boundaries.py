"""Bounded preparation API controls, not old-release behavior claims."""
from unittest.mock import Mock

import pytest

from orze.core import model_lineage as lineage
from test_model_lineage_prepared_finalization import prepared_project, _api


@pytest.mark.parametrize("malformed", ["missing", "non_path"])
def test_malformed_manifest_configuration_is_a_controlled_lineage_rejection(prepared_project, malformed):
    p = prepared_project
    prepare, _ = _api()
    if malformed == "missing":
        del p.cfg["data_separation"]["train_manifest"]
    else:
        p.cfg["data_separation"]["train_manifest"] = None
    with pytest.raises(lineage.ModelLineageError):
        prepare(p.tp, p.folder, p.cfg)
    assert not (p.folder / lineage.LINEAGE_FILE).exists()


def test_coordinator_receipt_directory_creation_does_not_invalidate_file_bindings(prepared_project):
    p = prepared_project
    prepare, publish = _api()
    prepared = prepare(p.tp, p.folder, p.cfg)
    (p.folder / "_execution_effects" / p.tp.attempt_id).mkdir(parents=True)
    assert publish(prepared, p.tp, p.folder, p.cfg)["artifact_kind"] == "file"


def test_binding_budget_rejects_before_large_artifact_hash(prepared_project, monkeypatch):
    p = prepared_project
    prepare, _ = _api()
    tripwire = Mock(side_effect=AssertionError("over-budget preparation hashed model"))
    monkeypatch.setattr(lineage, "_artifact_digest", tripwire)
    monkeypatch.setattr(lineage, "_MAX_PUBLICATION_BINDINGS", 1)
    with pytest.raises(lineage.ModelLineagePublicationUnsupported):
        prepare(p.tp, p.folder, p.cfg)
    tripwire.assert_not_called()
