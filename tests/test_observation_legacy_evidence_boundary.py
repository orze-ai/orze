"""Opt-in observation tasks cannot qualify unrelated legacy task files."""
import json

import pytest

from orze.reporting.evidence import (
    qualify_authoritative_report_evidence, qualify_local_report_evidence,
    qualify_result_artifacts,
)
from test_observation_contract_config import contract


@pytest.mark.parametrize("entry", ["local", "artifacts", "authoritative"])
def test_declared_observation_protocol_does_not_promote_shared_legacy_score(tmp_path, entry):
    results = tmp_path / "results"
    folder = results / "task"
    folder.mkdir(parents=True)
    (folder / "metrics.json").write_text(json.dumps({"status": "COMPLETED", "score": 999}))
    cfg = {"observation_contract": contract(), "report": {
        "primary_metric": "score", "sort": "descending",
        "columns": [{"key": "score", "label": "Score"}]}}
    if entry == "authoritative":
        result = qualify_authoritative_report_evidence("task", results, cfg, {"task"})
    else:
        call = qualify_local_report_evidence if entry == "local" else qualify_result_artifacts
        result = call(folder, cfg)
    assert result[2] is None
    assert result[3] == "observation_adapter_required"


def test_legacy_without_observation_protocol_retains_its_qualification(tmp_path):
    (tmp_path / "metrics.json").write_text('{"status":"COMPLETED","score":0}')
    cfg = {"report": {"primary_metric": "score", "columns": [{"key": "score"}]}}
    _, _, value, reason = qualify_result_artifacts(tmp_path, cfg)
    assert value == 0
    assert reason == "local_artifacts_verified"
