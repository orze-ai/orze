"""New v2 metadata requirements, not missing-API historical regressions."""
import copy
from dataclasses import asdict

import pytest

from orze.core.cpu_observation_contract import (
    cpu_observation_binding, cpu_observation_records, input_artifact_bindings,
)
from orze.core.execution_attempts import AttemptRef
from orze.core.observation_contract import validate_observation_publication_binding
from orze.core.research_observations import validate_observation_record


def source_record(index=1):
    return {"schema": 1, "artifact_id": "artifact-" + str(index),
        "producer": asdict(AttemptRef("source-" + str(index), "action", "run-" + str(index), 1)),
        "spec_fingerprint": str(index) * 64, "scope": "/results",
        "logical_name": "result", "path": "/artifacts/artifact-" + str(index) + "/content",
        "content_sha256": "c" * 64, "size_bytes": 3}


def binding():
    return cpu_observation_binding(adapter_id="domain.test.v1", protocol_fingerprint="d" * 64,
        spec_fingerprint="e" * 64, scope="/results", input_artifacts=[source_record(1), source_record(2)])


def claim(name="score"):
    return {"name": name, "values": {"zero": 0, "negative": -1},
            "validation": {"status": "unknown", "reason_code": "domain_unresolved"},
            "comparison_scope": None}


def test_v2_binding_is_detached_and_preserves_original_specs():
    sources = [source_record(1), source_record(2)]
    value = cpu_observation_binding(adapter_id="domain.test.v1", protocol_fingerprint="d" * 64,
        spec_fingerprint="e" * 64, scope="/results", input_artifacts=sources)
    original = copy.deepcopy(value)
    sources[0]["producer"]["generation"] = 7
    assert value == original
    assert value["version"] == 2
    assert [v["spec_fingerprint"] for v in value["input_artifact_bindings"].values()] == ["1" * 64, "2" * 64]
    assert value["spec_fingerprint"] == "e" * 64
    validated = validate_observation_publication_binding(value)
    validated["input_artifact_bindings"]["artifact-1"]["producer"]["generation"] = 9
    assert value == original


@pytest.mark.parametrize("fault", ["version_bool", "version_one", "no_version", "extra", "missing",
    "extra_input", "missing_input", "wrong_ref_fields", "generation_float", "generation_bool",
    "bad_spec", "bad_hash", "provenance_extra"])
def test_v2_binding_rejects_ambiguous_or_incomplete_provenance(fault):
    value = binding()
    item = value["input_artifact_bindings"]["artifact-1"]
    if fault == "version_bool": value["version"] = True
    elif fault == "version_one": value["version"] = 1
    elif fault == "no_version": del value["version"]
    elif fault == "extra": value["unknown"] = 0
    elif fault == "missing": del value["input_artifact_bindings"]
    elif fault == "extra_input": value["input_artifact_bindings"]["unlisted"] = copy.deepcopy(item)
    elif fault == "missing_input": del value["input_artifact_bindings"]["artifact-1"]
    elif fault == "wrong_ref_fields": del item["producer"]["phase"]
    elif fault == "generation_float": item["producer"]["generation"] = 1.0
    elif fault == "generation_bool": item["producer"]["generation"] = True
    elif fault == "bad_spec": item["spec_fingerprint"] = "g" * 64
    elif fault == "bad_hash": item["content_sha256"] = "A" * 64
    elif fault == "provenance_extra": item["unbound_path"] = "/tmp/foreign"
    with pytest.raises(ValueError):
        validate_observation_publication_binding(value)


def test_input_record_factory_rejects_duplicates_missing_types_and_foreign_scope():
    with pytest.raises(ValueError): input_artifact_bindings(None)
    with pytest.raises(ValueError): input_artifact_bindings([source_record()] * 2)
    with pytest.raises(ValueError): input_artifact_bindings([source_record()] * 33)
    with pytest.raises(ValueError):
        cpu_observation_binding(adapter_id="domain.test.v1", protocol_fingerprint="d" * 64,
            spec_fingerprint="e" * 64, scope="/foreign", input_artifacts=[source_record()])
    assert input_artifact_bindings(()) == {}


def test_cpu_occurrences_are_deterministic_detached_and_not_subject_identity():
    ref = AttemptRef("analysis", "action", "analyze-a", 1)
    source = claim()
    records = cpu_observation_records(ref, binding(), ("result-a",), (source,))
    assert len(records) == 1
    record = records[0]
    assert record["schema"] == 2 and "version" not in record
    assert record["evaluator"] == asdict(ref)
    assert record["values"] == {"zero": 0, "negative": -1}
    assert record["validation"]["status"] == "unknown" and record["comparison_scope"] is None
    assert records == cpu_observation_records(ref, binding(), ["result-a"], [source])
    second = cpu_observation_records(AttemptRef("analysis", "action", "analyze-b", 2),
                                    binding(), ["result-b"], [source])[0]
    assert second["observation_id"] != record["observation_id"]
    assert second["spec_fingerprint"] == record["spec_fingerprint"]
    source["values"]["zero"] = 99
    assert record["values"]["zero"] == 0


def test_empty_measurements_are_explicit_and_nonempty_require_result_artifacts():
    ref = AttemptRef("analysis", "action", "analyze-a", 1)
    assert cpu_observation_records(ref, binding(), [], ()) == []
    with pytest.raises(ValueError): cpu_observation_records(ref, binding(), [], None)
    with pytest.raises(ValueError): cpu_observation_records(ref, binding(), [], [claim()])
    with pytest.raises(ValueError): cpu_observation_records(ref, binding(), ["result"], [claim(), claim()])
    with pytest.raises(ValueError):
        cpu_observation_records(AttemptRef("analysis", "training", "train-a", 1), binding(), [], ())


@pytest.mark.parametrize("fault", ["schema_bool", "schema_one_with_v2", "record_version", "too_large", "nonfinite"])
def test_records_keep_exact_schema_and_bounded_json(fault):
    ref = AttemptRef("analysis", "action", "analyze-a", 1)
    record = cpu_observation_records(ref, binding(), ["result"], [claim()])[0]
    if fault == "schema_bool": record["schema"] = True
    elif fault == "schema_one_with_v2": record["schema"] = 1
    elif fault == "record_version": record["version"] = 2
    elif fault == "too_large": record["values"]["large"] = "x" * 32768
    elif fault == "nonfinite": record["values"]["bad"] = float("nan")
    with pytest.raises(ValueError): validate_observation_record(record)
