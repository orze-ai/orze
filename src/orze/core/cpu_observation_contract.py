"""Pure metadata factories for explicit v2 CPU observations.

input_artifact_bindings(records) projects detached, validated original artifact
metadata into exact producer/spec/content references. cpu_observation_binding
adds the separately supplied analysis subject/protocol and common scope.
cpu_observation_records(ref, binding, result_artifact_ids, claims) accepts an
explicit list/tuple of 0..32 domain claims and returns a detached list suitable
for register_observations. None is never converted into an empty result.

These functions do no filesystem/SQLite IO, process execution or scientific
inference. The executor must establish source closure and captured byte
identity, bind this metadata before GO, verify its own result artifacts, and
watch the complete publication through the final effect transaction. Labels
and comparison scopes remain domain claims, not Core approval.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

from orze.core.execution_attempts import AttemptRef
from orze.core.observation_contract import (
    _ids, validate_observation_publication_binding,
)
from orze.core.research_artifacts import _record as artifact_record
from orze.core.research_observations import validate_observation_record


def _fail(reason):
    raise ValueError("cpu_observation_contract: " + reason)


def input_artifact_bindings(records) -> dict:
    """Validate complete original records; return compact provenance only."""
    if type(records) not in (list, tuple) or len(records) > 32:
        _fail("input records must be an explicit bounded list or tuple")
    result = {}
    for original in records:
        record = json.loads(artifact_record(original))
        identity = record["artifact_id"]
        if identity in result:
            _fail("duplicate input artifact")
        result[identity] = {key: record[key] for key in (
            "producer", "spec_fingerprint", "content_sha256")}
    return result


def cpu_observation_binding(*, adapter_id, protocol_fingerprint,
                            spec_fingerprint, scope, input_artifacts) -> dict:
    """Bind a subject without relabeling any input's original specification."""
    inputs = input_artifact_bindings(input_artifacts)
    if any(record["scope"] != scope for record in input_artifacts):
        _fail("input artifact scope mismatch")
    return validate_observation_publication_binding({
        "version": 2, "adapter_id": adapter_id,
        "protocol_fingerprint": protocol_fingerprint, "spec_fingerprint": spec_fingerprint,
        "scope": scope, "input_artifact_ids": list(inputs),
        "input_artifact_bindings": inputs})


def cpu_observation_records(ref, binding, result_artifact_ids, claims) -> list[dict]:
    """Form bounded occurrence records; no registration or validity inference."""
    if type(ref) is not AttemptRef or ref.phase != "action":
        _fail("an exact action AttemptRef is required")
    publication = validate_observation_publication_binding(binding)
    if publication.get("version") != 2:
        _fail("explicit v2 publication binding is required")
    if type(result_artifact_ids) not in (list, tuple):
        _fail("result artifact IDs must be an explicit list or tuple")
    result_ids = _ids(list(result_artifact_ids))
    if type(claims) not in (list, tuple) or len(claims) > 32:
        _fail("claims must be an explicit bounded list or tuple")
    records, names = [], set()
    for claim in claims:
        if type(claim) is not dict or set(claim) != {
                "name", "values", "validation", "comparison_scope"}:
            _fail("invalid domain claim fields")
        name = claim["name"]
        if type(name) is not str or name in names:
            _fail("invalid or duplicate domain claim name")
        names.add(name)
        identity = {"schema": "orze.cpu_observation.v2", "evaluator": asdict(ref),
                    "scope": publication["scope"], "name": name}
        encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
        record = {"schema": 2, "observation_id": hashlib.sha256(encoded).hexdigest(),
                  "evaluator": asdict(ref),
                  **{key: value for key, value in publication.items() if key != "version"},
                  "result_artifact_ids": result_ids, **claim}
        records.append(validate_observation_record(record))
    return records
