"""Lock-free interpretation of a closed CPU action's captured result bytes.

Domain callbacks receive detached data, never Lake/transaction/attempt write
authority. Their labels remain domain claims. The native publisher owns the
same-effect artifact/observation/terminal transaction and its final watches.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from orze.core.cpu_observation_contract import cpu_observation_binding, cpu_observation_records
from orze.engine.artifact_publication import verify_prepared_artifacts
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
from orze.engine.observation_publication import _decode, _read

MAX_RESULT_BYTES = 1048576


def _encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def publication_binding(run, scope):
    from orze.core.research_interfaces import domain_run_metadata, domain_sources
    from orze.engine.cpu_action_sources import records
    declaration = domain_run_metadata(run)["observation"]
    if declaration is None:
        return None
    return cpu_observation_binding(
        adapter_id=declaration["adapter_id"],
        spec_fingerprint=declaration["spec_fingerprint"],
        protocol_fingerprint=declaration["protocol_fingerprint"],
        scope=str(scope), input_artifacts=records(domain_sources(run)))


@dataclass(frozen=True)
class PreparedDomainPublication:
    run: object
    metadata_json: str
    publication_json: str
    artifacts: object
    records_json: str
    records_sha256: str


def prepare(run, ref, prepared_artifacts):
    """Interpret only a bounded, exact declared result snapshot, outside SQL."""
    from orze.core.research_interfaces import domain_run_metadata, interpret_domain_run
    metadata = domain_run_metadata(run)
    artifacts = verify_prepared_artifacts(
        prepared_artifacts, ref, Path(prepared_artifacts.idea_dir),
        json.loads(prepared_artifacts.binding_json),
        source_dir=Path(prepared_artifacts.source_dir))
    binding = publication_binding(run, Path(prepared_artifacts.idea_dir).parent)
    declaration = metadata["observation"]
    envelope, result_ids = None, []
    if declaration is not None:
        name = declaration["result_output"]
        output = json.loads(prepared_artifacts.binding_json)["contract"]["outputs"].get(name)
        if (type(output) is not dict or type(output.get("max_bytes")) is not int
                or not 0 < output["max_bytes"] <= MAX_RESULT_BYTES):
            raise AttemptEffectInDoubt("cpu_domain_result_bound_invalid")
        selected = [record for record in artifacts if record["logical_name"] == name]
        if len(selected) != 1:
            raise AttemptEffectInDoubt("cpu_domain_result_missing")
        result = selected[0]
        raw = _read(Path(result["path"]), output["max_bytes"], envelope=True)
        if (len(raw) != result["size_bytes"]
                or hashlib.sha256(raw).hexdigest() != result["content_sha256"]):
            raise AttemptEffectInDoubt("cpu_domain_result_changed")
        envelope = _decode(raw)
        result_ids = [result["artifact_id"]]
    claims = interpret_domain_run(run, envelope)
    if type(claims) is not tuple or len(claims) > 32:
        raise AttemptEffectInDoubt("cpu_domain_claims_not_explicit")
    if declaration is None:
        if claims:
            raise AttemptEffectInDoubt("cpu_domain_undeclared_observations")
        observations = []
    else:
        observations = cpu_observation_records(ref, binding, result_ids, claims)
    encoded = _encoded(observations)
    prepared = PreparedDomainPublication(run, _encoded(metadata), _encoded(binding),
        prepared_artifacts, encoded, hashlib.sha256(encoded.encode()).hexdigest())
    verify(prepared, ref)
    return prepared


def verify(prepared, ref):
    """Recheck small metadata and file identities; never rerun the callback."""
    from orze.core.research_interfaces import domain_run_metadata
    if (type(prepared) is not PreparedDomainPublication
            or _encoded(domain_run_metadata(prepared.run)) != prepared.metadata_json
            or _encoded(publication_binding(prepared.run, Path(prepared.artifacts.idea_dir).parent))
               != prepared.publication_json
            or hashlib.sha256(prepared.records_json.encode()).hexdigest() != prepared.records_sha256):
        raise AttemptEffectInDoubt("cpu_domain_prepared_changed")
    verify_prepared_artifacts(prepared.artifacts, ref, Path(prepared.artifacts.idea_dir),
        json.loads(prepared.artifacts.binding_json), source_dir=Path(prepared.artifacts.source_dir))
    return json.loads(prepared.records_json)
