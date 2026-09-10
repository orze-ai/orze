"""Explicit JSON-observation adapter, separate task authority and worker IO.

CALLING SPEC:
    prepare_evaluation(...) -> PreparedEvaluation: source/artifact hashing,
        entrypoint copy and immutable input manifest, outside writer locks.
    verify_evaluation(...) -> dict: cheap identity/current source metadata only.
    prepare_observations(...) -> PreparedObservations: strict worker envelope
        parsing and independent result copy, outside writer locks.
    verify_observations(...) -> (artifact records, observation records): cheap
        checks; caller registers both in the native terminal SQL transaction.

The prepare/verify helpers have no lifecycle writes; finish_evaluation owns the
explicit terminal transaction. No provider/process launch, observation ranking
or scientific validation occurs here. Worker validity is recorded, not promoted.
Only the entrypoint is copied, not its imports/environment/data closure.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil

from orze.core.execution_attempts import AttemptRef, _json, current_attempt
from orze.core.research_artifacts import artifacts_for_attempt, get_artifact
from orze.engine.attempt_effect_lock import AttemptEffectBusy
from orze.engine import artifact_publication as files

_SCRIPT_LIMIT = 1024 * 1024
_MANIFEST_LIMIT = 65536


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _same(left, right):
    return _encode(left) == _encode(right)


def _verify_ids(identities):
    files._verify_identities(tuple((path, directory, tuple(identity))
                                 for path, directory, identity in identities))


def _ids(paths):
    result = {}
    for path in paths:
        for name, directory, identity in files._path_identities(Path(path)):
            result[(name, directory)] = identity
    return tuple((name, directory, identity)
                 for (name, directory), identity in sorted(result.items()))


def _read(path, maximum, *, envelope=False):
    identities = files._path_identities(Path(path))
    parent = files._open_directory(Path(path).parent)
    fd = None
    try:
        fd = os.open(Path(path).name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=parent)
        if files._identity(os.fstat(fd)) != identities[-1][2]:
            raise AttemptEffectBusy("observation_file_changed")
        chunks, total = [], 0
        while True:
            chunk = os.read(fd, min(65536, maximum - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                error = ValueError if envelope else AttemptEffectBusy
                raise error("observation_file_size_limit")
            chunks.append(chunk)
        if files._identity(os.fstat(fd)) != identities[-1][2]:
            raise AttemptEffectBusy("observation_file_changed")
        files._verify_identities(identities)
        return b"".join(chunks)
    finally:
        try:
            if fd is not None:
                os.close(fd)
        finally:
            os.close(parent)


def _write_once(path, raw):
    """Small create-only publication; exact readback precedes use."""
    if len(raw) > _MANIFEST_LIMIT:
        raise AttemptEffectBusy("observation_manifest_size_limit")
    parent = files._open_directory(Path(path).parent)
    fd = None
    try:
        fd = os.open(Path(path).name, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                     | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=parent)
        remaining = memoryview(raw)
        while remaining:
            written = os.write(fd, remaining)
            if type(written) is not int or not 0 < written <= len(remaining):
                raise OSError("observation_manifest_short_write")
            remaining = remaining[written:]
        os.fchmod(fd, 0o400)
        os.fsync(fd)
        closing, fd = fd, None
        os.close(closing)
        os.fsync(parent)
        if _read(path, _MANIFEST_LIMIT) != raw:
            raise AttemptEffectBusy("observation_manifest_readback_failed")
    finally:
        try:
            if fd is not None:
                os.close(fd)
        finally:
            os.close(parent)


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _declaration(cfg):
    from orze.core.observation_contract import get_observation_contract
    contract = get_observation_contract(cfg)
    if contract is None:
        raise AttemptEffectBusy("observation_contract_required")
    root = Path(cfg.get("_project_root") or ".").absolute()
    script = Path(cfg.get("eval_script") or "")
    if not script.is_absolute():
        script = root / script
    python = cfg.get("python") or os.sys.executable
    if type(python) is not str:
        raise AttemptEffectBusy("observation_python_invalid")
    python = shutil.which(python) if not Path(python).is_absolute() else python
    if not python:
        raise AttemptEffectBusy("observation_python_unavailable")
    arguments = cfg.get("eval_args") or []
    if not isinstance(arguments, (list, str)):
        raise AttemptEffectBusy("observation_arguments_invalid")
    environment = cfg.get("train_extra_env") or {}
    if type(environment) is not dict or any(type(key) is not str for key in environment):
        raise AttemptEffectBusy("observation_environment_invalid")
    # Pin the actual string conversion used by the legacy launcher while
    # rejecting unbounded/non-JSON declarations before filesystem effects.
    _json({"environment": environment, "arguments": arguments})
    declaration = {"contract": contract, "script": str(script.absolute()),
                   "python": str(Path(python).absolute()), "arguments": arguments,
                   "environment": {key: str(value) for key, value in environment.items()},
                   "sealed_files": cfg.get("sealed_files") or []}
    return json.loads(_json(declaration))


def verify_sealed(idea_dir, cfg):
    from orze.engine.sealed import load_sealed_manifest, verify_sealed_files
    paths = cfg.get("sealed_files") or []
    if paths and verify_sealed_files(paths, load_sealed_manifest(Path(idea_dir).parent)):
        raise AttemptEffectBusy("observation_sealed_files_changed")


def _sealed_ids(idea_dir, cfg):
    paths = cfg.get("sealed_files") or []
    if not paths:
        return ()
    try:
        return _ids([*(Path(path).absolute() for path in paths),
                     Path(idea_dir).parent / ".sealed_hashes"])
    except (OSError, AttemptEffectBusy) as exc:
        raise AttemptEffectBusy("observation_sealed_files_changed") from exc


def _verify_sealed_ids(identities):
    try:
        _verify_ids(identities)
    except (OSError, AttemptEffectBusy) as exc:
        raise AttemptEffectBusy("observation_sealed_files_changed") from exc


def _source(lake, idea_dir, source_event, names):
    from orze.engine.completion_events import require_completion
    if source_event is None or getattr(source_event, "attempt_ref", None) is None:
        raise AttemptEffectBusy("observation_native_training_source_required")
    row = require_completion(source_event, lake, Path(idea_dir).parent, phase="training")
    binding = row["binding"].get("artifact_publication") if row else None
    if (row is None or row["terminal"].get("outcome") != "completed"
            or not binding or binding["scope"] != str(Path(idea_dir).parent)):
        raise AttemptEffectBusy("observation_training_artifacts_required")
    records = artifacts_for_attempt(lake.conn, source_event.attempt_ref)
    if set(row["terminal"].get("artifact_ids", [])) != {r["artifact_id"] for r in records}:
        raise AttemptEffectBusy("observation_training_artifact_receipt_mismatch")
    by_name = {record["logical_name"]: record for record in records}
    if any(name not in by_name for name in names):
        raise AttemptEffectBusy("observation_declared_input_missing")
    selected = [by_name[name] for name in names]
    for record in selected:
        if (record["producer"] != asdict(source_event.attempt_ref)
                or record["scope"] != binding["scope"]
                or record["spec_fingerprint"] != binding["spec_fingerprint"]):
            raise AttemptEffectBusy("observation_input_producer_mismatch")
    return row, binding, selected


@dataclass(frozen=True)
class PreparedEvaluation:
    payload_json: str
    identities: tuple
    source_identities: tuple


def prepare_evaluation(idea_dir, cfg, lake, attempt_id, source_event):
    """All expensive input/script work precedes native begin's transaction."""
    try:
        idea_dir = Path(idea_dir).absolute()
        declaration = _declaration(cfg)
        sealed_ids = _sealed_ids(idea_dir, cfg)
        verify_sealed(idea_dir, cfg)
        _verify_sealed_ids(sealed_ids)
        _, source_binding, records = _source(
            lake, idea_dir, source_event, declaration["contract"]["inputs"])
        script = Path(declaration["script"])
        source_ids = _ids([script, *(Path(r["path"]) for r in records)])
        for record in records:
            path = Path(record["path"])
            if (path.stat().st_size != record["size_bytes"] or
                    files._snapshot_hash(path, record["size_bytes"] or 1)
                    != record["content_sha256"]):
                raise AttemptEffectBusy("observation_input_content_changed")
        _verify_ids(source_ids)
        _verify_sealed_ids(sealed_ids)
        parent = idea_dir / "_evaluation_attempts"
        fd = files._open_directory(parent, create=True)
        try:
            os.mkdir(attempt_id, 0o700, dir_fd=fd)
            os.fsync(fd)
        finally:
            os.close(fd)
        attempt_dir = parent / attempt_id
        work = attempt_dir / "work"
        work_fd = files._open_directory(work, create=True)
        os.close(work_fd)
        copied_script = attempt_dir / "entrypoint.py"
        script_sha, _, _ = files._copy_one(script, copied_script, _SCRIPT_LIMIT)
        sealed_manifest_sha = (_sha(_read(idea_dir.parent / ".sealed_hashes", _MANIFEST_LIMIT))
                               if sealed_ids else None)
        _verify_sealed_ids(sealed_ids)
        protocol = _sha(_encode({"declaration": declaration, "entrypoint_sha256": script_sha,
                                "sealed_manifest_sha256": sealed_manifest_sha}))
        inputs = [r["artifact_id"] for r in records]
        observation = {"adapter_id": declaration["contract"]["adapter"],
            "protocol_fingerprint": protocol, "spec_fingerprint": source_binding["spec_fingerprint"],
            "scope": str(idea_dir.parent), "input_artifact_ids": inputs}
        from orze.core.observation_contract import validate_observation_publication_binding
        observation = validate_observation_publication_binding(observation)
        previous = current_attempt(lake.conn, idea_dir.name, "evaluation")
        if previous is not None:
            old = retry_pin(previous)
            if (not _same(old["source_ref"], asdict(source_event.attempt_ref))
                    or not _same(old["inputs"], records)
                    or not _same(old["observation_publication"], observation)
                    or old["entrypoint_sha256"] != script_sha):
                raise AttemptEffectBusy("observation_retry_input_or_protocol_changed")
        output = work / declaration["contract"]["output"]["path"]
        # Output parents are explicit writable worker storage, never aliases
        # of task metrics or the controller-owned manifest/entrypoint.
        output_parent = files._open_directory(output.parent, create=True)
        os.close(output_parent)
        manifest = {"schema": 1, "task_id": idea_dir.name, "attempt_id": attempt_id,
            "source_ref": asdict(source_event.attempt_ref), "protocol_fingerprint": protocol,
            "inputs": records, "output_path": str(output)}
        manifest_path = attempt_dir / "input_manifest.json"
        _write_once(manifest_path, _encode(manifest))
        artifact = {"contract": {"version": 1, "outputs": {
            "observations": dict(declaration["contract"]["output"])}},
            "root": source_binding["root"], "scope": str(idea_dir.parent),
            "spec_fingerprint": _sha(_encode({"subject": observation["spec_fingerprint"],
                                              "protocol": protocol}))}
        identities = _ids([copied_script, manifest_path, *(Path(r["path"]) for r in records)])
        payload = {"observation_publication": observation, "artifact_publication": artifact,
            "evaluation_io": {"declaration": declaration, "attempt_dir": str(attempt_dir),
                "work": str(work), "output": str(output), "manifest": str(manifest_path),
                "entrypoint": str(copied_script), "entrypoint_sha256": script_sha,
                "inputs": records, "identities": identities,
                "sealed_identities": sealed_ids,
                "source_ref": asdict(source_event.attempt_ref)}}
        prepared = PreparedEvaluation(_json(json.loads(_encode(payload))), identities, source_ids)
        verify_evaluation(prepared, idea_dir, cfg, lake, source_event)
        return prepared
    except (OSError, ValueError, TypeError, UnicodeError) as exc:
        raise AttemptEffectBusy("observation_launch_preparation_failed") from exc


def verify_evaluation(prepared, idea_dir, cfg, lake, source_event):
    payload = json.loads(prepared.payload_json)
    io = payload["evaluation_io"]
    if not _same(io["declaration"], _declaration(cfg)):
        raise AttemptEffectBusy("observation_launch_declaration_changed")
    _verify_sealed_ids(io["sealed_identities"])
    _verify_ids(prepared.identities)
    _verify_ids(prepared.source_identities)
    _, _, records = _source(lake, idea_dir, source_event,
                            io["declaration"]["contract"]["inputs"])
    if not _same(records, io["inputs"]):
        raise AttemptEffectBusy("observation_launch_input_changed")
    return payload


def validate_bound_execution(row, idea_dir, cfg, *, verify_files=True):
    """No mutable cfg can redirect the actual stored evaluator contract."""
    io = row["binding"].get("evaluation_io")
    if not io or not _same(io["declaration"], _declaration(cfg)):
        raise AttemptEffectBusy("observation_launch_declaration_changed")
    base = Path(idea_dir).absolute() / "_evaluation_attempts" / row["attempt_id"]
    if (io["attempt_dir"] != str(base) or io["work"] != str(base / "work")
            or io["output"] != str(base / "work" / io["declaration"]["contract"]["output"]["path"])):
        raise AttemptEffectBusy("observation_output_scope_changed")
    if verify_files:
        _verify_sealed_ids(io["sealed_identities"])
        _verify_ids(io["identities"])
    return io


@dataclass(frozen=True)
class PreparedObservations:
    artifacts: files.PreparedArtifacts
    records_json: str
    records_sha256: str


def prepare_observations(lake, ref, idea_dir, cfg, row):
    """Read a strict bounded envelope; validity labels remain adapter claims."""
    io = validate_bound_execution(row, idea_dir, cfg)
    verify_sealed(idea_dir, cfg)
    for record in io["inputs"]:
        if not _same(get_artifact(lake.conn, record["artifact_id"]), record):
            raise AttemptEffectBusy("observation_input_record_changed")
    output = Path(io["output"])
    maximum = io["declaration"]["contract"]["output"]["max_bytes"]
    raw = _read(output, maximum, envelope=True)
    envelope = _decode(raw)
    if (type(envelope) is not dict or set(envelope) != {"schema", "observations"}
            or type(envelope["schema"]) is not int or envelope["schema"] != 1
            or type(envelope["observations"]) is not list or len(envelope["observations"]) > 32):
        raise ValueError("observation_envelope_invalid")
    publication = row["binding"]["observation_publication"]
    result_id = files.artifact_occurrence_id(ref, "observations", publication["scope"])
    records, names = [], set()
    from orze.core.research_observations import validate_observation_record
    for item in envelope["observations"]:
        if type(item) is not dict or set(item) != {"name", "values", "validation", "comparison_scope"}:
            raise ValueError("observation_item_invalid")
        name = item["name"]
        if type(name) is not str or name in names:
            raise ValueError("observation_name_duplicate_or_invalid")
        names.add(name)
        record = {"schema": 1, "observation_id": _sha(_encode({"evaluator": asdict(ref),
            "name": name, "scope": publication["scope"]})), "evaluator": asdict(ref),
            **publication, "result_artifact_ids": [result_id], **item}
        records.append(validate_observation_record(record))
    artifacts = files.prepare_artifacts(ref, idea_dir, row["binding"]["artifact_publication"],
                                       source_dir=Path(io["work"]))
    artifact_records = json.loads(artifacts.records_json)
    if len(artifact_records) != 1 or artifact_records[0]["content_sha256"] != _sha(raw):
        raise AttemptEffectBusy("observation_output_changed_during_parse")
    validate_bound_execution(row, idea_dir, cfg)
    encoded = _encode(sorted(records, key=lambda record: record["name"]))
    return PreparedObservations(artifacts, encoded.decode("utf-8"), _sha(encoded))


def verify_observations(prepared, ref, idea_dir, cfg, row):
    io = validate_bound_execution(row, idea_dir, cfg)
    if _sha(prepared.records_json.encode("utf-8")) != prepared.records_sha256:
        raise AttemptEffectBusy("observation_prepared_metadata_changed")
    artifacts = files.verify_prepared_artifacts(prepared.artifacts, ref, idea_dir,
        row["binding"]["artifact_publication"], source_dir=Path(io["work"]))
    return list(artifacts), json.loads(prepared.records_json)


def retry_pin(row):
    io = row["binding"].get("evaluation_io")
    if not io or not row["binding"].get("observation_publication"):
        raise ValueError("observation_retry_source_required")
    return {"source_ref": io["source_ref"], "observation_publication":
        row["binding"]["observation_publication"], "inputs": io["inputs"],
        "declaration": io["declaration"], "entrypoint_sha256": io["entrypoint_sha256"]}


def prepare_retry(idea_dir, cfg, failure_id, row):
    """Metadata-only retry admission. Old attempt directories remain intact."""
    if not _same(row["binding"]["evaluation_io"]["declaration"], _declaration(cfg)):
        raise ValueError("observation_retry_declaration_changed")
    path = Path(idea_dir) / "_evaluation_retries" / str(failure_id)
    fd = files._open_directory(path, create=True)
    os.close(fd)
    target = path / "manifest.json"
    value = {"schema_version": 2, "mode": "json_observations", "idea_id": Path(idea_dir).name,
             "failed_transition_id": failure_id, "pin": retry_pin(row)}
    if not target.exists():
        _write_once(target, _encode(value))
    if not _same(_decode(_read(target, _MANIFEST_LIMIT)), value):
        raise ValueError("observation_retry_pin_changed")
    return str(failure_id)


def verify_retry(idea_dir, cfg, failure_id, row):
    target = Path(idea_dir) / "_evaluation_retries" / str(failure_id) / "manifest.json"
    value = {"schema_version": 2, "mode": "json_observations", "idea_id": Path(idea_dir).name,
             "failed_transition_id": failure_id, "pin": retry_pin(row)}
    if (not _same(row["binding"]["evaluation_io"]["declaration"], _declaration(cfg))
            or not _same(_decode(_read(target, _MANIFEST_LIMIT)), value)):
        raise ValueError("observation_retry_pin_changed")


def finish_evaluation(lake, ep, idea_dir, cfg, ret, *, forced=None, not_started=False):
    """Native terminal writer; no legacy shared-output fallback or inference."""
    import socket
    from orze.core.execution_attempts import (
        AttemptAuthorityError, StaleAttempt, finish_attempt, mark_running)
    from orze.core.research_artifacts import register_artifacts
    from orze.core.research_observations import register_observations, observations_for_attempt
    from orze.engine.accounting import record_compute_terminal
    from orze.engine.execution_authority import execution_transaction, lifecycle_fence
    from orze.engine.native_evaluation import _owned, _ref, _verify_compute
    from orze.engine.completion_events import CompletionEvent

    row = _owned(lake, ep, states=("LAUNCHING", "RUNNING"))
    io = validate_bound_execution(row, idea_dir, cfg)
    ref = _ref(ep)
    prepared = None
    outcome, reason, detail = "failed", "observation_process_failed", f"Exit code {ret}"
    if forced is not None:
        outcome, reason, detail = forced
    elif ret == 0:
        try:
            prepared = prepare_observations(lake, ref, idea_dir, cfg, row)
        except (OSError, ValueError, UnicodeError, RecursionError) as exc:
            reason, detail = "observation_output_invalid", type(exc).__name__
        else:
            outcome, reason, detail = "completed", "observation_envelope_recorded", ""
    success = prepared is not None
    try:
        with execution_transaction(lake, idea_dir) as tx:
            row = _owned(lake, ep, states=("LAUNCHING", "RUNNING"))
            validate_bound_execution(row, idea_dir, cfg)
            records, observations = [], []
            if prepared is not None:
                records, observations = verify_observations(prepared, ref, idea_dir, cfg, row)
            plan = {"operation": "observation_evaluation_terminal", "outcome": outcome,
                    "return_code": ret, "reason_code": reason,
                    "artifact_ids": [record["artifact_id"] for record in records],
                    "observation_ids": [record["observation_id"] for record in observations]}
            digest = tx.prepare(ref, plan)
            if row["state"] == "LAUNCHING" and not not_started:
                mark_running(tx.conn, ref)
            artifact_ids = observation_ids = ()
            if prepared is not None:
                verify_observations(prepared, ref, idea_dir, cfg, row)
                artifact_ids = register_artifacts(tx.conn, ref, records)
                observation_ids = register_observations(tx.conn, ref, observations)
            else:
                # Diagnostic publication is attempt-local and never a worker
                # observation. No metrics.json/eval_output fallback is called.
                diagnostic = Path(io["attempt_dir"]) / "failure.json"
                if not diagnostic.exists():
                    _write_once(diagnostic, _encode({"schema": 1, "status": "FAILED",
                        "reason_code": reason, "detail": str(detail)[:500]}))
            if not not_started:
                receipt = record_compute_terminal(ep, idea_dir, outcome, reason,
                    phase="evaluation", return_code=ret)
                _verify_compute(idea_dir, receipt, process=ep, phase="evaluation", event="terminal",
                    outcome=outcome, reason_code=reason, return_code=ret,
                    require_start=row["state"] == "RUNNING")
            if not lake._record_state_transition_in_tx(ep.idea_id, "IN_PROGRESS",
                    "COMPLETE" if success else "FAILED", reason=reason,
                    host=socket.gethostname(), pid=os.getpid(), sop_type="training"):
                raise AttemptAuthorityError("observation_terminal_lifecycle_rejected")
            terminal = {"outcome": outcome, "reason_code": reason, "return_code": ret,
                "effect_receipt_sha256": digest, "artifact_ids": list(artifact_ids),
                "observation_ids": list(observation_ids),
                "lifecycle": lifecycle_fence(lake, ep.idea_id, "evaluation")}
            if finish_attempt(tx.conn, ref, terminal, not_started=not_started) != "committed":
                raise AttemptAuthorityError("observation_terminal_not_new")
            if (not _same(artifacts_for_attempt(tx.conn, ref),
                          sorted(records, key=lambda record: record["logical_name"]))
                    or not _same(observations_for_attempt(tx.conn, ref),
                                 sorted(observations, key=lambda record: record["name"]))):
                raise AttemptAuthorityError("observation_terminal_records_changed")
            if prepared is not None:
                verify_observations(prepared, ref, idea_dir, cfg, row)
            # Later lifecycle/terminal SQL triggers must not replace input
            # metadata after the store's earlier per-batch validation.
            for record in io["inputs"]:
                if not _same(get_artifact(tx.conn, record["artifact_id"]), record):
                    raise AttemptAuthorityError("observation_input_final_readback_failed")
    except StaleAttempt:
        return None
    return CompletionEvent(ep.idea_id, ep.gpu, ref)
