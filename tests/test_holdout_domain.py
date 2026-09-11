"""Independent toy-instance oracles and real sealed-FD worker units.

These are not the separate actual CLI/SQLite holdout campaign. No original
holdout jobs or expected winning schedule are embedded in this unit fixture.
"""
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

from examples.holdout import scheduling as domain


@pytest.fixture
def instance():
    return {"instance_id": "unit-schedule", "horizon": 4, "capacity": 2, "jobs": [
        {"id": "required", "release": 0, "deadline": 1, "duration": 1, "demand": 1,
         "value": 0, "required": True, "after": []},
        {"id": "left", "release": 1, "deadline": 3, "duration": 2, "demand": 1,
         "value": 4, "required": False, "after": ["required"]},
        {"id": "right", "release": 1, "deadline": 3, "duration": 2, "demand": 1,
         "value": 6, "required": False, "after": ["required"]},
    ]}


def encoded(instance, entries):
    return json.dumps({"instance_id": instance["instance_id"], "schedule": entries}).encode()


def feasible_entries():
    return [{"job_id": "required", "start": 0},
            {"job_id": "left", "start": 1}, {"job_id": "right", "start": 1}]


def independent_oracle(instance, candidate, capacity):
    # Event sweep, rather than the implementation's integer-tick evaluator.
    jobs = {item["id"]: item for item in instance["jobs"]}
    starts = {entry["job_id"]: entry["start"] for entry in candidate["schedule"]}
    assert len(starts) == len(candidate["schedule"])
    events = []
    for name, start in starts.items():
        job = jobs[name]
        assert start >= job["release"]
        assert start + job["duration"] <= min(instance["horizon"], job["deadline"])
        for previous in job["after"]:
            assert previous in starts and starts[previous] + jobs[previous]["duration"] <= start
        events.extend([(start, job["demand"]), (start + job["duration"], -job["demand"])])
    for job in instance["jobs"]:
        assert not job["required"] or job["id"] in starts
    total = 0
    for _, change in sorted(events):  # Negative end events precede start events.
        total += change
        assert 0 <= total <= capacity
    return sum(jobs[name]["value"] for name in starts)


@pytest.mark.parametrize("candidate,expected", [("baseline", 0), ("challenger", 10)])
def test_deterministic_generators_and_independent_finite_oracle(instance, candidate, expected):
    before = copy.deepcopy(instance)
    result = domain.produce(instance, candidate)
    assert domain.produce(instance, candidate) == result
    assert independent_oracle(instance, result, 2) == expected
    assert domain.evaluate(instance, json.dumps(result).encode(), domain.PROTOCOLS[0]) == {
        "status": "valid", "reason_code": "feasible", "scheduled_value": expected}
    assert instance == before


def test_exact_deadline_half_open_and_entry_order_have_literal_oracles(instance):
    first = encoded(instance, feasible_entries())
    reordered = encoded(instance, list(reversed(feasible_entries())))
    expected = {"status": "valid", "reason_code": "feasible", "scheduled_value": 10}
    assert domain.evaluate(instance, first, domain.PROTOCOLS[0]) == expected
    assert domain.evaluate(instance, reordered, domain.PROTOCOLS[0]) == expected


def test_same_bytes_only_protocol_capacity_changes_validity(instance):
    raw = encoded(instance, feasible_entries())
    assert domain.evaluate(instance, raw, domain.PROTOCOLS[0])["scheduled_value"] == 10
    assert domain.evaluate(instance, raw, domain.PROTOCOLS[1]) == {
        "status": "invalid", "reason_code": "capacity_overload"}
    assert instance["capacity"] == 2


@pytest.mark.parametrize("defect,reason", [
    ("unknown", "unknown_job"), ("duplicate", "duplicate_job"),
    ("required", "missing_required"), ("deadline", "time_window"),
    ("release", "time_window"), ("prerequisite", "missing_prerequisite"),
    ("precedence", "precedence_violation"), ("overload", "capacity_overload"),
])
def test_one_scientific_defect_has_typed_rejection_without_score(instance, defect, reason):
    entries = feasible_entries()
    if defect == "unknown":
        entries.append({"job_id": "foreign", "start": 0})
    elif defect == "duplicate":
        entries.append(dict(entries[1]))
    elif defect == "required":
        entries = []
    elif defect == "deadline":
        entries[1]["start"] = 2
    elif defect == "release":
        entries[1]["start"] = 0
    elif defect == "prerequisite":
        instance["jobs"][2]["after"] = ["left"]
        entries = [entries[0], entries[2]]
    elif defect == "precedence":
        instance["jobs"][1]["release"] = 0
        entries[1]["start"] = 0
    else:
        instance["jobs"][2]["demand"] = 2
    verdict = domain.evaluate(instance, encoded(instance, entries), domain.PROTOCOLS[0])
    assert verdict == {"status": "invalid", "reason_code": reason}
    assert "scheduled_value" not in verdict


@pytest.mark.parametrize("raw", [
    b"{", b"\xff", b'{"instance_id":"unit-schedule","instance_id":"unit-schedule","schedule":[]}',
    b'{"instance_id":"unit-schedule","schedule":[{"job_id":"required","start":0,"start":0}]}',
    b'{"instance_id":"unit-schedule","schedule":[{"job_id":"required","start":NaN}]}',
    b" " * 16385,
])
def test_bad_candidate_encoding_is_domain_invalid_not_exception(instance, raw):
    assert domain.evaluate(instance, raw, domain.PROTOCOLS[0]) == {
        "status": "invalid", "reason_code": "candidate_json_invalid"}


@pytest.mark.parametrize("change", ["top_extra", "entry_extra", "bool", "float", "negative", "identity", "array"])
def test_candidate_schema_is_exact(instance, change):
    value = {"instance_id": instance["instance_id"], "schedule": [{"job_id": "required", "start": 0}]}
    if change == "top_extra":
        value["score"] = 0
    elif change == "entry_extra":
        value["schedule"][0]["valid"] = True
    elif change in ("bool", "float", "negative"):
        value["schedule"][0]["start"] = {"bool": False, "float": 0.0, "negative": -1}[change]
    elif change == "identity":
        value["instance_id"] = "other-instance"
    else:
        value = []
    assert domain.evaluate(instance, json.dumps(value).encode(), domain.PROTOCOLS[0]) == {
        "status": "invalid", "reason_code": "candidate_schema_invalid"}


@pytest.mark.parametrize("change", ["extra", "bool_capacity", "float_duration", "duplicate", "unknown_after", "cycle"])
def test_instance_config_rejection_is_not_a_candidate_verdict(instance, change):
    if change == "extra":
        instance["schedule"] = []
    elif change == "bool_capacity":
        instance["capacity"] = True
    elif change == "float_duration":
        instance["jobs"][0]["duration"] = 1.0
    elif change == "duplicate":
        instance["jobs"][1]["id"] = "required"
    elif change == "unknown_after":
        instance["jobs"][0]["after"] = ["missing"]
    else:
        instance["jobs"][0]["after"] = ["left"]
    with pytest.raises(ValueError):
        domain.validate_instance(instance)


def test_configuration_and_request_are_detached_and_no_answer_is_in_producer(instance):
    configured = domain.SchedulingDomain({"instance": instance})
    instance["capacity"] = 999
    request = domain.make_request("produce", candidate="baseline")
    prepared = configured.prepare(request, ())
    request["outputs"]["candidate"]["path"] = "changed"
    assert configured.instance["capacity"] == 2
    assert prepared["action"]["outputs"] == domain.PRODUCER_OUTPUTS
    assert prepared["action"]["command"] == [sys.executable, str(Path(domain.__file__).resolve())]
    assert prepared["observation"] is None
    assert configured.interpret(prepared, None) == ()
    assert set(domain.produce(configured.instance, "baseline")) == {"instance_id", "schedule"}


def prepared_evaluation(instance, raw, protocol):
    source = {"artifact_id": "source-candidate", "content_sha256": hashlib.sha256(raw).hexdigest()}
    configured = domain.SchedulingDomain({"instance": instance})
    return configured, configured.prepare(domain.make_request("evaluate", protocol=protocol,
        source_id=source["artifact_id"]), (source,))


def envelope_for(prepared, verdict):
    inputs = prepared["action"]["inputs"]
    return {"version": 1, "protocol": inputs["payload"]["protocol"],
        "instance_sha256": domain._digest(inputs["instance"]),
        "source_artifact_ids": [value["artifact_id"] for value in inputs["source_bindings"]],
        "verdict": verdict, "worker_cpu_seconds": 0.0, "worker_wall_seconds": 0.01}


def test_protocol_observations_keep_same_subject_and_separate_comparison_domains(instance):
    raw = encoded(instance, feasible_entries())
    original, first = prepared_evaluation(instance, raw, domain.PROTOCOLS[0])
    _, second = prepared_evaluation(instance, raw, domain.PROTOCOLS[1])
    valid = original.interpret(first, envelope_for(first, {"status": "valid", "reason_code": "feasible", "scheduled_value": 10}))[0]
    invalid = original.interpret(second, envelope_for(second, {"status": "invalid", "reason_code": "capacity_overload"}))[0]
    assert first["observation"]["spec_fingerprint"] == second["observation"]["spec_fingerprint"]
    assert first["observation"]["protocol_fingerprint"] != second["observation"]["protocol_fingerprint"]
    assert valid["comparison_scope"] != invalid["comparison_scope"]
    assert valid["values"]["direction"] == "maximize" and valid["values"]["value"] == 10
    assert "value" not in invalid["values"] and invalid["validation"]["status"] == "invalid"


@pytest.mark.parametrize("change", ["source", "protocol", "instance", "version_bool", "timing_bool", "invalid_value"])
def test_interpret_rejects_bad_envelope_instead_of_fabricating_observation(instance, change):
    raw = encoded(instance, feasible_entries())
    configured, prepared = prepared_evaluation(instance, raw, domain.PROTOCOLS[0])
    envelope = envelope_for(prepared, {"status": "invalid", "reason_code": "capacity_overload"})
    if change == "source":
        envelope["source_artifact_ids"] = ["other"]
    elif change == "protocol":
        envelope["protocol"] = domain.PROTOCOLS[1]
    elif change == "instance":
        envelope["instance_sha256"] = "0" * 64
    elif change == "version_bool":
        envelope["version"] = True
    elif change == "timing_bool":
        envelope["worker_cpu_seconds"] = False
    else:
        envelope["verdict"]["scheduled_value"] = 0
    with pytest.raises(ValueError):
        configured.interpret(prepared, envelope)


@contextmanager
def sealed(raw):
    fd = os.memfd_create("holdout-domain-unit", os.MFD_ALLOW_SEALING)
    reader = None
    try:
        assert os.write(fd, raw) == len(raw)
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS,
            fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
        reader = os.open("/proc/self/fd/" + str(fd), os.O_RDONLY)
        yield reader
    finally:
        if reader is not None:
            os.close(reader)
        os.close(fd)


@pytest.mark.parametrize("operation", ["produce", "evaluate", "invalid", "partial"])
def test_real_stdlib_worker_sealed_source_and_explicit_partial_fault(instance, tmp_path, operation):
    from orze.engine.supervised_process import prepare_supervised, SupervisionUncertain
    raw = b"{" if operation == "invalid" else encoded(instance, feasible_entries())
    if operation == "produce":
        configured = domain.SchedulingDomain({"instance": instance})
        prepared = configured.prepare(domain.make_request("produce", candidate="baseline"), ())
    else:
        configured, prepared = prepared_evaluation(instance, raw, domain.PROTOCOLS[0])
    before = copy.deepcopy(prepared)
    process = None
    with sealed(raw) as source_fd, sealed(domain._canonical(prepared["action"]["inputs"])) as input_fd:
        env = {**os.environ, "ORZE_ACTION_INPUT_FD": str(input_fd),
            "ORZE_ACTION_SOURCE_FDS": json.dumps({} if operation == "produce" else {"source-candidate": source_fd}),
            "CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": ""}
        env.pop("ORZE_HOLDOUT_EVALUATOR_FAULT", None)
        if operation == "partial":
            env["ORZE_HOLDOUT_EVALUATOR_FAULT"] = "partial_exit_71"
        try:
            try:
                process = prepare_supervised(prepared["action"]["command"], identity={"unit": "holdout-domain"},
                    cwd=str(tmp_path), env=env, worker_only_fds=(input_fd, source_fd))
            except SupervisionUncertain as exc:
                process = exc.process
                raise
            process.start()
            code = process.wait(timeout=4)
            assert code == (71 if operation == "partial" else 0)
            if operation == "produce":
                assert json.loads((tmp_path / "candidate.json").read_bytes()) == {
                    "instance_id": instance["instance_id"], "schedule": [{"job_id": "required", "start": 0}]}
            elif operation == "partial":
                with pytest.raises(json.JSONDecodeError):
                    json.loads((tmp_path / "evaluation.json").read_bytes())
            else:
                envelope = json.loads((tmp_path / "evaluation.json").read_bytes())
                claims = configured.interpret(prepared, envelope)
                assert claims[0]["validation"]["status"] == ("invalid" if operation == "invalid" else "valid")
                assert envelope["source_artifact_ids"] == ["source-candidate"]
            assert prepared == before
        finally:
            if process is not None:
                if process.poll() is None:
                    process.stop(timeout=.5)
                assert type(process.poll()) is int


@pytest.mark.parametrize("defect", ["sha", "fd_ids"])
def test_source_mismatch_refuses_before_partial_fault(instance, tmp_path, monkeypatch, defect):
    raw = encoded(instance, feasible_entries())
    _, prepared = prepared_evaluation(instance, raw, domain.PROTOCOLS[0])
    inputs = prepared["action"]["inputs"]
    if defect == "sha":
        inputs["source_bindings"][0]["content_sha256"] = "0" * 64
    with sealed(raw) as source_fd, sealed(domain._canonical(inputs)) as input_fd:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ORZE_ACTION_INPUT_FD", str(input_fd))
        monkeypatch.setenv("ORZE_ACTION_SOURCE_FDS", json.dumps(
            {"foreign": source_fd} if defect == "fd_ids" else {"source-candidate": source_fd}))
        monkeypatch.setenv("ORZE_HOLDOUT_EVALUATOR_FAULT", "partial_exit_71")
        with pytest.raises(ValueError):
            domain.main()
        assert not (tmp_path / "evaluation.json").exists()
