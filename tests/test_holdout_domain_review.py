"""Independent finite scheduling verdicts, not the real CLI workflow.

The handwritten boundary schedule is an explicit test fixture, not a reference
optimum. The stdlib verifier never imports the tested Domain or generator. These
tests do not start producers/evaluators or manufacture SQLite execution proof.
"""
import copy
import hashlib
import json
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
ORACLE = runpy.run_path(str(ROOT / "docs/evidence/checks/v1-07b-domain-check.py"),
                        run_name="independent_holdout_verdict")
V1, V2 = "schedule-feasibility-v1", "schedule-feasibility-v2"
CHALLENGE_SHA = "fd1a4795c25829e5891680980bdaaef311985e92c4878f8a1552caa260023245"


@pytest.fixture
def instance():
    return json.loads((ROOT / "examples/holdout/instance.json").read_bytes())


@pytest.fixture
def domain():
    from examples.holdout import scheduling
    return scheduling


def candidate(instance, entries):
    return json.dumps({"instance_id": instance["instance_id"],
        "schedule": [{"job_id": identity, "start": start} for identity, start in entries]},
        separators=(",", ":")).encode()


def verdict(instance, domain, raw, protocol, expected):
    original = copy.deepcopy(instance)
    independently_computed = ORACLE["evaluate"](instance, raw, protocol)
    assert independently_computed == expected
    measured = domain.evaluate(instance, raw, protocol)
    assert ORACLE["verify_result"](instance, raw, protocol, measured) == expected
    assert instance == original
    return measured


def test_public_input_matches_the_unchanged_original_challenge(instance):
    raw = (ROOT / "docs/evidence/challenges/v1-07b-original.md").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CHALLENGE_SHA
    declared = json.loads(raw.decode().split("```json\n", 1)[1].split("\n```", 1)[0])
    assert instance == declared


@pytest.mark.parametrize("entries,score", [([("a", 0)], 0), ([("a", 0), ("b", 3), ("e", 6)], 20)])
def test_valid_zero_and_exact_deadline_halfopen_boundary(instance, domain, entries, score):
    # b ends exactly at deadline 6; e starts at 6. Both have demand 2, so an
    # inclusive-end capacity check would incorrectly reject this fixture.
    result = verdict(instance, domain, candidate(instance, entries), V1,
        {"status": "valid", "reason_code": "feasible", "scheduled_value": score})
    assert type(result["scheduled_value"]) is int


def test_entry_order_has_no_semantics_and_protocol_two_only_changes_capacity(instance, domain):
    entries = [("a", 0), ("b", 3), ("e", 6)]
    raw, reversed_raw = candidate(instance, entries), candidate(instance, list(reversed(entries)))
    expected = {"status": "valid", "reason_code": "feasible", "scheduled_value": 20}
    assert verdict(instance, domain, raw, V1, expected) == verdict(instance, domain, reversed_raw, V1, expected)
    invalid = {"status": "invalid", "reason_code": "capacity_overload"}
    result = verdict(instance, domain, raw, V2, invalid)
    assert "scheduled_value" not in result
    narrowed = copy.deepcopy(instance)
    narrowed["capacity"] = 1
    assert domain.evaluate(narrowed, raw, V1) == result
    assert ORACLE["evaluate"](narrowed, raw, V1) == result
    assert verdict(instance, domain, candidate(instance, [("a", 0)]), V2,
        {"status": "valid", "reason_code": "feasible", "scheduled_value": 0})["scheduled_value"] == 0


@pytest.mark.parametrize("fault,reason", [
    ("duplicate", "duplicate_job"),
    ("missing_prerequisite", "missing_prerequisite"),
    ("overload", "capacity_overload"),
    ("malformed", "candidate_json_invalid"),
])
def test_independent_single_fault_fixtures_have_typed_invalid_no_magic_score(instance, domain, fault, reason):
    rows = {"duplicate": [("a", 0), ("a", 0)],
            "missing_prerequisite": [("a", 0), ("e", 6)],
            "overload": [("a", 0), ("b", 0)]}
    raw = b'{"instance_id":' if fault == "malformed" else candidate(instance, rows[fault])
    result = verdict(instance, domain, raw, V1, {"status": "invalid", "reason_code": reason})
    assert set(result) == {"status", "reason_code"}


@pytest.mark.parametrize("fault,reason", [
    ("boolean_start", "candidate_schema_invalid"),
    ("float_start", "candidate_schema_invalid"),
    ("negative_start", "candidate_schema_invalid"),
    ("top_extra", "candidate_schema_invalid"),
    ("entry_extra", "candidate_schema_invalid"),
    ("instance_mismatch", "candidate_schema_invalid"),
    ("nonbytes", "candidate_schema_invalid"),
    ("utf8", "candidate_json_invalid"),
    ("duplicate_json_key", "candidate_json_invalid"),
    ("too_large", "candidate_json_invalid"),
    ("unknown", "unknown_job"),
    ("required", "missing_required"),
    ("deadline", "time_window"),
    ("precedence", "precedence_violation"),
])
def test_schema_types_and_remaining_constraints_are_not_coerced(instance, domain, fault, reason):
    starts = {"boolean_start": True, "float_start": 0.0, "negative_start": -1}
    rows = {"unknown": [("a", 0), ("unknown-id", 0)], "required": [],
            "deadline": [("a", 3)], "precedence": [("a", 1), ("c", 2)]}
    raw = candidate(instance, [("a", starts[fault])]) if fault in starts else candidate(instance, rows.get(fault, [("a", 0)]))
    value = json.loads(raw)
    if fault == "top_extra":
        value["scheduled_value"] = 999
    elif fault == "entry_extra":
        value["schedule"][0]["score"] = 0
    elif fault == "instance_mismatch":
        value["instance_id"] = "another-instance"
    if fault in ("top_extra", "entry_extra", "instance_mismatch"):
        raw = json.dumps(value).encode()
    elif fault == "nonbytes":
        raw = raw.decode()
    elif fault == "utf8":
        raw = b"\xff"
    elif fault == "duplicate_json_key":
        raw = raw.replace(b'"start":0', b'"start":0,"start":0')
    elif fault == "too_large":
        raw += b" " * 16385
    result = verdict(instance, domain, raw, V1, {"status": "invalid", "reason_code": reason})
    assert "scheduled_value" not in result


@pytest.mark.parametrize("mode", ["baseline", "challenger"])
def test_deterministic_generators_are_checked_independently_without_optimality_claim(instance, domain, mode):
    original = copy.deepcopy(instance)
    first = domain.produce(instance, mode)
    second = domain.produce(instance, mode)
    assert first == second
    assert set(first) == {"instance_id", "schedule"}
    raw = json.dumps(first, separators=(",", ":")).encode()
    actual = ORACLE["evaluate"](instance, raw, V1)
    assert actual["status"] == "valid"
    assert type(actual["scheduled_value"]) is int
    assert ORACLE["verify_result"](instance, raw, V1, domain.evaluate(instance, raw, V1)) == actual
    assert instance == original


@pytest.mark.parametrize("score", [False, 0.0, 999])
def test_independent_record_checker_rejects_magic_or_coerced_zero(instance, score):
    raw = candidate(instance, [("a", 0)])
    with pytest.raises(ValueError):
        ORACLE["verify_result"](instance, raw, V1,
            {"status": "valid", "reason_code": "feasible", "scheduled_value": score})
